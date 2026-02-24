#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件注释:
# - 文件作用: 多模型人脸匿名化（虚拟化）与批量推理
"""
Milestone 3 (multi-model): Batch face anonymization pipeline.

- 人脸检测（insightface）
- 生成柔边 mask（仅 landmarks 凸包，并裁剪到 bbox 内）
- ROI 推理（优先 inpaint(mask_image)，否则走 image-edit/img2img）
- 用同一张 mask 做贴回融合：mask 外背景严格保持原图不变
- 严格保证输出尺寸 == 输入尺寸

本版本新增：MODEL_ZOO（多模型），并把“加载/调用 pipeline”的逻辑抽象成统一接口，便于横向对比不同生成式模型本体。

- 不同 pipeline 的参数不完全一致；本代码用 inspect.signature 自动过滤不支持的参数，尽量“一套 cfg 跑多个模型”。

说明：
- 本工程仅做匿名化/虚拟化：不提供任何“身份注入/锁定为参考人脸”的功能。
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import cv2
from PIL import Image

import torch
from insightface.app import FaceAnalysis

# ---- diffusers：按需导入，避免版本不兼容直接炸 ----
from diffusers import AutoPipelineForInpainting, DiffusionPipeline  # 你原本就用它

try:
    # Kandinsky5 I2I：需要显式指定 pipeline（部分仓库的 model_index.json 可能会错误指向 T2I）
    from diffusers import Kandinsky5I2IPipeline
except Exception:
    Kandinsky5I2IPipeline = None  # type: ignore

try:
    from diffusers import FluxFillPipeline  # FLUX.1 Fill（用于 flux_fill_nf4）
except Exception:
    FluxFillPipeline = None  # type: ignore

try:
    from diffusers import FluxTransformer2DModel  # FLUX transformer（用于 NF4/4bit 组件替换）
except Exception:
    FluxTransformer2DModel = None  # type: ignore

try:
    from diffusers import StableDiffusion3Img2ImgPipeline  # SD3/3.5 的 img2img
except Exception:
    StableDiffusion3Img2ImgPipeline = None  # type: ignore

try:
    from transformers import T5EncoderModel  # FLUX 的 text_encoder_2（用于 NF4）
except Exception:
    T5EncoderModel = None  # type: ignore


# =========================
# 0) Model Zoo（你要对比的“生成式模型本体”）
# =========================
@dataclass(frozen=True)
class ModelSpec:
    """
    loader:
      - "auto_inpaint": diffusers AutoPipelineForInpainting（SD2/SDXL/Kandinsky2.2 等）
      - "flux_fill_nf4": FLUX.1 Fill dev 的 NF4（替换 transformer/text_encoder_2）
      - "sd3_img2img": diffusers StableDiffusion3Img2ImgPipeline（用 ROI img2img 近似 inpaint）
      - "diffusion_edit": diffusers DiffusionPipeline（仅图像编辑，无原生 mask）
      - "sdcpp_qwen_image_edit_2511_gguf": stable-diffusion.cpp Python 绑定（GGUF Q4_K_M），Qwen Image Edit 2511
    """
    key: str
    model_id: str
    loader: str

    # pipeline 级加载参数
    trust_remote_code: bool = False
    pad_multiple: Optional[int] = None
    torch_dtype: Optional[str] = None  # "fp16" | "bf16" | "fp32"
    variant: Optional[str] = None
    use_safetensors: Optional[bool] = None
    prefer_diffusion_pipeline: bool = False

    # 一些模型需要特殊 call 参数（比如 FLUX 的 max_sequence_length）
    extra_call_kwargs: Dict[str, Any] = field(default_factory=dict)

    # 备注：对比实验时写到 report 里会更清楚
    notes: str = ""


MODEL_ZOO: Dict[str, ModelSpec] = {
    # 1) SDXL Inpaint（你现在的 baseline）
    "sdxl_inpaint": ModelSpec(
        key="sdxl_inpaint",
        model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        loader="auto_inpaint",
        notes="强基线，高分辨率，适合做人脸 inpaint 对比。",
    ),
    # 3.65) Kandinsky 5.0 I2I Lite：无原生 mask，走 ROI-edit
    "kandinsky5_i2i_lite": ModelSpec(
        key="kandinsky5_i2i_lite",
        # 注意：`kandinskylab/Kandinsky-5.0-I2I-Lite` 本仓库不包含 diffusers 的 model_index.json（会 404）。
        # 这里使用官方提供的 Diffusers 版本（SFT，质量更好）。
        model_id="kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
        extra_call_kwargs={"max_sequence_length": 1024},
        notes="Kandinsky 5.0 I2I Lite（Diffusers SFT，图生图编辑）。内部会把分辨率映射到固定 resolutions 集合。",
    ),
    # 3.7) LongCat Image Edit Turbo（美团开源蒸馏版）：无原生 mask，走 ROI-edit
    "longcat_image_edit_turbo": ModelSpec(
        key="longcat_image_edit_turbo",
        model_id="meituan-longcat/LongCat-Image-Edit-Turbo",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
        notes="LongCat-Image-Edit-Turbo（蒸馏版，8 NFE，超低延迟）。文本渲染需用引号包裹目标文字。",
    ),
    # 3.8) Qwen Image Edit 2511（GGUF Q4_K_M via stable-diffusion.cpp python binding）：无原生 mask，走 ROI-edit
    "qwen_image_edit_2511_gguf": ModelSpec(
        key="qwen_image_edit_2511_gguf",
        model_id="unsloth/Qwen-Image-Edit-2511-GGUF",
        loader="sdcpp_qwen_image_edit_2511_gguf",
        pad_multiple=64,
        notes=(
            "Qwen-Image-Edit-2511（GGUF 4-bit：Q4_K_M），用 stable-diffusion-cpp-python 运行。"
            "使用 ROI-edit 外壳做匿名化：仅替换人脸内部区域并尽量保持姿态/表情/光照一致。"
        ),
    ),
    # 4.1) FLUX.1 Fill [dev] NF4：替换 transformer/text_encoder_2（更省显存）
    "flux_fill_nf4": ModelSpec(
        key="flux_fill_nf4",
        model_id="black-forest-labs/FLUX.1-Fill-dev",
        loader="flux_fill_nf4",
        extra_call_kwargs={"max_sequence_length": 512},
        pad_multiple=16,
        torch_dtype="bf16",
        notes="FLUX.1-Fill-dev 的 NF4 组件（transformer/text_encoder_2）来自 diffusers/FLUX.1-Fill-dev-nf4。",
    ),
    # 4.5) FLUX.2-klein-9B：无原生 mask，走 ROI-edit（本体需同意许可/可能需要 HF Token）
    "flux2_klein_9b": ModelSpec(
        key="flux2_klein_9b",
        model_id="black-forest-labs/FLUX.2-klein-9B",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
        notes=(
            "FLUX.2-klein-9B（4 steps distilled, 支持多参考编辑）。"
            "Repo 为 gated（flux-non-commercial-license），首次需要在 HF 网页同意条款并登录。"
            "推荐：steps=4, guidance_scale=1.0；单卡 4090 建议 enable_model_cpu_offload。"
        ),
    ),
    # 5) Stable Diffusion 3.5（没有标准 inpaint pipeline：用 ROI img2img 近似）
    "sd35_img2img": ModelSpec(
        key="sd35_img2img",
        model_id="stabilityai/stable-diffusion-3.5-large-turbo",
        loader="sd3_img2img",
        pad_multiple=16,
        notes="用 ROI img2img 近似 inpaint，主要用来对比“新一代 backbone”的生成质量/鲁棒性。",
    ),
}


def list_available_models() -> List[str]:
    """给 run 脚本用：打印/枚举可选 key。"""
    return list(MODEL_ZOO.keys())


def resolve_model_spec(model_id_or_key: str) -> ModelSpec:
    """
    允许你在 base_cfg.model_id 里直接传：
    - MODEL_ZOO 的 key（推荐，用于对比）
    - 或真实 Hugging Face repo id（兼容你旧代码）
    """
    if model_id_or_key in MODEL_ZOO:
        return MODEL_ZOO[model_id_or_key]

    # 兼容旧写法：如果用户直接给 repo id，就按 auto_inpaint 尝试加载
    mid = model_id_or_key.lower()
    if model_id_or_key == "flux2_dev_4bit" or ("flux.2-dev-bnb-4bit" in mid) or ("flux2-dev-bnb-4bit" in mid):
        raise ValueError(
            "`flux2_dev_4bit`（diffusers/FLUX.2-dev-bnb-4bit）已从工程中移除（含本地缓存权重）。"
            "请改用其它模型 key（例如 `flux2_klein_9b` / `flux_fill_nf4` / `sdxl_inpaint` 等）。"
        )
    # 显式兼容：Kandinsky 5 I2I（repo id 直接传入时，不能走 auto_inpaint）
    # 注意：`kandinskylab/Kandinsky-5.0-I2I-Lite` 本仓库不是 diffusers 格式，会缺少 model_index.json（导致 404）。
    # 这里自动映射到官方 Diffusers 版本。
    if "kandinsky-5.0-i2i-lite" in mid and "diffusers" not in mid:
        mapped = (
            "kandinskylab/Kandinsky-5.0-I2I-Lite-pretrain-Diffusers"
            if "pretrain" in mid
            else "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
        )
        return ModelSpec(
            key="custom",
            model_id=mapped,
            loader="diffusion_edit",
            pad_multiple=16,
            torch_dtype="bf16",
            use_safetensors=True,
            extra_call_kwargs={"max_sequence_length": 1024},
            notes=f"custom kandinsky5 i2i（repo id 直传，已映射到 Diffusers: {mapped}）",
        )

    if "kandinsky-5.0-i2i" in mid or ("kandinsky-5" in mid and "i2i" in mid):
        return ModelSpec(
            key="custom",
            model_id=model_id_or_key,
            loader="diffusion_edit",
            pad_multiple=16,
            torch_dtype="bf16",
            use_safetensors=True,
            extra_call_kwargs={"max_sequence_length": 1024},
            notes="custom kandinsky5 i2i（repo id 直传）",
        )

    # 显式兼容：Qwen Image Edit 2511 GGUF（repo id 直传时走 stable-diffusion.cpp Python 绑定）
    if "qwen-image-edit-2511-gguf" in mid or ("qwen" in mid and "image-edit-2511" in mid and "gguf" in mid):
        return ModelSpec(
            key="custom",
            model_id=model_id_or_key,
            loader="sdcpp_qwen_image_edit_2511_gguf",
            pad_multiple=64,
            notes="custom qwen image edit 2511 gguf（repo id 直传）",
        )

    # 显式兼容：FLUX.2 系列（文本生图/多参考编辑，属于 diffusion_edit，不应走 auto_inpaint）
    if "flux.2" in mid or "flux2" in mid:
        return ModelSpec(
            key="custom",
            model_id=model_id_or_key,
            loader="diffusion_edit",
            pad_multiple=16,
            torch_dtype="bf16",
            use_safetensors=True,
            notes="custom flux2（repo id 直传）",
        )

    return ModelSpec(
        key="custom",
        model_id=model_id_or_key,
        loader="auto_inpaint",
        notes="custom model_id（非 MODEL_ZOO key）",
    )


# =========================
# 1) 配置
# =========================
@dataclass
class DemoConfig:
    """基础配置：同一套参数尽量适配多模型（不支持的会被自动忽略）"""
    in_img: str
    out_img: str

    prompt: str = (
        "a realistic photo of a person, same head pose and facial expression as the input image, "
        "consistent lighting, natural skin texture, sharp eyes, high detail face"
    )
    negative_prompt: str = (
        "deformed, bad anatomy, extra face, extra eyes, blurry, low quality, "
        "cartoon, painting, uncanny, watermark, text"
    )

    seed: int = 42
    steps: int = 30
    guidance: float = 7.5
    strength: float = 0.85

    pad_ratio: float = 0.35
    mask_blur: int = 12

    # ---- 小脸/背脸处理（避免凭空补脸）----
    min_face_edit: int = 20          # 人脸短边小于该值时不走扩散
    skip_if_no_kps: bool = False     # 无关键点时是否跳过扩散（默认 False：尽量仍走 face parsing/扩散）

    # ---- 轻量 face parsing（可选）：比 landmarks 更贴合发际线/遮挡边界 ----
    # 说明：
    # - 需要你提供一个轻量 ONNX 分割模型（例如 BiSeNetV2 face parsing）。
    # - 若 onnxruntime / 模型文件不可用，会自动回退到 landmarks，不影响现有流程。
    enable_face_parsing: bool = True
    face_parsing_model: str = "models/face_parsing/bisenetv2.onnx"
    face_parsing_device: str = "auto"  # "auto" | "cuda" | "cpu"
    # 轻量推理输入分辨率（越大越准但更慢），建议 256 或 512
    face_parsing_input_hw: Tuple[int, int] = (512, 512)  # (H, W)
    # ROI 裁剪时对 bbox 的扩张（避免裁掉下颌/额头）
    face_parsing_pad_ratio: float = 0.35
    # 解析 mask 面积占 ROI 的合理范围（太小/太大认为失败，回退 landmarks）
    # NOTE: 当编辑域收紧为“内脸”时，mask 面积相对 ROI 会更小；适当降低下限避免误判失败。
    face_parsing_min_area_ratio: float = 0.03
    face_parsing_max_area_ratio: float = 0.92
    # label 选择：不同模型 label 定义不同；此默认适配 BiSeNetV2 19 类（CelebAMask-HQ）。
    # 目标：编辑域默认收紧为“内脸 + 眼镜”，减少头发/帽子/遮挡物被重绘导致的伪影与结构漂移。
    # - 包含：skin(1) / brow(2,3) / eye(4,5) / eyeglass(6) / nose(10) / mouth(11) / lip(12,13)
    # - 不包含：ear(7,8) / earring(9) / hair(17) / hat(18) / neck(14) / cloth(16) 等
    face_parsing_labels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)

    # ---- Qwen Image Edit 2511（GGUF）专用：避免风格跑偏/过度重绘 ----
    qwen_strength_min: float = 0.60
    qwen_strength_max: float = 0.90
    # 负面词不要太长（sd.cpp 有时对超长 negative 不稳）；这里提供一段短的“禁止动漫/尖耳”兜底
    qwen_extra_negative: str = (
        "anime, cartoon, illustration, painting, 3d render, doll, plastic skin, "
        "elf ears, pointy ears, animal ears, extra ears"
    )

    det_size: int = 640
    ctx_id: int = 0
    replace_all_faces: bool = True  # 多人同框时，是否替换全部人脸
    max_faces: int = 0              # >0 时限制最多替换的人脸数（0=不限制）

    # 这里既可以放 HF repo id，也可以放 MODEL_ZOO 的 key（如 "sdxl_inpaint"）
    model_id: str = "sdxl_inpaint"

    device: str = "cuda"
    pad_multiple: int = 8

    # ---- ROI 推理分辨率上限（避免大 ROI 直接爆显存）----
    # 说明：
    # - 当 ROI 的最长边 > roi_max_side 时，会等比缩放到该上限后再推理，
    #   推理完成后再 resize 回原 ROI 大小贴回，保证输出尺寸不变。
    # - <=0 表示不限制（不推荐，极易因大 ROI 触发显存峰值 / OOM）。
    roi_max_side: int = 1024

    # ---- 多卡切分（accelerate device_map）----
    # 例：device_map="balanced" / "auto" / "balanced_low_0"
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    # 例：{0: "22GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB", "cpu": "64GiB"}
    max_memory: Optional[Dict[Union[int, str], str]] = None

    # ---- 显存优化 ----
    # cpu_offload:
    # - "none": 不启用 offload，直接 `.to(device)`
    # - "model": enable_model_cpu_offload（推荐：速度/显存平衡）
    # - "sequential": enable_sequential_cpu_offload（更省显存但更慢）
    # - "auto": 先尝试 `.to(device)`，OOM 时自动回退为 offload
    cpu_offload: str = "auto"

    skip_existing: bool = True


# =========================
# 2) IO + face util
# =========================
def load_image_rgb(path: str) -> Image.Image:
    """读取图片并强制转换为 RGB"""
    return Image.open(path).convert("RGB")


def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray"""
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def init_face_app(ctx_id: int = 0, det_size: int = 640) -> FaceAnalysis:
    """
    初始化 insightface。

    说明：
    - 尽量不要在运行过程中反复调用 app.prepare() 试图切换 det_size。
      在部分 onnxruntime + CUDAExecutionProvider 组合下，这会触发段错误（core dumped）。
    - 因此这里在初始化阶段一次性设置 det_size，后续 get() 直接复用。
    """
    # 本工程需要 bbox + embedding + landmarks（用于精细 mask）
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "landmark_2d_106"])
    app.prepare(ctx_id=ctx_id, det_size=(int(det_size), int(det_size)))
    # 记录一下，后面 detect 时可以复用
    app.ctx_id = ctx_id  # type: ignore[attr-defined]
    app._demo_det_size = int(det_size)  # type: ignore[attr-defined]
    return app


def detect_largest_face_bbox(
    face_app: FaceAnalysis,
    img_pil: Image.Image,
    det_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    """返回最大人脸 bbox=(x1,y1,x2,y2)"""
    # 注意：这里不要重复 prepare()，避免 onnxruntime CUDA EP 段错误
    # det_size 参数保留仅用于接口兼容（由 init_face_app 决定实际 det_size）
    faces = face_app.get(pil_to_bgr_np(img_pil))
    if len(faces) == 0:
        return None

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces.sort(key=area, reverse=True)
    f = faces[0]
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    return (x1, y1, x2, y2)


def detect_all_faces(
    face_app: FaceAnalysis,
    img_pil: Image.Image,
    det_size: int,
) -> List[Any]:
    """返回所有人脸（按面积从大到小排序）"""
    faces = face_app.get(pil_to_bgr_np(img_pil))
    if not faces:
        return []

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces.sort(key=lambda f: (area(f), -float(f.bbox[0])), reverse=True)
    return list(faces)


def _get_face_landmarks(face) -> Optional[np.ndarray]:
    """返回可用的 2D landmarks（106 或 5 点），无则 None"""
    if face is None:
        return None
    pts = None
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        pts = np.array(face.landmark_2d_106, dtype=np.float32)
    elif hasattr(face, "kps") and face.kps is not None:
        pts = np.array(face.kps, dtype=np.float32)
    if pts is None or pts.size == 0:
        return None
    return pts


def _make_landmark_mask(
    size_wh: Tuple[int, int],
    landmarks: np.ndarray,
    pad_ratio: float,
    blur: int,
) -> Image.Image:
    """用 landmarks 的凸包生成更贴合人脸的柔边 mask"""
    W, H = size_wh
    pts = landmarks.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return Image.new("L", (W, H), 0)

    center = pts.mean(axis=0, keepdims=True)
    scale = 1.0 + float(max(pad_ratio, 0.0))
    pts = (pts - center) * scale + center
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((H, W), dtype=np.uint8)
    if hull is not None and len(hull) >= 3:
        cv2.fillConvexPoly(mask, hull, 255)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur)
    return Image.fromarray(mask, mode="L")


_FACE_PARSING_CACHE: Dict[str, Any] = {
    "session": None,
    "model_path": None,
    "input_name": None,
    "warned": False,
}


def _resolve_path_relative_to_this_file(path: str) -> str:
    """
    将相对路径解析到本文件所在目录（face_swapping/）。
    这样无论你从哪个 cwd 运行，默认的 models/face_parsing/... 都能正确找到。
    """
    p = Path(str(path or "").strip())
    if not p:
        return ""
    if p.is_absolute():
        return str(p)
    base = Path(__file__).resolve().parent
    return str((base / p).resolve())


def _load_face_parsing_session(model_path: str, device_pref: str = "auto"):
    """加载 ONNX face parsing 模型（失败返回 None）。"""
    model_path = _resolve_path_relative_to_this_file(model_path)
    if not model_path or not Path(model_path).is_file():
        if not _FACE_PARSING_CACHE["warned"]:
            print(f"[WARN] 未找到 face parsing 模型: {model_path}，将回退到 landmarks。")
            _FACE_PARSING_CACHE["warned"] = True
        return None

    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        if not _FACE_PARSING_CACHE["warned"]:
            print("[WARN] onnxruntime 未安装，face parsing 将回退到 landmarks。")
            _FACE_PARSING_CACHE["warned"] = True
        return None

    device_pref = (device_pref or "auto").lower().strip()
    providers: List[str] = ["CPUExecutionProvider"]
    if device_pref in ("auto", "cuda"):
        # 避免在未安装 GPU provider 时触发 onnxruntime warning
        try:
            avail = set(ort.get_available_providers())
        except Exception:
            avail = set()
        if avail and ("CUDAExecutionProvider" not in avail):
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # 先按偏好尝试；失败再降级 CPU（避免 CUDA EP 环境问题导致直接不可用）
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as e2:
            if not _FACE_PARSING_CACHE["warned"]:
                print(f"[WARN] 加载 face parsing ONNX 失败: {e2}")
                _FACE_PARSING_CACHE["warned"] = True
            return None

    try:
        inp = sess.get_inputs()[0]
        input_name = inp.name
    except Exception:
        input_name = None
    if not input_name:
        if not _FACE_PARSING_CACHE["warned"]:
            print("[WARN] face parsing ONNX 输入名获取失败，将回退到 landmarks。")
            _FACE_PARSING_CACHE["warned"] = True
        return None

    return (sess, input_name)


def _face_parsing_mask(
    img_pil: Image.Image,
    face,
    cfg: DemoConfig,
    blur: int,
) -> Optional[Image.Image]:
    """
    轻量 face parsing → mask（失败返回 None）。
    - 仅在 bbox 周围裁 ROI 推理，再 paste 回全图尺寸，成本较低。
    - 对小脸/侧脸：若解析 mask 面积异常会自动回退 landmarks。
    """
    if img_pil is None or face is None or not bool(getattr(cfg, "enable_face_parsing", False)):
        return None

    model_path = str(getattr(cfg, "face_parsing_model", "") or "").strip()
    device_pref = str(getattr(cfg, "face_parsing_device", "auto") or "auto")

    cached_ok = (
        _FACE_PARSING_CACHE["session"] is not None
        and _FACE_PARSING_CACHE["model_path"] == model_path
        and _FACE_PARSING_CACHE["input_name"] is not None
    )
    if not cached_ok:
        loaded = _load_face_parsing_session(model_path, device_pref)
        if loaded is None:
            _FACE_PARSING_CACHE["session"] = None
            return None
        sess, input_name = loaded
        _FACE_PARSING_CACHE.update({
            "session": sess,
            "model_path": model_path,
            "input_name": input_name,
        })

    sess = _FACE_PARSING_CACHE["session"]
    input_name = _FACE_PARSING_CACHE["input_name"]
    if sess is None or input_name is None:
        return None

    try:
        W, H = img_pil.size
        bbox = tuple(int(v) for v in face.bbox)
        bw, bh = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        if min(bw, bh) < 16:
            return None

        # ROI 裁剪（对小脸更宽一点，提供更多上下文）
        pad_ratio = float(getattr(cfg, "face_parsing_pad_ratio", 0.35) or 0.35)
        if min(bw, bh) < 120:
            pad_ratio = max(pad_ratio, 0.55)
        x1, y1, x2, y2 = _bbox_expand_and_clip(bbox, W, H, pad_ratio=pad_ratio)
        roi = img_pil.crop((x1, y1, x2, y2)).convert("RGB")
        if roi.size[0] < 32 or roi.size[1] < 32:
            return None

        # 轻量输入尺寸
        in_h, in_w = tuple(getattr(cfg, "face_parsing_input_hw", (512, 512)) or (512, 512))
        in_h, in_w = int(in_h), int(in_w)
        in_h = max(64, min(in_h, 1024))
        in_w = max(64, min(in_w, 1024))

        roi_in = roi.resize((in_w, in_h), resample=Image.BILINEAR)
        arr = np.asarray(roi_in).astype(np.float32) / 255.0

        # 默认：ImageNet normalize（适配常见 BiSeNetV2 face parsing ONNX 导出）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW

        out = sess.run(None, {input_name: arr})
        if not out:
            return None
        pred = out[0]
        if pred is None:
            return None

        # 输出兼容：(1,C,H,W) 或 (1,H,W) 等
        if pred.ndim == 4:
            seg = np.argmax(pred, axis=1)[0]
        elif pred.ndim == 3:
            # (C,H,W)
            seg = np.argmax(pred, axis=0)
        else:
            return None

        labels = tuple(int(x) for x in (getattr(cfg, "face_parsing_labels", ()) or ()))
        if not labels:
            return None
        mask_roi = np.isin(seg, labels).astype(np.uint8) * 255
        mask_roi = cv2.resize(mask_roi, (roi.size[0], roi.size[1]), interpolation=cv2.INTER_LINEAR)

        # 面积 sanity check：避免小脸/侧脸解析崩坏
        area = float((mask_roi > 0).sum())
        area_ratio = area / float(mask_roi.shape[0] * mask_roi.shape[1] + 1e-6)
        min_ar = float(getattr(cfg, "face_parsing_min_area_ratio", 0.06) or 0.06)
        max_ar = float(getattr(cfg, "face_parsing_max_area_ratio", 0.92) or 0.92)
        if area_ratio < min_ar or area_ratio > max_ar:
            return None

        full_mask = Image.new("L", (W, H), 0)
        full_mask.paste(Image.fromarray(mask_roi.astype(np.uint8), mode="L"), (x1, y1))
        if blur and int(blur) > 0:
            m = np.array(full_mask, dtype=np.uint8)
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=int(blur), sigmaY=int(blur))
            full_mask = Image.fromarray(m.astype(np.uint8), mode="L")
        return full_mask
    except Exception as e:
        if not _FACE_PARSING_CACHE["warned"]:
            print(f"[WARN] face parsing 失败，将回退到 landmarks: {e}")
            _FACE_PARSING_CACHE["warned"] = True
        return None


def make_face_mask_from_face(
    size_wh: Tuple[int, int],
    face,
    pad_ratio: float = 0.35,
    blur: int = 12,
    img_pil: Optional[Image.Image] = None,
    cfg: Optional[DemoConfig] = None,
) -> Image.Image:
    """
    仅用 landmarks 凸包生成 mask（不再回退 bbox 椭圆 / 解析模型）。
    - 无 landmarks 时返回空 mask（后续由 skip/fallback 处理）。
    """
    if face is None:
        return Image.new("L", size_wh, 0)
    # 优先：轻量 face parsing（可用则更贴合发际线/遮挡边界；失败回退 landmarks）
    if cfg is not None and img_pil is not None and bool(getattr(cfg, "enable_face_parsing", False)):
        m = _face_parsing_mask(img_pil=img_pil, face=face, cfg=cfg, blur=blur)
        if m is not None:
            return m
    pts = _get_face_landmarks(face)
    if pts is None:
        return Image.new("L", size_wh, 0)
    return _make_landmark_mask(size_wh, pts, pad_ratio=pad_ratio, blur=blur)


def clip_mask_to_bbox(mask_pil: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    将 mask 限制在 bbox 内（bbox 外强制为 0）。
    用途：按你的需求实现“bbox 重绘后，用 landmark mask 贴回并消除背景影响”。
    """
    if mask_pil is None:
        raise ValueError("mask_pil is None")
    W, H = mask_pil.size
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return Image.new("L", (W, H), 0)

    # NOTE:
    # - 直接用矩形 bbox “硬裁剪”很容易在脖子/皮肤等平滑区域出现“方框接缝”。
    # - 这里改为：bbox 内部保持 1，靠近 bbox 边缘做 soft falloff（基于距离变换），
    #   让 mask 在 bbox 边界处自然衰减，从而消除矩形割裂感。
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    feather_px = int(np.clip(float(min(bw, bh)) * 0.06, 6.0, 32.0))

    m01 = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
    bbox_bin = np.zeros((H, W), dtype=np.uint8)
    bbox_bin[y1:y2, x1:x2] = 1

    if feather_px <= 1:
        w01 = bbox_bin.astype(np.float32)
    else:
        dt = cv2.distanceTransform(bbox_bin, distanceType=cv2.DIST_L2, maskSize=5)
        w01 = np.clip(dt / float(feather_px), 0.0, 1.0).astype(np.float32)

    out01 = np.clip(m01 * w01, 0.0, 1.0)
    return Image.fromarray((out01 * 255.0).round().astype(np.uint8), mode="L")


def _make_feather_alpha_dt(mask_u8: np.ndarray, feather_px: int, blur_sigma: float = 0.0) -> np.ndarray:
    """
    用二值 mask 生成 0..1 的 feather alpha（仅在 mask 内侧渐变）。

    目的：替换“窄阈值硬切”式 alpha，降低 bbox 裁剪/ROI 贴回带来的接缝与方框割裂感。
    """
    if mask_u8 is None:
        raise ValueError("mask_u8 is None")
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be 2D")
    m = (mask_u8.astype(np.uint8) >= 128).astype(np.uint8)
    if int(m.sum()) < 16:
        return np.zeros(mask_u8.shape[:2], dtype=np.float32)

    feather_px = int(max(1, int(feather_px)))
    dt = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=5)
    alpha = np.clip(dt / float(feather_px), 0.0, 1.0).astype(np.float32)

    if blur_sigma is not None and float(blur_sigma) > 1e-6:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return alpha


def _make_change_guided_feather_alpha(
    base_patch_u8: np.ndarray,
    gen_patch_u8: np.ndarray,
    hard_u8: np.ndarray,
    strength_01: float,
    min_face: int,
    diff_thr: int = 28,
) -> np.ndarray:
    """
    ROI-edit 专用：用“变化区域”引导融合 alpha，降低发际线/轮廓的重影与错位感。

    核心思想：
    - ROI-edit（LongCat/Kandinsky/Flux2）常会产生轻微几何漂移；
    - 如果按整张 hard mask 做 feather 混合，边界处会把“新旧轮廓”混成双边缘；
    - 这里先估计 gen 与 base 的像素差分，只在“明显变化”的内侧区域进行融合，
      并对该区域做适度内缩 + 窄过渡带，从视觉上消除重影。
    """
    if base_patch_u8 is None or gen_patch_u8 is None or hard_u8 is None:
        raise ValueError("base/gen/hard is None")
    if base_patch_u8.shape != gen_patch_u8.shape:
        raise ValueError("base/gen shape mismatch")
    if hard_u8.ndim != 2:
        raise ValueError("hard_u8 must be 2D")

    h, w = hard_u8.shape[:2]
    if h < 32 or w < 32:
        return _make_feather_alpha_dt(hard_u8, feather_px=6, blur_sigma=0.8)

    hard01 = (hard_u8.astype(np.uint8) >= 128).astype(np.uint8)
    if int(hard01.sum()) < 64:
        return np.zeros((h, w), dtype=np.float32)

    # diff: 0..765
    diff = np.abs(base_patch_u8.astype(np.int16) - gen_patch_u8.astype(np.int16)).sum(axis=2).astype(np.int16)
    thr = int(np.clip(int(diff_thr), 10, 120))
    m = (hard01 > 0) & (diff >= thr)
    # 若阈值过严导致区域过小，则回退到 hard mask（避免“几乎不换脸”）
    if int(m.sum()) < int(hard01.sum()) * 0.18:
        m = (hard01 > 0)

    mask = (m.astype(np.uint8) * 255)
    # 形态学：填洞 + 去噪（kernel 随 ROI 尺寸自适应）
    k = int(np.clip(min(h, w) * 0.03, 7.0, 31.0))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    try:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)
    except Exception:
        pass

    # 内缩：把融合边界推向更内侧（重点是避开刘海/发际线边界）
    shrink_px = int(np.clip(min(h, w) * 0.040, 6.0, 26.0))
    try:
        ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1))
        mask2 = cv2.erode(mask, ker2, iterations=1)
        # 避免内缩过度导致区域过小
        if int((mask2 > 0).sum()) >= int((hard01 > 0).sum()) * 0.25:
            mask = mask2
    except Exception:
        pass

    # feather：过渡带更窄（越宽越容易“混出重影”）
    strength_01 = float(np.clip(float(strength_01), 0.0, 1.0))
    min_face = int(max(1, int(min_face)))
    feather_px = int(np.clip(float(min_face) * (0.045 + 0.010 * strength_01), 4.0, 16.0))
    alpha = _make_feather_alpha_dt(mask, feather_px=feather_px, blur_sigma=0.8)
    return alpha


def _bbox_from_mask(mask_pil: Image.Image, threshold: int = 128) -> Optional[Tuple[int, int, int, int]]:
    """从 mask 中提取最小外接框（x1,y1,x2,y2），失败返回 None。"""
    m = np.array(mask_pil.convert("L"))
    ys, xs = np.where(m >= int(threshold))
    if xs.size < 16 or ys.size < 16:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _bbox_union(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """返回两个 bbox 的并集。"""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
# =========================
# 3) 尺寸对齐：pad 到 multiple，最后再裁回
# =========================
def pad_image_and_mask_to_multiple(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    multiple: int = 8,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    许多扩散模型对尺寸有约束（8/16/64 对齐）。
    为了保证“输出尺寸==输入尺寸”，我们先 pad，再在输出阶段 crop 回原尺寸。
    """
    img = np.array(img_pil)
    msk = np.array(mask_pil.convert("L"))

    h, w = img.shape[:2]

    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    msk_pad = cv2.copyMakeBorder(msk, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return Image.fromarray(img_pad), Image.fromarray(msk_pad).convert("L"), (left, top, right, bottom)


def pad_mask_to_pad(mask_pil: Image.Image, pad: Tuple[int, int, int, int]) -> Image.Image:
    """把 mask 按 (left,top,right,bottom) 做同样 padding（用于双 mask 同步对齐）。"""
    left, top, right, bottom = (int(pad[0]), int(pad[1]), int(pad[2]), int(pad[3]))
    msk = np.array(mask_pil.convert("L"))
    msk_pad = cv2.copyMakeBorder(msk, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return Image.fromarray(msk_pad).convert("L")


def crop_back_to_original(
    out_pil: Image.Image,
    pad: Tuple[int, int, int, int],
    orig_size_wh: Tuple[int, int],
) -> Image.Image:
    """把 padding 后的输出裁回原始尺寸"""
    left, top, _, _ = pad
    orig_w, orig_h = orig_size_wh
    out_crop = out_pil.crop((left, top, left + orig_w, top + orig_h))
    if out_crop.size != (orig_w, orig_h):
        out_crop = out_crop.resize((orig_w, orig_h), resample=Image.LANCZOS)
    return out_crop


# =========================
# 5) pipe 构建（多模型）
# =========================
def _pick_dtype_for_spec(spec: ModelSpec, device: str) -> torch.dtype:
    """
    - SD/SDXL inpaint：fp16 常用
    - FLUX / SD3：官方示例常用 bfloat16（4090 支持 bf16）
    - 若 spec.torch_dtype 指定，则优先使用
    """
    if not device.startswith("cuda"):
        return torch.float32

    if spec.torch_dtype:
        dt = spec.torch_dtype.lower()
        if dt in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dt in ("fp16", "float16"):
            return torch.float16
        if dt in ("fp32", "float32"):
            return torch.float32

    if spec.loader in ("flux_fill_nf4", "sd3_img2img"):
        # 尽量用 bf16（4090 支持），但有些环境/权重可能更适合 fp16
        return torch.bfloat16
    return torch.float16


def _pipe_to_device(pipe, device: str, runtime_cfg: Optional[DemoConfig] = None):
    """
    把 diffusers pipeline 放到 device 上，并按需启用显存优化：
    - device_map（hf_device_map）存在时：不允许再显式 `.to()`（否则会破坏 accelerate 的调度）
    - cpu_offload="auto"：先尝试 `.to(device)`，OOM 时自动回退 enable_model_cpu_offload/enable_sequential_cpu_offload
    """
    # 如果 pipeline 使用了 accelerate device_map 进行分布式加载，
    # diffusers 会在 pipeline 上挂载 `hf_device_map`，此时不允许再显式 `.to()`。
    try:
        if getattr(pipe, "hf_device_map", None) is not None:
            return pipe
    except Exception:
        pass

    cpu_offload = "none"
    if runtime_cfg is not None:
        cpu_offload = str(getattr(runtime_cfg, "cpu_offload", cpu_offload) or cpu_offload).lower().strip()

    def _enable_cpu_offload(prefer: str) -> None:
        prefer = (prefer or "").lower().strip()
        # enable_model_cpu_offload / enable_sequential_cpu_offload 需要 accelerator
        if not (isinstance(device, str) and (device.startswith("cuda") or device.startswith("xpu"))):
            raise RuntimeError(f"cpu_offload 仅在加速设备上可用，但当前 device={device}")

        if prefer in ("model", "auto") and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload(device=device)
                return
            except Exception as e:
                print(f"[WARN] enable_model_cpu_offload 失败，尝试 sequential offload: {e}")

        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload(device=device)
                return
            except Exception as e:
                print(f"[WARN] enable_sequential_cpu_offload 失败: {e}")

        raise RuntimeError("当前 pipeline 不支持 CPU offload（或 accelerate 版本不满足）。")

    if cpu_offload in ("model", "sequential"):
        _enable_cpu_offload(cpu_offload)
        return pipe

    if cpu_offload == "auto":
        try:
            pipe = pipe.to(device)
            return pipe
        except Exception as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda out of memory" in msg)
            if not is_oom:
                raise
            print(f"[WARN] pipe.to({device}) 触发 OOM，将自动启用 CPU offload: {e}")
            # 尽量把已搬上 GPU 的部分清回 CPU，避免后续 offload 也失败
            try:
                pipe.to("cpu", silence_dtype_warnings=True)
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            _enable_cpu_offload("model")
            return pipe

    # 默认：不启用 offload
    pipe = pipe.to(device)
    return pipe


def _load_with_safetensors_fallback(load_fn, model_id_or_path: str, allow_fallback: bool = True, **kwargs):
    """
    部分仓库只提供 .bin（无 safetensors），这里做一次降级重试。
    """
    # 默认启用断点续传：遇到网络中断时，下一次重试能从已下载部分继续。
    kwargs.setdefault("resume_download", True)

    def _is_safetensors_header_error(exc: Exception) -> bool:
        """
        典型表现（本次你遇到的）：
        - safetensors_rust.SafetensorError: Error while deserializing header: invalid JSON ...
        往往是缓存里的 .safetensors 分片 0 字节/截断/损坏导致。
        """
        msg = str(exc)
        if "Error while deserializing header" in msg:
            return True
        if "invalid JSON" in msg and "safetensors" in msg:
            return True
        name = exc.__class__.__name__
        mod = getattr(exc.__class__, "__module__", "") or ""
        if name == "SafetensorError" or "safetensors" in mod:
            if "deserializing header" in msg or "invalid JSON" in msg:
                return True
        return False

    def _extract_checkpoint_file_from_tb(exc: Exception) -> Optional[str]:
        """
        从 transformers 的 traceback 里尝试抓到本地权重分片路径（checkpoint_file）。
        抓到了就能只重下这一片，避免 force_download=True 重拉整个模型快照。
        """
        tb = exc.__traceback__
        while tb is not None:
            loc = tb.tb_frame.f_locals
            for k in ("checkpoint_file", "resolved_archive_file", "archive_file", "weights_file", "shard_file"):
                v = loc.get(k, None)
                if isinstance(v, str) and v:
                    # 一般是 .safetensors / .bin 路径
                    if (".safetensors" in v) or v.endswith(".bin"):
                        return v
            tb = tb.tb_next
        return None

    def _extract_hf_snapshot_filename(local_path: str) -> Optional[str]:
        """
        由 HF cache 的 snapshots 路径提取 repo 内相对文件名：
          .../models--org--repo/snapshots/<rev>/<subdir>/xxx.safetensors  ->  <subdir>/xxx.safetensors
        """
        if not local_path:
            return None
        marker = os.sep + "snapshots" + os.sep
        if marker not in local_path:
            return None
        after = local_path.split(marker, 1)[1]  # "<rev>/..."
        parts = after.split(os.sep, 1)
        if len(parts) != 2:
            return None
        return parts[1]

    def _is_hf_consistency_error(err_msg: str) -> bool:
        # huggingface_hub/file_download.py: "Consistency check failed: file should be of size ..."
        return "Consistency check failed" in err_msg and "force_download=True" in err_msg

    def _extract_inconsistent_filename(err_msg: str) -> Optional[str]:
        # 取最后一段 "(xxx)"，例如 "(merges.txt)"
        l = err_msg.rfind("(")
        r = err_msg.rfind(")")
        if l == -1 or r == -1 or r <= l:
            return None
        name = err_msg[l + 1 : r].strip()
        return name or None

    def _redownload_hf_file(repo_id: str, filename: str) -> bool:
        """
        只强制重下“损坏/尺寸不一致”的单个文件，避免 force_download=True 重拉整个大模型快照。
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception:
            return False

        hub_kwargs: Dict[str, Any] = {}
        # 兼容 token / use_auth_token
        token = kwargs.get("token", None) or kwargs.get("use_auth_token", None)
        if token:
            hub_kwargs["token"] = token
        for k in (
            "cache_dir",
            "revision",
            "proxies",
            "local_files_only",
            "endpoint",
            "etag_timeout",
            "resume_download",
            "local_dir",
            "local_dir_use_symlinks",
        ):
            v = kwargs.get(k, None)
            if v is not None:
                hub_kwargs[k] = v
        hub_kwargs.setdefault("resume_download", True)

        # 如果报错只显示 basename（如 merges.txt），尝试常见子目录
        candidates: List[str] = []
        if filename:
            candidates.append(filename)
            if "/" not in filename:
                candidates.extend([f"tokenizer/{filename}", f"tokenizer_2/{filename}"])

        ok = False
        for fn in candidates:
            try:
                hf_hub_download(repo_id=repo_id, filename=fn, force_download=True, **hub_kwargs)
                ok = True
            except Exception:
                continue
        return ok

    def _is_transient_hf_network_error(exc: Exception) -> bool:
        """
        HuggingFace 下载大文件时常见的瞬态网络异常（断链/超时/分块读取失败）。
        命中后可通过 resume_download 断点续传，再重试即可恢复。
        """
        # 1) requests/urllib3 体系
        try:
            import requests  # type: ignore

            # 典型：requests.exceptions.ChunkedEncodingError / ConnectionError / Timeout
            if isinstance(exc, requests.exceptions.ChunkedEncodingError):
                return True
            if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                return True
            if isinstance(exc, requests.exceptions.RequestException):
                # 避免把明显的鉴权/404 这类不可恢复错误当成网络抖动
                msg = str(exc)
                if any(x in msg for x in ("401", "403", "404")):
                    return False
                return True
        except Exception:
            pass

        try:
            import urllib3  # type: ignore

            if isinstance(
                exc,
                (
                    urllib3.exceptions.ProtocolError,
                    urllib3.exceptions.ReadTimeoutError,
                    urllib3.exceptions.IncompleteRead,
                ),
            ):
                return True
        except Exception:
            pass

        # 2) 兜底：字符串特征
        msg = str(exc)
        transient_keys = (
            "Connection broken",
            "IncompleteRead",
            "Read timed out",
            "timed out",
            "RemoteDisconnected",
            "Connection reset by peer",
            "Connection aborted",
            "Max retries exceeded",
            "Temporary failure in name resolution",
            "Name or service not known",
            "EOF occurred in violation of protocol",
        )
        # 某些 HF 错误会包在更高层异常里（__cause__/__context__），这里仅做轻量匹配
        if any(k in msg for k in transient_keys):
            return True
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, Exception) and any(k in str(cause) for k in transient_keys):
            return True
        context = getattr(exc, "__context__", None)
        if isinstance(context, Exception) and any(k in str(context) for k in transient_keys):
            return True
        return False

    # 网络失败重试参数（仅针对瞬态网络异常；其它错误依旧立即抛出）
    max_net_retries = int(os.environ.get("HF_NET_RETRIES", "12"))
    base_sleep_s = float(os.environ.get("HF_NET_RETRY_BASE_SLEEP", "2.0"))
    max_sleep_s = float(os.environ.get("HF_NET_RETRY_MAX_SLEEP", "60.0"))
    fallback_done = False
    redownloaded_files: set[str] = set()
    net_retry_count = 0

    # 统一用 current_kwargs 在循环里逐步“增强”（例如 force_download / max_workers）
    current_kwargs: Dict[str, Any] = dict(kwargs)

    while True:
        try:
            return load_fn(model_id_or_path, **current_kwargs)

        except OSError as e:
            msg = str(e)

            # 1) 仓库无 safetensors：降级一次
            if (not fallback_done) and allow_fallback and ("safetensors" in msg) and ("Could not find the necessary" in msg):
                fallback_done = True
                current_kwargs = dict(current_kwargs)
                current_kwargs["use_safetensors"] = False
                continue

            # 2) HF 一致性校验失败：优先只重下坏的单文件；若仍失败再升级 force_download=True
            if _is_hf_consistency_error(msg):
                bad_file = _extract_inconsistent_filename(msg)
                if bad_file:
                    if bad_file not in redownloaded_files:
                        print(f"[WARN] HF 下载一致性校验失败（{bad_file}），将强制重下该文件并重试加载……")
                        redownloaded_files.add(bad_file)
                        _redownload_hf_file(model_id_or_path, bad_file)
                        continue

                    # 已经尝试过单文件重下仍失败：升级为拉整包
                    if not bool(current_kwargs.get("force_download", False)):
                        current_kwargs = dict(current_kwargs)
                        current_kwargs["force_download"] = True
                        print("[WARN] 仍失败，改为 force_download=True 重新拉取模型快照……")
                        continue
                else:
                    if not bool(current_kwargs.get("force_download", False)):
                        current_kwargs = dict(current_kwargs)
                        current_kwargs["force_download"] = True
                        print("[WARN] HF 下载一致性校验失败（未定位文件名），force_download=True 重试加载……")
                        continue
            raise

        except Exception as e:
            # 3) 网络瞬态异常：断点续传 + 降并发 + 退避重试
            if _is_transient_hf_network_error(e):
                net_retry_count += 1
                if net_retry_count > max_net_retries:
                    raise

                # 下载大模型遇到网络抖动时，降低 snapshot_download 并发更稳（默认不覆盖用户显式设置）
                if "max_workers" not in current_kwargs:
                    current_kwargs = dict(current_kwargs)
                    current_kwargs["max_workers"] = int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "1"))

                # 强制确保可断点续传
                current_kwargs.setdefault("resume_download", True)

                sleep_s = min(base_sleep_s * (2 ** (net_retry_count - 1)), max_sleep_s)
                # 只打印一行高信号日志，避免 tqdm 输出被刷屏
                print(
                    f"[WARN] HF 下载网络异常（{type(e).__name__}）：{str(e)[:240]}... "
                    f"将于 {sleep_s:.1f}s 后自动重试（{net_retry_count}/{max_net_retries}）"
                )
                time.sleep(sleep_s)
                continue

            # 4) 处理“缓存里的 safetensors 分片损坏/空文件”的常见情况
            if _is_safetensors_header_error(e) and not bool(current_kwargs.get("force_download", False)):
                ckpt_path = _extract_checkpoint_file_from_tb(e)
                bad_file = _extract_hf_snapshot_filename(ckpt_path) if ckpt_path else None
                if bad_file:
                    if bad_file not in redownloaded_files:
                        size_info = ""
                        try:
                            if ckpt_path and os.path.isfile(ckpt_path):
                                size_info = f"，本地 size={os.path.getsize(ckpt_path)} bytes"
                        except Exception:
                            pass
                        print(f"[WARN] 检测到 safetensors 文件疑似损坏：{bad_file}{size_info}。将强制重下该文件并重试加载……")
                        redownloaded_files.add(bad_file)
                        _redownload_hf_file(model_id_or_path, bad_file)
                        continue

                # 没抓到具体文件名，或单文件重下仍失败：升级 force_download=True 拉整包
                current_kwargs = dict(current_kwargs)
                current_kwargs["force_download"] = True
                print("[WARN] 检测到 safetensors 文件损坏但无法恢复，force_download=True 重试加载……")
                continue

            raise


def _require_longcat_diffusers_support():
    """
    LongCat-Image-Edit-Turbo 需要 diffusers 提供对应的 Pipeline/Model 类。
    当前环境若版本偏老，会在 from_pretrained 阶段直接 AttributeError。
    """
    try:
        import diffusers as _diffusers  # type: ignore
    except Exception as e:
        raise RuntimeError("未能导入 diffusers，无法加载 LongCat-Image-Edit-Turbo。") from e

    missing: List[str] = []
    for name in ("LongCatImageEditPipeline", "LongCatImageTransformer2DModel"):
        if not hasattr(_diffusers, name):
            missing.append(name)

    if missing:
        ver = getattr(_diffusers, "__version__", "unknown")
        raise RuntimeError(
            "当前 diffusers 版本不支持 LongCat-Image-Edit-Turbo。"
            f"（diffusers={ver}，缺少: {', '.join(missing)}）\n"
            "请升级 diffusers 到 >= 0.35.1（模型卡要求），例如：\n"
            "  pip install -U \"diffusers>=0.35.1\" \"transformers>=4.48.0\" accelerate safetensors\n"
            "或按官方建议直接安装 diffusers git 版：\n"
            "  pip install -U git+https://github.com/huggingface/diffusers\n"
            "升级后重启终端/解释器再运行。"
        )


def _require_stable_diffusion_cpp_python():
    """
    Qwen-Image-Edit-2511 GGUF 走 stable-diffusion.cpp 的 Python 绑定：
    pip 包名：stable-diffusion-cpp-python
    import 名：stable_diffusion_cpp
    """
    try:
        import stable_diffusion_cpp  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "未安装 stable-diffusion.cpp 的 Python 绑定，无法运行 Qwen-Image-Edit-2511-GGUF。\n"
            "请安装（单卡 4090 建议 CUDA 后端）：\n"
            "  CMAKE_ARGS=\"-DSD_CUDA=ON\" pip install stable-diffusion-cpp-python\n"
            "如仅想先跑通，也可不加 CMAKE_ARGS（但会走 CPU，极慢）。"
        ) from e
    return stable_diffusion_cpp


@dataclass
class _SdCppImagesOutput:
    images: List[Image.Image]


class SdCppQwenImageEdit2511GGUF:
    """
    用 stable-diffusion-cpp-python 封装成“像 diffusers 一样可调用”的 pipe。
    - 输入：ROI image 作为 ref_images[0]
    - 为避免“参考身份注入”，这里不支持额外 reference images。
    """

    def __init__(
        self,
        diffusion_model_path: str,
        llm_path: str,
        llm_vision_path: str,
        vae_path: str,
        *,
        offload_params_to_cpu: bool = True,
        diffusion_flash_attn: bool = True,
        flow_shift: float = 3.0,
        qwen_image_zero_cond_t: bool = True,
        enable_mmap: bool = True,
        verbose: bool = False,
    ):
        _require_stable_diffusion_cpp_python()
        from stable_diffusion_cpp import StableDiffusion  # type: ignore

        self._sd = StableDiffusion(
            diffusion_model_path=diffusion_model_path,
            llm_path=llm_path,
            llm_vision_path=llm_vision_path,
            vae_path=vae_path,
            offload_params_to_cpu=offload_params_to_cpu,
            diffusion_flash_attn=diffusion_flash_attn,
            qwen_image_zero_cond_t=qwen_image_zero_cond_t,
            flow_shift=float(flow_shift),
            enable_mmap=enable_mmap,
            image_resize_method="resize",  # ROI 不要被 crop
            verbose=bool(verbose),
        )

    def __call__(
        self,
        *,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        mask_image: Optional[Image.Image] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        strength: float = 0.75,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        **_kwargs,
    ) -> _SdCppImagesOutput:
        # 兼容 demo 的 generator：优先用 seed，其次 generator.initial_seed()
        if seed is None:
            try:
                seed = int(generator.initial_seed()) if generator is not None else 42
            except Exception:
                seed = 42

        image = image.convert("RGB")
        w = int(width) if width is not None else int(image.size[0])
        h = int(height) if height is not None else int(image.size[1])

        # Qwen Image Edit：用 ref_images 表达输入；仅传入当前 ROI，避免参考身份注入
        ref_images: List[Image.Image] = [image]

        # 进度条：stable-diffusion.cpp 默认不显示 diffusers 风格进度，这里用 progress_callback 做一个轻量终端进度。
        show_progress = (os.environ.get("DEMO_QWEN_PROGRESS", "1") or "").strip().lower() not in ("0", "false", "no")
        total_steps = max(1, int(num_inference_steps))
        last_step = -1
        t0 = time.perf_counter()
        tty_f = None
        if show_progress:
            # stable-diffusion-cpp-python 内部会在 verbose=False 时屏蔽 stdout/stderr；
            # 为了让进度条在终端可见，这里优先写 /dev/tty（若不可用则回退 stderr）。
            try:
                tty_f = open("/dev/tty", "w", encoding="utf-8")
            except Exception:
                tty_f = None

        def _write_progress(s: str) -> None:
            try:
                if tty_f is not None:
                    tty_f.write(s)
                    tty_f.flush()
                else:
                    sys.stderr.write(s)
                    sys.stderr.flush()
            except Exception:
                pass

        def _progress_cb(*args, **kwargs):
            nonlocal last_step, total_steps
            step = None
            try:
                if "step" in kwargs:
                    step = kwargs.get("step", None)
                elif len(args) >= 1:
                    step = args[0]
                # 有的实现会把 total_steps 也传进来
                if len(args) >= 2:
                    try:
                        ts = int(args[1])
                        if ts > 0:
                            total_steps = ts
                    except Exception:
                        pass
                if step is None:
                    return None

                if isinstance(step, float) and 0.0 <= step <= 1.0:
                    cur = int(step * float(total_steps))
                else:
                    cur = int(step)
                cur = max(0, min(int(total_steps) - 1, cur))
                if cur == last_step:
                    return None
                last_step = cur
                done = min(int(total_steps), cur + 1)
                pct = (float(done) / float(total_steps)) * 100.0
                elapsed = time.perf_counter() - t0
                _write_progress(f"\r[Qwen] {done:>3}/{int(total_steps):<3} {pct:5.1f}%  {elapsed:5.1f}s")
            except Exception:
                return None
            return None

        images = self._sd.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            ref_images=ref_images,
            mask_image=mask_image,
            width=w,
            height=h,
            sample_steps=int(num_inference_steps),
            cfg_scale=float(guidance_scale),
            strength=float(strength),
            seed=int(seed),
            sample_method="euler",
            progress_callback=_progress_cb if show_progress else None,
        )
        if show_progress:
            try:
                _write_progress("\n")
            except Exception:
                pass
        try:
            if tty_f is not None:
                tty_f.close()
        except Exception:
            pass
        return _SdCppImagesOutput(images=images)


def build_pipe(model_id_or_key: str, device: str, runtime_cfg: Optional[DemoConfig] = None):
    """
    返回：(pipe, device, spec)
    - pipe: 具体 pipeline 实例（diffusers）
    - device: 实际使用的 device
    - spec: ModelSpec（用于后续判断支持的功能）
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    # 兼容多卡机器上常见的 CUDA_VISIBLE_DEVICES 重映射：
    # 用户可能传入物理卡号（如 cuda:6），但当前进程只暴露 0..N-1 的可见 GPU。
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            d = torch.device(device)
            if d.index is not None:
                cnt = int(torch.cuda.device_count())
                if cnt > 0 and int(d.index) >= cnt:
                    print(f"[WARN] device={device} 超出当前可见 GPU 数量（count={cnt}），将回退为 cuda:0")
                    device = "cuda:0"
        except Exception:
            # 若解析失败则保持原样，后续 `.to()` 会抛出更明确的错误
            pass

    spec = resolve_model_spec(model_id_or_key)
    dtype = _pick_dtype_for_spec(spec, device)
    device_map = runtime_cfg.device_map if runtime_cfg is not None else None
    max_memory = runtime_cfg.max_memory if runtime_cfg is not None else None

    # SD3/3.5 属于“connected pipeline”，在部分 diffusers/accelerate 组合下使用 device_map
    # 容易出现 CPU/CUDA 混用（例如 embedding 的 index_select 报错），优先走单卡 + CPU offload 更稳。
    # 如确需强制启用 device_map，可设置环境变量：DEMO_SD3_ALLOW_DEVICE_MAP=1
    if spec.loader == "sd3_img2img" and device_map is not None:
        allow = (os.environ.get("DEMO_SD3_ALLOW_DEVICE_MAP", "0") or "").strip().lower() in ("1", "true", "yes")
        if not allow:
            print("[WARN] SD3 img2img 检测到 device_map 配置：为避免 CPU/CUDA 混用，将忽略 device_map/max_memory。"
                  "如需强制启用，请设置 DEMO_SD3_ALLOW_DEVICE_MAP=1")
            device_map = None
            max_memory = None

    # SD3/3.5 模型体量大，cpu_offload="auto" 时先 `.to(cuda)` 很容易直接 OOM。
    # 若未使用 device_map 且用户未显式指定 offload 策略，默认改为 model offload 更稳。
    if spec.loader == "sd3_img2img" and device_map is None and runtime_cfg is not None:
        try:
            co = str(getattr(runtime_cfg, "cpu_offload", "auto") or "auto").lower().strip()
            if co == "auto":
                runtime_cfg.cpu_offload = "model"
        except Exception:
            pass

    extra_pretrained_kwargs: Dict[str, Any] = {}
    if device_map is not None:
        extra_pretrained_kwargs["device_map"] = device_map
    if max_memory is not None:
        extra_pretrained_kwargs["max_memory"] = max_memory
    use_safetensors = spec.use_safetensors
    allow_fallback = (use_safetensors is True)

    if spec.loader == "auto_inpaint":
        def _load_auto_inpaint(pretrained_kwargs: Dict[str, Any]):
            if spec.prefer_diffusion_pipeline:
                return _load_with_safetensors_fallback(
                    DiffusionPipeline.from_pretrained,
                    spec.model_id,
                    allow_fallback=allow_fallback,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors,
                    trust_remote_code=spec.trust_remote_code,
                    variant=spec.variant,
                    **pretrained_kwargs,
                )

            try:
                return _load_with_safetensors_fallback(
                    AutoPipelineForInpainting.from_pretrained,
                    spec.model_id,
                    allow_fallback=allow_fallback,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors,
                    trust_remote_code=spec.trust_remote_code,
                    variant=spec.variant,
                    **pretrained_kwargs,
                )
            except Exception as e:
                print(f"[WARN] AutoPipelineForInpainting 加载失败，尝试 DiffusionPipeline 兜底: {e}")
                return _load_with_safetensors_fallback(
                    DiffusionPipeline.from_pretrained,
                    spec.model_id,
                    allow_fallback=allow_fallback,
                    torch_dtype=dtype,
                    use_safetensors=use_safetensors,
                    trust_remote_code=spec.trust_remote_code,
                    variant=spec.variant,
                    **pretrained_kwargs,
                )

        try:
            # 对 SDXL/SD1.5 这类非 connected pipeline，diffusers 是支持 device_map 的
            pipe = _load_auto_inpaint(extra_pretrained_kwargs)
        except NotImplementedError as e:
            msg = str(e)
            # Kandinsky 等 connected pipeline：diffusers 明确不支持 device_map
            if device_map is not None and "connected pipeline" in msg:
                print(f"[WARN] auto_inpaint 检测到 connected pipeline，device_map 将被忽略: {e}")
                retry_kwargs = dict(extra_pretrained_kwargs)
                retry_kwargs.pop("device_map", None)
                retry_kwargs.pop("max_memory", None)
                pipe = _load_auto_inpaint(retry_kwargs)
            else:
                raise
        pipe = _pipe_to_device(pipe, device, runtime_cfg=runtime_cfg)

    elif spec.loader == "flux_fill_nf4":
        # 使用 diffusers/FLUX.1-Fill-dev-nf4 的 NF4 组件（transformer + text_encoder_2），
        # 再用 black-forest-labs/FLUX.1-Fill-dev 的其余组件拼装成 FluxFillPipeline。
        if FluxFillPipeline is None:
            raise RuntimeError("你的 diffusers 版本没有 FluxFillPipeline。请升级 diffusers（pip install -U diffusers）。")
        if FluxTransformer2DModel is None:
            raise RuntimeError(
                "你的 diffusers 版本没有 FluxTransformer2DModel。请升级 diffusers（pip install -U diffusers）。"
            )
        if T5EncoderModel is None:
            raise RuntimeError(
                "当前环境缺少 transformers 的 T5EncoderModel（或导入失败）。请安装/升级 transformers。"
            )

        nf4_repo = "diffusers/FLUX.1-Fill-dev-nf4"

        # 1) 加载 NF4 transformer（允许 device_map/max_memory）
        transformer = FluxTransformer2DModel.from_pretrained(
            nf4_repo,
            subfolder="transformer",
            torch_dtype=dtype,
            **extra_pretrained_kwargs,
        )

        # 2) 加载 NF4 text_encoder_2（T5）（transformers 支持 device_map/max_memory）
        text_encoder_2 = T5EncoderModel.from_pretrained(
            nf4_repo,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            **extra_pretrained_kwargs,
        )

        # 3) 用 base pipeline 补齐其它组件（tokenizer/vae/text_encoder/scheduler 等）
        pipe = FluxFillPipeline.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            **extra_pretrained_kwargs,
        )
        pipe = _pipe_to_device(pipe, device, runtime_cfg=runtime_cfg)

    elif spec.loader == "sdcpp_qwen_image_edit_2511_gguf":
        # stable-diffusion.cpp Python 绑定：在 Python 进程内常驻模型，避免每次 sd-cli 重载权重。
        _require_stable_diffusion_cpp_python()
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 huggingface_hub，无法下载 Qwen GGUF 权重。请安装：pip install -U huggingface_hub") from e

        # 解决 HF XET/CAS 下载失败：沿用 run.py 的策略（若用户没设，也尽量兜底）
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        def _dl(repo_id: str, filename: str) -> str:
            return hf_hub_download(repo_id=repo_id, filename=filename)

        # 选择：单卡 4090 的 4-bit 版本（质量/体积平衡）
        diffusion_path = _dl(spec.model_id, "qwen-image-edit-2511-Q4_K_M.gguf")

        # Qwen Image 系列的 VAE（Comfy-Org 提供 split_files）
        vae_path = _dl("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors")

        # Text encoder（Qwen2.5-VL 7B）+ vision tower（mmproj）
        llm_repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
        llm_path = _dl(llm_repo, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
        llm_vision_path = _dl(llm_repo, "mmproj-BF16.gguf")

        pipe = SdCppQwenImageEdit2511GGUF(
            diffusion_model_path=diffusion_path,
            llm_path=llm_path,
            llm_vision_path=llm_vision_path,
            vae_path=vae_path,
            offload_params_to_cpu=True,
            diffusion_flash_attn=True,
            flow_shift=3.0,
            qwen_image_zero_cond_t=True,
            enable_mmap=True,
            verbose=False,
        )

    elif spec.loader in ("diffusion_edit",):
        mid_low = (spec.model_id or "").lower()
        key_low = (spec.key or "").lower()
        is_flux2 = key_low.startswith("flux2") or ("flux.2" in mid_low) or ("flux2" in mid_low)
        is_kandinsky = ("kandinsky" in key_low) or ("kandinsky" in mid_low)
        load_fn = DiffusionPipeline.from_pretrained
        if is_kandinsky and ("i2i" in key_low or "i2i" in mid_low):
            if Kandinsky5I2IPipeline is None:
                raise RuntimeError(
                    "你的 diffusers 版本没有 Kandinsky5I2IPipeline，无法加载 Kandinsky5 I2I 模型。"
                    "请升级 diffusers：pip install -U diffusers"
                )
            load_fn = Kandinsky5I2IPipeline.from_pretrained
        if spec.key == "longcat_image_edit_turbo":
            _require_longcat_diffusers_support()
        _warmup_mod = None
        _warmup_backup = None
        if is_flux2:
            try:
                from diffusers.models import model_loading_utils as _model_loading_utils
                _warmup_mod = _model_loading_utils
                _warmup_backup = _model_loading_utils._caching_allocator_warmup
                _model_loading_utils._caching_allocator_warmup = lambda *args, **kwargs: None
            except Exception as e:
                print(f"[WARN] flux2 禁用 CUDA warmup 失败，将继续默认加载: {e}")
        try:
            pipe = _load_with_safetensors_fallback(
                load_fn,
                spec.model_id,
                allow_fallback=allow_fallback,
                torch_dtype=dtype,
                use_safetensors=use_safetensors,
                trust_remote_code=spec.trust_remote_code,
                variant=spec.variant,
                **extra_pretrained_kwargs,
            )
        finally:
            if _warmup_mod is not None and _warmup_backup is not None:
                _warmup_mod._caching_allocator_warmup = _warmup_backup
        if is_flux2 and device_map is None and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"[WARN] flux2 启用 CPU offload 失败，将回退为全量上 GPU: {e}")
                pipe = _pipe_to_device(pipe, device, runtime_cfg=runtime_cfg)
        else:
            pipe = _pipe_to_device(pipe, device, runtime_cfg=runtime_cfg)

    elif spec.loader == "sd3_img2img":
        if StableDiffusion3Img2ImgPipeline is None:
            raise RuntimeError("你的 diffusers 版本没有 StableDiffusion3Img2ImgPipeline。请升级 diffusers（pip install -U diffusers）。")
        pipe = _load_with_safetensors_fallback(
            StableDiffusion3Img2ImgPipeline.from_pretrained,
            spec.model_id,
            allow_fallback=allow_fallback,
            torch_dtype=dtype,
            **extra_pretrained_kwargs,
        )
        pipe = _pipe_to_device(pipe, device, runtime_cfg=runtime_cfg)

    else:
        raise ValueError(f"Unknown loader: {spec.loader}")

    # 绑一个属性，方便 debug / 复用时判断
    try:
        setattr(pipe, "_demo_model_spec", spec)
    except Exception:
        pass

    return pipe, device, spec


# 兼容旧接口：你的老代码/脚本可能还在调用 build_inpaint_pipe
def build_inpaint_pipe(model_id_or_key: str, device: str, runtime_cfg: Optional[DemoConfig] = None):
    pipe, device, _spec = build_pipe(model_id_or_key, device, runtime_cfg=runtime_cfg)
    return pipe, device


def _get_pipe_param_dtype_device(pipe) -> Tuple[torch.dtype, str]:
    """
    不同 pipeline 内部模块名字不同：unet / transformer / etc.
    这里只为推断 pipeline 主要参数的 dtype/device（用于推断实际 device 等）。
    """
    # 常见顺序：unet（SD/SDXL）, transformer（flux/sd3）
    for attr in ("unet", "transformer"):
        if hasattr(pipe, attr):
            m = getattr(pipe, attr)
            try:
                p = next(m.parameters())
                return p.dtype, str(p.device)
            except Exception:
                pass
    # 兜底
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.float16 if dev.startswith("cuda") else torch.float32), dev


# =========================
# 7) 通用：按 signature 过滤 kwargs（避免不同 pipeline 参数不一致）
# =========================
def _call_pipe_filtered(pipe, **kwargs):
    sig = inspect.signature(pipe.__call__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**filtered)


def _to_pil_image(x) -> Image.Image:
    """把 pipeline 输出的单张结果统一转换成 PIL.Image（尽量稳健）。"""
    if isinstance(x, Image.Image):
        return x

    # torch tensor -> PIL
    if torch.is_tensor(x):
        t = x.detach().cpu()
        # 可能是 [B,C,H,W]
        if t.ndim == 4:
            t = t[0]
        # [C,H,W] -> [H,W,C]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        # [H,W] -> [H,W,1]
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.ndim != 3 or t.shape[-1] not in (1, 3):
            raise RuntimeError(f"Unsupported torch image tensor shape: {tuple(t.shape)}")

        arr = t.numpy()
        if np.issubdtype(arr.dtype, np.floating):
            # 兼容 [-1,1] 或 [0,1]
            if float(np.nanmin(arr)) < 0.0:
                arr = (arr + 1.0) / 2.0
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            # 兼容 uint8 / int
            if arr.max() <= 1:
                arr = (arr.astype(np.float32) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[-1] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        return Image.fromarray(arr, mode="RGB")

    # numpy -> PIL
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    if float(np.nanmin(arr)) < 0.0:
                        arr = (arr + 1.0) / 2.0
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255.0).round().astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                return Image.fromarray(arr, mode="RGBA").convert("RGB")
            if arr.shape[-1] == 1:
                return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
            return Image.fromarray(arr, mode="RGB")
        raise RuntimeError(f"Unsupported numpy image shape: {arr.shape}")

    raise RuntimeError(f"Unsupported image type from pipeline output: {type(x)}")


def _extract_images_from_pipe_output(output) -> List[Image.Image]:
    """
    兼容不同 pipeline 的输出字段：
    - 大多数 diffusers: output.images
    - Kandinsky5: output.image
    - 或 tuple/dict
    """
    if output is None:
        raise RuntimeError("Pipeline returned None.")

    raw = None
    if hasattr(output, "images"):
        raw = getattr(output, "images")
    elif hasattr(output, "image"):
        raw = getattr(output, "image")
    elif isinstance(output, dict):
        raw = output.get("images", output.get("image", None))
    elif isinstance(output, tuple) and len(output) > 0:
        raw = output[0]

    if raw is None:
        raise RuntimeError(f"Unsupported pipeline output object: {type(output)}")

    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]

    out: List[Image.Image] = []
    for it in items:
        out.append(_to_pil_image(it))
    return out


# =========================
# 7.5) 统一外壳：能力探测（A/B/C）
# =========================
def _pipe_supports_mask_inpaint(pipe) -> bool:
    """A：原生 mask inpaint（__call__ 支持 mask_image）。"""
    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    return "mask_image" in sig.parameters

# =========================
# 8) 一次推理：inpaint
# =========================
def run_inpaint_once(
    pipe,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    strength: float,
    device: str,
    extra_call_kwargs: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """
    统一 inpaint 调用，尽量兼容：
    - SD/SDXL/Kandinsky2.2 的 AutoPipelineForInpainting
    - FLUX Fill 的 FluxFillPipeline（参数子集不同）
    - 不做任何身份锁定/一致性注入
    """
    w, h = image_pil.size

    # 许多 pipeline 用 generator 控制随机性；FLUX 官方示例用 CPU generator
    gen_dev = "cpu" if "FluxFillPipeline" in pipe.__class__.__name__ else device
    generator = torch.Generator(gen_dev).manual_seed(seed)

    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        height=h,
        width=w,
        generator=generator,
    )

    # 额外参数（如 FLUX 的 max_sequence_length）
    if extra_call_kwargs:
        kwargs.update(extra_call_kwargs)

    result = _call_pipe_filtered(pipe, **kwargs)
    return _extract_images_from_pipe_output(result)[0]


# =========================
# 8.5) ROI inpaint：用于 FLUX 等，减少计算量
# =========================
def run_roi_inpaint_once(
    pipe,
    full_img_pil: Image.Image,
    full_mask_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    strength: float,
    device: str,
    pad_ratio: float,
    pad_multiple: int = 8,
    roi_max_side: int = 1024,
    model_spec: Optional[ModelSpec] = None,
    extra_call_kwargs: Optional[Dict[str, Any]] = None,
    enable_anonymize: bool = False,
) -> Image.Image:
    """
    只对人脸 ROI 做 inpaint，再贴回全图，显著减少计算量。
    ROI 处理逻辑尽量与 sd35_img2img 保持一致（动态 pad + 小脸上采样）。
    """
    W, H = full_img_pil.size
    bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    min_face = max(1, min(bw, bh))
    if min_face < 120:
        local_pad_ratio = 0.30
    elif min_face < 200:
        local_pad_ratio = 0.38
    else:
        local_pad_ratio = 0.45
    if pad_ratio > 0:
        local_pad_ratio = max(local_pad_ratio, pad_ratio)

    mask_bbox = _bbox_from_mask(full_mask_pil, threshold=128)
    base_bbox = _bbox_union(bbox, mask_bbox) if mask_bbox is not None else bbox
    X1, Y1, X2, Y2 = _bbox_expand_and_clip(base_bbox, W, H, pad_ratio=local_pad_ratio)

    roi_img = full_img_pil.crop((X1, Y1, X2, Y2)).convert("RGB")
    roi_mask = full_mask_pil.crop((X1, Y1, X2, Y2)).convert("L")
    roi_img_orig = roi_img
    roi_mask_orig = roi_mask

    roi_w, roi_h = roi_img.size
    min_dim = min(roi_w, roi_h)
    # 小脸上采样：提高可分辨细节，避免生成不全/发糊
    # NOTE:
    # - SDXL 系列在 ROI 分辨率过低时，容易在脸部生成明显的“彩色面具/抽象纹理”伪影。
    # - 这里对 SDXL 适当提高目标短边分辨率，并允许更大的上采样倍率；
    #   同时用 max_side 限制最长边，避免极端小脸导致过高计算/爆显存。
    spec_key = (getattr(model_spec, "key", "") or "").lower() if model_spec is not None else ""
    spec_mid = (getattr(model_spec, "model_id", "") or "").lower() if model_spec is not None else ""
    pipe_name = getattr(pipe, "__class__", type(pipe)).__name__.lower()
    is_sdxl = (
        ("sdxl" in spec_key)
        or ("stable-diffusion-xl" in spec_mid)
        or ("sdxl" in spec_mid)
        or ("diffusionxl" in pipe_name)  # StableDiffusionXL(Inpaint)Pipeline
        or ("sdxl" in pipe_name)
    )

    target_short_side = 768 if is_sdxl else 512
    # 小脸时：非 SDXL 模型也需要更大的上采样倍率，否则脸像素始终不够（导致糊/假/不稳定）
    if min_face < 120:
        max_scale = 8.0
    elif min_face < 200:
        max_scale = 4.0
    else:
        max_scale = 2.5
    if is_sdxl:
        max_scale = max(float(max_scale), 6.0)
    # 统一加上 ROI 最长边上限：
    # - 防止极端小脸上采样过大
    # - 也防止“原本就很大的 ROI”不经缩放直接推理触发显存峰值 / OOM
    _roi_max_side: Optional[int] = int(roi_max_side) if roi_max_side is not None else 1024
    if _roi_max_side <= 0:
        _roi_max_side = None

    # 同时满足两件事：
    # 1) ROI 的短边 >= target_short_side
    # 2) 人脸本身（bbox 短边）在生成空间里达到最小像素数（避免“脸太小模型没法画”）
    target_face_side = 220
    scale_dim = float(target_short_side) / float(max(1, min_dim)) if min_dim < target_short_side else 1.0
    scale_face = float(target_face_side) / float(max(1, min_face)) if min_face < target_face_side else 1.0
    scale = float(max(1.0, scale_dim, scale_face))
    scale = float(min(scale, float(max_scale)))
    new_w = int(round(roi_w * scale))
    new_h = int(round(roi_h * scale))
    if _roi_max_side is not None:
        max_dim = max(new_w, new_h)
        if max_dim > int(_roi_max_side):
            s2 = float(_roi_max_side) / float(max_dim)
            new_w = int(round(new_w * s2))
            new_h = int(round(new_h * s2))
            scale *= s2

    new_w = max(64, new_w)
    new_h = max(64, new_h)
    if (new_w, new_h) != (roi_w, roi_h):
        roi_img = roi_img.resize((new_w, new_h), resample=Image.LANCZOS)
        roi_mask = roi_mask.resize((new_w, new_h), resample=Image.BILINEAR)
        roi_w, roi_h = roi_img.size

    is_qwen = (model_spec is not None) and (getattr(model_spec, "loader", "") == "sdcpp_qwen_image_edit_2511_gguf")
    if enable_anonymize and is_qwen:
        try:
            roi_np = np.array(roi_img).astype(np.uint8)
            m_soft = np.array(roi_mask.convert("L")).astype(np.uint8)
            m_bin = (m_soft >= 128).astype(np.uint8) * 255
            min_dim2 = max(1, int(min(roi_w, roi_h)))
            size_factor = min(1.0, float(min_dim2) / 512.0)
            strength_01 = float(np.clip(float(strength), 0.0, 1.0))
            # Qwen 很容易“抠回原身份”，这里提高去特征化强度下限
            strength_01 = float(max(strength_01, 0.85))
            # strength 越大，抹除越强（更容易换出“不同身份”）
            # 方案A：扩大可编辑范围后，这里只做“轻微收紧”的去特征化区域，
            # 避免把刘海/耳朵等也强行抹掉；真正的 inpaint mask 仍使用更宽的 roi_mask（二值化）。
            # 这里尽量少 erosion，让去特征化覆盖到更多脸部区域，提升身份替换强度
            inner_erode_ratio = max(0.006, (0.012 + 0.012 * strength_01) * size_factor)
            m_inner = _erode_mask(m_bin, ratio=inner_erode_ratio)
            # 去特征化输入：更强的 gaussian 去细节，让模型更愿意替换身份
            roi_np = _apply_mask_obfuscate_rgb(roi_np, m_inner, strength_01=strength_01, mode="gaussian")
            roi_img = Image.fromarray(roi_np)
            # 推理 mask 用二值更稳（允许刘海/耳朵等轻度参与编辑）
            roi_mask = Image.fromarray(m_bin).convert("L")
        except Exception as e:
            print(f"[WARN] 匿名化预处理失败，将继续直接 inpaint: {e}")

    roi_img_pad, roi_mask_pad, roi_pad = pad_image_and_mask_to_multiple(
        roi_img, roi_mask, multiple=pad_multiple
    )

    out_roi_pad = run_inpaint_once(
        pipe=pipe,
        image_pil=roi_img_pad,
        mask_pil=roi_mask_pad,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        guidance=guidance,
        strength=strength,
        device=device,
        extra_call_kwargs=extra_call_kwargs,
    )
    out_roi = crop_back_to_original(out_roi_pad, roi_pad, roi_img.size)
    if out_roi.size != roi_img_orig.size:
        out_roi = out_roi.resize(roi_img_orig.size, resample=Image.LANCZOS)

    # 贴回：mask 外背景保持原图不变（使用 DT-feather alpha，避免“方框接缝”）
    base_rgb = np.array(full_img_pil.convert("RGB"))
    out_rgb = base_rgb.copy()
    gen_patch = np.array(out_roi.convert("RGB"))
    base_patch = base_rgb[Y1:Y2, X1:X2, :]
    alpha_u8 = np.array(roi_mask_orig.convert("L"), dtype=np.uint8)

    if gen_patch.shape[:2] != base_patch.shape[:2]:
        gen_patch = cv2.resize(gen_patch, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if alpha_u8.shape[:2] != base_patch.shape[:2]:
        alpha_u8 = cv2.resize(alpha_u8, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 1) 硬锁定：mask 外区域强制使用原图，避免 ROI/edit 模型在背景上留下矩形痕迹
    hard_u8 = (alpha_u8 >= 128).astype(np.uint8) * 255
    hard = (hard_u8.astype(np.float32) / 255.0)[..., None]
    gen_patch = (gen_patch.astype(np.float32) * hard + base_patch.astype(np.float32) * (1.0 - hard))
    gen_patch = np.clip(gen_patch, 0, 255).astype(np.uint8)

    # 模型特定修正：Qwen 输出常偏糊/偏色，做轻量锐化 + 色调匹配（mask 内）
    if (model_spec is not None) and (getattr(model_spec, "loader", "") == "sdcpp_qwen_image_edit_2511_gguf"):
        try:
            mask_bin = hard_u8

            gen_patch = _unsharp_mask_rgb(gen_patch, sigma=1.0, amount=0.18)
            matched = _match_color_mean_std(gen_patch, base_patch, mask_bin)
            alpha_cm = 0.45
            gen_patch = (
                gen_patch.astype(np.float32) * (1.0 - alpha_cm)
                + matched.astype(np.float32) * alpha_cm
            )
            gen_patch = np.clip(gen_patch, 0, 255).astype(np.uint8)
        except Exception:
            pass

    # 2) Feather alpha：用距离变换生成固定宽度的过渡带，减少硬边/方框割裂
    strength_01 = float(np.clip(float(strength), 0.0, 1.0))
    feather_px = int(np.clip(float(min_face) * (0.08 + 0.02 * strength_01), 6.0, 36.0))
    alpha = _make_feather_alpha_dt(hard_u8, feather_px=feather_px, blur_sigma=0.8)
    if alpha.shape[:2] != base_patch.shape[:2]:
        alpha = cv2.resize(alpha, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
    a3 = alpha[..., None]
    blended = base_patch.astype(np.float32) * (1.0 - a3) + gen_patch.astype(np.float32) * a3
    out_rgb[Y1:Y2, X1:X2, :] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(out_rgb)


# =========================
# 9) ROI img2img：用于 SD3.5 这类“没标准 inpaint”的情况
# =========================
def _bbox_expand_and_clip(
    bbox: Tuple[int, int, int, int],
    W: int,
    H: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)
    X1 = max(0, x1 - pad_w)
    Y1 = max(0, y1 - pad_h)
    X2 = min(W, x2 + pad_w)
    Y2 = min(H, y2 + pad_h)
    return X1, Y1, X2, Y2


def run_roi_edit_once(
    pipe,
    full_img_pil: Image.Image,
    full_mask_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    strength: float,
    pad_ratio: float,
    pad_multiple: int = 8,
    roi_max_side: int = 1024,
    extra_call_kwargs: Optional[Dict[str, Any]] = None,
    face_app: Optional[FaceAnalysis] = None,
) -> Image.Image:
    """
    无原生 mask 的编辑模型：ROI 裁剪 -> img2img -> 贴回。
    """
    pipe_name = pipe.__class__.__name__
    is_longcat_edit = ("LongCatImageEditPipeline" in pipe_name) or ("longcat" in pipe_name.lower())
    is_qwen_image_edit = ("qwen" in pipe_name.lower()) and ("edit" in pipe_name.lower())
    is_flux2 = ("flux2" in pipe_name.lower())
    is_kandinsky = ("kandinsky" in pipe_name.lower())

    # 对指令式编辑模型：强约束“姿态/视角/目光方向”，降低侧脸被“正脸化”或五官整体漂移的概率。
    # 说明：这里作为统一的“后缀约束”附加到 prompt 上，即使用户 prompt 已包含 replace/swap 也能生效。
    pose_lock_suffix = (
        " Keep the same head pose, viewing angle and gaze direction as the input image, and keep the same facial expression. "
        "Do not rotate the head/face towards the camera (no frontalization), and do not shift facial features. "
        "If the input is a profile/side view, keep the same profile/side view. "
        "Keep geometry aligned (no shifting or resizing). "
    )

    if is_longcat_edit:
        # LongCat-Image-Edit-Turbo 是指令式编辑模型：更偏好“明确编辑指令”，且 negative_prompt 过长容易干扰。
        negative_prompt = ""
        # Turbo 模型卡推荐：steps=8, guidance=1
        if steps <= 0 or steps > 16:
            steps = 8
        if guidance <= 0 or guidance > 3.0:
            guidance = 1.0

        p_low = (prompt or "").lower()
        has_edit_intent = any(k in p_low for k in (
            "replace", "swap", "change", "edit", "anonymize", "de-identify", "identity",
            "替换", "换脸", "更换", "修改", "匿名", "脱敏", "去标识",
        ))
        if not has_edit_intent:
            prompt = (
                "Replace the person's face with a different realistic face (different identity). "
                "Keep the same head pose, viewing angle and gaze direction, and keep the same facial expression and lighting. "
                "If the input is a profile/side view, keep the same profile/side view (no frontalization). "
                "Keep hairstyle, ears, neck, clothing and background unchanged. "
                "Primarily modify the inner face region; nearby hair fringe may be adjusted slightly for natural blending. "
                "The identity must be clearly different from the input (do not preserve recognizable facial features). "
                "Photo-realistic, natural skin texture, sharp eyes, high detail, no smears, no artifacts, no watermark, no text. "
            ) + prompt
        # 无论用户是否写了 edit intent，都补充姿态锁定约束（避免侧脸/目光被改）
        prompt = (prompt or "") + pose_lock_suffix

    if is_kandinsky:
        # Kandinsky 5 I2I Lite：对输入图像的“保留倾向”很强，需要更明确的编辑指令。
        # 官方示例 guidance_scale 默认 3.5；guidance 过高更容易导致结构漂移。
        # 同时 negative_prompt 不宜过长；这里补一段“拒绝相似身份”的短负面词，提升替换强度。
        neg_id = "same person, same identity, look alike, recognizable, original face, identical facial features"
        negative_prompt = (negative_prompt or "").strip()
        negative_prompt = (negative_prompt + ", " + neg_id).strip(", ") if negative_prompt else neg_id
        if steps <= 0 or steps > 80:
            steps = 35
        if guidance is None or guidance <= 0:
            guidance = 4.5
        if guidance > 7.0:
            guidance = 7.0

        p_low = (prompt or "").lower()
        has_edit_intent = any(k in p_low for k in (
            "replace", "swap", "change", "edit", "anonymize", "de-identify", "identity",
            "替换", "换脸", "更换", "修改", "匿名", "脱敏", "去标识",
        ))
        if not has_edit_intent:
            prompt = (
                "Using the input image as the base photo, replace only the person's face identity with a different realistic person (different identity). "
                "Keep the same head pose, viewing angle and gaze direction, and keep the same facial expression and lighting. "
                "If the input is a profile/side view, keep the same profile/side view (no frontalization). "
                "Primarily modify the inner face region and keep all geometry aligned (no shifting). "
                "Keep the same age, gender and approximate skin tone as the input image; do not make the person younger or older. "
                "The identity must be clearly different from the input; do not preserve recognizable identity cues (eyes, nose, mouth, jawline should differ). "
                "Photo-realistic, natural skin texture, sharp eyes, high detail. "
            ) + (prompt or "")
        prompt = (prompt or "") + pose_lock_suffix

    if is_qwen_image_edit:
        # Qwen Image Edit 更偏“指令式编辑”，并且推荐 cfg_scale 较低（sd.cpp 示例通常 2.5）。
        negative_prompt = ""
        if steps <= 0 or steps > 80:
            steps = 20
        if guidance <= 0 or guidance > 5.0:
            guidance = 2.5

        p_low = (prompt or "").lower()
        has_edit_intent = any(k in p_low for k in (
            "replace", "swap", "change", "edit", "anonymize", "de-identify", "identity",
            "替换", "换脸", "更换", "修改", "匿名", "脱敏", "去标识",
        ))
        if not has_edit_intent:
            prompt = (
                "Replace the person's face with a different realistic face (different identity). "
                "Keep the same head pose, viewing angle and gaze direction, and keep the same facial expression and lighting. "
                "If the input is a profile/side view, keep the same profile/side view (no frontalization). "
                "Keep hairstyle, ears, neck, clothing and background unchanged. Only modify the inner face region. "
                "Photo-realistic, natural skin texture, sharp eyes, high detail. "
            ) + prompt
        prompt = (prompt or "") + pose_lock_suffix

    if is_flux2:
        # FLUX.2 系列是 4 steps distilled，cfg_scale 推荐 1.0；negative_prompt 通常不需要。
        negative_prompt = ""
        if steps <= 0 or steps > 16:
            steps = 4
        # Flux2Pipeline 默认 guidance_scale=4.0；这里不过度限制，避免编辑指令不生效
        if guidance is None or guidance <= 0:
            guidance = 4.0
        if guidance > 10.0:
            guidance = 10.0

        p_low = (prompt or "").lower()
        has_edit_intent = any(k in p_low for k in (
            "replace", "swap", "change", "edit", "anonymize", "de-identify", "identity",
            "替换", "换脸", "更换", "修改", "匿名", "脱敏", "去标识",
        ))
        if not has_edit_intent:
            prompt = (
                "Replace the person's face with a different realistic face (different identity). "
                "Keep the same head pose, viewing angle and gaze direction, and keep the same facial expression and lighting. "
                "If the input is a profile/side view, keep the same profile/side view (no frontalization). "
                "Keep hairstyle, ears, neck, clothing and background unchanged. Primarily modify the inner face region. "
                "Photo-realistic, youthful-looking, clear smooth skin, even complexion, "
                "soft diffused lighting, sharp eyes, natural-looking details. "
            ) + (prompt or "")
        prompt = (prompt or "") + pose_lock_suffix

    W, H = full_img_pil.size
    bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    min_face = max(1, min(bw, bh))
    if min_face < 120:
        local_pad_ratio = 0.30
    elif min_face < 200:
        local_pad_ratio = 0.38
    else:
        local_pad_ratio = 0.45
    if pad_ratio > 0:
        local_pad_ratio = max(local_pad_ratio, pad_ratio)

    mask_bbox = _bbox_from_mask(full_mask_pil, threshold=128)
    base_bbox = _bbox_union(bbox, mask_bbox) if mask_bbox is not None else bbox
    X1, Y1, X2, Y2 = _bbox_expand_and_clip(base_bbox, W, H, pad_ratio=local_pad_ratio)

    roi_img = full_img_pil.crop((X1, Y1, X2, Y2)).convert("RGB")
    roi_mask = full_mask_pil.crop((X1, Y1, X2, Y2)).convert("L")
    roi_img_orig = roi_img
    roi_mask_orig = roi_mask

    roi_w, roi_h = roi_img.size
    min_dim = min(roi_w, roi_h)
    _roi_max_side: Optional[int] = int(roi_max_side) if roi_max_side is not None else 1024
    if _roi_max_side <= 0:
        _roi_max_side = None

    # Kandinsky5I2IPipeline 只支持固定 resolutions，且会按“最接近宽高比”强制 resize。
    # 若我们给任意 ROI 尺寸，再把输出 resize 回来，会引入拉伸/模糊/结构错位。
    # 这里先把 ROI 主动对齐到支持的分辨率集合，减少重复 resize 带来的细节损失。
    if is_kandinsky:
        try:
            tgt_w, tgt_h = _pick_kandinsky_resolution(pipe, roi_w, roi_h)
            if (tgt_w, tgt_h) != (roi_w, roi_h):
                roi_img = roi_img.resize((tgt_w, tgt_h), resample=Image.LANCZOS)
                roi_mask = roi_mask.resize((tgt_w, tgt_h), resample=Image.BILINEAR)
                roi_w, roi_h = roi_img.size
                min_dim = min(roi_w, roi_h)
        except Exception:
            pass

    # 小脸上采样：提高可分辨细节，避免生成不全/发糊
    # 对编辑模型（ROI-edit）而言，小脸最常见问题是“脸像素始终不足”，因此这里按 min_face 做兜底上采样。
    scale = 1.0
    if not is_kandinsky:
        min_roi_side = 512
        if min_face < 120:
            max_scale = 8.0
        elif min_face < 200:
            max_scale = 4.0
        else:
            max_scale = 2.5
        target_face_side = 220

        scale_dim = float(min_roi_side) / float(max(1, min_dim)) if min_dim < min_roi_side else 1.0
        scale_face = float(target_face_side) / float(max(1, min_face)) if min_face < target_face_side else 1.0
        scale = float(max(1.0, scale_dim, scale_face))
        scale = float(min(scale, float(max_scale)))

    # 统一加上 ROI 最长边上限：避免“原本就很大的 ROI”不经缩放直接推理触发显存峰值 / OOM
    new_w = int(round(roi_w * scale))
    new_h = int(round(roi_h * scale))
    if _roi_max_side is not None:
        max_dim = max(new_w, new_h)
        if max_dim > int(_roi_max_side):
            s2 = float(_roi_max_side) / float(max_dim)
            new_w = int(round(new_w * s2))
            new_h = int(round(new_h * s2))
            scale *= s2

    new_w = max(64, new_w)
    new_h = max(64, new_h)
    if (new_w, new_h) != (roi_w, roi_h):
        roi_img = roi_img.resize((new_w, new_h), resample=Image.LANCZOS)
        roi_mask = roi_mask.resize((new_w, new_h), resample=Image.BILINEAR)
        roi_w, roi_h = roi_img.size
        min_dim = min(roi_w, roi_h)

    if is_kandinsky:
        # Kandinsky I2I 不支持 strength 参数，但 image 条件很强：
        # 这里把 mask 内细节“抹掉”，让模型更容易生成不同身份，同时保留几何结构避免位移。
        roi_np = np.array(roi_img).astype(np.uint8)
        m_soft = np.array(roi_mask.convert("L")).astype(np.uint8)
        # 二值化更稳，避免边缘“半编辑”导致一圈发糊/阴影
        m_np = (m_soft >= 128).astype(np.uint8) * 255
        size_factor = min(1.0, float(min_dim) / 512.0)
        strength_01 = float(np.clip(float(strength), 0.0, 1.0))
        # Kandinsky 输出偏“保留输入”，这里给一个温和的去特征化下限；
        # 过强的高斯去细节会导致模型“保留模糊输入”→ 输出整体发糊。
        strength_01 = float(max(strength_01, 0.70))
        # 尽量少 erosion，让去特征化覆盖到更多脸部区域，提升身份替换强度
        inner_erode_ratio = max(0.006, (0.012 + 0.012 * strength_01) * size_factor)
        m_inner = _erode_mask(m_np, ratio=inner_erode_ratio)
        # 去特征化输入：用边缘保留的 bilateral 更利于保留清晰度（避免“糊脸”）
        roi_np = _apply_mask_obfuscate_rgb(roi_np, m_inner, strength_01=strength_01, mode="bilateral")
        roi_img = Image.fromarray(roi_np)

    if is_longcat_edit:
        # 给 LongCat 一个“去特征化但保结构”的脸部输入：
        # - 避免强高斯把侧脸结构/目光方向抹掉 → 模型更容易“正脸化”
        # - 改为：保边(bilateral)去纹理 + 轻量高斯做少量抹细节
        roi_np = np.array(roi_img).astype(np.uint8)
        m_soft = np.array(roi_mask.convert("L")).astype(np.uint8)
        m_np = (m_soft >= 128).astype(np.uint8) * 255
        size_factor = min(1.0, float(min_dim) / 512.0)
        strength_01 = float(np.clip(float(strength), 0.0, 1.0))
        # 给一个下限：避免去特征化太弱导致“抠回原身份”
        strength_01 = float(max(strength_01, 0.70))
        # strength 越大，抹除越强（更容易换出“不同身份”）
        inner_erode_ratio = max(0.010, (0.018 + 0.016 * strength_01) * size_factor)
        m_inner = _erode_mask(m_np, ratio=inner_erode_ratio)
        roi_np = _apply_mask_obfuscate_rgb(roi_np, m_inner, strength_01=strength_01, mode="bilateral")
        roi_np = _apply_mask_blur_rgb(
            roi_np,
            m_inner,
            blur_sigma=float(2.0 + 1.0 * strength_01),
            alpha_sigma=1.2,
        )
        roi_img = Image.fromarray(roi_np)

    if is_flux2:
        # Flux2(Klein/Dev) 的 image 条件很强，且 pipeline 不支持 strength；
        # 这里把脸内部细节“去特征化”，让模型更愿意替换身份，但尽量保留姿态/目光/侧脸结构。
        # 关键：避免强高斯（会抹掉侧脸结构 → 更容易 frontalization），改为保边 + 轻量模糊。
        roi_np = np.array(roi_img).astype(np.uint8)
        m_soft = np.array(roi_mask.convert("L")).astype(np.uint8)
        m_np = (m_soft >= 128).astype(np.uint8) * 255
        size_factor = min(1.0, float(min_dim) / 512.0)
        strength_01 = float(np.clip(float(strength), 0.0, 1.0))
        # 给一个温和下限：避免去特征化太弱导致身份替换不明显
        strength_01 = float(max(strength_01, 0.55))
        # strength 越大，抹除越强（更容易换出“不同身份”）
        inner_erode_ratio = max(0.010, (0.018 + 0.014 * strength_01) * size_factor)
        m_inner = _erode_mask(m_np, ratio=inner_erode_ratio)
        roi_np = _apply_mask_obfuscate_rgb(roi_np, m_inner, strength_01=strength_01, mode="bilateral")
        roi_np = _apply_mask_blur_rgb(
            roi_np,
            m_inner,
            blur_sigma=float(2.5 + 1.5 * strength_01),
            alpha_sigma=1.4,
        )
        roi_img = Image.fromarray(roi_np)

    roi_img_pad, roi_mask_pad, roi_pad = pad_image_and_mask_to_multiple(
        roi_img, roi_mask, multiple=pad_multiple
    )

    generator = torch.Generator("cpu").manual_seed(seed)
    image_arg: Any = roi_img_pad

    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_arg,
        # Qwen Image Edit（stable-diffusion.cpp）支持 mask_image：
        # 让模型只在 mask 内编辑，可显著降低“脸部 patch 与背景割裂/接缝”的概率。
        mask_image=roi_mask_pad if is_qwen_image_edit else None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        height=roi_img_pad.size[1],
        width=roi_img_pad.size[0],
        generator=generator,
    )
    if extra_call_kwargs:
        kwargs.update(extra_call_kwargs)

    out_roi_pad = _extract_images_from_pipe_output(_call_pipe_filtered(pipe, **kwargs))[0]
    # 某些 I2I pipeline（例如 Kandinsky5I2I）会把 height/width 映射到固定 resolutions，
    # 导致输出尺寸与输入 ROI pad 尺寸不一致；这里统一 resize 回来，避免后续 crop/paste 错位。
    if getattr(out_roi_pad, "size", None) is not None and out_roi_pad.size != roi_img_pad.size:
        out_roi_pad = out_roi_pad.resize(roi_img_pad.size, resample=Image.LANCZOS)
    out_roi = crop_back_to_original(out_roi_pad, roi_pad, roi_img.size)
    # 无论是小脸上采样还是 Kandinsky 固定 resolutions，对齐后统一 resize 回原 ROI 大小（保证贴回不变形）
    if out_roi.size != roi_img_orig.size:
        out_roi = out_roi.resize(roi_img_orig.size, resample=Image.LANCZOS)

    # 关键点对齐：修复编辑模型导致的“脸部整体偏移/错位”
    if face_app is not None:
        try:
            out_roi = _maybe_align_face_by_kps5(face_app=face_app, in_roi=roi_img_orig, out_roi=out_roi)
        except Exception:
            pass

    # 贴回：mask 外背景保持原图不变（使用 DT-feather alpha，避免“方框接缝”）
    base_rgb = np.array(full_img_pil.convert("RGB"))
    out_rgb = base_rgb.copy()
    gen_patch = np.array(out_roi.convert("RGB"))
    base_patch = base_rgb[Y1:Y2, X1:X2, :]
    alpha_u8 = np.array(roi_mask_orig.convert("L"), dtype=np.uint8)

    if gen_patch.shape[:2] != base_patch.shape[:2]:
        gen_patch = cv2.resize(gen_patch, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if alpha_u8.shape[:2] != base_patch.shape[:2]:
        alpha_u8 = cv2.resize(alpha_u8, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 1) 硬锁定：mask 外区域强制使用原图，避免 ROI/edit 模型在背景上留下矩形痕迹
    hard_u8 = (alpha_u8 >= 128).astype(np.uint8) * 255
    hard = (hard_u8.astype(np.float32) / 255.0)[..., None]
    gen_patch = (gen_patch.astype(np.float32) * hard + base_patch.astype(np.float32) * (1.0 - hard))
    gen_patch = np.clip(gen_patch, 0, 255).astype(np.uint8)

    # 模型特定修正：Kandinsky/Qwen 输出常偏糊/偏色，做轻量锐化 + 色调匹配（mask 内）
    if is_kandinsky or is_qwen_image_edit or is_flux2:
        try:
            mask_bin = hard_u8

            if is_kandinsky:
                gen_patch = _unsharp_mask_rgb(gen_patch, sigma=1.0, amount=0.45)
                alpha_cm = 0.65
            elif is_qwen_image_edit:
                gen_patch = _unsharp_mask_rgb(gen_patch, sigma=1.0, amount=0.18)
                alpha_cm = 0.45
            else:
                # FLUX2：常见问题是肤色/亮度轻微不一致，做轻量色调匹配可显著降低割裂感
                alpha_cm = 0.35

            matched = _match_color_mean_std(gen_patch, base_patch, mask_bin)
            gen_patch = (
                gen_patch.astype(np.float32) * (1.0 - alpha_cm)
                + matched.astype(np.float32) * alpha_cm
            )
            gen_patch = np.clip(gen_patch, 0, 255).astype(np.uint8)
        except Exception:
            pass

    # 2) Feather alpha：ROI-edit 更容易产生轻微漂移，直接对 hard mask 做 feather
    # 往往会在发际线/轮廓处混出“双边缘”。这里对 LongCat/Kandinsky/Flux2 启用“变化引导”的更稳融合。
    strength_01 = float(np.clip(float(strength), 0.0, 1.0))
    if is_longcat_edit or is_kandinsky or is_flux2:
        alpha = _make_change_guided_feather_alpha(
            base_patch_u8=base_patch,
            gen_patch_u8=gen_patch,
            hard_u8=hard_u8,
            strength_01=strength_01,
            min_face=int(min_face),
            diff_thr=28,
        )
    else:
        feather_px = int(np.clip(float(min_face) * (0.08 + 0.02 * strength_01), 6.0, 36.0))
        alpha = _make_feather_alpha_dt(hard_u8, feather_px=feather_px, blur_sigma=0.8)
    if alpha.shape[:2] != base_patch.shape[:2]:
        alpha = cv2.resize(alpha, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
    a3 = alpha[..., None]
    blended = base_patch.astype(np.float32) * (1.0 - a3) + gen_patch.astype(np.float32) * a3
    out_rgb[Y1:Y2, X1:X2, :] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(out_rgb)


def _apply_mask_blur_rgb(
    img_rgb: np.ndarray,
    mask_l: np.ndarray,
    blur_sigma: float,
    alpha_sigma: float,
) -> np.ndarray:
    """只在 mask 内做软融合模糊，用于“抹掉身份细节但保留结构”"""
    m = mask_l > 5
    if int(m.sum()) < 16:
        return img_rgb
    alpha = np.clip(mask_l.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(alpha_sigma), sigmaY=float(alpha_sigma))
    a3 = alpha[..., None]
    base = img_rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma)).astype(np.float32)
    out = base * (1.0 - a3) + blurred * a3
    return np.clip(out, 0, 255).astype(np.uint8)


def _unsharp_mask_rgb(img_rgb_u8: np.ndarray, sigma: float = 1.0, amount: float = 0.25) -> np.ndarray:
    """轻量反卷积锐化（unsharp mask），用于缓解部分编辑模型输出发糊。"""
    if img_rgb_u8 is None:
        raise ValueError("img_rgb_u8 is None")
    if float(amount) <= 1e-6:
        return img_rgb_u8
    blur = cv2.GaussianBlur(img_rgb_u8, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    out = cv2.addWeighted(img_rgb_u8, 1.0 + float(amount), blur, -float(amount), 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _match_color_mean_std(
    src_rgb_u8: np.ndarray,
    ref_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    在 mask 内对 src 做 per-channel 均值/方差匹配到 ref（轻量色调校正）。
    注意：只做颜色统计匹配，不改变几何结构；常用于压“偏灰/偏白/偏黄”的割裂感。
    """
    if src_rgb_u8 is None or ref_rgb_u8 is None or mask_u8 is None:
        raise ValueError("src/ref/mask is None")
    if src_rgb_u8.shape != ref_rgb_u8.shape:
        raise ValueError("src/ref shape mismatch")
    m = (mask_u8 > 5)
    if int(m.sum()) < 64:
        return src_rgb_u8

    src = src_rgb_u8.astype(np.float32)
    ref = ref_rgb_u8.astype(np.float32)
    out = src.copy()
    for c in range(3):
        s = src[..., c][m]
        r = ref[..., c][m]
        s_mean = float(s.mean())
        s_std = float(s.std() + eps)
        r_mean = float(r.mean())
        r_std = float(r.std() + eps)
        out[..., c][m] = (out[..., c][m] - s_mean) * (r_std / s_std) + r_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def match_color_simple(src_rgb_u8: np.ndarray, ref_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    兼容旧代码：在 mask 内做轻量色调匹配。
    当前实现等价于 `_match_color_mean_std`。
    """
    return _match_color_mean_std(src_rgb_u8, ref_rgb_u8, mask_u8)


def _blend_patch_rgb(
    base_patch_u8: np.ndarray,
    gen_patch_u8: np.ndarray,
    alpha_01: np.ndarray,
    mode: str = "multiband",
    levels: int = 4,
) -> np.ndarray:
    """
    对两个 patch 做融合：
    - mode="alpha": 直接按 alpha 做线性混合
    - mode="multiband": 拉普拉斯金字塔多频段融合（减少接缝）
    """
    if base_patch_u8 is None or gen_patch_u8 is None or alpha_01 is None:
        raise ValueError("base/gen/alpha is None")
    if base_patch_u8.shape != gen_patch_u8.shape:
        raise ValueError("base/gen shape mismatch")
    if base_patch_u8.ndim != 3 or base_patch_u8.shape[2] != 3:
        raise ValueError("base/gen must be HxWx3")

    h, w = base_patch_u8.shape[:2]
    a = alpha_01.astype(np.float32)
    if a.ndim == 2:
        a3 = a[..., None]
    elif a.ndim == 3 and a.shape[2] == 1:
        a3 = a
    else:
        raise ValueError("alpha must be HxW or HxWx1")
    if a3.shape[:2] != (h, w):
        a3 = cv2.resize(a3, (w, h), interpolation=cv2.INTER_LINEAR)
    a3 = np.clip(a3, 0.0, 1.0).astype(np.float32)

    mode = str(mode or "alpha").lower().strip()
    if mode not in ("multiband", "pyramid", "laplacian"):
        base = base_patch_u8.astype(np.float32)
        gen = gen_patch_u8.astype(np.float32)
        out = base * (1.0 - a3) + gen * a3
        return np.clip(out, 0, 255).astype(np.uint8)

    # 自动限制 pyramid 层数：太深会因为尺寸太小而失败
    levels = int(levels) if levels is not None else 4
    levels = int(np.clip(levels, 1, 8))
    max_levels = int(max(1, np.floor(np.log2(max(1, min(h, w)))) - 2))
    levels = int(max(1, min(levels, max_levels)))
    if levels <= 1 or min(h, w) < 32:
        base = base_patch_u8.astype(np.float32)
        gen = gen_patch_u8.astype(np.float32)
        out = base * (1.0 - a3) + gen * a3
        return np.clip(out, 0, 255).astype(np.uint8)

    base = base_patch_u8.astype(np.float32)
    gen = gen_patch_u8.astype(np.float32)

    gp_base = [base]
    gp_gen = [gen]
    gp_a = [a3]
    for _ in range(levels):
        gp_base.append(cv2.pyrDown(gp_base[-1]))
        gp_gen.append(cv2.pyrDown(gp_gen[-1]))
        ga = cv2.pyrDown(gp_a[-1])
        # OpenCV 对单通道图像经常返回 2D array，需要补回通道维度
        if ga.ndim == 2:
            ga = ga[..., None]
        gp_a.append(ga)

    lp_base: List[np.ndarray] = []
    lp_gen: List[np.ndarray] = []
    for i in range(levels):
        size = (gp_base[i].shape[1], gp_base[i].shape[0])
        base_up = cv2.pyrUp(gp_base[i + 1], dstsize=size)
        gen_up = cv2.pyrUp(gp_gen[i + 1], dstsize=size)
        lp_base.append(gp_base[i] - base_up)
        lp_gen.append(gp_gen[i] - gen_up)
    lp_base.append(gp_base[-1])
    lp_gen.append(gp_gen[-1])

    lp_blend: List[np.ndarray] = []
    for lb, lg, ga in zip(lp_base, lp_gen, gp_a):
        if ga.ndim == 2:
            ga = ga[..., None]
        if ga.shape[:2] != lb.shape[:2]:
            ga = cv2.resize(ga, (lb.shape[1], lb.shape[0]), interpolation=cv2.INTER_LINEAR)
            if ga.ndim == 2:
                ga = ga[..., None]
        ga = np.clip(ga, 0.0, 1.0).astype(np.float32)
        lp_blend.append(lg * ga + lb * (1.0 - ga))

    out = lp_blend[-1]
    for i in range(levels - 1, -1, -1):
        size = (lp_blend[i].shape[1], lp_blend[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + lp_blend[i]
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_kandinsky_resolutions(pipe) -> List[Tuple[int, int]]:
    """
    Kandinsky5I2IPipeline 只支持固定 resolutions（width,height）。
    优先从 pipe.resolutions 读取；读取失败则用已知默认值兜底。
    """
    r = getattr(pipe, "resolutions", None)
    if isinstance(r, (list, tuple)) and len(r) > 0:
        out: List[Tuple[int, int]] = []
        for it in r:
            try:
                w, h = int(it[0]), int(it[1])
                if w > 0 and h > 0:
                    out.append((w, h))
            except Exception:
                continue
        if out:
            return out
    # diffusers main: Kandinsky5I2IPipeline.resolutions
    return [
        (1024, 1024),
        (640, 1408),
        (1408, 640),
        (768, 1280),
        (1280, 768),
        (896, 1152),
        (1152, 896),
    ]


def _pick_kandinsky_resolution(pipe, w: int, h: int) -> Tuple[int, int]:
    """按 ROI 宽高比挑选最接近的 Kandinsky 支持分辨率（width,height）。"""
    w = max(1, int(w))
    h = max(1, int(h))
    ratio = float(w) / float(h)
    resolutions = _get_kandinsky_resolutions(pipe)
    # 以宽高比差为主；若并列则优先选像素数更大的（减少下采样带来的发糊/细节损失）
    scored = []
    for rw, rh in resolutions:
        r_ratio = float(rw) / float(rh)
        scored.append((abs(r_ratio - ratio), rw * rh, (rw, rh)))
    scored.sort(key=lambda x: (x[0], -x[1]))
    return scored[0][2]


def _apply_mask_obfuscate_rgb(
    img_rgb: np.ndarray,
    mask_l: np.ndarray,
    strength_01: float,
    mode: str = "bilateral",
) -> np.ndarray:
    """
    在 mask 内做“去特征化”预处理：
    - 目标：让编辑模型更愿意替换身份，同时尽量保留五官结构边缘，减少结构漂移与发糊。
    """
    strength_01 = float(np.clip(float(strength_01), 0.0, 1.0))
    m = mask_l > 5
    if int(m.sum()) < 16:
        return img_rgb

    alpha = np.clip(mask_l.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_sigma = 1.2 + 0.9 * strength_01
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(alpha_sigma), sigmaY=float(alpha_sigma))
    a3 = alpha[..., None]

    base = img_rgb.astype(np.float32)

    alt = img_rgb
    mode_l = (mode or "").lower().strip()
    if mode_l == "gaussian":
        blur_sigma = 6.0 + 10.0 * strength_01
        alt = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    elif mode_l == "bilateral":
        # bilateral：去掉皮肤高频纹理但尽量保留边缘（眼睛/鼻梁/嘴角）
        sigma_color = 40.0 + 70.0 * strength_01
        sigma_space = 7.0 + 10.0 * strength_01
        try:
            alt = cv2.bilateralFilter(img_rgb, d=0, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
        except Exception:
            alt = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=8.0, sigmaY=8.0)
        # 再加一点点高斯，避免残留毛孔/噪点让模型“抠回原身份”
        alt = cv2.GaussianBlur(alt, (0, 0), sigmaX=1.0 + 1.2 * strength_01, sigmaY=1.0 + 1.2 * strength_01)
    else:
        alt = img_rgb

    alt_f = alt.astype(np.float32)
    out = base * (1.0 - a3) + alt_f * a3
    return np.clip(out, 0, 255).astype(np.uint8)


def _detect_largest_face_kps5(face_app: FaceAnalysis, img_pil: Image.Image) -> Optional[np.ndarray]:
    """返回最大脸 5 点关键点 (5,2) float32（失败返回 None）。"""
    try:
        faces = face_app.get(pil_to_bgr_np(img_pil))
    except Exception:
        return None
    if not faces:
        return None

    def _area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    faces.sort(key=_area, reverse=True)
    f = faces[0]
    if not hasattr(f, "kps") or f.kps is None:
        return None
    kps = np.array(f.kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None
    return kps


def _normalized_landmark_error(kps_in: np.ndarray, kps_out: np.ndarray) -> float:
    """5 点 L2 平均误差 / 瞳距（越低越好）。"""
    iod = float(np.linalg.norm(kps_in[0] - kps_in[1])) + 1e-12
    err = np.linalg.norm(kps_in - kps_out, axis=1).mean()
    return float(err / iod)


def _maybe_align_face_by_kps5(face_app: FaceAnalysis, in_roi: Image.Image, out_roi: Image.Image) -> Image.Image:
    """
    若 out_roi 相对 in_roi 的五官整体发生偏移，则用 5 点关键点做相似变换对齐。
    """
    kps_in = _detect_largest_face_kps5(face_app, in_roi)
    kps_out = _detect_largest_face_kps5(face_app, out_roi)
    if kps_in is None or kps_out is None:
        return out_roi

    err0 = _normalized_landmark_error(kps_in, kps_out)
    if not (err0 > 0.18):
        return out_roi

    M, _ = cv2.estimateAffinePartial2D(kps_out, kps_in, method=cv2.LMEDS)
    if M is None:
        return out_roi

    out_np = np.array(out_roi.convert("RGB"))
    w, h = out_roi.size
    warped = cv2.warpAffine(
        out_np,
        M,
        dsize=(int(w), int(h)),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_pil = Image.fromarray(warped)

    kps_w = _detect_largest_face_kps5(face_app, warped_pil)
    if kps_w is None:
        return out_roi
    err1 = _normalized_landmark_error(kps_in, kps_w)
    if err1 < err0 * 0.92:
        return warped_pil
    return out_roi


def _apply_mask_noise_rgb(img_rgb: np.ndarray, mask_l: np.ndarray, noise_seed: int) -> np.ndarray:
    """
    SD3/3.5 的 img2img 没有原生 mask inpaint 能力，这里用“遮挡初始化”近似 inpaint：
    - **不再用高频随机噪声**（容易被模型当成纹理/面具，出现网格脸）
    - 改为：在 mask 内用“强模糊后的原图”覆盖，并用 soft mask 做平滑过渡

    这样模型仍能看到 ROI 的几何结构（姿态/轮廓/五官大致位置），更容易贴合原图并保持写实。
    """
    # noise_seed 保留用于可复现（当前实现不使用高频噪声）
    _ = noise_seed
    m = mask_l > 5
    if m.sum() < 16:
        return img_rgb

    # soft mask（0..1），并对 alpha 做模糊让边界柔和
    alpha = np.clip(mask_l.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha3 = alpha[..., None]

    base = img_rgb.astype(np.float32)
    # 强模糊版本：抹掉细节但保留大结构（sigma 过大会导致脸发糊）
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=6.0, sigmaY=6.0).astype(np.float32)
    out = base * (1.0 - alpha3) + blurred * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode_mask(mask_l: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    """
    收紧 mask，减少对脸外（头发/背景）的影响。
    ratio 基于 ROI 的最短边做腐蚀核大小估计。
    """
    h, w = mask_l.shape[:2]
    k = int(round(min(h, w) * ratio))
    k = max(1, min(k, 61))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return mask_l
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask_l, kernel, iterations=1)


def run_roi_img2img_once(
    pipe,
    full_img_pil: Image.Image,
    full_mask_pil: Image.Image,
    bbox: Tuple[int, int, int, int],
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    strength: float,
    device: str,
    pad_ratio: float = 0.0,
    pad_multiple: int = 8,
    roi_max_side: int = 1024,
    full_blend_mask_pil: Optional[Image.Image] = None,
    blend_mode: str = "multiband",
    blend_levels: int = 4,
) -> Image.Image:
    """
    SD3/3.5 的 img2img 没有原生 mask inpaint，这里实现“latent-mask img2img”：
    - 先把 ROI 编码为 clean latents
    - 每一步 denoise 结束后，把 **非 mask 区域** 的 latents 强制替换为「原图在同 timestep 的 noisy latents」
    - 从而在 latent 空间锁定结构，达到更接近 inpaint 的效果，并避免“输入破坏”造成的糊脸/网格脸
    """
    # 必须支持 callback_on_step_end，才能逐步替换 latents（否则只能回退 ROI-edit）
    try:
        sig = inspect.signature(pipe.__call__)
        _supports_cb = ("callback_on_step_end" in sig.parameters) and ("callback_on_step_end_tensor_inputs" in sig.parameters)
    except Exception:
        _supports_cb = False

    if not _supports_cb:
        print("[WARN] 当前 pipeline 不支持 callback_on_step_end，无法启用 latent-mask img2img，将回退到 ROI-edit。"
              "如需启用，请升级 diffusers（建议 pip install -U diffusers）。")
        return run_roi_edit_once(
            pipe=pipe,
            full_img_pil=full_img_pil,
            full_mask_pil=full_mask_pil,
            bbox=bbox,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            strength=strength,
            pad_ratio=float(pad_ratio) if pad_ratio is not None else 0.0,
            pad_multiple=pad_multiple,
            roi_max_side=roi_max_side,
        )

    W, H = full_img_pil.size
    bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    min_face = max(1, min(bw, bh))
    if min_face < 120:
        local_pad_ratio = 0.30
    elif min_face < 200:
        local_pad_ratio = 0.38
    else:
        local_pad_ratio = 0.45
    if pad_ratio > 0:
        local_pad_ratio = max(local_pad_ratio, float(pad_ratio))
    mask_bbox = _bbox_from_mask(full_mask_pil, threshold=128)
    if full_blend_mask_pil is not None:
        blend_bbox = _bbox_from_mask(full_blend_mask_pil, threshold=128)
        if blend_bbox is not None:
            mask_bbox = _bbox_union(mask_bbox, blend_bbox) if mask_bbox is not None else blend_bbox
    base_bbox = _bbox_union(bbox, mask_bbox) if mask_bbox is not None else bbox
    X1, Y1, X2, Y2 = _bbox_expand_and_clip(base_bbox, W, H, pad_ratio=local_pad_ratio)

    roi_img = full_img_pil.crop((X1, Y1, X2, Y2)).convert("RGB")
    roi_mask = full_mask_pil.crop((X1, Y1, X2, Y2)).convert("L")
    roi_blend_mask = (full_blend_mask_pil or full_mask_pil).crop((X1, Y1, X2, Y2)).convert("L")
    roi_img_orig = roi_img
    roi_mask_orig = roi_mask
    roi_blend_mask_orig = roi_blend_mask

    roi_w, roi_h = roi_img.size
    min_dim = min(roi_w, roi_h)
    _roi_max_side: Optional[int] = int(roi_max_side) if roi_max_side is not None else 1024
    if _roi_max_side <= 0:
        _roi_max_side = None

    # 小脸上采样：提高可分辨细节，避免生成不全/发糊
    min_roi_side = 512
    if min_face < 120:
        max_scale = 8.0
    elif min_face < 200:
        max_scale = 4.0
    else:
        max_scale = 2.5
    target_face_side = 220

    scale_dim = float(min_roi_side) / float(max(1, min_dim)) if min_dim < min_roi_side else 1.0
    scale_face = float(target_face_side) / float(max(1, min_face)) if min_face < target_face_side else 1.0
    scale = float(max(1.0, scale_dim, scale_face))
    scale = float(min(scale, float(max_scale)))
    new_w = int(round(roi_w * scale))
    new_h = int(round(roi_h * scale))
    if _roi_max_side is not None:
        max_dim = max(new_w, new_h)
        if max_dim > int(_roi_max_side):
            s2 = float(_roi_max_side) / float(max_dim)
            new_w = int(round(new_w * s2))
            new_h = int(round(new_h * s2))
            scale *= s2
    new_w = max(64, new_w)
    new_h = max(64, new_h)
    if (new_w, new_h) != (roi_w, roi_h):
        roi_img = roi_img.resize((new_w, new_h), resample=Image.LANCZOS)
        roi_mask = roi_mask.resize((new_w, new_h), resample=Image.BILINEAR)
        roi_w, roi_h = roi_img.size
        min_dim = min(roi_w, roi_h)
    # pad ROI 到 multiple（让 SD3 这类模型更稳）
    roi_img_pad, roi_mask_pad, pad = pad_image_and_mask_to_multiple(roi_img, roi_mask, multiple=pad_multiple)

    # =========================
    # SD3/3.5 的“真·mask img2img”（latent 空间逐步融合）
    # 思路：每一步把“非 mask 区域”的 latents 强制替换为「原图在同 timestep 的 noisy latents」
    # 这样 ROI 外结构完全不漂移，且不需要对输入图做模糊/噪声破坏（避免发糊、网格脸）。
    # =========================
    w2, h2 = roi_img_pad.size

    # 生成器：用 CPU generator 更兼容多设备/多卡
    generator = torch.Generator("cpu").manual_seed(seed)

    # 1) 准备 latent mask（主要在 edit_mask 内编辑；mask 适度扩大可提升自由度）
    roi_m = np.array(roi_mask_pad.convert("L"), dtype=np.uint8)
    roi_m_bin = (roi_m >= 128).astype(np.uint8) * 255
    size_factor = min(1.0, float(min_dim) / 512.0)
    # 方案A：减少对 mask 的过度收紧，允许刘海/耳朵等边界区域轻度参与编辑
    inner_erode_ratio = max(0.006, 0.02 * size_factor)
    # 贴回融合：避免过度 erosion 导致硬边
    blend_erode_ratio = max(0.008, 0.025 * size_factor)
    mask_sigma = max(0.2, 0.4 * size_factor)
    roi_m_inner = _erode_mask(roi_m_bin, ratio=inner_erode_ratio)
    mask_01 = (roi_m_inner.astype(np.float32) / 255.0)
    mask_01 = np.clip(mask_01, 0.0, 1.0)
    mask_01 = cv2.GaussianBlur(mask_01, (0, 0), sigmaX=mask_sigma, sigmaY=mask_sigma)

    # 2) 准备 clean latents（原 ROI 图像）
    if not hasattr(pipe, "vae") or pipe.vae is None:
        raise RuntimeError("当前 pipeline 没有 vae，无法进行 latent mask 融合。")

    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))

    # 注意：当 pipeline 启用了 enable_model_cpu_offload 时，参数常驻 CPU，但 forward 会被 hook 搬到 GPU。
    # 此时若把输入 tensor 放在 CPU，会触发 “cpu/cuda 混用” 的 conv2d device mismatch。
    exec_dev = None
    try:
        exec_dev = getattr(pipe, "_execution_device", None)
    except Exception:
        exec_dev = None
    if exec_dev is None:
        exec_dev = device
    exec_dev = str(exec_dev)

    roi_np = np.array(roi_img_pad.convert("RGB")).astype(np.float32) / 255.0
    roi_t = torch.from_numpy(roi_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W], 0..1
    roi_t = (roi_t * 2.0 - 1.0).to(device=exec_dev, dtype=vae_dtype)

    with torch.no_grad():
        clean_latents = vae.encode(roi_t).latent_dist.sample(generator=generator).to(dtype=vae_dtype)
        clean_latents = clean_latents * scaling_factor

    # latent mask 下采样到 latents 分辨率
    mh, mw = clean_latents.shape[-2], clean_latents.shape[-1]
    mask_t = torch.from_numpy(mask_01).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    mask_t = torch.nn.functional.interpolate(mask_t, size=(mh, mw), mode="bilinear", align_corners=False)
    mask_t = mask_t.to(device=clean_latents.device, dtype=clean_latents.dtype)
    mask_t = torch.clamp(mask_t, 0.0, 1.0)

    # 3) 预采样一次 noise，用于 scheduler.add_noise
    # 兼容不同 torch 版本：randn_like 可能不支持 generator 参数
    try:
        gen_dev = str(clean_latents.device)
        try:
            noise_gen = torch.Generator(device=gen_dev).manual_seed(seed)
        except TypeError:
            # 老版本 torch.Generator 不接受 device 参数
            noise_gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            clean_latents.shape,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=noise_gen,
        )
    except TypeError:
        # 兜底：不传 generator（仍尽量固定全局 seed）
        torch.manual_seed(seed)
        noise = torch.randn(clean_latents.shape, device=clean_latents.device, dtype=clean_latents.dtype)

    def _cb_on_step_end(pipe_self, step_idx: int, timestep: int, cb_kwargs: Dict):
        latents = cb_kwargs.get("latents", None)
        if latents is None:
            return cb_kwargs
        lat_dev = latents.device
        # 保证 callback 内所有张量设备一致（兼容 offload / 多设备）
        if clean_latents.device != lat_dev:
            clean_l = clean_latents.to(device=lat_dev)
        else:
            clean_l = clean_latents
        if mask_t.device != lat_dev:
            mask_l = mask_t.to(device=lat_dev)
        else:
            mask_l = mask_t
        if noise.device != lat_dev:
            noise_l = noise.to(device=lat_dev)
        else:
            noise_l = noise

        # timestep 可能是 tensor / int
        ts = timestep
        if not torch.is_tensor(ts):
            ts = torch.tensor([ts], device=lat_dev, dtype=torch.long)
        else:
            ts = ts.to(device=lat_dev)

        if not hasattr(pipe, "scheduler") or pipe.scheduler is None or not hasattr(pipe.scheduler, "add_noise"):
            # 兜底：直接用 clean_latents（不如 add_noise 精确，但仍能锁定结构）
            target = clean_l
        else:
            target = pipe.scheduler.add_noise(clean_l, noise_l, ts)

        # latents 可能是 [B,C,H,W]，mask_t 是 [1,1,H,W]，自动 broadcast
        latents = latents * mask_l + target * (1.0 - mask_l)
        cb_kwargs["latents"] = latents
        return cb_kwargs

    # 4) img2img 调用（参数会自动过滤）
    out = _extract_images_from_pipe_output(_call_pipe_filtered(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=roi_img_pad,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        height=h2,
        width=w2,
        generator=generator,
        callback_on_step_end=_cb_on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    ))[0]

    # crop 回 ROI 原大小
    out_roi = crop_back_to_original(out, pad, (roi_w, roi_h))
    if out_roi.size != roi_img_orig.size:
        out_roi = out_roi.resize(roi_img_orig.size, resample=Image.LANCZOS)

    # 用 mask 软融合贴回原图
    base_rgb = np.array(full_img_pil.convert("RGB"))
    out_rgb = base_rgb.copy()

    out_patch = np.array(out_roi.convert("RGB"))
    # 融合也用收紧后的 mask，避免背景被拖糊（先二值化再 erosion 更稳定）
    blend_src = np.array(roi_blend_mask_orig.convert("L"), dtype=np.uint8)
    blend_bin = (blend_src >= 128).astype(np.uint8) * 255
    blend_mask = _erode_mask(blend_bin, ratio=blend_erode_ratio)
    strength_01 = float(np.clip(float(strength), 0.0, 1.0))
    feather_px = int(np.clip(float(min_face) * (0.08 + 0.02 * strength_01), 6.0, 36.0))
    m = _make_feather_alpha_dt(blend_mask, feather_px=feather_px, blur_sigma=0.0)
    gamma = 1.10
    if gamma is not None and abs(float(gamma) - 1.0) > 1e-3:
        m = (np.clip(m, 0.0, 1.0) ** float(gamma)).astype(np.float32)

    base_patch = base_rgb[Y1:Y2, X1:X2, :]
    if base_patch.shape[:2] != out_patch.shape[:2]:
        out_patch = cv2.resize(out_patch, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        m = cv2.resize(m, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 贴回前颜色匹配：降低“脸 patch 与背景/脖子割裂感”
    try:
        # 颜色匹配只在 edit_mask 内做，避免头发/背景被拉色
        edit_src = np.array(roi_mask_orig.convert("L"), dtype=np.uint8)
        cm_mask = ((edit_src >= 128).astype(np.uint8) * 255).astype(np.uint8)
        matched = match_color_simple(out_patch, base_patch, cm_mask)
        alpha_cm = 0.7
        out_patch = (out_patch.astype(np.float32) * (1.0 - alpha_cm) + matched.astype(np.float32) * alpha_cm)
        out_patch = np.clip(out_patch, 0, 255).astype(np.uint8)
    except Exception:
        pass

    # 锁定 mask 外区域为原图，避免 ROI 矩形方框割裂
    try:
        edit_src = np.array(roi_mask_orig.convert("L"), dtype=np.uint8)
        hard_u8 = (edit_src >= 128).astype(np.uint8) * 255
        if hard_u8.shape[:2] != base_patch.shape[:2]:
            hard_u8 = cv2.resize(hard_u8, (base_patch.shape[1], base_patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        hard = (hard_u8.astype(np.float32) / 255.0)[..., None]
        out_patch = (out_patch.astype(np.float32) * hard + base_patch.astype(np.float32) * (1.0 - hard))
        out_patch = np.clip(out_patch, 0, 255).astype(np.uint8)
    except Exception:
        pass

    blended = _blend_patch_rgb(base_patch, out_patch, m, mode=blend_mode, levels=int(blend_levels))
    out_rgb[Y1:Y2, X1:X2, :] = blended
    # 轻量锐化（仅 ROI 区域），提升清晰度
    try:
        import cv2 as _cv2
        blur = _cv2.GaussianBlur(out_rgb, (0, 0), sigmaX=1.0, sigmaY=1.0)
        sharp = _cv2.addWeighted(out_rgb, 1.15, blur, -0.15, 0)
        out_rgb = sharp
    except Exception:
        pass
    return Image.fromarray(out_rgb)


# =========================
# 11) 单张：输入图 -> 输出图
# =========================
def run_identity_demo_once(
    cfg: DemoConfig,
    pipe=None,
    device: Optional[str] = None,
    face_app: Optional[FaceAnalysis] = None,
    model_spec: Optional[ModelSpec] = None,
) -> Image.Image:
    """
    单张处理（多模型）：
    - 自动检测人脸（默认替换全部人脸），生成柔边人脸 mask（仅 landmarks）
    - 根据模型类型做 inpaint / image-edit(img2img)
    - 输出严格裁回原始尺寸
    """
    # 1) 读输入
    img = load_image_rgb(cfg.in_img)
    orig_w, orig_h = img.size

    # 2) face_app
    if face_app is None:
        face_app = init_face_app(ctx_id=cfg.ctx_id, det_size=cfg.det_size)

    # 3) 预解析模型规格，决定 pad_multiple（也用于匿名化 gating）
    if model_spec is None:
        model_spec = resolve_model_spec(cfg.model_id)
    pad_multiple = model_spec.pad_multiple or cfg.pad_multiple

    # Qwen-Image-Edit-2511：强制走匿名化（去除非匿名化分支）
    enable_anonymize = (model_spec.loader == "sdcpp_qwen_image_edit_2511_gguf")

    # 4) 检测全部人脸
    faces = detect_all_faces(face_app, img, det_size=cfg.det_size)
    if not faces:
        raise RuntimeError(f"No face detected in {cfg.in_img}")
    if not cfg.replace_all_faces:
        faces = faces[:1]
    if cfg.max_faces and cfg.max_faces > 0:
        faces = faces[: int(cfg.max_faces)]

    # 5) build pipe（复用/新建）
    if pipe is None:
        pipe, device, model_spec = build_pipe(cfg.model_id, cfg.device, runtime_cfg=cfg)
    else:
        if device is None:
            _, device = _get_pipe_param_dtype_device(pipe)
        if model_spec is None:
            model_spec = getattr(pipe, "_demo_model_spec", None) or resolve_model_spec(cfg.model_id)

    assert model_spec is not None

    # 6.8) prompt / negative_prompt / 运行参数（按模型做适配）
    base_prompt = str(cfg.prompt or "")
    base_negative_prompt = str(cfg.negative_prompt or "")

    base_steps = int(cfg.steps)
    base_guidance = float(cfg.guidance)
    base_strength = float(cfg.strength)

    # 7) 推理（统一外壳：优先 inpaint(mask_image)，否则走 image-edit/img2img）
    use_mask_inpaint = _pipe_supports_mask_inpaint(pipe)

    def _infer_once(
        local_img: Image.Image,
        local_mask: Image.Image,
        local_bbox: Tuple[int, int, int, int],
        local_seed: int,
        local_steps: int,
        local_guidance: float,
        local_strength: float,
        local_prompt: str,
        local_negative_prompt: str,
    ) -> Image.Image:
        img_pad, mask_pad, pad = pad_image_and_mask_to_multiple(local_img, local_mask, multiple=pad_multiple)
        left, top, _, _ = pad
        bbox_pad = (local_bbox[0] + left, local_bbox[1] + top, local_bbox[2] + left, local_bbox[3] + top)
        if use_mask_inpaint:
            out_pad_local = run_roi_inpaint_once(
                pipe=pipe,
                full_img_pil=img_pad,
                full_mask_pil=mask_pad,
                bbox=bbox_pad,
                prompt=local_prompt,
                negative_prompt=local_negative_prompt,
                seed=int(local_seed),
                steps=int(local_steps),
                guidance=float(local_guidance),
                strength=float(local_strength),
                device=device or cfg.device,
                pad_ratio=cfg.pad_ratio,
                pad_multiple=pad_multiple,
                roi_max_side=getattr(cfg, "roi_max_side", 1024),
                model_spec=model_spec,
                extra_call_kwargs=model_spec.extra_call_kwargs,
                enable_anonymize=enable_anonymize,
            )
        else:
            # SD3/3.5：启用 latent-mask img2img，更接近 inpaint 且更稳
            if getattr(model_spec, "loader", "") == "sd3_img2img":
                out_pad_local = run_roi_img2img_once(
                    pipe=pipe,
                    full_img_pil=img_pad,
                    full_mask_pil=mask_pad,
                    bbox=bbox_pad,
                    prompt=local_prompt,
                    negative_prompt=local_negative_prompt,
                    seed=int(local_seed),
                    steps=int(local_steps),
                    guidance=float(local_guidance),
                    strength=float(local_strength),
                    device=device or cfg.device,
                    pad_ratio=cfg.pad_ratio,
                    pad_multiple=pad_multiple,
                    roi_max_side=getattr(cfg, "roi_max_side", 1024),
                )
            else:
                out_pad_local = run_roi_edit_once(
                    pipe=pipe,
                    full_img_pil=img_pad,
                    full_mask_pil=mask_pad,
                    bbox=bbox_pad,
                    prompt=local_prompt,
                    negative_prompt=local_negative_prompt,
                    seed=int(local_seed),
                    steps=int(local_steps),
                    guidance=float(local_guidance),
                    strength=float(local_strength),
                    pad_ratio=cfg.pad_ratio,
                    pad_multiple=pad_multiple,
                    roi_max_side=getattr(cfg, "roi_max_side", 1024),
                    extra_call_kwargs=model_spec.extra_call_kwargs,
                    face_app=face_app,
                )
        return crop_back_to_original(out_pad_local, pad, (orig_w, orig_h))

    # 8) 逐脸替换（多人同框）
    out = img
    for face_idx, face in enumerate(faces):
        bbox = tuple(int(v) for v in face.bbox)
        edit_mask = make_face_mask_from_face(
            size_wh=(orig_w, orig_h),
            face=face,
            pad_ratio=cfg.pad_ratio,
            blur=cfg.mask_blur,
            img_pil=img,
            cfg=cfg,
        )
        # 方案A：mask 与 bbox 融合（裁剪到“头部 bbox”而非仅人脸 bbox），把接缝尽量推到头发/头部边缘
        try:
            bbox_clip = _bbox_expand_and_clip(bbox, orig_w, orig_h, pad_ratio=0.80)
        except Exception:
            bbox_clip = bbox
        edit_mask = clip_mask_to_bbox(edit_mask, bbox_clip)

        bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        min_face = max(1, min(bw, bh))

        has_kps = True
        if cfg.skip_if_no_kps:
            pts = None
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                pts = np.array(face.landmark_2d_106)
            elif hasattr(face, "kps") and face.kps is not None:
                pts = np.array(face.kps)
            if pts is None or pts.size == 0:
                has_kps = False

        # 小脸/背脸/无关键点：避免凭空补脸（不再做 blur fallback，直接跳过/继续下一个脸）
        if min_face < int(cfg.min_face_edit) or (cfg.skip_if_no_kps and not has_kps):
            continue

        # 每个脸使用不同 seed，避免生成同一个虚拟人脸
        local_seed = int(cfg.seed) + (face_idx + 1) * 10007 + (bbox[0] + bbox[1]) % 997

        # 多人同框时增加“不同身份”提示
        identity_hint = ""
        negative_hint = ""
        if len(faces) > 1:
            identity_hint = f" different person, unique identity #{face_idx + 1}, distinct facial features."
            negative_hint = " same person, duplicate face, identical identity."

        prompt = (base_prompt or "") + identity_hint
        negative_prompt = (base_negative_prompt or "")
        if negative_hint:
            negative_prompt = (negative_prompt + ", " + negative_hint).strip(", ")

        # Qwen Image Edit（stable-diffusion.cpp GGUF）：推荐 cfg_scale 低一些；negative_prompt 为空更稳
        steps = base_steps
        guidance = base_guidance
        strength = base_strength
        if model_spec.loader == "sdcpp_qwen_image_edit_2511_gguf":
            if steps <= 0 or steps > 80:
                steps = 20
            if guidance <= 0 or guidance > 5.0:
                guidance = 2.5
            if enable_anonymize:
                # Qwen 容易在高 strength 下“重画+风格跑偏”，这里做上下限保护（仍允许通过 cfg 调参）
                s_min = float(getattr(cfg, "qwen_strength_min", 0.60) or 0.60)
                s_max = float(getattr(cfg, "qwen_strength_max", 0.90) or 0.90)
                strength = float(np.clip(float(strength), s_min, s_max))
                # negative_prompt 保留并追加“禁止动漫/尖耳”等兜底
                extra_neg = str(getattr(cfg, "qwen_extra_negative", "") or "").strip()
                if extra_neg:
                    if negative_prompt:
                        negative_prompt = (negative_prompt + ", " + extra_neg).strip(", ")
                    else:
                        negative_prompt = extra_neg

        strength = float(np.clip(strength, 0.0, 0.98))

        if enable_anonymize:
            # 兜底约束：避免耳朵/风格跑偏（对 run*.py 传入的 prompt 同样生效）
            prompt = (prompt or "").strip()
            prompt = (prompt + " Keep realistic human anatomy and photographic style; no anime/cartoon. "
                             "Do not change ear shape (no elf/pointy ears, no animal ears).").strip()
            p_low = (prompt or "").lower()
            has_edit_intent = any(k in p_low for k in (
                "replace", "swap", "change", "edit", "anonymize", "de-identify",
                "替换", "换脸", "更换", "修改", "匿名", "脱敏", "去标识",
            ))
            if not has_edit_intent:
                prompt = (
                    "Anonymize the person's face by replacing the face identity with a clearly different realistic identity (not the same person). "
                    "Do not preserve recognizable identity cues from the input face. "
                    "Keep the same head pose, gaze direction, facial expression, lighting and background. "
                    "Keep the overall hairstyle unchanged, but minor edits to hair fringe/ears around the face are allowed for natural blending. "
                    "Keep neck and clothing unchanged. "
                    "Photo-realistic, natural skin texture, sharp eyes, high detail. "
                ) + (prompt or "")

        out = _infer_once(
            local_img=out,
            local_mask=edit_mask,
            local_bbox=bbox,
            local_seed=local_seed,
            local_steps=steps,
            local_guidance=guidance,
            local_strength=strength,
            local_prompt=prompt,
            local_negative_prompt=negative_prompt,
        )

    # 10) 保存
    if cfg.out_img:
        Path(cfg.out_img).parent.mkdir(parents=True, exist_ok=True)
        out.save(cfg.out_img)

    return out


# =========================
# 12) 批量：一次加载模型 + 批量推理
# =========================
def _collect_image_pairs(
    in_paths: Union[str, Path, List[Union[str, Path]]],
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[Path, Path]]:
    """收集 (输入路径, 输出路径) 对"""
    pairs: List[Tuple[Path, Path]] = []
    in_paths = Path(in_paths) if isinstance(in_paths, (str, Path)) else in_paths

    if isinstance(in_paths, Path):
        if in_paths.is_dir():
            files = sorted(in_paths.glob("*"))
            files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
        else:
            files = [in_paths]
    else:
        files = []
        for p in in_paths:
            pp = Path(p)
            if pp.is_dir():
                files.extend(
                    f for f in sorted(pp.glob("*"))
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                )
            elif pp.exists() and pp.is_file():
                files.append(pp)

    out_base = Path(out_dir) if out_dir else None
    for f in files:
        f = Path(f)
        if out_base:
            out_path = out_base / f.name
        else:
            out_path = f.parent / f"{f.stem}_out{f.suffix}"
        pairs.append((f, out_path))
    return pairs


def run_identity_demo_batch(
    in_paths: Union[str, Path, List[Union[str, Path]]],
    out_dir: Union[str, Path],
    base_cfg: DemoConfig,
    skip_existing: bool = True,
    runtime_json_path: Optional[Union[str, Path]] = None,
) -> List[Image.Image]:
    """
    批量跑：
    - face_app / pipe / embedding 都复用
    - 输出尺寸严格一致
    """
    pairs = _collect_image_pairs(in_paths, out_dir)
    if not pairs:
        print("[WARN] No valid input images found.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading FaceAnalysis...")
    face_app = init_face_app(ctx_id=base_cfg.ctx_id, det_size=base_cfg.det_size)

    print(f"[INFO] Loading pipeline: model_id={base_cfg.model_id}")
    pipe, device, spec = build_pipe(base_cfg.model_id, base_cfg.device, runtime_cfg=base_cfg)

    results: List[Image.Image] = []
    runtime_map: Dict[str, Dict[str, float]] = {}
    total = len(pairs)

    for i, (in_p, out_p) in enumerate(pairs, 1):
        in_p = Path(in_p)
        out_p = Path(out_p)

        if skip_existing and out_p.exists():
            print(f"[SKIP] {in_p.name} -> {out_p.name}")
            continue

        t0 = time.perf_counter()
        try:
            cfg = DemoConfig(
                in_img=str(in_p),
                out_img=str(out_p),
                prompt=base_cfg.prompt,
                negative_prompt=base_cfg.negative_prompt,
                seed=base_cfg.seed,
                steps=base_cfg.steps,
                guidance=base_cfg.guidance,
                strength=base_cfg.strength,
                pad_ratio=base_cfg.pad_ratio,
                mask_blur=base_cfg.mask_blur,
                det_size=base_cfg.det_size,
                ctx_id=base_cfg.ctx_id,
                replace_all_faces=base_cfg.replace_all_faces,
                max_faces=base_cfg.max_faces,
                model_id=base_cfg.model_id,
                device=base_cfg.device,
                pad_multiple=base_cfg.pad_multiple,
                roi_max_side=base_cfg.roi_max_side,
                skip_existing=base_cfg.skip_existing,
            )

            out_img = run_identity_demo_once(
                cfg=cfg,
                pipe=pipe,
                device=device,
                face_app=face_app,
                model_spec=spec,
            )
            results.append(out_img)

            dt = time.perf_counter() - t0
            runtime_map[in_p.name] = {"seconds": round(dt, 3)}
            print(f"[{i:03d}/{total}] OK {in_p.name} -> {out_p.name} ({dt:.2f}s)")
        except Exception as e:
            print(f"[{i:03d}/{total}] ERR {in_p.name}: {e}")

    if runtime_json_path:
        runtime_json_path = Path(runtime_json_path)
        runtime_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(runtime_json_path, "w", encoding="utf-8") as f:
            json.dump(runtime_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Runtime saved to {runtime_json_path}")

    return results


# =========================
# 13) main：不走命令行的示例（你说 run 脚本单独写了，这里只是备用）
# =========================
def main():
    # 示例：跑 SDXL inpaint
    in_dir = "inputs"            # 待处理图片目录
    out_dir = "outputs_sdxl"     # 输出目录

    base_cfg = DemoConfig(
        in_img="",
        out_img="",
        model_id="sdxl_inpaint",   # 或直接写 HF repo id
        device="cuda",
        seed=1234,
        steps=28,
        guidance=6.0,
        strength=0.80,
        pad_ratio=0.35,
        mask_blur=12,
        det_size=640,
        ctx_id=0,
        pad_multiple=8,
    )

    run_identity_demo_batch(
        in_paths=in_dir,
        out_dir=out_dir,
        base_cfg=base_cfg,
        skip_existing=True,
        runtime_json_path=str(Path(out_dir) / "runtime.json"),
    )


if __name__ == "__main__":
    main()
