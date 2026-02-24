#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_m4 standalone pipeline (does NOT import demo_m3).

Batch local-redraw avatar swap:
- Keep background / pose / expression
- Replace only inner-face identity region
- Cover all model backbones used in demo_m3
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis

# Set HF environment before diffusers/hf_hub downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from diffusers import AutoPipelineForInpainting, DiffusionPipeline

try:
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
except Exception:
    FluxFillPipeline = None
    FluxTransformer2DModel = None

try:
    from diffusers import Kandinsky5I2IPipeline
except Exception:
    Kandinsky5I2IPipeline = None

try:
    from diffusers import StableDiffusion3Img2ImgPipeline
except Exception:
    StableDiffusion3Img2ImgPipeline = None

try:
    from transformers import T5EncoderModel
except Exception:
    T5EncoderModel = None


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
INNER_FACE_LABELS = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)

DEFAULT_PROMPT = (
    "Replace only the inner face identity with a different realistic person. "
    "Keep the same head pose, gaze direction, facial expression, hairstyle, lighting, "
    "camera perspective and full background. Keep ears, neck and clothing unchanged. "
    "Photo-realistic, natural skin texture, sharp eyes."
)
DEFAULT_NEGATIVE = (
    "deformed, bad anatomy, extra face, extra eyes, blurry, low quality, "
    "cartoon, anime, painting, uncanny, watermark, text, duplicated identity"
)
QWEN_PROMPT = (
    "Anonymize the person's face by replacing only the inner face area with a different realistic identity "
    "(not the same person). Keep the same head pose, gaze direction, facial expression, lighting, hairstyle "
    "and background. Keep ears, neck and clothing unchanged. Only modify the inner face area. "
    "Do not preserve recognizable identity cues from the input face. "
    "Photo-realistic, natural skin texture, sharp eyes, high detail."
)
FLUX2_PROMPT = (
    "Replace only the inner face identity with a different realistic person, same head pose and facial "
    "expression, consistent lighting, soft diffused lighting, clear smooth skin, even complexion, photorealistic."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    loader: str  # auto_inpaint | diffusion_edit | flux_fill_nf4 | sd3_img2img | qwen_cpp
    pad_multiple: int = 8
    torch_dtype: Optional[str] = None  # "fp16" | "bf16" | "fp32"
    trust_remote_code: bool = False
    variant: Optional[str] = None
    use_safetensors: Optional[bool] = None
    extra_call_kwargs: Dict[str, Any] = field(default_factory=dict)


MODEL_ZOO: Dict[str, ModelSpec] = {
    # 与 demo_m3 对齐的 7 个模型：这里作为“独立 pipeline”的固定模型池。
    "sdxl_inpaint": ModelSpec(
        key="sdxl_inpaint",
        model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        loader="auto_inpaint",
        pad_multiple=8,
    ),
    "kandinsky5_i2i_lite": ModelSpec(
        key="kandinsky5_i2i_lite",
        model_id="kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
        extra_call_kwargs={"max_sequence_length": 1024},
    ),
    "longcat_image_edit_turbo": ModelSpec(
        key="longcat_image_edit_turbo",
        model_id="meituan-longcat/LongCat-Image-Edit-Turbo",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
    ),
    "qwen_image_edit_2511_gguf": ModelSpec(
        key="qwen_image_edit_2511_gguf",
        model_id="unsloth/Qwen-Image-Edit-2511-GGUF",
        loader="qwen_cpp",
        pad_multiple=64,
    ),
    "flux_fill_nf4": ModelSpec(
        key="flux_fill_nf4",
        model_id="black-forest-labs/FLUX.1-Fill-dev",
        loader="flux_fill_nf4",
        pad_multiple=16,
        torch_dtype="bf16",
        extra_call_kwargs={"max_sequence_length": 512},
    ),
    "flux2_klein_9b": ModelSpec(
        key="flux2_klein_9b",
        model_id="black-forest-labs/FLUX.2-klein-9B",
        loader="diffusion_edit",
        pad_multiple=16,
        torch_dtype="bf16",
        use_safetensors=True,
    ),
    "sd35_img2img": ModelSpec(
        key="sd35_img2img",
        model_id="stabilityai/stable-diffusion-3.5-large-turbo",
        loader="sd3_img2img",
        pad_multiple=16,
    ),
}


@dataclass
class RuntimePlan:
    prompt: str
    negative_prompt: str
    steps: int
    guidance: float
    strength: float


@dataclass
class RunConfig:
    mode: str  # "single" | "batch"
    single_in_img: str
    single_out_img: str

    input_dir: str
    output_root: str
    report_root: str
    models: List[str]

    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance: float
    strength: float

    pad_ratio: float
    mask_blur: int
    roi_pad_ratio: float
    roi_max_side: int
    default_pad_multiple: int

    det_size: int
    ctx_id: int
    device: str
    cpu_offload: str

    replace_all_faces: bool
    max_faces: int
    min_face_edit: int
    skip_if_no_kps: bool

    # 小脸/侧脸难例自适应参数
    hard_face_scale_thr: float
    hard_yaw_thr: float
    hard_face_min_px: int
    hard_max_upscale: float
    hard_mask_blur_scale: float
    hard_roi_pad_extra: float
    hard_strength_drop: float
    hard_feather_scale: float
    hard_union_mask: bool

    skip_existing: bool
    no_eval: bool
    fail_fast: bool


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
    return obj


def _dataset_tag(path: str) -> str:
    p = os.path.normpath(path)
    tag = os.path.basename(p)
    return tag or "dataset"


def _collect_images(input_dir: str) -> List[Path]:
    # 仅收集当前目录下图片文件（不递归）。
    d = Path(input_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")
    return files


def _parse_models(text: str) -> List[str]:
    # "all" -> MODEL_ZOO 全部；否则按逗号拆分并校验 key。
    raw = [x.strip() for x in text.split(",") if x.strip()]
    if not raw or raw == ["all"]:
        return list(MODEL_ZOO.keys())
    unknown = [x for x in raw if x not in MODEL_ZOO]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}; available={list(MODEL_ZOO.keys())}")
    return raw


def _normalize_models_input(models: Any) -> List[str]:
    if isinstance(models, str):
        return _parse_models(models)
    if isinstance(models, (list, tuple)):
        raw = [str(x).strip() for x in models if str(x).strip()]
        if not raw:
            return list(MODEL_ZOO.keys())
        unknown = [x for x in raw if x not in MODEL_ZOO]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}; available={list(MODEL_ZOO.keys())}")
        return raw
    raise TypeError("models must be str or list/tuple of model keys")


def _normalize_device(device: str, ctx_id: int) -> str:
    # 若未显式传 device，则默认跟随 ctx_id；并做 CUDA 可见卡号兜底。
    dev = device.strip() if device.strip() else (f"cuda:{ctx_id}" if ctx_id >= 0 else "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if dev.startswith("cuda"):
        try:
            d = torch.device(dev)
            if d.index is not None:
                cnt = int(torch.cuda.device_count())
                if cnt > 0 and int(d.index) >= cnt:
                    return "cuda:0"
        except Exception:
            pass
    return dev


def _gpu_id_from_device(device: str) -> int:
    """Extract numeric GPU index from 'cuda:N' device string; defaults to 0."""
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            pass
    return 0


def _pick_dtype(spec: ModelSpec, device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    if spec.torch_dtype == "bf16":
        return torch.bfloat16
    if spec.torch_dtype == "fp32":
        return torch.float32
    if spec.torch_dtype == "fp16":
        return torch.float16
    if spec.loader in ("auto_inpaint", "sd3_img2img"):
        return torch.float16
    return torch.bfloat16


def _pretrained_kwargs(spec: ModelSpec, dtype: torch.dtype) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"torch_dtype": dtype}
    if spec.variant is not None:
        kw["variant"] = spec.variant
    if spec.use_safetensors is not None:
        kw["use_safetensors"] = spec.use_safetensors
    if spec.trust_remote_code:
        kw["trust_remote_code"] = True
    return kw


def _load_pretrained_flexible(load_fn, model_id: str, **kwargs):
    # 小容错：不同仓库对 use_safetensors 支持不一致，做 3 次尝试。
    attempts: List[Dict[str, Any]] = []
    if "use_safetensors" in kwargs:
        v = kwargs["use_safetensors"]
        attempts.append(dict(kwargs))
        attempts.append({**kwargs, "use_safetensors": (not bool(v))})
        attempts.append({k: v2 for k, v2 in kwargs.items() if k != "use_safetensors"})
    else:
        attempts.append(dict(kwargs))

    last_err: Optional[Exception] = None
    for attempt in attempts:
        try:
            return load_fn(model_id, **attempt)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("unexpected load_pretrained failure")


def _move_pipe_to_device(pipe: Any, device: str, cpu_offload: str) -> str:
    # 统一处理 full-GPU / model-offload / sequential-offload 三种上卡策略。
    # 所有 offload 调用必须传 gpu_id，否则 diffusers 默认使用 cuda:0。
    mode = str(cpu_offload or "auto").lower().strip()
    if not hasattr(pipe, "to"):
        return device

    gpu_id = _gpu_id_from_device(device)

    if not device.startswith("cuda"):
        pipe.to("cpu")
        return "cpu"

    if mode == "model" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        return device
    if mode == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        return device

    if mode == "auto":
        try:
            pipe.to(device)
            return device
        except Exception:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                return device
            raise

    pipe.to(device)
    return device


def _call_pipe_filtered(pipe: Any, **kwargs):
    # 多种 pipeline 的 __call__ 参数不一致，按签名自动过滤。
    sig = inspect.signature(pipe.__call__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return pipe(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return pipe(**filtered)


def _to_pil_image(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        arr = t.numpy()
        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmin(arr)) < 0.0:
                arr = (arr + 1.0) / 2.0
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.shape[-1] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        if arr.shape[-1] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")
        return Image.fromarray(arr, mode="RGB")

    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.shape[-1] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")
        if arr.shape[-1] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        return Image.fromarray(arr, mode="RGB")

    raise RuntimeError(f"Unsupported image output type: {type(x)}")


def _extract_images(output: Any) -> List[Image.Image]:
    if output is None:
        return []

    def _to_pil_list(value: Any) -> List[Image.Image]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_to_pil_image(x) for x in value]
        if torch.is_tensor(value) and value.ndim == 4:
            return [_to_pil_image(value[i]) for i in range(int(value.shape[0]))]
        if isinstance(value, np.ndarray) and value.ndim == 4:
            return [_to_pil_image(value[i]) for i in range(int(value.shape[0]))]
        return [_to_pil_image(value)]

    # Diffusers BaseOutput 通常是 dict-like（OrderedDict），不同 pipeline 的键名可能不同。
    if isinstance(output, dict):
        for k in ("images", "image", "sample", "samples", "frames"):
            if k in output and output[k] is not None:
                imgs = _to_pil_list(output[k])
                if imgs:
                    return imgs
        if len(output) > 0:
            first_val = next(iter(output.values()))
            imgs = _to_pil_list(first_val)
            if imgs:
                return imgs

    # 普通对象：优先读常见属性名
    for attr in ("images", "image", "sample", "samples", "frames"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            imgs = _to_pil_list(value)
            if imgs:
                return imgs

    if isinstance(output, list):
        return [_to_pil_image(x) for x in output]
    return [_to_pil_image(output)]


def _clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def init_face_app(ctx_id: int, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "landmark_2d_106"])
    app.prepare(ctx_id=int(ctx_id), det_size=(int(det_size), int(det_size)))
    return app


def detect_all_faces(face_app: FaceAnalysis, img_pil: Image.Image) -> List[Any]:
    faces = face_app.get(pil_to_bgr_np(img_pil))
    if not faces:
        return []

    def _area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    faces = sorted(faces, key=_area, reverse=True)
    return list(faces)


def _get_face_landmarks(face: Any) -> Optional[np.ndarray]:
    if face is None:
        return None
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        pts = np.asarray(face.landmark_2d_106, dtype=np.float32)
        if pts.size > 0:
            return pts
    if hasattr(face, "kps") and face.kps is not None:
        pts = np.asarray(face.kps, dtype=np.float32)
        if pts.size > 0:
            return pts
    return None


def _get_face_kps5(face: Any) -> Optional[np.ndarray]:
    if face is None:
        return None
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32)
        if kps.shape == (5, 2):
            return kps
    return None


def _kps5_yaw_proxy(kps5: Optional[np.ndarray]) -> Optional[float]:
    if kps5 is None or kps5.shape != (5, 2):
        return None
    left_eye, right_eye, nose = kps5[0], kps5[1], kps5[2]
    dl = float(np.linalg.norm(nose - left_eye))
    dr = float(np.linalg.norm(nose - right_eye))
    return float(abs(dl - dr) / (dl + dr + 1e-6))


def _detect_largest_face_kps5(face_app: FaceAnalysis, img_pil: Image.Image) -> Optional[np.ndarray]:
    faces = face_app.get(pil_to_bgr_np(img_pil))
    if not faces:
        return None

    def _area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    faces = sorted(faces, key=_area, reverse=True)
    return _get_face_kps5(faces[0])


def align_generated_roi_by_kps5(
    face_app: FaceAnalysis,
    ref_roi: Image.Image,
    gen_roi: Image.Image,
    max_err_norm: float = 0.06,
    max_abs_rot_deg: float = 7.0,
    max_scale_delta: float = 0.10,
) -> Image.Image:
    """
    用 5 点关键点做相似变换，把生成 ROI 对齐回参考 ROI，缓解几何错位。
    为避免“误对齐”导致人脸结构异常，加入轻量门控：
    - 关键点重投影误差过大 -> 跳过对齐
    - 旋转/缩放异常 -> 跳过对齐
    """
    kps_ref = _detect_largest_face_kps5(face_app, ref_roi)
    kps_gen = _detect_largest_face_kps5(face_app, gen_roi)
    if kps_ref is None or kps_gen is None:
        return gen_roi
    try:
        M, _ = cv2.estimateAffinePartial2D(kps_gen, kps_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return gen_roi

        # 对齐质量门控：避免把本来正常的人脸“拉坏”。
        pred = (M[:, :2] @ kps_gen.T + M[:, 2:3]).T
        err = np.linalg.norm(pred - kps_ref, axis=1)
        iod = float(np.linalg.norm(kps_ref[0] - kps_ref[1])) + 1e-6
        err_norm = float(err.mean() / iod)
        if err_norm > float(max_err_norm):
            return gen_roi

        a, b, _tx = M[0]
        _c, d, _ty = M[1]
        scale = float(np.sqrt(a * a + b * b))
        rot_deg = float(np.degrees(np.arctan2(b, a)))
        if abs(scale - 1.0) > float(max_scale_delta):
            return gen_roi
        if abs(rot_deg) > float(max_abs_rot_deg):
            return gen_roi

        arr = np.asarray(gen_roi.convert("RGB"))
        h, w = arr.shape[:2]
        warped = cv2.warpAffine(
            arr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return Image.fromarray(warped, mode="RGB")
    except Exception:
        return gen_roi


def _bbox_expand_and_clip(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(float(pad_ratio) * bw))
    py = int(round(float(pad_ratio) * bh))
    X1 = max(0, x1 - px)
    Y1 = max(0, y1 - py)
    X2 = min(width, x2 + px)
    Y2 = min(height, y2 + py)
    if X2 <= X1:
        X2 = min(width, X1 + 1)
    if Y2 <= Y1:
        Y2 = min(height, Y1 + 1)
    return (X1, Y1, X2, Y2)


def _bbox_union(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_from_mask(mask_pil: Image.Image, threshold: int = 12) -> Optional[Tuple[int, int, int, int]]:
    m = np.asarray(mask_pil.convert("L"), dtype=np.uint8)
    ys, xs = np.where(m > int(threshold))
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2, y2)


def _make_landmark_mask(
    size_wh: Tuple[int, int],
    landmarks: np.ndarray,
    pad_ratio: float,
    blur: int,
) -> Image.Image:
    W, H = size_wh
    pts = landmarks.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return Image.new("L", (W, H), 0)

    center = pts.mean(axis=0, keepdims=True)
    scale = 1.0 + max(0.0, float(pad_ratio))
    pts = (pts - center) * scale + center
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

    hull = cv2.convexHull(pts.astype(np.int32))
    m = np.zeros((H, W), dtype=np.uint8)
    if hull is not None and len(hull) >= 3:
        cv2.fillConvexPoly(m, hull, 255)
    if int(blur) > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=int(blur), sigmaY=int(blur))
    return Image.fromarray(m, mode="L")


def _make_bbox_ellipse_mask(
    size_wh: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    pad_ratio: float,
    blur: int,
) -> Image.Image:
    W, H = size_wh
    x1, y1, x2, y2 = _bbox_expand_and_clip(bbox, W, H, pad_ratio=pad_ratio)
    m = np.zeros((H, W), dtype=np.uint8)
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    ax = max(1, int(round((x2 - x1) * 0.5)))
    ay = max(1, int(round((y2 - y1) * 0.5)))
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    if int(blur) > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=int(blur), sigmaY=int(blur))
    return Image.fromarray(m, mode="L")


def make_face_mask(
    size_wh: Tuple[int, int],
    face: Any,
    pad_ratio: float,
    blur: int,
) -> Image.Image:
    pts = _get_face_landmarks(face)
    if pts is not None:
        return _make_landmark_mask(size_wh=size_wh, landmarks=pts, pad_ratio=pad_ratio, blur=blur)
    bbox = tuple(int(v) for v in face.bbox)
    return _make_bbox_ellipse_mask(size_wh=size_wh, bbox=bbox, pad_ratio=pad_ratio, blur=blur)


def merge_masks_max(mask_a: Image.Image, mask_b: Image.Image) -> Image.Image:
    a = np.asarray(mask_a.convert("L"), dtype=np.uint8)
    b = np.asarray(mask_b.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.maximum(a, b), mode="L")


def clip_mask_to_bbox_soft(mask_pil: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    m = np.asarray(mask_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = m.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return Image.new("L", (W, H), 0)

    bin_box = np.zeros((H, W), dtype=np.uint8)
    bin_box[y1:y2, x1:x2] = 1

    feather = int(np.clip(min(x2 - x1, y2 - y1) * 0.06, 6.0, 32.0))
    dt = cv2.distanceTransform(bin_box, cv2.DIST_L2, 5)
    weight = np.clip(dt / float(max(feather, 1)), 0.0, 1.0).astype(np.float32)

    out = np.clip(m * weight, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), mode="L")


def _compute_symmetric_pad(w: int, h: int, multiple: int) -> Tuple[int, int, int, int]:
    mm = max(1, int(multiple))
    pad_w = (mm - (w % mm)) % mm
    pad_h = (mm - (h % mm)) % mm
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return left, top, right, bottom


def pad_image_and_mask_to_multiple(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    multiple: int,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    w, h = image_pil.size
    left, top, right, bottom = _compute_symmetric_pad(w, h, multiple)
    img = np.asarray(image_pil.convert("RGB"))
    m = np.asarray(mask_pil.convert("L"))

    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    m_pad = cv2.copyMakeBorder(m, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return Image.fromarray(img_pad), Image.fromarray(m_pad, mode="L"), (left, top, right, bottom)


def crop_back_to_original(
    out_pil: Image.Image,
    pad: Tuple[int, int, int, int],
    orig_size_wh: Tuple[int, int],
) -> Image.Image:
    left, top, _, _ = pad
    w, h = orig_size_wh
    out_crop = out_pil.crop((left, top, left + w, top + h))
    if out_crop.size != (w, h):
        out_crop = out_crop.resize((w, h), resample=Image.LANCZOS)
    return out_crop


def resize_image_and_mask_max_side(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    max_side: int,
) -> Tuple[Image.Image, Image.Image, float]:
    if int(max_side) <= 0:
        return image_pil, mask_pil, 1.0
    w, h = image_pil.size
    side = max(w, h)
    if side <= int(max_side):
        return image_pil, mask_pil, 1.0
    scale = float(max_side) / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img2 = image_pil.resize((nw, nh), resample=Image.LANCZOS)
    mask2 = mask_pil.resize((nw, nh), resample=Image.BILINEAR)
    return img2, mask2, scale


def resize_image_and_mask_by_scale(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    scale: float,
) -> Tuple[Image.Image, Image.Image, float]:
    if abs(float(scale) - 1.0) <= 1e-6:
        return image_pil, mask_pil, 1.0
    w, h = image_pil.size
    nw = max(1, int(round(float(w) * float(scale))))
    nh = max(1, int(round(float(h) * float(scale))))
    img2 = image_pil.resize((nw, nh), resample=Image.LANCZOS)
    mask2 = mask_pil.resize((nw, nh), resample=Image.BILINEAR)
    actual = float(max(nw / max(w, 1), nh / max(h, 1)))
    return img2, mask2, actual


def make_blend_alpha(mask_u8: np.ndarray, feather_ratio: float = 0.12) -> np.ndarray:
    m = (mask_u8.astype(np.uint8) >= 128).astype(np.uint8)
    if int(m.sum()) < 16:
        return np.zeros(m.shape, dtype=np.float32)
    h, w = m.shape[:2]
    feather = int(np.clip(min(h, w) * float(feather_ratio), 4.0, 32.0))
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    alpha = np.clip(dt / float(max(feather, 1)), 0.0, 1.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.9, sigmaY=0.9)
    return np.clip(alpha, 0.0, 1.0)


def blend_patch(
    base_patch: Image.Image,
    gen_patch: Image.Image,
    mask_patch: Image.Image,
    feather_ratio: float = 0.12,
) -> Image.Image:
    base = np.asarray(base_patch.convert("RGB"), dtype=np.float32)
    gen = np.asarray(gen_patch.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask_patch.convert("L"), dtype=np.uint8)
    alpha = make_blend_alpha(m, feather_ratio=feather_ratio)
    out = base * (1.0 - alpha[:, :, None]) + gen * alpha[:, :, None]
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def match_color_mean_std_in_mask(
    gen_patch: Image.Image,
    ref_patch: Image.Image,
    mask_patch: Image.Image,
) -> Image.Image:
    """
    在 mask 内将生成结果的亮度/对比度对齐到原 ROI，抑制“发白雾”。
    """
    gen = np.asarray(gen_patch.convert("RGB"), dtype=np.float32)
    ref = np.asarray(ref_patch.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask_patch.convert("L"), dtype=np.uint8)
    sel = m >= 128
    if int(sel.sum()) < 32:
        return gen_patch

    out = gen.copy()
    for c in range(3):
        g = gen[:, :, c]
        r = ref[:, :, c]
        g_sel = g[sel]
        r_sel = r[sel]
        g_mu = float(g_sel.mean())
        r_mu = float(r_sel.mean())
        g_std = float(g_sel.std() + 1e-6)
        r_std = float(r_sel.std() + 1e-6)
        mapped = ((g - g_mu) * (r_std / g_std)) + r_mu
        out[:, :, c] = np.where(sel, mapped, out[:, :, c])

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def smooth_artifacts_in_mask(
    gen_patch: Image.Image,
    mask_patch: Image.Image,
    sigma_color: float = 22.0,
    sigma_space: float = 5.0,
    mix: float = 0.45,
) -> Image.Image:
    """
    在 mask 内对高频异常纹理做轻度抑制（双边滤波+按残差自适应混合）。
    主要用于缓解“花脸/拼贴纹理”，尽量保留正常细节。
    """
    gen = np.asarray(gen_patch.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask_patch.convert("L"), dtype=np.uint8)
    if int((m >= 128).sum()) < 32:
        return gen_patch

    smooth = cv2.bilateralFilter(gen, d=0, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    diff = cv2.absdiff(gen, smooth)
    hf = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    m01 = np.clip(m.astype(np.float32) / 255.0, 0.0, 1.0)
    w = np.clip((hf - 0.05) / 0.25, 0.0, 1.0) * m01 * float(np.clip(mix, 0.0, 1.0))
    out = gen.astype(np.float32) * (1.0 - w[:, :, None]) + smooth.astype(np.float32) * w[:, :, None]
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def obfuscate_inside_mask(img_pil: Image.Image, mask_pil: Image.Image, seed: int) -> Image.Image:
    img = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask_pil.convert("L"), dtype=np.uint8)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=1.2, sigmaY=1.2)
    m01 = np.clip(m.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]

    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=7.0, sigmaY=7.0)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    noise = rng.normal(127.0, 28.0, size=img.shape).astype(np.float32)
    inner = 0.70 * smooth + 0.30 * noise

    out = img * (1.0 - m01) + inner * m01
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@dataclass
class _SdCppImagesOutput:
    images: List[Image.Image]


class SdCppQwenImageEdit2511GGUF:
    def __init__(
        self,
        diffusion_model_path: str,
        llm_path: str,
        llm_vision_path: str,
        vae_path: str,
    ):
        try:
            from stable_diffusion_cpp import StableDiffusion  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "stable-diffusion-cpp-python is required for qwen_image_edit_2511_gguf.\n"
                "Install example: CMAKE_ARGS='-DSD_CUDA=ON' pip install stable-diffusion-cpp-python"
            ) from e

        self._sd = StableDiffusion(
            diffusion_model_path=diffusion_model_path,
            llm_path=llm_path,
            llm_vision_path=llm_vision_path,
            vae_path=vae_path,
            offload_params_to_cpu=True,
            diffusion_flash_attn=True,
            qwen_image_zero_cond_t=True,
            flow_shift=3.0,
            enable_mmap=True,
            image_resize_method="resize",
            verbose=False,
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
        strength: float = 0.85,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        **_kwargs,
    ) -> _SdCppImagesOutput:
        if seed is None:
            try:
                seed = int(generator.initial_seed()) if generator is not None else 42
            except Exception:
                seed = 42
        image = image.convert("RGB")
        w = int(width) if width is not None else int(image.size[0])
        h = int(height) if height is not None else int(image.size[1])

        out = self._sd.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            ref_images=[image],
            mask_image=mask_image,
            width=w,
            height=h,
            sample_steps=int(num_inference_steps),
            cfg_scale=float(guidance_scale),
            strength=float(strength),
            seed=int(seed),
            sample_method="euler",
        )
        return _SdCppImagesOutput(images=out)


def _build_qwen_cpp_pipe(spec: ModelSpec):
    # Qwen GGUF 走 stable-diffusion.cpp：下载 diffusion/vae/llm/mmproj 四个权重文件。
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        raise RuntimeError("huggingface_hub is required for qwen model download") from e

    def _dl(repo_id: str, filename: str) -> str:
        return hf_hub_download(repo_id=repo_id, filename=filename)

    diffusion_path = _dl(spec.model_id, "qwen-image-edit-2511-Q4_K_M.gguf")
    vae_path = _dl("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors")
    llm_repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    llm_path = _dl(llm_repo, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
    llm_vision_path = _dl(llm_repo, "mmproj-BF16.gguf")
    return SdCppQwenImageEdit2511GGUF(
        diffusion_model_path=diffusion_path,
        llm_path=llm_path,
        llm_vision_path=llm_vision_path,
        vae_path=vae_path,
    )


def build_pipe(spec: ModelSpec, device: str, cpu_offload: str):
    # 根据 loader 分派模型构建逻辑，返回 (pipe, real_device)。
    dtype = _pick_dtype(spec, device)
    pre_kw = _pretrained_kwargs(spec, dtype)

    if spec.loader == "auto_inpaint":
        pipe = _load_pretrained_flexible(AutoPipelineForInpainting.from_pretrained, spec.model_id, **pre_kw)
        real_device = _move_pipe_to_device(pipe, device, cpu_offload)
        return pipe, real_device

    if spec.loader == "flux_fill_nf4":
        if FluxFillPipeline is None or FluxTransformer2DModel is None or T5EncoderModel is None:
            raise RuntimeError("Current env lacks FLUX NF4 dependencies in diffusers/transformers")
        nf4_repo = "diffusers/FLUX.1-Fill-dev-nf4"
        transformer = FluxTransformer2DModel.from_pretrained(
            nf4_repo,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            nf4_repo,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )
        pipe = FluxFillPipeline.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
        )
        real_device = _move_pipe_to_device(pipe, device, cpu_offload)
        return pipe, real_device

    if spec.loader == "diffusion_edit":
        load_fn = DiffusionPipeline.from_pretrained
        mid_low = spec.model_id.lower()
        if "kandinsky" in mid_low and "i2i" in mid_low and Kandinsky5I2IPipeline is not None:
            load_fn = Kandinsky5I2IPipeline.from_pretrained

        pipe = _load_pretrained_flexible(load_fn, spec.model_id, **pre_kw)
        if spec.key == "flux2_klein_9b" and hasattr(pipe, "enable_model_cpu_offload"):
            if str(cpu_offload).lower().strip() in ("auto", "model"):
                gpu_id = _gpu_id_from_device(device)
                pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                return pipe, device
        real_device = _move_pipe_to_device(pipe, device, cpu_offload)
        return pipe, real_device

    if spec.loader == "sd3_img2img":
        if StableDiffusion3Img2ImgPipeline is None:
            raise RuntimeError("StableDiffusion3Img2ImgPipeline is unavailable in current diffusers")
        pipe = _load_pretrained_flexible(StableDiffusion3Img2ImgPipeline.from_pretrained, spec.model_id, **pre_kw)
        mode = str(cpu_offload).lower().strip()
        real_device = _move_pipe_to_device(pipe, device, "model" if mode == "auto" else mode)
        return pipe, real_device

    if spec.loader == "qwen_cpp":
        # stable-diffusion-cpp (ggml-cuda) 不走 PyTorch 的 device 分配，
        # 需要在构建前设置默认 CUDA 设备，使 ggml 使用正确的 GPU。
        if device.startswith("cuda"):
            gpu_id = _gpu_id_from_device(device)
            torch.cuda.set_device(gpu_id)
        pipe = _build_qwen_cpp_pipe(spec)
        return pipe, device

    raise ValueError(f"Unknown loader: {spec.loader}")


def supports_mask_inpaint(spec: ModelSpec) -> bool:
    return spec.loader in ("auto_inpaint", "flux_fill_nf4", "qwen_cpp")


def model_plan(spec: ModelSpec, cfg: RunConfig) -> RuntimePlan:
    # 模型专属推理参数覆盖：尽量贴合各 backbone 的稳定区间。
    prompt = cfg.prompt.strip() if cfg.prompt.strip() else DEFAULT_PROMPT
    negative = cfg.negative_prompt.strip() if cfg.negative_prompt.strip() else DEFAULT_NEGATIVE
    steps = int(cfg.steps)
    guidance = float(cfg.guidance)
    strength = float(cfg.strength)

    if spec.key == "sd35_img2img":
        steps = 22
        guidance = 2.5
        strength = min(strength, 0.40)
    elif spec.key == "flux2_klein_9b":
        steps = 4
        guidance = 1.0
        strength = min(strength, 0.70)
        prompt = FLUX2_PROMPT
    elif spec.key == "qwen_image_edit_2511_gguf":
        steps = 20
        guidance = 2.5
        strength = max(strength, 0.90)
        prompt = QWEN_PROMPT
        negative = (negative + ", same person, same identity, look-alike, recognizable face").strip(", ")
    elif spec.key == "longcat_image_edit_turbo":
        steps = max(8, min(steps, 28))
    elif spec.key == "kandinsky5_i2i_lite":
        guidance = min(guidance, 4.0)
        strength = min(strength, 0.60)
    elif spec.key == "sdxl_inpaint":
        # SDXL inpaint 在该数据集的小脸/复杂光照样本中容易出现“花脸”。
        # 这里做更稳健的全局上限，优先保证自然度。
        steps = min(steps, 36)
        guidance = min(guidance, 2.8)
        strength = min(strength, 0.40)
        negative = (
            negative
            + ", face paint, painted face, weird makeup, clown makeup, patchy skin, mosaic face, collage face"
            + ", text, letters, symbols, logo, watermark, glyph"
        ).strip(", ")

    strength = float(np.clip(strength, 0.05, 0.95))
    return RuntimePlan(
        prompt=prompt,
        negative_prompt=negative,
        steps=int(steps),
        guidance=float(guidance),
        strength=float(strength),
    )


def run_model_call(
    pipe: Any,
    spec: ModelSpec,
    device: str,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    plan: RuntimePlan,
    seed: int,
) -> Image.Image:
    # 统一一次模型调用：
    # - 支持 mask inpaint 的模型直接用原 ROI+mask
    # - 仅 edit/img2img 的模型先做 mask 内扰动再编辑
    if supports_mask_inpaint(spec):
        # Qwen 在部分样本上会保留较多身份特征；对 mask 内做轻扰动可提升去身份化。
        if spec.loader == "qwen_cpp":
            model_input = obfuscate_inside_mask(image_pil, mask_pil, seed=seed)
        else:
            model_input = image_pil
    else:
        # Kandinsky/SD3.5 对强扰动初值更敏感，容易出现局部发白和细节糊化。
        # 这里最小改动：这两类模型直接用原 ROI 作为编辑输入。
        if spec.key in ("kandinsky5_i2i_lite", "sd35_img2img"):
            model_input = image_pil
        else:
            model_input = obfuscate_inside_mask(image_pil, mask_pil, seed=seed)

    gen = None
    if spec.loader != "qwen_cpp":
        gdev = device if device.startswith("cuda") else "cpu"
        gen = torch.Generator(device=gdev).manual_seed(int(seed))

    kwargs = dict(
        prompt=plan.prompt,
        negative_prompt=plan.negative_prompt,
        image=model_input,
        mask_image=mask_pil,
        num_inference_steps=int(plan.steps),
        guidance_scale=float(plan.guidance),
        strength=float(plan.strength),
        width=int(image_pil.size[0]),
        height=int(image_pil.size[1]),
        generator=gen,
        seed=int(seed),
    )
    kwargs.update(spec.extra_call_kwargs)

    output = _call_pipe_filtered(pipe, **kwargs)
    images = _extract_images(output)
    if not images:
        raise RuntimeError("pipeline returned no images")
    out = images[0].convert("RGB")
    if out.size != image_pil.size:
        out = out.resize(image_pil.size, resample=Image.LANCZOS)
    return out


def run_local_swap_once(
    image_pil: Image.Image,
    face_app: FaceAnalysis,
    pipe: Any,
    spec: ModelSpec,
    cfg: RunConfig,
    plan: RuntimePlan,
) -> Image.Image:
    # 单张核心流水线：
    # 1) 检测人脸 -> 2) 生成局部 mask -> 3) 估计 ROI
    # 4) ROI 局部重绘 -> 5) 柔和融合贴回
    W, H = image_pil.size
    faces = detect_all_faces(face_app, image_pil)
    if not faces:
        raise RuntimeError("No face detected")

    if not cfg.replace_all_faces:
        faces = faces[:1]
    if int(cfg.max_faces) > 0:
        faces = faces[: int(cfg.max_faces)]

    out = image_pil.copy()
    for face_idx, face in enumerate(faces):
        bbox = tuple(int(v) for v in face.bbox)
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        if min(bw, bh) < int(cfg.min_face_edit):
            continue

        pts = _get_face_landmarks(face)
        if cfg.skip_if_no_kps and pts is None:
            continue

        # 难例判定：小脸或较强侧脸。
        face_scale = float(min(bw, bh) / max(1.0, min(W, H)))
        yaw_proxy = _kps5_yaw_proxy(_get_face_kps5(face))
        is_hard = (face_scale < float(cfg.hard_face_scale_thr)) or (
            yaw_proxy is not None and yaw_proxy >= float(cfg.hard_yaw_thr)
        )

        local_mask_blur = int(cfg.mask_blur)
        local_roi_pad_ratio = float(cfg.roi_pad_ratio)
        local_steps = int(plan.steps)
        local_guidance = float(plan.guidance)
        local_strength = float(plan.strength)
        local_feather = 0.08 if spec.key in ("kandinsky5_i2i_lite", "sd35_img2img") else 0.12
        if is_hard:
            local_mask_blur = max(2, int(round(float(cfg.mask_blur) * float(cfg.hard_mask_blur_scale))))
            local_roi_pad_ratio = float(cfg.roi_pad_ratio) + float(cfg.hard_roi_pad_extra)
            local_strength = max(0.05, float(plan.strength) - float(cfg.hard_strength_drop))
            local_feather = float(local_feather) * float(cfg.hard_feather_scale)
            # SDXL 在小脸/难脸上更容易出现局部“花脸”；收窄可编辑强度区间。
            if spec.key == "sdxl_inpaint":
                local_steps = min(local_steps, 36)
                local_guidance = min(local_guidance, 2.8)
                local_strength = min(local_strength, 0.34)

        # 人脸编辑域：关键点凸包（无关键点回退椭圆），再软裁到更大的头部范围。
        face_mask = make_face_mask(
            size_wh=(W, H),
            face=face,
            pad_ratio=float(cfg.pad_ratio),
            blur=int(local_mask_blur),
        )
        use_union_mask = bool(cfg.hard_union_mask) and (
            is_hard or (spec.key == "sdxl_inpaint" and face_scale < 0.16)
        )
        if use_union_mask:
            bbox_mask = _make_bbox_ellipse_mask(
                size_wh=(W, H),
                bbox=bbox,
                pad_ratio=float(cfg.pad_ratio),
                blur=int(local_mask_blur),
            )
            face_mask = merge_masks_max(face_mask, bbox_mask)
        clip_box = _bbox_expand_and_clip(bbox, W, H, pad_ratio=0.80)
        face_mask = clip_mask_to_bbox_soft(face_mask, clip_box)
        if np.asarray(face_mask.convert("L"), dtype=np.uint8).sum() < 128:
            continue

        # ROI 同时参考 mask 包围盒与 bbox 扩张框，取并集后再轻微扩展。
        roi_by_mask = _bbox_from_mask(face_mask, threshold=12)
        roi_by_face = _bbox_expand_and_clip(bbox, W, H, pad_ratio=float(local_roi_pad_ratio))
        roi_box = roi_by_face if roi_by_mask is None else _bbox_union(roi_by_mask, roi_by_face)
        roi_box = _bbox_expand_and_clip(roi_box, W, H, pad_ratio=0.08)
        x1, y1, x2, y2 = roi_box

        roi_img_base = out.crop((x1, y1, x2, y2)).convert("RGB")
        roi_mask_base = face_mask.crop((x1, y1, x2, y2)).convert("L")
        if np.asarray(roi_mask_base, dtype=np.uint8).sum() < 128:
            continue

        # 小脸/侧脸难例：先放大 ROI 到最小可编辑尺度，再按 max_side 控显存。
        roi_img_work = roi_img_base
        roi_mask_work = roi_mask_base
        pre_scale = 1.0
        if is_hard:
            # 用“脸框短边”估计放大倍率，避免 ROI 偏大时小脸仍然分辨率不足。
            face_short_edge = max(1.0, float(min(bw, bh)))
            want_scale = float(cfg.hard_face_min_px) / float(face_short_edge)
            want_scale = float(np.clip(want_scale, 1.0, float(cfg.hard_max_upscale)))
            roi_img_work, roi_mask_work, pre_scale = resize_image_and_mask_by_scale(
                roi_img_work, roi_mask_work, want_scale
            )

        # 大 ROI 再缩放控显存，后续统一还原回原尺寸。
        roi_img_rs, roi_mask_rs, scale = resize_image_and_mask_max_side(
            roi_img_work, roi_mask_work, max_side=int(cfg.roi_max_side)
        )
        multiple = int(spec.pad_multiple or cfg.default_pad_multiple)
        roi_img_pad, roi_mask_pad, pad = pad_image_and_mask_to_multiple(roi_img_rs, roi_mask_rs, multiple=multiple)

        # 每张图每张脸扰动 seed，减少多人同脸“同质化”。
        local_seed = int(cfg.seed) + (face_idx + 1) * 10007 + (int(x1) + int(y1)) % 997
        local_plan = RuntimePlan(
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            steps=int(local_steps),
            guidance=float(local_guidance),
            strength=float(local_strength),
        )

        out_pad = run_model_call(
            pipe=pipe,
            spec=spec,
            device=cfg.device,
            image_pil=roi_img_pad,
            mask_pil=roi_mask_pad,
            plan=local_plan,
            seed=local_seed,
        )

        out_rs = crop_back_to_original(out_pad, pad=pad, orig_size_wh=roi_img_rs.size)
        if abs(scale - 1.0) > 1e-6:
            out_roi_up = out_rs.resize(roi_img_work.size, resample=Image.LANCZOS)
        else:
            out_roi_up = out_rs

        if abs(pre_scale - 1.0) > 1e-6:
            out_roi = out_roi_up.resize(roi_img_base.size, resample=Image.LANCZOS)
        else:
            out_roi = out_roi_up

        # 几何漂移回对齐：扩展到所有模型。
        # 非难例使用更严格门控，减少“误对齐”导致的人脸异常。
        out_roi = align_generated_roi_by_kps5(
            face_app,
            roi_img_base,
            out_roi,
            max_err_norm=(0.09 if is_hard else 0.06),
            max_abs_rot_deg=(9.0 if is_hard else 7.0),
            max_scale_delta=0.10,
        )

        # SDXL 在复杂光照样本上先做颜色回对齐 + 高频伪影抑制，缓解“花脸”。
        if spec.key == "sdxl_inpaint":
            out_roi = match_color_mean_std_in_mask(out_roi, roi_img_base, roi_mask_base)

        # Kandinsky/SD3.5 先做 mask 内颜色回对齐，降低“白雾/灰蒙”。
        if spec.key in ("kandinsky5_i2i_lite", "sd35_img2img"):
            out_roi = match_color_mean_std_in_mask(out_roi, roi_img_base, roi_mask_base)

        # 用软 alpha 融合贴回，降低边缘接缝与贴片感。
        # 对 Kandinsky/SD3.5 收窄融合羽化带，减少边缘发白光晕感。
        blend = blend_patch(roi_img_base, out_roi, roi_mask_base, feather_ratio=float(local_feather))
        out.paste(blend, (x1, y1))

    return out


def _resolve_paths_for_model(cfg: RunConfig, model_key: str) -> Tuple[Path, Optional[Path], Optional[Path]]:
    # 当前工程约定：
    # outputs/<dataset>/demo_m4/<model_key>
    # reports/<dataset>/demo_m4/<model_key>
    # single 模式：不走 outputs 目录，直接使用 single_out_img 所在目录
    ds = _dataset_tag(cfg.input_dir)
    if cfg.mode == "single":
        single_out = Path(str(cfg.single_out_img).strip()) if str(cfg.single_out_img).strip() else None
        if single_out is None:
            raise ValueError("single mode requires single_out_img")
        out_dir = single_out.parent if str(single_out.parent) else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        # single 模式不产生 report/runtime 文件
        return out_dir, None, None

    out_dir = Path(cfg.output_root) / ds / "demo_m4" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path(cfg.report_root) / ds / "demo_m4" / model_key
    runtime_json = report_dir / "runtime.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, report_dir, runtime_json


def run_one_model(cfg: RunConfig, spec: ModelSpec, image_paths: List[Path]) -> Dict[str, Any]:
    out_dir, report_dir, runtime_json = _resolve_paths_for_model(cfg, spec.key)
    plan = model_plan(spec, cfg)

    print(f"\n[MODEL] {spec.key}")
    print(f"  - model_id={spec.model_id}")
    print(f"  - loader={spec.loader}")
    print(f"  - device={cfg.device}")
    print(f"  - output_dir={out_dir}")
    if report_dir is not None:
        print(f"  - report_dir={report_dir}")
    else:
        print("  - report_dir=<disabled in single mode>")
    print(
        "  - steps={s}, guidance={g}, strength={st}".format(
            s=plan.steps,
            g=plan.guidance,
            st=plan.strength,
        )
    )

    t0_model = time.perf_counter()
    runtime: Dict[str, Dict[str, float]] = {}
    num_ok, num_err, num_skip = 0, 0, 0
    eval_summary: Dict[str, Any] = {}

    pipe = None
    face_app = None
    status = "ok"
    error = ""
    try:
        # 在加载任何模型前钉住默认 CUDA 设备，
        # 防止第三方库（ggml、ONNX Runtime 等）偷跑 cuda:0。
        if cfg.device.startswith("cuda"):
            torch.cuda.set_device(_gpu_id_from_device(cfg.device))

        face_app = init_face_app(ctx_id=cfg.ctx_id, det_size=cfg.det_size)
        pipe, real_device = build_pipe(spec, device=cfg.device, cpu_offload=cfg.cpu_offload)
        model_cfg = RunConfig(**{**cfg.__dict__, "device": real_device})

        # 先做批量生成（或 single 的 1 张生成）。
        total = len(image_paths)
        for idx, in_path in enumerate(image_paths, 1):
            if cfg.mode == "single":
                single_out = Path(cfg.single_out_img.strip())
                out_path = single_out
            else:
                out_path = out_dir / in_path.name
            if cfg.skip_existing and out_path.exists():
                num_skip += 1
                print(f"[{idx:03d}/{total}] SKIP {in_path.name}")
                continue

            t0_img = time.perf_counter()
            try:
                in_img = load_image_rgb(str(in_path))
                out_img = run_local_swap_once(
                    image_pil=in_img,
                    face_app=face_app,
                    pipe=pipe,
                    spec=spec,
                    cfg=model_cfg,
                    plan=plan,
                )
                out_img.save(str(out_path))
                dt = time.perf_counter() - t0_img
                runtime[in_path.name] = {"seconds": round(dt, 3)}
                num_ok += 1
                print(f"[{idx:03d}/{total}] OK   {in_path.name} ({dt:.2f}s)")
            except Exception as e:
                num_err += 1
                print(f"[{idx:03d}/{total}] ERR  {in_path.name}: {e}")
                if cfg.fail_fast:
                    raise

        # batch 模式记录 runtime，single 模式不写 report/runtime 文件。
        if runtime_json is not None:
            with open(runtime_json, "w", encoding="utf-8") as f:
                json.dump(runtime, f, ensure_ascii=False, indent=2)

        # Batch 模式：强制先批量生成，再评估（与当前输出路径保持一致）
        if cfg.mode == "batch":
            if cfg.no_eval:
                print("[WARN] no_eval is ignored in batch mode; evaluation is always enabled.")
            if report_dir is None or runtime_json is None:
                raise RuntimeError("internal error: batch mode requires report_dir/runtime_json")
            existing_outputs = [p for p in image_paths if (out_dir / p.name).is_file()]
            if not existing_outputs:
                print("[WARN] batch eval skipped: no generated files found in output_dir")
            else:
                print("[INFO] batch generation finished, start evaluation ...")
                from eval import EvalConfig, evaluate_folder

                system_meta = {
                    "runner": "demo_m4_standalone",
                    "model_key": spec.key,
                    "model_id": spec.model_id,
                    "loader": spec.loader,
                    "pipeline": "local_roi_edit_and_mask_blend",
                    "goal": "swap_identity_keep_background_pose_expression",
                    "mask": f"inner_face_landmark_mask(pad_ratio={cfg.pad_ratio}, blur={cfg.mask_blur})",
                }
                eval_summary = evaluate_folder(
                    EvalConfig(
                        input_dir=cfg.input_dir,
                        output_dir=str(out_dir),
                        report_dir=str(report_dir),
                        ctx_id=cfg.ctx_id,
                        det_size=cfg.det_size,
                        pad_ratio=cfg.pad_ratio,
                        mask_blur=cfg.mask_blur,
                        runtime_json=str(runtime_json),
                        generation_seed=cfg.seed,
                        comparison_mode="system",
                        system_meta=system_meta,
                    )
                )
        elif (not cfg.no_eval) and (cfg.mode != "batch"):
            print("[INFO] single mode: skip eval (batch mode will run generation -> eval).")
    except Exception as e:
        status = "error"
        error = str(e)
    finally:
        pipe = None
        face_app = None
        _clear_gpu()

    elapsed = time.perf_counter() - t0_model
    return {
        "model_key": spec.key,
        "status": status,
        "error": error,
        "output_dir": (str(Path(cfg.single_out_img).parent) if cfg.mode == "single" else str(out_dir)),
        "report_dir": (str(report_dir) if report_dir is not None else ""),
        "runtime_json": (str(runtime_json) if runtime_json is not None else ""),
        "num_input": len(image_paths),
        "num_ok": num_ok,
        "num_err": num_err,
        "num_skip": num_skip,
        "elapsed_seconds": round(elapsed, 3),
        "eval_summary": eval_summary,
    }


def run_pipeline(
    *,
    mode: str = "batch",
    models: Any = "all",
    input_dir: str = "posture-demo",
    single_in_img: str = "",
    single_out_img: str = "",
    output_root: str = "outputs",
    report_root: str = "reports",
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 42,
    steps: int = 35,
    guidance: float = 4.5,
    strength: float = 0.70,
    pad_ratio: float = 0.35,
    mask_blur: int = 8,
    roi_pad_ratio: float = 0.45,
    roi_max_side: int = 1024,
    pad_multiple: int = 8,
    det_size: int = 640,
    ctx_id: int = 0,
    device: str = "",
    cpu_offload: str = "auto",
    only_largest_face: bool = False,
    max_faces: int = 0,
    min_face_edit: int = 20,
    skip_if_no_kps: bool = False,
    hard_face_scale_thr: float = 0.08,
    hard_yaw_thr: float = 0.14,
    hard_face_min_px: int = 160,
    hard_max_upscale: float = 3.0,
    hard_mask_blur_scale: float = 0.70,
    hard_roi_pad_extra: float = 0.08,
    hard_strength_drop: float = 0.12,
    hard_feather_scale: float = 0.85,
    hard_union_mask: bool = True,
    skip_existing: bool = False,
    no_eval: bool = False,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    model_keys = _normalize_models_input(models)
    mode2 = str(mode).strip().lower()
    if mode2 not in ("single", "batch"):
        raise ValueError("mode must be 'single' or 'batch'")

    if mode2 == "single":
        single_in = str(single_in_img).strip()
        if not single_in:
            raise ValueError("mode=single requires single_in_img")
        p = Path(single_in)
        if not p.is_file():
            raise FileNotFoundError(f"single input image not found: {single_in}")
        image_paths = [p]
        input_dir_for_report = str(p.parent)
    else:
        image_paths = _collect_images(str(input_dir))
        input_dir_for_report = str(input_dir)

    device2 = _normalize_device(str(device), int(ctx_id))
    cfg = RunConfig(
        mode=mode2,
        single_in_img=str(single_in_img),
        single_out_img=str(single_out_img),
        input_dir=input_dir_for_report,
        output_root=str(output_root),
        report_root=str(report_root),
        models=model_keys,
        prompt=str(prompt),
        negative_prompt=str(negative_prompt),
        seed=int(seed),
        steps=int(steps),
        guidance=float(guidance),
        strength=float(strength),
        pad_ratio=float(pad_ratio),
        mask_blur=int(mask_blur),
        roi_pad_ratio=float(roi_pad_ratio),
        roi_max_side=int(roi_max_side),
        default_pad_multiple=int(pad_multiple),
        det_size=int(det_size),
        ctx_id=int(ctx_id),
        device=device2,
        cpu_offload=str(cpu_offload),
        replace_all_faces=(not bool(only_largest_face)),
        max_faces=int(max_faces),
        min_face_edit=int(min_face_edit),
        skip_if_no_kps=bool(skip_if_no_kps),
        hard_face_scale_thr=float(hard_face_scale_thr),
        hard_yaw_thr=float(hard_yaw_thr),
        hard_face_min_px=int(hard_face_min_px),
        hard_max_upscale=float(hard_max_upscale),
        hard_mask_blur_scale=float(hard_mask_blur_scale),
        hard_roi_pad_extra=float(hard_roi_pad_extra),
        hard_strength_drop=float(hard_strength_drop),
        hard_feather_scale=float(hard_feather_scale),
        hard_union_mask=bool(hard_union_mask),
        skip_existing=bool(skip_existing),
        no_eval=bool(no_eval),
        fail_fast=bool(fail_fast),
    )

    print(f"[INFO] input_dir={cfg.input_dir}")
    print(f"[INFO] num_images={len(image_paths)}")
    print(f"[INFO] models={cfg.models}")
    print(f"[INFO] device={cfg.device}")

    t0_all = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for i, model_key in enumerate(cfg.models, 1):
        print(f"\n========== [{i}/{len(cfg.models)}] {model_key} ==========")
        spec = MODEL_ZOO[model_key]
        result = run_one_model(cfg=cfg, spec=spec, image_paths=image_paths)
        results.append(result)
        if result["status"] != "ok" and cfg.fail_fast:
            break

    wall = time.perf_counter() - t0_all
    ok_cnt = sum(1 for r in results if r.get("status") == "ok")
    err_cnt = sum(1 for r in results if r.get("status") != "ok")

    summary = {
        "runner": "demo_m4_standalone",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": cfg.input_dir,
        "num_images": len(image_paths),
        "models_requested": cfg.models,
        "num_models_ok": ok_cnt,
        "num_models_error": err_cnt,
        "wall_time_seconds": round(wall, 3),
        "results": results,
    }

    print("\n========== DONE ==========")
    if cfg.mode == "batch":
        ds = _dataset_tag(cfg.input_dir)
        summary_dir = Path(cfg.report_root) / ds / "demo_m4"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_safe_json(summary), f, ensure_ascii=False, indent=2)
        print(f"[INFO] summary_json={summary_path}")
    else:
        print("[INFO] single mode: summary/report files are disabled.")
    print(json.dumps(_safe_json(summary), ensure_ascii=False, indent=2))
    return summary
