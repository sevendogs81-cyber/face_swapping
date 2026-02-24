#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件作用: 统一入口运行多模型人脸替换与评估
"""
face_swapping 统一入口：single 单张 | batch → eval → report

使用 demo_m3 多模型人脸匿名化（虚拟化）
"""

import os

# =============================================================================
# 下载/缓存设置（需要在 import diffusers/huggingface_hub 之前设置环境变量）
# =============================================================================

# ---- 关闭 XET/CAS（部分网络环境下更稳定）----
HF_DISABLE_XET = True
if HF_DISABLE_XET:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ---- 下载超时（慢网/大模型更稳）----
# huggingface_hub 默认 10s，下载大文件时容易触发 Read timed out
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# ---- 下载加速（可选）：HuggingFace hf_transfer ----
# 说明：
# - hf_transfer 在部分网络环境会触发 “no permits available / error decoding response body”
# - 默认关闭更稳；如需开启请手动 export HF_HUB_ENABLE_HF_TRANSFER=1
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import json
from demo_m3 import (
    DemoConfig,
    run_identity_demo_once,
    run_identity_demo_batch,
    resolve_model_spec,
)
from eval import EvalConfig, evaluate_folder


# =============================================================================
# 可调参数（全部集中于此，便于复现与调参）
# =============================================================================

# ---- 运行模式 ----
MODE = "single"  # "single" 单张 | "batch" 批量+评估

# ---- I/O 路径 ----
INPUT_DIR = "posture-demo"              # 批量：输入图片目录
SINGLE_IN_IMG = "test.jpg"           # 单张模式：输入

# ---- 文本条件（匿名化：尽量保持姿态/同表情/同光照/肤色）----
PROMPT = (
    "a realistic photo of a person, same head pose and facial expression as the input image, "
    "consistent lighting, natural skin texture, sharp eyes, high detail face"
)
NEGATIVE_PROMPT = (
    "deformed, bad anatomy, extra face, extra eyes, blurry, low quality, "
    "cartoon, painting, uncanny, watermark, text"
)

# ---- 扩散推理超参数 ----
SEED = 42                            # 随机种子，固定可复现

STEPS = 35                           # 推理步数，越大越精细但更慢（25~35）
GUIDANCE = 7.5                       # CFG scale，越大越贴合 prompt（5~8）
STRENGTH = 0.7                      # Inpainting 强度：越低越保留原图姿态/光照，越高越“重画”（0.6~0.8 推荐）

# ---- Mask 参数（影响重绘区域与边缘融合）----
# 说明：你现在的贴回是“单 mask alpha 融合”，MASK_BLUR 过大容易出现脸部轮廓虚影/光晕
PAD_RATIO = 0.35                     # bbox 四周扩展比例，过小易裁到脸（0.3~0.5）
MASK_BLUR = 8                        # mask 高斯模糊 sigma，越大边缘越柔和但越容易虚影（6~12 更稳）

# ---- Face parsing labels（BiSeNetV2 19 类）----
# label 映射（常见）：1=skin,2/3=brow,4/5=eye,6=eyeglass,7/8=ear,9=earring,10=nose,11=mouth,12/13=lip,17=hair,18=hat
# 目标：把所有模型的编辑域收紧到“内脸”（含眼镜），避免头发/帽子/遮挡物被重绘造成贴片与结构漂移
FACE_PARSING_LABELS_HEAD = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)

# ---- 人脸检测 ----
DET_SIZE = 640                       # InsightFace 检测输入尺寸，越大对小脸更敏感（512/640）
CTX_ID = 7                           # GPU 编号，0=第一块，-1=CPU

# ---- 模型与设备 ----
# MODEL_ID 支持：MODEL_ZOO key（推荐，如 "sdxl_inpaint"）或 HF repo id
MODEL_ID = "sdxl_inpaint"
DEVICE = "cuda:7"                      # 多卡时用作默认 generator device
PAD_MULTIPLE = 8                       # 输出尺寸对齐倍数（多数模型 8/16）

# ---- 显存优化（单卡 OOM 时优先开这个）----
# cpu_offload: "none" | "model" | "sequential" | "auto"
CPU_OFFLOAD = "auto"

# ---- 输出目录（按模型隔离）----
_model_spec = resolve_model_spec(MODEL_ID)
MODEL_TAG = _model_spec.key
# NOTE: outputs/reports 将按「数据集名/模型」分目录，避免不同数据集互相覆盖
def _dataset_tag(path: str) -> str:
    p = os.path.normpath(path)
    tag = os.path.basename(p)
    return tag or "dataset"


def _resolve_io_paths(input_dir: str, model_tag: str):
    ds = _dataset_tag(input_dir)
    out_dir = os.path.join("outputs", ds, model_tag)
    report_dir = os.path.join("reports", ds, "eval", model_tag)
    runtime_json = os.path.join(report_dir, "runtime.json")
    return out_dir, report_dir, runtime_json


OUTPUT_DIR, REPORT_DIR, RUNTIME_JSON = _resolve_io_paths(INPUT_DIR, MODEL_TAG)
SINGLE_OUT_IMG = f"out_{MODEL_TAG}.jpg"           # 单张模式：输出（按模型分开）

# 模型若指定了 pad_multiple，优先使用
if _model_spec.pad_multiple is not None:
    PAD_MULTIPLE = _model_spec.pad_multiple

# FLUX 这类 pipeline 通常要求 16 对齐，提前 pad 可避免内部 resize
if _model_spec.loader in ("flux_fill_nf4",):
    PAD_MULTIPLE = 16
# SD3.5 Turbo 的推荐设置与 SDXL 不同：步数更少、guidance 更低，否则容易不写实/跑偏
if MODEL_ID == "sd35_img2img":
    STEPS = 22
    GUIDANCE = 2.5
    STRENGTH = 0.40

# FLUX.2-klein-9B 是 4 steps distilled：推荐 steps=4, guidance=1.0
if MODEL_ID == "flux2_klein_9b":
    STEPS = 4
    GUIDANCE = 1.0
    # flux2 更容易“过度皮肤细节”导致显老：用柔光+更干净的肤质描述来抑制皱纹/斑点
    PROMPT = (
        "a realistic photo of a person, same head pose and facial expression as the input image, "
        "consistent lighting, soft diffused lighting, clear smooth skin, even complexion, subtle retouch, "
        "sharp eyes, photorealistic"
    )

# Qwen Image Edit 2511（stable-diffusion.cpp GGUF）推荐：cfg_scale 低一些更稳
if MODEL_ID == "qwen_image_edit_2511_gguf":
    STEPS = 20
    GUIDANCE = 2.5
    STRENGTH = 0.75
    # Qwen：仅保留匿名化（去除非匿名化分支）
    # DET_SIZE 建议 640（对小脸更稳）；这里不强制覆盖，沿用顶部 DET_SIZE 配置
    STRENGTH = max(STRENGTH, 0.85)
    PROMPT = (
        "Anonymize the person's face by replacing only the inner face area with a different realistic identity (not the same person). "
        "Keep the same head pose, gaze direction, facial expression, lighting, hairstyle and background. "
        "Keep ears, neck and clothing unchanged. Only modify the inner face area. "
        "Photo-realistic, natural skin texture, sharp eyes, high detail. "
    )

# ---- 多卡分布式（不使用 CPU offload / slicing / 降 ROI）----
USE_MULTI_GPU = False                     # 有 4 张 4090 可设 True
MULTI_GPU_IDS = [0, 1]           # 多卡索引（按你的实际 GPU 号修改）
if USE_MULTI_GPU:
    # diffusers/accelerate 对不同 pipeline 支持的 device_map 字符串不一致：
    DEVICE_MAP = "balanced"
else:
    DEVICE_MAP = None
MAX_MEMORY = {i: "22GiB" for i in MULTI_GPU_IDS} if USE_MULTI_GPU else None
# 禁用 CPU/Disk offload：不允许把任何权重放到 cpu/disk（放不下就直接报错，便于定位）
if MAX_MEMORY is not None:
    MAX_MEMORY["cpu"] = "0GiB"
    MAX_MEMORY["disk"] = "0GiB"

# ---- 系统级对比元信息 ----
SYSTEM_META = {
    "model_key": _model_spec.key,
    "model_id": _model_spec.model_id,
    "loader": _model_spec.loader,
    "pipeline": "roi_redraw_then_landmark_mask_composite",
    "core_priority": "A=mask_inpaint,C=roi_edit",
    "mask": f"landmark_convex(pad_ratio={PAD_RATIO}, blur={MASK_BLUR})_clipped_to_bbox",
    "identity_locking": "false",
}
if MODEL_ID == "qwen_image_edit_2511_gguf":
    SYSTEM_META["pipeline"] = "qwen_anonymize_roi_landmark_mask_composite"
    SYSTEM_META["anonymize_enabled"] = "true"

# ---- 批量 ----
SKIP_EXISTING = False                 # 若输出已存在则跳过

# =============================================================================
# 主逻辑
# =============================================================================
# model 变化后需要重新解析 spec 与路径
_model_spec = resolve_model_spec(MODEL_ID)
MODEL_TAG = _model_spec.key
SYSTEM_META["model_key"] = _model_spec.key
SYSTEM_META["model_id"] = _model_spec.model_id
SYSTEM_META["loader"] = _model_spec.loader

if _model_spec.pad_multiple is not None:
    PAD_MULTIPLE = _model_spec.pad_multiple
OUTPUT_DIR, REPORT_DIR, RUNTIME_JSON = _resolve_io_paths(INPUT_DIR, MODEL_TAG)

# 保证目录存在（按数据集创建 outputs/reports）
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print(f"[I/O] dataset={_dataset_tag(INPUT_DIR)} model={MODEL_TAG}")
print(f"[I/O] OUTPUT_DIR={OUTPUT_DIR}")
print(f"[I/O] REPORT_DIR={REPORT_DIR}")

if MODE == "single":
    run_identity_demo_once(
        cfg=DemoConfig(
            in_img=SINGLE_IN_IMG,
            out_img=SINGLE_OUT_IMG,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=SEED,
            steps=STEPS,
            guidance=GUIDANCE,
            strength=STRENGTH,
            pad_ratio=PAD_RATIO,
            mask_blur=MASK_BLUR,
            face_parsing_labels=FACE_PARSING_LABELS_HEAD,
            det_size=DET_SIZE,
            ctx_id=CTX_ID,
            model_id=MODEL_ID,
            device=DEVICE,
            pad_multiple=PAD_MULTIPLE,
            device_map=DEVICE_MAP,
            max_memory=MAX_MEMORY,
            cpu_offload=CPU_OFFLOAD,
        ),
    )
else:
    # 1) 批量处理 + 记录耗时
    print("[1/2] Batch processing...")
    run_identity_demo_batch(
        in_paths=INPUT_DIR,
        out_dir=OUTPUT_DIR,
        base_cfg=DemoConfig(
            in_img="",
            out_img="",
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=SEED,
            steps=STEPS,
            guidance=GUIDANCE,
            strength=STRENGTH,
            pad_ratio=PAD_RATIO,
            mask_blur=MASK_BLUR,
            face_parsing_labels=FACE_PARSING_LABELS_HEAD,
            det_size=DET_SIZE,
            ctx_id=CTX_ID,
            model_id=MODEL_ID,
            device=DEVICE,
            pad_multiple=PAD_MULTIPLE,
            device_map=DEVICE_MAP,
            max_memory=MAX_MEMORY,
            cpu_offload=CPU_OFFLOAD,
        ),
        skip_existing=SKIP_EXISTING,
        runtime_json_path=RUNTIME_JSON,
    )

    # 2) 评估 + 生成报告
    print("\n[2/2] Evaluating...")
    summary = evaluate_folder(EvalConfig(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        report_dir=REPORT_DIR,
        ctx_id=CTX_ID,
        det_size=DET_SIZE,
        pad_ratio=PAD_RATIO,
        mask_blur=MASK_BLUR,
        runtime_json=RUNTIME_JSON,
        generation_seed=SEED,
        comparison_mode="system",
        system_meta=SYSTEM_META,
    ))
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[OK] Report saved to", REPORT_DIR + "/")
