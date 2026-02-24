#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 1 Demo (Python-callable):
- 单张/多图人脸 inpainting：检测最大人脸 -> 柔边 mask -> SDXL Inpainting 重绘
- 输出尺寸严格等于输入尺寸（padding 到 8 的倍数 + crop 回原尺寸）
- 批量处理时模型只加载一次，支持记录每张图耗时供 eval 使用

作者: didi
创建时间: 250202
修改时间: 250202
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

import torch
from diffusers import AutoPipelineForInpainting
from insightface.app import FaceAnalysis


# =========================
# 1) 配置：用 dataclass 管理参数（比 argparse 更适合脚本/IDE 直接跑）
# =========================
@dataclass
class DemoConfig:
    # I/O
    in_img: str
    out_img: str

    # 文本条件（你可以按任务逐步调整）
    prompt: str = (
        "a realistic photo of a person, natural skin texture, different face as the input image,"
        "consistent lighting, sharp eyes, high detail face, beautiful face, different style"
    )
    negative_prompt: str = (
        "deformed, bad anatomy, extra face, extra eyes, blurry, low quality, same face as the input image, "
        "cartoon, painting, uncanny, plastic, watermark, text, different color"
    )

    # 扩散推理超参数
    seed: int = 42
    steps: int = 30
    guidance: float = 7.5
    strength: float = 0.85

    # mask 参数（Milestone 1 重点调这两个就能救很多图）
    pad_ratio: float = 0.35
    mask_blur: int = 12

    # 检测参数
    det_size: int = 640 
    ctx_id: int = 0            # InsightFace: 0=GPU0, -1=CPU

    # 模型与设备
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    device: str = "cuda"       # "cuda" / "cuda:0" / "cpu"

    # 尺寸对齐：扩散模型通常要求宽高是 8 的倍数
    pad_multiple: int = 8

    # 批量处理
    skip_existing: bool = True   # 若输出已存在则跳过，节省时间


# =========================
# 2) 图像基础工具：读图/格式转换
# =========================
def load_image_rgb(path: str) -> Image.Image:
    """读取图片并强制转换为 RGB，避免 RGBA/灰度导致下游异常。"""
    return Image.open(path).convert("RGB")


def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray（InsightFace 使用）"""
    arr = np.array(img)  # HWC RGB
    return arr[:, :, ::-1].copy()  # HWC BGR


# =========================
# 3) 人脸检测：取最大脸 bbox
# =========================
def detect_largest_face_bbox(
    face_app: FaceAnalysis,
    img_pil: Image.Image,
    det_size: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    返回最大人脸 bbox=(x1,y1,x2,y2)。
    Milestone 1 只做单脸 demo，所以选最大脸最稳。
    """
    face_app.prepare(ctx_id=face_app.ctx_id, det_size=(det_size, det_size))
    faces = face_app.get(pil_to_bgr_np(img_pil))
    if len(faces) == 0:
        return None

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces.sort(key=area, reverse=True)
    x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
    return (x1, y1, x2, y2)


# =========================
# 4) Mask：bbox -> 柔边椭圆
# =========================
def make_soft_ellipse_mask(
    size_wh: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    pad_ratio: float,
    blur: int
) -> Image.Image:
    """
    输出灰度 mask（L 模式）：
    - 白色区域表示允许重绘
    - 黑色区域表示禁止改动
    """
    W, H = size_wh
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    X1 = max(0, x1 - pad_w)
    Y1 = max(0, y1 - pad_h)
    X2 = min(W - 1, x2 + pad_w)
    Y2 = min(H - 1, y2 + pad_h)

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse([X1, Y1, X2, Y2], fill=255)

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask


# =========================
# 5) 尺寸对齐：padding 到 8 的倍数 + crop 回原尺寸（硬保证尺寸一致）
# =========================
def _compute_symmetric_pad(w: int, h: int, multiple: int) -> Tuple[int, int, int, int]:
    """
    计算对称 padding：
    返回 (left, top, right, bottom)
    """
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return left, top, right, bottom


def pad_image_and_mask_to_multiple(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    multiple: int
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    对 image 和 mask 同步 padding，避免扩散模型内部 resize 改变输出尺寸。
    """
    w, h = image_pil.size
    left, top, right, bottom = _compute_symmetric_pad(w, h, multiple)

    img = np.array(image_pil)   # HWC RGB
    msk = np.array(mask_pil)    # HW

    # image 用反射 padding：边缘更自然
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    # mask 用 0 padding：新增区域不允许重绘
    msk_pad = cv2.copyMakeBorder(msk, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

    return Image.fromarray(img_pad), Image.fromarray(msk_pad).convert("L"), (left, top, right, bottom)


def crop_back_to_original(
    out_pil: Image.Image,
    pad: Tuple[int, int, int, int],
    orig_size_wh: Tuple[int, int]
) -> Image.Image:
    """把 padding 后的输出裁回原始尺寸，保证 output.size == input.size"""
    left, top, _, _ = pad
    orig_w, orig_h = orig_size_wh
    out_crop = out_pil.crop((left, top, left + orig_w, top + orig_h))
    # 双保险
    if out_crop.size != (orig_w, orig_h):
        out_crop = out_crop.resize((orig_w, orig_h), resample=Image.LANCZOS)
    return out_crop


# =========================
# 6) Inpainting：构建 pipe + 执行一次推理
# =========================
def build_inpaint_pipe(model_id: str, device: str):
    """
    构建 diffusers inpaint pipeline。
    建议：如果你之后做 batch，把 pipe 缓存起来复用，速度会快很多。
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)

    if device.startswith("cuda"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe, device


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
) -> Image.Image:
    """
    调用 pipeline 执行 inpainting。
    注意：这里显式传入 height/width，减少内部可能的尺寸重整。
    """
    w, h = image_pil.size
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
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
    return result.images[0]


# =========================
# 7) 核心入口：单张 / 批量（支持预加载模型复用，节省时间）
# =========================
def run_demo(
    cfg: DemoConfig,
    pipe=None,
    device: Optional[str] = None,
    face_app: Optional[FaceAnalysis] = None,
) -> Image.Image:
    """
    运行一次 demo（单张图片）：
    - 返回输出 PIL 图像，同时保存到 cfg.out_img
    - 若传入 pipe/face_app，则复用，避免重复加载模型（批量时关键优化）
    """
    # 1) 读图
    img = load_image_rgb(cfg.in_img)
    orig_w, orig_h = img.size

    # 2) 检测 bbox（复用 face_app 或新建）
    if face_app is None:
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.ctx_id = cfg.ctx_id
    bbox = detect_largest_face_bbox(face_app, img, det_size=cfg.det_size)
    if bbox is None:
        raise RuntimeError(
            f"No face detected in {cfg.in_img}. Try clearer image / larger det_size."
        )

    # 3) mask（原尺寸）
    mask = make_soft_ellipse_mask(
        size_wh=(orig_w, orig_h),
        bbox=bbox,
        pad_ratio=cfg.pad_ratio,
        blur=cfg.mask_blur,
    )

    # 4) padding 对齐（避免输出尺寸变化）
    img_pad, mask_pad, pad = pad_image_and_mask_to_multiple(img, mask, multiple=cfg.pad_multiple)

    # 5) inpaint（复用 pipe 或新建）
    if pipe is None:
        pipe, device = build_inpaint_pipe(cfg.model_id, cfg.device)
    elif device is None:
        device = str(next(pipe.unet.parameters()).device)

    out_pad = run_inpaint_once(
        pipe=pipe,
        image_pil=img_pad,
        mask_pil=mask_pad,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        seed=cfg.seed,
        steps=cfg.steps,
        guidance=cfg.guidance,
        strength=cfg.strength,
        device=device,
    )

    # 6) crop 回原尺寸（硬保证一致）
    out = crop_back_to_original(out_pad, pad=pad, orig_size_wh=(orig_w, orig_h))

    # 7) 保存 + 日志
    out.save(cfg.out_img)
    print(f"[OK] Saved: {cfg.out_img}")
    print(f"[INFO] input_size=({orig_w},{orig_h}), output_size={out.size}")
    print(f"[INFO] bbox={bbox}, pad={pad}")
    print(f"[INFO] seed={cfg.seed}, steps={cfg.steps}, guidance={cfg.guidance}, strength={cfg.strength}")

    return out


def _collect_image_pairs(
    in_paths: Union[str, Path, List[Union[str, Path]]],
    out_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[Path, Path]]:
    """
    收集 (输入路径, 输出路径) 对。
    - in_paths: 单文件、文件列表、或目录路径
    - out_dir: 输出目录，若为 None 则输出与输入同目录，后缀 _out
    """
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


def run_demo_batch(
    in_paths: Union[str, Path, List[Union[str, Path]]],
    out_dir: Optional[Union[str, Path]] = None,
    base_cfg: Optional[DemoConfig] = None,
    skip_existing: bool = True,
    runtime_json_path: Optional[str] = None,
) -> List[Image.Image]:
    """
    批量处理多张图片，模型只加载一次，显著节省运行时间。

    Args:
        in_paths: 输入路径（单文件、文件列表、或目录）
        out_dir: 输出目录，None 时输出到输入同目录，文件名加 _out
        base_cfg: 基础配置，None 时用默认 DemoConfig
        skip_existing: 若输出文件已存在则跳过
        runtime_json_path: 若指定，将每张图的推理耗时保存到此 JSON，供 eval 使用

    Returns:
        成功处理的 PIL 图像列表
    """
    pairs = _collect_image_pairs(in_paths, out_dir)
    if not pairs:
        print("[WARN] No valid input images found.")
        return []

    cfg = base_cfg or DemoConfig(in_img="", out_img="")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 只加载一次模型（核心优化）
    print("[INFO] Loading FaceAnalysis...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.ctx_id = cfg.ctx_id
    print("[INFO] Loading Inpainting pipeline...")
    pipe, device = build_inpaint_pipe(cfg.model_id, cfg.device)

    results: List[Image.Image] = []
    runtime_map: dict = {}  # {filename: {"seconds": float}}
    total = len(pairs)
    for i, (in_p, out_p) in enumerate(pairs):
        if skip_existing and out_p.exists():
            print(f"[SKIP] ({i+1}/{total}) {out_p.name} already exists")
            continue

        print(f"\n[PROC] ({i+1}/{total}) {in_p.name} -> {out_p}")
        try:
            t0_img = time.perf_counter()
            task_cfg = DemoConfig(
                in_img=str(in_p),
                out_img=str(out_p),
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                seed=cfg.seed,
                steps=cfg.steps,
                guidance=cfg.guidance,
                strength=cfg.strength,
                pad_ratio=cfg.pad_ratio,
                mask_blur=cfg.mask_blur,
                det_size=cfg.det_size,
                ctx_id=cfg.ctx_id,
                model_id=cfg.model_id,
                device=cfg.device,
                pad_multiple=cfg.pad_multiple,
            )
            out_img = run_demo(task_cfg, pipe=pipe, device=device, face_app=face_app)
            elapsed = time.perf_counter() - t0_img
            results.append(out_img)
            if runtime_json_path:
                runtime_map[in_p.name] = {"seconds": round(elapsed, 3)}
        except Exception as e:
            print(f"[ERR] {in_p.name}: {e}")

    if runtime_json_path and runtime_map:
        Path(runtime_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(runtime_json_path, "w", encoding="utf-8") as f:
            json.dump(runtime_map, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Runtime saved to {runtime_json_path}")

    print(f"\n[DONE] Processed {len(results)}/{total} images.")
    return results


# =========================
# 8) 命令行接口：支持单张与批量
# =========================
def _cli_main():
    import argparse
    p = argparse.ArgumentParser(description="人脸 Inpainting：单张或批量处理")
    p.add_argument("--in_img", type=str, help="单张输入图片路径")
    p.add_argument("--out_img", type=str, help="单张输出路径（与 --in_img 配合）")
    p.add_argument("--in_dir", type=str, default="SHHQ-1.0_samples", help="批量：输入目录")
    p.add_argument("--out_dir", type=str, default="outputs", help="批量：输出目录")
    p.add_argument("--no_skip", action="store_true", help="批量时不跳过已存在的输出")
    p.add_argument("--runtime_json", type=str, default="reports/eval/runtime.json", help="批量时保存耗时到此 JSON")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.in_img and args.out_img:
        # 单张模式
        cfg = DemoConfig(
            in_img=args.in_img,
            out_img=args.out_img,
            steps=args.steps,
            strength=args.strength,
            seed=args.seed,
        )
        run_demo(cfg)
    elif args.in_dir:
        run_demo_batch(
            in_paths=args.in_dir,
            out_dir=args.out_dir,
            base_cfg=DemoConfig(steps=args.steps, strength=args.strength, seed=args.seed),
            skip_existing=not args.no_skip,
            runtime_json_path=args.runtime_json,
        )
    else:
        p.print_help()
        print("\n示例:")
        print("  单张: python demo_m1.py --in_img test.jpg --out_img out.jpg")
        print("  批量: python demo_m1.py --in_dir SHHQ-1.0_samples/ --out_dir outputs/")


if __name__ == "__main__":
    _cli_main()
