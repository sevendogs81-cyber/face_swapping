#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件作用: 人脸匿名化（虚拟化）结果评估与报告生成（系统级对比）
# 编码: utf-8
"""
Eval module for face anonymization / face editing（对接 demo_m3 + run.py）.

评价目标：
1) 稳定性：输出图能否检测到人脸（success rate）
2) 匿名化强度：输入脸 vs 输出脸的 identity embedding 相似度（越低越匿名化）
3) 结构保真：关键点一致性（越低越好）
4) 背景保留：mask 外 SSIM（越高越好）
5) 伪影 proxy：检测失败/关键点缺失/脸 crop 模糊度等（越低越好）

特点：
- 默认评估每个输入人脸（按 IoU 匹配输出人脸）
- mask 外背景保留：优先使用人脸 landmarks 生成精细 mask（近似 face parsing），缺失时回退 bbox 椭圆
- 输出：report_dir 下 per_image.csv、metrics.json、report.md

使用方式：
    from eval import EvalConfig, evaluate_folder
    cfg = EvalConfig(input_dir="SHHQ-1.0", output_dir="outputs", report_dir="reports/eval")
    summary = evaluate_folder(cfg)

依赖：
pip install numpy opencv-python pillow insightface onnxruntime-gpu
"""

from __future__ import annotations

import os
import json
import csv
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

from insightface.app import FaceAnalysis


# =========================
# Config
# =========================
@dataclass
class EvalConfig:
    input_dir: str
    output_dir: str
    report_dir: str

    # file matching（与 demo_m1 _collect_image_pairs 一致）
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # face detection（det_size=512 可避免大图 broadcasting 报错）
    det_size: int = 512
    ctx_id: int = 0  # 0=GPU0, -1=CPU

    # mask（需与 run.py 的 pad_ratio、mask_blur 一致，否则 bg_ssim 不准确）
    pad_ratio: float = 0.40
    mask_blur: int = 16
    mask_threshold: int = 128  # binarize blurred mask (0..255)

    # background SSIM compute
    ssim_gauss_ksize: int = 11
    ssim_gauss_sigma: float = 1.5

    # optional runtime log (if you later save per-image time)
    # format: {"filename.jpg": {"seconds": 1.23, ...}, ...}
    runtime_json: Optional[str] = None

    # batch 生成时使用的 seed，便于复现（与 demo_m1 DemoConfig.seed 一致）
    generation_seed: Optional[int] = None

    # 对比模式：system 表示允许不同 pipeline；model 表示严格模型接口一致
    comparison_mode: str = "system"

    # 系统级对比元信息（例如：model_id/pipeline/备注）
    system_meta: Dict[str, str] = field(default_factory=dict)

    # if output size differs from input (shouldn't), how to handle:
    # "resize": resize output to input size for metric computation (and record flag)
    # "skip": skip that sample
    size_mismatch_policy: str = "resize"


@dataclass
class PerImageRecord:
    filename: str

    # sizes
    in_w: int
    in_h: int
    out_w: int
    out_h: int
    size_mismatch: int  # 0/1

    # face detect status
    in_face_ok: int
    out_face_ok: int

    # identity leakage: 输出 vs 输入（越低越匿名；越高越像输入）
    id_cosine_in_out: float   # cos(in_emb, out_emb)

    # structure preservation
    lm_norm_err: float  # normalized landmark error (lower => better)

    # background preservation
    bg_ssim: float  # SSIM on background (mask outside). higher => better

    # artifact proxies
    out_face_blur: float  # variance of Laplacian on output face crop. higher => sharper

    # optional runtime
    seconds: float  # -1 if unavailable


# =========================
# Utils: IO & matching
# =========================
def _is_image_file(name: str, exts: Tuple[str, ...]) -> bool:
    return os.path.splitext(name.lower())[1] in exts


def collect_pairs(input_dir: str, output_dir: str, exts: Tuple[str, ...]) -> List[Tuple[str, str, str]]:
    """
    Pair input/output images by identical filename.
    Returns list of (filename, input_path, output_path).
    """
    in_files = {f for f in os.listdir(input_dir) if _is_image_file(f, exts)}
    out_files = {f for f in os.listdir(output_dir) if _is_image_file(f, exts)}
    common = sorted(list(in_files & out_files))

    pairs = []
    for fn in common:
        pairs.append((fn, os.path.join(input_dir, fn), os.path.join(output_dir, fn)))
    return pairs


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1].copy()  # BGR


def ensure_same_size(
    in_img: Image.Image,
    out_img: Image.Image,
    policy: str
) -> Tuple[Optional[Image.Image], int]:
    """
    Ensure out_img matches in_img size for metric computation.
    Returns (possibly adjusted_out_img or None, size_mismatch_flag).
    """
    if in_img.size == out_img.size:
        return out_img, 0

    if policy == "skip":
        return None, 1

    # default: resize for computation
    out_resized = out_img.resize(in_img.size, resample=Image.LANCZOS)
    return out_resized, 1


# =========================
# Face analysis helpers
# =========================
def build_face_app(ctx_id: int, det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "landmark_2d_106"])
    app.prepare(ctx_id=ctx_id, det_size=(int(det_size), int(det_size)))
    app.ctx_id = ctx_id
    app._demo_det_size = int(det_size)  # type: ignore[attr-defined]
    return app


def detect_largest_face(app: FaceAnalysis, img_pil: Image.Image, det_size: int):
    """
    Returns face object or None. Picks the largest face.
    """
    faces = app.get(pil_to_bgr_np(img_pil))
    if not faces:
        return None

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces.sort(key=area, reverse=True)
    return faces[0]


def detect_all_faces(app: FaceAnalysis, img_pil: Image.Image, det_size: int) -> List[Any]:
    """返回所有人脸（按面积从大到小排序）"""
    faces = app.get(pil_to_bgr_np(img_pil))
    if not faces:
        return []

    def area(f) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces.sort(key=area, reverse=True)
    return list(faces)


def _bbox_iou(b1, b2) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, float(b1[2] - b1[0])) * max(0.0, float(b1[3] - b1[1]))
    a2 = max(0.0, float(b2[2] - b2[0])) * max(0.0, float(b2[3] - b2[1]))
    return float(inter / (a1 + a2 - inter + 1e-12))


def _match_faces_by_iou(in_faces: List[Any], out_faces: List[Any], min_iou: float = 0.10) -> List[Optional[Any]]:
    """按 IoU 贪心匹配输出人脸，返回与 in_faces 等长的匹配结果"""
    if not in_faces:
        return []
    if not out_faces:
        return [None] * len(in_faces)

    pairs = []
    for i, f_in in enumerate(in_faces):
        for j, f_out in enumerate(out_faces):
            iou = _bbox_iou(f_in.bbox, f_out.bbox)
            if iou > 0.0:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_out = set()
    matched = [None] * len(in_faces)
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if matched[i] is None and j not in used_out:
            matched[i] = out_faces[j]
            used_out.add(j)
    return matched


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
    """用 landmarks 的凸包生成更精细的 mask（近似 face parsing）"""
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def get_face_embedding(face) -> Optional[np.ndarray]:
    """
    InsightFace face object typically provides 'normed_embedding' (best) or 'embedding'.
    """
    if face is None:
        return None
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return np.array(face.normed_embedding)
    if hasattr(face, "embedding") and face.embedding is not None:
        emb = np.array(face.embedding)
        # normalize just in case
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
    return None


def get_face_kps5(face) -> Optional[np.ndarray]:
    """
    Return 5-point landmarks as (5,2) float32 in image coordinates.
    """
    if face is None:
        return None
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.array(face.kps, dtype=np.float32)
        if kps.shape == (5, 2):
            return kps
    return None


def normalized_landmark_error(kps_in: np.ndarray, kps_out: np.ndarray) -> float:
    """
    L2 average landmark distance normalized by inter-ocular distance.
    (kps: 5x2, usually [left_eye, right_eye, nose, left_mouth, right_mouth])
    """
    if kps_in is None or kps_out is None:
        return float("nan")

    # inter-ocular distance from input (more stable)
    iod = float(np.linalg.norm(kps_in[0] - kps_in[1])) + 1e-12
    err = np.linalg.norm(kps_in - kps_out, axis=1).mean()
    return float(err / iod)


def face_crop_from_bbox(img_pil: Image.Image, bbox, pad: float = 0.15) -> Image.Image:
    """
    Crop face region with small padding.
    """
    w, h = img_pil.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = (x2 - x1), (y2 - y1)
    px = bw * pad
    py = bh * pad
    X1 = int(max(0, x1 - px))
    Y1 = int(max(0, y1 - py))
    X2 = int(min(w - 1, x2 + px))
    Y2 = int(min(h - 1, y2 + py))
    return img_pil.crop((X1, Y1, X2, Y2))


def laplacian_blur_score(img_pil: Image.Image) -> float:
    """
    Blur proxy: variance of Laplacian on grayscale.
    Higher => sharper; lower => blurrier.
    """
    arr = np.array(img_pil.convert("L"))
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    return float(lap.var())


# =========================
# Mask & Background SSIM
# =========================
def make_soft_ellipse_mask(size_wh, bbox, pad_ratio: float, blur: int) -> Image.Image:
    """
    Same idea as demo_m1: ellipse mask around face bbox.
    White(255)=face region; black(0)=background.
    """
    W, H = size_wh
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    X1 = max(0, x1 - pad_w)
    Y1 = max(0, y1 - pad_h)
    X2 = min(W - 1, x2 + pad_w)
    Y2 = min(H - 1, y2 + pad_h)

    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([X1, Y1, X2, Y2], fill=255)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask


def make_face_mask_from_face(
    size_wh: Tuple[int, int],
    face,
    pad_ratio: float,
    blur: int,
) -> Image.Image:
    """
    优先用 landmarks 生成精细 mask（近似 face parsing），
    若 landmarks 不可用则回退到 bbox 椭圆 mask。
    """
    if face is None:
        return Image.new("L", size_wh, 0)
    pts = _get_face_landmarks(face)
    if pts is not None:
        return _make_landmark_mask(size_wh=size_wh, landmarks=pts, pad_ratio=pad_ratio, blur=blur)
    bbox = tuple(int(v) for v in face.bbox)
    return make_soft_ellipse_mask(size_wh, bbox, pad_ratio=pad_ratio, blur=blur)


def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float64)


def ssim_map_gray(
    x: np.ndarray,
    y: np.ndarray,
    ksize: int = 11,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Compute SSIM map for two grayscale images x,y in [0,255].
    Returns SSIM per-pixel map (float64).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    kernel = _gaussian_kernel(ksize, sigma)
    mu_x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mu_y = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.filter2D(x * x, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_x2
    sigma_y2 = cv2.filter2D(y * y, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_y2
    sigma_xy = cv2.filter2D(x * y, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_xy

    # constants for 8-bit images
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / (den + 1e-12)


def background_ssim(
    in_img: Image.Image,
    out_img: Image.Image,
    face_mask: Image.Image,
    threshold: int,
    ksize: int,
    sigma: float
) -> float:
    """
    Compute SSIM on the background region (mask outside).
    Steps:
      - binarize face_mask (face=1)
      - background = ~face
      - compute SSIM map on grayscale
      - average SSIM over background pixels
    """
    in_g = np.array(in_img.convert("L"))
    out_g = np.array(out_img.convert("L"))
    m = np.array(face_mask)  # 0..255
    face_bin = (m > threshold)
    bg = ~face_bin

    ssim_m = ssim_map_gray(in_g, out_g, ksize=ksize, sigma=sigma)

    if bg.sum() == 0:
        return float("nan")
    return float(ssim_m[bg].mean())


# =========================
# Runtime log
# =========================
def load_runtime_json(path: Optional[str]) -> Dict[str, float]:
    """
    Expect: {"a.jpg": {"seconds": 1.23}, "b.jpg": {"seconds": 0.98}, ...}
    Return: {"a.jpg": 1.23, ...}
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if isinstance(v, dict) and "seconds" in v:
            out[k] = float(v["seconds"])
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


# =========================
# Core evaluation
# =========================
def evaluate_folder(cfg: EvalConfig) -> Dict[str, float]:
    os.makedirs(cfg.report_dir, exist_ok=True)

    pairs = collect_pairs(cfg.input_dir, cfg.output_dir, cfg.exts)
    if not pairs:
        raise RuntimeError(f"No paired images found. Check dirs: {cfg.input_dir} & {cfg.output_dir}")

    runtime_map = load_runtime_json(cfg.runtime_json)

    face_app = build_face_app(cfg.ctx_id, cfg.det_size)

    records: List[PerImageRecord] = []
    t0 = time.time()

    for fn, in_path, out_path in pairs:
        in_img = load_rgb(in_path)
        out_img_raw = load_rgb(out_path)

        out_img, mismatch_flag = ensure_same_size(in_img, out_img_raw, cfg.size_mismatch_policy)
        if out_img is None:
            # skipped due to mismatch
            rec = PerImageRecord(
                filename=fn,
                in_w=in_img.size[0], in_h=in_img.size[1],
                out_w=out_img_raw.size[0], out_h=out_img_raw.size[1],
                size_mismatch=1,
                in_face_ok=0, out_face_ok=0,
                id_cosine_in_out=float("nan"),
                lm_norm_err=float("nan"),
                bg_ssim=float("nan"),
                out_face_blur=float("nan"),
                seconds=runtime_map.get(fn, -1.0),
            )
            records.append(rec)
            continue

        # detect faces (multi-face)
        in_faces = detect_all_faces(face_app, in_img, cfg.det_size)
        out_faces = detect_all_faces(face_app, out_img, cfg.det_size)

        in_face_ok = 1 if in_faces else 0
        if not in_faces:
            out_face_ok = 0
            id_cos_in_out = float("nan")
            lm_err = float("nan")
            bg_ssim_score = float("nan")
            blur_score = float("nan")
        else:
            matched = _match_faces_by_iou(in_faces, out_faces, min_iou=0.10)
            matched_count = sum(1 for m in matched if m is not None)
            out_face_ok = float(matched_count / max(len(in_faces), 1))

            id_list: List[float] = []
            lm_list: List[float] = []
            bg_list: List[float] = []
            blur_list: List[float] = []

            for in_face, out_face in zip(in_faces, matched):
                # identity: 输出 vs 输入
                emb_in = get_face_embedding(in_face)
                emb_out = get_face_embedding(out_face)
                if emb_in is not None and emb_out is not None:
                    id_list.append(cosine_sim(emb_in, emb_out))

                # landmark consistency (5-point)
                kps_in = get_face_kps5(in_face)
                kps_out = get_face_kps5(out_face)
                if kps_in is not None and kps_out is not None:
                    lm_list.append(normalized_landmark_error(kps_in, kps_out))

                # background preservation: mask from INPUT face (most meaningful)
                face_mask = make_face_mask_from_face(
                    size_wh=in_img.size,
                    face=in_face,
                    pad_ratio=cfg.pad_ratio,
                    blur=cfg.mask_blur,
                )
                bg_list.append(background_ssim(
                    in_img=in_img,
                    out_img=out_img,
                    face_mask=face_mask,
                    threshold=cfg.mask_threshold,
                    ksize=cfg.ssim_gauss_ksize,
                    sigma=cfg.ssim_gauss_sigma
                ))

                # blur proxy on output face crop
                if out_face is not None:
                    crop = face_crop_from_bbox(out_img, out_face.bbox, pad=0.15)
                    blur_list.append(laplacian_blur_score(crop))

            def _nanmean_local(xs: List[float]) -> float:
                xs = [x for x in xs if x is not None and not np.isnan(x)]
                return float(np.mean(xs)) if xs else float("nan")

            id_cos_in_out = _nanmean_local(id_list)
            lm_err = _nanmean_local(lm_list)
            bg_ssim_score = _nanmean_local(bg_list)
            blur_score = _nanmean_local(blur_list)

        rec = PerImageRecord(
            filename=fn,
            in_w=in_img.size[0], in_h=in_img.size[1],
            out_w=out_img_raw.size[0], out_h=out_img_raw.size[1],
            size_mismatch=mismatch_flag,
            in_face_ok=in_face_ok,
            out_face_ok=out_face_ok,
            id_cosine_in_out=id_cos_in_out,
            lm_norm_err=lm_err,
            bg_ssim=bg_ssim_score,
            out_face_blur=blur_score,
            seconds=runtime_map.get(fn, -1.0),
        )
        records.append(rec)

    wall = time.time() - t0

    # aggregate
    total = len(records)
    out_detect = sum(r.out_face_ok for r in records if not np.isnan(r.out_face_ok))
    success_rate = out_detect / total if total > 0 else 0.0

    def _nanmean(xs: List[float]) -> float:
        xs = [x for x in xs if x is not None and not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    id_cos_in_out_mean = _nanmean([r.id_cosine_in_out for r in records])
    id_cos_in_out_p90 = float(np.nanpercentile([r.id_cosine_in_out for r in records if not np.isnan(r.id_cosine_in_out)], 90)) \
        if any(not np.isnan(r.id_cosine_in_out) for r in records) else float("nan")

    lm_mean = _nanmean([r.lm_norm_err for r in records])
    bg_ssim_mean = _nanmean([r.bg_ssim for r in records])
    blur_mean = _nanmean([r.out_face_blur for r in records])

    mismatch_rate = sum(r.size_mismatch for r in records) / total if total > 0 else 0.0

    # artifact proxy: output face not detected => artifact
    artifact_rate = 1.0 - success_rate

    # runtime (optional)
    secs_list = [r.seconds for r in records if r.seconds >= 0]
    avg_seconds = float(np.mean(secs_list)) if secs_list else float("nan")

    summary = {
        "num_pairs": total,
        "generation_seed": cfg.generation_seed,  # batch 生成 seed，便于复现
        "comparison_mode": cfg.comparison_mode,
        "success_rate_out_face_detect": success_rate,
        "artifact_rate_proxy": artifact_rate,
        "size_mismatch_rate": mismatch_rate,

        # identity: 输出 vs 输入（匿名化强度 proxy）
        "identity_cosine_in_out_mean": id_cos_in_out_mean,
        "identity_cosine_in_out_p90": id_cos_in_out_p90,

        # structure & background
        "landmark_norm_err_mean": lm_mean,
        "background_ssim_mean": bg_ssim_mean,

        # quality proxy
        "out_face_blur_varLaplacian_mean": blur_mean,

        # performance
        "wall_time_seconds": wall,
        "avg_infer_seconds_from_log": avg_seconds,
    }

    # write outputs
    _write_per_image_csv(os.path.join(cfg.report_dir, "per_image.csv"), records)
    _write_json(os.path.join(cfg.report_dir, "metrics.json"), {"config": asdict(cfg), "summary": summary})
    _write_report_md(os.path.join(cfg.report_dir, "report.md"), cfg, summary)

    return summary


def _write_per_image_csv(path: str, records: List[PerImageRecord]) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def _json_safe(obj):
    """递归将 NaN/Inf 转为 None，保证 JSON 可解析"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)


def _write_report_md(path: str, cfg: EvalConfig, summary: Dict[str, float]) -> None:
    lines = []
    lines.append("# Face Anonymization Evaluation Report\n")
    lines.append("## Config\n")
    lines.append("```json\n")
    lines.append(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    lines.append("\n```\n")

    lines.append("## System Meta\n")
    if cfg.system_meta:
        for k, v in cfg.system_meta.items():
            lines.append(f"- **{k}**: {v}\n")
    else:
        lines.append("_None_\n")
    lines.append("\n")

    lines.append("## Summary Metrics\n")
    if cfg.comparison_mode == "system":
        lines.append("> NOTE: 系统级对比：不同模型可使用不同 pipeline/策略，报告中仅做效果对比。\n\n")
    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NaN"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    for k, v in summary.items():
        lines.append(f"- **{k}**: {fmt(v)}\n")

    lines.append("\n## How to interpret\n")
    lines.append("| 指标 | 说明 | 期望方向 |\n")
    lines.append("|------|------|----------|\n")
    lines.append("| **num_pairs** | 参与评估的输入-输出图像对数 | - |\n")
    lines.append("| **generation_seed** | batch 生成时使用的随机种子，便于复现 | - |\n")
    lines.append("| **comparison_mode** | 对比模式（system/model） | - |\n")
    lines.append("| **success_rate_out_face_detect** | 输出图能检测到人脸的比例（多人图为平均召回） | 越高越稳 |\n")
    lines.append("| **artifact_rate_proxy** | 输出脸检测失败的比例（=1-success_rate） | 越低越好 |\n")
    lines.append("| **size_mismatch_rate** | 输出尺寸与输入不一致的比例 | 越低越好 |\n")
    lines.append("| **identity_cosine_in_out_mean** | 输出脸与输入脸 embedding 余弦相似度均值（衡量是否像输入） | 越低越好 |\n")
    lines.append("| **identity_cosine_in_out_p90** | 上述 in_out 相似度的 90 分位数 | 视任务而定 |\n")
    lines.append("| **landmark_norm_err_mean** | 5 点关键点归一化误差均值（除以瞳距），衡量姿态/表情保持 | 越低越好 |\n")
    lines.append("| **background_ssim_mean** | mask 外背景区域 SSIM 均值，衡量背景是否被改动 | 越高越好 |\n")
    lines.append("| **out_face_blur_varLaplacian_mean** | 输出脸 crop 的 Laplacian 方差，模糊度 proxy | 越高越清晰 |\n")
    lines.append("| **wall_time_seconds** | 评估脚本总耗时（秒） | - |\n")
    lines.append("| **avg_infer_seconds_from_log** | 单张图平均推理耗时（秒），需 batch 时指定 runtime_json 才有值 | - |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# =========================
# CLI 与默认入口
# =========================
def _cli_main():
    import argparse
    p = argparse.ArgumentParser(description="评估人脸匿名化（虚拟化）效果")
    p.add_argument("--input_dir", type=str, default="SHHQ-1.0", help="输入目录（原始图）")
    p.add_argument("--output_dir", type=str, default="outputs", help="输出目录（生成结果）")
    p.add_argument("--report_dir", type=str, default="reports/eval", help="报告输出目录")
    p.add_argument("--ctx_id", type=int, default=0, help="InsightFace GPU id，-1=CPU")
    p.add_argument("--det_size", type=int, default=512)
    p.add_argument("--runtime_json", type=str, default="reports/eval/runtime.json", help="batch 耗时 JSON 路径")
    p.add_argument("--seed", type=int, default=42, help="batch 生成 seed")
    args = p.parse_args()

    cfg = EvalConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        ctx_id=args.ctx_id,
        det_size=args.det_size,
        runtime_json=args.runtime_json,
        generation_seed=args.seed,
    )
    s = evaluate_folder(cfg)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\n[OK] Report saved to {args.report_dir}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _cli_main()
    else:
        # 无参数时：用默认目录评估（对接 run.py 的 batch 输出）
        cfg = EvalConfig(
            input_dir="SHHQ-1.0",
            output_dir="outputs",
            report_dir="reports/eval",
            ctx_id=0,
            det_size=512,
            pad_ratio=0.40,
            mask_blur=16,
            runtime_json="reports/eval/runtime.json",
        )
        summary = evaluate_folder(cfg)
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        print("\n[OK] Report saved to reports/eval/")
