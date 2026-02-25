#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_m4 单模型入口：LongCat Image Edit Turbo
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from demo_m4 import run_pipeline


# 运行模式：
# - single: 仅处理 SINGLE_IN_IMG，结果另存到 SINGLE_OUT_IMG
# - batch : 批量处理 INPUT_DIR 目录下全部图片（并在 demo_m4 内自动走 batch->eval）
MODE = "single"  # "single" | "batch"
INPUT_DIR = "posture-demo"
SINGLE_IN_IMG = "test1.jpg"
SINGLE_OUT_IMG = "out_longcat_image_edit_turbo.jpg"

# 模型与设备（按单模型脚本固定）
MODEL_KEY = "longcat_image_edit_turbo"
CTX_ID = 2
DEVICE = "cuda:2"

SEED = 42
STEPS = 35
GUIDANCE = 4.5
STRENGTH = 0.70

PAD_RATIO = 0.35
MASK_BLUR = 8
ROI_PAD_RATIO = 0.45
ROI_MAX_SIDE = 1024

SKIP_EXISTING = False
NO_EVAL = False


def main() -> None:
    run_pipeline(
        mode=MODE,
        models=[MODEL_KEY],
        input_dir=INPUT_DIR,
        single_in_img=SINGLE_IN_IMG,
        single_out_img=SINGLE_OUT_IMG,
        ctx_id=CTX_ID,
        device=DEVICE,
        seed=SEED,
        steps=STEPS,
        guidance=GUIDANCE,
        strength=STRENGTH,
        pad_ratio=PAD_RATIO,
        mask_blur=MASK_BLUR,
        roi_pad_ratio=ROI_PAD_RATIO,
        roi_max_side=ROI_MAX_SIDE,
        skip_existing=SKIP_EXISTING,
        no_eval=NO_EVAL,
    )


if __name__ == "__main__":
    main()
