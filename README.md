# Face Swapping / Avatar Redraw

人像换脸与姿势保持重绘项目。在保持背景、姿势、表情不变的前提下，替换人脸身份区域。

## 环境

- Python 3.8+
- PyTorch
- diffusers
- insightface
- opencv-python
- Pillow

## 使用

- `run_m4_sdxl.py` - SDXL Inpaint 单模型入口
- `run_m4_kandinsky.py` - Kandinsky 5
- `run_m4_qwen.py` - Qwen Image Edit
- 其他 `run_m4_*.py` 对应不同模型

修改脚本中的 `INPUT_DIR`、`MODE` 等参数后运行。

## 数据与模型

- 输入图片放在 `posture-demo/` 或自定义目录
- 模型需单独下载到 `models/` 目录
