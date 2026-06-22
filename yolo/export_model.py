#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 YOLO 模型，例如导出 ONNX。

用法：
python yolo/export_model.py --weights output/yolo_runs/fish_activity_train/weights/best.pt --format onnx
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model")
    parser.add_argument("--weights", required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--format", default="onnx", help="导出格式，例如 onnx / torchscript")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸")
    parser.add_argument("--device", default=None, help="导出设备，例如 0 / cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(f"找不到模型权重: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先运行：pip install ultralytics") from exc

    model = YOLO(str(weights))
    kwargs = {"format": args.format, "imgsz": args.imgsz}
    if args.device is not None:
        kwargs["device"] = args.device

    print("[INFO] 开始导出模型")
    result = model.export(**kwargs)
    print("[DONE] 导出完成")
    print(f"[INFO] 导出结果: {result}")


if __name__ == "__main__":
    main()
