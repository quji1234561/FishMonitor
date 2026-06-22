#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的 YOLO 模型对图片、视频或文件夹进行预测。

用法：
python yolo/predict_yolo.py --weights output/yolo_runs/fish_activity_train/weights/best.pt --source output/enhanced/BT-001_yolo_enhanced/images/test
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with trained YOLO model")
    parser.add_argument("--weights", required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--source", required=True, help="输入图片、视频或文件夹路径")
    parser.add_argument("--imgsz", type=int, default=640, help="预测输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", default=None, help="预测设备，例如 0 / cpu")
    parser.add_argument("--project", default="output/yolo_runs", help="预测结果保存根目录")
    parser.add_argument("--name", default="fish_activity_predict", help="预测结果目录名")
    parser.add_argument("--save-txt", action="store_true", help="保存预测 txt 标签")
    parser.add_argument("--save-conf", action="store_true", help="保存预测置信度")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名预测目录")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)

    if not weights.exists():
        raise FileNotFoundError(f"找不到模型权重: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"找不到预测输入: {source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先运行：pip install ultralytics") from exc

    model = YOLO(str(weights))
    project_path = Path(args.project).resolve()
    if "/" in args.name or "\\" in args.name:
        raise ValueError(
            f"--name 只能是简单文件夹名，不能包含路径。当前 name={args.name}。"
            "正确示例：--project output/yolo_runs --name fish_activity_predict"
    )
    kwargs = {
        "source": str(source.resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": str(project_path),
        "name": args.name,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    print("[INFO] 开始 YOLO 推理")
    model.predict(**kwargs)
    print(f"[DONE] 推理完成，结果目录：{Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
