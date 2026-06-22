#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 YOLO 鱼类活跃状态检测模型。

用法：
python yolo/train_yolo.py --data output/enhanced/BT-001_yolo_enhanced/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for fish activity detection")
    parser.add_argument("--data", required=True, help="YOLO data.yaml 路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型，例如 yolov8n.pt / yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="batch size，显存不足时调小")
    parser.add_argument("--device", default=None, help="训练设备，例如 0 / cpu；默认自动选择")
    parser.add_argument("--project", default="output/yolo_runs", help="训练结果保存根目录")
    parser.add_argument("--name", default="fish_activity_train", help="本次训练名称")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心轮数")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名训练目录")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先运行：pip install ultralytics") from exc

    model = YOLO(args.model)
    project_path = Path(args.project).resolve()

    # 防止误把路径写进 name
    if "/" in args.name or "\\" in args.name:
        raise ValueError(
            f"--name 只能是简单名称，不能包含路径。当前 name={args.name}。"
            f"例如应该写 --project {project_path} / --name fish_activity_train"
        )
    
    kwargs = {
        "data": str(data_path.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    print("[INFO] 开始训练 YOLO 鱼类活跃状态检测模型")
    print(f"[INFO] data: {data_path}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] output: {Path(args.project) / args.name}")

    model.train(**kwargs)

    print("[DONE] 训练完成")
    print(f"[INFO] 最佳模型通常位于：{Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
