#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 val 或 test 数据上评估 YOLO 模型。

用法：
python yolo/evaluate_yolo.py --weights output/yolo_runs/fish_activity_train/weights/best.pt --data output/enhanced/BT-001_yolo_enhanced/data.yaml --split val
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--weights", required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--data", required=True, help="YOLO data.yaml 路径")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="评估 val 或 test")
    parser.add_argument("--imgsz", type=int, default=640, help="评估输入尺寸")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="评估置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", default=None, help="评估设备，例如 0 / cpu")
    parser.add_argument("--project", default="output/yolo_runs", help="评估结果保存根目录")
    parser.add_argument("--name", default="fish_activity_eval", help="评估结果目录名")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名评估目录")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    data = Path(args.data)

    if not weights.exists():
        raise FileNotFoundError(f"找不到模型权重: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先运行：pip install ultralytics") from exc

    model = YOLO(str(weights))
    project_path = Path(args.project).resolve()
    if "/" in args.name or "\\" in args.name:
        raise ValueError(
            f"--name 只能是简单文件夹名，不能包含路径。当前 name={args.name}。"
            "正确示例：--project output/yolo_runs --name fish_activity_eval"
        )
    kwargs = {
        "data": str(data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou, 
        "project": str(project_path),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    print("[INFO] 开始评估 YOLO 模型")
    metrics = model.val(**kwargs)
    print(f"[DONE] 评估完成，结果目录：{Path(args.project) / args.name}")

    try:
        print(f"[METRIC] mAP50-95: {metrics.box.map:.4f}")
        print(f"[METRIC] mAP50:    {metrics.box.map50:.4f}")
        print(f"[METRIC] mAP75:    {metrics.box.map75:.4f}")
    except Exception:
        print("[WARN] 未能直接打印 mAP 指标，请查看评估结果目录。")


if __name__ == "__main__":
    main()
