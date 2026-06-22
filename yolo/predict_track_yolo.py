#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 YOLO + ByteTrack/BoT-SORT 进行逐帧预测，并输出带 track_id 的 CSV。

推荐放置位置：
FishMonitor/yolo/predict_track_yolo.py

示例：
python yolo/predict_track_yolo.py \
  --weights output/yolo_runs/fish_activity_train/weights/best.pt \
  --source output/enhanced/BT-001_yolo_enhanced/images/test \
  --out-dir output/temporal/BT-001 \
  --save-video \
  --fps 10

输出：
output/temporal/BT-001/raw_predictions.csv
output/temporal/BT-001/raw_track_video.mp4  # 可选
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

STATE_SHORT = {
    "low_activity": "low",
    "normal_activity": "normal",
    "high_activity": "high",
}

STATE_COLOR = {
    0: (80, 160, 255),
    1: (80, 220, 80),
    2: (60, 80, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO tracking prediction to CSV")
    parser.add_argument("--weights", required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--source", required=True, help="图片文件夹、单张图片或视频路径")
    parser.add_argument("--out-dir", default="output/temporal/predict", help="输出目录")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置：bytetrack.yaml 或 botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", default=None, help="设备，例如 0 或 cpu")
    parser.add_argument("--save-video", action="store_true", help="是否保存带原始预测状态的视频")
    parser.add_argument("--fps", type=float, default=10.0, help="输出视频帧率")
    parser.add_argument("--resize", type=float, default=1.0, help="输出视频缩放比例，例如 0.5")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，调试用")
    return parser.parse_args()


def collect_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    if source.is_dir():
        imgs = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        return sorted(imgs, key=lambda p: p.name)
    return []


def frame_id_from_path(path: Path, index: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return index


def resize_if_needed(img, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def draw_box(img, row):
    state_id = int(row["state_id"])
    color = STATE_COLOR.get(state_id, (255, 255, 255))
    x1, y1, x2, y2 = [int(round(float(row[k]))) for k in ["x1", "y1", "x2", "y2"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    state_name = str(row["state_name"])
    state_short = STATE_SHORT.get(state_name, state_name)
    label = f"ID:{row['track_id']} {state_short} {float(row['conf']):.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(label, font, 0.55, 1)
    y_text = max(y1, th + 8)
    cv2.rectangle(img, (x1, y_text - th - base - 4), (x1 + tw + 6, y_text + base + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, y_text - 3), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def result_to_rows(result, frame_id: int, source_path: str, names: dict) -> List[dict]:
    rows = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        cls_id = int(clss[i])
        track_id = int(ids[i]) if ids is not None else -1
        rows.append({
            "frame": frame_id,
            "source_path": source_path,
            "track_id": track_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x": x1,
            "y": y1,
            "w": w,
            "h": h,
            "cx": x1 + w / 2.0,
            "cy": y1 + h / 2.0,
            "conf": float(confs[i]),
            "state_id": cls_id,
            "state_name": str(names.get(cls_id, cls_id)),
        })
    return rows


def open_video_writer(out_path: Path, first_img, fps: float, resize: float):
    first = resize_if_needed(first_img, resize)
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")
    return writer


def process_image_sequence(model, source: Path, args, out_csv: Path):
    images = collect_images(source)
    if not images:
        raise RuntimeError(f"没有找到图片: {source}")
    if args.max_frames:
        images = images[: args.max_frames]

    names = model.names
    all_rows = []
    writer = None
    video_path = Path(args.out_dir) / "raw_track_video.mp4"

    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}")
            continue

        frame_id = frame_id_from_path(img_path, idx)
        results = model.track(
            img,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        rows = result_to_rows(result, frame_id, str(img_path), names)
        all_rows.extend(rows)

        if args.save_video:
            if writer is None:
                writer = open_video_writer(video_path, img, args.fps, args.resize)
            draw = img.copy()
            for r in rows:
                draw_box(draw, r)
            cv2.putText(draw, f"frame: {frame_id}", (20, draw.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            writer.write(resize_if_needed(draw, args.resize))

    if writer is not None:
        writer.release()
        print(f"[OK] 原始跟踪视频: {video_path}")

    write_rows(out_csv, all_rows)


def process_video(model, source: Path, args, out_csv: Path):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    names = model.names
    all_rows = []
    writer = None
    video_path = Path(args.out_dir) / "raw_track_video.mp4"

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if args.max_frames and idx > args.max_frames:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        rows = result_to_rows(results[0], idx, str(source), names)
        all_rows.extend(rows)

        if args.save_video:
            if writer is None:
                writer = open_video_writer(video_path, frame, args.fps, args.resize)
            draw = frame.copy()
            for r in rows:
                draw_box(draw, r)
            cv2.putText(draw, f"frame: {idx}", (20, draw.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            writer.write(resize_if_needed(draw, args.resize))

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[OK] 原始跟踪视频: {video_path}")

    write_rows(out_csv, all_rows)


def write_rows(out_csv: Path, rows: List[dict]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame", "source_path", "track_id", "x1", "y1", "x2", "y2", "x", "y", "w", "h",
        "cx", "cy", "conf", "state_id", "state_name"
    ]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] 原始预测 CSV: {out_csv}")
    print(f"[OK] 预测框数量: {len(rows)}")


def main():
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "raw_predictions.csv"

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"找不到权重文件: {args.weights}")
    if not source.exists():
        raise FileNotFoundError(f"找不到输入: {source}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics，请先运行：pip install ultralytics") from exc

    model = YOLO(args.weights)

    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        process_video(model, source, args, out_csv)
    else:
        process_image_sequence(model, source, args, out_csv)


if __name__ == "__main__":
    main()
