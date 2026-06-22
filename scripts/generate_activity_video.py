#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据自动速度标注结果生成可视化视频：显示鱼框、track_id 和活跃状态。

推荐放置位置：
FishMonitor/scripts/generate_activity_video.py

推荐用法一：使用原始 MFT25 序列图片 + 自动标注 CSV
python scripts/generate_activity_video.py \
  --seq-dir data/MFT25/MFT25-train/BT-001 \
  --activity-csv output/activity/BT-001/auto_speed_labels_sampled.csv \
  --out output/video/BT-001_activity_labels.mp4 \
  --fps 10

推荐用法二：使用 YOLO 数据集里的图片 + 自动标注 CSV
python scripts/generate_activity_video.py \
  --yolo-root output/activity/BT-001/yolo_dataset \
  --activity-csv output/activity/BT-001/auto_speed_labels_sampled.csv \
  --out output/video/BT-001_activity_labels.mp4 \
  --fps 10

注意：
1. YOLO 的 labels/*.txt 默认只保存 class_id 和 bbox，不保存 track_id。
2. 如果要在视频里显示 ID，必须读取 auto_speed_labels_sampled.csv。
3. 该 CSV 是 auto_label_fish_activity.py 生成的，里面包含 frame、track_id、bbox、state_name。
"""

import argparse
import configparser
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import pandas as pd


STATE_COLOR = {
    0: (80, 160, 255),     # low_activity
    1: (80, 220, 80),      # normal_activity
    2: (60, 80, 255),      # high_activity
}

STATE_CN = {
    "low_activity": "low",
    "normal_activity": "normal",
    "high_activity": "high",
}


def read_seqinfo(seq_dir: Optional[Path]) -> Dict[str, object]:
    """读取 seqinfo.ini；没有就用默认值。"""
    default = {
        "frame_rate": 25,
        "im_width": 1920,
        "im_height": 1080,
        "im_ext": ".jpg",
        "im_dir": "img1",
    }

    if seq_dir is None:
        return default

    info_path = seq_dir / "seqinfo.ini"
    if not info_path.exists():
        print(f"[WARN] 未找到 {info_path}，使用默认参数")
        return default

    cfg = configparser.ConfigParser()
    cfg.read(info_path, encoding="utf-8")
    s = cfg["Sequence"]

    return {
        "frame_rate": int(s.get("frameRate", default["frame_rate"])),
        "im_width": int(s.get("imWidth", default["im_width"])),
        "im_height": int(s.get("imHeight", default["im_height"])),
        "im_ext": s.get("imExt", default["im_ext"]),
        "im_dir": s.get("imDir", default["im_dir"]),
    }


def load_activity_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 activity csv: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"frame", "track_id", "x", "y", "w", "h", "state_id", "state_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["state_id"] = df["state_id"].astype(int)

    for col in ["x", "y", "w", "h"]:
        df[col] = df[col].astype(float)

    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)


def collect_yolo_images(yolo_root: Path, split: str = "all") -> Dict[int, Path]:
    """
    收集 yolo_dataset/images 下的图片。
    文件名需要是 000001.jpg 这种帧号格式。
    """
    if not yolo_root.exists():
        raise FileNotFoundError(f"找不到 YOLO 数据集目录: {yolo_root}")

    image_map: Dict[int, Path] = {}

    splits = ["train", "val", "test"] if split == "all" else [split]
    for sp in splits:
        img_dir = yolo_root / "images" / sp
        if not img_dir.exists():
            print(f"[WARN] 找不到图片目录: {img_dir}")
            continue

        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            try:
                frame = int(p.stem)
            except ValueError:
                continue
            image_map[frame] = p

    return image_map


def get_image_path(
    frame: int,
    seq_dir: Optional[Path],
    seqinfo: Dict[str, object],
    yolo_image_map: Optional[Dict[int, Path]],
) -> Optional[Path]:
    """优先使用 yolo_image_map，否则使用 seq_dir/img1。"""
    if yolo_image_map is not None and frame in yolo_image_map:
        return yolo_image_map[frame]

    if seq_dir is not None:
        im_dir = str(seqinfo["im_dir"])
        im_ext = str(seqinfo["im_ext"])
        return seq_dir / im_dir / f"{frame:06d}{im_ext}"

    return None


def put_label(
    img,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
):
    """画带背景的文字，避免字看不清。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    y_text = max(y, th + 4)
    x_text = max(x, 0)

    cv2.rectangle(
        img,
        (x_text, y_text - th - baseline - 4),
        (x_text + tw + 6, y_text + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        img,
        text,
        (x_text + 3, y_text - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_annotations(
    img,
    frame_df: pd.DataFrame,
    show_speed: bool = False,
    line_thickness: int = 2,
):
    for _, row in frame_df.iterrows():
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        w = int(round(row["w"]))
        h = int(round(row["h"]))

        state_id = int(row["state_id"])
        state_name = str(row["state_name"])
        short_state = STATE_CN.get(state_name, state_name)

        color = STATE_COLOR.get(state_id, (255, 255, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)

        label = f"ID:{int(row['track_id'])} {short_state}"
        if show_speed and "speed_smooth" in frame_df.columns:
            try:
                label += f" v:{float(row['speed_smooth']):.2f}"
            except Exception:
                pass

        put_label(img, label, x, y - 4, color)


def draw_legend(img):
    """左上角画状态说明。"""
    lines = [
        ("low_activity", 0),
        ("normal_activity", 1),
        ("high_activity", 2),
    ]

    x0, y0 = 20, 30
    for i, (name, sid) in enumerate(lines):
        color = STATE_COLOR[sid]
        y = y0 + i * 28
        cv2.rectangle(img, (x0, y - 16), (x0 + 18, y + 2), color, -1)
        cv2.putText(
            img,
            f"{sid}: {name}",
            (x0 + 28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def resize_if_needed(img, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def generate_video(args):
    seq_dir = Path(args.seq_dir) if args.seq_dir else None
    yolo_root = Path(args.yolo_root) if args.yolo_root else None
    csv_path = Path(args.activity_csv)
    out_path = Path(args.out)

    seqinfo = read_seqinfo(seq_dir)
    df = load_activity_csv(csv_path)

    yolo_image_map = None
    if yolo_root is not None:
        yolo_image_map = collect_yolo_images(yolo_root, split=args.split)
        if not yolo_image_map:
            raise RuntimeError(f"没有在 {yolo_root}/images/{args.split} 中找到图片")

        # 只保留 YOLO 数据集里实际存在的帧
        df = df[df["frame"].isin(set(yolo_image_map.keys()))].copy()

    frames = sorted(df["frame"].unique().tolist())
    if args.start_frame is not None:
        frames = [f for f in frames if f >= args.start_frame]
    if args.end_frame is not None:
        frames = [f for f in frames if f <= args.end_frame]
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    if not frames:
        raise RuntimeError("没有可用帧，请检查 activity_csv、seq_dir/yolo_root 和 split 参数。")

    first_img_path = get_image_path(frames[0], seq_dir, seqinfo, yolo_image_map)
    if first_img_path is None or not first_img_path.exists():
        raise FileNotFoundError(f"找不到第一帧图片: {first_img_path}")

    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        raise RuntimeError(f"无法读取图片: {first_img_path}")

    first_img = resize_if_needed(first_img, args.resize)
    h, w = first_img.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")

    grouped = dict(tuple(df.groupby("frame")))

    written = 0
    for frame in frames:
        img_path = get_image_path(frame, seq_dir, seqinfo, yolo_image_map)
        if img_path is None or not img_path.exists():
            print(f"[WARN] 跳过缺失图片: frame={frame}, path={img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图片: {img_path}")
            continue

        frame_df = grouped.get(frame)
        if frame_df is not None:
            draw_annotations(
                img,
                frame_df,
                show_speed=args.show_speed,
                line_thickness=args.thickness,
            )

        if args.legend:
            draw_legend(img)

        cv2.putText(
            img,
            f"frame: {frame}",
            (20, img.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        img = resize_if_needed(img, args.resize)
        writer.write(img)
        written += 1

    writer.release()

    print(f"[OK] 已生成视频: {out_path}")
    print(f"[OK] 写入帧数: {written}")
    print(f"[INFO] 状态显示格式: ID:<track_id> <low/normal/high>")


def main():
    parser = argparse.ArgumentParser(
        description="根据 auto_speed_labels_sampled.csv 生成带 ID 和活跃状态标签的视频"
    )

    src = parser.add_argument_group("输入")
    src.add_argument("--seq-dir", default=None, help="MFT25 序列目录，例如 data/MFT25/MFT25-train/BT-001")
    src.add_argument("--yolo-root", default=None, help="YOLO 数据集目录，例如 output/activity/BT-001/yolo_dataset")
    src.add_argument("--split", default="all", choices=["all", "train", "val", "test"], help="使用 YOLO 数据集的哪个 split")
    src.add_argument("--activity-csv", required=True, help="auto_speed_labels_sampled.csv 路径")

    out = parser.add_argument_group("输出")
    out.add_argument("--out", required=True, help="输出视频路径，例如 output/video/BT-001_activity_labels.mp4")
    out.add_argument("--fps", type=float, default=10.0, help="输出视频帧率，默认 10，方便观察")
    out.add_argument("--codec", default="mp4v", help="视频编码，默认 mp4v；也可试 avc1")

    view = parser.add_argument_group("显示控制")
    view.add_argument("--resize", type=float, default=1.0, help="缩放比例，例如 0.5 表示输出 960x540")
    view.add_argument("--thickness", type=int, default=2, help="框线粗细")
    view.add_argument("--show-speed", action="store_true", help="标签中同时显示速度")
    view.add_argument("--no-legend", dest="legend", action="store_false", help="不显示左上角图例")
    view.set_defaults(legend=True)

    filt = parser.add_argument_group("帧范围")
    filt.add_argument("--start-frame", type=int, default=None, help="起始帧")
    filt.add_argument("--end-frame", type=int, default=None, help="结束帧")
    filt.add_argument("--max-frames", type=int, default=None, help="最多写入多少帧")

    args = parser.parse_args()

    if not args.seq_dir and not args.yolo_root:
        parser.error("--seq-dir 和 --yolo-root 至少提供一个。")

    generate_video(args)


if __name__ == "__main__":
    main()
