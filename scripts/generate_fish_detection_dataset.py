#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single-class YOLO fish-detection dataset from MFT25 MOT labels.

This dataset is for the first stage of the improved pipeline:
YOLO detects fish boxes only; temporal analysis assigns activity states later.
"""

import argparse
import configparser
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


def read_seqinfo(seq_dir: Path) -> Dict[str, object]:
    info_path = seq_dir / "seqinfo.ini"
    default = {
        "im_width": 1920,
        "im_height": 1080,
        "im_ext": ".jpg",
        "im_dir": "img1",
    }
    if not info_path.exists():
        return default

    cfg = configparser.ConfigParser()
    cfg.read(info_path, encoding="utf-8")
    s = cfg["Sequence"]
    return {
        "im_width": int(s.get("imWidth", default["im_width"])),
        "im_height": int(s.get("imHeight", default["im_height"])),
        "im_ext": s.get("imExt", default["im_ext"]),
        "im_dir": s.get("imDir", default["im_dir"]),
    }


def load_mot_gt(gt_path: Path) -> pd.DataFrame:
    if not gt_path.exists():
        raise FileNotFoundError(f"找不到 gt.txt: {gt_path}")
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "mark1", "mark2"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    df = df[(df["w"] > 0) & (df["h"] > 0)].copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df.sort_values(["frame", "track_id"]).reset_index(drop=True)


def split_frames(frames: Iterable[int], train_ratio: float, val_ratio: float) -> Tuple[set, set, set]:
    frames = sorted(int(f) for f in frames)
    n = len(frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = set(frames[:n_train])
    val = set(frames[n_train : n_train + n_val])
    test = set(frames[n_train + n_val :])
    return train, val, test


def yolo_fish_line(row: pd.Series, im_w: int, im_h: int) -> str:
    x_center = (float(row["x"]) + float(row["w"]) / 2.0) / im_w
    y_center = (float(row["y"]) + float(row["h"]) / 2.0) / im_h
    width = float(row["w"]) / im_w
    height = float(row["h"]) / im_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def output_stem(seq_dir: Path, frame: int, multi_sequence: bool) -> str:
    if multi_sequence:
        return f"{seq_dir.name}_{frame:06d}"
    return f"{frame:06d}"


def build_one_sequence(
    seq_dir: Path,
    out_dir: Path,
    frame_step: int,
    train_ratio: float,
    val_ratio: float,
    copy_images: bool,
    multi_sequence: bool,
) -> Dict[str, int]:
    info = read_seqinfo(seq_dir)
    im_w = int(info["im_width"])
    im_h = int(info["im_height"])
    im_ext = str(info["im_ext"])
    im_dir_name = str(info["im_dir"])
    img_src_dir = seq_dir / im_dir_name

    df = load_mot_gt(seq_dir / "gt" / "gt.txt")
    if frame_step > 1:
        df = df[df["frame"] % frame_step == 1].copy()

    train_set, val_set, test_set = split_frames(df["frame"].unique(), train_ratio, val_ratio)

    def split_name(frame: int) -> str:
        if frame in train_set:
            return "train"
        if frame in val_set:
            return "val"
        return "test"

    counts = {"frames": 0, "boxes": 0, "train": 0, "val": 0, "test": 0, "missing_images": 0}
    for frame, frame_df in df.groupby("frame"):
        split = split_name(int(frame))
        stem = output_stem(seq_dir, int(frame), multi_sequence)
        label_path = out_dir / "labels" / split / f"{stem}.txt"
        image_dst = out_dir / "images" / split / f"{stem}{im_ext}"
        image_src = img_src_dir / f"{int(frame):06d}{im_ext}"

        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_dst.parent.mkdir(parents=True, exist_ok=True)

        with label_path.open("w", encoding="utf-8") as f:
            for _, row in frame_df.iterrows():
                f.write(yolo_fish_line(row, im_w, im_h) + "\n")

        if copy_images:
            if image_src.exists():
                shutil.copy2(image_src, image_dst)
            else:
                counts["missing_images"] += 1

        counts["frames"] += 1
        counts["boxes"] += len(frame_df)
        counts[split] += 1

    return counts


def write_data_yaml(out_dir: Path) -> None:
    text = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: fish
"""
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate single-class fish YOLO dataset from MFT25")
    parser.add_argument(
        "--seq-dir",
        action="append",
        required=True,
        help="MFT25 sequence directory, e.g. data/MFT25/MFT25-train/BT-001. Can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame; 1 means all frames")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels and data.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    seq_dirs = [Path(p) for p in args.seq_dir]
    multi_sequence = len(seq_dirs) > 1
    total = {"frames": 0, "boxes": 0, "train": 0, "val": 0, "test": 0, "missing_images": 0}

    for seq_dir in seq_dirs:
        counts = build_one_sequence(
            seq_dir=seq_dir,
            out_dir=out_dir,
            frame_step=args.frame_step,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            copy_images=not args.no_copy_images,
            multi_sequence=multi_sequence,
        )
        for key, value in counts.items():
            total[key] += value
        print(f"[OK] {seq_dir.name}: {counts}")

    write_data_yaml(out_dir)
    summary = (
        "单类 fish YOLO 数据集生成摘要\n"
        "============================\n"
        f"序列数量: {len(seq_dirs)}\n"
        f"frame_step: {args.frame_step}\n"
        f"帧数: {total['frames']}\n"
        f"鱼框数: {total['boxes']}\n"
        f"train/val/test 帧数: {total['train']} / {total['val']} / {total['test']}\n"
        f"缺失图片数: {total['missing_images']}\n"
        f"输出目录: {out_dir}\n"
        "类别: 0 fish\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
