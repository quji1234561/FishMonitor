#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 MFT25 的 gt.txt 计算鱼的游动速度，并自动生成活跃状态标签。

输出：
1. auto_speed_labels_all_frames.csv：全量鱼框速度和状态
2. auto_speed_labels_sampled.csv：抽帧后的鱼框速度和状态
3. 可选 YOLO 数据集：images/train、labels/train、data.yaml 等

推荐用法：
python auto_label_fish_activity.py \
  --seq-dir data/MFT25/MFT25-train/BT-001 \
  --out-dir output/BT001_activity \
  --frame-step 10 \
  --make-yolo

状态类别：
0 = low_activity      低活跃
1 = normal_activity   正常活跃
2 = high_activity     高活跃
"""

import argparse
import configparser
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


STATE_NAMES = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def read_seqinfo(seq_dir: Path) -> Dict[str, object]:
    """读取 seqinfo.ini；如果不存在，就使用 MFT25 常见默认值。"""
    info_path = seq_dir / "seqinfo.ini"
    default = {
        "frame_rate": 25,
        "seq_length": None,
        "im_width": 1920,
        "im_height": 1080,
        "im_ext": ".jpg",
        "im_dir": "img1",
    }

    if not info_path.exists():
        print(f"[WARN] 未找到 {info_path}，使用默认参数：25fps, 1920x1080, .jpg")
        return default

    config = configparser.ConfigParser()
    config.read(info_path, encoding="utf-8")

    s = config["Sequence"]
    return {
        "frame_rate": int(s.get("frameRate", default["frame_rate"])),
        "seq_length": int(s.get("seqLength", 0)) or None,
        "im_width": int(s.get("imWidth", default["im_width"])),
        "im_height": int(s.get("imHeight", default["im_height"])),
        "im_ext": s.get("imExt", default["im_ext"]),
        "im_dir": s.get("imDir", default["im_dir"]),
    }


def load_mot_gt(gt_path: Path) -> pd.DataFrame:
    """读取 MOT 格式 gt.txt: frame,id,x,y,w,h,conf,*,*。"""
    if not gt_path.exists():
        raise FileNotFoundError(f"找不到 gt.txt: {gt_path}")

    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "mark1", "mark2"]
    df = pd.read_csv(gt_path, header=None, names=cols)

    # 只保留有效框
    df = df[(df["w"] > 0) & (df["h"] > 0)].copy()
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    # 中心点
    df["cx"] = df["x"] + df["w"] / 2.0
    df["cy"] = df["y"] + df["h"] / 2.0
    df["bbox_diag"] = np.sqrt(df["w"] ** 2 + df["h"] ** 2)
    df["bbox_size"] = np.sqrt(df["w"] * df["h"])

    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def compute_speed(
    df: pd.DataFrame,
    fps: int,
    max_gap: int = 5,
    smooth_window: int = 5,
    norm_mode: str = "bbox_diag",
) -> pd.DataFrame:
    """
    计算速度。
    speed_px_s: 像素/秒
    speed_norm_s: 归一化速度，默认约等于“每秒移动了几个鱼身对角线”
    """
    df = df.copy()

    g = df.groupby("track_id", group_keys=False)
    df["prev_frame"] = g["frame"].shift(1)
    df["prev_cx"] = g["cx"].shift(1)
    df["prev_cy"] = g["cy"].shift(1)

    df["frame_gap"] = df["frame"] - df["prev_frame"]
    df["dist_px"] = np.sqrt((df["cx"] - df["prev_cx"]) ** 2 + (df["cy"] - df["prev_cy"]) ** 2)

    # 只计算间隔合理的速度；gap 太大说明中间断了，不适合直接算速度
    valid = (df["frame_gap"] > 0) & (df["frame_gap"] <= max_gap)
    df["speed_px_s"] = np.nan
    df.loc[valid, "speed_px_s"] = df.loc[valid, "dist_px"] / df.loc[valid, "frame_gap"] * fps

    if norm_mode == "none":
        df["speed_norm_s"] = df["speed_px_s"]
    elif norm_mode == "bbox_size":
        denom = df["bbox_size"].replace(0, np.nan)
        df["speed_norm_s"] = df["speed_px_s"] / denom
    else:
        # 默认：用鱼框对角线归一化，减轻远近大小差异影响
        denom = df["bbox_diag"].replace(0, np.nan)
        df["speed_norm_s"] = df["speed_px_s"] / denom

    # 对每条鱼速度做平滑，减少一两帧抖动造成的误判
    def smooth_one_track(s: pd.Series) -> pd.Series:
        return s.rolling(window=smooth_window, min_periods=1).median()

    df["speed_smooth"] = df.groupby("track_id")["speed_norm_s"].transform(smooth_one_track)

    # 第一帧或断点可能没有速度，用同一条鱼附近速度填充；还没有就用全局中位数
    df["speed_smooth"] = df.groupby("track_id")["speed_smooth"].transform(lambda s: s.bfill().ffill())
    global_median = df["speed_smooth"].median()
    if np.isnan(global_median):
        global_median = 0.0
    df["speed_smooth"] = df["speed_smooth"].fillna(global_median)

    return df


def assign_state_by_percentile(
    df: pd.DataFrame,
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
    min_high_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float]:
    """
    按速度分位数自动切三档：
    <= low_thr: 低活跃
    low_thr ~ high_thr: 正常活跃
    >= high_thr: 高活跃
    """
    df = df.copy()
    speeds = df["speed_smooth"].dropna().values

    if len(speeds) == 0:
        low_thr = 0.0
        high_thr = 0.0
    else:
        low_thr = float(np.percentile(speeds, low_percentile))
        high_thr = float(np.percentile(speeds, high_percentile))

    if min_high_threshold is not None:
        high_thr = max(high_thr, min_high_threshold)

    def label(v: float) -> int:
        if v <= low_thr:
            return 0
        if v >= high_thr:
            return 2
        return 1

    df["state_id"] = df["speed_smooth"].apply(label).astype(int)
    df["state_name"] = df["state_id"].map(STATE_NAMES)

    return df, low_thr, high_thr


def filter_frame_step(df: pd.DataFrame, frame_step: int) -> pd.DataFrame:
    """抽帧：frame_step=10 表示每 10 帧取 1 帧。"""
    if frame_step <= 1:
        return df.copy()
    return df[df["frame"] % frame_step == 1].copy()


def write_csv(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "frame", "track_id", "x", "y", "w", "h",
        "cx", "cy", "speed_px_s", "speed_norm_s", "speed_smooth",
        "state_id", "state_name"
    ]
    df[keep_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已输出速度/状态表: {out_csv}")


def yolo_line(row: pd.Series, im_w: int, im_h: int) -> str:
    cls = int(row["state_id"])
    x_center = (float(row["x"]) + float(row["w"]) / 2.0) / im_w
    y_center = (float(row["y"]) + float(row["h"]) / 2.0) / im_h
    width = float(row["w"]) / im_w
    height = float(row["h"]) / im_h

    # 防止越界
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def split_frames_temporal(frames, train_ratio=0.7, val_ratio=0.2):
    """按时间顺序切分，避免相邻帧随机打乱导致结果虚高。"""
    frames = sorted(frames)
    n = len(frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = set(frames[:n_train])
    val = set(frames[n_train:n_train + n_val])
    test = set(frames[n_train + n_val:])
    return train, val, test


def make_yolo_dataset(
    df: pd.DataFrame,
    seq_dir: Path,
    out_dir: Path,
    im_w: int,
    im_h: int,
    im_ext: str,
    im_dir_name: str,
    copy_images: bool = True,
) -> None:
    yolo_root = out_dir / "yolo_dataset"
    img_src_dir = seq_dir / im_dir_name

    frames = sorted(df["frame"].unique().tolist())
    train_set, val_set, test_set = split_frames_temporal(frames)

    def subset_name(frame: int) -> str:
        if frame in train_set:
            return "train"
        if frame in val_set:
            return "val"
        return "test"

    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for frame, frame_df in df.groupby("frame"):
        split = subset_name(int(frame))
        stem = f"{int(frame):06d}"
        img_name = f"{stem}{im_ext}"
        label_name = f"{stem}.txt"

        label_path = yolo_root / "labels" / split / label_name
        with open(label_path, "w", encoding="utf-8") as f:
            for _, row in frame_df.iterrows():
                f.write(yolo_line(row, im_w, im_h) + "\n")

        src_img = img_src_dir / img_name
        dst_img = yolo_root / "images" / split / img_name

        if copy_images:
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                print(f"[WARN] 找不到图片: {src_img}")

    yaml_path = yolo_root / "data.yaml"
    yaml_text = f"""path: {yolo_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: low_activity
  1: normal_activity
  2: high_activity
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[OK] 已生成 YOLO 数据集: {yolo_root}")
    print(f"[OK] YOLO 配置文件: {yaml_path}")


def write_summary(
    df_all: pd.DataFrame,
    df_used: pd.DataFrame,
    out_dir: Path,
    low_thr: float,
    high_thr: float,
    norm_mode: str,
    fps: int,
    frame_step: int,
) -> None:
    summary_path = out_dir / "summary.txt"
    counts = df_used["state_name"].value_counts().to_dict()

    text = []
    text.append("鱼类活跃状态自动标注统计")
    text.append("=" * 40)
    text.append(f"fps: {fps}")
    text.append(f"frame_step: {frame_step}")
    text.append(f"speed norm mode: {norm_mode}")
    text.append(f"低活跃阈值 low_thr: {low_thr:.6f}")
    text.append(f"高活跃阈值 high_thr: {high_thr:.6f}")
    text.append("")
    text.append(f"原始标注框数量: {len(df_all)}")
    text.append(f"抽帧后标注框数量: {len(df_used)}")
    text.append("")
    text.append("状态数量：")
    for sid in [0, 1, 2]:
        name = STATE_NAMES[sid]
        text.append(f"- {sid} {name}: {counts.get(name, 0)}")
    text.append("")
    text.append("说明：")
    text.append("speed_px_s 表示像素/秒。")
    text.append("speed_norm_s 默认表示：每秒移动距离 / 当前鱼框对角线。")
    text.append("也就是说，数值越大，鱼相对自身大小移动得越快。")
    text.append("状态阈值使用分位数自动计算，适合时间紧的伪标签构建。")

    summary_path.write_text("\n".join(text), encoding="utf-8")
    print(f"[OK] 已输出统计摘要: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="根据 MFT25 gt.txt 自动计算速度并标注鱼类活跃状态")
    parser.add_argument("--seq-dir", required=True, help="序列目录，例如 data/MFT25/MFT25-train/BT-001")
    parser.add_argument("--gt-file", default=None, help="gt.txt 路径；默认使用 seq-dir/gt/gt.txt")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--frame-step", type=int, default=10, help="抽帧间隔，10 表示每 10 帧取 1 帧")
    parser.add_argument("--max-gap", type=int, default=5, help="计算速度允许的最大帧间隔")
    parser.add_argument("--smooth-window", type=int, default=5, help="速度平滑窗口，建议 5 或 10")
    parser.add_argument("--low-percentile", type=float, default=33.0, help="低活跃分位数阈值")
    parser.add_argument("--high-percentile", type=float, default=67.0, help="高活跃分位数阈值")
    parser.add_argument(
        "--norm-mode",
        choices=["bbox_diag", "bbox_size", "none"],
        default="bbox_diag",
        help="速度归一化方式：bbox_diag 推荐；none 表示直接用像素/秒"
    )
    parser.add_argument("--make-yolo", action="store_true", help="是否同时生成 YOLO 数据集")
    parser.add_argument("--no-copy-images", action="store_true", help="生成 YOLO 时不复制图片，只写 labels")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    gt_path = Path(args.gt_file) if args.gt_file else seq_dir / "gt" / "gt.txt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqinfo = read_seqinfo(seq_dir)
    fps = int(seqinfo["frame_rate"])
    im_w = int(seqinfo["im_width"])
    im_h = int(seqinfo["im_height"])
    im_ext = str(seqinfo["im_ext"])
    im_dir_name = str(seqinfo["im_dir"])

    print("[INFO] 读取 gt.txt ...")
    df = load_mot_gt(gt_path)

    print("[INFO] 计算速度 ...")
    df = compute_speed(
        df,
        fps=fps,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
        norm_mode=args.norm_mode,
    )

    print("[INFO] 按速度分位数划分低/中/高活跃 ...")
    df, low_thr, high_thr = assign_state_by_percentile(
        df,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )

    print(f"[INFO] 自动阈值: low <= {low_thr:.6f}, high >= {high_thr:.6f}")

    write_csv(df, out_dir / "auto_speed_labels_all_frames.csv")

    df_used = filter_frame_step(df, args.frame_step)
    write_csv(df_used, out_dir / "auto_speed_labels_sampled.csv")

    write_summary(
        df_all=df,
        df_used=df_used,
        out_dir=out_dir,
        low_thr=low_thr,
        high_thr=high_thr,
        norm_mode=args.norm_mode,
        fps=fps,
        frame_step=args.frame_step,
    )

    if args.make_yolo:
        print("[INFO] 生成 YOLO 数据集 ...")
        make_yolo_dataset(
            df=df_used,
            seq_dir=seq_dir,
            out_dir=out_dir,
            im_w=im_w,
            im_h=im_h,
            im_ext=im_ext,
            im_dir_name=im_dir_name,
            copy_images=not args.no_copy_images,
        )

    print("[DONE] 完成。")


if __name__ == "__main__":
    main()
