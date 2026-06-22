#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a multi-feature activity-state dataset from MFT25 gt.txt."""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_analysis.motion_features import (  # noqa: E402
    DEFAULT_MODEL_FEATURES,
    DEFAULT_SCORE_WEIGHTS,
    add_motion_features,
    apply_temporal_split,
    assign_labels_from_score,
    assign_persistent_labels_from_score,
    compute_activity_score,
)


STATE_COLOR = {
    0: (80, 160, 255),
    1: (80, 220, 80),
    2: (60, 80, 255),
}
STATE_SHORT = {
    "low_activity": "low",
    "normal_activity": "normal",
    "high_activity": "high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-feature fish activity-state dataset")
    parser.add_argument("--seq-dir", required=True, help="MFT25 sequence dir, e.g. data/MFT25/MFT25-train/PF-001")
    parser.add_argument("--gt-file", default=None, help="gt.txt path. Default: seq-dir/gt/gt.txt")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--frame-step", type=int, default=5, help="Use every Nth frame for the saved dataset")
    parser.add_argument("--smooth-window", type=int, default=5, help="Speed smoothing window")
    parser.add_argument("--feature-window", type=int, default=5, help="Motion feature rolling window")
    parser.add_argument(
        "--label-mode",
        choices=["persistent", "score_percentile"],
        default="persistent",
        help="persistent requires low/high motion to last inside a time window",
    )
    parser.add_argument("--low-percentile", type=float, default=20.0, help="Low state percentile on activity score")
    parser.add_argument("--high-percentile", type=float, default=80.0, help="High state percentile on activity score")
    parser.add_argument("--label-window-sec", type=float, default=3.0, help="Persistence window length in seconds")
    parser.add_argument("--low-required-ratio", type=float, default=0.8, help="Low label requires this low-motion ratio in the window")
    parser.add_argument("--high-required-ratio", type=float, default=0.6, help="High label requires this high-motion ratio in the window")
    parser.add_argument("--high-event-percentile", type=float, default=92.0, help="Sharp-turn event threshold percentile")
    parser.add_argument("--high-event-hold-sec", type=float, default=1.0, help="Hold high label after a sharp-turn event")
    parser.add_argument(
        "--min-low-label-confidence",
        type=float,
        default=0.0,
        help="Convert low labels with label_confidence below this value back to normal.",
    )
    parser.add_argument(
        "--min-high-label-confidence",
        type=float,
        default=0.0,
        help="Convert high labels with label_confidence below this value back to normal.",
    )
    parser.add_argument(
        "--max-non-normal-ratio",
        type=float,
        default=None,
        help="Keep only the most confident low/high labels when non-normal ratio exceeds this value.",
    )
    parser.add_argument(
        "--no-backfill-persistent-states",
        dest="backfill_persistent_states",
        action="store_false",
        default=True,
        help="Disable backfilling low/high persistent labels to the confirmation window",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Temporal train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Temporal val split ratio")
    parser.add_argument(
        "--feature-after-sampling",
        dest="feature_after_sampling",
        action="store_true",
        default=True,
        help="Filter frame-step first, then compute temporal features. This should match inference on sampled frames.",
    )
    parser.add_argument(
        "--feature-before-sampling",
        dest="feature_after_sampling",
        action="store_false",
        help="Old behavior: compute temporal features on all frames, then save sampled rows.",
    )
    parser.add_argument("--make-video", action="store_true", help="Also generate a label quality video")
    parser.add_argument("--video-out", default=None, help="Optional output path for label quality video")
    parser.add_argument("--video-max-frames", type=int, default=500, help="Maximum frames in quality video")
    parser.add_argument("--video-fps", type=float, default=25.0, help="Quality video fps")
    parser.add_argument("--resize", type=float, default=1.0, help="Quality video resize ratio")
    return parser.parse_args()


def read_seqinfo(seq_dir: Path) -> dict:
    info_path = seq_dir / "seqinfo.ini"
    default = {
        "frame_rate": 25,
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
        "frame_rate": int(s.get("frameRate", default["frame_rate"])),
        "im_width": int(s.get("imWidth", default["im_width"])),
        "im_height": int(s.get("imHeight", default["im_height"])),
        "im_ext": s.get("imExt", default["im_ext"]),
        "im_dir": s.get("imDir", default["im_dir"]),
    }


def load_mot_gt(gt_path: Path, sequence: str) -> pd.DataFrame:
    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "mark1", "mark2"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    df = df[(df["w"] > 0) & (df["h"] > 0)].copy()
    df["sequence"] = sequence
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["conf"] = 1.0
    df["x1"] = df["x"]
    df["y1"] = df["y"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    df["cx"] = df["x"] + df["w"] / 2.0
    df["cy"] = df["y"] + df["h"] / 2.0
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def filter_frame_step(df: pd.DataFrame, frame_step: int) -> pd.DataFrame:
    if frame_step <= 1:
        return df.copy()
    first_frame = int(df["frame"].min())
    return df[((df["frame"] - first_frame) % frame_step) == 0].copy()


def save_dataset_files(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "sequence", "split", "frame", "track_id", "image_path",
        "x", "y", "w", "h", "x1", "y1", "x2", "y2", "cx", "cy", "conf",
        "bbox_area", "bbox_diag", "bbox_aspect", "bbox_area_norm",
        "same_frame_max_iou", "overlap_suppression", "overlap_reliability",
        "center_x_norm", "center_y_norm", "edge_distance_norm",
        "prev_frame", "frame_gap", "dx", "dy", "dist_px",
        "speed_px_s", "speed_norm_s", "speed_smooth",
        "speed_mean_w5", "speed_std_w5", "speed_max_w5", "speed_p75_w5",
        "displacement_w5", "movement_ratio_w5",
        "speed_delta_abs", "accel_abs_smooth",
        "heading_angle", "turn_angle_abs", "turn_rate_w5", "direction_change_count_w5",
        "bbox_aspect_log", "bbox_aspect_change_abs", "bbox_aspect_change_w5",
        "bbox_size_change_abs", "bbox_size_change_w5",
        "edge_shape_change_norm", "edge_shape_change_w5",
        "bbox_shape_change_w5", "in_place_turn_score",
        "bbox_aspect_change_rate_s", "bbox_size_change_rate_s", "edge_shape_change_rate_s",
        "bbox_aspect_change_rate_w1s", "bbox_size_change_rate_w1s", "edge_shape_change_rate_w1s",
        "bbox_shape_change_rate_w1s",
        "bbox_diag_range_w1s", "bbox_aspect_range_w1s",
        "bbox_diag_reversal_flag", "bbox_aspect_reversal_flag", "bbox_reversal_count_w1s",
        "bbox_pulse_score_w1s",
        "sharp_turn_event_raw_score", "sharp_turn_event_score", "sharp_turn_event_w1s", "sharp_turn_event_w5",
        "fish_count_frame", "group_speed_mean", "group_speed_std", "group_high_motion_ratio",
        "nearest_neighbor_dist_norm", "crowding_index",
        "instant_low_flag", "instant_high_flag",
        "low_persistence_ratio", "high_persistence_ratio",
        "low_persistent_confirmed_flag", "high_persistent_confirmed_flag",
        "low_persistent_backfill_flag", "high_persistent_backfill_flag",
        "high_event_raw_flag", "high_event_hold_flag",
        "label_window_sec", "label_window_rows",
        "label_low_required_ratio", "label_high_required_ratio",
        "label_backfill_persistent_states", "label_confidence",
        "label_high_event_percentile", "label_high_event_score_threshold",
        "label_high_event_hold_sec", "label_high_event_hold_rows",
        "valid_motion", "is_track_start", "is_track_end", "track_len", "sample_weight",
        "state_score", "state_id", "state_name", "label_source", "label_version",
        "original_state_id", "original_state_name", "label_selection_confidence", "label_filter_reason",
    ]
    cols = [c for c in ordered_cols if c in df.columns]
    df[cols].to_csv(out_dir / "activity_state_all.csv", index=False, encoding="utf-8-sig")
    for split in ["train", "val", "test"]:
        df[df["split"] == split][cols].to_csv(
            out_dir / f"activity_state_{split}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def write_feature_config(out_dir: Path, args: argparse.Namespace, norm_stats: dict, low_thr: float, high_thr: float) -> None:
    config = {
        "label_version": "v9_event_below_low_overlap_suppressed" if args.label_mode == "persistent" else "v2_multifeature",
        "label_source": "auto_persistent_motion_rule" if args.label_mode == "persistent" else "auto_motion_rule",
        "model_features": DEFAULT_MODEL_FEATURES,
        "feature_weights": DEFAULT_SCORE_WEIGHTS,
        "normalization": norm_stats,
        "label_thresholds": {
            "low_score_threshold": low_thr,
            "high_score_threshold": high_thr,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
        },
        "feature_window": args.feature_window,
        "speed_smooth_window": args.smooth_window,
        "frame_step": args.frame_step,
        "feature_after_sampling": bool(args.feature_after_sampling),
        "label_mode": args.label_mode,
        "label_window_sec": args.label_window_sec,
        "low_required_ratio": args.low_required_ratio,
        "high_required_ratio": args.high_required_ratio,
        "backfill_persistent_states": bool(args.backfill_persistent_states),
        "high_event_percentile": args.high_event_percentile,
        "high_event_hold_sec": args.high_event_hold_sec,
        "min_low_label_confidence": args.min_low_label_confidence,
        "min_high_label_confidence": args.min_high_label_confidence,
        "max_non_normal_ratio": args.max_non_normal_ratio,
    }
    (out_dir / "feature_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_conservative_label_filter(
    df: pd.DataFrame,
    min_low_confidence: float = 0.0,
    min_high_confidence: float = 0.0,
    max_non_normal_ratio: float | None = None,
) -> pd.DataFrame:
    """Make low/high labels conservative for accuracy-oriented state recognition."""
    df = df.copy()
    if "label_confidence" not in df.columns:
        df["label_confidence"] = 1.0
    df["label_selection_confidence"] = df["label_confidence"].astype(float)
    if "high_event_hold_flag" in df.columns:
        high_mask = df["state_id"] == 2
        df.loc[high_mask, "label_selection_confidence"] = np.maximum(
            df.loc[high_mask, "label_selection_confidence"].astype(float),
            df.loc[high_mask, "high_event_hold_flag"].astype(float),
        )
    if "high_event_raw_flag" in df.columns:
        high_mask = df["state_id"] == 2
        df.loc[high_mask, "label_selection_confidence"] = np.maximum(
            df.loc[high_mask, "label_selection_confidence"].astype(float),
            df.loc[high_mask, "high_event_raw_flag"].astype(float),
        )
    df["original_state_id"] = df["state_id"].astype(int)
    df["original_state_name"] = df["state_name"].astype(str)
    df["label_filter_reason"] = ""

    low_uncertain = (df["state_id"] == 0) & (df["label_confidence"].astype(float) < float(min_low_confidence))
    high_uncertain = (df["state_id"] == 2) & (df["label_selection_confidence"].astype(float) < float(min_high_confidence))
    df.loc[low_uncertain, "label_filter_reason"] = "low_confidence_to_normal"
    df.loc[high_uncertain, "label_filter_reason"] = "high_confidence_to_normal"
    df.loc[low_uncertain | high_uncertain, "state_id"] = 1

    if max_non_normal_ratio is not None:
        ratio = float(max_non_normal_ratio)
        if not 0 < ratio < 1:
            raise ValueError("--max-non-normal-ratio must be between 0 and 1")
        non_normal = df[df["state_id"].isin([0, 2])].copy()
        max_keep = int(round(len(df) * ratio))
        if len(non_normal) > max_keep:
            keep_indices = set(
                non_normal.sort_values(
                    ["label_selection_confidence", "state_score"],
                    ascending=[False, False],
                ).head(max_keep).index.tolist()
            )
            cap_mask = df["state_id"].isin([0, 2]) & ~df.index.isin(keep_indices)
            df.loc[cap_mask & (df["label_filter_reason"] == ""), "label_filter_reason"] = "non_normal_ratio_cap_to_normal"
            df.loc[cap_mask, "state_id"] = 1

    df["state_id"] = df["state_id"].astype(int)
    df["state_name"] = df["state_id"].map({0: "low_activity", 1: "normal_activity", 2: "high_activity"})
    if (min_low_confidence > 0) or (min_high_confidence > 0) or (max_non_normal_ratio is not None):
        df["label_source"] = df["label_source"].astype(str) + "_conservative"
        df["label_version"] = df["label_version"].astype(str) + "_conservative"
    return df


def write_summary(df: pd.DataFrame, out_dir: Path, low_thr: float, high_thr: float) -> None:
    counts = df["state_name"].value_counts().to_dict()
    split_counts = df.groupby(["split", "state_name"]).size().unstack(fill_value=0).to_dict("index")
    filter_counts = {}
    if "label_filter_reason" in df.columns:
        filter_counts = {str(k): int(v) for k, v in df["label_filter_reason"].value_counts().to_dict().items()}
    summary = {
        "rows": int(len(df)),
        "frames": int(df["frame"].nunique()),
        "tracks": int(df["track_id"].nunique()),
        "state_counts": {k: int(v) for k, v in counts.items()},
        "split_state_counts": {
            split: {k: int(v) for k, v in values.items()}
            for split, values in split_counts.items()
        },
        "low_score_threshold": float(low_thr),
        "high_score_threshold": float(high_thr),
        "score_describe": {k: float(v) for k, v in df["state_score"].describe().to_dict().items()},
        "label_filter_counts": filter_counts,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_if_needed(img, scale: float):
    if abs(scale - 1.0) < 1e-6:
        return img
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def draw_label_video(df: pd.DataFrame, seq_dir: Path, seqinfo: dict, out_path: Path, max_frames: int, fps: float, resize: float) -> None:
    try:
        import cv2
    except ImportError:
        print("[WARN] opencv-python not installed, skip label quality video.")
        return

    frames = sorted(df["frame"].unique().tolist())[:max_frames]
    if not frames:
        return
    img_dir = seq_dir / str(seqinfo["im_dir"])
    im_ext = str(seqinfo["im_ext"])
    first = cv2.imread(str(img_dir / f"{frames[0]:06d}{im_ext}"))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    first = resize_if_needed(first, resize)
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {out_path}")
    grouped = dict(tuple(df.groupby("frame")))
    for frame in frames:
        img = cv2.imread(str(img_dir / f"{frame:06d}{im_ext}"))
        if img is None:
            continue
        for _, row in grouped.get(frame, pd.DataFrame()).iterrows():
            sid = int(row["state_id"])
            color = STATE_COLOR.get(sid, (255, 255, 255))
            x1, y1, x2, y2 = [int(round(float(row[c]))) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{int(row['track_id'])} {STATE_SHORT.get(row['state_name'], row['state_name'])} s:{float(row['state_score']):.2f}"
            cv2.putText(img, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"frame:{frame}", (20, img.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        writer.write(resize_if_needed(img, resize))
    writer.release()
    print(f"[OK] label quality video: {out_path}")


def main() -> None:
    args = parse_args()
    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    gt_path = Path(args.gt_file) if args.gt_file else seq_dir / "gt" / "gt.txt"
    seqinfo = read_seqinfo(seq_dir)
    sequence = seq_dir.name
    fps = float(seqinfo["frame_rate"])

    print("[INFO] load gt:", gt_path)
    df = load_mot_gt(gt_path, sequence=sequence)
    if args.feature_after_sampling:
        df = filter_frame_step(df, args.frame_step)
    print("[INFO] compute motion features")
    df = add_motion_features(
        df,
        fps=fps,
        frame_width=float(seqinfo["im_width"]),
        frame_height=float(seqinfo["im_height"]),
        smooth_window=args.smooth_window,
        feature_window=args.feature_window,
    )
    df, norm_stats = compute_activity_score(df, feature_weights=DEFAULT_SCORE_WEIGHTS)
    if args.label_mode == "persistent":
        df, low_thr, high_thr = assign_persistent_labels_from_score(
            df,
            fps=fps,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            window_sec=args.label_window_sec,
            low_required_ratio=args.low_required_ratio,
            high_required_ratio=args.high_required_ratio,
            high_event_percentile=args.high_event_percentile,
            high_event_hold_sec=args.high_event_hold_sec,
            backfill_persistent_states=bool(args.backfill_persistent_states),
        )
    else:
        df, low_thr, high_thr = assign_labels_from_score(
            df,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
        )
    df = apply_conservative_label_filter(
        df,
        min_low_confidence=args.min_low_label_confidence,
        min_high_confidence=args.min_high_label_confidence,
        max_non_normal_ratio=args.max_non_normal_ratio,
    )
    if not args.feature_after_sampling:
        df = filter_frame_step(df, args.frame_step)
    df = apply_temporal_split(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    img_dir = seq_dir / str(seqinfo["im_dir"])
    im_ext = str(seqinfo["im_ext"])
    df["image_path"] = df["frame"].astype(int).apply(lambda frame: str(img_dir / f"{frame:06d}{im_ext}"))

    print("[INFO] write dataset")
    save_dataset_files(df, out_dir)
    write_feature_config(out_dir, args, norm_stats, low_thr, high_thr)
    write_summary(df, out_dir, low_thr, high_thr)

    if args.make_video:
        video_out = Path(args.video_out) if args.video_out else out_dir / "label_quality_video.mp4"
        draw_label_video(
            df=df,
            seq_dir=seq_dir,
            seqinfo=seqinfo,
            out_path=video_out,
            max_frames=args.video_max_frames,
            fps=args.video_fps,
            resize=args.resize,
        )

    print("[DONE] activity state dataset:", out_dir)


if __name__ == "__main__":
    main()
