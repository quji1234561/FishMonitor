#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build activity-state training CSVs from YOLO boxes matched to GT state labels."""

from __future__ import annotations

import argparse
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
    add_motion_features,
)


STATE_ID_TO_NAME = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO-box activity-state dataset by matching YOLO boxes to GT labels")
    parser.add_argument("--gt-csv", required=True, help="activity_state_all.csv, or a split CSV")
    parser.add_argument("--pred-csv", required=True, help="YOLO raw_predictions.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--feature-config", default=None, help="feature_config.json from GT activity dataset")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="Minimum IoU for matching YOLO boxes to GT labels")
    parser.add_argument("--pred-frame-offset", type=int, default=0, help="Add this offset to YOLO frame ids")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS used for motion features")
    parser.add_argument("--speed-norm", choices=["bbox_diag", "bbox_size", "none"], default="bbox_diag")
    parser.add_argument("--smooth-window", type=int, default=5, help="Speed smoothing window")
    parser.add_argument("--feature-window", type=int, default=5, help="Motion feature rolling window")
    parser.add_argument("--keep-unmatched", action="store_true", help="Also save unmatched YOLO/GT rows for debugging")
    return parser.parse_args()


def ensure_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        return df
    required = {"x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing box columns: {missing}")
    df["x1"] = df["x"]
    df["y1"] = df["y"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    return df


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    gx1, gy1, gx2, gy2 = [gt_boxes[:, i][:, None] for i in range(4)]
    px1, py1, px2, py2 = [pred_boxes[:, i][None, :] for i in range(4)]
    ix1 = np.maximum(gx1, px1)
    iy1 = np.maximum(gy1, py1)
    ix2 = np.minimum(gx2, px2)
    iy2 = np.minimum(gy2, py2)
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    gt_area = np.maximum(gx2 - gx1, 0.0) * np.maximum(gy2 - gy1, 0.0)
    pred_area = np.maximum(px2 - px1, 0.0) * np.maximum(py2 - py1, 0.0)
    union = gt_area + pred_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match_frame(gt_frame: pd.DataFrame, pred_frame: pd.DataFrame, iou_thr: float) -> list[tuple[int, int, float]]:
    gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    mat = iou_matrix(gt_boxes, pred_boxes)
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    while mat.size:
        gt_idx, pred_idx = np.unravel_index(int(np.argmax(mat)), mat.shape)
        best_iou = float(mat[gt_idx, pred_idx])
        if best_iou < iou_thr:
            break
        if gt_idx not in used_gt and pred_idx not in used_pred:
            matches.append((gt_idx, pred_idx, best_iou))
            used_gt.add(gt_idx)
            used_pred.add(pred_idx)
        mat[gt_idx, :] = -1.0
        mat[:, pred_idx] = -1.0
    return matches


def load_feature_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_matched_rows(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    unmatched_gt_rows = []
    matched_pred_indices: set[int] = set()
    pred_groups = {int(k): v.reset_index(drop=False) for k, v in pred.groupby("frame")}
    for frame, gt_frame_raw in gt.groupby("frame"):
        gt_frame = gt_frame_raw.reset_index(drop=False)
        pred_frame = pred_groups.get(int(frame), pd.DataFrame(columns=list(pred.columns) + ["index"]))
        matches = greedy_match_frame(gt_frame, pred_frame, iou_thr=iou_thr)
        matched_gt_local = set()
        for gt_i, pred_i, iou in matches:
            gt_row = gt_frame.iloc[gt_i]
            pred_row = pred_frame.iloc[pred_i]
            matched_gt_local.add(gt_i)
            matched_pred_indices.add(int(pred_row["index"]))
            state_id = int(gt_row["state_id"])
            rows.append({
                "sequence": gt_row.get("sequence", ""),
                "split": str(gt_row.get("split", "")),
                "frame": int(frame),
                "track_id": int(pred_row.get("track_id", -1)),
                "gt_track_id": int(gt_row.get("track_id", -1)),
                "pred_track_id": int(pred_row.get("track_id", -1)),
                "source_path": pred_row.get("source_path", ""),
                "gt_iou": float(iou),
                "conf": float(pred_row.get("conf", 1.0)),
                "x1": float(pred_row["x1"]),
                "y1": float(pred_row["y1"]),
                "x2": float(pred_row["x2"]),
                "y2": float(pred_row["y2"]),
                "x": float(pred_row["x1"]),
                "y": float(pred_row["y1"]),
                "w": float(pred_row["x2"] - pred_row["x1"]),
                "h": float(pred_row["y2"] - pred_row["y1"]),
                "cx": float((pred_row["x1"] + pred_row["x2"]) / 2.0),
                "cy": float((pred_row["y1"] + pred_row["y2"]) / 2.0),
                "state_id": state_id,
                "state_name": STATE_ID_TO_NAME.get(state_id, str(state_id)),
                "gt_state_id": state_id,
                "gt_state_name": STATE_ID_TO_NAME.get(state_id, str(state_id)),
                "label_source": "gt_state_matched_to_yolo_box",
                "label_version": gt_row.get("label_version", ""),
            })
        for gt_i, gt_row in gt_frame.iterrows():
            if int(gt_i) not in matched_gt_local:
                unmatched_gt_rows.append(gt_row.to_dict())

    unmatched_pred = pred.loc[~pred.index.isin(matched_pred_indices)].copy()
    return pd.DataFrame(rows), pd.DataFrame(unmatched_gt_rows), unmatched_pred


def write_split_files(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "yolo_activity_state_all.csv", index=False, encoding="utf-8-sig")
    if "split" not in df.columns or df["split"].astype(str).eq("").all():
        return
    for split in ["train", "val", "test"]:
        part = df[df["split"].astype(str) == split].copy()
        part.to_csv(out_dir / f"yolo_activity_state_{split}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_config = load_feature_config(args.feature_config)
    gt = ensure_xyxy(pd.read_csv(args.gt_csv))
    pred = ensure_xyxy(pd.read_csv(args.pred_csv))
    if "state_id" not in gt.columns:
        raise ValueError("GT CSV must contain state_id")
    gt["frame"] = gt["frame"].astype(int)
    pred["frame"] = pred["frame"].astype(int) + int(args.pred_frame_offset)
    gt = gt.sort_values(["frame", "track_id"]).reset_index(drop=True)
    pred = pred.sort_values(["frame", "track_id"]).reset_index(drop=True)

    matched, unmatched_gt, unmatched_pred = build_matched_rows(gt, pred, iou_thr=args.iou_thr)
    if matched.empty:
        raise RuntimeError("No YOLO boxes matched to GT state labels. Check IoU threshold and frame ids.")

    matched = add_motion_features(
        matched,
        fps=args.fps,
        speed_norm=args.speed_norm,
        smooth_window=args.smooth_window,
        feature_window=args.feature_window,
    )

    # Preserve state labels after feature extraction.
    matched["state_id"] = matched["gt_state_id"].astype(int)
    matched["state_name"] = matched["gt_state_name"].astype(str)
    matched["label_source"] = "gt_state_matched_to_yolo_box"
    label_version = str(feature_config.get("label_version", ""))
    if label_version:
        matched["label_version"] = label_version
    elif "label_version" not in matched.columns:
        matched["label_version"] = ""

    write_split_files(matched, out_dir)
    summary = {
        "gt_csv": args.gt_csv,
        "pred_csv": args.pred_csv,
        "feature_config": args.feature_config or "",
        "iou_threshold": args.iou_thr,
        "gt_rows": int(len(gt)),
        "pred_rows": int(len(pred)),
        "matched_rows": int(len(matched)),
        "unmatched_gt_rows": int(len(unmatched_gt)),
        "unmatched_pred_rows": int(len(unmatched_pred)),
        "match_recall_gt": float(len(matched) / len(gt)) if len(gt) else 0.0,
        "match_precision_pred": float(len(matched) / len(pred)) if len(pred) else 0.0,
        "state_counts": {str(k): int(v) for k, v in matched["state_name"].value_counts().to_dict().items()},
        "split_state_counts": {
            split: {str(k): int(v) for k, v in part["state_name"].value_counts().to_dict().items()}
            for split, part in matched.groupby("split")
        } if "split" in matched.columns else {},
        "model_features": feature_config.get("model_features", DEFAULT_MODEL_FEATURES),
        "label_version": feature_config.get("label_version", ""),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.keep_unmatched:
        unmatched_gt.to_csv(out_dir / "unmatched_gt_rows.csv", index=False, encoding="utf-8-sig")
        unmatched_pred.to_csv(out_dir / "unmatched_yolo_rows.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
