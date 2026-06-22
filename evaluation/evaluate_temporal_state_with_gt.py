#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate temporal activity states by matching predicted boxes to GT activity labels.

Inputs:
1. temporal_results.csv from temporal_analysis/analyze_temporal_states.py
2. auto_speed_labels_all_frames.csv from scripts/auto_label_fish_activity.py

For each frame, predicted boxes are greedily matched to GT boxes by IoU. Matched
pairs are used to compare predicted smooth_state_id with GT state_id.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


STATE_NAMES = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temporal activity states against GT activity labels")
    parser.add_argument("--pred-csv", required=True, help="temporal_results.csv")
    parser.add_argument("--gt-activity-csv", required=True, help="auto_speed_labels_all_frames.csv")
    parser.add_argument("--out-dir", default="output/evaluation/temporal_state")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    return parser.parse_args()


def box_xyxy(row) -> Tuple[float, float, float, float]:
    x1 = float(row["x"])
    y1 = float(row["y"])
    x2 = x1 + float(row["w"])
    y2 = y1 + float(row["h"])
    return x1, y1, x2, y2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_frame: pd.DataFrame, gt_frame: pd.DataFrame, iou_thr: float) -> List[dict]:
    candidates = []
    for pred_idx, pred_row in pred_frame.iterrows():
        pred_box = box_xyxy(pred_row)
        for gt_idx, gt_row in gt_frame.iterrows():
            score = iou(pred_box, box_xyxy(gt_row))
            if score >= iou_thr:
                candidates.append((score, pred_idx, gt_idx))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_pred = set()
    used_gt = set()
    matches = []
    for score, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        pred_row = pred_frame.loc[pred_idx]
        gt_row = gt_frame.loc[gt_idx]
        pred_state = int(pred_row["smooth_state_id"])
        gt_state = int(gt_row["state_id"])
        matches.append(
            {
                "frame": int(pred_row["frame"]),
                "pred_track_id": int(pred_row["track_id"]),
                "gt_track_id": int(gt_row["track_id"]),
                "iou": float(score),
                "pred_state_id": pred_state,
                "pred_state_name": STATE_NAMES[pred_state],
                "gt_state_id": gt_state,
                "gt_state_name": STATE_NAMES[gt_state],
                "correct": int(pred_state == gt_state),
                "pred_x": float(pred_row["x"]),
                "pred_y": float(pred_row["y"]),
                "pred_w": float(pred_row["w"]),
                "pred_h": float(pred_row["h"]),
                "gt_x": float(gt_row["x"]),
                "gt_y": float(gt_row["y"]),
                "gt_w": float(gt_row["w"]),
                "gt_h": float(gt_row["h"]),
            }
        )
    return matches


def confusion_matrix(y_true, y_pred, n_classes: int = 3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm):
    total = int(cm.sum())
    acc = int(np.trace(cm)) / total if total else 0.0
    rows = []
    f1s = []
    for sid in range(cm.shape[0]):
        tp = int(cm[sid, sid])
        fp = int(cm[:, sid].sum() - tp)
        fn = int(cm[sid, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        rows.append(
            {
                "state_id": sid,
                "state_name": STATE_NAMES[sid],
                "support": int(cm[sid, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "support": total}, rows


def save_confusion_png(cm, out_path: Path):
    labels = ["low", "normal", "high"]
    cell = 130
    margin_left = 190
    margin_top = 150
    width = margin_left + cell * 3 + 40
    height = margin_top + cell * 3 + 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Temporal State Confusion Matrix", fill=(20, 20, 20))
    draw.text((margin_left + 70, 55), "Predicted", fill=(20, 20, 20))
    draw.text((25, margin_top + 150), "GT", fill=(20, 20, 20))
    max_value = max(int(cm.max()), 1)
    for i, label in enumerate(labels):
        draw.text((margin_left + i * cell + 30, margin_top - 35), label, fill=(20, 20, 20))
        draw.text((margin_left - 120, margin_top + i * cell + 55), label, fill=(20, 20, 20))
    for r in range(3):
        for c in range(3):
            value = int(cm[r, c])
            intensity = int(245 - 180 * (value / max_value))
            color = (intensity, intensity + 5 if intensity < 245 else 245, 255)
            x0 = margin_left + c * cell
            y0 = margin_top + r * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(80, 80, 80))
            draw.text((x0 + 45, y0 + 55), str(value), fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    gt = pd.read_csv(args.gt_activity_csv)

    pred_required = {"frame", "track_id", "x", "y", "w", "h", "smooth_state_id"}
    gt_required = {"frame", "track_id", "x", "y", "w", "h", "state_id"}
    pred_missing = pred_required - set(pred.columns)
    gt_missing = gt_required - set(gt.columns)
    if pred_missing:
        raise ValueError(f"pred CSV missing columns: {sorted(pred_missing)}")
    if gt_missing:
        raise ValueError(f"GT CSV missing columns: {sorted(gt_missing)}")

    pred["frame"] = pred["frame"].astype(int)
    gt["frame"] = gt["frame"].astype(int)
    pred["smooth_state_id"] = pred["smooth_state_id"].astype(int)
    gt["state_id"] = gt["state_id"].astype(int)

    all_matches = []
    frames = sorted(set(pred["frame"].unique()).intersection(gt["frame"].unique()))
    gt_by_frame = {frame: part.copy() for frame, part in gt.groupby("frame")}
    pred_by_frame = {frame: part.copy() for frame, part in pred.groupby("frame")}
    for frame in frames:
        all_matches.extend(greedy_match(pred_by_frame[frame], gt_by_frame[frame], args.iou_thr))

    matches = pd.DataFrame(all_matches)
    matches_path = out_dir / "matched_state_results.csv"
    matches.to_csv(matches_path, index=False, encoding="utf-8-sig")

    if matches.empty:
        raise RuntimeError("No matched prediction/GT pairs. Try lowering --iou-thr or checking frame ids.")

    cm = confusion_matrix(matches["gt_state_id"].to_numpy(), matches["pred_state_id"].to_numpy())
    summary, per_class = metrics_from_cm(cm)
    match_recall = len(matches) / len(gt[gt["frame"].isin(frames)]) if frames else 0.0
    pred_match_rate = len(matches) / len(pred[pred["frame"].isin(frames)]) if frames else 0.0

    pd.DataFrame(cm, index=STATE_NAMES.values(), columns=STATE_NAMES.values()).to_csv(
        out_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame([summary | {"gt_match_recall": match_recall, "pred_match_rate": pred_match_rate}]).to_csv(
        out_dir / "overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(per_class).to_csv(out_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    save_confusion_png(cm, out_dir / "confusion_matrix.png")

    report = (
        "时序状态识别匹配评估摘要\n"
        "========================\n"
        f"预测结果: {args.pred_csv}\n"
        f"GT 状态标签: {args.gt_activity_csv}\n"
        f"IoU 阈值: {args.iou_thr:.2f}\n"
        f"参与评估帧数: {len(frames)}\n"
        f"匹配框数量: {len(matches)}\n"
        f"GT 匹配召回率: {match_recall:.4f}\n"
        f"预测框匹配率: {pred_match_rate:.4f}\n"
        f"匹配后状态准确率: {summary['accuracy']:.4f}\n"
        f"匹配后宏平均 F1: {summary['macro_f1']:.4f}\n"
        f"输出明细: {matches_path}\n"
    )
    (out_dir / "evaluation_summary.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
