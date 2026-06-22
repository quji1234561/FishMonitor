#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight multi-feature activity-state model from generated labels."""

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
    DEFAULT_SCORE_WEIGHTS,
    STATE_ID_TO_NAME,
    apply_persistent_state_postprocess,
    compute_activity_score,
)


WEIGHT_CANDIDATES = {
    "balanced": DEFAULT_SCORE_WEIGHTS,
    "speed_dominant": {
        "speed_smooth": 0.24,
        "speed_mean_w5": 0.30,
        "speed_max_w5": 0.16,
        "speed_std_w5": 0.08,
        "accel_abs_smooth": 0.08,
        "turn_rate_w5": 0.02,
        "bbox_aspect_change_w5": 0.01,
        "bbox_size_change_w5": 0.01,
        "edge_shape_change_w5": 0.01,
        "in_place_turn_score": 0.01,
        "bbox_shape_change_rate_w1s": 0.01,
        "bbox_pulse_score_w1s": 0.01,
        "sharp_turn_event_w5": 0.01,
        "displacement_w5": 0.07,
        "movement_ratio_w5": 0.02,
        "group_speed_mean": 0.02,
    },
    "burst_turn": {
        "speed_smooth": 0.12,
        "speed_mean_w5": 0.16,
        "speed_max_w5": 0.12,
        "speed_std_w5": 0.10,
        "accel_abs_smooth": 0.14,
        "turn_rate_w5": 0.03,
        "bbox_aspect_change_w5": 0.05,
        "bbox_size_change_w5": 0.03,
        "edge_shape_change_w5": 0.05,
        "in_place_turn_score": 0.06,
        "bbox_shape_change_rate_w1s": 0.10,
        "bbox_pulse_score_w1s": 0.10,
        "sharp_turn_event_w5": 0.10,
        "displacement_w5": 0.04,
        "movement_ratio_w5": 0.04,
        "group_speed_mean": 0.02,
    },
    "window_stability": {
        "speed_smooth": 0.12,
        "speed_mean_w5": 0.28,
        "speed_max_w5": 0.10,
        "speed_std_w5": 0.10,
        "accel_abs_smooth": 0.10,
        "turn_rate_w5": 0.03,
        "bbox_aspect_change_w5": 0.03,
        "bbox_size_change_w5": 0.02,
        "edge_shape_change_w5": 0.03,
        "in_place_turn_score": 0.03,
        "bbox_shape_change_rate_w1s": 0.04,
        "bbox_pulse_score_w1s": 0.04,
        "sharp_turn_event_w5": 0.03,
        "displacement_w5": 0.12,
        "movement_ratio_w5": 0.04,
        "group_speed_mean": 0.04,
    },
    "group_aware": {
        "speed_smooth": 0.16,
        "speed_mean_w5": 0.20,
        "speed_max_w5": 0.10,
        "speed_std_w5": 0.08,
        "accel_abs_smooth": 0.10,
        "turn_rate_w5": 0.03,
        "bbox_aspect_change_w5": 0.03,
        "bbox_size_change_w5": 0.02,
        "edge_shape_change_w5": 0.03,
        "in_place_turn_score": 0.02,
        "bbox_shape_change_rate_w1s": 0.04,
        "bbox_pulse_score_w1s": 0.04,
        "sharp_turn_event_w5": 0.03,
        "displacement_w5": 0.06,
        "movement_ratio_w5": 0.04,
        "group_speed_mean": 0.10,
        "group_high_motion_ratio": 0.06,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-feature activity-state model")
    parser.add_argument("--train-csv", required=True, help="activity_state_train.csv")
    parser.add_argument("--val-csv", required=True, help="activity_state_val.csv")
    parser.add_argument("--out-dir", required=True, help="Output model directory")
    parser.add_argument("--feature-config", default=None, help="feature_config.json from generated dataset")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument(
        "--candidate-percentiles",
        default=None,
        help="Deprecated. Comma-separated shared threshold percentiles searched on train scores.",
    )
    parser.add_argument("--low-candidate-percentiles", default="10,15,20,25,30", help="Low-threshold percentiles")
    parser.add_argument("--high-candidate-percentiles", default="70,75,80,85,90", help="High-threshold percentiles")
    parser.add_argument("--resume-from", default=None, help="Optional previous last_activity_model.json")
    return parser.parse_args()


def load_feature_config(path: str | None) -> dict:
    if not path:
        return {
            "model_features": DEFAULT_MODEL_FEATURES,
            "feature_weights": DEFAULT_SCORE_WEIGHTS,
        }
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_dataset(path: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "state_id" not in df.columns:
        raise ValueError(f"{path} missing state_id")
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0.0
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["state_id"] = df["state_id"].astype(int)
    return df


def predict_by_score(scores: pd.Series, low_thr: float, high_thr: float) -> np.ndarray:
    pred = np.ones(len(scores), dtype=np.int64)
    values = scores.to_numpy(dtype=float)
    pred[values <= low_thr] = 0
    pred[values >= high_thr] = 2
    return pred


def build_postprocess_config(feature_config: dict, train_df: pd.DataFrame) -> dict:
    if feature_config.get("label_mode") != "persistent":
        return {"mode": "none"}
    if "label_window_rows" in train_df.columns and not train_df["label_window_rows"].dropna().empty:
        window_rows = int(round(float(train_df["label_window_rows"].dropna().median())))
    else:
        window_rows = 1
    if "label_high_event_score_threshold" in train_df.columns and not train_df["label_high_event_score_threshold"].dropna().empty:
        high_event_threshold = float(train_df["label_high_event_score_threshold"].dropna().median())
    elif "sharp_turn_event_w5" in train_df.columns:
        high_event_threshold = float(np.percentile(train_df["sharp_turn_event_w5"].dropna().to_numpy(), feature_config.get("high_event_percentile", 92.0)))
    else:
        high_event_threshold = None
    if "label_high_event_hold_rows" in train_df.columns and not train_df["label_high_event_hold_rows"].dropna().empty:
        high_event_hold_rows = int(round(float(train_df["label_high_event_hold_rows"].dropna().median())))
    else:
        high_event_hold_rows = max(1, int(round(window_rows / 3)))
    return {
        "mode": "persistent",
        "window_sec": float(feature_config.get("label_window_sec", 3.0)),
        "window_rows": max(1, window_rows),
        "low_required_ratio": float(feature_config.get("low_required_ratio", 0.8)),
        "high_required_ratio": float(feature_config.get("high_required_ratio", 0.6)),
        "backfill_persistent_states": bool(feature_config.get("backfill_persistent_states", True)),
        "high_event_col": "sharp_turn_event_w5",
        "high_event_threshold": high_event_threshold,
        "high_event_hold_sec": float(feature_config.get("high_event_hold_sec", 1.0)),
        "high_event_hold_rows": max(1, high_event_hold_rows),
    }


def predict_labels(
    scored_df: pd.DataFrame,
    low_thr: float,
    high_thr: float,
    postprocess: dict,
) -> np.ndarray:
    cols = ["frame", "track_id", "state_score"]
    high_event_col = str(postprocess.get("high_event_col", "sharp_turn_event_w5"))
    if high_event_col in scored_df.columns:
        cols.append(high_event_col)
    df = scored_df[cols].copy()
    df["state_id"] = predict_by_score(df["state_score"], low_thr, high_thr)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    if postprocess.get("mode") == "persistent":
        df = apply_persistent_state_postprocess(
            df,
            model={"postprocess": postprocess},
            fps=None,
        )
    return df["state_id"].to_numpy(dtype=np.int64)


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for cls in [0, 1, 2]:
        tp = float(((y_true == cls) & (y_pred == cls)).sum())
        fp = float(((y_true != cls) & (y_pred == cls)).sum())
        fn = float(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) <= 2 and 0 <= int(p) <= 2:
            mat[int(t), int(p)] += 1
    return mat.tolist()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": macro_f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def normalize_weights(weights: dict[str, float], features: list[str]) -> dict[str, float]:
    out = {feature: float(weights.get(feature, 0.0)) for feature in features}
    total = sum(max(v, 0.0) for v in out.values())
    if total <= 0:
        return {feature: 1.0 / len(features) for feature in features}
    return {feature: max(v, 0.0) / total for feature, v in out.items()}


def load_resume_candidate(path: str | None, features: list[str]) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    return {"resume_previous": normalize_weights(model.get("feature_weights", {}), features)}


def make_model_bundle(
    name: str,
    features: list[str],
    weights: dict[str, float],
    norm_stats: dict,
    low_thr: float,
    high_thr: float,
    train_metrics: dict,
    val_metrics: dict,
    source: dict,
    label_version: str,
) -> dict:
    return {
        "model_type": "weighted_motion_score_v1",
        "model_name": name,
        "label_version": label_version,
        "state_names": STATE_ID_TO_NAME,
        "model_features": features,
        "feature_weights": weights,
        "normalization": norm_stats,
        "low_score_threshold": float(low_thr),
        "high_score_threshold": float(high_thr),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "source": source,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_config = load_feature_config(args.feature_config)
    features = [str(x) for x in feature_config.get("model_features", DEFAULT_MODEL_FEATURES)]
    candidates = {
        name: normalize_weights(weights, features)
        for name, weights in WEIGHT_CANDIDATES.items()
    }
    candidates.update(load_resume_candidate(args.resume_from, features))

    train_df = load_dataset(args.train_csv, features)
    val_df = load_dataset(args.val_csv, features)
    postprocess = build_postprocess_config(feature_config, train_df)
    if args.candidate_percentiles:
        percentiles = [float(x.strip()) for x in args.candidate_percentiles.split(",") if x.strip()]
        low_percentiles = percentiles
        high_percentiles = percentiles
    else:
        low_percentiles = [float(x.strip()) for x in args.low_candidate_percentiles.split(",") if x.strip()]
        high_percentiles = [float(x.strip()) for x in args.high_candidate_percentiles.split(",") if x.strip()]

    y_train = train_df["state_id"].to_numpy(dtype=np.int64)
    y_val = val_df["state_id"].to_numpy(dtype=np.int64)
    rows = []
    best = None
    last_bundle = None

    for candidate_name, weights in candidates.items():
        train_scored, norm_stats = compute_activity_score(train_df, feature_weights=weights)
        val_scored, _ = compute_activity_score(val_df, feature_weights=weights, norm_stats=norm_stats)
        for low_p in low_percentiles:
            for high_p in high_percentiles:
                if high_p <= low_p:
                    continue
                low_thr = float(np.percentile(train_scored["state_score"], low_p))
                high_thr = float(np.percentile(train_scored["state_score"], high_p))
                pred_train = predict_labels(train_scored, low_thr, high_thr, postprocess)
                pred_val = predict_labels(val_scored, low_thr, high_thr, postprocess)
                train_metrics = evaluate(y_train, pred_train)
                val_metrics = evaluate(y_val, pred_val)
                row = {
                    "candidate": candidate_name,
                    "low_percentile": low_p,
                    "high_percentile": high_p,
                    "low_score_threshold": low_thr,
                    "high_score_threshold": high_thr,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
                rows.append(row)
                bundle = make_model_bundle(
                    name=candidate_name,
                    features=features,
                    weights=weights,
                    norm_stats=norm_stats,
                    low_thr=low_thr,
                    high_thr=high_thr,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    source={
                        "train_csv": args.train_csv,
                        "val_csv": args.val_csv,
                        "feature_config": args.feature_config or "",
                        "low_percentile": low_p,
                        "high_percentile": high_p,
                    },
                    label_version=str(feature_config.get("label_version", "v9_event_below_low_overlap_suppressed")),
                )
                bundle["postprocess"] = postprocess
                last_bundle = bundle
                score = val_metrics[args.metric]
                if best is None or score > best["score"]:
                    best = {"score": score, "bundle": bundle, "row": row}

    if best is None or last_bundle is None:
        raise RuntimeError("No candidate model trained.")

    pd.DataFrame(rows).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    (out_dir / "best_activity_model.json").write_text(
        json.dumps(best["bundle"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "last_activity_model.json").write_text(
        json.dumps(last_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_metrics = {
        "selection_metric": args.metric,
        "best_score": best["score"],
        "best_row": best["row"],
        "best_train_metrics": best["bundle"]["train_metrics"],
        "best_val_metrics": best["bundle"]["val_metrics"],
    }
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] best model: {out_dir / 'best_activity_model.json'}")
    print(f"[OK] last model: {out_dir / 'last_activity_model.json'}")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
