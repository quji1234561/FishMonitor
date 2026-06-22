#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for supervised activity-state classifiers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_analysis.motion_features import DEFAULT_MODEL_FEATURES, DEFAULT_SCORE_WEIGHTS, STATE_ID_TO_NAME, compute_activity_score


DEFAULT_CLASSIFIER_FEATURES = [
    *DEFAULT_MODEL_FEATURES,
    "conf",
    "bbox_area",
    "bbox_diag",
    "bbox_aspect",
    "same_frame_max_iou",
    "overlap_reliability",
    "nearest_neighbor_dist_norm",
    "crowding_index",
    "fish_count_frame",
    "group_speed_std",
    "frame_gap",
    "valid_motion",
    "track_len",
    "classifier_state_score",
    "classifier_score_mean_w1s",
    "classifier_score_mean_w3s",
    "classifier_score_min_w3s",
    "classifier_score_max_w3s",
    "classifier_low_ratio_w3s",
    "classifier_high_ratio_w3s",
    "classifier_sharp_event_max_w1s",
    "classifier_sharp_event_max_w3s",
    "classifier_speed_mean_w3s",
    "classifier_speed_max_w3s",
]


def load_activity_classifier(path: Path) -> dict:
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid activity classifier bundle: {path}")
    if bundle.get("model_type") != "sklearn_activity_classifier_v1":
        raise ValueError(f"Unsupported activity classifier type: {bundle.get('model_type')}")
    return bundle


def save_activity_classifier(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def prepare_feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    data = df.copy()
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0.0
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    return data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _infer_window_rows(df: pd.DataFrame, fps: float, window_sec: float) -> int:
    gaps = df.get("frame_gap", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    gaps = gaps[gaps > 0].dropna()
    median_gap = float(gaps.median()) if not gaps.empty else 1.0
    if not np.isfinite(median_gap) or median_gap <= 0:
        median_gap = 1.0
    return max(1, int(round(float(fps) / median_gap * float(window_sec))))


def add_classifier_context_features(
    df: pd.DataFrame,
    context: dict | None = None,
    *,
    fit: bool = False,
    feature_weights: dict[str, float] | None = None,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
    fps: float = 25.0,
) -> tuple[pd.DataFrame, dict]:
    """Add score/window features that mirror the label-generation rules."""
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)
    if context is None:
        context = {}
    weights = feature_weights or context.get("feature_weights") or DEFAULT_SCORE_WEIGHTS
    if fit:
        df, norm_stats = compute_activity_score(df, feature_weights=weights)
        scores = df["state_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        low_thr = float(np.percentile(scores, low_percentile)) if len(scores) else 0.0
        high_thr = float(np.percentile(scores, high_percentile)) if len(scores) else 0.0
        context = {
            "feature_weights": weights,
            "score_norm_stats": norm_stats,
            "low_score_threshold": low_thr,
            "high_score_threshold": high_thr,
            "fps": float(fps),
        }
    else:
        df, _ = compute_activity_score(
            df,
            feature_weights=weights,
            norm_stats=context.get("score_norm_stats"),
        )
        low_thr = float(context.get("low_score_threshold", 0.0))
        high_thr = float(context.get("high_score_threshold", 1.0))

    df["classifier_state_score"] = df["state_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["classifier_low_score_flag"] = (df["classifier_state_score"] <= low_thr).astype(float)
    df["classifier_high_score_flag"] = (df["classifier_state_score"] >= high_thr).astype(float)

    w1 = _infer_window_rows(df, fps=float(context.get("fps", fps)), window_sec=1.0)
    w3 = _infer_window_rows(df, fps=float(context.get("fps", fps)), window_sec=3.0)
    g = df.groupby("track_id", group_keys=False)
    df["classifier_score_mean_w1s"] = g["classifier_state_score"].transform(
        lambda s: s.rolling(window=w1, min_periods=1).mean()
    )
    df["classifier_score_mean_w3s"] = g["classifier_state_score"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).mean()
    )
    df["classifier_score_min_w3s"] = g["classifier_state_score"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).min()
    )
    df["classifier_score_max_w3s"] = g["classifier_state_score"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).max()
    )
    df["classifier_low_ratio_w3s"] = g["classifier_low_score_flag"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).mean()
    )
    df["classifier_high_ratio_w3s"] = g["classifier_high_score_flag"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).mean()
    )
    if "sharp_turn_event_w5" not in df.columns:
        df["sharp_turn_event_w5"] = 0.0
    df["classifier_sharp_event_max_w1s"] = g["sharp_turn_event_w5"].transform(
        lambda s: s.rolling(window=w1, min_periods=1).max()
    )
    df["classifier_sharp_event_max_w3s"] = g["sharp_turn_event_w5"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).max()
    )
    if "speed_smooth" not in df.columns:
        df["speed_smooth"] = 0.0
    df["classifier_speed_mean_w3s"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).mean()
    )
    df["classifier_speed_max_w3s"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=w3, min_periods=1).max()
    )

    for col in DEFAULT_CLASSIFIER_FEATURES:
        if col.startswith("classifier_"):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, context


def predict_with_activity_classifier(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = df.copy()
    features = [str(x) for x in bundle["model_features"]]
    estimator = bundle["estimator"]
    df, _ = add_classifier_context_features(df, context=bundle.get("feature_context", {}), fit=False)
    x = prepare_feature_matrix(df, features)
    pred = estimator.predict(x).astype(int)
    adjusted_pred = pred.copy()
    decision = bundle.get("decision", {})
    df["classifier_state_id"] = pred
    df["classifier_state_name"] = pd.Series(pred, index=df.index).map(STATE_ID_TO_NAME)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        classes = [int(c) for c in getattr(estimator, "classes_", [])]
        for state_id, state_name in STATE_ID_TO_NAME.items():
            col = f"prob_{state_name}"
            if state_id in classes:
                df[col] = proba[:, classes.index(state_id)]
            else:
                df[col] = 0.0
        df["classifier_confidence"] = df[[f"prob_{name}" for name in STATE_ID_TO_NAME.values()]].max(axis=1)
        non_normal_min_prob = float(decision.get("non_normal_min_prob", 0.0))
        if non_normal_min_prob > 0:
            for i, state_id in enumerate(pred):
                state_id = int(state_id)
                if state_id == 1 or state_id not in classes:
                    continue
                state_prob = float(proba[i, classes.index(state_id)])
                if state_prob < non_normal_min_prob:
                    adjusted_pred[i] = 1
    else:
        df["classifier_confidence"] = 1.0

    df["classifier_adjusted_state_id"] = adjusted_pred.astype(int)
    df["classifier_adjusted_state_name"] = pd.Series(adjusted_pred, index=df.index).map(STATE_ID_TO_NAME)
    df["state_id"] = df["classifier_adjusted_state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    return df
