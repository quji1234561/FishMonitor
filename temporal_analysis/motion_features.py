#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared motion feature extraction for activity-state labeling and inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STATE_ID_TO_NAME = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}
STATE_NAME_TO_ID = {v: k for k, v in STATE_ID_TO_NAME.items()}

DEFAULT_MODEL_FEATURES = [
    "speed_smooth",
    "speed_mean_w5",
    "speed_max_w5",
    "speed_std_w5",
    "accel_abs_smooth",
    "turn_rate_w5",
    "bbox_aspect_change_w5",
    "bbox_size_change_w5",
    "edge_shape_change_w5",
    "in_place_turn_score",
    "bbox_shape_change_rate_w1s",
    "bbox_pulse_score_w1s",
    "sharp_turn_event_w5",
    "displacement_w5",
    "movement_ratio_w5",
    "group_speed_mean",
    "group_high_motion_ratio",
]

DEFAULT_SCORE_WEIGHTS = {
    "speed_smooth": 0.16,
    "speed_mean_w5": 0.20,
    "speed_max_w5": 0.10,
    "speed_std_w5": 0.07,
    "accel_abs_smooth": 0.11,
    "turn_rate_w5": 0.03,
    "bbox_aspect_change_w5": 0.04,
    "bbox_size_change_w5": 0.03,
    "edge_shape_change_w5": 0.03,
    "in_place_turn_score": 0.03,
    "bbox_shape_change_rate_w1s": 0.06,
    "bbox_pulse_score_w1s": 0.06,
    "sharp_turn_event_w5": 0.03,
    "displacement_w5": 0.07,
    "movement_ratio_w5": 0.03,
    "group_speed_mean": 0.02,
    "group_high_motion_ratio": 0.01,
}


def ensure_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "x1" not in df.columns:
        df["x1"] = df["x"]
    if "y1" not in df.columns:
        df["y1"] = df["y"]
    if "x2" not in df.columns:
        df["x2"] = df["x"] + df["w"]
    if "y2" not in df.columns:
        df["y2"] = df["y"] + df["h"]
    if "cx" not in df.columns:
        df["cx"] = df["x"] + df["w"] / 2.0
    if "cy" not in df.columns:
        df["cy"] = df["y"] + df["h"] / 2.0
    return df


def robust_normalize(series: pd.Series, lower_q: float = 10.0, upper_q: float = 90.0) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    lo = float(np.nanpercentile(s, lower_q))
    hi = float(np.nanpercentile(s, upper_q))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    out = (s.fillna(lo) - lo) / (hi - lo)
    return out.clip(0.0, 1.0)


def robust_normalize_open_upper(series: pd.Series, lower_q: float = 10.0, upper_q: float = 90.0) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    lo = float(np.nanpercentile(s, lower_q))
    hi = float(np.nanpercentile(s, upper_q))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    out = (s.fillna(lo) - lo) / (hi - lo)
    return out.clip(lower=0.0)


def angle_diff_abs(a: pd.Series, b: pd.Series) -> pd.Series:
    diff = a - b
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff.abs()


def add_overlap_features(
    df: pd.DataFrame,
    soft_iou: float = 0.15,
    hard_iou: float = 0.40,
) -> pd.DataFrame:
    """Add same-frame overlap reliability for shape-based event features."""
    df = df.copy()
    max_ious = pd.Series(0.0, index=df.index, dtype=float)
    for _, frame_df in df.groupby("frame", sort=False):
        if len(frame_df) <= 1:
            continue
        boxes = frame_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        x1 = boxes[:, 0][:, None]
        y1 = boxes[:, 1][:, None]
        x2 = boxes[:, 2][:, None]
        y2 = boxes[:, 3][:, None]
        ox1 = boxes[:, 0][None, :]
        oy1 = boxes[:, 1][None, :]
        ox2 = boxes[:, 2][None, :]
        oy2 = boxes[:, 3][None, :]
        ix1 = np.maximum(x1, ox1)
        iy1 = np.maximum(y1, oy1)
        ix2 = np.minimum(x2, ox2)
        iy2 = np.minimum(y2, oy2)
        inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
        area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        union = area[:, None] + area[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        np.fill_diagonal(iou, 0.0)
        max_ious.loc[frame_df.index] = np.max(iou, axis=1)

    df["same_frame_max_iou"] = max_ious.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = max(hard_iou - soft_iou, 1e-6)
    df["overlap_suppression"] = ((df["same_frame_max_iou"] - soft_iou) / denom).clip(0.0, 1.0)
    df["overlap_reliability"] = (1.0 - df["overlap_suppression"]).clip(0.0, 1.0)
    return df


def add_motion_features(
    df: pd.DataFrame,
    fps: float,
    frame_width: float | None = None,
    frame_height: float | None = None,
    speed_norm: str = "bbox_diag",
    smooth_window: int = 5,
    feature_window: int = 5,
) -> pd.DataFrame:
    """Add lightweight per-fish and per-frame motion features.

    Input rows should represent one fish box in one frame and contain at least:
    frame, track_id, x, y, w, h.
    """
    df = ensure_xyxy(df)
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    df["bbox_area"] = df["w"].astype(float) * df["h"].astype(float)
    df["bbox_diag"] = np.sqrt(df["w"].astype(float) ** 2 + df["h"].astype(float) ** 2).replace(0, np.nan)
    df["bbox_size"] = np.sqrt(df["bbox_area"]).replace(0, np.nan)
    df["bbox_aspect"] = df["w"].astype(float) / df["h"].replace(0, np.nan).astype(float)
    df = add_overlap_features(df)

    if frame_width and frame_height:
        df["bbox_area_norm"] = df["bbox_area"] / float(frame_width * frame_height)
        df["center_x_norm"] = df["cx"] / float(frame_width)
        df["center_y_norm"] = df["cy"] / float(frame_height)
        left = df["cx"]
        right = float(frame_width) - df["cx"]
        top = df["cy"]
        bottom = float(frame_height) - df["cy"]
        df["edge_distance_norm"] = np.minimum.reduce([left, right, top, bottom]) / max(frame_width, frame_height)
    else:
        df["bbox_area_norm"] = np.nan
        df["center_x_norm"] = np.nan
        df["center_y_norm"] = np.nan
        df["edge_distance_norm"] = np.nan

    g = df.groupby("track_id", group_keys=False)
    df["prev_frame"] = g["frame"].shift(1)
    df["prev_cx"] = g["cx"].shift(1)
    df["prev_cy"] = g["cy"].shift(1)
    df["frame_gap"] = df["frame"] - df["prev_frame"]
    valid_gap = df["frame_gap"] > 0
    positive_gaps = df.loc[valid_gap, "frame_gap"].replace([np.inf, -np.inf], np.nan).dropna()
    median_gap = float(positive_gaps.median()) if not positive_gaps.empty else 1.0
    if not np.isfinite(median_gap) or median_gap <= 0:
        median_gap = 1.0
    window_1s = max(1, int(round(float(fps) / median_gap)))
    df["dx"] = df["cx"] - df["prev_cx"]
    df["dy"] = df["cy"] - df["prev_cy"]
    df.loc[~valid_gap, ["dx", "dy"]] = np.nan
    df["dist_px"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["speed_px_s"] = np.nan
    df.loc[valid_gap, "speed_px_s"] = df.loc[valid_gap, "dist_px"] / df.loc[valid_gap, "frame_gap"] * fps

    if speed_norm == "none":
        df["speed_norm_s"] = df["speed_px_s"]
        denom = pd.Series(1.0, index=df.index)
    else:
        denom = df["bbox_size"] if speed_norm == "bbox_size" else df["bbox_diag"]
        df["speed_norm_s"] = df["speed_px_s"] / denom.replace(0, np.nan)

    med_speed = df["speed_norm_s"].median()
    if not np.isfinite(med_speed):
        med_speed = 0.0
    df["speed_norm_s"] = g["speed_norm_s"].transform(lambda s: s.bfill().ffill()).fillna(med_speed)

    df["speed_smooth"] = g["speed_norm_s"].transform(
        lambda s: s.rolling(window=smooth_window, min_periods=1).median()
    )
    df["speed_smooth"] = g["speed_smooth"].transform(lambda s: s.bfill().ffill()).fillna(med_speed)

    df["prev_x1"] = g["x1"].shift(1)
    df["prev_y1"] = g["y1"].shift(1)
    df["prev_x2"] = g["x2"].shift(1)
    df["prev_y2"] = g["y2"].shift(1)
    df["prev_bbox_diag"] = g["bbox_diag"].shift(1)
    df["bbox_diag_log"] = np.log(df["bbox_diag"].replace(0, np.nan))
    df["prev_bbox_diag_log"] = g["bbox_diag_log"].shift(1)
    df["bbox_aspect_log"] = np.log(df["bbox_aspect"].replace(0, np.nan))
    df["prev_bbox_aspect_log"] = g["bbox_aspect_log"].shift(1)
    df["bbox_aspect_change_abs"] = (df["bbox_aspect_log"] - df["prev_bbox_aspect_log"]).abs()
    diag_ratio = df["bbox_diag"] / df["prev_bbox_diag"].replace(0, np.nan)
    df["bbox_size_change_abs"] = np.log(diag_ratio.replace(0, np.nan)).abs()

    edge_residual = (
        (df["x1"] - df["prev_x1"] - df["dx"]).abs()
        + (df["x2"] - df["prev_x2"] - df["dx"]).abs()
        + (df["y1"] - df["prev_y1"] - df["dy"]).abs()
        + (df["y2"] - df["prev_y2"] - df["dy"]).abs()
    ) / (2.0 * denom.replace(0, np.nan))
    df["edge_shape_change_norm"] = edge_residual
    shape_raw_cols = ["bbox_aspect_change_abs", "bbox_size_change_abs", "edge_shape_change_norm"]
    df.loc[~valid_gap, shape_raw_cols] = np.nan
    for col in shape_raw_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dt_s = df["frame_gap"] / float(fps)
    dt_s = dt_s.where(valid_gap & (dt_s > 0), np.nan)
    df["bbox_aspect_change_rate_s"] = df["bbox_aspect_change_abs"] / dt_s
    df["bbox_size_change_rate_s"] = df["bbox_size_change_abs"] / dt_s
    df["edge_shape_change_rate_s"] = df["edge_shape_change_norm"] / dt_s
    rate_cols = ["bbox_aspect_change_rate_s", "bbox_size_change_rate_s", "edge_shape_change_rate_s"]
    for col in rate_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["bbox_aspect_change_w5"] = g["bbox_aspect_change_abs"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).mean()
    ).fillna(0.0)
    df["bbox_size_change_w5"] = g["bbox_size_change_abs"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).mean()
    ).fillna(0.0)
    df["edge_shape_change_w5"] = g["edge_shape_change_norm"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).mean()
    ).fillna(0.0)
    df["bbox_aspect_change_rate_w1s"] = g["bbox_aspect_change_rate_s"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).mean()
    ).fillna(0.0)
    df["bbox_size_change_rate_w1s"] = g["bbox_size_change_rate_s"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).mean()
    ).fillna(0.0)
    df["edge_shape_change_rate_w1s"] = g["edge_shape_change_rate_s"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).mean()
    ).fillna(0.0)
    df["bbox_shape_change_rate_w1s"] = (
        0.45 * df["bbox_aspect_change_rate_w1s"]
        + 0.25 * df["bbox_size_change_rate_w1s"]
        + 0.30 * df["edge_shape_change_rate_w1s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["bbox_diag_range_w1s"] = g["bbox_diag_log"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).max()
        - s.rolling(window=window_1s, min_periods=1).min()
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bbox_aspect_range_w1s"] = g["bbox_aspect_log"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).max()
        - s.rolling(window=window_1s, min_periods=1).min()
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bbox_diag_delta_log"] = g["bbox_diag_log"].diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bbox_aspect_delta_log"] = g["bbox_aspect_log"].diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bbox_diag_delta_prev"] = g["bbox_diag_delta_log"].shift(1).fillna(0.0)
    df["bbox_aspect_delta_prev"] = g["bbox_aspect_delta_log"].shift(1).fillna(0.0)
    df["bbox_diag_reversal_flag"] = (
        (df["bbox_diag_delta_log"] * df["bbox_diag_delta_prev"] < 0)
        & ((df["bbox_diag_delta_log"].abs() + df["bbox_diag_delta_prev"].abs()) > 1e-3)
    ).astype(float)
    df["bbox_aspect_reversal_flag"] = (
        (df["bbox_aspect_delta_log"] * df["bbox_aspect_delta_prev"] < 0)
        & ((df["bbox_aspect_delta_log"].abs() + df["bbox_aspect_delta_prev"].abs()) > 1e-3)
    ).astype(float)
    df["bbox_reversal_count_w1s"] = g["bbox_diag_reversal_flag"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).sum()
    ) + g["bbox_aspect_reversal_flag"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).sum()
    )
    reversal_boost = (1.0 + 0.35 * df["bbox_reversal_count_w1s"].clip(0.0, 2.0)).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1.0)
    df["bbox_pulse_score_w1s"] = (
        (0.55 * df["bbox_diag_range_w1s"] + 0.45 * df["bbox_aspect_range_w1s"])
        * reversal_boost
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["bbox_shape_change_w5"] = (
        0.45 * df["bbox_aspect_change_w5"]
        + 0.25 * df["bbox_size_change_w5"]
        + 0.30 * df["edge_shape_change_w5"]
    )
    speed_low_factor = 1.0 - robust_normalize(df["speed_smooth"])
    df["in_place_turn_score"] = (df["bbox_shape_change_w5"] * speed_low_factor).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    df["speed_delta"] = g["speed_smooth"].diff()
    df["speed_delta_abs"] = df["speed_delta"].abs().fillna(0.0)
    df["accel_abs_smooth"] = g["speed_delta_abs"].transform(
        lambda s: s.rolling(window=smooth_window, min_periods=1).median()
    ).fillna(0.0)

    df["heading_angle"] = np.arctan2(df["dy"], df["dx"])
    df["prev_heading_angle"] = g["heading_angle"].shift(1)
    df["turn_angle_abs"] = angle_diff_abs(df["heading_angle"], df["prev_heading_angle"]).fillna(0.0)
    df.loc[df["dist_px"].fillna(0) < 1e-6, "turn_angle_abs"] = 0.0
    df["turn_rate_w5"] = g["turn_angle_abs"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).mean()
    ).fillna(0.0)
    df["direction_change_count_w5"] = g["turn_angle_abs"].transform(
        lambda s: (s > (math.pi / 4)).rolling(window=feature_window, min_periods=1).sum()
    ).fillna(0.0)
    shape_rate_component = robust_normalize_open_upper(df["bbox_shape_change_rate_w1s"])
    shape_pulse_component = robust_normalize_open_upper(df["bbox_pulse_score_w1s"])
    in_place_component = robust_normalize_open_upper(df["in_place_turn_score"])
    df["sharp_turn_event_raw_score"] = np.maximum.reduce([
        shape_rate_component.to_numpy(dtype=float),
        shape_pulse_component.to_numpy(dtype=float),
        in_place_component.to_numpy(dtype=float),
    ])
    df["sharp_turn_event_raw_score"] = pd.Series(df["sharp_turn_event_raw_score"], index=df.index).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)
    df["sharp_turn_event_score"] = (
        df["sharp_turn_event_raw_score"] * df["overlap_reliability"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)
    df["sharp_turn_event_w1s"] = g["sharp_turn_event_score"].transform(
        lambda s: s.rolling(window=window_1s, min_periods=1).max()
    ).fillna(0.0)
    df["sharp_turn_event_w5"] = g["sharp_turn_event_score"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).max()
    ).fillna(0.0)

    df["speed_mean_w5"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).mean()
    )
    df["speed_std_w5"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=2).std()
    ).fillna(0.0)
    df["speed_max_w5"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).max()
    )
    df["speed_p75_w5"] = g["speed_smooth"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).quantile(0.75)
    )

    df["cx_lag_w"] = g["cx"].shift(feature_window - 1)
    df["cy_lag_w"] = g["cy"].shift(feature_window - 1)
    df["displacement_px_w5"] = np.sqrt((df["cx"] - df["cx_lag_w"]) ** 2 + (df["cy"] - df["cy_lag_w"]) ** 2)
    df["path_px_w5"] = g["dist_px"].transform(
        lambda s: s.rolling(window=feature_window, min_periods=1).sum()
    )
    df["displacement_w5"] = (df["displacement_px_w5"] / denom.replace(0, np.nan)).fillna(0.0)
    df["path_norm_w5"] = (df["path_px_w5"] / denom.replace(0, np.nan)).fillna(0.0)
    df["movement_ratio_w5"] = (df["path_px_w5"] / df["displacement_px_w5"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)
    df["movement_ratio_w5"] = df["movement_ratio_w5"].clip(1.0, 10.0)

    high_motion_thr = float(np.nanpercentile(df["speed_smooth"], 75)) if len(df) else 0.0
    frame_group = df.groupby("frame")
    df["fish_count_frame"] = frame_group["track_id"].transform("count")
    df["group_speed_mean"] = frame_group["speed_smooth"].transform("mean").fillna(0.0)
    df["group_speed_std"] = frame_group["speed_smooth"].transform("std").fillna(0.0)
    df["is_high_motion_tmp"] = (df["speed_smooth"] >= high_motion_thr).astype(float)
    df["group_high_motion_ratio"] = frame_group["is_high_motion_tmp"].transform("mean").fillna(0.0)

    df = add_neighbor_features(df)

    for col in DEFAULT_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["valid_motion"] = valid_gap.astype(int)
    df["track_len"] = g["frame"].transform("count")
    df["is_track_start"] = (g.cumcount() == 0).astype(int)
    df["is_track_end"] = (g.cumcount(ascending=False) == 0).astype(int)
    df["sample_weight"] = np.where(df["valid_motion"] == 1, 1.0, 0.5)
    return df


def add_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nearest = []
    crowd = []
    for _, frame_df in df.groupby("frame", sort=False):
        centers = frame_df[["cx", "cy"]].to_numpy(dtype=float)
        diags = frame_df["bbox_diag"].replace(0, np.nan).to_numpy(dtype=float)
        if len(frame_df) <= 1:
            nearest.extend([np.nan] * len(frame_df))
            crowd.extend([0.0] * len(frame_df))
            continue
        diff = centers[:, None, :] - centers[None, :, :]
        dmat = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dmat, np.inf)
        nn = np.min(dmat, axis=1)
        nn_norm = nn / np.nanmedian(diags)
        nearest.extend(nn_norm.tolist())
        crowd.extend((1.0 / np.maximum(nn_norm, 1e-6)).clip(0, 10).tolist())
    df["nearest_neighbor_dist_norm"] = pd.Series(nearest, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["crowding_index"] = pd.Series(crowd, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def compute_activity_score(
    df: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    norm_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = df.copy()
    weights = feature_weights or DEFAULT_SCORE_WEIGHTS
    if norm_stats is None:
        norm_stats = {}
        for feature in weights:
            s = df[feature].astype(float).replace([np.inf, -np.inf], np.nan)
            lo = float(np.nanpercentile(s, 10)) if not s.dropna().empty else 0.0
            hi = float(np.nanpercentile(s, 90)) if not s.dropna().empty else 1.0
            if not np.isfinite(hi) or hi <= lo:
                hi = lo + 1e-6
            norm_stats[feature] = {"p10": lo, "p90": hi}

    score = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for feature, weight in weights.items():
        if feature not in df.columns:
            continue
        stats = norm_stats.get(feature, {"p10": 0.0, "p90": 1.0})
        lo = float(stats["p10"])
        hi = float(stats["p90"])
        if hi <= lo:
            hi = lo + 1e-6
        normed = ((df[feature].astype(float).fillna(lo) - lo) / (hi - lo)).clip(0.0, 1.0)
        score += float(weight) * normed
        total_weight += float(weight)
    if total_weight <= 0:
        total_weight = 1.0
    df["state_score"] = (score / total_weight).clip(0.0, 1.0)
    return df, norm_stats


def assign_labels_from_score(
    df: pd.DataFrame,
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
    label_source: str = "auto_motion_rule",
    label_version: str = "v2_multifeature",
) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()
    scores = df["state_score"].dropna().to_numpy()
    if len(scores) == 0:
        low_thr = high_thr = 0.0
    else:
        low_thr = float(np.percentile(scores, low_percentile))
        high_thr = float(np.percentile(scores, high_percentile))

    df["state_id"] = 1
    df.loc[df["state_score"] <= low_thr, "state_id"] = 0
    df.loc[df["state_score"] >= high_thr, "state_id"] = 2
    df["state_id"] = df["state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    df["label_source"] = label_source
    df["label_version"] = label_version
    return df, low_thr, high_thr


def infer_window_rows(df: pd.DataFrame, fps: float, window_sec: float) -> int:
    gaps = df.get("frame_gap", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    gaps = gaps[gaps > 0].dropna()
    median_gap = float(gaps.median()) if not gaps.empty else 1.0
    if not np.isfinite(median_gap) or median_gap <= 0:
        median_gap = 1.0
    sampled_fps = float(fps) / median_gap
    return max(1, int(round(float(window_sec) * sampled_fps)))


def hold_event_forward(flags: pd.Series, hold_rows: int) -> pd.Series:
    hold_rows = max(1, int(hold_rows))
    held = flags.astype(float).rolling(window=hold_rows, min_periods=1).max()
    return held.reindex(flags.index).fillna(0.0)


def backfill_confirmed_flags(flags: pd.Series, lookback_rows: int) -> pd.Series:
    lookback_rows = max(1, int(lookback_rows))
    values = flags.astype(float).iloc[::-1]
    held = values.rolling(window=lookback_rows, min_periods=1).max().iloc[::-1]
    return held.reindex(flags.index).fillna(0.0)


def assign_persistent_labels_from_score(
    df: pd.DataFrame,
    fps: float,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
    window_sec: float = 3.0,
    low_required_ratio: float = 0.8,
    high_required_ratio: float = 0.6,
    high_event_percentile: float = 92.0,
    high_event_hold_sec: float = 1.0,
    backfill_persistent_states: bool = True,
    label_source: str = "auto_persistent_motion_rule",
    label_version: str = "v9_event_below_low_overlap_suppressed",
) -> tuple[pd.DataFrame, float, float]:
    """Assign activity labels only when low/high motion persists inside a time window."""
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)
    scores = df["state_score"].dropna().to_numpy()
    if len(scores) == 0:
        low_thr = high_thr = 0.0
    else:
        low_thr = float(np.percentile(scores, low_percentile))
        high_thr = float(np.percentile(scores, high_percentile))

    window_rows = infer_window_rows(df, fps=fps, window_sec=window_sec)
    df["instant_low_flag"] = (df["state_score"] <= low_thr).astype(float)
    df["instant_high_flag"] = (df["state_score"] >= high_thr).astype(float)
    event_col = "sharp_turn_event_w5"
    if event_col in df.columns and not df[event_col].dropna().empty:
        high_event_thr = float(np.percentile(df[event_col].dropna().to_numpy(), high_event_percentile))
    else:
        high_event_thr = float("inf")
    event_hold_rows = infer_window_rows(df, fps=fps, window_sec=high_event_hold_sec)

    parts = []
    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame").copy()
        group["low_persistence_ratio"] = group["instant_low_flag"].rolling(
            window=window_rows,
            min_periods=window_rows,
        ).mean().fillna(0.0)
        group["high_persistence_ratio"] = group["instant_high_flag"].rolling(
            window=window_rows,
            min_periods=window_rows,
        ).mean().fillna(0.0)
        group["low_persistent_confirmed_flag"] = (
            group["low_persistence_ratio"] >= low_required_ratio
        ).astype(float)
        group["high_persistent_confirmed_flag"] = (
            group["high_persistence_ratio"] >= high_required_ratio
        ).astype(float)
        if backfill_persistent_states:
            group["low_persistent_backfill_flag"] = backfill_confirmed_flags(
                group["low_persistent_confirmed_flag"],
                window_rows,
            )
            group["high_persistent_backfill_flag"] = backfill_confirmed_flags(
                group["high_persistent_confirmed_flag"],
                window_rows,
            )
        else:
            group["low_persistent_backfill_flag"] = group["low_persistent_confirmed_flag"]
            group["high_persistent_backfill_flag"] = group["high_persistent_confirmed_flag"]
        group["high_event_raw_flag"] = (group.get(event_col, pd.Series(0.0, index=group.index)) >= high_event_thr).astype(float)
        group["high_event_hold_flag"] = hold_event_forward(group["high_event_raw_flag"], event_hold_rows)
        parts.append(group)
    df = pd.concat(parts, ignore_index=True).sort_values(["frame", "track_id"]).reset_index(drop=True)

    df["state_id"] = 1
    df.loc[df["high_event_hold_flag"] >= 1.0, "state_id"] = 2
    df.loc[df["low_persistent_backfill_flag"] >= 1.0, "state_id"] = 0
    df.loc[df["high_persistent_backfill_flag"] >= 1.0, "state_id"] = 2
    conflict = (
        (df["low_persistent_backfill_flag"] >= 1.0)
        & (df["high_persistent_backfill_flag"] >= 1.0)
    )
    df.loc[conflict, "state_id"] = np.where(
        df.loc[conflict, "high_persistence_ratio"] >= df.loc[conflict, "low_persistence_ratio"],
        2,
        0,
    )
    df["state_id"] = df["state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    df["label_window_sec"] = float(window_sec)
    df["label_window_rows"] = int(window_rows)
    df["label_low_required_ratio"] = float(low_required_ratio)
    df["label_high_required_ratio"] = float(high_required_ratio)
    df["label_backfill_persistent_states"] = bool(backfill_persistent_states)
    df["label_high_event_percentile"] = float(high_event_percentile)
    df["label_high_event_score_threshold"] = float(high_event_thr)
    df["label_high_event_hold_sec"] = float(high_event_hold_sec)
    df["label_high_event_hold_rows"] = int(event_hold_rows)
    df["label_confidence"] = np.where(
        df["state_id"] == 0,
        df["low_persistence_ratio"],
        np.where(
            df["state_id"] == 2,
            df["high_persistence_ratio"],
            1.0 - np.maximum(df["low_persistence_ratio"], df["high_persistence_ratio"]),
        ),
    )
    df["label_source"] = label_source
    df["label_version"] = label_version
    return df, low_thr, high_thr


def split_frames_temporal(
    frames: Iterable[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> tuple[set[int], set[int], set[int]]:
    frames = sorted(int(f) for f in frames)
    n = len(frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = set(frames[:n_train])
    val = set(frames[n_train:n_train + n_val])
    test = set(frames[n_train + n_val:])
    return train, val, test


def apply_temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> pd.DataFrame:
    df = df.copy()
    train, val, test = split_frames_temporal(df["frame"].unique(), train_ratio, val_ratio)

    def split_name(frame: int) -> str:
        if frame in train:
            return "train"
        if frame in val:
            return "val"
        return "test"

    df["split"] = df["frame"].astype(int).map(split_name)
    return df


def load_activity_model(path: Path) -> dict:
    data = pd.io.common.stringify_path(path)
    import json

    with open(data, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_persistent_state_postprocess(df: pd.DataFrame, model: dict, fps: float | None = None) -> pd.DataFrame:
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)
    post = model.get("postprocess", {})
    if post.get("mode") != "persistent":
        return df

    if fps is not None and "window_sec" in post:
        window_rows = infer_window_rows(df, fps=fps, window_sec=float(post.get("window_sec", 3.0)))
    else:
        window_rows = int(post.get("window_rows", 1))
    window_rows = max(1, window_rows)
    low_required_ratio = float(post.get("low_required_ratio", 0.8))
    high_required_ratio = float(post.get("high_required_ratio", 0.6))
    high_event_col = str(post.get("high_event_col", "sharp_turn_event_w5"))
    high_event_threshold = post.get("high_event_threshold")
    high_event_hold_rows = int(post.get("high_event_hold_rows", max(1, window_rows // 3)))
    backfill_persistent_states = bool(post.get("backfill_persistent_states", True))

    df["instant_low_flag"] = (df["state_id"] == 0).astype(float)
    df["instant_high_flag"] = (df["state_id"] == 2).astype(float)
    parts = []
    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame").copy()
        group["low_persistence_ratio"] = group["instant_low_flag"].rolling(
            window=window_rows,
            min_periods=window_rows,
        ).mean().fillna(0.0)
        group["high_persistence_ratio"] = group["instant_high_flag"].rolling(
            window=window_rows,
            min_periods=window_rows,
        ).mean().fillna(0.0)
        group["low_persistent_confirmed_flag"] = (
            group["low_persistence_ratio"] >= low_required_ratio
        ).astype(float)
        group["high_persistent_confirmed_flag"] = (
            group["high_persistence_ratio"] >= high_required_ratio
        ).astype(float)
        if backfill_persistent_states:
            group["low_persistent_backfill_flag"] = backfill_confirmed_flags(
                group["low_persistent_confirmed_flag"],
                window_rows,
            )
            group["high_persistent_backfill_flag"] = backfill_confirmed_flags(
                group["high_persistent_confirmed_flag"],
                window_rows,
            )
        else:
            group["low_persistent_backfill_flag"] = group["low_persistent_confirmed_flag"]
            group["high_persistent_backfill_flag"] = group["high_persistent_confirmed_flag"]
        if high_event_threshold is not None and high_event_col in group.columns:
            group["high_event_raw_flag"] = (group[high_event_col] >= float(high_event_threshold)).astype(float)
            group["high_event_hold_flag"] = hold_event_forward(group["high_event_raw_flag"], high_event_hold_rows)
        else:
            group["high_event_raw_flag"] = 0.0
            group["high_event_hold_flag"] = 0.0
        parts.append(group)
    df = pd.concat(parts, ignore_index=True).sort_values(["frame", "track_id"]).reset_index(drop=True)

    df["state_id"] = 1
    df.loc[df["high_event_hold_flag"] >= 1.0, "state_id"] = 2
    df.loc[df["low_persistent_backfill_flag"] >= 1.0, "state_id"] = 0
    df.loc[df["high_persistent_backfill_flag"] >= 1.0, "state_id"] = 2
    conflict = (
        (df["low_persistent_backfill_flag"] >= 1.0)
        & (df["high_persistent_backfill_flag"] >= 1.0)
    )
    df.loc[conflict, "state_id"] = np.where(
        df.loc[conflict, "high_persistence_ratio"] >= df.loc[conflict, "low_persistence_ratio"],
        2,
        0,
    )
    df["state_id"] = df["state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    return df


def predict_with_activity_model(df: pd.DataFrame, model: dict, fps: float | None = None) -> pd.DataFrame:
    df = df.copy()
    weights = {str(k): float(v) for k, v in model["feature_weights"].items()}
    norm_stats = model.get("normalization", {})
    df, _ = compute_activity_score(df, feature_weights=weights, norm_stats=norm_stats)
    low_thr = float(model["low_score_threshold"])
    high_thr = float(model["high_score_threshold"])
    df["state_id"] = 1
    df.loc[df["state_score"] <= low_thr, "state_id"] = 0
    df.loc[df["state_score"] >= high_thr, "state_id"] = 2
    df["state_id"] = df["state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    df = apply_persistent_state_postprocess(df, model=model, fps=fps)
    return df
