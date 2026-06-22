#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TCN helpers for fish activity-state recognition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from temporal_analysis.activity_classifier import (
    DEFAULT_CLASSIFIER_FEATURES,
    add_classifier_context_features,
    prepare_feature_matrix,
)
from temporal_analysis.motion_features import STATE_ID_TO_NAME


DEFAULT_TCN_FEATURES = DEFAULT_CLASSIFIER_FEATURES


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 PyTorch。请先安装 torch 后再训练或推理 TCN，例如：\n"
            "  pip install torch\n"
            "如果你用 pixi，请在项目环境中安装 pytorch。"
        ) from exc
    return torch, nn


def make_scaler(x: np.ndarray) -> dict:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def build_centered_windows(
    df: pd.DataFrame,
    x: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("--window-size must be an odd integer >= 3")
    radius = window_size // 2
    windows = []
    row_indices = []
    for _, group in df.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        idx = group.index.to_numpy(dtype=int)
        seq = x[idx]
        if len(seq) == 0:
            continue
        padded = np.pad(seq, ((radius, radius), (0, 0)), mode="edge")
        for i, row_idx in enumerate(idx):
            windows.append(padded[i:i + window_size])
            row_indices.append(row_idx)
    return np.asarray(windows, dtype=np.float32), np.asarray(row_indices, dtype=int)


def build_tcn_model(input_dim: int, channels: int = 64, levels: int = 4, kernel_size: int = 5, dropout: float = 0.2):
    torch, nn = require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int):
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            return self.net(x) + self.proj(x)

    class ActivityTCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = input_dim
            for level in range(levels):
                blocks.append(ResidualBlock(in_channels, channels, dilation=2 ** level))
                in_channels = channels
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels, 3),
            )

        def forward(self, x):
            # x: batch, window, features
            x = x.transpose(1, 2)
            y = self.tcn(x)
            center = y[:, :, y.shape[-1] // 2]
            return self.head(center)

    return ActivityTCN()


def load_activity_tcn(path: Path, map_location: str = "cpu") -> dict:
    torch, _ = require_torch()
    return torch.load(str(path), map_location=map_location, weights_only=False)


def predict_with_activity_tcn(
    df: pd.DataFrame,
    bundle: dict,
    batch_size: int = 512,
    device: str = "cpu",
) -> pd.DataFrame:
    torch, _ = require_torch()
    df = df.copy().sort_values(["track_id", "frame"]).reset_index(drop=True)
    features = [str(x) for x in bundle["model_features"]]
    df, _ = add_classifier_context_features(df, context=bundle.get("feature_context", {}), fit=False)
    x = prepare_feature_matrix(df, features)
    x = apply_scaler(x, bundle["scaler"])
    windows, row_indices = build_centered_windows(df, x, int(bundle["window_size"]))

    model_cfg = bundle["model_config"]
    model = build_tcn_model(
        input_dim=len(features),
        channels=int(model_cfg.get("channels", 64)),
        levels=int(model_cfg.get("levels", 4)),
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    probs_all = np.zeros((len(df), 3), dtype=np.float32)
    preds_all = np.ones(len(df), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1).astype(np.int64)
            rows = row_indices[start:start + len(pred)]
            probs_all[rows] = probs
            preds_all[rows] = pred

    decision = bundle.get("decision", {})
    non_normal_min_prob = float(decision.get("non_normal_min_prob", 0.0))
    adjusted = preds_all.copy()
    if non_normal_min_prob > 0:
        for i, state_id in enumerate(preds_all):
            if int(state_id) == 1:
                continue
            if probs_all[i, int(state_id)] < non_normal_min_prob:
                adjusted[i] = 1

    df["tcn_state_id"] = preds_all.astype(int)
    df["tcn_state_name"] = pd.Series(preds_all, index=df.index).map(STATE_ID_TO_NAME)
    df["prob_low_activity"] = probs_all[:, 0]
    df["prob_normal_activity"] = probs_all[:, 1]
    df["prob_high_activity"] = probs_all[:, 2]
    df["tcn_confidence"] = probs_all.max(axis=1)
    df["tcn_adjusted_state_id"] = adjusted.astype(int)
    df["tcn_adjusted_state_name"] = pd.Series(adjusted, index=df.index).map(STATE_ID_TO_NAME)
    df["state_id"] = df["tcn_adjusted_state_id"].astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    return df
