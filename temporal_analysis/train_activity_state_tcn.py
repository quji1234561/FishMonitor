#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a lightweight TCN for fish activity-state recognition."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_analysis.activity_classifier import add_classifier_context_features, prepare_feature_matrix  # noqa: E402
from temporal_analysis.activity_tcn import (  # noqa: E402
    DEFAULT_TCN_FEATURES,
    apply_scaler,
    build_centered_windows,
    build_tcn_model,
    make_scaler,
    require_torch,
)
from temporal_analysis.train_activity_state_classifier import (  # noqa: E402
    evaluate,
    parse_float_list,
    parse_int_list,
    rolling_majority_by_track,
)


STATE_ID_TO_NAME = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN activity-state model")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--window-size", type=int, default=75, help="Odd sequence window length, e.g. 75 for 3s at 25fps")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--class-weight-mode", choices=["none", "balanced", "sqrt_balanced"], default="sqrt_balanced")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--post-windows", default="1,15,30")
    parser.add_argument("--min-vote-ratios", default="0.5,0.6")
    parser.add_argument("--non-normal-min-probs", default="0,0.25,0.35,0.45,0.55")
    parser.add_argument("--min-val-non-normal-ratio", type=float, default=0.06)
    parser.add_argument("--min-val-low-ratio", type=float, default=0.005)
    parser.add_argument("--min-val-high-ratio", type=float, default=0.03)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_feature_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_features(args: argparse.Namespace, feature_config: dict) -> list[str]:
    if args.features:
        return [x.strip() for x in args.features.split(",") if x.strip()]
    configured = [str(x) for x in feature_config.get("model_features", [])]
    features = []
    for feature in [*configured, *DEFAULT_TCN_FEATURES]:
        if feature not in features:
            features.append(feature)
    return features


def load_dataset(path: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "state_id" not in df.columns:
        raise ValueError(f"{path} missing state_id")
    for col in ["frame", "track_id"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing {col}")
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0.0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["state_id"] = df["state_id"].astype(int)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def make_class_weights(y: np.ndarray, mode: str):
    torch, _ = require_torch()
    if mode == "none":
        return None
    counts = np.bincount(y.astype(int), minlength=3).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (3.0 * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def predict_logits(model, windows: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    logits_all = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            logits = model(batch).cpu().numpy()
            logits_all.append(logits)
    return np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 3), dtype=np.float32)


def apply_probability_decision(probs: np.ndarray, non_normal_min_prob: float) -> np.ndarray:
    pred = probs.argmax(axis=1).astype(np.int64)
    if non_normal_min_prob <= 0:
        return pred
    adjusted = pred.copy()
    for i, state_id in enumerate(pred):
        if int(state_id) == 1:
            continue
        if probs[i, int(state_id)] < non_normal_min_prob:
            adjusted[i] = 1
    return adjusted


def passes_constraints(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("pred_non_normal_ratio", 0.0)) >= args.min_val_non_normal_ratio
        and float(metrics.get("pred_low_ratio", 0.0)) >= args.min_val_low_ratio
        and float(metrics.get("pred_high_ratio", 0.0)) >= args.min_val_high_ratio
    )


def search_decision_and_postprocess(
    logits_train: np.ndarray,
    logits_val: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict | None, list[dict]]:
    torch, _ = require_torch()
    probs_train = torch.softmax(torch.from_numpy(logits_train), dim=1).numpy()
    probs_val = torch.softmax(torch.from_numpy(logits_val), dim=1).numpy()
    rows = []
    best = None
    for non_normal_min_prob in parse_float_list(args.non_normal_min_probs):
        raw_train = apply_probability_decision(probs_train, non_normal_min_prob)
        raw_val = apply_probability_decision(probs_val, non_normal_min_prob)
        for window in parse_int_list(args.post_windows):
            for min_vote_ratio in parse_float_list(args.min_vote_ratios):
                pred_train = rolling_majority_by_track(train_df, raw_train, window, min_vote_ratio)
                pred_val = rolling_majority_by_track(val_df, raw_val, window, min_vote_ratio)
                train_metrics = evaluate(y_train, pred_train)
                val_metrics = evaluate(y_val, pred_val)
                row = {
                    "non_normal_min_prob": float(non_normal_min_prob),
                    "post_window": int(window),
                    "min_vote_ratio": float(min_vote_ratio),
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "train_pred_low_ratio": train_metrics["pred_low_ratio"],
                    "train_pred_high_ratio": train_metrics["pred_high_ratio"],
                    "train_pred_non_normal_ratio": train_metrics["pred_non_normal_ratio"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_pred_low_ratio": val_metrics["pred_low_ratio"],
                    "val_pred_high_ratio": val_metrics["pred_high_ratio"],
                    "val_pred_non_normal_ratio": val_metrics["pred_non_normal_ratio"],
                    "passes_selection_constraints": passes_constraints(val_metrics, args),
                }
                rows.append(row)
                if not row["passes_selection_constraints"]:
                    continue
                score = val_metrics[args.metric]
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "row": row,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    }
    return best, rows


def main() -> None:
    args = parse_args()
    torch, nn = require_torch()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd, e.g. 75")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_config = load_feature_config(args.feature_config)
    features = resolve_features(args, feature_config)

    train_df = load_dataset(args.train_csv, features)
    val_df = load_dataset(args.val_csv, features)
    label_thresholds = feature_config.get("label_thresholds", {})
    train_df, feature_context = add_classifier_context_features(
        train_df,
        fit=True,
        feature_weights=feature_config.get("feature_weights"),
        low_percentile=float(label_thresholds.get("low_percentile", 20.0)),
        high_percentile=float(label_thresholds.get("high_percentile", 80.0)),
        fps=args.fps,
    )
    val_df, _ = add_classifier_context_features(val_df, context=feature_context, fit=False, fps=args.fps)

    x_train_base = prepare_feature_matrix(train_df, features)
    scaler = make_scaler(x_train_base)
    x_train = apply_scaler(x_train_base, scaler)
    x_val = apply_scaler(prepare_feature_matrix(val_df, features), scaler)
    train_windows, train_row_indices = build_centered_windows(train_df, x_train, args.window_size)
    val_windows, val_row_indices = build_centered_windows(val_df, x_val, args.window_size)
    y_train_all = train_df["state_id"].to_numpy(dtype=np.int64)
    y_val_all = val_df["state_id"].to_numpy(dtype=np.int64)
    y_train = y_train_all[train_row_indices]
    y_val = y_val_all[val_row_indices]

    model = build_tcn_model(
        input_dim=len(features),
        channels=args.channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    class_weights = make_class_weights(y_train, args.class_weight_mode)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log(f"[INFO] device: {device}")
    log(f"[INFO] train windows: {len(train_windows)}, val windows: {len(val_windows)}, features: {len(features)}")
    log(f"[INFO] model: TCN channels={args.channels}, levels={args.levels}, kernel={args.kernel_size}, window={args.window_size}")
    log(
        "[INFO] selection constraints: min_non_normal=%.3f, min_low=%.3f, min_high=%.3f"
        % (args.min_val_non_normal_ratio, args.min_val_low_ratio, args.min_val_high_ratio)
    )

    indices = np.arange(len(train_windows))
    rows = []
    best = None
    last_bundle = None
    no_improve = 0
    start_all = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(indices)
        losses = []
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            xb = torch.from_numpy(train_windows[batch_idx]).to(device)
            yb = torch.from_numpy(y_train[batch_idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_logits = predict_logits(model, train_windows, args.batch_size, device)
        val_logits = predict_logits(model, val_windows, args.batch_size, device)
        epoch_best, epoch_rows = search_decision_and_postprocess(
            train_logits,
            val_logits,
            train_df.iloc[train_row_indices].reset_index(drop=True),
            val_df.iloc[val_row_indices].reset_index(drop=True),
            y_train,
            y_val,
            args,
        )
        for row in epoch_rows:
            row["epoch"] = epoch
            row["loss"] = float(np.mean(losses)) if losses else 0.0
        rows.extend(epoch_rows)

        if epoch_best is None:
            log(f"[EPOCH {epoch:03d}] loss={np.mean(losses):.4f} no candidate passed constraints")
            no_improve += 1
        else:
            score = epoch_best["score"]
            row = epoch_best["row"]
            log(
                "[EPOCH %03d] loss=%.4f best_%s=%.4f val_acc=%.4f val_non_normal=%.3f low=%.3f high=%.3f"
                % (
                    epoch,
                    np.mean(losses),
                    args.metric,
                    score,
                    row["val_accuracy"],
                    row["val_pred_non_normal_ratio"],
                    row["val_pred_low_ratio"],
                    row["val_pred_high_ratio"],
                )
            )
            bundle = {
                "model_type": "torch_activity_tcn_v1",
                "model_name": "activity_tcn",
                "label_version": str(feature_config.get("label_version", "")),
                "state_names": STATE_ID_TO_NAME,
                "model_features": features,
                "feature_context": feature_context,
                "scaler": scaler,
                "window_size": int(args.window_size),
                "model_config": {
                    "channels": int(args.channels),
                    "levels": int(args.levels),
                    "kernel_size": int(args.kernel_size),
                    "dropout": float(args.dropout),
                },
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "decision": {"non_normal_min_prob": float(row["non_normal_min_prob"])},
                "postprocess": {
                    "mode": "rolling_majority",
                    "window": int(row["post_window"]),
                    "min_vote_ratio": float(row["min_vote_ratio"]),
                },
                "train_metrics": epoch_best["train_metrics"],
                "val_metrics": epoch_best["val_metrics"],
                "source": {
                    "train_csv": args.train_csv,
                    "val_csv": args.val_csv,
                    "feature_config": args.feature_config or "",
                    "selection_metric": args.metric,
                },
            }
            last_bundle = bundle
            if best is None or score > best["score"]:
                best = {"score": score, "bundle": bundle, "row": row, "epoch": epoch}
                torch.save(bundle, out_dir / "best_activity_tcn.pt")
                log(
                    "[BEST] epoch=%d %s=%.4f acc=%.4f non_normal_min_prob=%.2f post_window=%d"
                    % (epoch, args.metric, score, row["val_accuracy"], row["non_normal_min_prob"], row["post_window"])
                )
                no_improve = 0
            else:
                no_improve += 1

        if last_bundle is not None:
            torch.save(last_bundle, out_dir / "last_activity_tcn.pt")
        if no_improve >= args.patience:
            log(f"[STOP] no improvement for {args.patience} epochs")
            break

    pd.DataFrame(rows).to_csv(out_dir / "tcn_train_log.csv", index=False, encoding="utf-8-sig")
    if best is None:
        raise RuntimeError("No TCN epoch satisfied output-ratio constraints. Lower min-val ratio constraints.")
    metrics = {
        "selection_metric": args.metric,
        "best_score": best["score"],
        "best_epoch": best["epoch"],
        "best_row": best["row"],
        "best_train_metrics": best["bundle"]["train_metrics"],
        "best_val_metrics": best["bundle"]["val_metrics"],
        "elapsed_sec": time.perf_counter() - start_all,
        "state_names": STATE_ID_TO_NAME,
    }
    (out_dir / "best_tcn_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[OK] best TCN: {out_dir / 'best_activity_tcn.pt'}")
    log(f"[OK] last TCN: {out_dir / 'last_activity_tcn.pt'}")
    log(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
