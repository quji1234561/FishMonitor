#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a supervised activity-state classifier from YOLO-box motion features."""

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

from temporal_analysis.activity_classifier import (  # noqa: E402
    DEFAULT_CLASSIFIER_FEATURES,
    add_classifier_context_features,
    prepare_feature_matrix,
    save_activity_classifier,
)
from temporal_analysis.train_activity_state_model import confusion_matrix, macro_f1_score  # noqa: E402


STATE_ID_TO_NAME = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised activity-state classifier")
    parser.add_argument("--train-csv", required=True, help="YOLO-box activity train CSV")
    parser.add_argument("--val-csv", required=True, help="YOLO-box activity val CSV")
    parser.add_argument("--out-dir", required=True, help="Output model directory")
    parser.add_argument("--feature-config", default=None, help="feature_config.json from generated activity labels")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS used to build derived temporal classifier features")
    parser.add_argument("--features", default=None, help="Optional comma-separated feature list")
    parser.add_argument("--post-windows", default="1,5,15,30,75", help="Rolling majority windows searched on validation")
    parser.add_argument("--min-vote-ratios", default="0.5,0.6,0.7", help="Rolling majority vote ratios")
    parser.add_argument(
        "--non-normal-min-probs",
        default="0,0.45,0.55,0.65,0.75",
        help="If low/high probability is below this value, convert it to normal. Searched for accuracy.",
    )
    parser.add_argument(
        "--estimator-verbose",
        type=int,
        default=1,
        help="Verbosity passed to sklearn tree models. Use 0 to silence tree-building logs.",
    )
    parser.add_argument(
        "--advanced-models",
        action="store_true",
        help="Also try stronger optional models: MLP plus installed XGBoost/LightGBM/CatBoost.",
    )
    parser.add_argument(
        "--min-val-non-normal-ratio",
        type=float,
        default=0.0,
        help="Require validation predictions to contain at least this low+high ratio when selecting the best model.",
    )
    parser.add_argument(
        "--min-val-low-ratio",
        type=float,
        default=0.0,
        help="Require validation predictions to contain at least this low ratio when selecting the best model.",
    )
    parser.add_argument(
        "--min-val-high-ratio",
        type=float,
        default=0.0,
        help="Require validation predictions to contain at least this high ratio when selecting the best model.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 scikit-learn。请先安装后再训练分类器，例如：\n"
            "  pixi add scikit-learn\n"
            "或：\n"
            "  pip install scikit-learn\n"
        ) from exc
    return ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, MLPClassifier, make_pipeline, StandardScaler


def load_feature_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(float(x.strip())) for x in value.split(",") if x.strip()]


def resolve_features(args: argparse.Namespace, feature_config: dict) -> list[str]:
    if args.features:
        return [x.strip() for x in args.features.split(",") if x.strip()]
    configured = [str(x) for x in feature_config.get("model_features", [])]
    features = []
    for feature in [*configured, *DEFAULT_CLASSIFIER_FEATURES]:
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


def rolling_majority_by_track(df: pd.DataFrame, raw_pred: np.ndarray, window: int, min_vote_ratio: float) -> np.ndarray:
    if window <= 1:
        return raw_pred.astype(int)
    work = df[["track_id", "frame"]].copy()
    work["_pred"] = raw_pred.astype(int)
    out = pd.Series(index=work.index, dtype=int)
    for _, group in work.groupby("track_id", sort=False):
        group = group.sort_values("frame")
        values = group["_pred"].astype(int).tolist()
        smoothed: list[int] = []
        for i, value in enumerate(values):
            start = max(0, i - window + 1)
            window_values = values[start:i + 1]
            counts = pd.Series(window_values).value_counts()
            best_state = int(counts.index[0])
            best_count = int(counts.iloc[0])
            if best_count / len(window_values) >= min_vote_ratio:
                smoothed.append(best_state)
            else:
                smoothed.append(int(value))
        out.loc[group.index] = smoothed
    return out.sort_index().to_numpy(dtype=int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    low_ratio = float((y_pred == 0).mean())
    normal_ratio = float((y_pred == 1).mean())
    high_ratio = float((y_pred == 2).mean())
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": macro_f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "pred_low_ratio": low_ratio,
        "pred_normal_ratio": normal_ratio,
        "pred_high_ratio": high_ratio,
        "pred_non_normal_ratio": low_ratio + high_ratio,
    }


def passes_selection_constraints(metrics: dict, args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("pred_non_normal_ratio", 0.0)) >= float(args.min_val_non_normal_ratio)
        and float(metrics.get("pred_low_ratio", 0.0)) >= float(args.min_val_low_ratio)
        and float(metrics.get("pred_high_ratio", 0.0)) >= float(args.min_val_high_ratio)
    )


def predict_with_decision(estimator, x: np.ndarray, non_normal_min_prob: float) -> np.ndarray:
    pred = estimator.predict(x).astype(int)
    if non_normal_min_prob <= 0 or not hasattr(estimator, "predict_proba"):
        return pred
    proba = estimator.predict_proba(x)
    classes = [int(c) for c in getattr(estimator, "classes_", [])]
    adjusted = pred.copy()
    for i, state_id in enumerate(pred):
        state_id = int(state_id)
        if state_id == 1 or state_id not in classes:
            continue
        state_prob = float(proba[i, classes.index(state_id)])
        if state_prob < non_normal_min_prob:
            adjusted[i] = 1
    return adjusted


def add_optional_boosters(candidates: dict, random_state: int, verbose: int = 1) -> dict:
    try:
        from xgboost import XGBClassifier
        candidates["xgboost_hist"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=max(0, min(int(verbose), 2)),
        )
        log("[INFO] optional model enabled: xgboost_hist")
    except ModuleNotFoundError:
        log("[INFO] optional model skipped: xgboost is not installed")

    try:
        from lightgbm import LGBMClassifier
        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=700,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1 if verbose == 0 else 1,
        )
        log("[INFO] optional model enabled: lightgbm")
    except ModuleNotFoundError:
        log("[INFO] optional model skipped: lightgbm is not installed")

    try:
        from catboost import CatBoostClassifier
        candidates["catboost"] = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.04,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=bool(verbose),
            allow_writing_files=False,
        )
        log("[INFO] optional model enabled: catboost")
    except ModuleNotFoundError:
        log("[INFO] optional model skipped: catboost is not installed")
    return candidates


def build_candidates(random_state: int, verbose: int = 1, advanced_models: bool = False):
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, MLPClassifier, make_pipeline, StandardScaler = require_sklearn()
    candidates = {
        "extra_trees_acc": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=None,
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose,
        ),
        "random_forest_acc": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=None,
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            verbose=verbose,
        ),
        "hist_gradient_boosting": make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=300,
                l2_regularization=0.02,
                random_state=random_state,
            ),
        ),
    }
    if advanced_models:
        candidates["mlp_128_64"] = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=250,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=random_state,
                verbose=bool(verbose),
            ),
        )
        candidates = add_optional_boosters(candidates, random_state=random_state, verbose=verbose)
    return candidates


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_config = load_feature_config(args.feature_config)
    features = resolve_features(args, feature_config)
    post_windows = parse_int_list(args.post_windows)
    min_vote_ratios = parse_float_list(args.min_vote_ratios)
    non_normal_min_probs = parse_float_list(args.non_normal_min_probs)

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
    val_df, _ = add_classifier_context_features(
        val_df,
        context=feature_context,
        fit=False,
        fps=args.fps,
    )
    x_train = prepare_feature_matrix(train_df, features)
    x_val = prepare_feature_matrix(val_df, features)
    y_train = train_df["state_id"].to_numpy(dtype=int)
    y_val = val_df["state_id"].to_numpy(dtype=int)

    candidates = build_candidates(
        args.random_state,
        verbose=args.estimator_verbose,
        advanced_models=bool(args.advanced_models),
    )
    rows = []
    best = None
    last_bundle = None
    total_grid = len(non_normal_min_probs) * len(post_windows) * len(min_vote_ratios)

    log("[INFO] train rows: %d, val rows: %d, features: %d" % (len(train_df), len(val_df), len(features)))
    log("[INFO] candidates: %d, decision/postprocess combinations per candidate: %d" % (len(candidates), total_grid))
    log("[INFO] selection metric: %s" % args.metric)
    log(
        "[INFO] selection constraints: min_non_normal=%.3f, min_low=%.3f, min_high=%.3f"
        % (args.min_val_non_normal_ratio, args.min_val_low_ratio, args.min_val_high_ratio)
    )

    for candidate_idx, (name, estimator) in enumerate(candidates.items(), start=1):
        start = time.perf_counter()
        log("[TRAIN] %d/%d fitting candidate: %s" % (candidate_idx, len(candidates), name))
        estimator.fit(x_train, y_train)
        fit_elapsed = time.perf_counter() - start
        log("[TRAIN] %s fit done in %.1fs; searching thresholds/windows..." % (name, fit_elapsed))
        candidate_best = None
        for non_normal_min_prob in non_normal_min_probs:
            raw_train = predict_with_decision(estimator, x_train, non_normal_min_prob)
            raw_val = predict_with_decision(estimator, x_val, non_normal_min_prob)
            for window in post_windows:
                for min_vote_ratio in min_vote_ratios:
                    pred_train = rolling_majority_by_track(train_df, raw_train, window, min_vote_ratio)
                    pred_val = rolling_majority_by_track(val_df, raw_val, window, min_vote_ratio)
                    train_metrics = evaluate(y_train, pred_train)
                    val_metrics = evaluate(y_val, pred_val)
                    row = {
                        "candidate": name,
                        "non_normal_min_prob": float(non_normal_min_prob),
                        "post_window": int(window),
                        "min_vote_ratio": float(min_vote_ratio),
                        "train_accuracy": train_metrics["accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "train_pred_low_ratio": train_metrics["pred_low_ratio"],
                        "train_pred_normal_ratio": train_metrics["pred_normal_ratio"],
                        "train_pred_high_ratio": train_metrics["pred_high_ratio"],
                        "train_pred_non_normal_ratio": train_metrics["pred_non_normal_ratio"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_pred_low_ratio": val_metrics["pred_low_ratio"],
                        "val_pred_normal_ratio": val_metrics["pred_normal_ratio"],
                        "val_pred_high_ratio": val_metrics["pred_high_ratio"],
                        "val_pred_non_normal_ratio": val_metrics["pred_non_normal_ratio"],
                        "passes_selection_constraints": passes_selection_constraints(val_metrics, args),
                    }
                    rows.append(row)
                    bundle = {
                        "model_type": "sklearn_activity_classifier_v1",
                        "model_name": name,
                        "label_version": str(feature_config.get("label_version", "")),
                        "state_names": STATE_ID_TO_NAME,
                        "model_features": features,
                        "estimator": estimator,
                        "decision": {
                            "non_normal_min_prob": float(non_normal_min_prob),
                        },
                        "postprocess": {
                            "mode": "rolling_majority",
                            "window": int(window),
                            "min_vote_ratio": float(min_vote_ratio),
                        },
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "source": {
                            "train_csv": args.train_csv,
                            "val_csv": args.val_csv,
                        "feature_config": args.feature_config or "",
                        "selection_metric": args.metric,
                    },
                    "feature_context": feature_context,
                }
                    last_bundle = bundle
                    score = val_metrics[args.metric]
                    if candidate_best is None or score > candidate_best["score"]:
                        candidate_best = {"score": score, "row": row}
                    if not row["passes_selection_constraints"]:
                        continue
                    if best is None or score > best["score"]:
                        best = {"score": score, "bundle": bundle, "row": row}
                        log(
                            "[BEST] %s=%.4f candidate=%s non_normal_min_prob=%.2f post_window=%d min_vote_ratio=%.2f val_non_normal=%.3f"
                            % (
                                args.metric,
                                score,
                                name,
                                non_normal_min_prob,
                                window,
                                min_vote_ratio,
                                val_metrics["pred_non_normal_ratio"],
                            )
                        )
        elapsed = time.perf_counter() - start
        if candidate_best:
            log(
                "[DONE] %s best_%s=%.4f, elapsed %.1fs"
                % (name, args.metric, candidate_best["score"], elapsed)
            )

    if best is None or last_bundle is None:
        pd.DataFrame(rows).to_csv(out_dir / "classifier_train_log.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "No classifier candidate satisfied the validation output-ratio constraints. "
            "Lower --min-val-non-normal-ratio/--min-val-low-ratio/--min-val-high-ratio, "
            f"then inspect {out_dir / 'classifier_train_log.csv'}."
        )

    pd.DataFrame(rows).to_csv(out_dir / "classifier_train_log.csv", index=False, encoding="utf-8-sig")
    save_activity_classifier(best["bundle"], out_dir / "best_activity_classifier.pkl")
    save_activity_classifier(last_bundle, out_dir / "last_activity_classifier.pkl")
    best_metrics = {
        "selection_metric": args.metric,
        "best_score": best["score"],
        "best_row": best["row"],
        "best_train_metrics": best["bundle"]["train_metrics"],
        "best_val_metrics": best["bundle"]["val_metrics"],
        "selection_constraints": {
            "min_val_non_normal_ratio": args.min_val_non_normal_ratio,
            "min_val_low_ratio": args.min_val_low_ratio,
            "min_val_high_ratio": args.min_val_high_ratio,
        },
        "state_names": STATE_ID_TO_NAME,
    }
    (out_dir / "best_classifier_metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] best classifier: {out_dir / 'best_activity_classifier.pkl'}")
    print(f"[OK] last classifier: {out_dir / 'last_activity_classifier.pkl'}")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
