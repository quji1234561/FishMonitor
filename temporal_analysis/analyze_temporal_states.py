#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼类状态时序分析脚本：
1. 读取 YOLO 跟踪预测结果 raw_predictions.csv
2. 按 track_id 计算速度
3. 对同一条鱼的状态做滑动窗口多数投票平滑
4. 使用速度对明显跳变状态做轻量修正
5. 计算每帧鱼群整体活跃度指数
6. 输出 CSV、趋势图，可选输出平滑后视频

推荐放置位置：
FishMonitor/temporal_analysis/analyze_temporal_states.py

示例：
python temporal_analysis/analyze_temporal_states.py \
  --pred-csv output/temporal/BT-001/raw_predictions.csv \
  --out-dir output/temporal/BT-001 \
  --source output/enhanced/BT-001_yolo_enhanced/images/test \
  --save-video \
  --fps 10
"""

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temporal_analysis.motion_features import (  # noqa: E402
    add_motion_features,
    load_activity_model,
    predict_with_activity_model,
)
from temporal_analysis.activity_classifier import (  # noqa: E402
    load_activity_classifier,
    predict_with_activity_classifier,
)
from temporal_analysis.activity_tcn import (  # noqa: E402
    load_activity_tcn,
    predict_with_activity_tcn,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

STATE_ID_TO_NAME = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}
STATE_NAME_TO_ID = {v: k for k, v in STATE_ID_TO_NAME.items()}
STATE_SHORT = {
    "low_activity": "low",
    "normal_activity": "normal",
    "high_activity": "high",
}
STATE_SCORE = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
}
STATE_COLOR = {
    0: (80, 160, 255),
    1: (80, 220, 80),
    2: (60, 80, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal smoothing and activity analysis")
    parser.add_argument("--pred-csv", required=True, help="predict_track_yolo.py 输出的 raw_predictions.csv")
    parser.add_argument("--out-dir", default="output/temporal/analyze", help="输出目录")
    parser.add_argument("--fps", type=float, default=10.0, help="视频帧率；也用于速度计算的秒换算")
    parser.add_argument("--window", type=int, default=5, help="滑动窗口大小，建议 5 或 7")
    parser.add_argument("--min-vote-ratio", type=float, default=0.6, help="多数投票最低占比")
    parser.add_argument("--speed-norm", choices=["bbox_diag", "bbox_size", "none"], default="bbox_diag", help="速度归一化方式")
    parser.add_argument("--speed-smooth-window", type=int, default=5, help="速度中值平滑窗口，需与阈值训练保持一致")
    parser.add_argument(
        "--state-source",
        choices=["auto", "model", "speed", "activity_model", "activity_classifier", "activity_tcn"],
        default="auto",
        help="状态来源：model 使用 YOLO 类别；speed 使用速度阈值；activity_model 使用多特征加权模型；activity_classifier 使用监督分类器；activity_tcn 使用 TCN 时序模型",
    )
    parser.add_argument("--low-speed-p", type=float, default=33.0, help="低速阈值分位数")
    parser.add_argument("--high-speed-p", type=float, default=67.0, help="高速阈值分位数")
    parser.add_argument("--threshold-file", default=None, help="训练得到的速度阈值 JSON；提供后优先使用固定阈值")
    parser.add_argument("--activity-model", default=None, help="训练得到的多特征状态模型 JSON；提供后优先使用")
    parser.add_argument("--activity-classifier", default=None, help="训练得到的监督活动状态分类器 PKL；提供后优先使用")
    parser.add_argument("--activity-tcn", default=None, help="训练得到的 TCN 活动状态模型 PT；提供后优先使用")
    parser.add_argument("--tcn-device", default="cpu", help="TCN 推理设备：cpu 或 cuda")
    parser.add_argument("--no-speed-correct", action="store_true", help="关闭速度辅助修正")
    parser.add_argument("--source", default=None, help="原始图片文件夹、单张图片或视频；用于生成平滑后视频")
    parser.add_argument("--save-video", action="store_true", help="是否生成平滑后状态视频")
    parser.add_argument("--resize", type=float, default=1.0, help="视频缩放比例，例如 0.5")
    parser.add_argument("--max-frames", type=int, default=None, help="最多输出多少帧视频")
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到预测 CSV: {path}")
    df = pd.read_csv(path)
    required = {"frame", "track_id", "x", "y", "w", "h", "cx", "cy", "conf", "state_id", "state_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"预测 CSV 缺少字段: {missing}")
    df = df.copy()
    df = df[df["track_id"] >= 0].copy()
    if df.empty:
        raise RuntimeError("CSV 中没有有效 track_id。请确认 predict_track_yolo.py 使用了 model.track。")
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["state_id"] = df["state_id"].astype(int)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def compute_speed(df: pd.DataFrame, fps: float, norm_mode: str, smooth_window: int = 5) -> pd.DataFrame:
    return add_motion_features(
        df,
        fps=fps,
        speed_norm=norm_mode,
        smooth_window=smooth_window,
        feature_window=5,
    )


def resolve_state_source(df: pd.DataFrame, requested: str) -> str:
    if requested == "activity_model":
        return requested
    if requested != "auto":
        return requested
    names = set(str(x) for x in df["state_name"].dropna().unique())
    activity_names = set(STATE_NAME_TO_ID.keys())
    if names and names.issubset(activity_names):
        return "model"
    # A single-class fish detector usually outputs state_name == "fish".
    return "speed"


def load_threshold_file(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"找不到速度阈值文件: {threshold_path}")
    data = json.loads(threshold_path.read_text(encoding="utf-8"))
    required = {"low_speed_threshold", "high_speed_threshold"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"阈值文件缺少字段: {missing}")
    return data


def assign_states_from_speed(
    df: pd.DataFrame,
    low_p: float,
    high_p: float,
    threshold_data: Optional[dict] = None,
) -> tuple[pd.DataFrame, float, float, str]:
    df = df.copy()

    speed_col = "speed_smooth"
    if threshold_data is not None:
        speed_col = str(threshold_data.get("speed_column", "speed_smooth"))
        if speed_col not in df.columns:
            raise ValueError(f"预测结果中没有阈值文件要求的速度列: {speed_col}")
        low_thr = float(threshold_data["low_speed_threshold"])
        high_thr = float(threshold_data["high_speed_threshold"])
    else:
        speeds = df[speed_col].dropna().to_numpy()
        if len(speeds) == 0:
            low_thr = high_thr = 0.0
        else:
            low_thr = float(np.percentile(speeds, low_p))
            high_thr = float(np.percentile(speeds, high_p))

    def label(v: float) -> int:
        if v <= low_thr:
            return 0
        if v >= high_thr:
            return 2
        return 1

    df["model_state_id"] = df["state_id"]
    df["model_state_name"] = df["state_name"]
    df["state_id"] = df[speed_col].apply(label).astype(int)
    df["state_name"] = df["state_id"].map(STATE_ID_TO_NAME)
    return df, low_thr, high_thr, speed_col


def assign_states_from_activity_model(df: pd.DataFrame, model_path: str, fps: float) -> tuple[pd.DataFrame, dict]:
    model = load_activity_model(Path(model_path))
    df = df.copy()
    df["model_state_id"] = df["state_id"]
    df["model_state_name"] = df["state_name"]
    df = predict_with_activity_model(df, model, fps=fps)
    return df, model


def assign_states_from_activity_classifier(df: pd.DataFrame, model_path: str) -> tuple[pd.DataFrame, dict]:
    model = load_activity_classifier(Path(model_path))
    df = df.copy()
    df["model_state_id"] = df["state_id"]
    df["model_state_name"] = df["state_name"]
    df = predict_with_activity_classifier(df, model)
    return df, model


def assign_states_from_activity_tcn(df: pd.DataFrame, model_path: str, device: str) -> tuple[pd.DataFrame, dict]:
    model = load_activity_tcn(Path(model_path), map_location=device)
    df = df.copy()
    df["model_state_id"] = df["state_id"]
    df["model_state_name"] = df["state_name"]
    df = predict_with_activity_tcn(df, model, device=device)
    return df, model


def rolling_majority(states: List[int], window: int, min_vote_ratio: float) -> List[int]:
    q = deque(maxlen=window)
    out = []
    for s in states:
        q.append(int(s))
        counter = Counter(q)
        best_state, best_count = counter.most_common(1)[0]
        if best_count / len(q) >= min_vote_ratio:
            out.append(best_state)
        else:
            out.append(int(s))
    return out


def smooth_states(df: pd.DataFrame, window: int, min_vote_ratio: float) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("track_id"):
        group = group.sort_values("frame").copy()
        group["vote_state_id"] = rolling_majority(group["state_id"].tolist(), window, min_vote_ratio)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["frame", "track_id"]).reset_index(drop=True)


def speed_correct(df: pd.DataFrame, low_p: float, high_p: float, enabled: bool) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()
    speeds = df["speed_norm_s"].dropna().to_numpy()
    if len(speeds) == 0:
        low_thr = high_thr = 0.0
    else:
        low_thr = float(np.percentile(speeds, low_p))
        high_thr = float(np.percentile(speeds, high_p))

    final = []
    for _, row in df.iterrows():
        s = int(row["vote_state_id"])
        v = float(row["speed_norm_s"])
        if enabled:
            # 轻量修正：只修正明显矛盾，不强行覆盖所有结果
            if s == 2 and v <= low_thr:
                s = 1
            elif s == 0 and v >= high_thr:
                s = 1
        final.append(s)

    df["smooth_state_id"] = final
    df["smooth_state_name"] = df["smooth_state_id"].map(STATE_ID_TO_NAME)
    return df, low_thr, high_thr


def make_frame_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame, g in df.groupby("frame"):
        total = len(g)
        low = int((g["smooth_state_id"] == 0).sum())
        normal = int((g["smooth_state_id"] == 1).sum())
        high = int((g["smooth_state_id"] == 2).sum())
        low_ratio = low / total if total else 0.0
        high_ratio = high / total if total else 0.0
        mean_speed = float(g["speed_norm_s"].mean()) if total else 0.0
        mean_state_score = float(g["smooth_state_id"].map(STATE_SCORE).mean()) if total else 0.0
        # 简单活跃度指数：状态得分为主，速度为辅；为了稳定，速度只参与相对趋势展示
        activity_index = 0.7 * mean_state_score + 0.3 * min(mean_speed, 1.0)
        rows.append({
            "frame": int(frame),
            "fish_count": total,
            "low_count": low,
            "normal_count": normal,
            "high_count": high,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
            "mean_speed_norm": mean_speed,
            "mean_state_score": mean_state_score,
            "activity_index": activity_index,
        })
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def save_trend_plot(summary: pd.DataFrame, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过趋势图。可运行 pip install matplotlib")
        return
    plt.figure(figsize=(10, 4))
    plt.plot(summary["frame"], summary["activity_index"], label="activity_index")
    plt.plot(summary["frame"], summary["high_ratio"], label="high_ratio")
    plt.plot(summary["frame"], summary["low_ratio"], label="low_ratio")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Fish Activity Trend")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] 活跃趋势图: {out_path}")


def collect_images(source: Path) -> Dict[int, Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        try:
            frame = int(source.stem)
        except ValueError:
            frame = 1
        return {frame: source}
    if source.is_dir():
        images = sorted([p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
        result = {}
        for i, p in enumerate(images, start=1):
            try:
                frame = int(p.stem)
            except ValueError:
                frame = i
            result[frame] = p
        return result
    return {}


def resize_if_needed(img, scale: float):
    import cv2

    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def draw_box(img, row):
    import cv2

    sid = int(row["smooth_state_id"])
    color = STATE_COLOR.get(sid, (255, 255, 255))
    x1 = int(round(float(row["x"])))
    y1 = int(round(float(row["y"])))
    x2 = int(round(float(row["x"] + row["w"])))
    y2 = int(round(float(row["y"] + row["h"])))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    state_short = STATE_SHORT.get(str(row["smooth_state_name"]), str(row["smooth_state_name"]))
    raw_short = STATE_SHORT.get(str(row["state_name"]), str(row["state_name"]))
    label = f"ID:{int(row['track_id'])} {state_short}"
    if state_short != raw_short:
        label += f"<-{raw_short}"
    label += f" v:{float(row['speed_norm_s']):.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(label, font, 0.52, 1)
    y_text = max(y1, th + 8)
    cv2.rectangle(img, (x1, y_text - th - base - 4), (x1 + tw + 6, y_text + base + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, y_text - 3), font, 0.52, (255,255,255), 1, cv2.LINE_AA)


def make_video_from_images(source: Path, df: pd.DataFrame, summary: pd.DataFrame, out_path: Path, fps: float, resize: float, max_frames: Optional[int]):
    import cv2

    image_map = collect_images(source)
    if not image_map:
        raise RuntimeError(f"没有找到图片: {source}")
    frames = sorted(set(df["frame"].unique()).intersection(image_map.keys()))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError("没有可用于生成视频的帧，请检查 source 与 CSV 的 frame 是否对应。")

    first = cv2.imread(str(image_map[frames[0]]))
    if first is None:
        raise RuntimeError(f"无法读取图片: {image_map[frames[0]]}")
    first_r = resize_if_needed(first, resize)
    h, w = first_r.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")

    grouped = dict(tuple(df.groupby("frame")))
    summary_map = summary.set_index("frame").to_dict("index")
    for frame in frames:
        img = cv2.imread(str(image_map[frame]))
        if img is None:
            continue
        for _, row in grouped.get(frame, pd.DataFrame()).iterrows():
            draw_box(img, row)
        s = summary_map.get(frame, {})
        text = f"frame:{frame} activity:{s.get('activity_index', 0):.2f} high:{s.get('high_ratio', 0):.2f} low:{s.get('low_ratio', 0):.2f}"
        cv2.putText(img, text, (20, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        writer.write(resize_if_needed(img, resize))
    writer.release()
    print(f"[OK] 时序平滑视频: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(Path(args.pred_csv))
    df = compute_speed(df, fps=args.fps, norm_mode=args.speed_norm, smooth_window=args.speed_smooth_window)
    if args.activity_tcn:
        state_source = "activity_tcn"
    elif args.activity_classifier:
        state_source = "activity_classifier"
    elif args.activity_model:
        state_source = "activity_model"
    else:
        state_source = resolve_state_source(df, args.state_source)
    threshold_data = load_threshold_file(args.threshold_file)
    threshold_mode = "trained_file" if threshold_data is not None else "runtime_percentile"
    threshold_speed_col = "speed_smooth"
    activity_model_info = {}
    classifier_info = {}
    tcn_info = {}
    actual_window = args.window
    actual_min_vote_ratio = args.min_vote_ratio

    if state_source == "activity_tcn":
        df, tcn_info = assign_states_from_activity_tcn(df, args.activity_tcn, device=args.tcn_device)
        post = tcn_info.get("postprocess", {})
        post_window = int(post.get("window", args.window))
        post_min_vote_ratio = float(post.get("min_vote_ratio", args.min_vote_ratio))
        actual_window = post_window
        actual_min_vote_ratio = post_min_vote_ratio
        df = smooth_states(df, window=post_window, min_vote_ratio=post_min_vote_ratio)
        df["smooth_state_id"] = df["vote_state_id"].astype(int)
        df["smooth_state_name"] = df["smooth_state_id"].map(STATE_ID_TO_NAME)
        low_thr = 0.0
        high_thr = 0.0
        threshold_mode = "activity_tcn"
        threshold_speed_col = "tcn"
    elif state_source == "activity_classifier":
        df, classifier_info = assign_states_from_activity_classifier(df, args.activity_classifier)
        post = classifier_info.get("postprocess", {})
        post_window = int(post.get("window", args.window))
        post_min_vote_ratio = float(post.get("min_vote_ratio", args.min_vote_ratio))
        actual_window = post_window
        actual_min_vote_ratio = post_min_vote_ratio
        df = smooth_states(df, window=post_window, min_vote_ratio=post_min_vote_ratio)
        df["smooth_state_id"] = df["vote_state_id"].astype(int)
        df["smooth_state_name"] = df["smooth_state_id"].map(STATE_ID_TO_NAME)
        low_thr = 0.0
        high_thr = 0.0
        threshold_mode = "activity_classifier"
        threshold_speed_col = "classifier"
    elif state_source == "activity_model":
        df, activity_model_info = assign_states_from_activity_model(df, args.activity_model, fps=args.fps)
        df = smooth_states(df, window=args.window, min_vote_ratio=args.min_vote_ratio)
        df["smooth_state_id"] = df["vote_state_id"].astype(int)
        df["smooth_state_name"] = df["smooth_state_id"].map(STATE_ID_TO_NAME)
        low_thr = float(activity_model_info.get("low_score_threshold", 0.0))
        high_thr = float(activity_model_info.get("high_score_threshold", 0.0))
        threshold_mode = "activity_model"
        threshold_speed_col = "state_score"
    elif state_source == "speed":
        df, low_thr, high_thr, threshold_speed_col = assign_states_from_speed(
            df,
            args.low_speed_p,
            args.high_speed_p,
            threshold_data=threshold_data,
        )
        df = smooth_states(df, window=args.window, min_vote_ratio=args.min_vote_ratio)
        # For speed-derived states, the thresholding step already defines the state.
        df, _, _ = speed_correct(df, args.low_speed_p, args.high_speed_p, enabled=False)
    else:
        df = smooth_states(df, window=args.window, min_vote_ratio=args.min_vote_ratio)
        df, low_thr, high_thr = speed_correct(df, args.low_speed_p, args.high_speed_p, enabled=not args.no_speed_correct)

    result_csv = out_dir / "temporal_results.csv"
    summary_csv = out_dir / "frame_activity_summary.csv"
    trend_png = out_dir / "activity_trend.png"

    df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    summary = make_frame_summary(df)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    save_trend_plot(summary, trend_png)

    report = out_dir / "temporal_summary.txt"
    report.write_text(
        "时序分析结果摘要\n"
        "================\n"
        f"输入预测框数量: {len(df)}\n"
        f"轨迹数量: {df['track_id'].nunique()}\n"
        f"状态来源: {state_source}\n"
        f"阈值模式: {threshold_mode}\n"
        f"阈值速度列: {threshold_speed_col}\n"
        f"阈值文件: {args.threshold_file or ''}\n"
        f"状态模型: {args.activity_model or ''}\n"
        f"TCN状态模型: {args.activity_tcn or ''}\n"
        f"状态分类器: {args.activity_classifier or ''}\n"
        f"TCN状态模型类型: {tcn_info.get('model_type', '')}\n"
        f"TCN状态模型名称: {tcn_info.get('model_name', '')}\n"
        f"状态模型类型: {activity_model_info.get('model_type', '')}\n"
        f"状态分类器类型: {classifier_info.get('model_type', '')}\n"
        f"状态分类器名称: {classifier_info.get('model_name', '')}\n"
        f"滑动窗口大小: {actual_window}\n"
        f"多数投票比例: {actual_min_vote_ratio}\n"
        f"速度平滑窗口: {args.speed_smooth_window}\n"
        f"速度低阈值: {low_thr:.6f}\n"
        f"速度高阈值: {high_thr:.6f}\n"
        f"输出: {result_csv}\n"
        f"帧级统计: {summary_csv}\n"
        f"趋势图: {trend_png}\n",
        encoding="utf-8",
    )

    print(f"[OK] 时序分析结果: {result_csv}")
    print(f"[OK] 帧级统计: {summary_csv}")
    print(f"[OK] 摘要: {report}")
    print(f"[INFO] 状态来源: {state_source}")
    print(f"[INFO] 阈值模式: {threshold_mode}")
    print(f"[INFO] 速度阈值: low<={low_thr:.6f}, high>={high_thr:.6f}")

    if args.save_video:
        if not args.source:
            raise ValueError("--save-video 需要同时提供 --source")
        source = Path(args.source)
        if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
            print("[WARN] 当前版本主要支持图片序列生成时序视频；视频输入建议先拆帧。")
        else:
            make_video_from_images(
                source=source,
                df=df,
                summary=summary,
                out_path=out_dir / "temporal_smoothed_video.mp4",
                fps=args.fps,
                resize=args.resize,
                max_frames=args.max_frames,
            )


if __name__ == "__main__":
    main()
