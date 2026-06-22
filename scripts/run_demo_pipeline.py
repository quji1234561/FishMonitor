#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the deliverable demo pipeline:
video -> extracted frames -> YOLO tracking -> temporal state analysis -> result video.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


DEFAULT_WEIGHT_CANDIDATES = [
    "models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt",
    "weights/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt",
]
DEFAULT_THRESHOLD_CANDIDATES = [
    "configs/activity_speed_thresholds.json",
    "weights/activity_speed_thresholds.json",
    "models/activity_speed_thresholds.json",
]
DEFAULT_ACTIVITY_MODEL_CANDIDATES = [
    "models/activity_state/best_activity_model.json",
    "weights/activity_state/best_activity_model.json",
    "configs/activity_state/best_activity_model.json",
    "models/best_activity_model.json",
]
DEFAULT_ACTIVITY_CLASSIFIER_CANDIDATES = [
    "models/activity_state/best_activity_classifier.pkl",
    "weights/activity_state/best_activity_classifier.pkl",
    "configs/activity_state/best_activity_classifier.pkl",
    "models/best_activity_classifier.pkl",
]
DEFAULT_ACTIVITY_TCN_CANDIDATES = [
    "models/activity_state/best_activity_tcn.pt",
    "weights/activity_state/best_activity_tcn.pt",
    "configs/activity_state/best_activity_tcn.pt",
    "models/best_activity_tcn.pt",
]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FishMonitor video demo pipeline")
    parser.add_argument("--input-video", required=True, help="Input fish video")
    parser.add_argument("--out-dir", default="output/web_demo/manual_run", help="Output directory")
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO fish detector weights. If omitted, the script searches models/ and weights/.",
    )
    parser.add_argument(
        "--threshold-file",
        default=None,
        help="Trained speed threshold JSON. If omitted, the script searches configs/, weights/ and models/.",
    )
    parser.add_argument(
        "--activity-model",
        default=None,
        help="Trained multi-feature activity-state model JSON. If omitted, the script searches models/activity_state/.",
    )
    parser.add_argument(
        "--activity-classifier",
        default=None,
        help="Trained supervised activity-state classifier PKL. If omitted, the script searches models/activity_state/.",
    )
    parser.add_argument(
        "--activity-tcn",
        default=None,
        help="Trained TCN activity-state model PT. If omitted, the script searches models/activity_state/.",
    )
    parser.add_argument("--tcn-device", default="cpu", help="TCN inference device: cpu or cuda")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="YOLO tracker config")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame for faster demo")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--state-window", type=int, default=None, help="Temporal smoothing window in sampled frames")
    parser.add_argument("--state-window-sec", type=float, default=3.0, help="Temporal smoothing window in seconds")
    parser.add_argument("--low-speed-p", type=float, default=33.0, help="Low activity speed percentile")
    parser.add_argument("--high-speed-p", type=float, default=67.0, help="High activity speed percentile")
    parser.add_argument("--resize", type=float, default=1.0, help="Output video resize ratio")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def make_browser_mp4(input_path: Path) -> Path:
    """Convert OpenCV mp4v output to browser-friendly H.264 when ffmpeg exists."""
    ffmpeg = shutil.which("ffmpeg")
    output_path = input_path.with_name(input_path.stem + "_browser.mp4")
    if not ffmpeg:
        print("[WARN] ffmpeg not found, skip browser-compatible H.264 conversion.")
        return input_path

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print("[WARN] ffmpeg conversion failed, keep original video.")
        if exc.stdout:
            print(exc.stdout.decode("utf-8", errors="replace")[-2000:])
        return input_path

    print("[OK] Browser-compatible video:", output_path)
    return output_path


def resolve_default_weights(repo_root: Path) -> Path:
    for rel in DEFAULT_WEIGHT_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path

    patterns = [
        "*PF001*mix*best*.pt",
        "*PF001*mixed*best*.pt",
        "*pf001*mix*best*.pt",
        "*mixed*mAP50*0982*best*.pt",
        "*mixed*best*.pt",
    ]
    for root_name in ["models", "weights"]:
        root = repo_root / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]

    searched = ", ".join(DEFAULT_WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"YOLO weights not found. Tried: {searched}")


def resolve_default_threshold_file(repo_root: Path) -> Path | None:
    for rel in DEFAULT_THRESHOLD_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path

    patterns = [
        "*activity*threshold*.json",
        "*speed*threshold*.json",
    ]
    for root_name in ["configs", "weights", "models"]:
        root = repo_root / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def resolve_default_activity_model(repo_root: Path) -> Path | None:
    for rel in DEFAULT_ACTIVITY_MODEL_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path

    patterns = [
        "*best*activity*model*.json",
        "*activity*state*model*.json",
    ]
    for root_name in ["models", "weights", "configs"]:
        root = repo_root / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def resolve_default_activity_classifier(repo_root: Path) -> Path | None:
    for rel in DEFAULT_ACTIVITY_CLASSIFIER_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path

    patterns = [
        "*best*activity*classifier*.pkl",
        "*activity*classifier*.pkl",
    ]
    for root_name in ["models", "weights", "configs"]:
        root = repo_root / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def resolve_default_activity_tcn(repo_root: Path) -> Path | None:
    for rel in DEFAULT_ACTIVITY_TCN_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path

    patterns = [
        "*best*activity*tcn*.pt",
        "*activity*tcn*.pt",
    ]
    for root_name in ["models", "weights", "configs"]:
        root = repo_root / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def extract_frames(video_path: Path, frames_dir: Path, frame_step: int, max_frames: int | None) -> float:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video type: {video_path.suffix}")
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % frame_step != 0:
            continue
        out_path = frames_dir / f"{frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
        if max_frames and saved >= max_frames:
            break

    cap.release()
    if saved == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    print(f"[OK] Extracted frames: {saved}, source fps: {fps:.3f}")
    return fps


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_video = Path(args.input_video).resolve()
    out_dir = Path(args.out_dir).resolve()
    if args.weights:
        weights = Path(args.weights)
        if not weights.is_absolute():
            weights = repo_root / weights
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
    else:
        weights = resolve_default_weights(repo_root)
    print(f"[INFO] Using weights: {weights}")

    threshold_file = None
    if args.threshold_file:
        threshold_file = Path(args.threshold_file)
        if not threshold_file.is_absolute():
            threshold_file = repo_root / threshold_file
        if not threshold_file.exists():
            raise FileNotFoundError(f"Threshold file not found: {threshold_file}")
    else:
        threshold_file = resolve_default_threshold_file(repo_root)
    if threshold_file:
        print(f"[INFO] Using trained thresholds: {threshold_file}")
    else:
        print("[INFO] No trained threshold file found; use runtime percentiles.")

    activity_model = None
    activity_classifier = None
    activity_tcn = None
    if args.activity_tcn:
        activity_tcn = Path(args.activity_tcn)
        if not activity_tcn.is_absolute():
            activity_tcn = repo_root / activity_tcn
        if not activity_tcn.exists():
            raise FileNotFoundError(f"Activity TCN not found: {activity_tcn}")
    else:
        activity_tcn = resolve_default_activity_tcn(repo_root)
    if activity_tcn:
        print(f"[INFO] Using activity-state TCN: {activity_tcn}")
    else:
        print("[INFO] No activity-state TCN found; try activity-state classifier.")

    if args.activity_classifier:
        activity_classifier = Path(args.activity_classifier)
        if not activity_classifier.is_absolute():
            activity_classifier = repo_root / activity_classifier
        if not activity_classifier.exists():
            raise FileNotFoundError(f"Activity classifier not found: {activity_classifier}")
    else:
        activity_classifier = resolve_default_activity_classifier(repo_root)
    if activity_classifier:
        print(f"[INFO] Using activity-state classifier: {activity_classifier}")
    else:
        print("[INFO] No activity-state classifier found; try activity-state model.")

    if args.activity_model:
        activity_model = Path(args.activity_model)
        if not activity_model.is_absolute():
            activity_model = repo_root / activity_model
        if not activity_model.exists():
            raise FileNotFoundError(f"Activity model not found: {activity_model}")
    else:
        activity_model = resolve_default_activity_model(repo_root)
    if activity_model:
        print(f"[INFO] Using activity-state model: {activity_model}")
    else:
        print("[INFO] No activity-state model found; use speed thresholds.")

    frames_dir = out_dir / "frames"
    track_dir = out_dir / "tracking"
    temporal_dir = out_dir / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = extract_frames(
        video_path=input_video,
        frames_dir=frames_dir,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )
    output_fps = fps / args.frame_step
    state_window = args.state_window
    if state_window is None:
        state_window = max(1, int(round(args.state_window_sec * output_fps)))
    print(f"[INFO] Temporal state window: {state_window} sampled frames ({args.state_window_sec:.2f}s)")

    predict_cmd = [
        sys.executable,
        "yolo/predict_track_yolo.py",
        "--weights",
        str(weights),
        "--source",
        str(frames_dir),
        "--out-dir",
        str(track_dir),
        "--tracker",
        args.tracker,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--fps",
        str(output_fps),
    ]
    if args.device:
        predict_cmd += ["--device", args.device]
    run(predict_cmd, cwd=repo_root)

    temporal_cmd = [
        sys.executable,
        "temporal_analysis/analyze_temporal_states.py",
        "--pred-csv",
        str(track_dir / "raw_predictions.csv"),
        "--out-dir",
        str(temporal_dir),
        "--source",
        str(frames_dir),
        "--fps",
        str(fps),
        "--window",
        str(state_window),
        "--low-speed-p",
        str(args.low_speed_p),
        "--high-speed-p",
        str(args.high_speed_p),
        "--resize",
        str(args.resize),
        "--save-video",
    ]
    if activity_tcn:
        temporal_cmd += [
            "--state-source",
            "activity_tcn",
            "--activity-tcn",
            str(activity_tcn),
            "--tcn-device",
            args.tcn_device,
        ]
    elif activity_classifier:
        temporal_cmd += [
            "--state-source",
            "activity_classifier",
            "--activity-classifier",
            str(activity_classifier),
        ]
    elif activity_model:
        temporal_cmd += [
            "--state-source",
            "activity_model",
            "--activity-model",
            str(activity_model),
        ]
    else:
        temporal_cmd += ["--state-source", "speed"]
    if threshold_file and not activity_model and not activity_classifier and not activity_tcn:
        temporal_cmd += ["--threshold-file", str(threshold_file)]
    if args.max_frames:
        temporal_cmd += ["--max-frames", str(args.max_frames)]
    run(temporal_cmd, cwd=repo_root)

    result_video = temporal_dir / "temporal_smoothed_video.mp4"
    browser_video = make_browser_mp4(result_video)
    print("[DONE] Demo result video:", result_video)
    print("[DONE] Browser video:", browser_video)
    print("[DONE] Tracking CSV:", track_dir / "raw_predictions.csv")
    print("[DONE] Temporal CSV:", temporal_dir / "temporal_results.csv")


if __name__ == "__main__":
    main()
