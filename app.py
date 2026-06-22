#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight web demo server for FishMonitor."""

from __future__ import annotations

import subprocess
import sys
import traceback
import uuid
import os
import csv
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


REPO_ROOT = Path(__file__).resolve().parent
UPLOAD_ROOT = REPO_ROOT / "output" / "web_demo" / "uploads"
JOB_ROOT = REPO_ROOT / "output" / "web_demo" / "jobs"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt"
DEFAULT_WEIGHT_CANDIDATES = [
    REPO_ROOT / "models" / "fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt",
    REPO_ROOT / "weights" / "fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt",
]
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
STATE_LABELS = {
    "0": {"name": "low_activity", "label": "低活跃", "color": "orange"},
    "1": {"name": "normal_activity", "label": "正常活跃", "color": "green"},
    "2": {"name": "high_activity", "label": "高活跃", "color": "red"},
}

app = Flask(__name__, static_folder="frontend_demo", static_url_path="")


@app.errorhandler(Exception)
def handle_exception(exc):
    traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "error": "后端内部异常",
            "log": traceback.format_exc()[-6000:],
        }), 500
    raise exc


def find_default_weights() -> Path | None:
    for path in DEFAULT_WEIGHT_CANDIDATES:
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
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def summarize_labels(result_csv: Path) -> dict:
    counts = {state_id: 0 for state_id in STATE_LABELS}
    frames = set()
    tracks = set()
    total = 0

    if not result_csv.exists():
        return {
            "total_boxes": 0,
            "frame_count": 0,
            "track_count": 0,
            "items": [
                {**meta, "state_id": int(state_id), "count": 0, "ratio": 0.0}
                for state_id, meta in STATE_LABELS.items()
            ],
        }

    with result_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            state_id = str(row.get("smooth_state_id") or row.get("state_id") or "").strip()
            if state_id not in counts:
                continue
            counts[state_id] += 1
            total += 1
            frame = row.get("frame")
            track_id = row.get("track_id")
            if frame not in (None, ""):
                frames.add(frame)
            if track_id not in (None, ""):
                tracks.add(track_id)

    items = []
    for state_id, meta in STATE_LABELS.items():
        count = counts[state_id]
        ratio = count / total if total else 0.0
        items.append({
            **meta,
            "state_id": int(state_id),
            "count": count,
            "ratio": ratio,
        })

    return {
        "total_boxes": total,
        "frame_count": len(frames),
        "track_count": len(tracks),
        "items": items,
    }


@app.get("/")
def index():
    return send_from_directory(REPO_ROOT / "frontend_demo", "index.html")


@app.get("/health")
def health():
    weights = find_default_weights()
    return jsonify({
        "ok": True,
        "weights_exists": weights is not None,
        "weights": str(weights.relative_to(REPO_ROOT)) if weights else "",
    })


@app.post("/api/process")
def process_video():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "请先选择视频文件"}), 400

    upload = request.files["video"]
    original_name = upload.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        return jsonify({"ok": False, "error": f"不支持的视频格式: {suffix}"}), 400
    weights = find_default_weights()
    if weights is None:
        return jsonify({
            "ok": False,
            "error": "找不到模型权重，请将 PF-001 mixed best.pt 放到 models/ 或 weights/ 目录。",
        }), 500

    job_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOAD_ROOT / job_id
    job_dir = JOB_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = Path(secure_filename(original_name)).stem or "input"
    filename = f"{safe_stem}{suffix}"
    input_path = upload_dir / filename
    upload.save(input_path)

    frame_step = request.form.get("frame_step", "1")
    max_frames = request.form.get("max_frames", "").strip()
    cmd = [
        sys.executable,
        "scripts/run_demo_pipeline.py",
        "--input-video",
        str(input_path),
        "--out-dir",
        str(job_dir),
        "--weights",
        str(weights),
        "--frame-step",
        frame_step,
        "--resize",
        "1.0",
    ]
    if max_frames:
        cmd += ["--max-frames", max_frames]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        return jsonify({
            "ok": False,
            "error": "视频处理失败",
            "log": exc.stdout[-6000:] if exc.stdout else "",
        }), 500

    print(completed.stdout, flush=True)

    summary_path = job_dir / "temporal" / "temporal_summary.txt"
    temporal_summary = ""
    if summary_path.exists():
        temporal_summary = summary_path.read_text(encoding="utf-8", errors="replace")

    browser_video = job_dir / "temporal" / "temporal_smoothed_video_browser.mp4"
    result_csv = job_dir / "temporal" / "temporal_results.csv"
    label_stats = summarize_labels(result_csv)
    result_name = "temporal_smoothed_video_browser.mp4" if browser_video.exists() else "temporal_smoothed_video.mp4"
    result_rel = f"{job_id}/temporal/{result_name}"
    summary_rel = f"{job_id}/temporal/temporal_summary.txt"
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "result_video": f"/results/{result_rel}",
        "summary": f"/results/{summary_rel}",
        "temporal_summary": temporal_summary,
        "label_stats": label_stats,
        "log": completed.stdout[-4000:],
    })


@app.get("/results/<job_id>/<path:filename>")
def result_file(job_id: str, filename: str):
    job_dir = JOB_ROOT / job_id
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    debug = os.environ.get("FISHMONITOR_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=7860, debug=debug)
