# YOLO 模块

YOLO 单类鱼体检测模型的训练、预测和跟踪模块。在本项目最终架构中，YOLO **只负责鱼体检测**（单类 `fish`），活跃状态由独立的时序分类器判定。

## 文件说明

| 文件 | 状态 | 说明 |
|------|------|------|
| `predict_track_yolo.py` | ★ 核心 | YOLO + ByteTrack 跟踪预测，输出带 `track_id` 的 CSV |
| `predict_yolo.py` | 参考 | YOLO 预测（无跟踪），不用于主流水线 |
| `train_yolo.py` | 参考 | YOLO 训练封装，fish 检测模型在 Kaggle 上训练 |
| `evaluate_yolo.py` | 参考 | YOLO 评估，针对旧三分类架构 |
| `export_model.py` | 参考 | 模型导出（ONNX 等），当前未使用 |

> 标注为"参考"的脚本在当前主流水线中不直接使用，保留供参考或备选。

## 核心脚本：predict_track_yolo.py

用训练好的 YOLO fish 检测模型对图像目录做逐帧推理，ByteTrack 跨帧关联生成 `track_id`，输出 `raw_predictions.csv`。

### 命令

```bash
python yolo/predict_track_yolo.py \
  --weights models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt \
  --source data/MFT25/MFT25-train/PF-001/img1 \
  --out-dir output/activity_state_yolo/PF-001_v9_step1/tracking_all \
  --tracker bytetrack.yaml \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.7 \
  --fps 25
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--weights` | YOLO 权重路径 | `models/fish_...best.pt` |
| `--source` | 图像目录或视频路径 | — |
| `--out-dir` | 输出目录 | — |
| `--tracker` | 跟踪器配置 | `bytetrack.yaml` |
| `--imgsz` | 输入尺寸 | 640 |
| `--conf` | 检测置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.7 |
| `--fps` | 视频帧率（视频输入时用） | 25 |
| `--save-video` | 同时输出跟踪可视化视频 | — |

### 输出 CSV 字段

```
frame, source_path, track_id, x1, y1, x2, y2, x, y, w, h, cx, cy, conf, state_id, state_name
```

> 注意：YOLO 为单类 fish 模型，`state_name` 固定为 `fish`。真正的活跃状态由下游 `analyze_temporal_states.py` 计算。

## 参考脚本

### train_yolo.py — YOLO 训练

fish 检测模型在 Kaggle 上训练。本地如需重训：

```bash
python yolo/train_yolo.py \
  --data output/fish_detection/PF-001_yolo_fish_step5_mixed/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --name fish_yolov8n_640_pf001_mixed_e50
```

### predict_yolo.py — 无跟踪预测

不生成 track_id，仅用于单帧/目录的快速推理测试。

### evaluate_yolo.py / export_model.py

旧架构的三分类评估和 ONNX 导出，当前流水线不使用。

## 当前模型

| 权重 | 说明 | 指标 |
|------|------|------|
| `models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt` | PF-001 step5 Mixed 数据集训练 | mAP50=98.23%, P=99.3%, R=95.2% |

## 在流水线中的位置

```
┌─────────────────────────────────────────┐
│  Stage 1: YOLO 鱼体检测 (本模块)          │
│  输入: 帧图像                             │
│  输出: raw_predictions.csv (bbox + track_id)│
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  Stage 2: 时序分析 + 状态分类器            │
│  模块: temporal_analysis/                │
└─────────────────────────────────────────┘
```

详见项目主文档 [FINAL_METRICS_MINVIS_PIPELINE.md](../FINAL_METRICS_MINVIS_PIPELINE.md)。
