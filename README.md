# FishMonitor

基于 MFT25 数据集的**鱼类活跃状态监测**课程设计项目。采用「YOLO 单类鱼体检测 + 时序特征分类器」两阶段架构，从水下视频中检测鱼体位置并识别其活跃状态（低活跃 / 正常活跃 / 高活跃），提供完整的数据处理、模型训练、评估和 Web 演示流程。

---

## 项目架构

```
上传视频 / 帧图像
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: YOLO 鱼体检测      │
│  单类 fish，mAP50 = 98.23%  │
│  输出：鱼体 bbox + track_id  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2: 活跃状态分类器      │
│  基于运动特征的三分类模型      │
│  输出：low/normal/high        │
│  验证集 accuracy = 85.3%     │
└─────────────┬───────────────┘
              │
              ▼
       结果视频 / Web 展示
```

**核心设计**：YOLO 只负责检测"鱼在哪里"（单类 fish），活跃状态由独立分类器基于跨帧运动特征判定，训练输入与推理输入一致（均为 YOLO 预测框），避免 GT 框与预测框的分布差异。

---

## 项目结构

```
FishMonitor/
├── data/MFT25/                           # MFT25 数据集
│   ├── train.json / train_half.json      # COCO 标注
│   ├── val_half.json
│   ├── MFT25-train/                      # 8 个训练序列
│   │   └── PF-001/                       # ★ 主训练序列 (15,000 帧, 2 条/帧)
│   └── MFT25-test/                       # 测试序列（无标注）
│
├── scripts/                              # 数据处理与编排
│   ├── generate_fish_detection_dataset.py # 生成 YOLO fish 单类数据集
│   ├── generate_activity_state_dataset.py # ★ 生成活动状态标签 (metrics_minvis)
│   ├── build_yolo_activity_state_dataset.py # ★ YOLO 框匹配状态标签
│   ├── make_mixed_yolo_dataset.py        # 合并原图+增强为 Mixed 训练集
│   ├── run_demo_pipeline.py              # ★ Web 演示编排脚本
│   ├── generate_annotated_video.py       # 生成标注/原始视频
│   └── generate_activity_video.py        # 生成状态标签可视化视频
│
├── temporal_analysis/                    # 时序分析与状态分类
│   ├── train_activity_state_classifier.py # ★ 训练活动状态分类器
│   ├── analyze_temporal_states.py        # ★ 时序分析 & 分类器推理
│   ├── motion_features.py               # 运动特征提取
│   ├── activity_classifier.py           # 分类器包装模块
│   └── activity_tcn.py                  # TCN 时序模型
│
├── yolo/                                 # YOLO 训练与推理
│   ├── predict_track_yolo.py             # ★ YOLO + ByteTrack 跟踪预测
│   ├── train_yolo.py                     # YOLO 训练封装
│   ├── predict_yolo.py                   # YOLO 预测（无跟踪）
│   ├── evaluate_yolo.py                  # YOLO 评估
│   └── export_model.py                   # 模型导出
│
├── image_enhancement/                    # 水下图像增强
│   ├── enhance_yolo_dataset.py           # 批量增强 YOLO 数据集
│   └── enhancement_methods.py            # CLAHE/Gamma/去噪/锐化
│
├── evaluation/                           # 评估模块
│   └── evaluate_activity_predictions_with_gt.py  # ★ 状态预测 vs GT 评估
│
├── app.py                                # ★ Flask Web 演示后端
├── frontend_demo/index.html              # ★ Web 演示前端页面
│
├── models/                               # 训练好的模型
│   ├── fish_yolov8n_640_PF001_..._best.pt        # ★ YOLO 鱼体检测模型
│   └── activity_state_metrics_minvis/            # ★ 活动状态分类器
│       ├── best_activity_classifier.pkl
│       └── best_classifier_metrics.json
│
├── configs/
│   └── activity_speed_thresholds.json     # 速度阈值配置（旧版兼容）
│
├── output/
│   ├── fish_detection/                   # YOLO fish 数据集（含 Mixed）
│   ├── activity_state/                   # 活动状态标签数据
│   ├── activity_state_yolo/              # YOLO 框匹配后的训练数据
│   ├── evaluation/                       # 评估结果
│   └── web_demo/                         # Web 演示运行时输出
│
├── docs/                                 # 文档
│   ├── deliverable_demo_runbook.md       # 演示操作手册
│   └── metrics/                          # 训练指标记录
│
├── FINAL_METRICS_MINVIS_PIPELINE.md      # ★ 最终流程详细文档
├── PROJECT_STRUCTURE.md                  # 完整项目结构说明
├── DATA.md                               # 数据集详细说明
└── pixi.toml                             # pixi 环境配置
```

---

## 环境依赖

本项目使用 [pixi](https://pixi.sh) 管理 conda 环境：

```bash
pixi install
```

等价 pip 命令：

```bash
pip install opencv-python numpy pandas ultralytics matplotlib flask scikit-learn
```

---

## 最终流程（metrics_minvis）

详细完整流程见 **[FINAL_METRICS_MINVIS_PIPELINE.md](FINAL_METRICS_MINVIS_PIPELINE.md)**。以下是各步骤概览：

### Step 1：生成活动状态标签

从 PF-001 的 GT 框生成 low / normal / high 三类活动状态标签（指标优先策略，保守标注 low 和 high）。

```bash
python scripts/generate_activity_state_dataset.py \
  --seq-dir data/MFT25/MFT25-train/PF-001 \
  --out-dir output/activity_state/PF-001_v9_step1_metrics_minvis \
  --frame-step 1 --label-window-sec 3 \
  --low-percentile 25 --high-percentile 92 \
  --max-non-normal-ratio 0.18 \
  --make-video
```

> 检查点：打开 `dataset_summary.json` 确认 state_counts 中三类均有分布。

### Step 2：YOLO 鱼体检测

用训练好的 YOLO fish 检测模型对 PF-001 全帧预测，生成 YOLO 预测框 CSV。

```bash
python yolo/predict_track_yolo.py \
  --weights models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt \
  --source data/MFT25/MFT25-train/PF-001/img1 \
  --out-dir output/activity_state_yolo/PF-001_v9_step1/tracking_all \
  --tracker bytetrack.yaml --imgsz 640 --conf 0.25 --iou 0.7 --fps 25
```

### Step 3：YOLO 框匹配状态标签

将 YOLO 预测框与 Step 1 的状态标签按 IoU 匹配，用 YOLO 框重新计算运动特征。

```bash
python scripts/build_yolo_activity_state_dataset.py \
  --gt-csv output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_all.csv \
  --pred-csv output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv \
  --feature-config output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json \
  --out-dir output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all \
  --iou-thr 0.5 --fps 25
```

### Step 4：训练活动状态分类器

基于 YOLO 框运动特征训练三分类模型。

```bash
python temporal_analysis/train_activity_state_classifier.py \
  --train-csv output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_train.csv \
  --val-csv output/activity_state_yolo/PF-001_v9_step1_metrics_minvis/matched_all/yolo_activity_state_val.csv \
  --feature-config output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json \
  --out-dir models/activity_state_metrics_minvis \
  --metric accuracy --advanced-models \
  --post-windows 1,15,30,45,75 \
  --min-vote-ratios 0.5,0.6,0.7
```

> 检查点：打开 `best_classifier_metrics.json`，确认 `val_accuracy > 0.8` 且 `val_pred_low_ratio > 0`、`val_pred_high_ratio > 0`。

### Step 5：测试集评估

模拟完整系统，评估状态识别准确率。

```bash
# 5a: 分类器预测
python temporal_analysis/analyze_temporal_states.py \
  --pred-csv output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv \
  --out-dir output/evaluation/PF-001_yolo_state_metrics_minvis/temporal \
  --state-source activity_classifier \
  --activity-classifier models/activity_state_metrics_minvis/best_activity_classifier.pkl \
  --fps 25 --window 75

# 5b: 与 GT 对比评估
python evaluation/evaluate_activity_predictions_with_gt.py \
  --gt-csv output/activity_state/PF-001_v9_step1_metrics_minvis/activity_state_test.csv \
  --pred-csv output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/temporal_results.csv \
  --out-dir output/evaluation/PF-001_yolo_state_metrics_minvis/metrics \
  --iou-thr 0.5
```

> 最终指标见 `output/evaluation/PF-001_yolo_state_metrics_minvis/metrics/metrics.json`。

### Step 6：Web 演示

```bash
# 手动运行演示管线
python scripts/run_demo_pipeline.py \
  --input-video output/video/PF-002_raw.mp4 \
  --out-dir output/web_demo/manual_pf002_metrics_minvis \
  --weights models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt \
  --activity-classifier models/activity_state_metrics_minvis/best_activity_classifier.pkl \
  --frame-step 1 --state-window-sec 3

# 启动 Web 界面
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

---

## 当前模型指标

| 阶段 | 模型 | 指标 | 值 |
|------|------|------|------|
| Stage 1 | YOLOv8n fish 检测 | mAP50 | **98.23%** |
| Stage 1 | YOLOv8n fish 检测 | Precision / Recall | 99.3% / 95.2% |
| Stage 2 | 活动状态分类器 (ExtraTrees) | 验证集 accuracy | **85.3%** |
| Stage 2 | 活动状态分类器 | 验证集 macro F1 | 0.461 |

> 数据集：PF-001 step5 Mixed；模型权重：`models/fish_yolov8n_640_PF001_step5_mixed_mAP50_0982_best.pt`
>
> 活动状态分类器：`models/activity_state_metrics_minvis/best_activity_classifier.pkl`

---

## 活跃状态类别

| 标签 | 类别名 | 含义 |
|------|--------|------|
| `0` | `low_activity` | 低活跃 —— 鱼体移动极少或静止 |
| `1` | `normal_activity` | 正常活跃 —— 常规游动 |
| `2` | `high_activity` | 高活跃 —— 快速游动/急转弯 |

> ⚠️ 状态标签基于运动特征自动生成，采用指标优先的保守标注策略。low 和 high 只保留特征明显的高置信度样本。标签不代表真实生理状态（缺氧、进食、饥饿等）。

---

## YOLO 鱼体检测模型训练

如需重新训练 YOLO fish 检测模型：

```bash
# 1. 生成 fish 检测数据集
python scripts/generate_fish_detection_dataset.py \
  --seq-dir data/MFT25/MFT25-train/PF-001 \
  --out-dir output/fish_detection/PF-001_yolo_fish_step5 \
  --frame-step 5

# 2. 图像增强
python image_enhancement/enhance_yolo_dataset.py \
  --input-yolo output/fish_detection/PF-001_yolo_fish_step5 \
  --output-yolo output/fish_detection/PF-001_yolo_fish_step5_all_enhanced \
  --enhance-splits all --overwrite

# 3. 生成 Mixed 数据集
python scripts/make_mixed_yolo_dataset.py \
  --original-yolo output/fish_detection/PF-001_yolo_fish_step5 \
  --enhanced-yolo output/fish_detection/PF-001_yolo_fish_step5_all_enhanced \
  --output-yolo output/fish_detection/PF-001_yolo_fish_step5_mixed

# 4. 训练（本地或 Kaggle）
python yolo/train_yolo.py \
  --data output/fish_detection/PF-001_yolo_fish_step5_mixed/data.yaml \
  --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8
```

---

## 数据集说明

| 项目 | 数值 |
|------|------|
| 数据集 | MFT25（Multi-Fish Tracking 2025） |
| 类别 | 1（fish） |
| 图像分辨率 | 1920 × 1080 |
| 帧率 | 25 fps |
| 主训练序列 | PF-001（15,000 帧，~2 条/帧） |

MFT25 原始标注仅提供 `gt.txt`（frame, track_id, x, y, w, h, conf），不提供活跃状态标签。本项目通过运动特征自动生成状态标签。

详见 **[DATA.md](DATA.md)**。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 项目概览、架构、命令速查 |
| [FINAL_METRICS_MINVIS_PIPELINE.md](FINAL_METRICS_MINVIS_PIPELINE.md) | ★ 最终流程详细步骤（7 步） |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 完整项目结构说明 |
| [DATA.md](DATA.md) | MFT25 数据集标注格式与统计 |
| [docs/deliverable_demo_runbook.md](docs/deliverable_demo_runbook.md) | 可交付演示操作手册 |
| [image_enhancement/README.md](image_enhancement/README.md) | 图像增强模块说明 |
| [yolo/README.md](yolo/README.md) | YOLO 模块说明 |
| [temporal_analysis/README.md](temporal_analysis/README.md) | 时序分析模块说明 |
