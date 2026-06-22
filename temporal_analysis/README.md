# 时序分析与状态分类模块

本模块是 FishMonitor 的 **Stage 2 核心推理模块**，负责将 YOLO 的鱼体检测结果转化为活跃状态判定。支持两种状态判定模式，并提供完整的分类器训练管线。

## 文件说明

| 文件 | 状态 | 说明 |
|------|------|------|
| `analyze_temporal_states.py` | ★ 核心 | 时序分析 & 分类器推理入口 |
| `train_activity_state_classifier.py` | ★ 核心 | 训练活动状态分类器（ExtraTrees 等） |
| `motion_features.py` | ★ 核心 | 运动特征提取（速度/加速度/位移/形态变化） |
| `activity_classifier.py` | ★ 核心 | 分类器包装、加载、预测接口 |
| `activity_tcn.py` | 辅助 | TCN 时序模型实现 |
| `train_activity_state_model.py` | 遗留 | 旧训练入口，被 `train_activity_state_classifier.py` 取代 |
| `train_activity_state_tcn.py` | 遗留 | 独立 TCN 训练，已整合到主训练脚本 |
| `train_speed_thresholds.py` | 遗留 | 旧速度阈值方案，被分类器方案取代 |

> 标注为"遗留"的脚本不在当前主流水线中使用，保留供参考。

## 架构

```
YOLO 预测框 CSV (raw_predictions.csv)
        │
        ▼
motion_features.py
  提取每条鱼的运动特征：
  - 速度 (speed_norm_s, speed_smooth)
  - 加速度 (acceleration)
  - 窗口位移 (displacement)
  - 框形态变化 (aspect_ratio_change, area_change)
  - 急转弯事件 (sharp_turn)
  - 群体运动水平 (group_motion)
        │
        ▼
┌──────────────────────────────────────┐
│  状态判定 (analyze_temporal_states.py) │
│                                      │
│  三种模式 (--state-source)：           │
│  1. speed             速度阈值判定    │
│  2. activity_classifier  分类器判定 ★ │
│  3. (默认)              YOLO 预测标签  │
└──────────────────────────────────────┘
        │
        ▼
temporal_results.csv        每条鱼每帧的最终状态
frame_activity_summary.csv  每帧活跃度统计
activity_trend.png          活跃度趋势图
temporal_smoothed_video.mp4 结果视频
```

## 核心脚本

### analyze_temporal_states.py — 推理入口

对 YOLO 跟踪预测结果做时序状态判定。

**模式 1：分类器判定（当前主方案）★**

```bash
python temporal_analysis/analyze_temporal_states.py \
  --pred-csv output/activity_state_yolo/PF-001_v9_step1/tracking_all/raw_predictions.csv \
  --out-dir output/evaluation/PF-001_yolo_state_metrics_minvis/temporal \
  --state-source activity_classifier \
  --activity-classifier models/activity_state_metrics_minvis/best_activity_classifier.pkl \
  --fps 25 \
  --window 75
```

**模式 2：速度阈值判定（备选方案）**

```bash
python temporal_analysis/analyze_temporal_states.py \
  --pred-csv raw_predictions.csv \
  --out-dir output/temporal/ \
  --state-source speed \
  --threshold-file configs/activity_speed_thresholds.json \
  --fps 25 --window 5
```

| 参数 | 说明 |
|------|------|
| `--pred-csv` | YOLO 跟踪预测 CSV（来自 `predict_track_yolo.py`） |
| `--out-dir` | 输出目录 |
| `--state-source` | `activity_classifier` / `speed` / 默认 (YOLO) |
| `--activity-classifier` | 分类器 PKL 文件路径 |
| `--threshold-file` | 速度阈值 JSON（仅 `--state-source speed`） |
| `--fps` | 帧率 |
| `--window` | 滑动窗口大小 |
| `--save-video` | 输出可视化视频 |

### train_activity_state_classifier.py — 分类器训练

基于运动特征训练 low/normal/high 三分类模型。

```bash
python temporal_analysis/train_activity_state_classifier.py \
  --train-csv output/activity_state_yolo/.../matched_all/yolo_activity_state_train.csv \
  --val-csv output/activity_state_yolo/.../matched_all/yolo_activity_state_val.csv \
  --feature-config output/activity_state/PF-001_v9_step1_metrics_minvis/feature_config.json \
  --out-dir models/activity_state_metrics_minvis \
  --metric accuracy \
  --advanced-models \
  --post-windows 1,15,30,45,75 \
  --min-vote-ratios 0.5,0.6,0.7
```

#### 训练流程

1. 从 CSV 提取运动特征（`motion_features.py`）
2. 网格搜索最优分类器（ExtraTrees / RandomForest / GradientBoosting / TCN）
3. 搜索最优后处理参数（窗口大小 + 多数投票比例）
4. 验证集评估 + non-normal 比例约束
5. 保存最佳模型为 `best_activity_classifier.pkl`

#### 核心参数

| 参数 | 说明 |
|------|------|
| `--train-csv` / `--val-csv` | 训练/验证 CSV |
| `--feature-config` | 特征配置 JSON |
| `--out-dir` | 模型输出目录 |
| `--metric` | 优化指标（默认 `accuracy`） |
| `--advanced-models` | 启用 ExtraTrees/RF/GB 等高级模型 |
| `--post-windows` | 后处理窗口大小候选 |
| `--min-vote-ratios` | 多数投票比例候选 |
| `--min-val-non-normal-ratio` | 验证集 non-normal 最低比例（防退化） |

## 输出文件

```
<out_dir>/
├── temporal_results.csv             # 每条鱼每帧的最终状态
│   (frame, track_id, bbox, state_id, state_name, speed, ...)
├── frame_activity_summary.csv       # 每帧状态统计
│   (fish_count, low_count, normal_count, high_count, activity_index, ...)
├── activity_trend.png               # 活跃度趋势图
├── temporal_summary.txt             # 分析摘要
└── temporal_smoothed_video.mp4      # 结果视频（可选）
```

## 训练输出

```
models/<model_name>/
├── best_activity_classifier.pkl     # 最佳分类器
├── last_activity_classifier.pkl     # 最后一次训练的分类器
├── best_classifier_metrics.json     # 最佳模型指标详情
└── classifier_train_log.csv         # 训练日志
```

## 在流水线中的位置

```
┌───────────────────────────────────┐
│  Stage 1: YOLO (yolo/)             │
│  → raw_predictions.csv             │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Stage 2: 时序分析 (本模块)         │
│  → temporal_results.csv            │
│  → frame_activity_summary.csv      │
│  → 结果视频                         │
└───────────────────────────────────┘
```

详见项目主文档 [FINAL_METRICS_MINVIS_PIPELINE.md](../FINAL_METRICS_MINVIS_PIPELINE.md)。
