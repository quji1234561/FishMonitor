# NO_WORK — 不再使用的文件与目录

本文档列出项目中已不再使用的文件和目录，说明其原用途、废弃原因及当前替代方案。仅供参考，可用于后续清理。

---

## 一、废弃脚本（scripts/）

### `scripts/auto_label_fish_activity.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 从 gt.txt 计算速度，按分位数生成 low/normal/high 伪标签 |
| **废弃原因** | 被 `generate_activity_state_dataset.py` 取代。旧脚本仅基于速度分位数分档，缺少运动特征（加速度、窗口位移、急转弯等），标签质量不足以支撑分类器训练 |
| **替代方案** | `scripts/generate_activity_state_dataset.py`（指标优先的多特征标签策略） |

---

## 二、废弃 YOLO 脚本（yolo/）

### `yolo/train_yolo.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 本地训练 YOLO 模型 |
| **废弃原因** | YOLO fish 检测模型在 Kaggle 上训练（GPU 资源更充足），本地未实际使用该脚本训练最终模型 |
| **替代方案** | Kaggle notebook 训练；本地仅保留脚本框架供参考 |
| **如需要** | 修改 `--data` 路径后仍可本地运行 |

### `yolo/predict_yolo.py`

| 项目 | 说明 |
|------|------|
| **原用途** | YOLO 图像预测（无跟踪） |
| **废弃原因** | 当前流水线需要 track_id 用于时序分析，统一使用 `predict_track_yolo.py`（带 ByteTrack 跟踪） |
| **替代方案** | `yolo/predict_track_yolo.py` |

### `yolo/evaluate_yolo.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 评估 YOLO 三分类活跃状态检测模型 |
| **废弃原因** | 当前架构改为「YOLO 单类 fish 检测 + 独立分类器」，不再用 YOLO 做三分类，该评估脚本针对旧架构 |
| **替代方案** | `evaluation/evaluate_activity_predictions_with_gt.py` |

### `yolo/export_model.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 导出 ONNX 等格式 |
| **废弃原因** | 当前仅使用 PyTorch 原生 .pt 权重，未部署到 ONNX 环境 |
| **如需要** | 仍可直接使用 |

---

## 三、废弃时序分析脚本（temporal_analysis/）

### `temporal_analysis/train_speed_thresholds.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 从速度标签学习固定 low/high 速度阈值，输出 `activity_speed_thresholds.json` |
| **废弃原因** | 当前方案用运动特征分类器（ExtraTrees/TCN）替代简单速度阈值比较，分类器利用多维度运动特征，效果好于单一速度阈值 |
| **替代方案** | `temporal_analysis/train_activity_state_classifier.py` |

### `temporal_analysis/train_activity_state_model.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 早期活动状态模型训练入口 |
| **废弃原因** | 被 `train_activity_state_classifier.py` 取代，后者支持 advanced-models、post-windows 网格搜索等完备功能 |
| **替代方案** | `temporal_analysis/train_activity_state_classifier.py` |

### `temporal_analysis/train_activity_state_tcn.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 独立训练 TCN 时序模型 |
| **废弃原因** | 功能已整合到 `train_activity_state_classifier.py` 的 `--advanced-models` 选项中 |
| **替代方案** | `temporal_analysis/train_activity_state_classifier.py --advanced-models` |

---

## 四、废弃评估脚本（evaluation/）

### `evaluation/evaluate_state_recognition.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 旧版状态识别评估 |
| **废弃原因** | 被 `evaluate_activity_predictions_with_gt.py` 取代，新脚本支持 IoU 匹配、混淆矩阵、matched_predictions 等完整输出 |
| **替代方案** | `evaluation/evaluate_activity_predictions_with_gt.py` |

### `evaluation/evaluate_temporal_state_with_gt.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 旧版时序状态评估（基于速度阈值） |
| **废弃原因** | 同样被新评估脚本取代，旧脚本基于简单的速度阈值而非分类器输出 |
| **替代方案** | `evaluation/evaluate_activity_predictions_with_gt.py` |

---

## 五、废弃可视化脚本（scripts/）

### `scripts/generate_activity_video.py`

| 项目 | 说明 |
|------|------|
| **原用途** | 从 `auto_speed_labels_sampled.csv` 生成带状态标签的可视化视频 |
| **废弃原因** | 输入 CSV 来自旧的 `auto_label_fish_activity.py`；当前演示视频由 `run_demo_pipeline.py` 和 `analyze_temporal_states.py` 内置生成 |
| **如需要** | 修改 CSV 路径后仍可用于快速可视化 |

---

## 六、废弃模型目录

### `models/activity_state/`

| 项目 | 说明 |
|------|------|
| **原用途** | 存放早期训练的活动状态模型（`best_activity_model.json`、`last_activity_model.json` 等） |
| **废弃原因** | 这些是旧标签策略（分位数 / step1 / visible）训练的模型，指标未达标或不适用 |
| **当前模型** | `models/activity_state_metrics_minvis/` |

### `models/temp/`

| 项目 | 说明 |
|------|------|
| **原用途** | 临时模型存放 |
| **废弃原因** | 临时目录，无实际使用 |

---

## 七、废弃输出目录

### `output/activity/`

| 项目 | 说明 |
|------|------|
| **原用途** | `auto_label_fish_activity.py` 的输出（BT-001 的 yolo_dataset 和速度标签 CSV） |
| **废弃原因** | 旧脚本输出，当前使用 `output/activity_state/` |

### `output/yolo_runs/`

| 项目 | 说明 |
|------|------|
| **原用途** | 本地 YOLO 训练和预测输出（fish_activity_train、fish_activity_predict 等） |
| **废弃原因** | 旧三分类 YOLO 训练产物，检测效果不理想；当前 YOLO 训练在 Kaggle，推理结果在 `output/activity_state_yolo/` |
| **子目录** | `fish_activity_train/`、`fish_activity_predict/`、`fish_activity_eval_val/`、`new/` 均为废弃产物 |

### `output/temporal/BT-001/`

| 项目 | 说明 |
|------|------|
| **原用途** | BT-001 旧时序分析结果 |
| **废弃原因** | BT-001 仅为早期实验序列，当前主序列为 PF-001 |
| **替代目录** | `output/evaluation/PF-001_yolo_state_metrics_minvis/temporal/` |

### `output/activity_state/` 旧版本子目录

| 子目录 | 状态 |
|--------|------|
| `PF-001/` | 旧版（无 step 标签） |
| `PF-001_step1/` | 旧版（较早的 step1） |
| `PF-001_v9/` | 旧版 |
| `PF-001_v9_step1/` | 旧版（v9 代 step1） |
| `PF-001_v9_step1_low_more/` | 旧版（尝试更多 low 样本） |
| `PF-001_v9_step1_visible/` | 旧版 |
| **`PF-001_v9_step1_metrics_minvis/`** | ✅ **当前版本** |

---

## 八、废弃配置文件

### `configs/activity_speed_thresholds.json`

| 项目 | 说明 |
|------|------|
| **原用途** | 旧 `train_speed_thresholds.py` 输出的固定速度阈值 |
| **废弃原因** | 当前方案用分类器替代简单速度阈值比较 |
| **仍保留** | `analyze_temporal_states.py` 的 `--state-source speed` 模式仍需此文件（备选模式），建议保留 |

---

## 九、废弃文档

### `docs/temporal_threshold_training.md`

| 项目 | 说明 |
|------|------|
| **原用途** | 速度阈值训练说明文档 |
| **废弃原因** | 速度阈值方案已废弃，该文档内容过时 |

---

## 清理建议

### 可直接删除的目录

```
output/activity/                    # 旧 auto_label 输出
output/yolo_runs/                   # 旧三分类 YOLO 训练产物
output/temporal/BT-001/             # 旧 BT-001 时序分析
output/activity_state/PF-001/       # 旧版状态标签
output/activity_state/PF-001_step1/ # 旧版状态标签
output/activity_state/PF-001_v9/    # 旧版状态标签
output/activity_state/PF-001_v9_step1/           # 旧版状态标签
output/activity_state/PF-001_v9_step1_low_more/  # 旧版状态标签
output/activity_state/PF-001_v9_step1_visible/   # 旧版状态标签
models/temp/                        # 临时目录
models/activity_state/              # 旧模型（如已复制 best_activity_classifier.pkl 到此则保留）
```

### 可直接删除的文件

```
docs/temporal_threshold_training.md
```

### 建议保留但不再使用的文件

以下文件不再参与主流程，但作为参考代码保留：

```
scripts/auto_label_fish_activity.py
scripts/generate_activity_video.py
yolo/train_yolo.py
yolo/predict_yolo.py
yolo/evaluate_yolo.py
yolo/export_model.py
temporal_analysis/train_speed_thresholds.py
temporal_analysis/train_activity_state_model.py
temporal_analysis/train_activity_state_tcn.py
evaluation/evaluate_state_recognition.py
evaluation/evaluate_temporal_state_with_gt.py
```

---

## 总结

| 类别 | 数量 | 处理建议 |
|------|------|----------|
| 可删除的废弃输出目录 | ~9 个 | 删除以释放磁盘空间 |
| 可删除的废弃模型目录 | 2 个 | 确认不需要后删除 |
| 可删除的废弃文档 | 1 个 | 直接删除 |
| 保留作参考的旧脚本 | 11 个 | 不移除，但不参与主流程 |
| **当前活跃文件** | **~16 个** | 见 FINAL_METRICS_MINVIS_PIPELINE.md |
