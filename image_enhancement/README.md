# 图像增强模块

对 YOLO 训练集图像进行水下图像增强，提升模型对低对比度、模糊水下场景的泛化能力。

## 增强策略

| 数据划分 | 处理方式 | 原因 |
|----------|----------|------|
| train | 增强 | 提升训练数据多样性 |
| val | 保持原图 | 保证评估真实性 |
| test | 保持原图 | 保证评估真实性 |
| labels | 全部原样复制 | 增强不改变几何位置，标签可直接复用 |

> 本模块只做 CLAHE、Gamma 校正、去噪、锐化等像素级处理，不进行裁剪、旋转、翻转等几何变换，鱼框位置不变。

## 文件说明

```
image_enhancement/
├── enhancement_methods.py      # CLAHE、Gamma、去噪、锐化等增强算法
├── enhance_yolo_dataset.py     # 批量生成增强版 YOLO 数据集
└── README.md                   # 本说明文件
```

## 使用方法

从项目根目录运行。

### 仅增强 train（推荐给活跃状态分类用）

```bash
python image_enhancement/enhance_yolo_dataset.py \
  --input-yolo output/activity/BT-001/yolo_dataset \
  --output-yolo output/enhanced/BT-001_yolo_enhanced
```

### 全增强（fish 检测数据集用）

用于生成 YOLO fish 检测的 Mixed 数据集时，需要全部增强（train + val + test）：

```bash
python image_enhancement/enhance_yolo_dataset.py \
  --input-yolo output/fish_detection/PF-001_yolo_fish_step5 \
  --output-yolo output/fish_detection/PF-001_yolo_fish_step5_all_enhanced \
  --enhance-splits all \
  --overwrite
```

覆盖已有输出加 `--overwrite`。

## 输出结构

```
<output_dir>/
├── images/
│   ├── train/              # 增强后的训练图像
│   ├── val/                # 原始验证图像（仅增强 train 时）
│   └── test/               # 原始测试图像（仅增强 train 时）
├── labels/
│   ├── train/              # 原样复制的标签
│   ├── val/
│   └── test/
├── data.yaml               # 自动更新的 YOLO 配置
└── enhance_summary.txt     # 增强参数记录
```

## 可调参数

```bash
--gamma 0.8                 # Gamma 校正：<1 变亮，>1 变暗
--clahe-clip-limit 2.0      # CLAHE 对比度限制
--clahe-tile-grid-size 8    # CLAHE 网格大小
--denoise-h 5               # 去噪强度
--sharpen-amount 0.8        # 锐化强度
```

关闭某个增强步骤：

```bash
--no-clahe
--no-gamma
--no-denoise
--no-sharpen
```

## 在流水线中的位置

```
generate_fish_detection_dataset.py → 原图 YOLO 数据集
  ↓
enhance_yolo_dataset.py             → 全增强数据集
  ↓
make_mixed_yolo_dataset.py          → Mixed 训练集 (原图 train + 增强 train)
  ↓
Kaggle / 本地 YOLO 训练
```

详见项目主文档 [FINAL_METRICS_MINVIS_PIPELINE.md](../FINAL_METRICS_MINVIS_PIPELINE.md)。
