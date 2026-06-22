# FishMonitor 数据集说明

本项目使用 **MFT25**（Multi-Fish Tracking 2025）水下鱼类多目标跟踪数据集。

---

## 项目结构

```
FishMonitor/
├── data/
│   ├── MFT25/                          # 多目标跟踪数据集
│   │   ├── train.json                  # 完整训练集 COCO 标注 (45.6 MB)
│   │   ├── train_half.json             # 训练集前半拆分 (22.6 MB)
│   │   ├── val_half.json               # 训练集后半拆分 / 验证集 (22.7 MB)
│   │   ├── MFT25-train/                # 训练图像 + MOT 格式标注
│   │   │   ├── BT-001/                 # 序列 1
│   │   │   ├── BT-003/                 # 序列 2
│   │   │   ├── BT-005/                 # 序列 3
│   │   │   ├── MSK-002/                # 序列 4
│   │   │   ├── PF-001/                 # 序列 5
│   │   │   ├── SN-001/                 # 序列 6
│   │   │   ├── SN-013/                 # 序列 7
│   │   │   └── SN-015/                 # 序列 8
│   │   └── MFT25-test/                 # 测试图像（无标注）
│   │       ├── BT-002/
│   │       ├── BT-004/
│   │       ├── MSK-003/
│   │       ├── PF-002/
│   │       ├── SN-009/
│   │       ├── SN-011/
│   │       └── SN-014/
│   ├── play.py                         # 帧序列播放器
│   ├── play2.py                        # COCO JSON 标注查看器
│   └── play3.py                        # 检测 vs 真值对比查看器
│
├── scripts/
│   └── generate_annotated_video.py     # 生成带标注框的视频
│
├── output/
│   └── video/                          # 生成的视频输出目录
│
└── DATA.md                             # 本文件
```

---

## 概述

- **任务**：多目标跟踪（Multi-Object Tracking, MOT）
- **类别数**：1（`fish`）
- **训练序列**：8 个
- **测试序列**：7 个
- **图像分辨率**：1920 × 1080
- **帧率**：25 fps
- **图像格式**：JPG
- **总训练帧数**：32,085
- **总标注数**：263,666
- **唯一跟踪 ID 数**：263,659

### 场景前缀含义

| 前缀 | 含义推测 | 场景特征 |
|------|----------|----------|
| **BT** | Bait（诱饵） | 有诱饵装置吸引鱼群，鱼群较集中 |
| **MSK** | Mask（遮蔽/掩体） | 有礁石或结构物，鱼群密度高 |
| **PF** | Pelagic Fish（远洋鱼类） | 开阔水域，目标稀疏 |
| **SN** | Seafloor/Scene（海底） | 海底场景，鱼群密度中等 |

---

## 每个序列的内部结构（以 BT-001 为例）

```
BT-001/
├── seqinfo.ini          # 序列元数据
├── img1/                # 帧图像 (000001.jpg, 000002.jpg, ...)
├── gt/                  # 真值标注
│   ├── gt.txt           # 完整真值
│   ├── gt_train_half.txt
│   └── gt_val_half.txt
└── det/                 # 检测结果
    ├── det.txt
    ├── det_train_half.txt
    └── det_val_half.txt
```

测试序列仅包含 `img1/` 和 `seqinfo.ini`，无 `gt/` 和 `det/` 目录。

### 序列元数据（seqinfo.ini）

```ini
[Sequence]
name=BT-001
imDir=img1
frameRate=25
seqLength=3000
imWidth=1920
imHeight=1080
imExt=.jpg
```

---

## 标注格式

### COCO JSON（train.json / train_half.json / val_half.json）

```json
{
  "images": [
    {
      "file_name": "BT-001/img1/000001.jpg",
      "id": 1,
      "frame_id": 1,
      "prev_image_id": -1,
      "next_image_id": 2,
      "video_id": 1,
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "category_id": 1,
      "image_id": 1,
      "track_id": 1,
      "bbox": [709.0, 520.0, 63.0, 204.0],
      "conf": 1.0,
      "iscrowd": 0,
      "area": 12852.0
    }
  ],
  "videos": [
    {"id": 1, "file_name": "BT-001"}
  ],
  "categories": [
    {"id": 1, "name": "fish"}
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox` | `[x, y, w, h]` | 边界框（左上角坐标 + 宽高） |
| `track_id` | int | 跨帧跟踪 ID，同一鱼在所有帧中保持相同 ID |
| `category_id` | int | 固定为 1（fish） |
| `conf` | float | 置信度（真值全为 1.0） |
| `iscrowd` | int | 是否为密集目标（全为 0） |
| `area` | float | 边界框面积（w × h） |
| `prev_image_id` / `next_image_id` | int | 前后帧链接（-1 表示首/尾帧） |

### MOT TXT 格式（gt.txt / det.txt）

```
<frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <1>, <1>
```

示例：
```
1,1,709,520,63,204,1,1,1
1,2,832,460,168,68,1,1,1
```

> 注意：MOT 格式中 `<id>` 即 track_id，且每一帧每条鱼一行。det.txt 与 gt.txt 格式完全相同，但 det.txt 来自检测器输出。

---

## 各序列详细统计

| 序列 | 帧数 | 总标注数 | 唯一 track | 平均鱼数/帧 | 说明 |
|------|------|----------|------------|-------------|------|
| BT-001 | 3,000 | 30,000 | 10 | 10.0 | 诱饵场景，鱼群稳定 |
| BT-003 | 243 | 2,430 | 10 | 10.0 | 短序列 |
| BT-005 | 1,834 | 18,340 | 10 | 10.0 | — |
| MSK-002 | 1,754 | 85,385 | 49 | 48.7 | 密度最高（掩体/遮蔽场景） |
| PF-001 | 15,000 | 30,000 | 2 | 2.0 | 最长序列，仅 2 条鱼（远洋场景） |
| SN-001 | 684 | 6,632 | 13 | 9.7 | — |
| SN-013 | 3,000 | 28,125 | 11 | 9.4 | — |
| SN-015 | 6,570 | 62,754 | 14 | 9.6 | — |
| **合计** | **32,085** | **263,666** | — | **~8.2** | — |

### 标注统计

- **bbox 面积范围**：120 ~ 426,700 像素
- **iscrowd**：全部为 0（无密集目标标记）
- **conf**：真值全为 1.0
- **各帧鱼数范围**：2 ~ 49 条

---

## Train / Val / Test 划分

| 划分 | 序列 | 帧数 | 标注文件 |
|------|------|------|----------|
| **Train** | BT-001, BT-003, BT-005, MSK-002, PF-001, SN-001, SN-013, SN-015 | 32,085 | `train.json` |
| **Train Half** | 上述序列的前半帧 | 16,050 | `train_half.json` |
| **Val Half** | 上述序列的后半帧 | 16,035 | `val_half.json` |
| **Test** | BT-002, BT-004, MSK-003, PF-002, SN-009, SN-011, SN-014 | 15,985 | 无（仅图像） |

---

## 工具脚本

### 数据浏览（`data/`）

| 脚本 | 功能 | 用法 |
|------|------|------|
| `data/play.py` | 帧序列播放器 | 逐帧显示序列 JPG 图像，支持暂停/前进/后退 |
| `data/play2.py` | COCO JSON 标注查看器 | 加载 JSON 标注，在帧上绘制边界框并浏览 |
| `data/play3.py` | 检测 vs 真值对比 | 同时读取 det.txt 和 gt.txt，对比显示差异 |

**play.py 控件**：
- `Space` — 暂停/继续
- `A` / `D` — 上一帧 / 下一帧
- `Q` — 退出

**play2.py 控件**：
- 任意键 — 下一帧
- `Q` — 退出

**play3.py 控件**：
- `Space` — 暂停/继续
- 方向键 — 手动浏览帧（检测与真值同框对比，重叠区域黄色高亮）

### 视频生成（`scripts/`）

| 脚本 | 功能 |
|------|------|
| `scripts/generate_annotated_video.py` | 读取序列帧和标注，生成带标注框的 MP4 视频 |

```bash
# 单个序列（使用 gt.txt）
python scripts/generate_annotated_video.py -s BT-001

# 使用 COCO JSON 标注
python scripts/generate_annotated_video.py -s BT-001 --json data/MFT25/train.json

# 批量生成所有训练序列
python scripts/generate_annotated_video.py --all

# 指定帧范围 + 细框线（高密度场景）
python scripts/generate_annotated_video.py -s MSK-002 --start 100 --end 500 --thin

# 自定义帧率和编码器
python scripts/generate_annotated_video.py -s BT-001 --fps 30 --codec avc1
```

详细参数见 `python scripts/generate_annotated_video.py --help`。

---

## 数据来源与许可

待补充。

---

## 更新记录

| 日期 | 说明 |
|------|------|
| 2026-06-01 | 初始版本：完成数据集结构和标注格式说明 |
| 2026-06-01 | 移除 fish_image 分类数据集，新增 `scripts/generate_annotated_video.py` 说明 |
