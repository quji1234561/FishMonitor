#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量增强 YOLO 数据集中的图片，并生成一个新的 YOLO 数据集。

默认策略：
1. 只增强 train 图片；
2. val/test 图片保持原图复制；
3. labels 原样复制，因为 CLAHE、Gamma、去噪、锐化都不改变鱼框位置；
4. 自动生成新的 data.yaml。

如果检测链路默认包含图像增强预处理，可以使用：
--enhance-splits all
将 train/val/test 全部增强，再用增强后的 val/test 评估检测效果。

推荐放置位置：
FishMonitor/image_enhancement/enhance_yolo_dataset.py

推荐运行：
python image_enhancement/enhance_yolo_dataset.py ^
  --input-yolo output/activity/BT-001/yolo_dataset ^
  --output-yolo output/enhanced/BT-001_yolo_enhanced

Linux / PowerShell 可使用反斜杠换行：
python image_enhancement/enhance_yolo_dataset.py \
  --input-yolo output/activity/BT-001/yolo_dataset \
  --output-yolo output/enhanced/BT-001_yolo_enhanced
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Set

import cv2

# 兼容两种运行方式：
# 1. python image_enhancement/enhance_yolo_dataset.py
# 2. 在 image_enhancement 目录下 python enhance_yolo_dataset.py
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from enhancement_methods import enhance_underwater_bgr  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = {
    0: "low_activity",
    1: "normal_activity",
    2: "high_activity",
}


def iter_images(image_dir: Path) -> Iterable[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_label_dir(input_yolo: Path, output_yolo: Path, split: str) -> None:
    src_dir = input_yolo / "labels" / split
    dst_dir = output_yolo / "labels" / split
    ensure_dir(dst_dir)

    if not src_dir.exists():
        print(f"[WARN] 标签目录不存在，跳过: {src_dir}")
        return

    for label_path in sorted(src_dir.glob("*.txt")):
        shutil.copy2(label_path, dst_dir / label_path.name)


def copy_images(input_yolo: Path, output_yolo: Path, split: str) -> None:
    src_dir = input_yolo / "images" / split
    dst_dir = output_yolo / "images" / split
    ensure_dir(dst_dir)

    if not src_dir.exists():
        print(f"[WARN] 图片目录不存在，跳过: {src_dir}")
        return

    images = list(iter_images(src_dir))
    for i, img_path in enumerate(images, start=1):
        shutil.copy2(img_path, dst_dir / img_path.name)
        if i % 100 == 0:
            print(f"[INFO] copy {split}: {i}/{len(images)}")

    print(f"[OK] {split} 原图复制完成: {len(images)} 张")


def enhance_images(input_yolo: Path, output_yolo: Path, split: str, args: argparse.Namespace) -> None:
    src_dir = input_yolo / "images" / split
    dst_dir = output_yolo / "images" / split
    ensure_dir(dst_dir)

    if not src_dir.exists():
        print(f"[WARN] 图片目录不存在，跳过: {src_dir}")
        return

    images = list(iter_images(src_dir))
    if not images:
        print(f"[WARN] 没有找到图片: {src_dir}")
        return

    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图片，跳过: {img_path}")
            continue

        enhanced = enhance_underwater_bgr(
            img,
            use_clahe=not args.no_clahe,
            use_gamma=not args.no_gamma,
            use_denoise=not args.no_denoise,
            use_sharpen=not args.no_sharpen,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_grid_size=args.clahe_tile_grid_size,
            gamma=args.gamma,
            denoise_h=args.denoise_h,
            denoise_h_color=args.denoise_h_color,
            sharpen_amount=args.sharpen_amount,
        )

        out_path = dst_dir / img_path.name
        ok = cv2.imwrite(str(out_path), enhanced)
        if not ok:
            print(f"[WARN] 保存失败: {out_path}")

        if i % 50 == 0 or i == len(images):
            print(f"[INFO] enhance {split}: {i}/{len(images)}")

    print(f"[OK] {split} 图像增强完成: {len(images)} 张")


def read_names_block(data_yaml: Path) -> str:
    """
    尽量从原始 data.yaml 中保留 names 部分。
    如果读取失败，就使用默认三类活跃状态。
    """
    if not data_yaml.exists():
        return "names:\n  0: low_activity\n  1: normal_activity\n  2: high_activity\n"

    text = data_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        if line.strip().startswith("names:"):
            return "\n".join(lines[idx:]).rstrip() + "\n"

    return "names:\n  0: low_activity\n  1: normal_activity\n  2: high_activity\n"


def write_data_yaml(input_yolo: Path, output_yolo: Path) -> None:
    names_block = read_names_block(input_yolo / "data.yaml")
    yaml_text = f"""path: {output_yolo.as_posix()}
train: images/train
val: images/val
test: images/test

{names_block}"""
    out_path = output_yolo / "data.yaml"
    out_path.write_text(yaml_text, encoding="utf-8")
    print(f"[OK] 已生成 data.yaml: {out_path}")


def write_enhance_summary(output_yolo: Path, args: argparse.Namespace) -> None:
    enhance_splits = normalize_enhance_splits(args.enhance_splits)
    text = f"""图像增强数据集说明
====================

输入数据集: {args.input_yolo}
输出数据集: {args.output_yolo}
增强策略: 增强 {', '.join(sorted(enhance_splits))}；其他 split 保持原图
标签处理: labels 原样复制，不改变 bbox

增强方法:
- CLAHE: {'关闭' if args.no_clahe else '开启'}, clip_limit={args.clahe_clip_limit}, tile_grid_size={args.clahe_tile_grid_size}
- Gamma 校正: {'关闭' if args.no_gamma else '开启'}, gamma={args.gamma}
- 去噪: {'关闭' if args.no_denoise else '开启'}, h={args.denoise_h}, h_color={args.denoise_h_color}
- 锐化: {'关闭' if args.no_sharpen else '开启'}, amount={args.sharpen_amount}

说明:
本增强流程只进行像素级处理，不进行裁剪、旋转、翻转等几何变换，
因此不会改变鱼体框位置，YOLO 标签可以直接复用。
"""
    out_path = output_yolo / "enhance_summary.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] 已生成增强说明: {out_path}")


def check_input_yolo(input_yolo: Path) -> None:
    if not input_yolo.exists():
        raise FileNotFoundError(f"输入 YOLO 数据集不存在: {input_yolo}")
    if not (input_yolo / "images").exists():
        raise FileNotFoundError(f"缺少 images 目录: {input_yolo / 'images'}")
    if not (input_yolo / "labels").exists():
        raise FileNotFoundError(f"缺少 labels 目录: {input_yolo / 'labels'}")


def normalize_enhance_splits(raw_splits: List[str]) -> Set[str]:
    splits = set(raw_splits)
    if "all" in splits:
        return {"train", "val", "test"}
    return splits


def zip_yolo_dataset(output_yolo: Path) -> Path:
    zip_path = output_yolo.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_yolo.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(output_yolo.parent)))

    print(f"[OK] 已打包 Kaggle 上传 zip: {zip_path}")
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增强 YOLO 数据集，并保持 YOLO 标签不变")

    parser.add_argument("--input-yolo", required=True, help="原始 YOLO 数据集目录")
    parser.add_argument("--output-yolo", required=True, help="增强后 YOLO 数据集输出目录")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，先删除再重新生成",
    )
    parser.add_argument(
        "--enhance-splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test", "all"],
        help="需要增强的 split。默认只增强 train；检测链路默认包含增强预处理时可设为 all",
    )
    parser.add_argument(
        "--zip-output",
        action="store_true",
        help="生成完成后，将整个 YOLO 数据集打包成 zip，方便上传 Kaggle",
    )

    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE 对比度限制")
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8, help="CLAHE 网格大小")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Gamma 值。gamma<1 变亮，gamma>1 变暗，默认 0.8",
    )
    parser.add_argument("--denoise-h", type=float, default=5.0, help="去噪强度，建议 3~7")
    parser.add_argument("--denoise-h-color", type=float, default=5.0, help="彩色去噪强度，建议 3~7")
    parser.add_argument("--sharpen-amount", type=float, default=0.8, help="锐化强度，建议 0.5~1.2")

    parser.add_argument("--no-clahe", action="store_true", help="关闭 CLAHE")
    parser.add_argument("--no-gamma", action="store_true", help="关闭 Gamma 校正")
    parser.add_argument("--no-denoise", action="store_true", help="关闭去噪")
    parser.add_argument("--no-sharpen", action="store_true", help="关闭锐化")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_yolo = Path(args.input_yolo)
    output_yolo = Path(args.output_yolo)

    check_input_yolo(input_yolo)

    if output_yolo.exists():
        if args.overwrite:
            shutil.rmtree(output_yolo)
        else:
            raise FileExistsError(
                f"输出目录已存在: {output_yolo}\n"
                f"如需覆盖，请添加 --overwrite"
            )

    for split in ["train", "val", "test"]:
        ensure_dir(output_yolo / "images" / split)
        ensure_dir(output_yolo / "labels" / split)

    print("[INFO] 开始处理 YOLO 数据集")
    print(f"[INFO] input : {input_yolo}")
    print(f"[INFO] output: {output_yolo}")
    enhance_splits = normalize_enhance_splits(args.enhance_splits)
    print(f"[INFO] 策略: 增强 {sorted(enhance_splits)}，其他 split 保持原图")

    for split in ["train", "val", "test"]:
        if split in enhance_splits:
            enhance_images(input_yolo, output_yolo, split, args)
        else:
            copy_images(input_yolo, output_yolo, split)
        copy_label_dir(input_yolo, output_yolo, split)

    write_data_yaml(input_yolo, output_yolo)
    write_enhance_summary(output_yolo, args)

    if args.zip_output:
        zip_yolo_dataset(output_yolo)

    print("[DONE] 增强版 YOLO 数据集生成完成。")
    print(f"[NEXT] 可用于训练: yolo detect train data={output_yolo.as_posix()}/data.yaml model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == "__main__":
    main()
