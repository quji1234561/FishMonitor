#!/usr/bin/env python3
"""
根据 MFT25 数据集的帧图像和标注，生成带标注框的视频。

支持两种标注来源：
  1. MOT txt 格式 (gt/gt.txt)        —— 每个序列独立，速度快
  2. COCO JSON 格式 (train.json 等)   —— 跨序列统一文件

用法示例：
  # 使用 MOT txt 标注（默认）
  python scripts/generate_annotated_video.py -s BT-001

  # 使用 COCO JSON 标注
  python scripts/generate_annotated_video.py -s BT-001 --json data/MFT25/train.json

  # 批量生成所有训练序列
  python scripts/generate_annotated_video.py --all

  # 指定帧范围
  python scripts/generate_annotated_video.py -s PF-001 --start 100 --end 500
"""

import cv2
import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# 修复 Windows GBK 终端编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# 颜色生成 —— 为每个 track_id 分配稳定且区分度高的颜色
# ============================================================

def generate_track_colors(track_ids):
    """为每个 track_id 生成一个 BGR 颜色（HSV 色相均匀分布）。"""
    n = max(len(track_ids), 1)
    colors = {}
    for i, tid in enumerate(sorted(track_ids)):
        hue = int(180 * i / n)  # 0~179
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[tid] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return colors


# ============================================================
# 标注加载
# ============================================================

def load_annotations_from_mot(txt_path):
    """
    从 MOT txt 格式加载标注。
    格式: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <1>, <1>

    Returns:
        frame_anns: dict {frame_num: [{track_id, bbox: [x,y,w,h]}, ...]}
        track_ids: set of all track IDs
    """
    frame_anns = defaultdict(list)
    track_ids = set()

    if not os.path.exists(txt_path):
        print(f"  [警告] 标注文件不存在: {txt_path}")
        return frame_anns, track_ids

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                frame_num = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = map(int, map(float, parts[2:6]))
            except ValueError:
                continue

            frame_anns[frame_num].append({
                'track_id': track_id,
                'bbox': [x, y, w, h],
            })
            track_ids.add(track_id)

    return frame_anns, track_ids


def load_annotations_from_json(json_path, seq_name):
    """
    从 COCO JSON 格式加载指定序列的标注。

    Returns:
        frame_anns: dict {frame_num: [{track_id, bbox: [x,y,w,h]}, ...]}
        track_ids: set of all track IDs
    """
    frame_anns = defaultdict(list)
    track_ids = set()

    if not os.path.exists(json_path):
        print(f"  [警告] JSON 文件不存在: {json_path}")
        return frame_anns, track_ids

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 筛选属于目标序列的 image
    seq_images = [img for img in data['images'] if seq_name in img['file_name']]
    if not seq_images:
        print(f"  [警告] JSON 中未找到序列 {seq_name} 的数据")
        return frame_anns, track_ids

    # 构建 image_id -> frame_num 映射
    img_id_to_frame = {}
    for img in seq_images:
        frame_num = int(os.path.splitext(os.path.basename(img['file_name']))[0])
        img_id_to_frame[img['id']] = frame_num

    for ann in data['annotations']:
        if ann['image_id'] in img_id_to_frame:
            frame_num = img_id_to_frame[ann['image_id']]
            bbox = ann['bbox']  # [x, y, w, h]
            track_id = ann.get('track_id', ann.get('id', 0))
            frame_anns[frame_num].append({
                'track_id': track_id,
                'bbox': [int(v) for v in bbox],
            })
            track_ids.add(track_id)

    print(f"  JSON 中共找到 {len(seq_images)} 帧, {sum(len(v) for v in frame_anns.values())} 条标注")
    return frame_anns, track_ids


# ============================================================
# 视频生成核心
# ============================================================

def generate_video(
    seq_name,
    data_dir,
    subset,
    output_path,
    frame_anns,
    track_colors,
    fps=25,
    codec='mp4v',
    start_frame=1,
    end_frame=0,
    show_track_id=True,
    show_frame_counter=True,
    draw_boxes=True,
    box_thickness=2,
    font_scale=0.6,
):
    """
    逐帧读取图像、绘制标注框、写入视频。

    Args:
        seq_name: 序列名称
        data_dir: MFT25 数据根目录
        subset: train 或 test，对应 MFT25-train / MFT25-test
        output_path: 输出视频路径
        frame_anns: {frame_num: [{track_id, bbox}, ...]}
        track_colors: {track_id: (B, G, R)}
        fps: 输出帧率
        codec: 四字符编码 (mp4v, avc1, XVID, etc.)
        start_frame: 起始帧号
        end_frame: 结束帧号 (0=全部)
        show_track_id: 是否在框上显示 track ID
        show_frame_counter: 是否显示帧号
        draw_boxes: 是否绘制标注框；False 时生成原始无标注视频
        box_thickness: 框线宽
        font_scale: 字体大小
    """
    img_dir = os.path.join(data_dir, f"MFT25-{subset}", seq_name, "img1")
    if not os.path.isdir(img_dir):
        print(f"  [错误] 图像目录不存在: {img_dir}")
        return False

    # 获取所有帧文件
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    if not img_files:
        print(f"  [错误] 图像目录为空: {img_dir}")
        return False

    total_available = int(os.path.splitext(img_files[-1])[0])

    if end_frame == 0:
        end_frame = max(frame_anns.keys()) if frame_anns else total_available
    end_frame = min(end_frame, total_available)

    # 确定视频尺寸（读第一帧）
    first_img_path = os.path.join(img_dir, f"{start_frame:06d}.jpg")
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"  [错误] 无法读取第一帧: {first_img_path}")
        return False
    height, width = first_img.shape[:2]

    # 初始化 VideoWriter
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"  [错误] 无法创建视频文件: {output_path}")
        print(f"         请检查编码器 {codec} 是否可用")
        return False

    total_frames = end_frame - start_frame + 1
    frames_with_anns = 0
    total_boxes = 0

    print(f"  输出: {output_path}")
    print(f"  帧范围: {start_frame} → {end_frame} (共 {total_frames} 帧)")
    print(f"  分辨率: {width}x{height}, 帧率: {fps} fps")
    print(f"  编码: {codec}, track ID 显示: {'是' if show_track_id else '否'}")

    # 逐帧处理
    for idx, frame_num in enumerate(range(start_frame, end_frame + 1)):
        img_file = f"{frame_num:06d}.jpg"
        img_path = os.path.join(img_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            # 帧文件不存在时写入空白帧（保持视频长度）
            img = np.zeros((height, width, 3), dtype=np.uint8)

        # 绘制标注框
        anns = frame_anns.get(frame_num, [])
        if anns:
            frames_with_anns += 1
            total_boxes += len(anns)

        if draw_boxes:
            for ann in anns:
                tid = ann['track_id']
                x, y, w, h = ann['bbox']

                # 边界检查
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                color = track_colors.get(tid, (0, 255, 0))

                # 绘制矩形框
                cv2.rectangle(img, (x, y), (x + w, y + h), color, box_thickness)

                # 绘制 track ID 标签
                if show_track_id:
                    label = f"ID:{tid}"
                    # 文字背景
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    cv2.rectangle(img,
                                  (x, y - th - baseline - 4),
                                  (x + tw + 4, y),
                                  color, -1)
                    cv2.putText(img, label, (x + 2, y - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # 帧号叠加（左上角）
        if show_frame_counter:
            seq_label = f"{seq_name} | Frame: {frame_num}"
            cv2.putText(img, seq_label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

        # 进度显示（ASCII 字符，兼容所有终端）
        progress = (idx + 1) / total_frames * 100
        bar_len = 30
        filled = int(bar_len * (idx + 1) / total_frames)
        bar = '#' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {progress:5.1f}%  ({idx+1}/{total_frames})")
        sys.stdout.flush()

    writer.release()
    if draw_boxes:
        print(f"\n  完成! {frames_with_anns}/{total_frames} 帧有标注, 共绘制 {total_boxes} 个框")
    else:
        print(f"\n  完成! 已生成无标注原始视频，共写入 {total_frames} 帧")

    # 获取文件大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {size_mb:.1f} MB")
    return True


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成带标注框的 MFT25 鱼群跟踪视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -s BT-001                          # 用 gt.txt 生成 BT-001 视频
  %(prog)s -s BT-001 --json data/MFT25/train.json  # 用 COCO JSON 生成
  %(prog)s --all                               # 批量生成所有训练序列
  %(prog)s -s MSK-002 --start 100 --end 300    # 只生成第 100~300 帧
  %(prog)s -s PF-001 --fps 30 --codec avc1     # 自定义帧率和编码器
  %(prog)s -s PF-001 --raw --start 1 --end 300  # 生成无标注原始视频
        """
    )

    # ---- 输入源 ----
    parser.add_argument('-s', '--seq', default=None,
                        help='序列名称 (如 BT-001, MSK-002, PF-001)')
    parser.add_argument('--all', action='store_true',
                        help='批量生成所有训练序列')
    parser.add_argument('--data-dir', default='data/MFT25',
                        help='MFT25 数据根目录 (默认: data/MFT25)')
    parser.add_argument('--subset', choices=['train', 'test'], default='train',
                        help='数据划分：train 对应 MFT25-train，test 对应 MFT25-test (默认: train)')
    parser.add_argument('--json', default=None,
                        help='COCO JSON 标注文件路径 (不指定则使用 gt/gt.txt)')

    # ---- 输出 ----
    parser.add_argument('-o', '--output-dir', default='output/video',
                        help='输出目录 (默认: output/video)')
    parser.add_argument('--prefix', default='',
                        help='输出文件名前缀')

    # ---- 帧范围 ----
    parser.add_argument('--start', type=int, default=1,
                        help='起始帧号 (默认: 1)')
    parser.add_argument('--end', type=int, default=0,
                        help='结束帧号 (默认: 0=到最后)')

    # ---- 视频参数 ----
    parser.add_argument('--fps', type=int, default=25,
                        help='输出帧率 (默认: 25)')
    parser.add_argument('--codec', default='mp4v',
                        help='编码器四字符码 (默认: mp4v, 也可用 avc1/h264/XVID)')

    # ---- 显示选项 ----
    parser.add_argument('--no-track-id', action='store_true',
                        help='不显示 track ID')
    parser.add_argument('--no-frame-counter', action='store_true',
                        help='不显示帧号')
    parser.add_argument('--raw', action='store_true',
                        help='生成无标注原始视频：不画框、不显示 track ID、不显示帧号')
    parser.add_argument('--thin', action='store_true',
                        help='使用细框线 (适合目标较小的场景如 MSK)')

    args = parser.parse_args()

    # ---- 确定要处理的序列列表 ----
    split_dir = os.path.join(args.data_dir, f"MFT25-{args.subset}")
    if not os.path.isdir(split_dir):
        print(f"[错误] 数据目录不存在: {split_dir}")
        sys.exit(1)

    all_seqs = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d, "img1"))
    ])

    if args.all:
        seq_list = all_seqs
    elif args.seq:
        if args.seq not in all_seqs:
            print(f"[错误] 未找到序列 '{args.seq}'")
            print(f"  可用序列: {', '.join(all_seqs)}")
            sys.exit(1)
        seq_list = [args.seq]
    else:
        parser.print_help()
        print(f"\n可用序列: {', '.join(all_seqs)}")
        sys.exit(1)

    print(f"将处理 {len(seq_list)} 个序列: {', '.join(seq_list)}")
    print()

    # ---- 逐序列生成 ----
    for seq_name in seq_list:
        print(f"{'='*60}")
        print(f"  序列: {seq_name}")
        print(f"{'='*60}")

        # 加载标注
        if args.json:
            print(f"  标注来源: COCO JSON ({args.json})")
            frame_anns, track_ids = load_annotations_from_json(args.json, seq_name)
        else:
            gt_path = os.path.join(split_dir, seq_name, "gt", "gt.txt")
            print(f"  标注来源: MOT txt ({gt_path})")
            frame_anns, track_ids = load_annotations_from_mot(gt_path)

        if not frame_anns and not args.raw:
            print(f"  [警告] 序列 {seq_name} 无标注数据，跳过\n")
            continue

        if frame_anns:
            print(f"  标注帧数: {len(frame_anns)}, 标注总数: {sum(len(v) for v in frame_anns.values())}")
            print(f"  唯一 track ID 数: {len(track_ids)}")
        elif args.raw:
            print("  无标注数据；--raw 模式将直接使用原始帧生成视频")

        # 生成颜色映射
        track_colors = generate_track_colors(track_ids)

        # 确定输出文件名
        prefix = args.prefix + '_' if args.prefix else ''
        source_tag = 'raw' if args.raw else ('json' if args.json else 'gt')
        output_name = f"{prefix}{seq_name}_{source_tag}.mp4"
        output_path = os.path.join(args.output_dir, output_name)

        # 生成视频
        success = generate_video(
            seq_name=seq_name,
            data_dir=args.data_dir,
            subset=args.subset,
            output_path=output_path,
            frame_anns=frame_anns,
            track_colors=track_colors,
            fps=args.fps,
            codec=args.codec,
            start_frame=args.start,
            end_frame=args.end,
            show_track_id=(not args.no_track_id) and (not args.raw),
            show_frame_counter=(not args.no_frame_counter) and (not args.raw),
            draw_boxes=not args.raw,
            box_thickness=1 if args.thin else 2,
        )

        if success:
            print()
        else:
            print(f"  [错误] 序列 {seq_name} 生成失败\n")

    print("全部完成!")


if __name__ == '__main__':
    main()
