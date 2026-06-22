#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强方法集合。

本文件只做“像素级增强”，不做裁剪、旋转、翻转等几何变换，
因此不会改变鱼框位置，YOLO 标签可以原样复用。
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_bgr(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    CLAHE 局部直方图均衡。

    作用：增强水下图像局部对比度，让鱼体和背景更容易区分。
    做法：转到 LAB 颜色空间，只增强亮度通道 L，尽量减少颜色失真。
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_gamma_correction(image_bgr: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Gamma 校正。

    gamma < 1：整体变亮，适合偏暗水下图像。
    gamma = 1：不变。
    gamma > 1：整体变暗。
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    table = np.array([
        ((i / 255.0) ** gamma) * 255.0 for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image_bgr, table)


def apply_denoise_bgr(
    image_bgr: np.ndarray,
    h: float = 5.0,
    h_color: float = 5.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    非局部均值去噪。

    作用：减少水下图像细小噪声。
    h 越大，去噪越强，但也可能让鱼体细节变糊。
    课程项目建议 h=3~7，不要太大。
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    return cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=float(h),
        hColor=float(h_color),
        templateWindowSize=int(template_window_size),
        searchWindowSize=int(search_window_size),
    )


def apply_sharpen_bgr(
    image_bgr: np.ndarray,
    amount: float = 0.8,
    blur_kernel_size: int = 0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Unsharp Mask 锐化。

    作用：增强鱼体边缘和纹理。
    amount 越大锐化越明显，过大可能产生噪点和边缘光晕。
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if amount <= 0:
        return image_bgr.copy()

    blurred = cv2.GaussianBlur(
        image_bgr,
        ksize=(int(blur_kernel_size), int(blur_kernel_size)),
        sigmaX=float(sigma),
    )
    sharpened = cv2.addWeighted(image_bgr, 1.0 + float(amount), blurred, -float(amount), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_underwater_bgr(
    image_bgr: np.ndarray,
    use_clahe: bool = True,
    use_gamma: bool = True,
    use_denoise: bool = True,
    use_sharpen: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    gamma: float = 0.8,
    denoise_h: float = 5.0,
    denoise_h_color: float = 5.0,
    sharpen_amount: float = 0.8,
) -> np.ndarray:
    """
    水下图像增强流水线。

    默认顺序：CLAHE → Gamma 校正 → 去噪 → 锐化。
    所有操作都不会改变图像尺寸和鱼框位置。
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    out = image_bgr.copy()

    if use_clahe:
        out = apply_clahe_bgr(
            out,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

    if use_gamma:
        out = apply_gamma_correction(out, gamma=gamma)

    if use_denoise:
        out = apply_denoise_bgr(
            out,
            h=denoise_h,
            h_color=denoise_h_color,
        )

    if use_sharpen:
        out = apply_sharpen_bgr(out, amount=sharpen_amount)

    return out
