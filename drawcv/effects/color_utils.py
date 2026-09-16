"""Centralized utilities for safe straight-color operations on premultiplied float32 buffers."""

from __future__ import annotations
from typing import Callable
import numpy as np


def apply_straight_color_op(
    buffer: np.ndarray,
    roi: tuple[int, int, int, int],
    color_transform_fn: Callable[..., np.ndarray],
    include_alpha: bool = False,
) -> np.ndarray:
    """Safely apply a straight normalized-color transformation within roi = (x1, y1, x2, y2).

    The incoming buffer must be float32 BGRA where:
    - BGR channels are premultiplied in range [0, 255 * alpha]
    - Alpha channel is in range [0, 1]

    The color_transform_fn receives normalized straight BGR in range [0.0, 1.0] (and optionally
    alpha in range [0.0, 1.0] if include_alpha is True) and returns transformed normalized
    straight BGR in range [0.0, 1.0].

    Fully transparent pixels (alpha <= 1e-6) strictly remain (0, 0, 0, 0).
    The premultiplied invariant (0 <= BGR_pm <= 255 * alpha) is strictly preserved.
    """
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return buffer

    sub = buffer[y1:y2, x1:x2]
    alpha = sub[..., 3:4]

    # Safe float32 unpremultiplication: normalize to [0, 1] straight BGR
    straight_bgr = np.zeros_like(sub[..., :3])
    valid_mask = (alpha > 1e-6)[..., 0]
    if np.any(valid_mask):
        alpha_valid = alpha[valid_mask]
        straight_bgr[valid_mask] = sub[valid_mask, :3] / (alpha_valid * 255.0)
        np.clip(straight_bgr, 0.0, 1.0, out=straight_bgr)

        # Apply transformation in normalized straight color space
        if include_alpha:
            transformed = color_transform_fn(straight_bgr, alpha)
        else:
            transformed = color_transform_fn(straight_bgr)
        np.clip(transformed, 0.0, 1.0, out=transformed)

        # Re-premultiply by alpha * 255.0 into DrawCV internal scale
        sub[valid_mask, :3] = transformed[valid_mask] * (alpha_valid * 255.0)

    # Sanitize transparent pixels
    sub[~valid_mask, :] = 0.0
    buffer[y1:y2, x1:x2] = sub
    return buffer


def apply_spatial_straight_filter(
    buffer: np.ndarray,
    ctx: Any,
    filter_fn: Callable[[np.ndarray, tuple[int, int, int, int]], np.ndarray],
) -> np.ndarray:
    """Safely apply a spatial filter operating on normalized straight BGR colors.

    The incoming buffer must be float32 BGRA where:
    - BGR channels are premultiplied in range [0, 255 * alpha]
    - Alpha channel is in range [0, 1]

    The filter_fn receives:
    - src_straight_bgr: normalized straight BGR in [0.0, 1.0] covering the padded requested source ROI.
    - local_target_box: (tx1, ty1, tx2, ty2) bounding the destination output region within src_straight_bgr.
    and returns transformed straight BGR for the target region of shape (ty2 - ty1, tx2 - tx1, 3).

    Output pixel alpha is strictly preserved.
    Fully transparent pixels (alpha <= 1e-6) strictly remain (0, 0, 0, 0).
    The premultiplied invariant (0 <= BGR_pm <= 255 * alpha) is strictly preserved.
    """
    ox1, oy1, ox2, oy2 = ctx.output_roi()
    if ox2 <= ox1 or oy2 <= oy1:
        return buffer

    sx1, sy1, sx2, sy2 = ctx.clamped_source_roi()
    if sx2 <= sx1 or sy2 <= sy1:
        return buffer

    src_slice = buffer[sy1:sy2, sx1:sx2]
    tx1 = ox1 - sx1
    ty1 = oy1 - sy1
    tx2 = ox2 - sx1
    ty2 = oy2 - sy1

    if tx2 <= tx1 or ty2 <= ty1 or src_slice.shape[0] == 0 or src_slice.shape[1] == 0:
        return buffer

    src_alpha = src_slice[..., 3:4]
    valid_src = (src_alpha > 1e-6)[..., 0]
    src_straight = np.zeros_like(src_slice[..., :3])
    if np.any(valid_src):
        src_straight[valid_src] = src_slice[valid_src, :3] / (src_alpha[valid_src] * 255.0)
        np.clip(src_straight, 0.0, 1.0, out=src_straight)

    filtered_straight = filter_fn(src_straight, (tx1, ty1, tx2, ty2))
    np.clip(filtered_straight, 0.0, 1.0, out=filtered_straight)

    out_sub = buffer[oy1:oy2, ox1:ox2]
    out_alpha = out_sub[..., 3:4]
    valid_out = (out_alpha > 1e-6)[..., 0]

    if np.any(valid_out):
        out_sub[valid_out, :3] = filtered_straight[valid_out] * (out_alpha[valid_out] * 255.0)

    out_sub[~valid_out, :] = 0.0
    buffer[oy1:oy2, ox1:ox2] = out_sub
    return buffer
