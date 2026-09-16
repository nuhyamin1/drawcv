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
