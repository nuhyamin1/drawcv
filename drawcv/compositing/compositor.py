"""Alpha-aware compositing engine implementing W3C blend-mode composition."""

from __future__ import annotations
import numpy as np

from drawcv.core.enums import BlendMode
from drawcv.compositing.blend import blend_rgb


def composite_blend(
    dst_buffer: np.ndarray,
    src_buffer: np.ndarray,
    region: tuple[int, int, int, int],
    mode: BlendMode,
    is_destination_isolated: bool,
) -> None:
    """Composite src_buffer into dst_buffer over region (x1, y1, x2, y2) using mode.

    Supports both float32 premultiplied BGRA destination buffers (_IsolatedSurface)
    and uint8 BGR canvas destinations (Canvas).
    """
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return

    sub_src = src_buffer[y1:y2, x1:x2]
    src_rgb = sub_src[:, :, :3]
    src_a = sub_src[:, :, 3:4]

    # Fast-path for BlendMode.NORMAL: strictly preserves existing renderer numerical semantics
    if mode == BlendMode.NORMAL:
        if is_destination_isolated:
            sub_dst = dst_buffer[y1:y2, x1:x2]
            dst_rgb = sub_dst[:, :, :3]
            dst_a = sub_dst[:, :, 3:4]
            sub_dst[:, :, :3] = src_rgb + dst_rgb * (1.0 - src_a)
            sub_dst[:, :, 3] = (src_a + dst_a * (1.0 - src_a))[:, :, 0]
        else:
            sub_dst = dst_buffer[y1:y2, x1:x2].astype(np.float32)
            blended = sub_dst * (1.0 - src_a) + src_rgb
            dst_buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
        return

    # Non-normal blend modes:
    if is_destination_isolated:
        sub_dst = dst_buffer[y1:y2, x1:x2]
        dst_rgb = sub_dst[:, :, :3]
        dst_a = sub_dst[:, :, 3:4]

        # Safe unpremultiplication strictly following drawcv alpha convention: where alpha > 0
        cb = np.zeros_like(dst_rgb)
        np.divide(dst_rgb, 255.0 * dst_a, out=cb, where=dst_a > 0)
        np.clip(cb, 0.0, 1.0, out=cb)

        cs = np.zeros_like(src_rgb)
        np.divide(src_rgb, 255.0 * src_a, out=cs, where=src_a > 0)
        np.clip(cs, 0.0, 1.0, out=cs)

        b_255 = blend_rgb(cb, cs, mode) * 255.0

        # W3C Compositing & Blending Level 1 general formula:
        # co_pm = (1 - ab) * cs_pm + (1 - as) * cb_pm + as * ab * B(cb, cs)
        # ao = as + ab * (1 - as)
        pm_rgb_o = (1.0 - dst_a) * src_rgb + (1.0 - src_a) * dst_rgb + (src_a * dst_a) * b_255
        alpha_o = src_a + dst_a * (1.0 - src_a)

        # Clamping inside valid premultiplied cone
        alpha_o = np.clip(alpha_o, 0.0, 1.0)
        pm_rgb_o = np.clip(pm_rgb_o, 0.0, alpha_o * 255.0)

        sub_dst[:, :, :3] = pm_rgb_o
        sub_dst[:, :, 3] = alpha_o[:, :, 0]
    else:
        # Destination is ordinary BGR Canvas (opaque backdrop, alpha_b = 1.0)
        sub_dst = dst_buffer[y1:y2, x1:x2].astype(np.float32)
        cb = np.clip(sub_dst / 255.0, 0.0, 1.0)

        cs = np.zeros_like(src_rgb)
        np.divide(src_rgb, 255.0 * src_a, out=cs, where=src_a > 0)
        np.clip(cs, 0.0, 1.0, out=cs)

        b_255 = blend_rgb(cb, cs, mode) * 255.0

        # Simplified for opaque backdrop: co_pm = (1 - as) * cb + as * B(cb, cs)
        blended = sub_dst * (1.0 - src_a) + b_255 * src_a
        dst_buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
