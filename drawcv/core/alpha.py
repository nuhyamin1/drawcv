"""Internal conversions: straight uint8 BGRA <-> premultiplied float32 BGRA.

Premultiplied BGR channels use [0, 255], alpha uses [0, 1]. Arithmetic is
in stored color space (no implicit linear-light/sRGB conversion).
"""
import numpy as np


def premultiply(bgra: np.ndarray) -> np.ndarray:
    result = bgra.astype(np.float32)
    result[..., 3] /= 255.0
    result[..., :3] *= result[..., 3:4]
    return result


def clamp_premultiplied(buffer: np.ndarray) -> np.ndarray:
    """Keep interpolation overshoot inside the valid premultiplied color cone."""
    result = buffer.copy()
    result[..., 3] = np.clip(result[..., 3], 0.0, 1.0)
    result[..., :3] = np.clip(result[..., :3], 0.0, result[..., 3:4] * 255.0)
    return result


def unpremultiply(buffer: np.ndarray) -> np.ndarray:
    source = clamp_premultiplied(buffer)
    alpha = source[..., 3:4]
    straight = np.zeros_like(source)
    np.divide(source[..., :3], alpha, out=straight[..., :3], where=alpha > 0)
    straight[..., 3:4] = alpha * 255.0
    result = np.rint(straight).clip(0, 255).astype(np.uint8)
    result[result[..., 3] == 0, :3] = 0
    return result
