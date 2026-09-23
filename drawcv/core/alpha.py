"""Internal conversions: straight uint8 BGRA <-> premultiplied float32 BGRA.

Premultiplied BGR channels use [0, 255], alpha uses [0, 1]. Arithmetic is
in stored color space (no implicit linear-light/sRGB conversion).
"""
import cv2
import numpy as np


def spread_channels(values: np.ndarray, count: int) -> np.ndarray:
    """Repeat values along a new last axis as a contiguous array.

    Multiplying by the result gives the same elements as broadcasting
    ``values[..., None]``, but NumPy's broadcast over a 3- or 4-wide last axis
    runs an order of magnitude slower than an elementwise operation.
    """
    if values.ndim == 2 and values.size and values.dtype in (np.float32, np.float64):
        return cv2.merge([np.ascontiguousarray(values)] * count)
    return np.repeat(values[..., None], count, axis=-1)


def tint(coverage: np.ndarray, color, dtype=np.float32) -> np.ndarray:
    """Premultiplied BGRA ``(color * coverage, coverage)``, one channel at a time.

    Each element equals ``np.array(color, dtype) * coverage[..., None]``.
    """
    result = np.empty(coverage.shape + (len(color) + 1,), dtype=np.result_type(dtype, coverage))
    for channel, value in enumerate(np.asarray(color, dtype=dtype)):
        np.multiply(coverage, value, out=result[..., channel])
    result[..., -1] = coverage
    return result


def source_over(destination: np.ndarray, source: np.ndarray) -> None:
    """In place ``destination = source + destination * (1 - source_alpha)``.

    Premultiplied BGRA of matching shape; the alpha channel follows the same
    formula. Adding the scaled destination to the source is commutative, so
    each element is bit-identical to the textbook expression.
    """
    if source.dtype != destination.dtype:
        # Keep mixed-precision intermediates exactly as the plain expression does.
        destination[...] = source + destination * (1.0 - source[..., 3:4])
        return
    inverse = 1.0 - source[..., 3]
    np.multiply(destination, spread_channels(inverse, destination.shape[-1]), out=destination)
    np.add(destination, source, out=destination)


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
    """Quantize to straight uint8 BGRA, clamping into the premultiplied cone first.

    Equivalent to clamp_premultiplied followed by division, rounding and
    saturation, without the intermediate copies. When every pixel is opaque the
    cone bound is exactly 255 and division by 1 is exact, so both are skipped.
    """
    alpha = np.clip(buffer[..., 3], 0.0, 1.0)
    alpha_255 = alpha * 255.0
    result = np.empty(buffer.shape, dtype=np.uint8)
    if alpha.size and alpha.min() == 1.0:
        straight = np.clip(buffer[..., :3], 0.0, 255.0)
    else:
        premultiplied = np.minimum(np.maximum(buffer[..., :3], 0.0), alpha_255[..., None])
        straight = np.zeros_like(premultiplied)
        np.divide(premultiplied, alpha[..., None], out=straight, where=alpha[..., None] > 0)
    np.rint(straight, out=straight)
    np.clip(straight, 0, 255, out=straight)
    result[..., :3] = straight
    np.rint(alpha_255, out=alpha_255)
    np.clip(alpha_255, 0, 255, out=alpha_255)
    result[..., 3] = alpha_255
    result[result[..., 3] == 0, :3] = 0
    return result
