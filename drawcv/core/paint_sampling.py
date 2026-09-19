"""Renderer-side sampling of retained paints at integer pixel centers."""

from __future__ import annotations
import cv2
import numpy as np

from drawcv.core.alpha import clamp_premultiplied, premultiply
from drawcv.core.enums import ImageInterpolation
from drawcv.core.geometry import Point
from drawcv.core.exceptions import ValidationError
from drawcv.styles.paint import (
    ConicGradient,
    ImagePaint,
    LinearGradient,
    Paint,
    RadialGradient,
)


def apply_spread(t: np.ndarray, mode: str = "pad") -> np.ndarray:
    """Map real parameter t according to gradient spread mode."""
    if mode == "repeat":
        return t % 1.0
    elif mode == "reflect":
        m = t % 2.0
        return np.where(m > 1.0, 2.0 - m, m)
    return np.clip(t, 0.0, 1.0)


def _sample_stops(stops, t: np.ndarray) -> np.ndarray:
    """Sample ordered gradient color stops at parameter t in [0, 1]."""
    positions = np.array([s.position for s in stops], dtype=float)
    colors = np.array([[s.color.b, s.color.g, s.color.r, s.color.a] for s in stops], dtype=float)
    right = np.searchsorted(positions, t, side="right")
    lo = np.clip(right - 1, 0, len(positions) - 1)
    hi = np.clip(right, 0, len(positions) - 1)
    span = positions[hi] - positions[lo]
    mix = np.zeros_like(t)
    np.divide(t - positions[lo], span, out=mix, where=span > 0)
    mix = mix.clip(0.0, 1.0)[..., None]
    samples = colors[lo] * (1.0 - mix) + colors[hi] * mix
    samples[..., :3] *= samples[..., 3:4]
    return samples.astype(np.float32)


def _evaluate_coordinates(
    paint: Paint,
    matrix: np.ndarray,
    width: int,
    height: int,
    origin: tuple[int, int],
    *,
    pixel_center: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Evaluate grid coordinates through entity world matrix and paint transform."""
    y, x = np.indices((height, width), dtype=float)
    if pixel_center:
        x += 0.5
        y += 0.5
    x += origin[0]
    y += origin[1]

    # Step 1: If object space, apply inverse entity matrix
    if paint.space == "object":
        try:
            inv_entity = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None, None
        x, y = (
            inv_entity[0, 0] * x + inv_entity[0, 1] * y + inv_entity[0, 2],
            inv_entity[1, 0] * x + inv_entity[1, 1] * y + inv_entity[1, 2],
        )

    # Step 2: Apply inverse paint-local transform
    if paint.transform is not None and not paint.transform.is_identity():
        paint_mat = paint.transform.get_matrix(default_pivot=Point(0.0, 0.0))
        try:
            inv_paint = np.linalg.inv(paint_mat)
        except np.linalg.LinAlgError:
            return None, None
        x, y = (
            inv_paint[0, 0] * x + inv_paint[0, 1] * y + inv_paint[0, 2],
            inv_paint[1, 0] * x + inv_paint[1, 1] * y + inv_paint[1, 2],
        )

    return x, y


def _get_cv_interpolation(interp: ImageInterpolation) -> int:
    mapping = {
        ImageInterpolation.NEAREST: cv2.INTER_NEAREST,
        ImageInterpolation.LINEAR: cv2.INTER_LINEAR,
        ImageInterpolation.CUBIC: cv2.INTER_CUBIC,
        ImageInterpolation.LANCZOS: cv2.INTER_LANCZOS4,
    }
    return mapping.get(interp, cv2.INTER_LINEAR)


def _prepare_source_image(src_img: np.ndarray) -> np.ndarray:
    """Ensure image is 4-channel BGRA float32 premultiplied."""
    if src_img.ndim == 2 or src_img.shape[2] == 1:
        bgr = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        source = np.dstack([bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)])
    elif src_img.shape[2] == 3:
        source = np.dstack([src_img, np.full(src_img.shape[:2], 255, dtype=np.uint8)])
    else:
        source = src_img
    return premultiply(source)


def sample_linear(paint: LinearGradient, matrix: np.ndarray, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    x, y = _evaluate_coordinates(paint, matrix, width, height, origin)
    if x is None:
        return np.zeros((height, width, 4), dtype=np.float32)

    dx = paint.end.x - paint.start.x
    dy = paint.end.y - paint.start.y
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        t = np.zeros_like(x)
    else:
        t = ((x - paint.start.x) * dx + (y - paint.start.y) * dy) / len_sq

    t = apply_spread(t, paint.spread)
    samples = _sample_stops(paint.stops, t)
    opacity = getattr(paint, "opacity", 1.0)
    if opacity < 1.0:
        samples *= float(opacity)
    return samples.astype(np.float32)


def sample_radial(paint: RadialGradient, matrix: np.ndarray, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    x, y = _evaluate_coordinates(paint, matrix, width, height, origin)
    if x is None:
        return np.zeros((height, width, 4), dtype=np.float32)

    if paint.radius <= 0.0:
        return np.zeros((height, width, 4), dtype=np.float32)

    t = np.hypot(x - paint.center.x, y - paint.center.y) / paint.radius
    t = apply_spread(t, paint.spread)
    samples = _sample_stops(paint.stops, t)
    opacity = getattr(paint, "opacity", 1.0)
    if opacity < 1.0:
        samples *= float(opacity)
    return samples.astype(np.float32)


def sample_conic(paint: ConicGradient, matrix: np.ndarray, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    x, y = _evaluate_coordinates(paint, matrix, width, height, origin)
    if x is None:
        return np.zeros((height, width, 4), dtype=np.float32)

    dx = x - paint.center.x
    dy = y - paint.center.y
    theta = np.arctan2(dy, dx)
    deg = (np.degrees(theta) - paint.start_angle) % 360.0
    t = deg / 360.0
    at_center = (dx == 0.0) & (dy == 0.0)
    t = np.where(at_center, 0.0, t)

    samples = _sample_stops(paint.stops, t)
    opacity = getattr(paint, "opacity", 1.0)
    if opacity < 1.0:
        samples *= float(opacity)
    return samples.astype(np.float32)


def sample_image(paint: ImagePaint, matrix: np.ndarray, width: int, height: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    x, y = _evaluate_coordinates(paint, matrix, width, height, origin, pixel_center=True)
    if x is None:
        return np.zeros((height, width, 4), dtype=np.float32)

    # 4. origin/scale image mapping
    u = (x - paint.origin.x) / paint.scale[0]
    v = (y - paint.origin.y) / paint.scale[1]

    pm_source = _prepare_source_image(paint.image)
    h, w = pm_source.shape[:2]

    cv_interp = _get_cv_interpolation(paint.interpolation)

    if paint.repeat == "repeat":
        u_w = u % float(w)
        v_w = v % float(h)
        map_x = (u_w - 0.5).astype(np.float32)
        map_y = (v_w - 0.5).astype(np.float32)
        sampled = cv2.remap(
            pm_source,
            map_x,
            map_y,
            interpolation=cv_interp,
            borderMode=cv2.BORDER_WRAP,
        )
    elif paint.repeat == "reflect":
        mu = u % (2.0 * float(w))
        mv = v % (2.0 * float(h))
        u_w = np.where(mu > float(w), 2.0 * float(w) - mu, mu)
        v_w = np.where(mv > float(h), 2.0 * float(h) - mv, mv)
        map_x = (u_w - 0.5).astype(np.float32)
        map_y = (v_w - 0.5).astype(np.float32)
        sampled = cv2.remap(
            pm_source,
            map_x,
            map_y,
            interpolation=cv_interp,
            borderMode=cv2.BORDER_REFLECT,
        )
    elif paint.repeat == "pad":
        u_w = np.clip(u, 0.0, float(w))
        v_w = np.clip(v, 0.0, float(h))
        map_x = (u_w - 0.5).astype(np.float32)
        map_y = (v_w - 0.5).astype(np.float32)
        sampled = cv2.remap(
            pm_source,
            map_x,
            map_y,
            interpolation=cv_interp,
            borderMode=cv2.BORDER_REPLICATE,
        )
    elif paint.repeat == "none":
        map_x = (u - 0.5).astype(np.float32)
        map_y = (v - 0.5).astype(np.float32)
        sampled = cv2.remap(
            pm_source,
            map_x,
            map_y,
            interpolation=cv_interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )
    else:
        u_w = u % float(w)
        v_w = v % float(h)
        map_x = (u_w - 0.5).astype(np.float32)
        map_y = (v_w - 0.5).astype(np.float32)
        sampled = cv2.remap(
            pm_source,
            map_x,
            map_y,
            interpolation=cv_interp,
            borderMode=cv2.BORDER_WRAP,
        )

    if paint.opacity < 1.0:
        sampled = sampled * float(paint.opacity)

    return clamp_premultiplied(sampled)


def sample_paint(paint: Paint, matrix: np.ndarray, width: int, height: int, *, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Sample any retained spatial paint at integer pixel centers."""
    from drawcv.styles.pattern import VectorPattern, sample_vector_pattern
    if isinstance(paint, VectorPattern):
        return sample_vector_pattern(paint, matrix, width, height, origin)
    if isinstance(paint, LinearGradient):
        return sample_linear(paint, matrix, width, height, origin)
    elif isinstance(paint, RadialGradient):
        return sample_radial(paint, matrix, width, height, origin)
    elif isinstance(paint, ConicGradient):
        return sample_conic(paint, matrix, width, height, origin)
    elif isinstance(paint, ImagePaint):
        return sample_image(paint, matrix, width, height, origin)
    else:
        raise ValidationError(f"Cannot sample unsupported paint type: {type(paint).__name__}")


# Backward-compatible alias
sample_gradient = sample_paint
