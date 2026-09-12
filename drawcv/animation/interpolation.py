"""Interpolation functions for scalars, points, colors, transforms, and bounds."""

from __future__ import annotations
import math
from typing import Any, Callable

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.core.exceptions import ValidationError


def lerp(a: float | int, b: float | int, t: float) -> float:
    """Linear interpolation between two numbers."""
    return float((1.0 - t) * a + t * b)


def lerp_point(p1: Point, p2: Point, t: float) -> Point:
    """Linear interpolation between two 2D points."""
    return Point(
        lerp(p1.x, p2.x, t),
        lerp(p1.y, p2.y, t),
    )


def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    """Interpolate RGB using float arithmetic, quantized to [0, 255] ints, with float alpha."""
    r = int(round(max(0.0, min(255.0, lerp(c1.r, c2.r, t)))))
    g = int(round(max(0.0, min(255.0, lerp(c1.g, c2.g, t)))))
    b = int(round(max(0.0, min(255.0, lerp(c1.b, c2.b, t)))))
    a = float(max(0.0, min(1.0, lerp(c1.a, c2.a, t))))
    return Color(r, g, b, a)


def lerp_transform(t1: Transform, t2: Transform, t: float) -> Transform:
    """Linear interpolation between two 2D affine transforms."""
    tx = lerp(t1.translation_x, t2.translation_x, t)
    ty = lerp(t1.translation_y, t2.translation_y, t)
    rot = lerp(t1.rotation, t2.rotation, t)
    # Scale factors must be strictly positive
    sx = max(1e-5, lerp(t1.scale_x, t2.scale_x, t))
    sy = max(1e-5, lerp(t1.scale_y, t2.scale_y, t))

    pivot: Point | None = None
    if t1.pivot is not None and t2.pivot is not None:
        pivot = lerp_point(t1.pivot, t2.pivot, t)
    elif t1.pivot is not None:
        pivot = t1.pivot
    elif t2.pivot is not None:
        pivot = t2.pivot

    return Transform(
        translation_x=tx,
        translation_y=ty,
        rotation=rot,
        scale_x=sx,
        scale_y=sy,
        pivot=pivot,
    )


def lerp_bounds(b1: BoundingBox, b2: BoundingBox, t: float) -> BoundingBox:
    """Linear interpolation between two bounding boxes."""
    x = lerp(b1.x, b2.x, t)
    y = lerp(b1.y, b2.y, t)
    w = max(0.0, lerp(b1.width, b2.width, t))
    h = max(0.0, lerp(b1.height, b2.height, t))
    return BoundingBox(x, y, w, h)


def resolve_interpolator(val1: Any, val2: Any) -> Callable[[Any, Any, float], Any]:
    """Determine the appropriate interpolator callable for two values."""
    if isinstance(val1, Color) and isinstance(val2, Color):
        return lerp_color
    if isinstance(val1, Point) and isinstance(val2, Point):
        return lerp_point
    if isinstance(val1, Transform) and isinstance(val2, Transform):
        return lerp_transform
    if isinstance(val1, BoundingBox) and isinstance(val2, BoundingBox):
        return lerp_bounds
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return lerp
    raise ValidationError(
        f"Cannot automatically interpolate between {type(val1).__name__} and {type(val2).__name__}"
    )
