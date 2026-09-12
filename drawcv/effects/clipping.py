"""Clipping boundaries (rectangular and arbitrary vector paths)."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import cv2
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


@dataclass(frozen=True)
class ClipRect:
    """Rectangular clipping boundary defined in entity-local coordinates.
    
    Attributes:
        x: Left edge in entity-local space.
        y: Top edge in entity-local space.
        width: Width of the clipping box (>= 0).
        height: Height of the clipping box (>= 0).
    """
    x: float
    y: float
    width: float
    height: float

    def __init__(self, x: float | int, y: float | int, width: float | int, height: float | int):
        for name, val in (("x", x), ("y", y), ("width", width), ("height", height)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"ClipRect '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"ClipRect '{name}' must be a finite number, got {val}")

        if float(width) < 0.0 or float(height) < 0.0:
            raise ValidationError(f"ClipRect dimensions must be non-negative, got ({width}, {height})")

        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "height", float(height))

    @property
    def bounds(self) -> BoundingBox:
        """Local bounding box of the clip rectangle."""
        return BoundingBox(self.x, self.y, self.width, self.height)

    @property
    def corners(self) -> list[Point]:
        """Return 4 local corners: top-left, top-right, bottom-right, bottom-left."""
        return [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height),
        ]


@dataclass(frozen=True)
class ClipPath:
    """Arbitrary vector path clipping boundary defined in entity-local coordinates.
    
    Attributes:
        points: Ordered sequence of local vertices forming a closed clipping boundary.
    """
    points: tuple[Point, ...]

    def __init__(self, points: Sequence[Point | tuple[float, float]]):
        if not isinstance(points, (list, tuple)):
            raise ValidationError(f"ClipPath points must be a sequence, got {type(points).__name__}")
        if len(points) < 3:
            raise ValidationError(f"ClipPath requires at least 3 points, got {len(points)}")

        norm_pts: list[Point] = []
        for idx, pt in enumerate(points):
            if isinstance(pt, Point):
                norm_pts.append(pt)
            elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                norm_pts.append(Point(pt[0], pt[1]))
            else:
                raise ValidationError(f"Point at index {idx} must be a Point or (x, y) tuple, got {pt}")

        object.__setattr__(self, "points", tuple(norm_pts))

    @property
    def bounds(self) -> BoundingBox:
        """Local bounding box of the clip path."""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
