"""Geometric primitives and vector mathematics for DrawCV."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np

from drawcv.core.exceptions import ValidationError


@dataclass(frozen=True)
class Point:
    """Immutable 2D point supporting floating-point coordinates and vector operations."""
    x: float
    y: float

    def __init__(self, x: float | int, y: float | int):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValidationError(f"Point coordinates must be numeric, got ({type(x).__name__}, {type(y).__name__})")
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            raise ValidationError(f"Point coordinates must be finite numbers, got ({x}, {y})")
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))

    def translate(self, dx: float | int, dy: float | int) -> Point:
        """Return a new Point translated by (dx, dy)."""
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ValidationError("Translation deltas must be numeric")
        return Point(self.x + dx, self.y + dy)

    def distance_to(self, other: Point) -> float:
        """Compute Euclidean distance to another Point."""
        if not isinstance(other, Point):
            raise ValidationError(f"Expected Point, got {type(other).__name__}")
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_tuple(self) -> tuple[float, float]:
        """Return coordinates as a (x, y) float tuple."""
        return (self.x, self.y)

    def to_numpy(self) -> np.ndarray:
        """Return coordinates as a 1D NumPy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float64)

    def copy(self) -> Point:
        """Return a copy of the point (for API convenience, points are immutable)."""
        return Point(self.x, self.y)

    def __add__(self, other: Any) -> Point:
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: Any) -> Point:
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __mul__(self, scalar: Any) -> Point:
        if isinstance(scalar, (int, float)):
            return Point(self.x * scalar, self.y * scalar)
        return NotImplemented

    def __rmul__(self, scalar: Any) -> Point:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Any) -> Point:
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValidationError("Cannot divide Point by zero")
            return Point(self.x / scalar, self.y / scalar)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Point({self.x:g}, {self.y:g})"


def distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    """Calculate the shortest Euclidean distance from a Point to a line segment [start, end]."""
    if not isinstance(point, Point) or not isinstance(start, Point) or not isinstance(end, Point):
        raise ValidationError("All arguments to distance_point_to_segment must be Point instances")

    dx = end.x - start.x
    dy = end.y - start.y
    l2 = dx * dx + dy * dy

    if l2 == 0.0:
        return point.distance_to(start)

    # Project point onto segment, clamped to [0, 1]
    t = max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / l2))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return point.distance_to(projection)
