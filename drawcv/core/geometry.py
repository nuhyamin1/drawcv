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


@dataclass(frozen=True)
class StrokePoint:
    """Immutable point with physical capture metadata (pressure, timestamp, velocity)."""
    x: float
    y: float
    pressure: float | None = None
    timestamp: float | None = None
    velocity: float | None = None

    def __init__(
        self,
        x: float | int,
        y: float | int,
        pressure: float | int | None = None,
        timestamp: float | int | None = None,
        velocity: float | int | None = None,
    ):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValidationError(f"StrokePoint coordinates must be numeric, got ({type(x).__name__}, {type(y).__name__})")
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            raise ValidationError(f"StrokePoint coordinates must be finite numbers, got ({x}, {y})")

        if pressure is not None:
            if not isinstance(pressure, (int, float)) or math.isnan(pressure) or math.isinf(pressure):
                raise ValidationError(f"StrokePoint pressure must be a numeric value, got {pressure}")
            if not (0.0 <= float(pressure) <= 1.0):
                raise ValidationError(f"StrokePoint pressure must be between 0.0 and 1.0 inclusive, got {pressure}")
            pressure = float(pressure)

        if timestamp is not None:
            if not isinstance(timestamp, (int, float)) or math.isnan(timestamp) or math.isinf(timestamp):
                raise ValidationError(f"StrokePoint timestamp must be a numeric value, got {timestamp}")
            timestamp = float(timestamp)

        if velocity is not None:
            if not isinstance(velocity, (int, float)) or math.isnan(velocity) or math.isinf(velocity):
                raise ValidationError(f"StrokePoint velocity must be a numeric value, got {velocity}")
            if float(velocity) < 0.0:
                raise ValidationError(f"StrokePoint velocity must be non-negative, got {velocity}")
            velocity = float(velocity)

        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "pressure", pressure)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "velocity", velocity)

    def translate(self, dx: float | int, dy: float | int) -> StrokePoint:
        """Return a new StrokePoint translated by (dx, dy), preserving metadata."""
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ValidationError("Translation deltas must be numeric")
        return StrokePoint(self.x + dx, self.y + dy, self.pressure, self.timestamp, self.velocity)

    def distance_to(self, other: Any) -> float:
        """Compute Euclidean distance to another point."""
        if not hasattr(other, "x") or not hasattr(other, "y"):
            raise ValidationError("Expected object with x and y attributes")
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_point(self) -> Point:
        """Convert to basic geometric Point."""
        return Point(self.x, self.y)

    def to_tuple(self) -> tuple[float, float]:
        """Return coordinates as a (x, y) float tuple."""
        return (self.x, self.y)

    def to_numpy(self) -> np.ndarray:
        """Return coordinates as a 1D NumPy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float64)

    def copy(self) -> StrokePoint:
        """Return a copy of the StrokePoint."""
        return StrokePoint(self.x, self.y, self.pressure, self.timestamp, self.velocity)

    @classmethod
    def from_point(
        cls,
        point: Any,
        pressure: float | None = None,
        timestamp: float | None = None,
        velocity: float | None = None,
    ) -> StrokePoint:
        """Create a StrokePoint from a Point, StrokePoint, or (x, y) tuple."""
        if isinstance(point, StrokePoint):
            p_val = pressure if pressure is not None else point.pressure
            t_val = timestamp if timestamp is not None else point.timestamp
            v_val = velocity if velocity is not None else point.velocity
            return cls(point.x, point.y, p_val, t_val, v_val)
        if isinstance(point, Point):
            return cls(point.x, point.y, pressure, timestamp, velocity)
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return cls(point[0], point[1], pressure, timestamp, velocity)
        raise ValidationError(f"Cannot convert {type(point).__name__} to StrokePoint")

    def __repr__(self) -> str:
        parts = [f"{self.x:g}", f"{self.y:g}"]
        if self.pressure is not None:
            parts.append(f"p={self.pressure:.2f}")
        if self.timestamp is not None:
            parts.append(f"t={self.timestamp:g}")
        if self.velocity is not None:
            parts.append(f"v={self.velocity:.1f}")
        return f"StrokePoint({', '.join(parts)})"


def distance_point_to_segment(point: Point | StrokePoint, start: Point | StrokePoint, end: Point | StrokePoint) -> float:
    """Calculate the shortest Euclidean distance from a Point to a line segment [start, end]."""
    if not hasattr(point, "x") or not hasattr(start, "x") or not hasattr(end, "x"):
        raise ValidationError("All arguments to distance_point_to_segment must have x and y attributes")

    dx = end.x - start.x
    dy = end.y - start.y
    l2 = dx * dx + dy * dy

    if l2 == 0.0:
        return point.distance_to(start)

    # Project point onto segment, clamped to [0, 1]
    t = max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / l2))
    proj_x = start.x + t * dx
    proj_y = start.y + t * dy
    return math.hypot(point.x - proj_x, point.y - proj_y)

