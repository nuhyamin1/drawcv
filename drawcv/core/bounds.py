"""Bounding box representation for spatial queries and bounds calculation."""

from __future__ import annotations
from dataclasses import dataclass
import math

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


@dataclass(frozen=True)
class BoundingBox:
    """Immutable 2D axis-aligned bounding box.
    
    Attributes:
        x: minimum x coordinate (left)
        y: minimum y coordinate (top)
        width: horizontal extent (width >= 0)
        height: vertical extent (height >= 0)
    """
    x: float
    y: float
    width: float
    height: float

    def __init__(self, x: float | int, y: float | int, width: float | int, height: float | int):
        for name, val in (("x", x), ("y", y), ("width", width), ("height", height)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"BoundingBox '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"BoundingBox '{name}' must be finite, got {val}")

        if width < 0:
            raise ValidationError(f"BoundingBox width must be non-negative, got {width}")
        if height < 0:
            raise ValidationError(f"BoundingBox height must be non-negative, got {height}")

        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "height", float(height))

    def to_dict(self) -> dict[str, float]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | tuple | list) -> BoundingBox:
        """Construct a BoundingBox from a dict or (x, y, width, height) sequence."""
        if isinstance(data, (list, tuple)) and len(data) >= 4:
            return cls(data[0], data[1], data[2], data[3])
        if isinstance(data, dict):
            if "x" not in data or "y" not in data or "width" not in data or "height" not in data:
                raise ValidationError("BoundingBox dict requires 'x', 'y', 'width', and 'height'")
            return cls(data["x"], data["y"], data["width"], data["height"])
        raise ValidationError(f"Cannot construct BoundingBox from {type(data).__name__}")


    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def top_left(self) -> Point:
        return Point(self.x, self.y)

    @property
    def top_right(self) -> Point:
        return Point(self.x + self.width, self.y)

    @property
    def bottom_left(self) -> Point:
        return Point(self.x, self.y + self.height)

    @property
    def bottom_right(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    @property
    def corners(self) -> list[Point]:
        """Return the 4 corner points [top_left, top_right, bottom_right, bottom_left]."""
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def contains(self, point: Point) -> bool:
        """Check if a Point lies within this bounding box."""
        if not isinstance(point, Point):
            raise ValidationError(f"Expected Point, got {type(point).__name__}")
        return (self.left <= point.x <= self.right) and (self.top <= point.y <= self.bottom)

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box overlaps with another bounding box."""
        if not isinstance(other, BoundingBox):
            raise ValidationError(f"Expected BoundingBox, got {type(other).__name__}")
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )

    def union(self, other: BoundingBox) -> BoundingBox:
        """Compute the smallest bounding box enclosing both this and another bounding box."""
        if not isinstance(other, BoundingBox):
            raise ValidationError(f"Expected BoundingBox, got {type(other).__name__}")
        new_left = min(self.left, other.left)
        new_top = min(self.top, other.top)
        new_right = max(self.right, other.right)
        new_bottom = max(self.bottom, other.bottom)
        return BoundingBox(new_left, new_top, new_right - new_left, new_bottom - new_top)

    def expand(self, padding: float | int) -> BoundingBox:
        """Expand or contract the bounding box uniformly on all sides.
        
        Contracts if padding < 0, clamped so width and height do not become negative.
        """
        if not isinstance(padding, (int, float)) or isinstance(padding, bool):
            raise ValidationError("Padding must be a numeric value")
        p = float(padding)
        new_width = max(0.0, self.width + 2 * p)
        new_height = max(0.0, self.height + 2 * p)
        new_x = self.x - p if new_width > 0 else self.center.x
        new_y = self.y - p if new_height > 0 else self.center.y
        return BoundingBox(new_x, new_y, new_width, new_height)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x:g}, y={self.y:g}, w={self.width:g}, h={self.height:g})"
