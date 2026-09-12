"""RoundedRectangle shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import flatten_arc
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class RoundedRectangle(Drawable):
    """Retained-mode 2D rectangle with smoothly rounded corners.
    
    Attributes:
        x: Top-left x-coordinate in local space.
        y: Top-left y-coordinate in local space.
        width: Width of the rectangle (width > 0).
        height: Height of the rectangle (height > 0).
        corner_radius: Radius of the 4 corner arcs (0 <= corner_radius <= min(w, h)/2).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering.
    """
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    corner_radius: float = 0.0
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_rounded_rectangle()

    def _validate_rounded_rectangle(self):
        for attr, val in (("x", self.x), ("y", self.y), ("width", self.width),
                          ("height", self.height), ("corner_radius", self.corner_radius)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"RoundedRectangle '{attr}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"RoundedRectangle '{attr}' must be finite, got {val}")

        if self.width <= 0:
            raise ValidationError(f"RoundedRectangle 'width' must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValidationError(f"RoundedRectangle 'height' must be > 0, got {self.height}")

        max_radius = min(self.width, self.height) / 2.0
        if self.corner_radius < 0 or self.corner_radius > max_radius + 1e-9:
            raise ValidationError(
                f"RoundedRectangle 'corner_radius' must be between 0 and {max_radius}, got {self.corner_radius}"
            )

        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"RoundedRectangle 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"RoundedRectangle 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    def get_contour_points(self, tolerance: float = 0.5) -> list[Point]:
        """Generate a closed sequence of local Points forming the rounded rectangle perimeter."""
        r = float(self.corner_radius)
        if r <= 0.0:
            return [
                Point(self.x, self.y),
                Point(self.x + self.width, self.y),
                Point(self.x + self.width, self.y + self.height),
                Point(self.x, self.y + self.height),
            ]

        pts: list[Point] = []
        # Top-left arc: center (x + r, y + r), from 180 to 270 deg (sweep +90)
        pts.extend(flatten_arc(Point(self.x + r, self.y + r), r, r, 180.0, 90.0, tolerance))
        # Top-right arc: center (x + w - r, y + r), from 270 to 360 deg (sweep +90)
        pts.extend(flatten_arc(Point(self.x + self.width - r, self.y + r), r, r, 270.0, 90.0, tolerance))
        # Bottom-right arc: center (x + w - r, y + h - r), from 0 to 90 deg (sweep +90)
        pts.extend(flatten_arc(Point(self.x + self.width - r, self.y + self.height - r), r, r, 0.0, 90.0, tolerance))
        # Bottom-left arc: center (x + r, y + h - r), from 90 to 180 deg (sweep +90)
        pts.extend(flatten_arc(Point(self.x + r, self.y + self.height - r), r, r, 90.0, 90.0, tolerance))
        return pts

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        return BoundingBox(self.x, self.y, self.width, self.height)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule."""
        corners = [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height),
        ]
        w_corners = [self.to_world(c) for c in corners]
        xs = [c.x for c in w_corners]
        ys = [c.y for c in w_corners]
        geom_aabb = BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates."""
        norm = name.strip().lower()
        x, y, w, h = self.x, self.y, self.width, self.height

        mapping = {
            "top_left": Point(x, y),
            "top_right": Point(x + w, y),
            "bottom_left": Point(x, y + h),
            "bottom_right": Point(x + w, y + h),
            "top": Point(x + w / 2.0, y),
            "bottom": Point(x + w / 2.0, y + h),
            "left": Point(x, y + h / 2.0),
            "right": Point(x + w, y + h / 2.0),
            "center": Point(x + w / 2.0, y + h / 2.0),
        }

        if norm in mapping:
            return self.to_world(mapping[norm])

        valid = ", ".join(mapping.keys())
        raise ValidationError(f"Unknown anchor '{name}' for RoundedRectangle. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within this rounded rectangle (geometry-selection)."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        lp = self.to_local(world_point)
        x, y, w, h, r = self.x, self.y, self.width, self.height, float(self.corner_radius)

        # Outside outer bounding rectangle
        if lp.x < x or lp.x > x + w or lp.y < y or lp.y > y + h:
            return False

        if r <= 0.0:
            return True

        # Inside horizontal or vertical central bands
        if (x + r <= lp.x <= x + w - r) or (y + r <= lp.y <= y + h - r):
            return True

        # Check the 4 corner sectors
        # Top-Left
        if lp.x < x + r and lp.y < y + r:
            return math.hypot(lp.x - (x + r), lp.y - (y + r)) <= r + 1e-9
        # Top-Right
        if lp.x > x + w - r and lp.y < y + r:
            return math.hypot(lp.x - (x + w - r), lp.y - (y + r)) <= r + 1e-9
        # Bottom-Right
        if lp.x > x + w - r and lp.y > y + h - r:
            return math.hypot(lp.x - (x + w - r), lp.y - (y + h - r)) <= r + 1e-9
        # Bottom-Left
        if lp.x < x + r and lp.y > y + h - r:
            return math.hypot(lp.x - (x + r), lp.y - (y + h - r)) <= r + 1e-9

        return False
