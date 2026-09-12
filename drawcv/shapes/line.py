"""Line shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point, distance_point_to_segment
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Line(Drawable):
    """Retained-mode 2D line segment connecting two points.
    
    Attributes:
        start: Starting Point in local space.
        end: Ending Point in local space.
        stroke: StrokeStyle for outline rendering.
    """
    start: Point = field(default_factory=lambda: Point(0.0, 0.0))
    end: Point = field(default_factory=lambda: Point(0.0, 0.0))
    stroke: StrokeStyle = field(default_factory=StrokeStyle)

    def __post_init__(self):
        super().__post_init__()
        self._validate_line()

    def _validate_line(self):
        if not isinstance(self.start, Point):
            raise ValidationError(f"Line 'start' must be a Point, got {type(self.start).__name__}")
        if not isinstance(self.end, Point):
            raise ValidationError(f"Line 'end' must be a Point, got {type(self.end).__name__}")
        if not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Line 'stroke' must be a StrokeStyle, got {type(self.stroke).__name__}")

    @property
    def length(self) -> float:
        """Euclidean distance between start and end in local space."""
        return self.start.distance_to(self.end)

    @property
    def angle(self) -> float:
        """Angle in degrees from start to end (relative to horizontal axis) in local space."""
        return math.degrees(math.atan2(self.end.y - self.start.y, self.end.x - self.start.x))

    @property
    def center(self) -> Point:
        """Midpoint of the line segment in local space."""
        return Point((self.start.x + self.end.x) / 2.0, (self.start.y + self.end.y) / 2.0)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        min_x = min(self.start.x, self.end.x)
        min_y = min(self.start.y, self.end.y)
        max_x = max(self.start.x, self.end.x)
        max_y = max(self.start.y, self.end.y)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule."""
        w_start = self.to_world(self.start)
        w_end = self.to_world(self.end)
        min_x = min(w_start.x, w_end.x)
        min_y = min(w_start.y, w_end.y)
        max_x = max(w_start.x, w_end.x)
        max_y = max(w_start.y, w_end.y)
        geom_aabb = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Line:
            'start', 'center', 'end'
        """
        normalized = name.strip().lower()
        if normalized == "start":
            return self.to_world(self.start)
        elif normalized == "center":
            return self.to_world(self.center)
        elif normalized == "end":
            return self.to_world(self.end)
        else:
            raise ValidationError(f"Unknown anchor '{name}' for Line. Valid anchors are: start, center, end")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies near the line segment within screen stroke tolerance."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        world_start = self.to_world(self.start)
        world_end = self.to_world(self.end)
        tol = max((self.stroke.width / 2.0) if self.stroke else 0.0, 2.0)
        dist = distance_point_to_segment(world_point, world_start, world_end)
        return dist <= tol
