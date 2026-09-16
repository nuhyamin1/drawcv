"""Line shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
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
    supports_progressive_rendering: bool = True

    start: Point = field(default_factory=lambda: Point(0.0, 0.0))
    end: Point = field(default_factory=lambda: Point(0.0, 0.0))
    stroke: StrokeStyle = field(default_factory=StrokeStyle)

    def slice_at_progress(self, progress: float) -> Line:
        """Return a transient partial Line sliced along arc length."""
        p = max(0.0, min(1.0, float(progress)))
        cut = Point(
            (1.0 - p) * self.start.x + p * self.end.x,
            (1.0 - p) * self.start.y + p * self.end.y,
        )
        sliced = Line(
            start=self.start.copy(),
            end=cut,
            stroke=self.stroke.copy() if self.stroke else StrokeStyle(),
            name=self.name,
            visible=self.visible,
            locked=self.locked,
            opacity=self.opacity,
            z_index=self.z_index,
            transform=self.transform.copy(),
            clip=copy.deepcopy(self.clip),
            mask=copy.deepcopy(self.mask),
            effects=[copy.deepcopy(e) for e in self.effects],
        )
        sliced.render_progress = 1.0
        return sliced

    def __post_init__(self):
        super().__post_init__()
        self._validate_line()

    def _validate(self) -> None:
        super()._validate()
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
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
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
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
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

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "start": self.start.copy(),
            "end": self.end.copy(),
            "stroke": self.stroke.copy() if self.stroke else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "start" in state:
            self.start = state["start"].copy() if isinstance(state["start"], Point) else Point.from_dict(state["start"])
        if "end" in state:
            self.end = state["end"].copy() if isinstance(state["end"], Point) else Point.from_dict(state["end"])
        if "stroke" in state:
            self.stroke = state["stroke"].copy() if isinstance(state["stroke"], StrokeStyle) else StrokeStyle.from_dict(state["stroke"])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "line",
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "stroke": self.stroke.to_dict() if self.stroke else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Line:
        """Construct a Line from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        start_pt = Point.from_dict(data["start"]) if "start" in data else Point(0.0, 0.0)
        end_pt = Point.from_dict(data["end"]) if "end" in data else Point(0.0, 0.0)
        stroke_val = StrokeStyle.from_dict(data.get("stroke")) if data.get("stroke") is not None else StrokeStyle()
        return cls(
            start=start_pt,
            end=end_pt,
            stroke=stroke_val,
            **base_kwargs,
        )

