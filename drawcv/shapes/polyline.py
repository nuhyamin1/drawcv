"""Polyline shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    distance_point_to_segment,
    point_at_polyline_length,
    polyline_length,
)
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Polyline(Drawable):
    """Retained-mode 2D polyline (connected sequence of line segments).
    
    Attributes:
        points: Ordered list of local-space Points (len >= 2).
        closed: Whether the last point connects back to the first point.
        stroke: Optional StrokeStyle for stroke rendering.
    """
    points: list[Point] = field(default_factory=list)
    closed: bool = False
    stroke: StrokeStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_polyline()

    def _validate_polyline(self):
        if not isinstance(self.points, list):
            raise ValidationError(f"Polyline 'points' must be a list of Points, got {type(self.points).__name__}")
        if len(self.points) < 2:
            raise ValidationError(f"Polyline requires at least 2 points, got {len(self.points)}")
        for idx, pt in enumerate(self.points):
            if not isinstance(pt, Point):
                raise ValidationError(f"Polyline point at index {idx} must be a Point, got {type(pt).__name__}")
        if not isinstance(self.closed, bool):
            raise ValidationError(f"Polyline 'closed' must be a boolean, got {type(self.closed).__name__}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Polyline 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")

    @property
    def length(self) -> float:
        """Cumulative length of the polyline in local space."""
        base_len = polyline_length(self.points)
        if self.closed and len(self.points) >= 2:
            base_len += self.points[-1].distance_to(self.points[0])
        return base_len

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule."""
        w_pts = [self.to_world(p) for p in self.points]
        xs = [p.x for p in w_pts]
        ys = [p.y for p in w_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        geom_aabb = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Polyline:
            'start': First point
            'end': Last point
            'midpoint': Point at 50% cumulative length
            'point_0', 'point_1', ...: Specific point by index
        """
        norm = name.strip().lower()
        if norm == "start":
            return self.to_world(self.points[0])
        if norm == "end":
            return self.to_world(self.points[-1])
        if norm == "midpoint":
            half_len = self.length / 2.0
            if self.closed:
                # Append first point temporarily for closed polyline interpolation
                pts = self.points + [self.points[0]]
            else:
                pts = self.points
            mid_pt = point_at_polyline_length(pts, half_len)
            return self.to_world(mid_pt)

        if norm.startswith("point_") or norm.startswith("p"):
            prefix = "point_" if norm.startswith("point_") else "p"
            suffix = norm[len(prefix):]
            if suffix.isdigit():
                idx = int(suffix)
                if 0 <= idx < len(self.points):
                    return self.to_world(self.points[idx])

        valid = "start, end, midpoint, point_0 .. point_N-1"
        raise ValidationError(f"Unknown anchor '{name}' for Polyline. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within stroke distance of any polyline segment.
        
        Evaluates in world space with non-scaling stroke tolerance.
        """
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        half_stroke = (self.stroke.width / 2.0) if self.stroke else 1.0
        tolerance = max(5.0, half_stroke)

        w_pts = [self.to_world(p) for p in self.points]
        n = len(w_pts)
        for i in range(n - 1):
            if distance_point_to_segment(world_point, w_pts[i], w_pts[i + 1]) <= tolerance:
                return True

        if self.closed and n >= 2:
            if distance_point_to_segment(world_point, w_pts[-1], w_pts[0]) <= tolerance:
                return True

        return False

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "points": [p.copy() for p in self.points],
            "closed": bool(self.closed),
            "stroke": self.stroke.copy() if self.stroke else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "points" in state:
            self.points = [
                p.copy() if isinstance(p, Point) else Point.from_dict(p)
                for p in state["points"]
            ]
        if "closed" in state:
            self.closed = bool(state["closed"])
        if "stroke" in state:
            st = state["stroke"]
            self.stroke = st.copy() if isinstance(st, StrokeStyle) else StrokeStyle.from_dict(st)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "polyline",
            "points": [p.to_dict() for p in self.points],
            "closed": bool(self.closed),
            "stroke": self.stroke.to_dict() if self.stroke else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Polyline:
        """Construct a Polyline from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        pts = [Point.from_dict(p) for p in data.get("points", [])]
        closed_val = bool(data.get("closed", False))
        stroke_val = StrokeStyle.from_dict(data.get("stroke")) if data.get("stroke") is not None else StrokeStyle()
        return cls(
            points=pts,
            closed=closed_val,
            stroke=stroke_val,
            **base_kwargs,
        )

