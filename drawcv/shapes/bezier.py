"""BezierCurve shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import math

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    bezier_extrema_bounds,
    distance_point_to_segment,
    flatten_cubic_bezier,
    flatten_quadratic_bezier,
)
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class BezierCurve(Drawable):
    """Retained-mode Quadratic or Cubic Bézier curve.
    
    Attributes:
        p0: Start Point.
        p1: First control Point.
        p2: Second control Point (if cubic) or End Point (if quadratic).
        p3: Optional End Point (if cubic).
        stroke: Optional StrokeStyle for stroke rendering.
    """
    supports_progressive_rendering: bool = True

    p0: Point = field(default_factory=lambda: Point(0.0, 0.0))
    p1: Point = field(default_factory=lambda: Point(50.0, 100.0))
    p2: Point = field(default_factory=lambda: Point(100.0, 0.0))
    p3: Point | None = None
    stroke: StrokeStyle | None = None

    def slice_at_progress(self, progress: float) -> BezierCurve:
        """Return a transient partial BezierCurve sliced along arc length."""
        from drawcv.core.path_processing import slice_bezier
        p = max(0.0, min(1.0, float(progress)))

        if self.is_cubic:
            q0, q1, q2, q3 = slice_bezier(self.p0, self.p1, self.p2, self.p3, p)
        else:
            c1 = Point(
                self.p0.x + (2.0 / 3.0) * (self.p1.x - self.p0.x),
                self.p0.y + (2.0 / 3.0) * (self.p1.y - self.p0.y),
            )
            c2 = Point(
                self.p2.x + (2.0 / 3.0) * (self.p1.x - self.p2.x),
                self.p2.y + (2.0 / 3.0) * (self.p1.y - self.p2.y),
            )
            q0, q1, q2, q3 = slice_bezier(self.p0, c1, c2, self.p2, p)

        sliced = BezierCurve(
            p0=q0, p1=q1, p2=q2, p3=q3,
            stroke=self.stroke.copy() if self.stroke else None,
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
        self._validate_bezier()

    def _validate_bezier(self):
        for name, pt in (("p0", self.p0), ("p1", self.p1), ("p2", self.p2)):
            if not isinstance(pt, Point):
                raise ValidationError(f"BezierCurve '{name}' must be a Point, got {type(pt).__name__}")
        if self.p3 is not None and not isinstance(self.p3, Point):
            raise ValidationError(f"BezierCurve 'p3' must be a Point or None, got {type(self.p3).__name__}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"BezierCurve 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")

    @classmethod
    def quadratic(cls, p0: Point, p1: Point, p2: Point, stroke: StrokeStyle | None = None, **kwargs) -> BezierCurve:
        """Construct a Quadratic Bézier curve (p0 -> control p1 -> p2)."""
        return cls(p0=p0, p1=p1, p2=p2, p3=None, stroke=stroke, **kwargs)

    @classmethod
    def cubic(cls, p0: Point, p1: Point, p2: Point, p3: Point, stroke: StrokeStyle | None = None, **kwargs) -> BezierCurve:
        """Construct a Cubic Bézier curve (p0 -> control p1 -> control p2 -> p3)."""
        return cls(p0=p0, p1=p1, p2=p2, p3=p3, stroke=stroke, **kwargs)

    @property
    def is_cubic(self) -> bool:
        """Whether this curve is cubic (has 4 control points)."""
        return self.p3 is not None

    @property
    def is_quadratic(self) -> bool:
        """Whether this curve is quadratic (has 3 control points)."""
        return self.p3 is None

    def flatten(self, tolerance: float = 0.5) -> list[Point]:
        """Adaptively flatten the curve in local space."""
        if self.p3 is None:
            return flatten_quadratic_bezier(self.p0, self.p1, self.p2, tolerance)
        return flatten_cubic_bezier(self.p0, self.p1, self.p2, self.p3, tolerance)

    def flatten_world(self, tolerance: float = 0.5) -> list[Point]:
        """Adaptively flatten the curve in world space, preserving screen-space fidelity."""
        w0 = self.to_world(self.p0)
        w1 = self.to_world(self.p1)
        w2 = self.to_world(self.p2)
        if self.p3 is None:
            return flatten_quadratic_bezier(w0, w1, w2, tolerance)
        w3 = self.to_world(self.p3)
        return flatten_cubic_bezier(w0, w1, w2, w3, tolerance)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space using analytical extrema."""
        x, y, w, h = bezier_extrema_bounds(self.p0, self.p1, self.p2, self.p3)
        return BoundingBox(x, y, w, h)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space."""
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule using analytical extrema of transformed curve."""
        w0 = self.to_world(self.p0)
        w1 = self.to_world(self.p1)
        w2 = self.to_world(self.p2)
        w3 = self.to_world(self.p3) if self.p3 is not None else None
        x, y, w, h = bezier_extrema_bounds(w0, w1, w2, w3)
        geom_aabb = BoundingBox(x, y, w, h)
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates."""
        norm = name.strip().lower()
        if norm == "start":
            return self.to_world(self.p0)
        if norm == "end":
            return self.to_world(self.p3 if self.p3 is not None else self.p2)
        if norm in ("control_1", "c1"):
            return self.to_world(self.p1)
        if norm in ("control_2", "c2"):
            if self.p3 is None:
                raise ValidationError("Quadratic Bézier curve does not have 'control_2'")
            return self.to_world(self.p2)
        if norm == "midpoint":
            # Point at t=0.5
            if self.p3 is None:
                mx = 0.25 * self.p0.x + 0.5 * self.p1.x + 0.25 * self.p2.x
                my = 0.25 * self.p0.y + 0.5 * self.p1.y + 0.25 * self.p2.y
            else:
                mx = 0.125 * self.p0.x + 0.375 * self.p1.x + 0.375 * self.p2.x + 0.125 * self.p3.x
                my = 0.125 * self.p0.y + 0.375 * self.p1.y + 0.375 * self.p2.y + 0.125 * self.p3.y
            return self.to_world(Point(mx, my))

        valid = "start, end, control_1, midpoint" + (", control_2" if self.p3 is not None else "")
        raise ValidationError(f"Unknown anchor '{name}' for BezierCurve. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within stroke distance of the flattened curve."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        half_stroke = (self.stroke.width / 2.0) if self.stroke else 1.0
        tolerance = max(5.0, half_stroke)

        w_pts = self.flatten_world(tolerance=0.5)
        for i in range(len(w_pts) - 1):
            if distance_point_to_segment(world_point, w_pts[i], w_pts[i + 1]) <= tolerance:
                return True

        return False

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "p0": self.p0.copy(),
            "p1": self.p1.copy(),
            "p2": self.p2.copy(),
            "p3": self.p3.copy() if self.p3 is not None else None,
            "stroke": self.stroke.copy() if self.stroke else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "p0" in state:
            self.p0 = state["p0"].copy() if isinstance(state["p0"], Point) else Point.from_dict(state["p0"])
        if "p1" in state:
            self.p1 = state["p1"].copy() if isinstance(state["p1"], Point) else Point.from_dict(state["p1"])
        if "p2" in state:
            self.p2 = state["p2"].copy() if isinstance(state["p2"], Point) else Point.from_dict(state["p2"])
        if "p3" in state:
            p3_val = state["p3"]
            self.p3 = p3_val.copy() if isinstance(p3_val, Point) else (Point.from_dict(p3_val) if p3_val is not None else None)
        if "stroke" in state:
            st = state["stroke"]
            self.stroke = st.copy() if isinstance(st, StrokeStyle) else (StrokeStyle.from_dict(st) if st else None)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "bezier",
            "p0": self.p0.to_dict(),
            "p1": self.p1.to_dict(),
            "p2": self.p2.to_dict(),
            "p3": self.p3.to_dict() if self.p3 is not None else None,
            "stroke": self.stroke.to_dict() if self.stroke else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BezierCurve:
        """Construct a BezierCurve from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        p0 = Point.from_dict(data["p0"]) if "p0" in data else Point(0.0, 0.0)
        p1 = Point.from_dict(data["p1"]) if "p1" in data else Point(50.0, 100.0)
        p2 = Point.from_dict(data["p2"]) if "p2" in data else Point(100.0, 0.0)
        p3 = Point.from_dict(data["p3"]) if data.get("p3") is not None else None
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        return cls(
            p0=p0,
            p1=p1,
            p2=p2,
            p3=p3,
            stroke=stroke,
            **base_kwargs,
        )

