"""Arc shape implementation."""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
import math
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import ArcClosure
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    arc_transformed_extrema_bounds,
    distance_point_to_segment,
    flatten_arc,
    is_angle_in_sweep,
    point_in_polygon,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Arc(Drawable):
    """Retained-mode 2D elliptical arc with explicit sweep angle and closure semantics.
    
    Attributes:
        center: Center Point in local space.
        radius_x: Horizontal semi-axis radius (radius_x > 0).
        radius_y: Vertical semi-axis radius (radius_y > 0).
        start_angle: Starting angle in degrees (clockwise from positive x-axis).
        sweep_angle: Signed sweep angle in degrees (clockwise positive, [-360, 360]).
        closure: Arc closure type (ArcClosure.OPEN, CHORD, or PIE).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering (applicable to CHORD and PIE).
    """
    supports_progressive_rendering: bool = True

    center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    radius_x: float = 0.0
    radius_y: float = 0.0
    start_angle: float = 0.0
    sweep_angle: float = 0.0
    closure: ArcClosure = ArcClosure.OPEN
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def slice_at_progress(self, progress: float) -> Arc:
        """Return a transient partial Arc sliced by sweep angle."""
        p = max(0.0, min(1.0, float(progress)))
        partial_sweep = self.sweep_angle * p
        closure_val = self.closure if p >= 1.0 else ArcClosure.OPEN
        fill_val = self.fill.copy() if (p >= 1.0 and self.fill) else None

        sliced = Arc(
            center=self.center.copy(),
            radius_x=self.radius_x,
            radius_y=self.radius_y,
            start_angle=self.start_angle,
            sweep_angle=partial_sweep,
            closure=closure_val,
            stroke=self.stroke.copy() if self.stroke else None,
            fill=fill_val,
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
        self._validate_arc()

    def _validate_arc(self):
        if not isinstance(self.center, Point):
            raise ValidationError(f"Arc 'center' must be a Point, got {type(self.center).__name__}")
        for attr, val in (("radius_x", self.radius_x), ("radius_y", self.radius_y)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Arc '{attr}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValidationError(f"Arc '{attr}' must be a positive finite number, got {val}")

        for attr, val in (("start_angle", self.start_angle), ("sweep_angle", self.sweep_angle)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Arc '{attr}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"Arc '{attr}' must be finite, got {val}")

        if abs(self.sweep_angle) > 360.0:
            raise ValidationError(f"Arc 'sweep_angle' magnitude must be <= 360 degrees, got {self.sweep_angle}")

        if not isinstance(self.closure, ArcClosure):
            raise ValidationError(f"Arc 'closure' must be an ArcClosure enum value, got {type(self.closure).__name__}")

        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Arc 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Arc 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    def get_contour_points(self, tolerance: float = 0.5) -> list[Point]:
        """Generate sampled points forming the arc curve and optional closure geometry in local space."""
        arc_pts = flatten_arc(
            self.center, float(self.radius_x), float(self.radius_y),
            float(self.start_angle), float(self.sweep_angle), tolerance
        )
        if self.closure == ArcClosure.PIE:
            return arc_pts + [self.center]
        elif self.closure == ArcClosure.CHORD:
            return arc_pts
        return arc_pts

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space using analytical parametric extrema."""
        x, y, w, h = arc_transformed_extrema_bounds(
            self.center, float(self.radius_x), float(self.radius_y),
            float(self.start_angle), float(self.sweep_angle),
            np.eye(3, dtype=np.float64), self.closure
        )
        return BoundingBox(x, y, w, h)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule using transformed parametric extrema."""
        x, y, w, h = arc_transformed_extrema_bounds(
            self.center, float(self.radius_x), float(self.radius_y),
            float(self.start_angle), float(self.sweep_angle),
            self.world_matrix, self.closure

        )
        geom_aabb = BoundingBox(x, y, w, h)
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates."""
        norm = name.strip().lower()
        rx, ry = float(self.radius_x), float(self.radius_y)
        c = self.center

        if norm == "center":
            return self.to_world(c)

        if norm == "start":
            rad = math.radians(self.start_angle)
            return self.to_world(Point(c.x + rx * math.cos(rad), c.y + ry * math.sin(rad)))

        if norm == "end":
            rad = math.radians(self.start_angle + self.sweep_angle)
            return self.to_world(Point(c.x + rx * math.cos(rad), c.y + ry * math.sin(rad)))

        if norm == "midpoint":
            rad = math.radians(self.start_angle + self.sweep_angle / 2.0)
            return self.to_world(Point(c.x + rx * math.cos(rad), c.y + ry * math.sin(rad)))

        valid = "center, start, end, midpoint"
        raise ValidationError(f"Unknown anchor '{name}' for Arc. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test point containment (for PIE/CHORD) or proximity (for OPEN) in world space."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        if self.closure == ArcClosure.PIE:
            # Analytical pie sector test in local space
            lp = self.to_local(world_point)
            rx, ry = float(self.radius_x), float(self.radius_y)
            dx = (lp.x - self.center.x) / rx
            dy = (lp.y - self.center.y) / ry
            if dx * dx + dy * dy > 1.0 + 1e-9:
                return False
            angle_rad = math.atan2(dy, dx)
            start_rad = math.radians(self.start_angle)
            sweep_rad = math.radians(self.sweep_angle)
            return is_angle_in_sweep(angle_rad, start_rad, sweep_rad)

        if self.closure == ArcClosure.CHORD:
            # Closed polygon test in local space
            lp = self.to_local(world_point)
            pts = self.get_contour_points(tolerance=0.5)
            return point_in_polygon(lp, pts, include_boundary=True)

        # ArcClosure.OPEN: world-space curve stroke proximity test
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 1.0
        tolerance = max(5.0, half_stroke)

        local_pts = self.get_contour_points(tolerance=0.5)
        world_pts = [self.to_world(p) for p in local_pts]
        for i in range(len(world_pts) - 1):
            if distance_point_to_segment(world_point, world_pts[i], world_pts[i + 1]) <= tolerance:
                return True

        return False

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "center": self.center.copy(),
            "radius_x": float(self.radius_x),
            "radius_y": float(self.radius_y),
            "start_angle": float(self.start_angle),
            "sweep_angle": float(self.sweep_angle),
            "closure": self.closure.value if hasattr(self.closure, "value") else str(self.closure),
            "stroke": self.stroke.copy() if self.stroke else None,
            "fill": self.fill.copy() if self.fill else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "center" in state:
            self.center = state["center"].copy() if isinstance(state["center"], Point) else Point.from_dict(state["center"])
        if "radius_x" in state:
            self.radius_x = float(state["radius_x"])
        if "radius_y" in state:
            self.radius_y = float(state["radius_y"])
        if "start_angle" in state:
            self.start_angle = float(state["start_angle"])
        if "sweep_angle" in state:
            self.sweep_angle = float(state["sweep_angle"])
        if "closure" in state:
            c_val = state["closure"]
            self.closure = ArcClosure(c_val) if isinstance(c_val, str) else c_val
        if "stroke" in state:
            st = state["stroke"]
            self.stroke = st.copy() if isinstance(st, StrokeStyle) else (StrokeStyle.from_dict(st) if st else None)
        if "fill" in state:
            fi = state["fill"]
            self.fill = fi.copy() if isinstance(fi, FillStyle) else (FillStyle.from_dict(fi) if fi else None)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "arc",
            "center": self.center.to_dict(),
            "radius_x": float(self.radius_x),
            "radius_y": float(self.radius_y),
            "start_angle": float(self.start_angle),
            "sweep_angle": float(self.sweep_angle),
            "closure": self.closure.value if hasattr(self.closure, "value") else str(self.closure),
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Arc:
        """Construct an Arc from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        center = Point.from_dict(data["center"]) if "center" in data else Point(0.0, 0.0)
        rx = float(data.get("radius_x", 0.0))
        ry = float(data.get("radius_y", 0.0))
        start_angle = float(data.get("start_angle", 0.0))
        sweep_angle = float(data.get("sweep_angle", 0.0))
        closure_val = ArcClosure(data["closure"]) if "closure" in data else ArcClosure.OPEN
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            center=center,
            radius_x=rx,
            radius_y=ry,
            start_angle=start_angle,
            sweep_angle=sweep_angle,
            closure=closure_val,
            stroke=stroke,
            fill=fill,
            **base_kwargs,
        )

