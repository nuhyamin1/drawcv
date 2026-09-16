"""Ellipse shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Ellipse(Drawable):
    """Retained-mode 2D ellipse.
    
    Attributes:
        center: Center Point in local space.
        radius_x: Horizontal semi-axis radius (radius_x > 0).
        radius_y: Vertical semi-axis radius (radius_y > 0).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering.
    """
    center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    radius_x: float = 0.0
    radius_y: float = 0.0
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_ellipse()

    def _validate(self) -> None:
        super()._validate()
        self._validate_ellipse()

    def _validate_ellipse(self):
        if not isinstance(self.center, Point):
            raise ValidationError(f"Ellipse 'center' must be a Point, got {type(self.center).__name__}")
        for attr, val in (("radius_x", self.radius_x), ("radius_y", self.radius_y)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Ellipse '{attr}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValidationError(f"Ellipse '{attr}' must be a positive finite number, got {val}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Ellipse 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Ellipse 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    @property
    def area(self) -> float:
        """Area of the ellipse in local space."""
        return math.pi * float(self.radius_x) * float(self.radius_y)

    @property
    def circumference(self) -> float:
        """Ramanujan approximation of ellipse circumference in local space."""
        a = float(self.radius_x)
        b = float(self.radius_y)
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        rx = float(self.radius_x)
        ry = float(self.radius_y)
        return BoundingBox(self.center.x - rx, self.center.y - ry, 2.0 * rx, 2.0 * ry)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule using analytical ellipse extrema."""
        w_center = self.to_world(self.center)
        M = self.world_matrix
        a = float(M[0, 0])
        c = float(M[0, 1])
        b = float(M[1, 0])
        d = float(M[1, 1])
        rx = float(self.radius_x)
        ry = float(self.radius_y)

        x_ext = math.sqrt((a * rx) ** 2 + (c * ry) ** 2)
        y_ext = math.sqrt((b * rx) ** 2 + (d * ry) ** 2)

        geom_aabb = BoundingBox(w_center.x - x_ext, w_center.y - y_ext, 2.0 * x_ext, 2.0 * y_ext)
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Ellipse:
            'center', 'top', 'bottom', 'left', 'right'
        """
        norm = name.strip().lower()
        c = self.center
        rx = float(self.radius_x)
        ry = float(self.radius_y)

        if norm == "center":
            local_pt = c
        elif norm == "top":
            local_pt = Point(c.x, c.y - ry)
        elif norm == "bottom":
            local_pt = Point(c.x, c.y + ry)
        elif norm == "left":
            local_pt = Point(c.x - rx, c.y)
        elif norm == "right":
            local_pt = Point(c.x + rx, c.y)
        else:
            valid = "center, top, bottom, left, right"
            raise ValidationError(f"Unknown anchor '{name}' for Ellipse. Valid anchors are: {valid}")

        return self.to_world(local_pt)

    def anchor_at_angle(self, degrees: float | int) -> Point:
        """Calculate a point on the ellipse's circumference at the given parametric angle (in degrees),
        mapped to world coordinates through the object's transform.
        """
        if not isinstance(degrees, (int, float)):
            raise ValidationError("Angle must be numeric")
        rad = math.radians(float(degrees))
        rx = float(self.radius_x)
        ry = float(self.radius_y)
        local_pt = Point(self.center.x + rx * math.cos(rad), self.center.y + ry * math.sin(rad))
        return self.to_world(local_pt)

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within this ellipse (geometry-selection)."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        local_pt = self.to_local(world_point)
        rx = float(self.radius_x)
        ry = float(self.radius_y)
        dx = (local_pt.x - self.center.x) / rx
        dy = (local_pt.y - self.center.y) / ry
        return (dx * dx + dy * dy) <= 1.0 + 1e-9

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "center": self.center.copy(),
            "radius_x": float(self.radius_x),
            "radius_y": float(self.radius_y),
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
            "type": "ellipse",
            "center": self.center.to_dict(),
            "radius_x": float(self.radius_x),
            "radius_y": float(self.radius_y),
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ellipse:
        """Construct an Ellipse from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        center = Point.from_dict(data["center"]) if "center" in data else Point(0.0, 0.0)
        rx = float(data.get("radius_x", 0.0))
        ry = float(data.get("radius_y", 0.0))
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            center=center,
            radius_x=rx,
            radius_y=ry,
            stroke=stroke,
            fill=fill,
            **base_kwargs,
        )

