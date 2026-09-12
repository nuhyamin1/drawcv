"""Circle shape implementation."""

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
class Circle(Drawable):
    """Retained-mode 2D circle.
    
    Attributes:
        center: Center Point in local space.
        radius: Radius of the circle (radius >= 0).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering.
    """
    center: Point = field(default_factory=lambda: Point(0.0, 0.0))
    radius: float = 0.0
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_circle()

    def _validate_circle(self):
        if not isinstance(self.center, Point):
            raise ValidationError(f"Circle 'center' must be a Point, got {type(self.center).__name__}")
        if not isinstance(self.radius, (int, float)) or isinstance(self.radius, bool):
            raise ValidationError(f"Circle 'radius' must be numeric, got {type(self.radius).__name__}")
        if math.isnan(self.radius) or math.isinf(self.radius) or self.radius < 0:
            raise ValidationError(f"Circle 'radius' must be a non-negative finite number, got {self.radius}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Circle 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Circle 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    @property
    def diameter(self) -> float:
        """Diameter of the circle in local space."""
        return 2.0 * float(self.radius)

    @property
    def area(self) -> float:
        """Area of the circle in local space."""
        return math.pi * float(self.radius) ** 2

    @property
    def circumference(self) -> float:
        """Circumference of the circle in local space."""
        return 2.0 * math.pi * float(self.radius)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        r = float(self.radius)
        return BoundingBox(self.center.x - r, self.center.y - r, 2.0 * r, 2.0 * r)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule using analytical ellipse extrema."""
        w_center = self.to_world(self.center)
        M = self.world_matrix
        a = float(M[0, 0])
        c = float(M[0, 1])
        b = float(M[1, 0])
        d = float(M[1, 1])
        r = float(self.radius)

        # Analytical semi-extrema for rotated/transformed circle
        x_ext = math.sqrt((a * r) ** 2 + (c * r) ** 2)
        y_ext = math.sqrt((b * r) ** 2 + (d * r) ** 2)

        geom_aabb = BoundingBox(w_center.x - x_ext, w_center.y - y_ext, 2.0 * x_ext, 2.0 * y_ext)
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Circle:
            'center', 'top', 'bottom', 'left', 'right'
        """
        norm = name.strip().lower()
        c = self.center
        r = float(self.radius)

        if norm == "center":
            local_pt = c
        elif norm == "top":
            local_pt = Point(c.x, c.y - r)
        elif norm == "bottom":
            local_pt = Point(c.x, c.y + r)
        elif norm == "left":
            local_pt = Point(c.x - r, c.y)
        elif norm == "right":
            local_pt = Point(c.x + r, c.y)
        else:
            valid = "center, top, bottom, left, right"
            raise ValidationError(f"Unknown anchor '{name}' for Circle. Valid anchors are: {valid}")

        return self.to_world(local_pt)

    def anchor_at_angle(self, degrees: float | int) -> Point:
        """Calculate a point on the circle's circumference at the given angle (in degrees),
        mapped to world coordinates through the object's transform.
        """
        if not isinstance(degrees, (int, float)):
            raise ValidationError("Angle must be numeric")
        rad = math.radians(float(degrees))
        r = float(self.radius)
        local_pt = Point(self.center.x + r * math.cos(rad), self.center.y + r * math.sin(rad))
        return self.to_world(local_pt)

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within this circle (geometry-selection)."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        local_pt = self.to_local(world_point)
        return local_pt.distance_to(self.center) <= float(self.radius)
