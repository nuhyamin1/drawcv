"""Rectangle shape implementation."""

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
class Rectangle(Drawable):
    """Retained-mode 2D rectangle.
    
    Attributes:
        position: Top-left corner Point in local space.
        width: Width of the rectangle (width >= 0).
        height: Height of the rectangle (height >= 0).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering.
    """
    position: Point = field(default_factory=lambda: Point(0.0, 0.0))
    width: float = 0.0
    height: float = 0.0
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_rectangle()

    def _validate_rectangle(self):
        if not isinstance(self.position, Point):
            raise ValidationError(f"Rectangle 'position' must be a Point, got {type(self.position).__name__}")
        for name, val in (("width", self.width), ("height", self.height)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Rectangle '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val) or val < 0:
                raise ValidationError(f"Rectangle '{name}' must be a non-negative finite number, got {val}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Rectangle 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Rectangle 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    @property
    def area(self) -> float:
        """Area of the rectangle in local space."""
        return float(self.width * self.height)

    @property
    def center(self) -> Point:
        """Center Point of the rectangle in local space."""
        return Point(self.position.x + self.width / 2.0, self.position.y + self.height / 2.0)

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """Return (top_left, top_right, bottom_right, bottom_left) in local space."""
        tl = self.position
        tr = Point(self.position.x + self.width, self.position.y)
        br = Point(self.position.x + self.width, self.position.y + self.height)
        bl = Point(self.position.x, self.position.y + self.height)
        return (tl, tr, br, bl)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        return BoundingBox(self.position.x, self.position.y, self.width, self.height)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule."""
        world_corners = [self.to_world(c) for c in self.corners]
        min_x = min(c.x for c in world_corners)
        max_x = max(c.x for c in world_corners)
        min_y = min(c.y for c in world_corners)
        max_y = max(c.y for c in world_corners)
        geom_aabb = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Rectangle:
            'center', 'top', 'bottom', 'left', 'right',
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
        """
        norm = name.strip().lower().replace("-", "_").replace(" ", "_")
        tl = self.position
        w, h = self.width, self.height

        if norm == "center":
            local_pt = self.center
        elif norm == "top":
            local_pt = Point(tl.x + w / 2.0, tl.y)
        elif norm == "bottom":
            local_pt = Point(tl.x + w / 2.0, tl.y + h)
        elif norm == "left":
            local_pt = Point(tl.x, tl.y + h / 2.0)
        elif norm == "right":
            local_pt = Point(tl.x + w, tl.y + h / 2.0)
        elif norm in ("top_left", "topleft"):
            local_pt = tl
        elif norm in ("top_right", "topright"):
            local_pt = Point(tl.x + w, tl.y)
        elif norm in ("bottom_left", "bottomleft"):
            local_pt = Point(tl.x, tl.y + h)
        elif norm in ("bottom_right", "bottomright"):
            local_pt = Point(tl.x + w, tl.y + h)
        else:
            valid = "center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right"
            raise ValidationError(f"Unknown anchor '{name}' for Rectangle. Valid anchors are: {valid}")

        return self.to_world(local_pt)

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within this rectangle (geometry-selection)."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        local_pt = self.to_local(world_point)
        x, y = self.position.x, self.position.y
        return (x <= local_pt.x <= x + self.width) and (y <= local_pt.y <= y + self.height)

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "position": self.position.copy(),
            "width": float(self.width),
            "height": float(self.height),
            "stroke": self.stroke.copy() if self.stroke else None,
            "fill": self.fill.copy() if self.fill else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "position" in state:
            self.position = state["position"].copy() if isinstance(state["position"], Point) else Point.from_dict(state["position"])
        if "width" in state:
            self.width = float(state["width"])
        if "height" in state:
            self.height = float(state["height"])
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
            "type": "rectangle",
            "position": self.position.to_dict(),
            "width": float(self.width),
            "height": float(self.height),
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rectangle:
        """Construct a Rectangle from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        pos = Point.from_dict(data["position"]) if "position" in data else Point(0.0, 0.0)
        width = float(data.get("width", 0.0))
        height = float(data.get("height", 0.0))
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            position=pos,
            width=width,
            height=height,
            stroke=stroke,
            fill=fill,
            **base_kwargs,
        )

