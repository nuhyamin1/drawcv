"""Polygon shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    point_in_polygon,
    polygon_area,
    polygon_centroid,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Polygon(Drawable):
    """Retained-mode 2D closed polygon.
    
    Attributes:
        vertices: Ordered list of local-space Point vertices (len >= 3).
        stroke: Optional StrokeStyle for outline rendering.
        fill: Optional FillStyle for interior rendering.
    """
    vertices: list[Point] = field(default_factory=list)
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def __post_init__(self):
        super().__post_init__()
        self._validate_polygon()

    def _validate(self) -> None:
        super()._validate()
        self._validate_polygon()

    def _validate_polygon(self):
        if not isinstance(self.vertices, list):
            raise ValidationError(f"Polygon 'vertices' must be a list of Points, got {type(self.vertices).__name__}")
        if len(self.vertices) < 3:
            raise ValidationError(f"Polygon requires at least 3 vertices, got {len(self.vertices)}")
        for idx, pt in enumerate(self.vertices):
            if not isinstance(pt, Point):
                raise ValidationError(f"Polygon vertex at index {idx} must be a Point, got {type(pt).__name__}")
        
        area = abs(polygon_area(self.vertices))
        if area <= 1e-6:
            raise ValidationError(f"Polygon vertices are collinear or degenerate (area = {area})")

        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Polygon 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Polygon 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    @property
    def area(self) -> float:
        """Geometric area of the polygon in local space."""
        return abs(polygon_area(self.vertices))

    @property
    def centroid(self) -> Point:
        """Centroid (center of mass) of the polygon in local space."""
        return polygon_centroid(self.vertices)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space (excluding stroke)."""
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space (including stroke width expansion)."""
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB under non-scaling stroke rule."""
        w_pts = [self.to_world(p) for p in self.vertices]
        xs = [p.x for p in w_pts]
        ys = [p.y for p in w_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        geom_aabb = BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Polygon:
            'centroid': Centroid (center of mass)
            'center': Center of geometric bounding box
            'vertex_0', 'vertex_1', ...: Specific vertex by index
        """
        norm = name.strip().lower()
        if norm == "centroid":
            return self.to_world(self.centroid)
        if norm == "center":
            gb = self.get_geometry_bounds()
            return self.to_world(Point(gb.x + gb.width / 2.0, gb.y + gb.height / 2.0))

        if norm.startswith("vertex_") or norm.startswith("v"):
            prefix = "vertex_" if norm.startswith("vertex_") else "v"
            suffix = norm[len(prefix):]
            if suffix.isdigit():
                idx = int(suffix)
                if 0 <= idx < len(self.vertices):
                    return self.to_world(self.vertices[idx])

        valid = "centroid, center, vertex_0 .. vertex_N-1"
        raise ValidationError(f"Unknown anchor '{name}' for Polygon. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within this polygon (geometry-selection)."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        local_pt = self.to_local(world_point)
        return point_in_polygon(local_pt, self.vertices, include_boundary=True)

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "vertices": [p.copy() for p in self.vertices],
            "stroke": self.stroke.copy() if self.stroke else None,
            "fill": self.fill.copy() if self.fill else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "vertices" in state:
            self.vertices = [
                p.copy() if isinstance(p, Point) else Point.from_dict(p)
                for p in state["vertices"]
            ]
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
            "type": "polygon",
            "vertices": [p.to_dict() for p in self.vertices],
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Polygon:
        """Construct a Polygon from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        vertices = [Point.from_dict(p) for p in data.get("vertices", [])]
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            vertices=vertices,
            stroke=stroke,
            fill=fill,
            **base_kwargs,
        )

