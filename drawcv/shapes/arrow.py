"""Arrow shape implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import math

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import ArrowHeadStyle
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    distance_point_to_segment,
    point_in_polygon,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class Arrow(Drawable):
    """Retained-mode 2D directed arrow with screen-space arrowhead geometry.
    
    Attributes:
        start: Local-space start Point of shaft.
        end: Local-space end Point (tip) of shaft.
        head_length: Screen-space length of arrowhead (pixels > 0).
        head_width: Screen-space total width of arrowhead (pixels > 0).
        head_style: Visual style of the arrowhead.
        stroke: StrokeStyle for shaft and head outline.
        fill: Optional FillStyle for filled head styles (TRIANGLE, DIAMOND, CIRCLE).
    """
    supports_progressive_rendering: bool = True

    start: Point = field(default_factory=lambda: Point(0.0, 0.0))
    end: Point = field(default_factory=lambda: Point(100.0, 0.0))
    head_length: float = 15.0
    head_width: float = 10.0
    head_style: ArrowHeadStyle = ArrowHeadStyle.TRIANGLE
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def slice_at_progress(self, progress: float) -> Arrow:
        """Return a transient partial Arrow sliced along the shaft."""
        p = max(0.0, min(1.0, float(progress)))
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        cut_x = self.start.x + p * dx
        cut_y = self.start.y + p * dy
        cut_pt = Point(cut_x, cut_y)

        if cut_pt == self.start:
            cut_pt = Point(self.start.x + 1e-4 * dx, self.start.y + 1e-4 * dy)

        fill_val = self.fill.copy() if (p >= 1.0 and self.fill) else None

        sliced = Arrow(
            start=self.start.copy(),
            end=cut_pt,
            head_length=self.head_length,
            head_width=self.head_width,
            head_style=self.head_style,
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
        self._validate_arrow()

    def _validate_arrow(self):
        if not isinstance(self.start, Point):
            raise ValidationError(f"Arrow 'start' must be a Point, got {type(self.start).__name__}")
        if not isinstance(self.end, Point):
            raise ValidationError(f"Arrow 'end' must be a Point, got {type(self.end).__name__}")
        if self.start == self.end:
            raise ValidationError("Arrow start and end points cannot be identical")

        for attr, val in (("head_length", self.head_length), ("head_width", self.head_width)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Arrow '{attr}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val) or val <= 0:
                raise ValidationError(f"Arrow '{attr}' must be a positive finite number, got {val}")

        if not isinstance(self.head_style, ArrowHeadStyle):
            raise ValidationError(f"Arrow 'head_style' must be an ArrowHeadStyle enum, got {type(self.head_style).__name__}")

        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Arrow 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Arrow 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    @property
    def length(self) -> float:
        """Euclidean length of the arrow shaft in local space."""
        return self.start.distance_to(self.end)

    def get_world_head_geometry(self) -> tuple[Point, list[Point]]:
        """Calculate the world-space arrowhead geometry along the shaft direction vector.
        
        Returns:
            (w_end, head_points): tip Point and list of polygon/marker Points in world space.
        """
        w_start = self.to_world(self.start)
        w_end = self.to_world(self.end)

        dx = w_end.x - w_start.x
        dy = w_end.y - w_start.y
        w_len = math.hypot(dx, dy)
        if w_len <= 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / w_len, dy / w_len

        nx, ny = -uy, ux  # Perpendicular normal
        L = float(self.head_length)
        half_w = float(self.head_width) / 2.0

        if self.head_style == ArrowHeadStyle.TRIANGLE:
            base_center_x = w_end.x - ux * L
            base_center_y = w_end.y - uy * L
            p_left = Point(base_center_x + nx * half_w, base_center_y + ny * half_w)
            p_right = Point(base_center_x - nx * half_w, base_center_y - ny * half_w)
            return w_end, [w_end, p_left, p_right]

        elif self.head_style == ArrowHeadStyle.OPEN:
            base_center_x = w_end.x - ux * L
            base_center_y = w_end.y - uy * L
            p_left = Point(base_center_x + nx * half_w, base_center_y + ny * half_w)
            p_right = Point(base_center_x - nx * half_w, base_center_y - ny * half_w)
            return w_end, [p_left, w_end, p_right]

        elif self.head_style == ArrowHeadStyle.DIAMOND:
            p_base = Point(w_end.x - ux * L, w_end.y - uy * L)
            mid_x = w_end.x - ux * (L / 2.0)
            mid_y = w_end.y - uy * (L / 2.0)
            p_left = Point(mid_x + nx * half_w, mid_y + ny * half_w)
            p_right = Point(mid_x - nx * half_w, mid_y - ny * half_w)
            return w_end, [w_end, p_left, p_base, p_right]

        elif self.head_style == ArrowHeadStyle.CIRCLE:
            radius = L / 2.0
            c_x = w_end.x - ux * radius
            c_y = w_end.y - uy * radius
            return w_end, [Point(c_x, c_y), Point(radius, radius)]

        return w_end, [w_end]

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space for the shaft."""
        min_x = min(self.start.x, self.end.x)
        max_x = max(self.start.x, self.end.x)
        min_y = min(self.start.y, self.end.y)
        max_y = max(self.start.y, self.end.y)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space."""
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        extra = max(half_stroke, self.head_width / 2.0, self.head_length)
        return self.get_geometry_bounds().expand(extra)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB including screen-space arrowhead points."""
        w_start = self.to_world(self.start)
        w_end, head_pts = self.get_world_head_geometry()

        all_pts = [w_start, w_end]
        if self.head_style == ArrowHeadStyle.CIRCLE:
            c = head_pts[0]
            r = head_pts[1].x
            all_pts.extend([Point(c.x - r, c.y - r), Point(c.x + r, c.y + r)])
        else:
            all_pts.extend(head_pts)

        xs = [p.x for p in all_pts]
        ys = [p.y for p in all_pts]
        geom_aabb = BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        half_stroke = (self.stroke.width / 2.0) if self.stroke else 0.0
        return geom_aabb.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates."""
        norm = name.strip().lower()
        if norm == "start":
            return self.to_world(self.start)
        if norm in ("end", "tip"):
            return self.to_world(self.end)
        if norm == "midpoint":
            mid = Point((self.start.x + self.end.x) / 2.0, (self.start.y + self.end.y) / 2.0)
            return self.to_world(mid)

        valid = "start, end, tip, midpoint"
        raise ValidationError(f"Unknown anchor '{name}' for Arrow. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies on the shaft or within the arrowhead."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        half_stroke = (self.stroke.width / 2.0) if self.stroke else 1.0
        tolerance = max(5.0, half_stroke)

        w_start = self.to_world(self.start)
        w_end, head_pts = self.get_world_head_geometry()

        # Shaft proximity
        if distance_point_to_segment(world_point, w_start, w_end) <= tolerance:
            return True

        # Head containment / proximity
        if self.head_style in (ArrowHeadStyle.TRIANGLE, ArrowHeadStyle.DIAMOND):
            if point_in_polygon(world_point, head_pts, include_boundary=True):
                return True

        elif self.head_style == ArrowHeadStyle.CIRCLE:
            c = head_pts[0]
            r = head_pts[1].x
            if world_point.distance_to(c) <= r + tolerance:
                return True

        elif self.head_style == ArrowHeadStyle.OPEN:
            p_left, tip, p_right = head_pts[0], head_pts[1], head_pts[2]
            if distance_point_to_segment(world_point, tip, p_left) <= tolerance:
                return True
            if distance_point_to_segment(world_point, tip, p_right) <= tolerance:
                return True

        return False

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "start": self.start.copy(),
            "end": self.end.copy(),
            "head_length": float(self.head_length),
            "head_width": float(self.head_width),
            "head_style": self.head_style.value if hasattr(self.head_style, "value") else str(self.head_style),
            "stroke": self.stroke.copy() if self.stroke else None,
            "fill": self.fill.copy() if self.fill else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "start" in state:
            self.start = state["start"].copy() if isinstance(state["start"], Point) else Point.from_dict(state["start"])
        if "end" in state:
            self.end = state["end"].copy() if isinstance(state["end"], Point) else Point.from_dict(state["end"])
        if "head_length" in state:
            self.head_length = float(state["head_length"])
        if "head_width" in state:
            self.head_width = float(state["head_width"])
        if "head_style" in state:
            hs = state["head_style"]
            self.head_style = ArrowHeadStyle(hs) if isinstance(hs, str) else hs
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
            "type": "arrow",
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "head_length": float(self.head_length),
            "head_width": float(self.head_width),
            "head_style": self.head_style.value if hasattr(self.head_style, "value") else str(self.head_style),
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Arrow:
        """Construct an Arrow from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        start_pt = Point.from_dict(data["start"]) if "start" in data else Point(0.0, 0.0)
        end_pt = Point.from_dict(data["end"]) if "end" in data else Point(100.0, 0.0)
        head_len = float(data.get("head_length", 15.0))
        head_wid = float(data.get("head_width", 10.0))
        head_st = ArrowHeadStyle(data["head_style"]) if "head_style" in data else ArrowHeadStyle.TRIANGLE
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            start=start_pt,
            end=end_pt,
            head_length=head_len,
            head_width=head_wid,
            head_style=head_st,
            stroke=stroke,
            fill=fill,
            **base_kwargs,
        )

