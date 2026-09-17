"""Retained-mode Path and Subpath implementation with semantic vector commands."""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import math
from typing import Any, Callable, Sequence


from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import FillRule, PathBooleanOp
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    bezier_extrema_bounds,
    distance_point_to_segment,
    flatten_cubic_bezier,
    flatten_quadratic_bezier,
    point_in_polygon,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


# -----------------------------------------------------------------------------
# Semantic Path Commands
# -----------------------------------------------------------------------------

class PathCommand:
    """Abstract base class for semantic vector path commands."""
    pass


@dataclass(frozen=True)
class MoveTo(PathCommand):
    point: Point

    def __post_init__(self):
        if not isinstance(self.point, Point):
            raise ValidationError(f"MoveTo 'point' must be a Point, got {type(self.point).__name__}")


@dataclass(frozen=True)
class LineTo(PathCommand):
    point: Point

    def __post_init__(self):
        if not isinstance(self.point, Point):
            raise ValidationError(f"LineTo 'point' must be a Point, got {type(self.point).__name__}")


@dataclass(frozen=True)
class QuadraticTo(PathCommand):
    control: Point
    end: Point

    def __post_init__(self):
        if not isinstance(self.control, Point) or not isinstance(self.end, Point):
            raise ValidationError("QuadraticTo control and end must be Point instances")


@dataclass(frozen=True)
class CubicTo(PathCommand):
    control1: Point
    control2: Point
    end: Point

    def __post_init__(self):
        if not isinstance(self.control1, Point) or not isinstance(self.control2, Point) or not isinstance(self.end, Point):
            raise ValidationError("CubicTo control1, control2, and end must be Point instances")


@dataclass(frozen=True)
class Close(PathCommand):
    pass


# -----------------------------------------------------------------------------
# Subpath
# -----------------------------------------------------------------------------

@dataclass
class Subpath:
    """A single continuous contour within a Path."""
    commands: list[PathCommand] = field(default_factory=list)
    closed: bool = False

    @property
    def start_point(self) -> Point | None:
        if self.commands and isinstance(self.commands[0], MoveTo):
            return self.commands[0].point
        return None

    @property
    def current_point(self) -> Point | None:
        if not self.commands:
            return None
        last = self.commands[-1]
        if isinstance(last, (MoveTo, LineTo)):
            return last.point
        if isinstance(last, (QuadraticTo, CubicTo)):
            return last.end
        if isinstance(last, Close) and self.start_point is not None:
            return self.start_point
        return None


# -----------------------------------------------------------------------------
# Subpath Serialization Codecs
# -----------------------------------------------------------------------------

def serialize_subpaths(subpaths: list[Subpath]) -> list[dict[str, Any]]:
    """Serialize a list of Subpaths to canonical JSON dictionary representation."""
    subpaths_data = []
    for sp in subpaths:
        cmds_data = []
        for cmd in sp.commands:
            if isinstance(cmd, MoveTo):
                cmds_data.append({"type": "move_to", "point": cmd.point.to_dict()})
            elif isinstance(cmd, LineTo):
                cmds_data.append({"type": "line_to", "point": cmd.point.to_dict()})
            elif isinstance(cmd, QuadraticTo):
                cmds_data.append({"type": "quadratic_to", "control": cmd.control.to_dict(), "end": cmd.end.to_dict()})
            elif isinstance(cmd, CubicTo):
                cmds_data.append({
                    "type": "cubic_to",
                    "control1": cmd.control1.to_dict(),
                    "control2": cmd.control2.to_dict(),
                    "end": cmd.end.to_dict(),
                })
            elif isinstance(cmd, Close):
                cmds_data.append({"type": "close"})
        subpaths_data.append({"commands": cmds_data, "closed": bool(sp.closed)})
    return subpaths_data


def deserialize_subpaths(subpaths_data: list[dict[str, Any]]) -> list[Subpath]:
    """Construct a list of Subpaths from canonical serialized dictionaries."""
    if not isinstance(subpaths_data, list):
        raise ValidationError("Path 'subpaths' data must be a list")
    subpaths = []
    for sp_dict in subpaths_data:
        cmds = []
        for c_dict in sp_dict.get("commands", []):
            c_type = c_dict.get("type", "")
            if c_type in ("move_to", "moveTo"):
                cmds.append(MoveTo(Point.from_dict(c_dict["point"])))
            elif c_type in ("line_to", "lineTo"):
                cmds.append(LineTo(Point.from_dict(c_dict["point"])))
            elif c_type in ("quadratic_to", "quadraticTo"):
                cmds.append(QuadraticTo(Point.from_dict(c_dict["control"]), Point.from_dict(c_dict["end"])))
            elif c_type in ("cubic_to", "cubicTo"):
                cmds.append(CubicTo(
                    Point.from_dict(c_dict["control1"]),
                    Point.from_dict(c_dict["control2"]),
                    Point.from_dict(c_dict["end"]),
                ))
            elif c_type in ("close", "closePath"):
                cmds.append(Close())
            else:
                raise ValidationError(f"Unknown path command type '{c_type}'")
        subpaths.append(Subpath(commands=cmds, closed=bool(sp_dict.get("closed", False))))
    return subpaths


# -----------------------------------------------------------------------------
# Path Shape
# -----------------------------------------------------------------------------

@dataclass(eq=False)
class Path(Drawable):
    """Retained-mode 2D compound vector Path.
    
    Preserves exact semantic commands inside Subpaths and adaptively derives
    screen-space flattening dynamically at render or hit-test time.
    
    Attributes:
        subpaths: List of Subpath contours.
        fill_rule: FillRule.EVEN_ODD or FillRule.NON_ZERO.
        stroke: Optional StrokeStyle.
        fill: Optional FillStyle.
    """
    supports_progressive_rendering: bool = True

    subpaths: list[Subpath] = field(default_factory=list)
    fill_rule: FillRule = FillRule.EVEN_ODD
    stroke: StrokeStyle | None = None
    fill: FillStyle | None = None

    def slice_at_progress(self, progress: float) -> Path:
        """Return a transient partial Path sliced along arc length."""
        from drawcv.core.path_processing import slice_path
        p = max(0.0, min(1.0, float(progress)))
        sliced_path = slice_path(self, p)
        sliced_path.stroke = self.stroke.copy() if self.stroke else None
        sliced_path.fill = self.fill.copy() if (p >= 1.0 and self.fill) else None
        sliced_path.name = self.name
        sliced_path.visible = self.visible
        sliced_path.locked = self.locked
        sliced_path.opacity = self.opacity
        sliced_path.z_index = self.z_index
        sliced_path.transform = self.transform.copy()
        sliced_path.clip = copy.deepcopy(self.clip)
        sliced_path.mask = copy.deepcopy(self.mask)
        sliced_path.effects = [copy.deepcopy(e) for e in self.effects]
        sliced_path.render_progress = 1.0
        return sliced_path

    def __post_init__(self):
        super().__post_init__()
        self._validate_path()

    def _validate(self) -> None:
        super()._validate()
        self._validate_path()

    def _validate_path(self):
        if not isinstance(self.subpaths, list):
            raise ValidationError("Path 'subpaths' must be a list")
        if not isinstance(self.fill_rule, FillRule):
            raise ValidationError(f"Path 'fill_rule' must be a FillRule enum, got {type(self.fill_rule).__name__}")
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"Path 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")
        if self.fill is not None and not isinstance(self.fill, FillStyle):
            raise ValidationError(f"Path 'fill' must be a FillStyle or None, got {type(self.fill).__name__}")

    # -------------------------------------------------------------------------
    # Fluent Builder API
    # -------------------------------------------------------------------------

    def _ensure_active_subpath(self, fallback_start: Point | None = None) -> Subpath:
        if not self.subpaths or self.subpaths[-1].closed:
            sp = Subpath()
            if fallback_start is not None:
                sp.commands.append(MoveTo(fallback_start))
            self.subpaths.append(sp)
            return sp
        return self.subpaths[-1]

    def move_to(self, x_or_pt: float | Point, y: float | None = None) -> Path:
        """Start a new subpath at the given point."""
        pt = x_or_pt if isinstance(x_or_pt, Point) else Point(float(x_or_pt), float(y))
        sp = Subpath(commands=[MoveTo(pt)])
        self.subpaths.append(sp)
        return self

    def line_to(self, x_or_pt: float | Point, y: float | None = None) -> Path:
        """Add a straight line segment from the current point to the specified point."""
        pt = x_or_pt if isinstance(x_or_pt, Point) else Point(float(x_or_pt), float(y))
        sp = self._ensure_active_subpath(fallback_start=Point(0.0, 0.0))
        sp.commands.append(LineTo(pt))
        return self

    def quadratic_to(
        self,
        ctrl_x_or_pt: float | Point,
        ctrl_y_or_end: float | Point,
        end_x: float | None = None,
        end_y: float | None = None
    ) -> Path:
        """Add a quadratic Bézier curve to the active subpath."""
        if isinstance(ctrl_x_or_pt, Point) and isinstance(ctrl_y_or_end, Point):
            ctrl = ctrl_x_or_pt
            end = ctrl_y_or_end
        else:
            ctrl = Point(float(ctrl_x_or_pt), float(ctrl_y_or_end))
            end = Point(float(end_x), float(end_y))

        sp = self._ensure_active_subpath(fallback_start=Point(0.0, 0.0))
        sp.commands.append(QuadraticTo(ctrl, end))
        return self

    def cubic_to(
        self,
        c1: Point,
        c2: Point,
        end: Point
    ) -> Path:
        """Add a cubic Bézier curve to the active subpath."""
        if not (isinstance(c1, Point) and isinstance(c2, Point) and isinstance(end, Point)):
            raise ValidationError("cubic_to expects Point instances for c1, c2, and end")

        sp = self._ensure_active_subpath(fallback_start=Point(0.0, 0.0))
        sp.commands.append(CubicTo(c1, c2, end))
        return self

    def close(self) -> Path:
        """Close the current subpath with a straight segment back to its start point."""
        if self.subpaths and not self.subpaths[-1].closed:
            sp = self.subpaths[-1]
            sp.commands.append(Close())
            sp.closed = True
        return self

    # -------------------------------------------------------------------------
    # Vector Path Boolean Operations
    # -------------------------------------------------------------------------

    def boolean(self, other: Path, operation: PathBooleanOp | str) -> Path:
        """Execute a 2D vector path boolean operation against another Path.
        
        Args:
            other: Secondary Path operand.
            operation: Boolean operation ('union', 'intersection', 'difference', 'xor' or PathBooleanOp).
            
        Returns:
            A new detached Path representing the world-space boolean geometry with identity transform.
        """
        from drawcv.core.path_boolean import apply_path_boolean
        return apply_path_boolean(self, other, operation)

    def union(self, other: Path) -> Path:
        """Return the geometric union of this Path and another Path."""
        return self.boolean(other, PathBooleanOp.UNION)

    def intersection(self, other: Path) -> Path:
        """Return the geometric intersection of this Path and another Path."""
        return self.boolean(other, PathBooleanOp.INTERSECTION)

    def difference(self, other: Path) -> Path:
        """Return the geometric difference (subtraction: self - other) of this Path and another Path."""
        return self.boolean(other, PathBooleanOp.DIFFERENCE)

    def xor(self, other: Path) -> Path:
        """Return the symmetric difference (exclusive-or) of this Path and another Path."""
        return self.boolean(other, PathBooleanOp.XOR)

    # -------------------------------------------------------------------------
    # Screen-Space Flattening
    # -------------------------------------------------------------------------

    def flatten_world(self, tolerance: float = 0.5, *, include_closed: bool = False):
        """Derive flattened point contours mapped directly into world space.
        
        Evaluates de Casteljau adaptive subdivision on transformed control points
        so the tolerance (in screen pixels) remains scale-invariant.
        """
        world_contours: list[list[Point]] = []
        closed_flags: list[bool] = []

        for sp in self.subpaths:
            if not sp.commands:
                continue

            current_w: Point | None = None
            start_w: Point | None = None
            contour: list[Point] = []

            for cmd in sp.commands:
                if isinstance(cmd, MoveTo):
                    pt_w = self.to_world(cmd.point)
                    if contour:
                        world_contours.append(contour)
                        closed_flags.append(False)
                        contour = []
                    current_w = pt_w
                    start_w = pt_w
                    contour.append(pt_w)

                elif isinstance(cmd, LineTo):
                    if current_w is None:
                        current_w = self.to_world(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    pt_w = self.to_world(cmd.point)
                    contour.append(pt_w)
                    current_w = pt_w

                elif isinstance(cmd, QuadraticTo):
                    if current_w is None:
                        current_w = self.to_world(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    ctrl_w = self.to_world(cmd.control)
                    end_w = self.to_world(cmd.end)
                    subdiv = flatten_quadratic_bezier(current_w, ctrl_w, end_w, tolerance=tolerance)
                    # Skip the first point since it matches current_w
                    contour.extend(subdiv[1:])
                    current_w = end_w

                elif isinstance(cmd, CubicTo):
                    if current_w is None:
                        current_w = self.to_world(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    c1_w = self.to_world(cmd.control1)
                    c2_w = self.to_world(cmd.control2)
                    end_w = self.to_world(cmd.end)
                    subdiv = flatten_cubic_bezier(current_w, c1_w, c2_w, end_w, tolerance=tolerance)
                    contour.extend(subdiv[1:])
                    current_w = end_w

                elif isinstance(cmd, Close):
                    if start_w is not None and current_w is not None and current_w != start_w:
                        contour.append(start_w)
                        current_w = start_w

            if contour:
                world_contours.append(contour)
                closed_flags.append(sp.closed)

        if include_closed:
            return list(zip(world_contours, closed_flags))
        return world_contours

    def flatten_with_mapper(
        self,
        map_point: Callable[[Point], Point],
        tolerance: float = 0.5,
        *,
        include_closed: bool = False,
    ):
        """Derive flattened point contours transformed through a custom point mapper.

        Evaluates de Casteljau adaptive subdivision on mapped control points
        so the tolerance (in screen pixels) remains scale-invariant.
        """
        world_contours: list[list[Point]] = []
        closed_flags: list[bool] = []

        for sp in self.subpaths:
            if not sp.commands:
                continue

            current_w: Point | None = None
            start_w: Point | None = None
            contour: list[Point] = []

            for cmd in sp.commands:
                if isinstance(cmd, MoveTo):
                    pt_w = map_point(cmd.point)
                    if contour:
                        world_contours.append(contour)
                        closed_flags.append(False)
                        contour = []
                    current_w = pt_w
                    start_w = pt_w
                    contour.append(pt_w)

                elif isinstance(cmd, LineTo):
                    if current_w is None:
                        current_w = map_point(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    pt_w = map_point(cmd.point)
                    contour.append(pt_w)
                    current_w = pt_w

                elif isinstance(cmd, QuadraticTo):
                    if current_w is None:
                        current_w = map_point(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    ctrl_w = map_point(cmd.control)
                    end_w = map_point(cmd.end)
                    subdiv = flatten_quadratic_bezier(current_w, ctrl_w, end_w, tolerance=tolerance)
                    # Skip the first point since it matches current_w
                    contour.extend(subdiv[1:])
                    current_w = end_w

                elif isinstance(cmd, CubicTo):
                    if current_w is None:
                        current_w = map_point(Point(0.0, 0.0))
                        start_w = current_w
                        contour.append(current_w)
                    c1_w = map_point(cmd.control1)
                    c2_w = map_point(cmd.control2)
                    end_w = map_point(cmd.end)
                    subdiv = flatten_cubic_bezier(current_w, c1_w, c2_w, end_w, tolerance=tolerance)
                    contour.extend(subdiv[1:])
                    current_w = end_w

                elif isinstance(cmd, Close):
                    if start_w is not None and current_w is not None and current_w != start_w:
                        contour.append(start_w)
                        current_w = start_w

            if contour:
                world_contours.append(contour)
                closed_flags.append(sp.closed)

        if include_closed:
            return list(zip(world_contours, closed_flags))
        return world_contours

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local space using analytical command extrema."""
        all_boxes: list[BoundingBox] = []

        for sp in self.subpaths:
            cur: Point = Point(0.0, 0.0)
            for cmd in sp.commands:
                if isinstance(cmd, MoveTo):
                    cur = cmd.point
                    all_boxes.append(BoundingBox(cur.x, cur.y, 0.0, 0.0))
                elif isinstance(cmd, LineTo):
                    min_x, max_x = min(cur.x, cmd.point.x), max(cur.x, cmd.point.x)
                    min_y, max_y = min(cur.y, cmd.point.y), max(cur.y, cmd.point.y)
                    all_boxes.append(BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y))
                    cur = cmd.point
                elif isinstance(cmd, QuadraticTo):
                    x, y, w, h = bezier_extrema_bounds(cur, cmd.control, cmd.end)
                    all_boxes.append(BoundingBox(x, y, w, h))
                    cur = cmd.end
                elif isinstance(cmd, CubicTo):
                    x, y, w, h = bezier_extrema_bounds(cur, cmd.control1, cmd.control2, cmd.end)
                    all_boxes.append(BoundingBox(x, y, w, h))
                    cur = cmd.end

        if not all_boxes:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)

        result = all_boxes[0]
        for b in all_boxes[1:]:
            result = result.union(b)
        return result

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space."""
        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return self.get_geometry_bounds().expand(half_stroke)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB using analytical extrema of transformed segments."""
        all_boxes: list[BoundingBox] = []

        for sp in self.subpaths:
            cur_w: Point = self.to_world(Point(0.0, 0.0))
            for cmd in sp.commands:
                if isinstance(cmd, MoveTo):
                    cur_w = self.to_world(cmd.point)
                    all_boxes.append(BoundingBox(cur_w.x, cur_w.y, 0.0, 0.0))
                elif isinstance(cmd, LineTo):
                    p_w = self.to_world(cmd.point)
                    min_x, max_x = min(cur_w.x, p_w.x), max(cur_w.x, p_w.x)
                    min_y, max_y = min(cur_w.y, p_w.y), max(cur_w.y, p_w.y)
                    all_boxes.append(BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y))
                    cur_w = p_w
                elif isinstance(cmd, QuadraticTo):
                    c_w = self.to_world(cmd.control)
                    e_w = self.to_world(cmd.end)
                    x, y, w, h = bezier_extrema_bounds(cur_w, c_w, e_w)
                    all_boxes.append(BoundingBox(x, y, w, h))
                    cur_w = e_w
                elif isinstance(cmd, CubicTo):
                    c1_w = self.to_world(cmd.control1)
                    c2_w = self.to_world(cmd.control2)
                    e_w = self.to_world(cmd.end)
                    x, y, w, h = bezier_extrema_bounds(cur_w, c1_w, c2_w, e_w)
                    all_boxes.append(BoundingBox(x, y, w, h))
                    cur_w = e_w

        if not all_boxes:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)

        result = all_boxes[0]
        for b in all_boxes[1:]:
            result = result.union(b)

        half_stroke = (self.stroke.bounds_padding) if self.stroke else 0.0
        return result.expand(half_stroke)

    # -------------------------------------------------------------------------
    # Anchors & Hit Testing
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates."""
        norm = name.strip().lower()
        if norm == "start":
            if self.subpaths and self.subpaths[0].start_point is not None:
                return self.to_world(self.subpaths[0].start_point)
            return self.to_world(Point(0.0, 0.0))

        if norm == "center":
            gb = self.get_geometry_bounds()
            return self.to_world(Point(gb.x + gb.width / 2.0, gb.y + gb.height / 2.0))

        valid = "start, center"
        raise ValidationError(f"Unknown anchor '{name}' for Path. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test if a world-space point lies within stroke distance or interior fill of the Path."""
        if not isinstance(world_point, Point):
            raise ValidationError(f"Expected Point, got {type(world_point).__name__}")

        half_stroke = (self.stroke.width / 2.0) if self.stroke else 1.0
        tolerance = max(5.0, half_stroke)

        world_contours = self.flatten_world(tolerance=0.5)

        # 1. Stroke proximity check along all segments
        for contour in world_contours:
            for i in range(len(contour) - 1):
                if distance_point_to_segment(world_point, contour[i], contour[i + 1]) <= tolerance:
                    return True

        # 2. Interior containment check for closed subpaths
        if self.fill is not None:
            # Test interior according to fill_rule
            closed_contours = [c for idx, c in enumerate(world_contours)
                               if idx < len(self.subpaths) and self.subpaths[idx].closed and len(c) >= 3]
            if closed_contours:
                if self.fill_rule == FillRule.EVEN_ODD:
                    inside_count = sum(1 for c in closed_contours if point_in_polygon(world_point, c, include_boundary=True))
                    if inside_count % 2 == 1:
                        return True
                else:  # NON_ZERO (orientation winding)
                    winding = 0
                    for c in closed_contours:
                        if point_in_polygon(world_point, c, include_boundary=True):
                            # Orientation via Shoelace
                            area = sum(c[i].x * c[(i+1)%len(c)].y - c[(i+1)%len(c)].x * c[i].y for i in range(len(c)))
                            winding += 1 if area >= 0 else -1
                    if winding != 0:
                        return True

        return False

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "subpaths": [copy.deepcopy(sp) for sp in self.subpaths],
            "fill_rule": self.fill_rule.value if hasattr(self.fill_rule, "value") else str(self.fill_rule),
            "stroke": self.stroke.copy() if self.stroke else None,
            "fill": self.fill.copy() if self.fill else None,
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "subpaths" in state:
            self.subpaths = [copy.deepcopy(sp) for sp in state["subpaths"]]
        if "fill_rule" in state:
            fr = state["fill_rule"]
            self.fill_rule = FillRule(fr) if isinstance(fr, str) else fr
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
            "type": "path",
            "subpaths": serialize_subpaths(self.subpaths),
            "fill_rule": self.fill_rule.value if hasattr(self.fill_rule, "value") else str(self.fill_rule),
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "fill": self.fill.to_dict() if self.fill else None,
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Path:
        """Construct a Path from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        subpaths = deserialize_subpaths(data.get("subpaths", []))
        fill_rule_val = FillRule(data["fill_rule"]) if "fill_rule" in data else FillRule.NON_ZERO
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        fill = FillStyle.from_dict(data["fill"]) if data.get("fill") is not None else None
        return cls(
            subpaths=subpaths,
            stroke=stroke,
            fill=fill,
            fill_rule=fill_rule_val,
            **base_kwargs,
        )

