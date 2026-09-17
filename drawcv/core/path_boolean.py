"""Internal Skia PathOps geometry adapter for DrawCV true vector path boolean operations."""

from __future__ import annotations
import copy
import math
import numpy as np
import pathops

from drawcv.core.color import Color
from drawcv.core.enums import FillRule, PathBooleanOp
from drawcv.core.exceptions import PathBooleanError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.shapes.path import (
    Close,
    CubicTo,
    EllipticalArcTo,
    LineTo,
    MoveTo,
    Path,
    QuadraticTo,
    Subpath,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.paint import LinearGradient, RadialGradient
from drawcv.styles.stroke import StrokeStyle


# Explicit tolerance for converting rational conics (if emitted by backend) into quadratics.
# DrawCV retains lines, quadratics, and cubics natively; any conics produced by Skia
# are approximated to quadratics with this canvas-coordinate tolerance.
_CONIC_TO_QUAD_TOLERANCE: float = 0.25


def _validate_finite_point(pt: Point, context: str) -> None:
    if not isinstance(pt, Point) or not (math.isfinite(pt.x) and math.isfinite(pt.y)):
        raise ValidationError(f"{context} must contain finite coordinates, got {pt}")


def _validate_backend_point(pt: Point, context: str) -> None:
    if not isinstance(pt, Point) or not (math.isfinite(pt.x) and math.isfinite(pt.y)):
        raise PathBooleanError(
            f"{context} contains non-finite coordinates: {pt}"
        )


def _to_backend_point(raw_x: float, raw_y: float, context: str) -> Point:
    x = float(raw_x)
    y = float(raw_y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise PathBooleanError(
            f"{context} contains non-finite coordinates: ({x}, {y})"
        )
    pt = Point(x, y)
    _validate_backend_point(pt, context)
    return pt


def drawcv_to_pathops(path: Path) -> pathops.Path:
    """Convert a DrawCV Path into a PathOps Path in common world space.
    
    Transforms endpoints and control points through the full ancestor world hierarchy
    without flattening Bezier curves. Implicitly closes open subpaths for boolean fill evaluation.
    """
    if not isinstance(path, Path):
        raise ValidationError(f"Expected Path instance, got {type(path).__name__}")

    p_ops = pathops.Path()
    p_ops.fillType = (
        pathops.FillType.EVEN_ODD
        if path.fill_rule == FillRule.EVEN_ODD
        else pathops.FillType.WINDING
    )

    for sp in path.subpaths:
        if not sp.commands:
            continue

        def to_w(p: Point, name: str) -> Point:
            _validate_finite_point(p, f"Local command point '{name}'")
            pw = path.to_world(p)
            _validate_finite_point(pw, f"World-space transformed point '{name}'")
            return pw

        has_close = False
        has_drawing_command = False

        for cmd in sp.commands:
            if isinstance(cmd, MoveTo):
                pw = to_w(cmd.point, "MoveTo")
                p_ops.moveTo(pw.x, pw.y)

            elif isinstance(cmd, LineTo):
                pw = to_w(cmd.point, "LineTo")
                p_ops.lineTo(pw.x, pw.y)
                has_drawing_command = True

            elif isinstance(cmd, QuadraticTo):
                ctrl_w = to_w(cmd.control, "QuadraticTo.control")
                end_w = to_w(cmd.end, "QuadraticTo.end")
                p_ops.quadTo(ctrl_w.x, ctrl_w.y, end_w.x, end_w.y)
                has_drawing_command = True

            elif isinstance(cmd, CubicTo):
                c1_w = to_w(cmd.control1, "CubicTo.control1")
                c2_w = to_w(cmd.control2, "CubicTo.control2")
                end_w = to_w(cmd.end, "CubicTo.end")
                p_ops.cubicTo(c1_w.x, c1_w.y, c2_w.x, c2_w.y, end_w.x, end_w.y)
                has_drawing_command = True

            elif isinstance(cmd, EllipticalArcTo):
                raise PathBooleanError(
                    "EllipticalArcTo commands are not supported in PathOps boolean operations without flattening."
                )

            elif isinstance(cmd, Close):
                if not has_close:
                    p_ops.close()
                    has_close = True

        # Single-close and implicit fill-closure normalization:
        # If not explicitly closed by a Close command, close once if marked closed
        # or if it has drawing commands (boolean operations operate on fill geometry).
        if not has_close and (sp.closed or has_drawing_command):
            p_ops.close()

    return p_ops


def pathops_to_drawcv(p_ops: pathops.Path) -> Path:
    """Convert a PathOps Path back into an editable DrawCV Path with NON_ZERO fill rule."""
    p_ops.convertConicsToQuads(tolerance=_CONIC_TO_QUAD_TOLERANCE)

    subpaths: list[Subpath] = []
    current_subpath: Subpath | None = None

    try:
        for verb, pts in p_ops:
            if verb == pathops.PathVerb.MOVE:
                pt = _to_backend_point(pts[0][0], pts[0][1], "PathOps MOVE point")
                current_subpath = Subpath(commands=[MoveTo(pt)], closed=False)
                subpaths.append(current_subpath)

            elif verb == pathops.PathVerb.LINE:
                if current_subpath is None:
                    raise PathBooleanError("PathOps backend emitted a LINE verb before a MOVE verb")
                pt = _to_backend_point(pts[0][0], pts[0][1], "PathOps LINE point")
                current_subpath.commands.append(LineTo(pt))

            elif verb == pathops.PathVerb.QUAD:
                if current_subpath is None:
                    raise PathBooleanError("PathOps backend emitted a QUAD verb before a MOVE verb")
                ctrl = _to_backend_point(pts[0][0], pts[0][1], "PathOps QUAD control")
                end = _to_backend_point(pts[1][0], pts[1][1], "PathOps QUAD end")
                current_subpath.commands.append(QuadraticTo(ctrl, end))

            elif verb == pathops.PathVerb.CUBIC:
                if current_subpath is None:
                    raise PathBooleanError("PathOps backend emitted a CUBIC verb before a MOVE verb")
                c1 = _to_backend_point(pts[0][0], pts[0][1], "PathOps CUBIC control1")
                c2 = _to_backend_point(pts[1][0], pts[1][1], "PathOps CUBIC control2")
                end = _to_backend_point(pts[2][0], pts[2][1], "PathOps CUBIC end")
                current_subpath.commands.append(CubicTo(c1, c2, end))

            elif verb == pathops.PathVerb.CLOSE:
                if current_subpath is None:
                    raise PathBooleanError("PathOps backend emitted a CLOSE verb without an active subpath")
                current_subpath.commands.append(Close())
                current_subpath.closed = True
                current_subpath = None

            elif verb == pathops.PathVerb.CONIC:
                raise PathBooleanError("Unexpected unconverted CONIC verb from pathops backend")

            else:
                raise PathBooleanError(f"Unexpected verb {verb} from pathops backend")

    except ValidationError as err:
        raise PathBooleanError(f"Malformed backend geometry: {err}") from err

    # The boolean result is normalized to FillRule.NON_ZERO by fix_winding=True
    return Path(
        subpaths=subpaths,
        fill_rule=FillRule.NON_ZERO,
        transform=Transform(),
    )


def _resolve_boolean_styles(left: Path) -> tuple[FillStyle | None, StrokeStyle | None]:
    """Resolve deterministic fill and stroke styles from the primary (left) operand."""
    resolved_stroke = copy.deepcopy(left.stroke) if left.stroke is not None else None
    resolved_fill: FillStyle | None = None

    if left.fill is not None:
        paint = left.fill.paint
        if isinstance(paint, Color):
            resolved_fill = copy.deepcopy(left.fill)

        elif getattr(paint, "space", None) == "world":
            resolved_fill = copy.deepcopy(left.fill)

        elif getattr(paint, "space", None) == "object":
            # Derive the authoritative effective linear 2x2 transform from to_world
            p_origin = left.to_world(Point(0.0, 0.0))
            p_ex = left.to_world(Point(1.0, 0.0))
            p_ey = left.to_world(Point(0.0, 1.0))
            col_x = np.array([p_ex.x - p_origin.x, p_ex.y - p_origin.y], dtype=float)
            col_y = np.array([p_ey.x - p_origin.x, p_ey.y - p_origin.y], dtype=float)
            A = np.column_stack([col_x, col_y])
            det = float(np.linalg.det(A))

            if abs(det) > 1e-12:
                if isinstance(paint, LinearGradient):
                    # Linear gradient is an affine scalar field:
                    # t(p_obj) = dot(p_obj - start, v) / dot(v, v)
                    # Under world = A * p_obj + b, the world-space gradient vector is:
                    # g = A^(-T) * v / dot(v, v)
                    # Equivalent world endpoints satisfy w = g / dot(g, g) with start at left.to_world(start).
                    v = np.array([paint.end.x - paint.start.x, paint.end.y - paint.start.y], dtype=float)
                    v_dot_v = float(np.dot(v, v))
                    if v_dot_v > 1e-14:
                        A_inv_T = np.linalg.inv(A).T
                        g = A_inv_T @ v / v_dot_v
                        g_dot_g = float(np.dot(g, g))
                        if g_dot_g > 1e-14:
                            w = g / g_dot_g
                            p0_w = left.to_world(paint.start)
                            p1_w = Point(p0_w.x + float(w[0]), p0_w.y + float(w[1]))
                            if (
                                math.isfinite(p0_w.x)
                                and math.isfinite(p0_w.y)
                                and math.isfinite(p1_w.x)
                                and math.isfinite(p1_w.y)
                            ):
                                world_grad = LinearGradient(
                                    start=p0_w,
                                    end=p1_w,
                                    stops=copy.deepcopy(paint.stops),
                                    space="world",
                                )
                                resolved_fill = FillStyle(
                                    paint=world_grad,
                                    opacity=left.fill.opacity,
                                    enabled=left.fill.enabled,
                                )

                elif isinstance(paint, RadialGradient):
                    # Radial gradient requires a similarity transform (uniform scale, no shear)
                    # to remain an exact circular radial gradient in world space.
                    norm_x = float(np.linalg.norm(col_x))
                    norm_y = float(np.linalg.norm(col_y))
                    dot_xy = float(np.dot(col_x, col_y))
                    is_similarity = (
                        abs(norm_x - norm_y) / max(norm_x, norm_y, 1e-12) < 1e-5
                        and abs(dot_xy) / (norm_x * norm_y + 1e-12) < 1e-5
                    )
                    if is_similarity:
                        scale_factor = norm_x
                        world_center = left.to_world(paint.center)
                        world_radius = paint.radius * scale_factor
                        if (
                            math.isfinite(world_center.x)
                            and math.isfinite(world_center.y)
                            and math.isfinite(world_radius)
                            and world_radius > 0
                        ):
                            world_grad = RadialGradient(
                                center=world_center,
                                radius=world_radius,
                                stops=copy.deepcopy(paint.stops),
                                space="world",
                            )
                            resolved_fill = FillStyle(
                                paint=world_grad,
                                opacity=left.fill.opacity,
                                enabled=left.fill.enabled,
                            )
                    # Non-uniform scale or shear turns circles into ellipses; fallback conservatively to fill=None

    return resolved_fill, resolved_stroke


def apply_path_boolean(a: Path, b: Path, operation: PathBooleanOp | str) -> Path:
    """Execute a 2D vector boolean operation between two retained Path objects.
    
    Args:
        a: Primary (left) Path operand.
        b: Secondary (right) Path operand.
        operation: PathBooleanOp enum or case-insensitive string ('union', 'intersection', 'difference', 'xor').
        
    Returns:
        A new detached Path representing the resulting world-space vector geometry with identity transform.
        
    Raises:
        ValidationError: If inputs are invalid types or contain non-finite coordinates.
        PathBooleanError: If the geometry backend encounters an unrecoverable failure.
    """
    if not isinstance(a, Path):
        raise ValidationError(f"Expected Path instance for operand 'a', got {type(a).__name__}")
    if not isinstance(b, Path):
        raise ValidationError(f"Expected Path instance for operand 'b', got {type(b).__name__}")

    if isinstance(operation, str):
        try:
            op = PathBooleanOp(operation.lower().strip())
        except ValueError:
            valid_ops = ", ".join(o.value for o in PathBooleanOp)
            raise ValidationError(
                f"Invalid boolean operation '{operation}'. Valid operations are: {valid_ops}"
            )
    elif isinstance(operation, PathBooleanOp):
        op = operation
    else:
        raise ValidationError(
            f"Expected PathBooleanOp or str for 'operation', got {type(operation).__name__}"
        )

    op_map = {
        PathBooleanOp.UNION: pathops.PathOp.UNION,
        PathBooleanOp.INTERSECTION: pathops.PathOp.INTERSECTION,
        PathBooleanOp.DIFFERENCE: pathops.PathOp.DIFFERENCE,
        PathBooleanOp.XOR: pathops.PathOp.XOR,
    }
    skia_op = op_map[op]

    # Convert operands to PathOps and execute backend boolean operation inside backend exception boundary
    try:
        p_a = drawcv_to_pathops(a)
        p_b = drawcv_to_pathops(b)
        res_ops = pathops.op(
            p_a,
            p_b,
            skia_op,
            fix_winding=True,
            keep_starting_points=False,
        )
        res = pathops_to_drawcv(res_ops)
    except ValidationError:
        raise
    except PathBooleanError:
        raise
    except Exception as err:
        raise PathBooleanError(
            f"Path boolean operation '{op.value}' failed: {err}"
        ) from err

    # Apply detached left-side style policy
    resolved_fill, resolved_stroke = _resolve_boolean_styles(a)
    res.fill = resolved_fill
    res.stroke = resolved_stroke

    return res
