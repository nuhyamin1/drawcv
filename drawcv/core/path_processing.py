"""Modular path processing algorithms for freehand strokes and curve geometry."""

from __future__ import annotations
import math
from typing import Sequence

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point, StrokePoint, distance_point_to_segment


def interpolate_metadata(
    val1: float | None,
    val2: float | None,
    t: float,
    clamp_range: tuple[float, float] | None = None,
) -> float | None:
    """Linearly interpolate metadata between two points according to deterministic None rules.
    
    Rules:
        - Both values present: linear interpolation (1-t)*val1 + t*val2 (optionally clamped).
        - Only one present: return the present value.
        - Both None: return None.
    """
    if val1 is not None and val2 is not None:
        res = (1.0 - t) * val1 + t * val2
        if clamp_range is not None:
            res = max(clamp_range[0], min(clamp_range[1], res))
        return res
    if val1 is not None:
        return val1
    if val2 is not None:
        return val2
    return None


def interpolate_stroke_point(p1: StrokePoint, p2: StrokePoint, t: float) -> StrokePoint:
    """Linearly interpolate coordinates and physical metadata between two StrokePoints."""
    x = (1.0 - t) * p1.x + t * p2.x
    y = (1.0 - t) * p1.y + t * p2.y
    pressure = interpolate_metadata(p1.pressure, p2.pressure, t, clamp_range=(0.0, 1.0))
    timestamp = interpolate_metadata(p1.timestamp, p2.timestamp, t)
    velocity = interpolate_metadata(p1.velocity, p2.velocity, t, clamp_range=(0.0, float("inf")))
    return StrokePoint(x, y, pressure=pressure, timestamp=timestamp, velocity=velocity)


def compute_path_length(points: Sequence[StrokePoint | Point]) -> float:
    """Calculate the cumulative Euclidean length of a sequence of points."""
    if len(points) < 2:
        return 0.0
    return sum(points[i].distance_to(points[i + 1]) for i in range(len(points) - 1))


def rdp_simplify(points: list[StrokePoint], epsilon: float) -> list[StrokePoint]:
    """Ramer-Douglas-Peucker (RDP) path simplification algorithm.
    
    Reduces the number of points in a stroke while preserving features exceeding
    the perpendicular distance tolerance epsilon. Start and end points and all
    StrokePoint attributes are strictly preserved.
    """
    if not isinstance(points, list):
        raise ValidationError(f"points must be a list, got {type(points).__name__}")
    if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon) or epsilon < 0.0:
        raise ValidationError(f"Simplification tolerance (epsilon) must be a non-negative number, got {epsilon}")

    if len(points) <= 2 or epsilon == 0.0:
        return [p.copy() for p in points]

    start = points[0]
    end = points[-1]
    dmax = 0.0
    index = 0

    for i in range(1, len(points) - 1):
        d = distance_point_to_segment(points[i], start, end)
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        left = rdp_simplify(points[: index + 1], epsilon)
        right = rdp_simplify(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0].copy(), points[-1].copy()]


def chaikin_smooth(
    points: list[StrokePoint],
    iterations: int = 1,
    ratio: float = 0.25,
) -> list[StrokePoint]:
    """Chaikin's corner-cutting curve smoothing algorithm.
    
    Iteratively refines an open polyline by cutting each corner at `ratio`
    and `1.0 - ratio`. Preserves open stroke endpoints and linearly interpolates
    all physical metadata (pressure, timestamp, velocity).
    """
    if not isinstance(points, list):
        raise ValidationError(f"points must be a list, got {type(points).__name__}")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValidationError(f"Smoothing iterations must be a non-negative integer, got {iterations}")
    if not isinstance(ratio, (int, float)) or math.isnan(ratio) or not (0.0 < float(ratio) < 0.5):
        raise ValidationError(f"Smoothing ratio must be between 0.0 and 0.5 exclusive, got {ratio}")

    if len(points) < 3 or iterations == 0:
        return [p.copy() for p in points]

    curr = [p.copy() for p in points]
    r = float(ratio)

    for _ in range(iterations):
        smoothed = [curr[0].copy()]
        for i in range(len(curr) - 1):
            p_a = curr[i]
            p_b = curr[i + 1]
            q = interpolate_stroke_point(p_a, p_b, r)
            s = interpolate_stroke_point(p_a, p_b, 1.0 - r)
            smoothed.append(q)
            smoothed.append(s)
        smoothed.append(curr[-1].copy())
        curr = smoothed

    return curr


def catmull_rom_spline(
    points: list[StrokePoint],
    samples_per_segment: int = 8,
    alpha: float = 0.5,
) -> list[StrokePoint]:
    """Centripetal Catmull-Rom cubic spline interpolation.
    
    Generates a smooth C1-continuous curve that passes exactly through all input control
    points. Uses centripetal parametrization (alpha=0.5) to strongly reduce pathological
    behavior and avoid cusps and self-intersections within individual spline segments.
    
    Physical metadata (pressure, timestamp, velocity) is linearly interpolated between
    adjacent control points to ensure valid ranges and non-decreasing timestamps.
    """
    if not isinstance(points, list):
        raise ValidationError(f"points must be a list, got {type(points).__name__}")
    if not isinstance(samples_per_segment, int) or samples_per_segment < 1:
        raise ValidationError(f"Interpolation samples per segment must be an integer >= 1, got {samples_per_segment}")
    if not isinstance(alpha, (int, float)) or math.isnan(alpha) or float(alpha) < 0.0:
        raise ValidationError(f"alpha must be non-negative, got {alpha}")

    n = len(points)
    if n < 2:
        return [p.copy() for p in points]

    if n == 2:
        return [
            interpolate_stroke_point(points[0], points[1], i / samples_per_segment)
            for i in range(samples_per_segment + 1)
        ]

    # Extend ghost control points at boundaries: P_{-1} = 2*P_0 - P_1, P_n = 2*P_{n-1} - P_{n-2}
    p_first = points[0]
    p_second = points[1]
    p_last = points[-1]
    p_prev_last = points[-2]

    ghost_start = StrokePoint(
        2.0 * p_first.x - p_second.x,
        2.0 * p_first.y - p_second.y,
        pressure=p_first.pressure,
        timestamp=p_first.timestamp,
        velocity=p_first.velocity,
    )
    ghost_end = StrokePoint(
        2.0 * p_last.x - p_prev_last.x,
        2.0 * p_last.y - p_prev_last.y,
        pressure=p_last.pressure,
        timestamp=p_last.timestamp,
        velocity=p_last.velocity,
    )

    ctrl = [ghost_start] + [p.copy() for p in points] + [ghost_end]
    result: list[StrokePoint] = []
    a = float(alpha)

    for i in range(n - 1):
        p0 = ctrl[i]
        p1 = ctrl[i + 1]
        p2 = ctrl[i + 2]
        p3 = ctrl[i + 3]

        d01 = math.hypot(p1.x - p0.x, p1.y - p0.y)
        d12 = math.hypot(p2.x - p1.x, p2.y - p1.y)
        d23 = math.hypot(p3.x - p2.x, p3.y - p2.y)

        t0 = 0.0
        t1 = t0 + (max(1e-6, d01) ** a)
        t2 = t1 + (max(1e-6, d12) ** a)
        t3 = t2 + (max(1e-6, d23) ** a)

        # Include end vertex only on the last segment to avoid duplicated points
        num_steps = samples_per_segment + 1 if i == n - 2 else samples_per_segment

        for step in range(num_steps):
            u = step / samples_per_segment
            t = t1 + u * (t2 - t1)

            # Pyramidal de Casteljau-like evaluation for Catmull-Rom
            a1_x = ((t1 - t) * p0.x + (t - t0) * p1.x) / (t1 - t0)
            a1_y = ((t1 - t) * p0.y + (t - t0) * p1.y) / (t1 - t0)

            a2_x = ((t2 - t) * p1.x + (t - t1) * p2.x) / (t2 - t1)
            a2_y = ((t2 - t) * p1.y + (t - t1) * p2.y) / (t2 - t1)

            a3_x = ((t3 - t) * p2.x + (t - t2) * p3.x) / (t3 - t2)
            a3_y = ((t3 - t) * p2.y + (t - t2) * p3.y) / (t3 - t2)

            b1_x = ((t2 - t) * a1_x + (t - t0) * a2_x) / (t2 - t0)
            b1_y = ((t2 - t) * a1_y + (t - t0) * a2_y) / (t2 - t0)

            b2_x = ((t3 - t) * a2_x + (t - t1) * a3_x) / (t3 - t1)
            b2_y = ((t3 - t) * a2_y + (t - t1) * a3_y) / (t3 - t1)

            cx = ((t2 - t) * b1_x + (t - t1) * b2_x) / (t2 - t1)
            cy = ((t2 - t) * b1_y + (t - t1) * b2_y) / (t2 - t1)

            # Metadata is strictly linearly interpolated between control points p1 and p2
            pres = interpolate_metadata(p1.pressure, p2.pressure, u, clamp_range=(0.0, 1.0))
            ts = interpolate_metadata(p1.timestamp, p2.timestamp, u)
            vel = interpolate_metadata(p1.velocity, p2.velocity, u, clamp_range=(0.0, float("inf")))

            result.append(StrokePoint(cx, cy, pressure=pres, timestamp=ts, velocity=vel))

    return result


# -----------------------------------------------------------------------------
# Progressive Path & Curve Slicing (Phase 8)
# -----------------------------------------------------------------------------

def slice_polyline(points: list[Point], progress: float, closed: bool = False) -> list[Point]:
    """Arc-length slice of a polyline at fractional progress in [0, 1]."""
    if not points:
        return []
    if len(points) == 1 or progress <= 0.0:
        return [points[0].copy()]

    pts = [p.copy() for p in points]
    if closed and (pts[-1].x != pts[0].x or pts[-1].y != pts[0].y):
        pts.append(pts[0].copy())

    if progress >= 1.0:
        return pts

    total_len = compute_path_length(pts)
    if total_len <= 1e-9:
        return [pts[0].copy()]

    target_dist = float(progress) * total_len
    accum = 0.0
    result = [pts[0].copy()]

    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        seg_len = p1.distance_to(p2)
        if seg_len <= 1e-9:
            continue
        if accum + seg_len >= target_dist:
            remaining = target_dist - accum
            t = max(0.0, min(1.0, remaining / seg_len))
            cut = Point(
                (1.0 - t) * p1.x + t * p2.x,
                (1.0 - t) * p1.y + t * p2.y,
            )
            result.append(cut)
            return result
        accum += seg_len
        result.append(p2.copy())

    return result


def slice_stroke_points(points: list[StrokePoint], progress: float) -> list[StrokePoint]:
    """Arc-length slice of StrokePoints with interpolated pressure, timestamp, velocity."""
    if not points:
        return []
    if len(points) == 1 or progress <= 0.0:
        return [points[0].copy()]
    if progress >= 1.0:
        return [p.copy() for p in points]

    total_len = compute_path_length(points)
    if total_len <= 1e-9:
        return [points[0].copy()]

    target_dist = float(progress) * total_len
    accum = 0.0
    result = [points[0].copy()]

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        seg_len = p1.distance_to(p2)
        if seg_len <= 1e-9:
            continue
        if accum + seg_len >= target_dist:
            remaining = target_dist - accum
            t = max(0.0, min(1.0, remaining / seg_len))
            cut = interpolate_stroke_point(p1, p2, t)
            result.append(cut)
            return result
        accum += seg_len
        result.append(p2.copy())

    return result


def _cubic_bezier_length(p0: Point, p1: Point, p2: Point, p3: Point, t_end: float = 1.0, num_steps: int = 40) -> float:
    """Numerically approximate the cumulative arc length of a cubic Bézier curve on [0, t_end]."""
    if t_end <= 0.0:
        return 0.0
    
    # Sample points on [0, t_end]
    length = 0.0
    prev_pt = p0
    for i in range(1, num_steps + 1):
        t = (i / num_steps) * t_end
        u = 1.0 - t
        x = u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x
        y = u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y
        curr_pt = Point(x, y)
        length += prev_pt.distance_to(curr_pt)
        prev_pt = curr_pt
    return length


def slice_bezier(p0: Point, p1: Point, p2: Point, p3: Point, progress: float) -> tuple[Point, Point, Point, Point]:
    """Numerically estimate cumulative arc length, solve parameter t, and de Casteljau split."""
    if progress <= 0.0:
        return (p0.copy(), p0.copy(), p0.copy(), p0.copy())
    if progress >= 1.0:
        return (p0.copy(), p1.copy(), p2.copy(), p3.copy())

    total_len = _cubic_bezier_length(p0, p1, p2, p3, t_end=1.0)
    if total_len <= 1e-9:
        return (p0.copy(), p0.copy(), p0.copy(), p0.copy())

    target_len = float(progress) * total_len

    # Bisection root-finding for parameter t where length(t) = target_len
    low = 0.0
    high = 1.0
    best_t = float(progress)  # initial guess

    for _ in range(25):
        mid = (low + high) / 2.0
        cur_len = _cubic_bezier_length(p0, p1, p2, p3, t_end=mid)
        if abs(cur_len - target_len) < 0.05:
            best_t = mid
            break
        if cur_len < target_len:
            low = mid
            best_t = mid
        else:
            high = mid
            best_t = mid

    # De Casteljau subdivision at parameter best_t
    t = max(0.0, min(1.0, best_t))
    u = 1.0 - t

    m01 = Point(u * p0.x + t * p1.x, u * p0.y + t * p1.y)
    m12 = Point(u * p1.x + t * p2.x, u * p1.y + t * p2.y)
    m23 = Point(u * p2.x + t * p3.x, u * p2.y + t * p3.y)

    m012 = Point(u * m01.x + t * m12.x, u * m01.y + t * m12.y)
    m123 = Point(u * m12.x + t * m23.x, u * m12.y + t * m23.y)

    m0123 = Point(u * m012.x + t * m123.x, u * m012.y + t * m123.y)

    return (p0.copy(), m01, m012, m0123)


def slice_path(path: Any, progress: float) -> Any:
    """Arc-length slice of a compound Path up to fractional progress in [0, 1]."""
    from drawcv.shapes.path import Path, Subpath, MoveTo, LineTo, QuadraticTo, CubicTo, Close

    if progress <= 0.0:
        empty = Path(fill_rule=path.fill_rule, stroke=path.stroke, fill=None)
        if path.subpaths and path.subpaths[0].commands:
            first_cmd = path.subpaths[0].commands[0]
            if isinstance(first_cmd, MoveTo):
                empty.move_to(first_cmd.point.x, first_cmd.point.y)
        return empty

    if progress >= 1.0:
        return path

    # 1. Compute arc-length of each command across all subpaths
    cmd_records: list[tuple[Subpath, Any, float, Point, Point]] = []
    # (subpath, cmd, length, start_pt, end_pt)
    total_length = 0.0

    for sp in path.subpaths:
        current_pt = Point(0.0, 0.0)
        start_pt = Point(0.0, 0.0)

        for cmd in sp.commands:
            if isinstance(cmd, MoveTo):
                current_pt = cmd.point
                start_pt = cmd.point
                cmd_records.append((sp, cmd, 0.0, current_pt, current_pt))
            elif isinstance(cmd, LineTo):
                seg_len = current_pt.distance_to(cmd.point)
                cmd_records.append((sp, cmd, seg_len, current_pt, cmd.point))
                total_length += seg_len
                current_pt = cmd.point
            elif isinstance(cmd, Close):
                seg_len = current_pt.distance_to(start_pt)
                cmd_records.append((sp, cmd, seg_len, current_pt, start_pt))
                total_length += seg_len
                current_pt = start_pt
            elif isinstance(cmd, CubicTo):
                seg_len = _cubic_bezier_length(current_pt, cmd.control1, cmd.control2, cmd.end)
                cmd_records.append((sp, cmd, seg_len, current_pt, cmd.end))
                total_length += seg_len
                current_pt = cmd.end
            elif isinstance(cmd, QuadraticTo):
                # Elevate quadratic to cubic for arc-length
                c1 = Point(
                    current_pt.x + (2.0 / 3.0) * (cmd.control.x - current_pt.x),
                    current_pt.y + (2.0 / 3.0) * (cmd.control.y - current_pt.y),
                )
                c2 = Point(
                    cmd.end.x + (2.0 / 3.0) * (cmd.control.x - cmd.end.x),
                    cmd.end.y + (2.0 / 3.0) * (cmd.control.y - cmd.end.y),
                )
                seg_len = _cubic_bezier_length(current_pt, c1, c2, cmd.end)
                cmd_records.append((sp, cmd, seg_len, current_pt, cmd.end))
                total_length += seg_len
                current_pt = cmd.end

    if total_length <= 1e-9:
        return path

    target_dist = float(progress) * total_length
    accum = 0.0

    sliced_path = Path(fill_rule=path.fill_rule, stroke=path.stroke, fill=None)
    current_sp = Subpath()
    sliced_path.subpaths.append(current_sp)

    for sp, cmd, cmd_len, p_start, p_end in cmd_records:
        if isinstance(cmd, MoveTo):
            if current_sp.commands:
                current_sp = Subpath()
                sliced_path.subpaths.append(current_sp)
            current_sp.commands.append(MoveTo(cmd.point.copy()))
            continue

        if cmd_len <= 1e-9:
            continue

        if accum + cmd_len >= target_dist:
            remaining = target_dist - accum
            frac = max(0.0, min(1.0, remaining / cmd_len))

            if isinstance(cmd, (LineTo, Close)):
                cut_pt = Point(
                    (1.0 - frac) * p_start.x + frac * p_end.x,
                    (1.0 - frac) * p_start.y + frac * p_end.y,
                )
                current_sp.commands.append(LineTo(cut_pt))
            elif isinstance(cmd, CubicTo):
                _, q1, q2, q3 = slice_bezier(p_start, cmd.control1, cmd.control2, cmd.end, frac)
                current_sp.commands.append(CubicTo(q1, q2, q3))
            elif isinstance(cmd, QuadraticTo):
                # Quadratic de Casteljau split
                t = frac
                u = 1.0 - t
                m1 = Point(u * p_start.x + t * cmd.control.x, u * p_start.y + t * cmd.control.y)
                m2 = Point(u * cmd.control.x + t * cmd.end.x, u * cmd.control.y + t * cmd.end.y)
                m12 = Point(u * m1.x + t * m2.x, u * m1.y + t * m2.y)
                current_sp.commands.append(QuadraticTo(m1, m12))
            break
        else:
            current_sp.commands.append(cmd)
            accum += cmd_len

    return sliced_path

