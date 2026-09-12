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
