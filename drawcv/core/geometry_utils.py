"""Computational geometry utilities, adaptive curve flattening, and topological fill rule evaluation."""

from __future__ import annotations
import math
import cv2
import numpy as np

from drawcv.core.enums import ArcClosure, FillRule
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


# -----------------------------------------------------------------------------
# Point and Segment Metrics
# -----------------------------------------------------------------------------

def distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    """Calculate the shortest Euclidean distance from a Point to a line segment [start, end]."""
    if not isinstance(point, Point) or not isinstance(start, Point) or not isinstance(end, Point):
        raise ValidationError("All arguments to distance_point_to_segment must be Point instances")

    dx = end.x - start.x
    dy = end.y - start.y
    l2 = dx * dx + dy * dy

    if l2 == 0.0:
        return point.distance_to(start)

    t = max(0.0, min(1.0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / l2))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return point.distance_to(projection)


# -----------------------------------------------------------------------------
# Polygon Metrics and Containment
# -----------------------------------------------------------------------------

def polygon_area(vertices: list[Point]) -> float:
    """Calculate signed area of a polygon using the Shoelace formula.
    
    Sign depends on vertex winding.
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y
    return area2 / 2.0


def polygon_centroid(vertices: list[Point]) -> Point:
    """Calculate the centroid (center of mass) of a non-degenerate polygon."""
    n = len(vertices)
    if n < 3:
        raise ValidationError("Centroid requires at least 3 vertices")

    signed_area = polygon_area(vertices)
    if abs(signed_area) <= 1e-6:
        # Collinear or degenerate: fallback to arithmetic mean
        avg_x = sum(v.x for v in vertices) / n
        avg_y = sum(v.y for v in vertices) / n
        return Point(avg_x, avg_y)

    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        factor = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y
        cx += (vertices[i].x + vertices[j].x) * factor
        cy += (vertices[i].y + vertices[j].y) * factor

    denom = 6.0 * signed_area
    return Point(cx / denom, cy / denom)


def point_in_polygon(point: Point, vertices: list[Point], include_boundary: bool = True) -> bool:
    """Test point containment within a closed polygon using ray-casting.
    
    If include_boundary is True, points on polygon edges are considered inside.
    """
    n = len(vertices)
    if n < 3:
        return False

    # Check boundary proximity first if requested
    if include_boundary:
        for i in range(n):
            j = (i + 1) % n
            if distance_point_to_segment(point, vertices[i], vertices[j]) <= 1e-5:
                return True

    inside = False
    px, py = point.x, point.y

    j = n - 1
    for i in range(n):
        xi, yi = vertices[i].x, vertices[i].y
        xj, yj = vertices[j].x, vertices[j].y

        intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-14) + xi)
        if intersect:
            inside = not inside
        j = i

    return inside


# -----------------------------------------------------------------------------
# Polyline Length & Interpolation
# -----------------------------------------------------------------------------

def polyline_length(points: list[Point]) -> float:
    """Compute cumulative Euclidean length of a sequence of points."""
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(len(points) - 1):
        total += points[i].distance_to(points[i + 1])
    return total


def point_at_polyline_length(points: list[Point], target_length: float) -> Point:
    """Locate a Point at target_length along the polyline path."""
    if not points:
        raise ValidationError("Points list cannot be empty")
    if len(points) == 1 or target_length <= 0.0:
        return points[0]

    accum = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        seg_len = p1.distance_to(p2)
        if accum + seg_len >= target_length:
            if seg_len == 0.0:
                return p1
            t = (target_length - accum) / seg_len
            return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))
        accum += seg_len

    return points[-1]


# -----------------------------------------------------------------------------
# Adaptive Curve Flattening (de Casteljau)
# -----------------------------------------------------------------------------

def flatten_quadratic_bezier(
    p0: Point, p1: Point, p2: Point, tolerance: float = 0.5, max_depth: int = 12
) -> list[Point]:
    """Adaptively subdivide a Quadratic Bézier curve until chord deviation <= tolerance."""
    result: list[Point] = [p0]

    def _subdivide(p0: Point, p1: Point, p2: Point, depth: int):
        # Midpoint of chord p0-p2
        mx = (p0.x + p2.x) / 2.0
        my = (p0.y + p2.y) / 2.0
        # Point on curve at t=0.5
        cx = (p0.x + 2.0 * p1.x + p2.x) / 4.0
        cy = (p0.y + 2.0 * p1.y + p2.y) / 4.0
        deviation = math.hypot(cx - mx, cy - my)

        if deviation <= tolerance or depth >= max_depth:
            result.append(p2)
            return

        # de Casteljau split at t=0.5
        q0 = Point((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0)
        q1 = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
        q_mid = Point((q0.x + q1.x) / 2.0, (q0.y + q1.y) / 2.0)

        _subdivide(p0, q0, q_mid, depth + 1)
        _subdivide(q_mid, q1, p2, depth + 1)

    _subdivide(p0, p1, p2, 0)
    return result


def flatten_cubic_bezier(
    p0: Point, p1: Point, p2: Point, p3: Point, tolerance: float = 0.5, max_depth: int = 14
) -> list[Point]:
    """Adaptively subdivide a Cubic Bézier curve until chord deviation <= tolerance."""
    result: list[Point] = [p0]

    def _subdivide(p0: Point, p1: Point, p2: Point, p3: Point, depth: int):
        d1 = distance_point_to_segment(p1, p0, p3)
        d2 = distance_point_to_segment(p2, p0, p3)
        dev = max(d1, d2)

        if dev <= tolerance or depth >= max_depth:
            result.append(p3)
            return

        # de Casteljau split at t=0.5
        q0 = Point((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0)
        q1 = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
        q2 = Point((p2.x + p3.x) / 2.0, (p2.y + p3.y) / 2.0)

        r0 = Point((q0.x + q1.x) / 2.0, (q0.y + q1.y) / 2.0)
        r1 = Point((q1.x + q2.x) / 2.0, (q1.y + q2.y) / 2.0)

        s0 = Point((r0.x + r1.x) / 2.0, (r0.y + r1.y) / 2.0)

        _subdivide(p0, q0, r0, s0, depth + 1)
        _subdivide(s0, r1, q2, p3, depth + 1)

    _subdivide(p0, p1, p2, p3, 0)
    return result


def flatten_arc(
    center: Point, rx: float, ry: float, start_angle: float, sweep_angle: float, tolerance: float = 0.5
) -> list[Point]:
    """Sample an arc into points so chord deviation <= tolerance."""
    if sweep_angle == 0.0 or rx <= 0.0 or ry <= 0.0:
        return [Point(center.x + rx * math.cos(math.radians(start_angle)),
                      center.y + ry * math.sin(math.radians(start_angle)))]

    max_r = max(rx, ry)
    # Chord deviation: sagitta h = R * (1 - cos(delta_theta / 2)) <= tol
    if tolerance < max_r:
        cos_val = max(-1.0, min(1.0, 1.0 - (tolerance / max_r)))
        max_delta_deg = math.degrees(2.0 * math.acos(cos_val))
    else:
        max_delta_deg = 30.0

    max_delta_deg = max(2.0, min(max_delta_deg, 45.0))
    steps = max(6, int(math.ceil(abs(sweep_angle) / max_delta_deg)))

    points: list[Point] = []
    for i in range(steps + 1):
        t = i / steps
        ang_deg = start_angle + t * sweep_angle
        rad = math.radians(ang_deg)
        points.append(Point(center.x + rx * math.cos(rad), center.y + ry * math.sin(rad)))

    return points


# -----------------------------------------------------------------------------
# Bézier Analytical Extrema Bounds
# -----------------------------------------------------------------------------

def bezier_extrema_bounds(
    p0: Point, p1: Point, p2: Point, p3: Point | None = None
) -> tuple[float, float, float, float]:
    """Calculate exact analytical bounding box (min_x, min_y, width, height) of a Bézier curve.
    
    Evaluates endpoints and all roots of dx/dt = 0 and dy/dt = 0 within t in (0, 1).
    """
    x_candidates = [p0.x, (p3.x if p3 is not None else p2.x)]
    y_candidates = [p0.y, (p3.y if p3 is not None else p2.y)]

    if p3 is None:
        # Quadratic: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        # d/dt = 2(1-t)(P1 - P0) + 2t(P2 - P1) = 2(P0 - 2P1 + P2)t + 2(P1 - P0) = 0
        for axis, pts, cands in (("x", (p0.x, p1.x, p2.x), x_candidates),
                                 ("y", (p0.y, p1.y, p2.y), y_candidates)):
            a = pts[0] - 2.0 * pts[1] + pts[2]
            b = pts[1] - pts[0]
            if a != 0.0:
                t = -b / a
                if 0.0 < t < 1.0:
                    val = (1.0 - t) ** 2 * pts[0] + 2.0 * (1.0 - t) * t * pts[1] + t ** 2 * pts[2]
                    cands.append(val)
    else:
        # Cubic: B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2) = a t^2 + b t + c = 0
        for axis, pts, cands in (("x", (p0.x, p1.x, p2.x, p3.x), x_candidates),
                                 ("y", (p0.y, p1.y, p2.y, p3.y), y_candidates)):
            a = -pts[0] + 3.0 * pts[1] - 3.0 * pts[2] + pts[3]
            b = 2.0 * (pts[0] - 2.0 * pts[1] + pts[2])
            c = -pts[0] + pts[1]

            if a == 0.0:
                if b != 0.0:
                    t = -c / b
                    if 0.0 < t < 1.0:
                        val = ((1.0 - t) ** 3 * pts[0] + 3.0 * (1.0 - t) ** 2 * t * pts[1] +
                               3.0 * (1.0 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])
                        cands.append(val)
            else:
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_disc = math.sqrt(disc)
                    for sign in (-1.0, 1.0):
                        t = (-b + sign * sqrt_disc) / (2.0 * a)
                        if 0.0 < t < 1.0:
                            val = ((1.0 - t) ** 3 * pts[0] + 3.0 * (1.0 - t) ** 2 * t * pts[1] +
                                   3.0 * (1.0 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])
                            cands.append(val)

    min_x = min(x_candidates)
    max_x = max(x_candidates)
    min_y = min(y_candidates)
    max_y = max(y_candidates)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


# -----------------------------------------------------------------------------
# Arc Transformed Extrema Bounds
# -----------------------------------------------------------------------------

def is_angle_in_sweep(angle_rad: float, start_rad: float, sweep_rad: float) -> bool:
    """Check if an angle in radians lies within the signed sweep interval [start, start + sweep]."""
    if abs(sweep_rad) >= 2.0 * math.pi - 1e-9:
        return True
    
    two_pi = 2.0 * math.pi
    if sweep_rad >= 0.0:
        delta = (angle_rad - start_rad) % two_pi
        return delta <= sweep_rad + 1e-9
    else:
        delta_ccw = (start_rad - angle_rad) % two_pi
        return delta_ccw <= abs(sweep_rad) + 1e-9


def arc_transformed_extrema_bounds(
    center: Point,
    rx: float,
    ry: float,
    start_angle_deg: float,
    sweep_angle_deg: float,
    transform_matrix: np.ndarray,
    closure: ArcClosure = ArcClosure.OPEN
) -> tuple[float, float, float, float]:
    """Calculate exact analytical world-space bounding box (min_x, min_y, width, height)
    of a transformed elliptical arc under arbitrary affine transformation (rotation, non-uniform scaling).
    
    Derived from the parametric transformed ellipse equations:
        x'(t) = C_x + A_x cos(t) + B_x sin(t)
        y'(t) = C_y + A_y cos(t) + B_y sin(t)
    where critical extrema angles occur where dx'/dt = 0 and dy'/dt = 0:
        t_x = atan2(B_x, A_x) and t_x + pi
        t_y = atan2(B_y, A_y) and t_y + pi
    Critical angles are filtered against the signed sweep_angle.
    """
    a = float(transform_matrix[0, 0])
    c = float(transform_matrix[0, 1])
    tx = float(transform_matrix[0, 2])
    b = float(transform_matrix[1, 0])
    d = float(transform_matrix[1, 1])
    ty = float(transform_matrix[1, 2])

    Cx = a * center.x + c * center.y + tx
    Cy = b * center.x + d * center.y + ty
    Ax = a * rx
    Bx = c * ry
    Ay = b * rx
    By = d * ry

    def eval_pt(t: float) -> tuple[float, float]:
        return (Cx + Ax * math.cos(t) + Bx * math.sin(t),
                Cy + Ay * math.cos(t) + By * math.sin(t))

    start_rad = math.radians(start_angle_deg)
    sweep_rad = math.radians(sweep_angle_deg)
    end_rad = start_rad + sweep_rad

    # Endpoints are always candidate boundary points
    p_start = eval_pt(start_rad)
    p_end = eval_pt(end_rad)
    candidates: list[tuple[float, float]] = [p_start, p_end]

    # If closure is PIE, the center is a boundary vertex
    if closure == ArcClosure.PIE:
        candidates.append((Cx, Cy))

    # Analytical candidate extrema angles for x'(t)
    tx1 = math.atan2(Bx, Ax)
    tx2 = tx1 + math.pi
    for t_cand in (tx1, tx2):
        if is_angle_in_sweep(t_cand, start_rad, sweep_rad):
            candidates.append(eval_pt(t_cand))

    # Analytical candidate extrema angles for y'(t)
    ty1 = math.atan2(By, Ay)
    ty2 = ty1 + math.pi
    for t_cand in (ty1, ty2):
        if is_angle_in_sweep(t_cand, start_rad, sweep_rad):
            candidates.append(eval_pt(t_cand))

    min_x = min(pt[0] for pt in candidates)
    max_x = max(pt[0] for pt in candidates)
    min_y = min(pt[1] for pt in candidates)
    max_y = max(pt[1] for pt in candidates)

    return (min_x, min_y, max_x - min_x, max_y - min_y)


# -----------------------------------------------------------------------------
# Topological Fill Rule Mask Evaluator
# -----------------------------------------------------------------------------

def evaluate_fill_rule_mask(
    subpath_contours: list[list[Point]],
    fill_rule: FillRule,
    width: int,
    height: int,
    supersample: int = 2
) -> np.ndarray:
    """Generate a high-fidelity alpha coverage mask using EVEN_ODD or NON_ZERO rule.
    
    Architectural Contract:
      1. Keeps anti-aliasing and fill-rule topology strictly separate.
         The topological interior is evaluated on an integer grid at `supersample` x
         resolution to prevent XOR grayscale edge-intensity artifacts.
      2. Downsampling via cv2.INTER_AREA derives true geometric sub-pixel edge coverage.
      3. NON_ZERO fill assumes simple closed contours (using orientation sign: +1 for CW,
         -1 for CCW). Arbitrary self-intersecting subpaths are currently unsupported/undefined.
    """
    if not subpath_contours or width <= 0 or height <= 0:
        return np.zeros((height, width), dtype=np.uint8)

    scale = max(1, supersample)
    sw = width * scale
    sh = height * scale

    # Scale Point contours to supersampled coordinate space
    cv_contours: list[np.ndarray] = []
    for contour in subpath_contours:
        if len(contour) < 3:
            continue
        arr = np.array(
            [[int(round(p.x * scale)), int(round(p.y * scale))] for p in contour],
            dtype=np.int32
        ).reshape((-1, 1, 2))
        cv_contours.append(arr)

    if not cv_contours:
        return np.zeros((height, width), dtype=np.uint8)

    # Evaluate purely binary topological interior on supersampled grid
    if fill_rule == FillRule.EVEN_ODD:
        # Parity-based XOR accumulation across individual closed contours
        topological_mask = np.zeros((sh, sw), dtype=np.uint8)
        for c in cv_contours:
            sub_mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillPoly(sub_mask, [c], 255)
            topological_mask = cv2.bitwise_xor(topological_mask, sub_mask)
    else:  # FillRule.NON_ZERO
        # Signed winding accumulation (+1 for CW, -1 for CCW) across simple closed contours
        winding_accum = np.zeros((sh, sw), dtype=np.int32)
        for c in cv_contours:
            oriented_area = cv2.contourArea(c, oriented=True)
            w = 1 if oriented_area >= 0 else -1
            sub_mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillPoly(sub_mask, [c], 1)
            winding_accum += w * sub_mask.astype(np.int32)

        topological_mask = (winding_accum != 0).astype(np.uint8) * 255

    # Derive smooth anti-aliased edge coverage via area downsampling
    if scale > 1:
        return cv2.resize(topological_mask, (width, height), interpolation=cv2.INTER_AREA)
    return topological_mask

