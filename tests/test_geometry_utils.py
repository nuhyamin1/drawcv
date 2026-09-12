"""Unit tests for computational geometry utilities and analytical extrema."""

import math
import numpy as np
import pytest

from drawcv.core.enums import ArcClosure
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    arc_transformed_extrema_bounds,
    bezier_extrema_bounds,
    distance_point_to_segment,
    flatten_arc,
    flatten_cubic_bezier,
    flatten_quadratic_bezier,
    is_angle_in_sweep,
    point_at_polyline_length,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    polyline_length,
)


def test_distance_point_to_segment():
    s = Point(0.0, 0.0)
    e = Point(10.0, 0.0)

    # Perpendicular projection onto segment
    p1 = Point(5.0, 3.0)
    assert math.isclose(distance_point_to_segment(p1, s, e), 3.0)

    # Point past the end
    p2 = Point(14.0, 3.0)
    assert math.isclose(distance_point_to_segment(p2, s, e), 5.0)

    # Point before the start
    p3 = Point(-3.0, 4.0)
    assert math.isclose(distance_point_to_segment(p3, s, e), 5.0)

    # Degenerate segment (point)
    assert math.isclose(distance_point_to_segment(Point(3.0, 4.0), s, s), 5.0)


def test_polygon_area_and_centroid():
    # 10x10 square clockwise
    cw_square = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    assert math.isclose(polygon_area(cw_square), 100.0)
    c = polygon_centroid(cw_square)
    assert math.isclose(c.x, 5.0) and math.isclose(c.y, 5.0)

    # Counter-clockwise square (signed area is negative)
    ccw_square = list(reversed(cw_square))
    assert math.isclose(polygon_area(ccw_square), -100.0)
    c_ccw = polygon_centroid(ccw_square)
    assert math.isclose(c_ccw.x, 5.0) and math.isclose(c_ccw.y, 5.0)

    # Triangle
    tri = [Point(0, 0), Point(6, 0), Point(0, 6)]
    assert math.isclose(polygon_area(tri), 18.0)
    c_tri = polygon_centroid(tri)
    assert math.isclose(c_tri.x, 2.0) and math.isclose(c_tri.y, 2.0)


def test_point_in_polygon():
    verts = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    assert point_in_polygon(Point(5, 5), verts) is True
    assert point_in_polygon(Point(15, 5), verts) is False
    # Boundary points
    assert point_in_polygon(Point(0, 5), verts, include_boundary=True) is True
    assert point_in_polygon(Point(10, 10), verts, include_boundary=True) is True


def test_polyline_metrics():
    pts = [Point(0, 0), Point(3, 4), Point(3, 9)]
    # First segment = 5.0, second segment = 5.0, total = 10.0
    assert math.isclose(polyline_length(pts), 10.0)

    # Midpoint
    mid = point_at_polyline_length(pts, 5.0)
    assert math.isclose(mid.x, 3.0) and math.isclose(mid.y, 4.0)

    # Quarter point
    q1 = point_at_polyline_length(pts, 2.5)
    assert math.isclose(q1.x, 1.5) and math.isclose(q1.y, 2.0)

    # Beyond length clamps to last point
    end = point_at_polyline_length(pts, 20.0)
    assert end == pts[-1]


def test_adaptive_bezier_flattening():
    # Straight quadratic curve (degenerate into line)
    p0 = Point(0, 0)
    p1 = Point(5, 0)
    p2 = Point(10, 0)
    flat_quad = flatten_quadratic_bezier(p0, p1, p2, tolerance=0.5)
    assert len(flat_quad) == 2  # No subdivision needed for straight line

    # Curved quadratic
    p1_curved = Point(5, 50)
    flat_curved = flatten_quadratic_bezier(p0, p1_curved, p2, tolerance=0.5)
    assert len(flat_curved) > 4
    # All points must be continuous
    assert flat_curved[0] == p0
    assert flat_curved[-1] == p2

    # Cubic curve
    c0 = Point(0, 0)
    c1 = Point(10, 40)
    c2 = Point(40, 40)
    c3 = Point(50, 0)
    flat_cubic = flatten_cubic_bezier(c0, c1, c2, c3, tolerance=0.5)
    assert len(flat_cubic) > 5
    assert flat_cubic[0] == c0
    assert flat_cubic[-1] == c3


def test_bezier_extrema_bounds():
    # Symmetric arch quadratic: P0=(0,0), P1=(50, 100), P2=(100, 0)
    # Peak at t=0.5, y=50
    p0 = Point(0, 0)
    p1 = Point(50, 100)
    p2 = Point(100, 0)
    x, y, w, h = bezier_extrema_bounds(p0, p1, p2)
    assert math.isclose(x, 0.0)
    assert math.isclose(w, 100.0)
    assert math.isclose(y, 0.0)
    assert math.isclose(h, 50.0)

    # S-curve cubic: control points exceed bounding box
    c0 = Point(0, 0)
    c1 = Point(0, 100)
    c2 = Point(100, -100)
    c3 = Point(100, 0)
    x, y, w, h = bezier_extrema_bounds(c0, c1, c2, c3)
    assert math.isclose(x, 0.0)
    assert math.isclose(w, 100.0)
    # Exact curve extrema will not reach 100 or -100, but will extend beyond endpoints
    assert y < 0.0
    assert y + h > 0.0


def test_arc_transformed_extrema_bounds_analytical_precision():
    center = Point(100, 100)
    rx, ry = 80.0, 40.0
    start_angle = 30.0
    sweep_angle = 120.0  # from 30 deg to 150 deg (covers 90 deg where sin=1, max y)

    # Identity transform
    M_id = np.eye(3, dtype=np.float64)
    x, y, w, h = arc_transformed_extrema_bounds(
        center, rx, ry, start_angle, sweep_angle, M_id, ArcClosure.OPEN
    )

    # Top extreme occurs at 90 deg: y = 100 + 40 = 140
    assert math.isclose(y + h, 140.0, abs_tol=1e-5)

    # Rotated & non-uniformly scaled transform
    theta = math.radians(45.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    M_rot = np.array([
        [2.0 * cos_t, -2.0 * sin_t, 50.0],
        [1.5 * sin_t, 1.5 * cos_t, 30.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    bx, by, bw, bh = arc_transformed_extrema_bounds(
        center, rx, ry, 0.0, 360.0, M_rot, ArcClosure.OPEN
    )

    # Verify against full ellipse extrema formula
    a = float(M_rot[0, 0])
    c = float(M_rot[0, 1])
    tx = float(M_rot[0, 2])
    b = float(M_rot[1, 0])
    d = float(M_rot[1, 1])
    ty = float(M_rot[1, 2])

    Cx = a * center.x + c * center.y + tx
    Cy = b * center.x + d * center.y + ty
    Ax = a * rx
    Bx = c * ry
    Ay = b * rx
    By = d * ry

    expected_w = 2.0 * math.sqrt(Ax * Ax + Bx * Bx)
    expected_h = 2.0 * math.sqrt(Ay * Ay + By * By)

    assert math.isclose(bw, expected_w, abs_tol=1e-5)
    assert math.isclose(bh, expected_h, abs_tol=1e-5)
