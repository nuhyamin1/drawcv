"""Unit tests for Phase 3 Advanced Geometry shapes."""

import math
import numpy as np
import pytest

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, ArrowHeadStyle, FillRule
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.path import Close, CubicTo, LineTo, MoveTo, Path, QuadraticTo, Subpath
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


# -----------------------------------------------------------------------------
# Ellipse Tests
# -----------------------------------------------------------------------------

def test_ellipse_properties_and_bounds():
    e = Ellipse(center=Point(100, 100), radius_x=50, radius_y=25, stroke=StrokeStyle(width=4.0))
    assert math.isclose(e.area, math.pi * 50 * 25)
    assert e.circumference > 0

    # Local bounds
    gb = e.get_geometry_bounds()
    assert gb == BoundingBox(50, 75, 100, 50)
    lb = e.get_local_bounds()
    assert lb == BoundingBox(48, 73, 104, 54)

    # World bounds under non-uniform scaling & non-scaling stroke
    e.scale(2.0, 1.0)
    wb = e.get_bounds()
    # Scaled radius_x = 100, radius_y = 25, center at 100, 100
    # geom bounds: 0, 75, 200, 50. Expanded by stroke.width/2 = 2.0
    assert math.isclose(wb.x, -2.0)
    assert math.isclose(wb.y, 73.0)
    assert math.isclose(wb.width, 204.0)
    assert math.isclose(wb.height, 54.0)


def test_ellipse_anchors_and_hit_testing():
    e = Ellipse(center=Point(100, 100), radius_x=50, radius_y=25)
    assert e.anchor("center") == Point(100, 100)
    assert e.anchor("top") == Point(100, 75)
    assert e.anchor("right") == Point(150, 100)

    # Parametric angle anchor
    pt_45 = e.anchor_at_angle(45)
    assert math.isclose(pt_45.x, 100 + 50 * math.cos(math.radians(45)))
    assert math.isclose(pt_45.y, 100 + 25 * math.sin(math.radians(45)))

    # Containment
    assert e.contains_point(Point(100, 100)) is True
    assert e.contains_point(Point(145, 100)) is True
    assert e.contains_point(Point(155, 100)) is False


# -----------------------------------------------------------------------------
# Polygon Tests
# -----------------------------------------------------------------------------

def test_polygon_properties_and_anchors():
    verts = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]
    poly = Polygon(vertices=verts, stroke=StrokeStyle(width=6.0))
    assert math.isclose(poly.area, 10000.0)

    c = poly.anchor("centroid")
    assert math.isclose(c.x, 50.0) and math.isclose(c.y, 50.0)
    assert poly.anchor("vertex_1") == Point(100, 0)

    # Bounds
    gb = poly.get_geometry_bounds()
    assert gb == BoundingBox(0, 0, 100, 100)
    wb = poly.get_bounds()
    assert wb == BoundingBox(-3, -3, 106, 106)

    # Containment
    assert poly.contains_point(Point(50, 50)) is True
    assert poly.contains_point(Point(150, 50)) is False


def test_polygon_validation_degenerate():
    # Collinear points
    with pytest.raises(ValidationError):
        Polygon(vertices=[Point(0, 0), Point(5, 5), Point(10, 10)])


# -----------------------------------------------------------------------------
# Polyline Tests
# -----------------------------------------------------------------------------

def test_polyline_metrics_and_anchors():
    pts = [Point(0, 0), Point(100, 0), Point(100, 50)]
    pl = Polyline(points=pts, closed=False, stroke=StrokeStyle(width=4.0))
    assert math.isclose(pl.length, 150.0)

    assert pl.anchor("start") == Point(0, 0)
    assert pl.anchor("end") == Point(100, 50)
    mid = pl.anchor("midpoint")
    assert math.isclose(mid.x, 75.0) and math.isclose(mid.y, 0.0)

    # World-space segment proximity hit-testing
    assert pl.contains_point(Point(50, 2)) is True
    assert pl.contains_point(Point(50, 20)) is False


# -----------------------------------------------------------------------------
# RoundedRectangle Tests
# -----------------------------------------------------------------------------

def test_rounded_rectangle_anchors_and_hit_testing():
    rr = RoundedRectangle(x=10, y=20, width=100, height=80, corner_radius=15)
    assert rr.anchor("top_left") == Point(10, 20)
    assert rr.anchor("center") == Point(60, 60)

    # Center is inside
    assert rr.contains_point(Point(60, 60)) is True
    # The sharp top-left corner point (11, 21) is cut off by corner radius 15
    # Center of TL arc is at (25, 35). Distance from (11, 21) is hypot(14, 14) ~= 19.8 > 15
    assert rr.contains_point(Point(11, 21)) is False
    # A point within the rounded arc
    assert rr.contains_point(Point(25, 25)) is True


# -----------------------------------------------------------------------------
# Arc Tests
# -----------------------------------------------------------------------------

def test_arc_closure_and_hit_testing():
    # 90-degree quadrant arc from 0 to 90
    arc_pie = Arc(
        center=Point(100, 100), radius_x=50, radius_y=50,
        start_angle=0.0, sweep_angle=90.0, closure=ArcClosure.PIE
    )
    # Point inside the 0..90 quadrant (e.g. at 45 deg, radius 25)
    p_in = Point(100 + 25 * math.cos(math.radians(45)), 100 + 25 * math.sin(math.radians(45)))
    assert arc_pie.contains_point(p_in) is True

    # Point outside the sweep (at 180 deg)
    p_out = Point(50, 100)
    assert arc_pie.contains_point(p_out) is False


def test_arc_negative_sweep():
    # Arc sweeping counter-clockwise from 90 to 0 (sweep = -90)
    arc = Arc(
        center=Point(0, 0), radius_x=40, radius_y=40,
        start_angle=90.0, sweep_angle=-90.0, closure=ArcClosure.PIE
    )
    # Point at 45 deg should be inside
    pt_45 = Point(20 * math.cos(math.radians(45)), 20 * math.sin(math.radians(45)))
    assert arc.contains_point(pt_45) is True


# -----------------------------------------------------------------------------
# Arrow Tests
# -----------------------------------------------------------------------------

def test_arrow_screen_space_head_invariance():
    arrow = Arrow(
        start=Point(0, 0), end=Point(100, 0),
        head_length=20.0, head_width=10.0, head_style=ArrowHeadStyle.TRIANGLE
    )

    # Scale the arrow non-uniformly: 5x along x, 0.2x along y from start point
    arrow.scale(5.0, 0.2, pivot=Point(0, 0))

    # Head geometry must remain in screen-space dimensions
    w_end, head_pts = arrow.get_world_head_geometry()
    # Tip is at (500, 0)
    assert math.isclose(w_end.x, 500.0)
    assert math.isclose(w_end.y, 0.0)

    # Base points should be 20 pixels behind along world x-axis (x = 480)
    # and +-5 pixels along world y-axis
    p_left, p_right = head_pts[1], head_pts[2]
    assert math.isclose(p_left.x, 480.0)
    assert math.isclose(abs(p_left.y), 5.0)
    assert math.isclose(p_right.x, 480.0)
    assert math.isclose(abs(p_right.y), 5.0)


# -----------------------------------------------------------------------------
# BezierCurve Tests
# -----------------------------------------------------------------------------

def test_bezier_curve_retention_and_scale():
    b = BezierCurve.quadratic(p0=Point(0, 0), p1=Point(50, 100), p2=Point(100, 0))
    assert b.is_quadratic is True
    assert b.is_cubic is False

    # Original control points are preserved
    assert b.p1 == Point(50, 100)

    # World flattening derives points dynamically
    w_pts = b.flatten_world(tolerance=0.5)
    assert len(w_pts) > 5

    # After scaling 10x from origin, original control points are still (50, 100)
    b.scale(10.0, pivot=Point(0, 0))

    assert b.p1 == Point(50, 100)
    # World points now span 1000px
    w_pts_scaled = b.flatten_world(tolerance=0.5)
    assert math.isclose(w_pts_scaled[-1].x, 1000.0)
    # Number of flattened segments naturally increases to maintain 0.5px tolerance
    assert len(w_pts_scaled) > len(w_pts)


# -----------------------------------------------------------------------------
# Path & Subpath Tests
# -----------------------------------------------------------------------------

def test_path_preserves_commands_and_subpaths():
    p = Path()
    p.move_to(0, 0).line_to(100, 0).quadratic_to(150, 50, 200, 0).close()

    assert len(p.subpaths) == 1
    sp = p.subpaths[0]
    assert sp.closed is True
    assert len(sp.commands) == 4
    assert isinstance(sp.commands[0], MoveTo)
    assert isinstance(sp.commands[1], LineTo)
    assert isinstance(sp.commands[2], QuadraticTo)
    assert isinstance(sp.commands[3], Close)

    # Second subpath
    p.move_to(300, 300).cubic_to(Point(350, 350), Point(400, 350), Point(450, 300))
    assert len(p.subpaths) == 2
    assert isinstance(p.subpaths[1].commands[1], CubicTo)

    # Retained commands remain untouched after transform
    p.scale(2.0)
    assert p.subpaths[0].commands[1].point == Point(100, 0)
