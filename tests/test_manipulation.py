"""Unit tests for Phase 2 manipulation, bounds hierarchy, anchors, and deep cloning."""

import math
import pytest

from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_tripartite_bounds_hierarchy_rectangle():
    rect = Rectangle(
        position=Point(100, 100),
        width=200,
        height=100,
        stroke=StrokeStyle(width=10.0)
    )

    # 1. Geometry bounds (intrinsic local, excluding stroke)
    geom_b = rect.get_geometry_bounds()
    assert geom_b.left == 100.0
    assert geom_b.top == 100.0
    assert geom_b.width == 200.0
    assert geom_b.height == 100.0

    # 2. Local bounds (including stroke width expansion: half-stroke = 5.0)
    local_b = rect.get_local_bounds()
    assert local_b.left == 95.0
    assert local_b.top == 95.0
    assert local_b.width == 210.0
    assert local_b.height == 110.0

    # 3. World bounds under translation and scale
    rect.move(50, 50)
    rect.scale(2.0, 1.0)  # Width doubles to 400 around center (200, 150)
    world_b = rect.get_bounds()
    # Non-scaling stroke rule: stroke remains 10 px on screen (+5 on all sides)
    assert math.isclose(world_b.width, 410.0, abs_tol=1e-4)


def test_analytical_ellipse_world_bounds():
    # Circle of radius 50 at (100, 100) with stroke width 4
    # Scaled sx=2.0 (semi-axis a=100), sy=1.0 (semi-axis b=50), rotated 45°
    circle = Circle(
        center=Point(100, 100),
        radius=50,
        stroke=StrokeStyle(width=4.0)
    )
    circle.scale(2.0, 1.0)
    circle.rotate(45.0)

    bounds = circle.get_bounds()
    # Analytical extrema: sqrt((100*cos(45))^2 + (50*sin(45))^2)
    # = sqrt(5000 + 1250) = sqrt(6250) ≈ 79.0569
    # Semi-extrema horizontal and vertical are equal at 45°
    expected_ext = math.sqrt(6250)
    expected_w = 2 * expected_ext + 4.0  # + stroke width (4.0)
    assert math.isclose(bounds.width, expected_w, rel_tol=1e-4)
    assert math.isclose(bounds.height, expected_w, rel_tol=1e-4)


def test_shape_specific_anchors():
    # Line anchors
    line = Line(start=Point(0, 0), end=Point(100, 0))
    line.move(50, 50)
    assert line.anchor("start") == Point(50, 50)
    assert line.anchor("center") == Point(100, 50)
    assert line.anchor("end") == Point(150, 50)
    with pytest.raises(ValidationError):
        line.anchor("top_right")

    # Rectangle anchors
    rect = Rectangle(position=Point(0, 0), width=100, height=50)
    rect.rotate(90.0)  # Rotates 90° clockwise around center (50, 25)
    tr = rect.anchor("top_right")  # Local (100, 0)
    # (100, 0) relative to (50, 25) is (50, -25). Rotated 90° is (25, 50) -> (75, 75)
    assert math.isclose(tr.x, 75.0, abs_tol=1e-4)
    assert math.isclose(tr.y, 75.0, abs_tol=1e-4)

    # Circle circumference anchor at angle
    circle = Circle(center=Point(100, 100), radius=50)
    # Angle 0°: Point(150, 100)
    assert circle.anchor_at_angle(0) == Point(150, 100)
    # Angle 90°: Point(100, 150)
    p90 = circle.anchor_at_angle(90)
    assert math.isclose(p90.x, 100.0, abs_tol=1e-4)
    assert math.isclose(p90.y, 150.0, abs_tol=1e-4)


def test_deep_cloning():
    rect = Rectangle(
        position=Point(10, 20),
        width=100,
        height=50,
        stroke=StrokeStyle(color=Color.red(), width=3.0),
        fill=FillStyle(color=Color.blue())
    )
    rect.tags.add("box")
    rect.metadata["author"] = "alice"
    rect.move(10, 10)

    clone = rect.clone()
    assert clone.id != rect.id
    assert clone.position == rect.position
    assert clone.width == rect.width
    assert clone.height == rect.height
    assert clone.tags == rect.tags
    assert clone.metadata == rect.metadata
    assert clone.transform == rect.transform

    # Mutate clone, verify original is unaffected
    clone.tags.add("clone_box")
    clone.metadata["author"] = "bob"
    clone.stroke.color = Color.green()
    clone.move(20, 20)

    assert "clone_box" not in rect.tags
    assert rect.metadata["author"] == "alice"
    assert rect.stroke.color == Color.red()
    assert rect.transform.translation_x == 10.0
    assert clone.transform.translation_x == 30.0
