"""Unit tests for geometry-selection hit-testing and Scene spatial queries."""

from drawcv.core.color import Color
from drawcv.core.geometry import Point
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle


def test_hit_test_rotated_rectangle():
    # 100x100 square centered at (100, 100) rotated 45°
    rect = Rectangle(
        position=Point(50, 50),
        width=100,
        height=100,
        fill=None  # Even with fill=None, geometry-selection selects interior!
    )
    rect.rotate(45.0)

    # Center is at (100, 100) -> inside
    assert rect.contains_point(Point(100, 100))

    # Corner of the original axis-aligned box (52, 52) is OUTSIDE the 45° rotated diamond
    assert not rect.contains_point(Point(52, 52))

    # Diamond tip along x-axis is at 100 + 50*sqrt(2) ≈ 170.7
    # Point at (165, 100) is inside diamond
    assert rect.contains_point(Point(165, 100))
    # Point at (175, 100) is outside diamond
    assert not rect.contains_point(Point(175, 100))


def test_hit_test_transformed_circle():
    # Circle radius 50 at (100, 100) scaled sx=2.0 (horizontal width 200, vertical height 100)
    circle = Circle(center=Point(100, 100), radius=50)
    circle.scale(2.0, 1.0)

    # Center (100, 100) is inside
    assert circle.contains_point(Point(100, 100))

    # Point at (180, 100) is inside horizontally stretched ellipse (distance from center = 80 < 100)
    assert circle.contains_point(Point(180, 100))

    # Point at (100, 160) is outside vertical extent (distance from center = 60 > 50)
    assert not circle.contains_point(Point(100, 160))


def test_hit_test_line():
    line = Line(start=Point(0, 0), end=Point(100, 100))
    line.move(50, 50)

    # Point on line (100, 100) -> inside
    assert line.contains_point(Point(100, 100))
    # Point near line (101, 100) -> inside tolerance
    assert line.contains_point(Point(101, 100))
    # Point far from line (100, 150) -> outside
    assert not line.contains_point(Point(100, 150))


def test_hit_test_scaled_line_constant_screen_tolerance():
    from drawcv.styles.stroke import StrokeStyle
    # Line along horizontal axis from (0, 100) to (100, 100) with stroke width 10 (screen tolerance = 5.0)
    line = Line(start=Point(0, 100), end=Point(100, 100), stroke=StrokeStyle(width=10.0))
    # Scale 4x horizontally and vertically
    line.scale(4.0)

    # Transformed line is from (-150, 100) to (250, 100)
    # 4 screen pixels above line (y=104): inside 5px screen tolerance
    assert line.contains_point(Point(50, 104))

    # 8 screen pixels above line (y=108): OUTSIDE 5px screen tolerance
    # (If calculated in local space, 8 / 4 = 2 <= 5 would have erroneously returned True!)
    assert not line.contains_point(Point(50, 108))


def test_scene_hit_test_z_ordering():
    scene = Scene(300, 300)

    # Object 1: Low z-index (0)
    r1 = Rectangle(position=Point(50, 50), width=100, height=100, z_index=0, name="bottom")
    # Object 2: High z-index (5)
    r2 = Rectangle(position=Point(80, 80), width=100, height=100, z_index=5, name="top")

    scene.add(r1)
    scene.add(r2)

    # Point (90, 90) overlaps both rectangles
    hits = scene.hit_test(90, 90)
    assert len(hits) == 2
    # Topmost first
    assert hits[0] is r2
    assert hits[1] is r1

    assert scene.hit_test_top(90, 90) is r2

    # Point (60, 60) hits only r1
    assert scene.hit_test_top(60, 60) is r1

    # Point (10, 10) hits nothing
    assert scene.hit_test_top(10, 10) is None


def test_scene_hit_test_ignores_invisible():
    scene = Scene(200, 200)
    rect = Rectangle(position=Point(20, 20), width=50, height=50, visible=False)
    scene.add(rect)

    assert scene.hit_test(30, 30) == []
    assert scene.hit_test_top(30, 30) is None
