"""Unit tests for Line, Rectangle, and Circle shape entities."""

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


def test_line_properties_and_metrics():
    p1 = Point(100, 100)
    p2 = Point(400, 500)
    line = Line(start=p1, end=p2, stroke=StrokeStyle(width=4.0))

    assert line.length == 500.0
    assert line.center == Point(250, 300)
    assert math.isclose(line.angle, math.degrees(math.atan2(400, 300)))

    # Bounds must include stroke padding (half stroke width on all sides)
    bounds = line.bounds
    assert bounds.left == 98.0  # 100 - 2
    assert bounds.right == 402.0  # 400 + 2
    assert bounds.top == 98.0
    assert bounds.bottom == 502.0


def test_line_movement():
    line = Line(start=Point(10, 20), end=Point(30, 40))
    line.move(5, -10)
    # Intrinsic geometry remains intact in local space
    assert line.start == Point(10, 20)
    assert line.end == Point(30, 40)
    # World points reflect the translation
    assert line.to_world(line.start) == Point(15, 10)
    assert line.to_world(line.end) == Point(35, 30)


def test_rectangle_properties():
    rect = Rectangle(
        position=Point(50, 100),
        width=200,
        height=100,
        stroke=StrokeStyle(width=6.0),
        fill=FillStyle(color=Color.blue())
    )

    assert rect.area == 20000.0
    assert rect.center == Point(150, 150)
    tl, tr, br, bl = rect.corners
    assert tl == Point(50, 100)
    assert tr == Point(250, 100)
    assert br == Point(250, 200)
    assert bl == Point(50, 200)

    # Stroke-aware bounds
    bounds = rect.bounds
    assert bounds.left == 47.0  # 50 - 3
    assert bounds.top == 97.0   # 100 - 3
    assert bounds.width == 206.0
    assert bounds.height == 106.0


def test_rectangle_movement_and_mutation():
    rect = Rectangle(position=Point(10, 10), width=50, height=50)
    rect.move(20, 30)
    # Intrinsic local position stays intact
    assert rect.position == Point(10, 10)
    assert rect.to_world(rect.position) == Point(30, 40)

    # Deliberate intrinsic editing updates local geometry
    rect.position = Point(25, 25)
    rect.width = 100
    rect.height = 80
    assert rect.area == 8000.0
    assert rect.to_world(rect.position) == Point(45, 55)


def test_rectangle_validation():
    with pytest.raises(ValidationError):
        Rectangle(position=Point(0, 0), width=-10, height=20)


def test_circle_properties():
    circle = Circle(
        center=Point(200, 200),
        radius=50,
        stroke=StrokeStyle(width=4.0),
        fill=FillStyle(color=Color.red())
    )

    assert circle.diameter == 100.0
    assert math.isclose(circle.area, math.pi * 2500)
    assert math.isclose(circle.circumference, 2 * math.pi * 50)

    # Stroke-aware bounds
    bounds = circle.bounds
    assert bounds.left == 148.0   # 200 - 50 - 2
    assert bounds.top == 148.0
    assert bounds.width == 104.0
    assert bounds.height == 104.0


def test_circle_movement_and_mutation():
    circle = Circle(center=Point(100, 100), radius=30)
    circle.move(-20, 50)
    # Intrinsic local center stays intact
    assert circle.center == Point(100, 100)
    assert circle.to_world(circle.center) == Point(80, 150)

    # Deliberate intrinsic editing
    circle.center = Point(200, 200)
    circle.radius = 60
    assert circle.diameter == 120.0
    assert circle.to_world(circle.center) == Point(180, 250)


def test_circle_validation():
    with pytest.raises(ValidationError):
        Circle(center=Point(0, 0), radius=-5)


def test_anchors():
    rect = Rectangle(position=Point(100, 100), width=200, height=100)
    # Stroke is None, bounds are exact
    assert rect.anchor("center") == Point(200, 150)
    assert rect.anchor("top") == Point(200, 100)
    assert rect.anchor("bottom") == Point(200, 200)
    assert rect.anchor("left") == Point(100, 150)
    assert rect.anchor("right") == Point(300, 150)
    assert rect.anchor("top_left") == Point(100, 100)
    assert rect.anchor("bottom_right") == Point(300, 200)

    with pytest.raises(ValidationError):
        rect.anchor("invalid_anchor_name")


def test_shared_mutable_defaults_isolation():
    c1 = Circle(center=Point(0, 0), radius=10)
    c2 = Circle(center=Point(10, 10), radius=20)

    # Modify tags and metadata on c1
    c1.tags.add("tag1")
    c1.metadata["key"] = "val"

    assert "tag1" not in c2.tags
    assert "key" not in c2.metadata
    assert c1.transform is not c2.transform
