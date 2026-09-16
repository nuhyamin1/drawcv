"""Tests for paintable strokes and gradient strokes across DrawCV primitives."""

import numpy as np
import pytest

from drawcv import (
    Arc,
    ArcClosure,
    Arrow,
    BezierCurve,
    CapStyle,
    Circle,
    Color,
    GradientStop,
    Group,
    JoinStyle,
    Line,
    LinearGradient,
    OpenCVRenderer,
    Path,
    Point,
    Polygon,
    Polyline,
    RadialGradient,
    Rectangle,
    Scene,
    StrokeStyle,
    Transform,
    ValidationError,
)

STOPS = (
    GradientStop(0.0, Color.red()),
    GradientStop(1.0, Color.blue()),
)


def render_scene(drawable, width=200, height=200):
    scene = Scene(width, height, background=Color(0, 0, 0, 0))
    scene.add(drawable)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def test_stroke_style_api_and_validation():
    # 1. Backward compatible default and color
    s1 = StrokeStyle()
    assert s1.color == Color.black()
    assert s1.paint == Color.black()

    s2 = StrokeStyle(Color.red(), width=3.0)
    assert s2.color == Color.red()
    assert s2.paint == Color.red()

    # 2. Paint initialization
    grad = LinearGradient(Point(0, 0), Point(100, 0), STOPS)
    s3 = StrokeStyle(paint=grad, width=4.0)
    assert s3.paint is grad
    with pytest.raises(ValidationError):
        _ = s3.color  # non-solid has no single color

    # 3. Invalid combinations
    with pytest.raises(ValidationError):
        StrokeStyle(color=Color.red(), paint=grad)
    with pytest.raises(ValidationError):
        StrokeStyle(color=grad)  # passing gradient to color parameter
    with pytest.raises(ValidationError):
        StrokeStyle(paint="not_a_paint")

    # 4. In-place mutation
    s3.color = Color.green()
    assert s3.color == Color.green()
    assert s3.paint == Color.green()
    s3.paint = grad
    assert s3.paint is grad

    # 5. Copy independence
    s3_copy = s3.copy()
    assert s3_copy.paint == s3.paint and s3_copy.paint is not s3.paint


def test_stroke_style_serialization_roundtrip():
    grad = LinearGradient(Point(10, 20), Point(100, 20), STOPS, space="world")
    style = StrokeStyle(paint=grad, width=5.0, dash_array=(10, 5), dash_offset=2.0)
    data = style.to_dict()

    assert "paint" in data
    assert "color" not in data
    assert data["paint"]["type"] == "linear"
    assert data["width"] == 5.0
    assert data["dash_array"] == [10.0, 5.0]

    restored = StrokeStyle.from_dict(data)
    assert restored.width == 5.0
    assert restored.dash_array == (10.0, 5.0)
    assert isinstance(restored.paint, LinearGradient)
    assert restored.paint.start == Point(10, 20)


def test_gradient_stroke_line():
    # Horizontal line from (20, 50) to (120, 50)
    grad = LinearGradient(Point(20, 50), Point(120, 50), STOPS)
    line = Line(start=Point(20, 50), end=Point(120, 50), stroke=StrokeStyle(paint=grad, width=8))
    buf = render_scene(line)

    # Near start (30, 50) is red (BGR: 0, 0, 255)
    assert buf[50, 30, 3] > 0
    assert buf[50, 30, 2] > 200 and buf[50, 30, 0] < 50

    # Near middle (70, 50) is purple
    assert buf[50, 70, 3] > 0
    assert abs(int(buf[50, 70, 2]) - int(buf[50, 70, 0])) < 30

    # Near end (110, 50) is blue (BGR: 255, 0, 0)
    assert buf[50, 110, 3] > 0
    assert buf[50, 110, 0] > 200 and buf[50, 110, 2] < 50


def test_gradient_stroke_circle():
    # Radial gradient stroke around circle
    grad = RadialGradient(Point(100, 100), 50, STOPS)
    circle = Circle(center=Point(100, 100), radius=50, stroke=StrokeStyle(paint=grad, width=10))
    buf = render_scene(circle)

    # Interior (100, 100) is hollow (no fill)
    assert buf[100, 100, 3] == 0

    # Ring pixels at r=50 (e.g. 150, 100) are rendered
    assert buf[100, 150, 3] > 0
    assert buf[100, 50, 3] > 0


def test_gradient_stroke_rectangle():
    grad = LinearGradient(Point(10, 10), Point(110, 10), STOPS)
    rect = Rectangle(position=Point(10, 10), width=100, height=80, stroke=StrokeStyle(paint=grad, width=6))
    buf = render_scene(rect)

    # Left edge around (10, 50) is red
    assert buf[50, 10, 3] > 0
    assert buf[50, 10, 2] > 200

    # Right edge around (110, 50) is blue
    assert buf[50, 110, 3] > 0
    assert buf[50, 110, 0] > 200


def test_gradient_stroke_dashed():
    grad = LinearGradient(Point(0, 50), Point(150, 50), STOPS)
    style = StrokeStyle(paint=grad, width=8, dash_array=(20, 20), cap_style=CapStyle.BUTT)
    line = Line(start=Point(0, 50), end=Point(150, 50), stroke=style)
    buf = render_scene(line)

    # First dash (x=5 to 15) is drawn (red)
    assert buf[50, 10, 3] > 0
    assert buf[50, 10, 2] > 200

    # Gap (x=25 to 35) is empty
    assert buf[50, 30, 3] == 0

    # Second dash (x=45 to 55) is drawn
    assert buf[50, 50, 3] > 0


def test_gradient_stroke_nested_transform():
    grad = LinearGradient(Point(0, 0), Point(100, 0), STOPS, space="object")
    line = Line(start=Point(0, 0), end=Point(100, 0), stroke=StrokeStyle(paint=grad, width=6))
    line.transform = Transform(translation_x=20, translation_y=40)
    group = Group(children=[line], transform=Transform(rotation=45, pivot=Point(50, 50)))
    buf = render_scene(group, width=150, height=150)

    # Entity is rendered through nested transforms without error
    assert np.count_nonzero(buf[..., 3]) > 0
