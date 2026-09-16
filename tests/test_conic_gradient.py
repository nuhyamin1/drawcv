"""Tests for ConicGradient construction, validation, sampling, and transformations."""

import numpy as np
import pytest

from drawcv import (
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    Transform,
    ValidationError,
)

STOPS_4 = (
    GradientStop(0.0, Color.red()),      # 0 deg (right)
    GradientStop(0.25, Color.green()),   # 90 deg (down)
    GradientStop(0.5, Color.blue()),     # 180 deg (left)
    GradientStop(0.75, Color.yellow()),  # 270 deg (up)
    GradientStop(1.0, Color.red()),      # 360 deg
)


def render_conic(conic_paint, width=100, height=100, move_entity=(0, 0)):
    rect = Rectangle(position=Point(0, 0), width=width, height=height, fill=FillStyle(paint=conic_paint))
    if move_entity != (0, 0):
        rect.move(*move_entity)
    scene = Scene(width + abs(move_entity[0]), height + abs(move_entity[1]), background=Color(0, 0, 0, 0))
    scene.add(rect)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def test_conic_quadrant_sampling():
    # Center at (50, 50), start_angle = 0
    paint = ConicGradient(Point(50, 50), STOPS_4)
    buf = render_conic(paint)

    # Center singular point should be valid color (stop 0: red)
    assert tuple(buf[50, 50]) == (0, 0, 255, 255)

    # 0 deg: (x=70, y=50) -> East -> Red (BGR: 0, 0, 255)
    assert tuple(buf[50, 70]) == (0, 0, 255, 255)

    # 90 deg: (x=50, y=70) -> South -> Green (BGR: 0, 255, 0)
    assert tuple(buf[70, 50]) == (0, 255, 0, 255)

    # 180 deg: (x=30, y=50) -> West -> Blue (BGR: 255, 0, 0)
    assert tuple(buf[50, 30]) == (255, 0, 0, 255)

    # 270 deg: (x=50, y=30) -> North -> Yellow (BGR: 0, 255, 255)
    assert tuple(buf[30, 50]) == (0, 255, 255, 255)


def test_conic_start_angle():
    # Rotating start_angle by +90 degrees moves stop 0 (red) to South (90 deg)
    paint = ConicGradient(Point(50, 50), STOPS_4, start_angle=90.0)
    buf = render_conic(paint)

    # East (0 deg in world) is now (0 - 90) % 360 = 270 deg -> Yellow
    assert tuple(buf[50, 70]) == (0, 255, 255, 255)
    # South (90 deg in world) is now (90 - 90) = 0 deg -> Red
    assert tuple(buf[70, 50]) == (0, 0, 255, 255)


def test_conic_paint_transform():
    # Rotate paint around center (50, 50) by 90 degrees via paint transform
    paint = ConicGradient(Point(50, 50), STOPS_4, transform=Transform(rotation=90, pivot=Point(50, 50)))
    buf = render_conic(paint)

    # Paint rotation of +90 deg rotates visual output clockwise by 90 deg:
    # South (90 deg) now displays what was previously at 0 deg (Red)
    assert tuple(buf[70, 50]) == (0, 0, 255, 255)


def test_conic_object_vs_world_space():
    paint_obj = ConicGradient(Point(50, 50), STOPS_4, space="object")
    paint_world = ConicGradient(Point(50, 50), STOPS_4, space="world")

    # Move rectangle by (20, 0)
    buf_obj = render_conic(paint_obj, move_entity=(20, 0))
    buf_world = render_conic(paint_world, move_entity=(20, 0))

    # In object space, the center moves with the rectangle (now at 70, 50)
    assert tuple(buf_obj[50, 70]) == (0, 0, 255, 255)
    # East of local center (90, 50) is red
    assert tuple(buf_obj[50, 90]) == (0, 0, 255, 255)

    # In world space, center remains at world (50, 50)
    # Position (50, 50) inside the moved rect (x=20 to 120) is the center
    assert tuple(buf_world[50, 50]) == (0, 0, 255, 255)


def test_conic_validation():
    with pytest.raises(ValidationError):
        ConicGradient("not_a_point", STOPS_4)
    with pytest.raises(ValidationError):
        ConicGradient(Point(0, 0), STOPS_4, start_angle="bad_angle")
    with pytest.raises(ValidationError):
        ConicGradient(Point(0, 0), (GradientStop(0.0, Color.red()),))  # < 2 stops
    with pytest.raises(ValidationError):
        ConicGradient(Point(0, 0), STOPS_4, space="invalid_space")
    with pytest.raises(ValidationError):
        ConicGradient(Point(0, 0), STOPS_4, transform="not_a_transform")


def test_conic_serialization_roundtrip():
    paint = ConicGradient(Point(40, 60), STOPS_4, start_angle=45.0, space="world", transform=Transform(rotation=15))
    data = paint.to_dict()
    assert data["type"] == "conic"
    assert data["center"] == {"x": 40.0, "y": 60.0}
    assert data["start_angle"] == 45.0
    assert data["space"] == "world"

    restored = ConicGradient.from_dict(data)
    assert restored.center == paint.center
    assert restored.start_angle == paint.start_angle
    assert restored.space == paint.space
    assert restored.transform.rotation == 15.0
    assert len(restored.stops) == len(paint.stops)
