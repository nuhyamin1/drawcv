"""Tests for animating paint properties via AnimationTrack."""

import numpy as np
import pytest

from drawcv import (
    AnimationTrack,
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    ImageInterpolation,
    ImagePaint,
    LinearGradient,
    Line,
    OpenCVRenderer,
    Point,
    RadialGradient,
    Rectangle,
    Scene,
    StrokeStyle,
    Timing,
    Transform,
)


def test_animate_linear_gradient_endpoints():
    grad = LinearGradient(
        start=Point(0, 0),
        end=Point(100, 0),
        stops=(GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=grad))

    # Track to animate start point from (0, 0) to (50, 0)
    track = AnimationTrack(
        target_id=rect.id,
        property_path="fill.paint.start",
        start_value=Point(0, 0),
        end_value=Point(50, 0),
        timing=Timing(duration=1.0),
    )

    # At t=0.5, start should be at (25, 0)
    val = track.evaluate(0.5, target=rect)
    assert val == Point(25, 0)
    assert rect.fill.paint.start == Point(25, 0)


def test_animate_conic_gradient_start_angle():
    grad = ConicGradient(
        center=Point(50, 50),
        stops=(GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
        start_angle=0.0,
    )
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=grad))

    track = AnimationTrack(
        target_id=rect.id,
        property_path="fill.paint.start_angle",
        start_value=0.0,
        end_value=360.0,
        timing=Timing(duration=2.0),
    )

    # At t=1.0 (midpoint of duration=2.0), start_angle should be 180.0
    val = track.evaluate(1.0, target=rect)
    assert pytest.approx(val, 1e-4) == 180.0
    assert pytest.approx(rect.fill.paint.start_angle, 1e-4) == 180.0


def test_animate_stroke_paint_center():
    grad = RadialGradient(
        center=Point(20, 20),
        radius=30.0,
        stops=(GradientStop(0.0, Color.red()), GradientStop(1.0, Color.green())),
    )
    line = Line(start=Point(0, 0), end=Point(100, 100), stroke=StrokeStyle(paint=grad, width=5.0))

    track = AnimationTrack(
        target_id=line.id,
        property_path="stroke.paint.center",
        start_value=Point(20, 20),
        end_value=Point(80, 80),
        timing=Timing(duration=1.0),
    )

    val = track.evaluate(0.5, target=line)
    assert val == Point(50, 50)
    assert line.stroke.paint.center == Point(50, 50)


def test_animate_image_paint_transform():
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    paint = ImagePaint(image=img, transform=Transform(translation_x=0.0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=paint))

    track = AnimationTrack(
        target_id=rect.id,
        property_path="fill.paint.transform",
        start_value=Transform(translation_x=0.0),
        end_value=Transform(translation_x=100.0),
        timing=Timing(duration=1.0),
    )

    val = track.evaluate(0.5, target=rect)
    assert pytest.approx(val.translation_x) == 50.0
    assert pytest.approx(rect.fill.paint.transform.translation_x) == 50.0
