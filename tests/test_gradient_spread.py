"""Tests for linear and radial gradient spread modes: pad, repeat, reflect."""

import numpy as np
import pytest

from drawcv import (
    Color,
    GradientStop,
    LinearGradient,
    OpenCVRenderer,
    Point,
    RadialGradient,
    Rectangle,
    Scene,
    FillStyle,
    Transform,
    ValidationError,
)

STOPS = (
    GradientStop(0.0, Color.red()),
    GradientStop(1.0, Color.blue()),
)


def render_rect(fill_paint, width=200, height=100):
    rect = Rectangle(position=Point(0, 0), width=width, height=height, fill=FillStyle(paint=fill_paint))
    scene = Scene(width, height, background=Color(0, 0, 0, 0))
    scene.add(rect)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def test_linear_spread_pad_default():
    # Gradient from x=50 to x=100
    paint = LinearGradient(Point(50, 0), Point(100, 0), STOPS, spread="pad")
    buf = render_rect(paint)
    # Before start (x < 50): clamped to red (BGR: 0, 0, 255)
    assert tuple(buf[50, 10]) == (0, 0, 255, 255)
    assert tuple(buf[50, 50]) == (0, 0, 255, 255)
    # Midpoint (x = 75): 50% blend of red and blue (BGR: 128, 0, 128)
    assert tuple(buf[50, 75]) == (128, 0, 128, 255)
    # At and past end (x >= 100): clamped to blue (BGR: 255, 0, 0)
    assert tuple(buf[50, 100]) == (255, 0, 0, 255)
    assert tuple(buf[50, 150]) == (255, 0, 0, 255)


def test_linear_spread_repeat():
    # Gradient span is 50px (x=0 to x=50)
    paint = LinearGradient(Point(0, 0), Point(50, 0), STOPS, spread="repeat")
    buf = render_rect(paint)
    # Cycle 0 (0 to 50): starts red, mid purple, ends blue
    assert tuple(buf[50, 0]) == (0, 0, 255, 255)
    assert tuple(buf[50, 25]) == (128, 0, 128, 255)
    # Cycle 1 (50 to 100): restarts red at 50, mid purple at 75
    assert tuple(buf[50, 50]) == (0, 0, 255, 255)
    assert tuple(buf[50, 75]) == (128, 0, 128, 255)
    # Cycle 2 (100 to 150): restarts red at 100
    assert tuple(buf[50, 100]) == (0, 0, 255, 255)
    assert tuple(buf[50, 125]) == (128, 0, 128, 255)


def test_linear_spread_reflect():
    # Gradient span is 50px (x=0 to x=50)
    paint = LinearGradient(Point(0, 0), Point(50, 0), STOPS, spread="reflect")
    buf = render_rect(paint)
    # Cycle 0 (0 to 50): red at 0 -> blue at 50
    assert tuple(buf[50, 0]) == (0, 0, 255, 255)
    assert tuple(buf[50, 25]) == (128, 0, 128, 255)
    assert tuple(buf[50, 50]) == (255, 0, 0, 255)
    # Cycle 1 (50 to 100): reflects! blue at 50 -> red at 100
    assert tuple(buf[50, 75]) == (128, 0, 128, 255)
    assert tuple(buf[50, 100]) == (0, 0, 255, 255)
    # Cycle 2 (100 to 150): red at 100 -> blue at 150
    assert tuple(buf[50, 125]) == (128, 0, 128, 255)
    assert tuple(buf[50, 150]) == (255, 0, 0, 255)


def test_radial_spread_repeat_and_reflect():
    # Center at (50, 50), radius 20
    paint_repeat = RadialGradient(Point(50, 50), 20, STOPS, spread="repeat")
    buf_rep = render_rect(paint_repeat, width=100, height=100)
    # Center: r=0 -> red
    assert tuple(buf_rep[50, 50]) == (0, 0, 255, 255)
    # r=10 -> purple
    assert tuple(buf_rep[50, 60]) == (128, 0, 128, 255)
    # r=20 -> restart at red
    assert tuple(buf_rep[50, 70]) == (0, 0, 255, 255)
    # r=30 -> purple
    assert tuple(buf_rep[50, 80]) == (128, 0, 128, 255)

    paint_reflect = RadialGradient(Point(50, 50), 20, STOPS, spread="reflect")
    buf_ref = render_rect(paint_reflect, width=100, height=100)
    # Center: r=0 -> red
    assert tuple(buf_ref[50, 50]) == (0, 0, 255, 255)
    # r=20 -> blue
    assert tuple(buf_ref[50, 70]) == (255, 0, 0, 255)
    # r=30 -> reflects back towards red: purple
    assert tuple(buf_ref[50, 80]) == (128, 0, 128, 255)
    # r=40 -> red
    assert tuple(buf_ref[50, 90]) == (0, 0, 255, 255)


def test_invalid_spread_mode_rejected():
    with pytest.raises(ValidationError):
        LinearGradient(Point(0, 0), Point(10, 0), STOPS, spread="invalid_spread")
    with pytest.raises(ValidationError):
        RadialGradient(Point(0, 0), 10, STOPS, spread="invalid_spread")
