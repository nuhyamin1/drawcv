"""Tests for interpolation, easing functions, and Timing model."""

import math
import pytest

from drawcv.animation.easing import (
    EASING_FUNCTIONS,
    get_easing,
    linear,
    ease_in_elastic,
    ease_out_bounce,
)
from drawcv.animation.interpolation import (
    lerp,
    lerp_bounds,
    lerp_color,
    lerp_point,
    lerp_transform,
    resolve_interpolator,
)
from drawcv.animation.timing import Timing
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.exceptions import SerializationError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform


def test_scalar_lerp():
    assert lerp(10, 20, 0.0) == 10.0
    assert lerp(10, 20, 1.0) == 20.0
    assert lerp(10, 20, 0.5) == 15.0
    assert lerp(10, 20, 1.5) == 25.0


def test_lerp_point():
    p1 = Point(0, 10)
    p2 = Point(100, 50)
    mid = lerp_point(p1, p2, 0.5)
    assert mid.x == 50.0
    assert mid.y == 30.0


def test_lerp_color():
    c1 = Color(0, 100, 200, 0.2)
    c2 = Color(100, 200, 50, 0.8)
    mid = lerp_color(c1, c2, 0.5)
    assert mid.r == 50
    assert mid.g == 150
    assert mid.b == 125
    assert pytest.approx(mid.a) == 0.5
    assert isinstance(mid.r, int)
    assert isinstance(mid.g, int)
    assert isinstance(mid.b, int)
    assert isinstance(mid.a, float)


def test_lerp_transform():
    t1 = Transform(translation_x=0, translation_y=0, rotation=0, scale_x=1.0, scale_y=1.0)
    t2 = Transform(translation_x=100, translation_y=50, rotation=90, scale_x=2.0, scale_y=4.0)
    mid = lerp_transform(t1, t2, 0.5)
    assert mid.translation_x == 50.0
    assert mid.translation_y == 25.0
    assert mid.rotation == 45.0
    assert mid.scale_x == 1.5
    assert mid.scale_y == 2.5


def test_lerp_bounds():
    b1 = BoundingBox(0, 0, 100, 100)
    b2 = BoundingBox(50, 50, 200, 300)
    mid = lerp_bounds(b1, b2, 0.5)
    assert mid.x == 25.0
    assert mid.y == 25.0
    assert mid.width == 150.0
    assert mid.height == 200.0


def test_resolve_interpolator():
    assert resolve_interpolator(10, 20) == lerp
    assert resolve_interpolator(Point(0, 0), Point(1, 1)) == lerp_point
    assert resolve_interpolator(Color(0, 0, 0), Color(255, 255, 255)) == lerp_color
    assert resolve_interpolator(Transform(), Transform()) == lerp_transform
    assert resolve_interpolator(BoundingBox(0, 0, 1, 1), BoundingBox(0, 0, 2, 2)) == lerp_bounds

    with pytest.raises(ValidationError):
        resolve_interpolator("hello", 123)


def test_all_easings_endpoints_and_finite():
    for name, fn in EASING_FUNCTIONS.items():
        assert pytest.approx(fn(0.0), abs=1e-4) == 0.0, f"Easing '{name}' failed at t=0"
        assert pytest.approx(fn(1.0), abs=1e-4) == 1.0, f"Easing '{name}' failed at t=1"

        # Check finite across 20 intermediate steps
        for i in range(1, 20):
            t = i / 20.0
            val = fn(t)
            assert not math.isnan(val) and not math.isinf(val), f"Easing '{name}' produced non-finite value at t={t}"


def test_monotonic_easings():
    monotonic_names = [
        "linear",
        "ease_in_quad", "ease_out_quad", "ease_in_out_quad",
        "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
        "ease_in_sine", "ease_out_sine", "ease_in_out_sine",
        "ease_in_expo", "ease_out_expo", "ease_in_out_expo",
        "ease_in_circ", "ease_out_circ", "ease_in_out_circ",
    ]
    for name in monotonic_names:
        fn = EASING_FUNCTIONS[name]
        prev = -1e-7
        for i in range(21):
            t = i / 20.0
            val = fn(t)
            assert 0.0 - 1e-6 <= val <= 1.0 + 1e-6, f"Easing '{name}' exceeded [0, 1] at t={t}: {val}"
            assert val >= prev - 1e-6, f"Easing '{name}' was not monotonic at t={t}"
            prev = val


def test_oscillatory_easings_sample_values():
    # Elastic and bounce can overshoot or bounce; verify they produce expected behavior
    elastic_in = EASING_FUNCTIONS["ease_in_elastic"]
    elastic_out = EASING_FUNCTIONS["ease_out_elastic"]
    bounce_out = EASING_FUNCTIONS["ease_out_bounce"]

    assert elastic_in(0.0) == 0.0
    assert elastic_in(1.0) == 1.0
    assert elastic_out(0.0) == 0.0
    assert elastic_out(1.0) == 1.0

    # ease_out_elastic overshoots 1.0 around t=0.3-0.4
    peak = max(elastic_out(t / 100.0) for t in range(101))
    assert peak > 1.0, "ease_out_elastic should overshoot 1.0"

    # ease_out_bounce stays within [0, 1] but oscillates
    for i in range(21):
        t = i / 20.0
        val = bounce_out(t)
        assert 0.0 <= val <= 1.0 + 1e-5


def test_timing_validation():
    # Valid
    t = Timing(start_time=1.0, duration=2.0, delay=0.5, speed=2.0, easing="ease_in")
    assert t.cycle_duration == 1.0
    assert t.cycle_end_time == 2.5
    assert t.end_time == 2.5

    # Negative duration
    with pytest.raises(ValidationError):
        Timing(duration=-1.0)

    # Non-positive speed
    with pytest.raises(ValidationError):
        Timing(speed=0.0)
    with pytest.raises(ValidationError):
        Timing(speed=-0.5)

    # duration == 0 with loop == True must be rejected
    with pytest.raises(ValidationError, match="Looping animation cannot have duration=0"):
        Timing(duration=0.0, loop=True)

    # Unknown easing
    with pytest.raises(ValidationError):
        Timing(easing="super_magic_curve")


def test_timing_progress_and_raw_evaluate():
    # Test clamped get_progress vs raw evaluate
    t_elastic = Timing(duration=1.0, easing="ease_out_elastic")

    # Before start
    assert t_elastic.get_progress(-0.5) == 0.0
    assert t_elastic.evaluate(-0.5) == 0.0

    # At end
    assert t_elastic.get_progress(1.5) == 1.0
    assert t_elastic.evaluate(1.5) == 1.0

    # During overshoot (peaks around t=0.15)
    overshoot_time = 0.15
    raw_val = t_elastic.evaluate(overshoot_time)
    clamped_val = t_elastic.get_progress(overshoot_time)
    assert raw_val > 1.0
    assert clamped_val == 1.0


def test_timing_looping():
    t = Timing(start_time=0.0, duration=2.0, loop=True)
    assert t.end_time == float("inf")
    assert t.get_progress(1.0) == 0.5
    assert t.get_progress(2.0) == 0.0  # wraps around
    assert t.get_progress(3.0) == 0.5


def test_timing_serialization():
    t = Timing(start_time=1.0, duration=2.5, delay=0.2, speed=1.5, easing="ease_in_out", loop=False)
    data = t.to_dict()
    assert data["easing"] == "ease_in_out"
    assert data["duration"] == 2.5

    t2 = Timing.from_dict(data)
    assert t2.start_time == 1.0
    assert t2.duration == 2.5
    assert t2.delay == 0.2
    assert t2.speed == 1.5
    assert t2.easing == "ease_in_out"
    assert t2.loop is False

    # Callable easing raises SerializationError
    t_callable = Timing(easing=lambda x: x)
    with pytest.raises(SerializationError):
        t_callable.to_dict()
