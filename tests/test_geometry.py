"""Unit tests for Point geometry and vector operations."""

import math
import pytest
from dataclasses import FrozenInstanceError

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


def test_point_construction():
    p = Point(10, 20.5)
    assert p.x == 10.0
    assert p.y == 20.5
    assert isinstance(p.x, float)
    assert isinstance(p.y, float)


def test_point_immutability():
    p = Point(5, 10)
    with pytest.raises(FrozenInstanceError):
        p.x = 20  # type: ignore


def test_point_invalid_coordinates():
    with pytest.raises(ValidationError):
        Point("10", 20)  # type: ignore

    with pytest.raises(ValidationError):
        Point(float("nan"), 10)

    with pytest.raises(ValidationError):
        Point(10, float("inf"))


def test_point_arithmetic():
    p1 = Point(10, 20)
    p2 = Point(5, 8)

    # Addition
    p_add = p1 + p2
    assert p_add == Point(15, 28)

    # Subtraction
    p_sub = p1 - p2
    assert p_sub == Point(5, 12)

    # Scalar multiplication
    p_mul1 = p1 * 2.5
    assert p_mul1 == Point(25, 50)
    p_mul2 = 2 * p1
    assert p_mul2 == Point(20, 40)

    # Scalar division
    p_div = p1 / 2
    assert p_div == Point(5, 10)

    with pytest.raises(ValidationError):
        _ = p1 / 0


def test_point_translation():
    p = Point(10, 20)
    p_trans = p.translate(5, -10)
    assert p_trans == Point(15, 10)
    assert p == Point(10, 20)  # Original remains unchanged


def test_point_distance():
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert math.isclose(p1.distance_to(p2), 5.0)

    with pytest.raises(ValidationError):
        p1.distance_to((3, 4))  # type: ignore


def test_point_conversions():
    p = Point(12.5, 34.5)
    assert p.to_tuple() == (12.5, 34.5)
    arr = p.to_numpy()
    assert arr.shape == (2,)
    assert arr[0] == 12.5 and arr[1] == 34.5
    assert repr(p) == "Point(12.5, 34.5)"
