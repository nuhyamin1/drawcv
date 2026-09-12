"""Unit tests for BoundingBox spatial operations and validations."""

import pytest
from dataclasses import FrozenInstanceError

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


def test_bounding_box_construction():
    b = BoundingBox(10, 20, 100, 50)
    assert b.left == 10.0
    assert b.top == 20.0
    assert b.right == 110.0
    assert b.bottom == 70.0
    assert b.width == 100.0
    assert b.height == 50.0
    assert b.center == Point(60, 45)


def test_bounding_box_immutability():
    b = BoundingBox(0, 0, 10, 10)
    with pytest.raises(FrozenInstanceError):
        b.x = 5  # type: ignore


def test_bounding_box_validation():
    with pytest.raises(ValidationError):
        BoundingBox(0, 0, -10, 50)

    with pytest.raises(ValidationError):
        BoundingBox(0, 0, 50, -1)

    with pytest.raises(ValidationError):
        BoundingBox("0", 0, 10, 10)  # type: ignore


def test_bounding_box_corners():
    b = BoundingBox(10, 20, 100, 50)
    assert b.top_left == Point(10, 20)
    assert b.top_right == Point(110, 20)
    assert b.bottom_left == Point(10, 70)
    assert b.bottom_right == Point(110, 70)


def test_bounding_box_contains():
    b = BoundingBox(10, 10, 50, 50)
    assert b.contains(Point(20, 20))
    assert b.contains(Point(10, 10))  # Boundary
    assert b.contains(Point(60, 60))  # Boundary
    assert not b.contains(Point(5, 20))
    assert not b.contains(Point(61, 20))


def test_bounding_box_intersects():
    b1 = BoundingBox(0, 0, 100, 100)
    b2 = BoundingBox(50, 50, 100, 100)
    b3 = BoundingBox(200, 200, 50, 50)

    assert b1.intersects(b2)
    assert b2.intersects(b1)
    assert not b1.intersects(b3)


def test_bounding_box_union():
    b1 = BoundingBox(10, 10, 20, 20)
    b2 = BoundingBox(40, 50, 10, 10)
    u = b1.union(b2)
    assert u.left == 10
    assert u.top == 10
    assert u.right == 50
    assert u.bottom == 60


def test_bounding_box_expand():
    b = BoundingBox(20, 20, 40, 40)
    b_exp = b.expand(5)
    assert b_exp.left == 15
    assert b_exp.top == 15
    assert b_exp.width == 50
    assert b_exp.height == 50
