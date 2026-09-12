"""Unit tests for positioning and alignment utilities in DrawCV."""

import math
import pytest

from drawcv import (
    BoundingBox,
    Circle,
    Point,
    Rectangle,
    ValidationError,
    align_bottom,
    align_center_x,
    align_center_y,
    align_centers,
    align_left,
    align_right,
    align_top,
    distribute_horizontally,
    distribute_vertically,
    place_above,
    place_below,
    place_left_of,
    place_right_of,
)


def test_alignments():
    ref = Rectangle(position=Point(200, 200), width=100, height=80)
    target = Rectangle(position=Point(0, 0), width=40, height=30)

    # align_left
    align_left(target, ref)
    assert math.isclose(target.get_bounds().left, 200.0)

    # align_right
    align_right(target, ref)
    assert math.isclose(target.get_bounds().right, 300.0)

    # align_top
    align_top(target, ref)
    assert math.isclose(target.get_bounds().top, 200.0)

    # align_bottom
    align_bottom(target, ref)
    assert math.isclose(target.get_bounds().bottom, 280.0)

    # align_center_x
    align_center_x(target, ref)
    assert math.isclose(target.get_bounds().center.x, 250.0)

    # align_center_y
    align_center_y(target, ref)
    assert math.isclose(target.get_bounds().center.y, 240.0)

    # align_centers
    target.move(100, 50)
    align_centers(target, ref)
    assert math.isclose(target.get_bounds().center.x, 250.0)
    assert math.isclose(target.get_bounds().center.y, 240.0)

    # With BoundingBox as reference
    box = BoundingBox(500, 500, 50, 50)
    align_left(target, box)
    assert math.isclose(target.get_bounds().left, 500.0)


def test_relative_placements_default_gap_20():
    ref = Rectangle(position=Point(200, 200), width=100, height=100)
    t1 = Rectangle(position=Point(0, 0), width=40, height=40)
    t2 = Rectangle(position=Point(0, 0), width=40, height=40)
    t3 = Rectangle(position=Point(0, 0), width=40, height=40)
    t4 = Rectangle(position=Point(0, 0), width=40, height=40)

    # place_above: default gap is 20.0
    place_above(t1, ref)
    assert math.isclose(t1.get_bounds().bottom, 180.0)  # 200 - 20 = 180

    # place_below: default gap is 20.0
    place_below(t2, ref)
    assert math.isclose(t2.get_bounds().top, 320.0)  # 300 + 20 = 320

    # place_left_of: default gap is 20.0
    place_left_of(t3, ref)
    assert math.isclose(t3.get_bounds().right, 180.0)  # 200 - 20 = 180

    # place_right_of: default gap is 20.0
    place_right_of(t4, ref)
    assert math.isclose(t4.get_bounds().left, 320.0)  # 300 + 20 = 320


def test_relative_placement_cross_alignment():
    ref = Rectangle(position=Point(200, 200), width=100, height=100)
    target = Rectangle(position=Point(0, 0), width=40, height=40)

    # place_below with align="center"
    place_below(target, ref, gap=15, align="center")
    assert math.isclose(target.get_bounds().top, 315.0)
    assert math.isclose(target.get_bounds().center.x, ref.get_bounds().center.x)

    # place_right_of with align="bottom"
    place_right_of(target, ref, gap=10, align="bottom")
    assert math.isclose(target.get_bounds().left, 310.0)
    assert math.isclose(target.get_bounds().bottom, ref.get_bounds().bottom)


def test_distribution():
    # 4 rectangles of width 20
    items = [
        Rectangle(position=Point(i * 10, 0), width=20, height=20)
        for i in range(4)
    ]

    # Distribute with fixed spacing = 15.0
    distribute_horizontally(items, spacing=15.0)
    for i in range(1, len(items)):
        gap = items[i].get_bounds().left - items[i - 1].get_bounds().right
        assert math.isclose(gap, 15.0)

    # Distribute vertically with fixed spacing = 25.0
    distribute_vertically(items, spacing=25.0)
    for i in range(1, len(items)):
        gap = items[i].get_bounds().top - items[i - 1].get_bounds().bottom
        assert math.isclose(gap, 25.0)


def test_positioning_type_validation_and_lock_check():
    target = Rectangle(position=Point(0, 0), width=20, height=20, locked=True)
    ref = Rectangle(position=Point(50, 50), width=50, height=50)

    # Target is locked -> raises ValidationError
    with pytest.raises(ValidationError, match="locked"):
        align_left(target, ref)

    # Target is not a Drawable -> raises ValidationError
    with pytest.raises(ValidationError, match="Expected Drawable"):
        align_left(BoundingBox(0, 0, 10, 10), ref)  # type: ignore

    # Reference is invalid -> raises ValidationError
    with pytest.raises(ValidationError, match="Expected Drawable or BoundingBox"):
        align_left(ref, "invalid")  # type: ignore
