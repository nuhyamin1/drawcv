"""Declarative relative positioning and alignment utilities for DrawCV drawables."""

from __future__ import annotations
from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import atomic_transforms


def _get_ref_bounds(b: Drawable | BoundingBox) -> BoundingBox:
    if isinstance(b, Drawable):
        return b.get_bounds()
    elif isinstance(b, BoundingBox):
        return b
    else:
        raise ValidationError(f"Expected Drawable or BoundingBox for reference 'b', got {type(b).__name__}")


def _check_target(a: Drawable) -> None:
    if not isinstance(a, Drawable):
        raise ValidationError(f"Expected Drawable for target 'a', got {type(a).__name__}")
    if a.effective_locked:
        raise ValidationError(f"Cannot reposition locked drawable '{a.id}'")


def _apply_cross_align_horizontal(a: Drawable, ref_box: BoundingBox, align: str) -> None:
    norm = align.strip().lower()
    if norm in ("center", "center_x"):
        align_center_x(a, ref_box)
    elif norm == "left":
        align_left(a, ref_box)
    elif norm == "right":
        align_right(a, ref_box)
    else:
        raise ValidationError(f"Unknown horizontal alignment '{align}'. Valid options: center, left, right")


def _apply_cross_align_vertical(a: Drawable, ref_box: BoundingBox, align: str) -> None:
    norm = align.strip().lower()
    if norm in ("center", "center_y"):
        align_center_y(a, ref_box)
    elif norm == "top":
        align_top(a, ref_box)
    elif norm == "bottom":
        align_bottom(a, ref_box)
    else:
        raise ValidationError(f"Unknown vertical alignment '{align}'. Valid options: center, top, bottom")


# -----------------------------------------------------------------------------
# Alignment Functions
# -----------------------------------------------------------------------------

def align_left(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' horizontally so its left edge aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dx = rb.left - a.get_bounds().left
    a.move(dx, 0.0)
    return a


def align_right(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' horizontally so its right edge aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dx = rb.right - a.get_bounds().right
    a.move(dx, 0.0)
    return a


def align_top(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' vertically so its top edge aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dy = rb.top - a.get_bounds().top
    a.move(0.0, dy)
    return a


def align_bottom(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' vertically so its bottom edge aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dy = rb.bottom - a.get_bounds().bottom
    a.move(0.0, dy)
    return a


def align_center_x(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' horizontally so its horizontal center aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dx = rb.center.x - a.get_bounds().center.x
    a.move(dx, 0.0)
    return a


def align_center_y(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' vertically so its vertical center aligns with 'b'."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dy = rb.center.y - a.get_bounds().center.y
    a.move(0.0, dy)
    return a


@atomic_transforms
def align_centers(a: Drawable, b: Drawable | BoundingBox) -> Drawable:
    """Move target 'a' so its center aligns with 'b' in both dimensions."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    ab = a.get_bounds()
    dx = rb.center.x - ab.center.x
    dy = rb.center.y - ab.center.y
    a.move(dx, dy)
    return a


# -----------------------------------------------------------------------------
# Relative Placement Functions (Default gap=20.0)
# -----------------------------------------------------------------------------

@atomic_transforms
def place_above(
    a: Drawable,
    b: Drawable | BoundingBox,
    gap: float | int = 20.0,
    align: str | None = None
) -> Drawable:
    """Place target 'a' directly above 'b' with spacing 'gap' (default 20.0)."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dy = (rb.top - float(gap)) - a.get_bounds().bottom
    a.move(0.0, dy)
    if align is not None:
        _apply_cross_align_horizontal(a, rb, align)
    return a


@atomic_transforms
def place_below(
    a: Drawable,
    b: Drawable | BoundingBox,
    gap: float | int = 20.0,
    align: str | None = None
) -> Drawable:
    """Place target 'a' directly below 'b' with spacing 'gap' (default 20.0)."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dy = (rb.bottom + float(gap)) - a.get_bounds().top
    a.move(0.0, dy)
    if align is not None:
        _apply_cross_align_horizontal(a, rb, align)
    return a


@atomic_transforms
def place_left_of(
    a: Drawable,
    b: Drawable | BoundingBox,
    gap: float | int = 20.0,
    align: str | None = None
) -> Drawable:
    """Place target 'a' directly to the left of 'b' with spacing 'gap' (default 20.0)."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dx = (rb.left - float(gap)) - a.get_bounds().right
    a.move(dx, 0.0)
    if align is not None:
        _apply_cross_align_vertical(a, rb, align)
    return a


@atomic_transforms
def place_right_of(
    a: Drawable,
    b: Drawable | BoundingBox,
    gap: float | int = 20.0,
    align: str | None = None
) -> Drawable:
    """Place target 'a' directly to the right of 'b' with spacing 'gap' (default 20.0)."""
    _check_target(a)
    rb = _get_ref_bounds(b)
    dx = (rb.right + float(gap)) - a.get_bounds().left
    a.move(dx, 0.0)
    if align is not None:
        _apply_cross_align_vertical(a, rb, align)
    return a


# -----------------------------------------------------------------------------
# Distribution Functions
# -----------------------------------------------------------------------------

@atomic_transforms
def distribute_horizontally(objects: list[Drawable], spacing: float | int | None = None) -> list[Drawable]:
    """Distribute a sequence of objects horizontally left-to-right.
    
    If spacing is provided, spaces each consecutive item by spacing pixels.
    If spacing is None (requires at least 3 items), distributes items evenly between first and last.
    """
    if len(objects) < 2:
        return objects

    for obj in objects:
        _check_target(obj)

    if spacing is not None:
        gap = float(spacing)
        for i in range(1, len(objects)):
            place_right_of(objects[i], objects[i - 1], gap=gap)
    else:
        if len(objects) < 3:
            return objects
        first = objects[0].get_bounds()
        last = objects[-1].get_bounds()
        total_span = last.left - first.right
        intermediate_widths = sum(obj.get_bounds().width for obj in objects[1:-1])
        remaining_gap = total_span - intermediate_widths
        num_intervals = len(objects) - 1
        gap = remaining_gap / num_intervals

        for i in range(1, len(objects) - 1):
            place_right_of(objects[i], objects[i - 1], gap=gap)

    return objects


@atomic_transforms
def distribute_vertically(objects: list[Drawable], spacing: float | int | None = None) -> list[Drawable]:
    """Distribute a sequence of objects vertically top-to-bottom.
    
    If spacing is provided, spaces each consecutive item by spacing pixels.
    If spacing is None (requires at least 3 items), distributes items evenly between first and last.
    """
    if len(objects) < 2:
        return objects

    for obj in objects:
        _check_target(obj)

    if spacing is not None:
        gap = float(spacing)
        for i in range(1, len(objects)):
            place_below(objects[i], objects[i - 1], gap=gap)
    else:
        if len(objects) < 3:
            return objects
        first = objects[0].get_bounds()
        last = objects[-1].get_bounds()
        total_span = last.top - first.bottom
        intermediate_heights = sum(obj.get_bounds().height for obj in objects[1:-1])
        remaining_gap = total_span - intermediate_heights
        num_intervals = len(objects) - 1
        gap = remaining_gap / num_intervals

        for i in range(1, len(objects) - 1):
            place_below(objects[i], objects[i - 1], gap=gap)

    return objects
