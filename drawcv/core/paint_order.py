"""SVG paint-order values: the sequence fill, stroke and markers are drawn in."""
from __future__ import annotations

from drawcv.core.exceptions import ValidationError

PAINT_STEPS = ("fill", "stroke", "markers")


def normalize_paint_order(value: object) -> str:
    """Canonical form: 'normal' or all three steps, e.g. 'stroke fill markers'.

    As in SVG, listed steps come first and omitted steps follow in normal order,
    so 'stroke' means 'stroke fill markers'.
    """
    if not isinstance(value, str):
        raise ValidationError(f"paint_order must be a string, got {type(value).__name__}")
    tokens = value.replace(",", " ").lower().split()
    if tokens == ["normal"]:
        return "normal"
    if not tokens or any(t not in PAINT_STEPS for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValidationError(f"Invalid paint_order {value!r}; use 'normal' or an order of fill, stroke, markers")
    order = tokens + [step for step in PAINT_STEPS if step not in tokens]
    return "normal" if tuple(order) == PAINT_STEPS else " ".join(order)


def paint_steps(value: str) -> tuple[str, ...]:
    """The three steps of a normalized paint order."""
    return PAINT_STEPS if value == "normal" else tuple(value.split())
