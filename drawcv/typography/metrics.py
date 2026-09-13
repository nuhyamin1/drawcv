"""Immutable local-coordinate typography measurements."""
from dataclasses import dataclass
from drawcv.core.bounds import BoundingBox


@dataclass(frozen=True)
class TextLineMetrics:
    text: str
    advance_width: float
    baseline: float
    line_box: BoundingBox
    ink_bounds: BoundingBox | None


@dataclass(frozen=True)
class TextMetrics:
    advance_width: float
    layout_bounds: BoundingBox
    paragraph_bounds: BoundingBox
    ink_bounds: BoundingBox | None
    lines: tuple[TextLineMetrics, ...]
