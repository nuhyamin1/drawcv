"""Luminance-based spatial edge detection effect."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.enums import EdgeDetectionMethod, coerce_edge_detection_method
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_real_number
from drawcv.effects.effect import Effect


@dataclass
class EdgeDetectionEffect(Effect):
    """Spatial edge detection effect operating on Rec.709 luminance.

    Computes grayscale edge response using either:
    - SOBEL: Gradient magnitude from 3x3 horizontal and vertical Sobel filters.
    - LAPLACIAN: Absolute second derivative response from 3x3 discrete Laplacian filter.

    Uses fixed mathematical scaling with no image-dependent normalization.
    Uniform-color input produces zero edge response (or 1.0 when inverted).
    Preserves original output pixel alpha. Transparent pixels remain strictly (0, 0, 0, 0).

    Attributes:
        method: Edge detection algorithm (EdgeDetectionMethod.SOBEL or EdgeDetectionMethod.LAPLACIAN).
        strength: Edge response multiplier (>= 0.0, default: 1.0).
        invert: If True, produces inverted grayscale edges (white background, dark edges; default: False).
    """
    method: EdgeDetectionMethod = EdgeDetectionMethod.SOBEL
    strength: float = 1.0
    invert: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        self.method = coerce_edge_detection_method(self.method)
        self.strength = validate_real_number(self.strength, "strength")
        if self.strength < 0.0:
            raise ValidationError(f"strength must be non-negative, got {self.strength}")

        if not isinstance(self.invert, bool):
            raise ValidationError(f"invert must be a boolean, got {type(self.invert).__name__}")

    @property
    def effect_type(self) -> str:
        return "edge_detection"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Edge detection preserves visual bounds."""
        return input_bounds

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Support padding for 3x3 edge operator."""
        return (1.0, 1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "edge_detection",
            "method": self.method.value,
            "strength": float(self.strength),
            "invert": bool(self.invert),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeDetectionEffect:
        """Construct an EdgeDetectionEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"EdgeDetectionEffect data must be a dict, got {type(data).__name__}")
        raw_method = data.get("method", EdgeDetectionMethod.SOBEL)
        return cls(
            method=coerce_edge_detection_method(raw_method),
            strength=data.get("strength", 1.0),
            invert=data.get("invert", False),
        )
