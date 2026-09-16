"""Unsharp masking spatial sharpening effect."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_real_number
from drawcv.effects.effect import Effect
from drawcv.effects.gaussian import gaussian_pad_for_radius


@dataclass
class SharpenEffect(Effect):
    """Unsharp masking spatial sharpening effect operating on straight color.

    Enhances high-frequency spatial gradients using unsharp masking:

        sharpened = original + amount * (original - gaussian_blurred(radius))

    Preserves original output pixel alpha. Transparent pixels remain strictly (0, 0, 0, 0).

    Attributes:
        amount: Sharpening strength multiplier (>= 0.0; 0.0 is an exact no-op, default: 1.0).
        radius: Gaussian standard deviation/radius for unsharp mask (>= 0.0; 0.0 is an exact no-op, default: 1.0).
    """
    amount: float = 1.0
    radius: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        self.amount = validate_real_number(self.amount, "amount")
        if self.amount < 0.0:
            raise ValidationError(f"amount must be non-negative, got {self.amount}")

        self.radius = validate_real_number(self.radius, "radius")
        if self.radius < 0.0:
            raise ValidationError(f"radius must be non-negative, got {self.radius}")

    @property
    def effect_type(self) -> str:
        return "sharpen"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Sharpening preserves visual bounds."""
        return input_bounds

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Support padding derived from finite Gaussian kernel support."""
        pad = float(gaussian_pad_for_radius(self.radius))
        return (pad, pad, pad, pad)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "sharpen",
            "amount": float(self.amount),
            "radius": float(self.radius),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharpenEffect:
        """Construct a SharpenEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"SharpenEffect data must be a dict, got {type(data).__name__}")
        return cls(
            amount=data.get("amount", 1.0),
            radius=data.get("radius", 1.0),
        )
