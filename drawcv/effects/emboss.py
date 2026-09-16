"""Directional luminance gradient emboss effect."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_real_number
from drawcv.effects.effect import Effect


@dataclass
class EmbossEffect(Effect):
    """Directional luminance gradient relief emboss effect.

    Computes directional luminance gradient from straight color:

        directional = cos(angle) * gx + sin(angle) * gy
        embossed = bias + strength * directional

    Screen angle semantics:
        0 degrees   = +X (right)
        90 degrees  = +Y (down)
        135 degrees = down-right (default light source from top-left)

    Produces neutral-centered grayscale relief inside the original alpha footprint.
    Uniform-color input produces exact neutral bias (default: 0.5 gray).
    Preserves original output pixel alpha. Transparent pixels remain strictly (0, 0, 0, 0).

    Attributes:
        strength: Multiplier for directional gradient relief (>= 0.0, default: 1.0).
        angle: Light source direction in degrees (0 = right, 90 = down, default: 135.0).
        bias: Neutral baseline luminance in range [0.0, 1.0] (default: 0.5).
    """
    strength: float = 1.0
    angle: float = 135.0
    bias: float = 0.5

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        self.strength = validate_real_number(self.strength, "strength")
        if self.strength < 0.0:
            raise ValidationError(f"strength must be non-negative, got {self.strength}")

        raw_angle = validate_real_number(self.angle, "angle")
        self.angle = float(raw_angle) % 360.0

        self.bias = validate_real_number(self.bias, "bias")
        if not (0.0 <= self.bias <= 1.0):
            raise ValidationError(f"bias must be in range [0.0, 1.0], got {self.bias}")

    @property
    def effect_type(self) -> str:
        return "emboss"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Embossing preserves visual bounds."""
        return input_bounds

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Support padding for 3x3 gradient kernel."""
        return (1.0, 1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "emboss",
            "strength": float(self.strength),
            "angle": float(self.angle),
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbossEffect:
        """Construct an EmbossEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"EmbossEffect data must be a dict, got {type(data).__name__}")
        return cls(
            strength=data.get("strength", 1.0),
            angle=data.get("angle", 135.0),
            bias=data.get("bias", 0.5),
        )
