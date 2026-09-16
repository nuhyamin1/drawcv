"""Outer glow post-processing visual effect."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.effects.effect import Effect
from drawcv.effects.gaussian import gaussian_pad_for_radius


@dataclass
class GlowEffect(Effect):
    """True outer glow effect radiating outward from an entity's alpha contour.

    Attributes:
        blur_radius: Gaussian penumbra radius for outer glow spread (>= 0.0).
        color: Base color of the glow (defaults to bright yellow Color(255, 255, 0, 1.0)).
        opacity: Multiplier for glow opacity in range [0.0, 1.0].
    """
    blur_radius: float = 10.0
    color: Color = field(default_factory=lambda: Color(255, 255, 0, 1.0))
    opacity: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.blur_radius, (int, float)) or isinstance(self.blur_radius, bool):
            raise ValidationError(f"blur_radius must be numeric, got {type(self.blur_radius).__name__}")
        if math.isnan(self.blur_radius) or math.isinf(self.blur_radius) or float(self.blur_radius) < 0.0:
            raise ValidationError(f"blur_radius must be a finite non-negative number, got {self.blur_radius}")
        self.blur_radius = float(self.blur_radius)

        if not isinstance(self.color, Color):
            raise ValidationError(f"Glow color must be a Color instance, got {type(self.color).__name__}")

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise ValidationError(f"opacity must be numeric, got {type(self.opacity).__name__}")
        if not (0.0 <= float(self.opacity) <= 1.0):
            raise ValidationError(f"opacity must be in range [0.0, 1.0], got {self.opacity}")
        self.opacity = float(self.opacity)

    @property
    def effect_type(self) -> str:
        return "glow"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Calculate visual bounds expanded symmetrically by outer glow penumbra."""
        blur_pad = float(gaussian_pad_for_radius(self.blur_radius))
        glow_bounds = BoundingBox(
            input_bounds.left - blur_pad,
            input_bounds.top - blur_pad,
            input_bounds.width + 2.0 * blur_pad,
            input_bounds.height + 2.0 * blur_pad,
        )
        return input_bounds.union(glow_bounds)

    def get_padding(self) -> tuple[float, float, float, float]:
        """Return symmetric padding for glow expansion (left, right, top, bottom)."""
        pad = float(gaussian_pad_for_radius(self.blur_radius))
        return (pad, pad, pad, pad)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "glow",
            "blur_radius": float(self.blur_radius),
            "color": self.color.to_dict(),
            "opacity": float(self.opacity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlowEffect:
        """Construct a GlowEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"GlowEffect data must be a dict, got {type(data).__name__}")
        color_val = Color.from_dict(data["color"]) if "color" in data else Color(255, 255, 0, 1.0)
        return cls(
            blur_radius=data.get("blur_radius", 10.0),
            color=color_val,
            opacity=data.get("opacity", 1.0),
        )
