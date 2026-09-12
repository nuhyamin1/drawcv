"""Drop shadow post-processing visual effect."""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.effects.effect import Effect


@dataclass
class ShadowEffect(Effect):
    """Blurred offset drop shadow effect rendered behind an entity.
    
    Attributes:
        offset_x: Horizontal shadow displacement in pixels (positive = right, negative = left).
        offset_y: Vertical shadow displacement in pixels (positive = down, negative = up).
        blur_radius: Gaussian blur standard deviation for shadow penumbra (>= 0.0).
        color: Base color of the shadow silhouette (defaults to semi-transparent black).
        opacity: Multiplier for shadow opacity in range [0.0, 1.0].
    """
    offset_x: float = 8.0
    offset_y: float = 8.0
    blur_radius: float = 10.0
    color: Color = field(default_factory=lambda: Color(0, 0, 0, 0.5))
    opacity: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for name, val in (("offset_x", self.offset_x), ("offset_y", self.offset_y)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"{name} must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"{name} must be a finite number, got {val}")

        self.offset_x = float(self.offset_x)
        self.offset_y = float(self.offset_y)

        if not isinstance(self.blur_radius, (int, float)) or isinstance(self.blur_radius, bool):
            raise ValidationError(f"blur_radius must be numeric, got {type(self.blur_radius).__name__}")
        if math.isnan(self.blur_radius) or math.isinf(self.blur_radius) or float(self.blur_radius) < 0.0:
            raise ValidationError(f"blur_radius must be non-negative, got {self.blur_radius}")
        self.blur_radius = float(self.blur_radius)

        if not isinstance(self.color, Color):
            raise ValidationError(f"Shadow color must be a Color instance, got {type(self.color).__name__}")

        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise ValidationError(f"opacity must be numeric, got {type(self.opacity).__name__}")
        if not (0.0 <= float(self.opacity) <= 1.0):
            raise ValidationError(f"opacity must be in range [0.0, 1.0], got {self.opacity}")
        self.opacity = float(self.opacity)

    def get_padding(self) -> tuple[float, float, float, float]:
        """Return asymmetric padding for signed shadow offset and blur expansion (left, right, top, bottom)."""
        blur_padding = float(math.ceil(3.0 * self.blur_radius)) if self.blur_radius > 0.0 else 0.0
        left = blur_padding + max(0.0, -self.offset_x)
        right = blur_padding + max(0.0, self.offset_x)
        top = blur_padding + max(0.0, -self.offset_y)
        bottom = blur_padding + max(0.0, self.offset_y)
        return (left, right, top, bottom)
