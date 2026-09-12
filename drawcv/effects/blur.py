"""Spatial blur effect implementation."""

from __future__ import annotations
from dataclasses import dataclass
import math

from drawcv.core.enums import BlurType
from drawcv.core.exceptions import ValidationError
from drawcv.effects.effect import Effect


@dataclass
class BlurEffect(Effect):
    """Spatial content blur effect applied to an entity's isolated buffer.
    
    Attributes:
        kernel_size: Blur window dimension (must be an odd positive integer, default: 15).
        sigma: Gaussian standard deviation (>= 0.0; if 0.0, automatically computed from kernel_size).
        blur_type: Blur algorithm (BlurType.GAUSSIAN or BlurType.BOX).
    """
    kernel_size: int = 15
    sigma: float = 0.0
    blur_type: BlurType = BlurType.GAUSSIAN

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.kernel_size, int) or isinstance(self.kernel_size, bool):
            raise ValidationError(f"kernel_size must be an integer, got {type(self.kernel_size).__name__}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValidationError(f"kernel_size must be an odd positive integer, got {self.kernel_size}")

        if not isinstance(self.sigma, (int, float)) or isinstance(self.sigma, bool):
            raise ValidationError(f"sigma must be numeric, got {type(self.sigma).__name__}")
        if math.isnan(self.sigma) or math.isinf(self.sigma) or float(self.sigma) < 0.0:
            raise ValidationError(f"sigma must be non-negative, got {self.sigma}")
        self.sigma = float(self.sigma)

        if isinstance(self.blur_type, str):
            try:
                self.blur_type = BlurType(self.blur_type.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid blur_type: '{self.blur_type}'. Supported: 'gaussian', 'box'")
        elif not isinstance(self.blur_type, BlurType):
            raise ValidationError(f"blur_type must be a BlurType enum or string, got {type(self.blur_type).__name__}")

    def get_padding(self) -> tuple[float, float, float, float]:
        """Return symmetric padding for blur expansion (left, right, top, bottom)."""
        if self.sigma > 0.0:
            pad = float(math.ceil(3.0 * self.sigma))
        else:
            pad = float(self.kernel_size // 2)
        return (pad, pad, pad, pad)
