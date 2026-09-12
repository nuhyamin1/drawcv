"""Fill style specification for interior shape rendering."""

from __future__ import annotations
from dataclasses import dataclass, field

from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError


@dataclass
class FillStyle:
    """Fill appearance style for closed shapes.
    
    Attributes:
        enabled: Whether fill is active.
        color: Fill color.
        opacity: Fill opacity multiplier in range [0.0, 1.0].
    """
    enabled: bool = True
    color: Color = field(default_factory=Color.white)
    opacity: float = 1.0

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_initialized", True)

    def _validate(self):
        if not isinstance(self.enabled, bool):
            raise ValidationError(f"FillStyle 'enabled' must be a boolean, got {type(self.enabled).__name__}")
        if not isinstance(self.color, Color):
            raise ValidationError(f"FillStyle 'color' must be a Color, got {type(self.color).__name__}")
        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise ValidationError(f"FillStyle 'opacity' must be numeric, got {type(self.opacity).__name__}")
        if not (0.0 <= float(self.opacity) <= 1.0):
            raise ValidationError(f"FillStyle 'opacity' must be in range [0.0, 1.0], got {self.opacity}")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if getattr(self, "_initialized", False):
            self._validate()
