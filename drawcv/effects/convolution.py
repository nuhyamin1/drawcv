"""Discrete 2D spatial convolution effect."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_real_number
from drawcv.effects.effect import Effect


@dataclass
class ConvolutionEffect(Effect):
    """Generalized discrete 2D spatial filtering effect.

    Computes normalized 2D spatial cross-correlation on normalized straight BGR colors
    with explicit centered anchor and transparent-zero boundaries:

        C'(x, y) = factor * sum_{u=0}^{W-1} sum_{v=0}^{H-1} K(v, u) * C(x + u - W//2, y + v - H//2) + bias

    Preserves original output pixel alpha. Transparent pixels remain strictly (0, 0, 0, 0).

    Attributes:
        kernel: 2D rectangular matrix of finite real numbers with odd dimensions (1 <= W, H <= 63).
        factor: Multiplicative scalar applied to the convolution response (default: 1.0).
        bias: Additive offset in normalized straight-color units [0.0, 1.0] (default: 0.0).
    """
    kernel: Sequence[Sequence[float]] | np.ndarray
    factor: float = 1.0
    bias: float = 0.0

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if isinstance(self.kernel, np.ndarray):
            raw_kernel = self.kernel.tolist()
        elif isinstance(self.kernel, (list, tuple)):
            raw_kernel = self.kernel
        else:
            raise ValidationError(f"kernel must be a 2D sequence or NumPy array, got {type(self.kernel).__name__}")

        if not isinstance(raw_kernel, (list, tuple)) or len(raw_kernel) == 0:
            raise ValidationError("kernel must be a non-empty 2D sequence")

        h = len(raw_kernel)
        if h <= 0 or h > 63 or h % 2 == 0:
            raise ValidationError(f"kernel height must be an odd integer in range [1, 63], got {h}")

        first_row = raw_kernel[0]
        if not isinstance(first_row, (list, tuple)):
            raise ValidationError("kernel rows must be sequences of numbers")
        w = len(first_row)
        if w <= 0 or w > 63 or w % 2 == 0:
            raise ValidationError(f"kernel width must be an odd integer in range [1, 63], got {w}")

        validated_rows = []
        for r_idx, row in enumerate(raw_kernel):
            if not isinstance(row, (list, tuple)):
                raise ValidationError(f"kernel row {r_idx} must be a sequence, got {type(row).__name__}")
            if len(row) != w:
                raise ValidationError(f"kernel row {r_idx} length {len(row)} does not match width {w}")
            row_vals = []
            for c_idx, val in enumerate(row):
                v = validate_real_number(val, f"kernel[{r_idx}][{c_idx}]")
                row_vals.append(v)
            validated_rows.append(tuple(row_vals))

        self.kernel = tuple(validated_rows)
        self.factor = validate_real_number(self.factor, "factor")
        self.bias = validate_real_number(self.bias, "bias")

    @property
    def effect_type(self) -> str:
        return "convolution"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Convolution preserves visual bounds."""
        return input_bounds

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Support padding derived from kernel dimensions."""
        h = len(self.kernel)
        w = len(self.kernel[0])
        pad_x = float(w // 2)
        pad_y = float(h // 2)
        return (pad_x, pad_x, pad_y, pad_y)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "convolution",
            "kernel": [list(row) for row in self.kernel],
            "factor": float(self.factor),
            "bias": float(self.bias),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvolutionEffect:
        """Construct a ConvolutionEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"ConvolutionEffect data must be a dict, got {type(data).__name__}")
        if "kernel" not in data:
            raise ValidationError("ConvolutionEffect requires 'kernel'")
        return cls(
            kernel=data["kernel"],
            factor=data.get("factor", 1.0),
            bias=data.get("bias", 0.0),
        )
