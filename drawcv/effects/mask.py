"""Grayscale mask abstraction for modulating transparency."""

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from drawcv.core.enums import MaskMapping
from drawcv.core.exceptions import ValidationError


@dataclass
class Mask:
    """Grayscale mask modulating the alpha transparency of a drawable, group, or layer.
    
    Attributes:
        buffer: 2D uint8 NumPy array where 0 = completely hidden, 255 = completely opaque.
        mapping: Mapping mode over the target entity (FIT_BOUNDS or ABSOLUTE).
        inverted: When True, inverts the mask (0 becomes opaque, 255 becomes hidden).
    """
    buffer: np.ndarray
    mapping: MaskMapping = MaskMapping.FIT_BOUNDS
    inverted: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.buffer, np.ndarray):
            raise ValidationError(f"Mask buffer must be a NumPy array, got {type(self.buffer).__name__}")
        if self.buffer.ndim != 2:
            raise ValidationError(f"Mask buffer must be a 2D single-channel array, got shape {self.buffer.shape}")
        if self.buffer.dtype != np.uint8:
            raise ValidationError(f"Mask buffer must have dtype uint8, got {self.buffer.dtype}")
        if self.buffer.shape[0] <= 0 or self.buffer.shape[1] <= 0:
            raise ValidationError("Mask buffer dimensions must be strictly positive")

        if isinstance(self.mapping, str):
            try:
                self.mapping = MaskMapping(self.mapping.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid MaskMapping: '{self.mapping}'. Supported: 'fit_bounds', 'absolute'")
        elif not isinstance(self.mapping, MaskMapping):
            raise ValidationError(f"mapping must be a MaskMapping enum or string, got {type(self.mapping).__name__}")

        if not isinstance(self.inverted, bool):
            raise ValidationError(f"inverted must be a boolean, got {type(self.inverted).__name__}")

    @property
    def width(self) -> int:
        return int(self.buffer.shape[1])

    @property
    def height(self) -> int:
        return int(self.buffer.shape[0])

    def get_coverage(self, target_width: int, target_height: int) -> np.ndarray:
        """Return a float32 coverage mask in range [0.0, 1.0] sized (target_height, target_width)."""
        if target_width <= 0 or target_height <= 0:
            return np.zeros((max(0, target_height), max(0, target_width)), dtype=np.float32)

        if self.mapping == MaskMapping.FIT_BOUNDS:
            if self.width == target_width and self.height == target_height:
                resized = self.buffer
            else:
                resized = cv2.resize(self.buffer, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        else:
            # ABSOLUTE: Crop or pad onto target buffer at local origin (0, 0)
            resized = np.zeros((target_height, target_width), dtype=np.uint8)
            copy_h = min(self.height, target_height)
            copy_w = min(self.width, target_width)
            resized[:copy_h, :copy_w] = self.buffer[:copy_h, :copy_w]

        coverage = resized.astype(np.float32) / 255.0
        if self.inverted:
            coverage = 1.0 - coverage
        return coverage
