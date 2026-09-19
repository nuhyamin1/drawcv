"""Grayscale mask abstraction for modulating transparency."""

from __future__ import annotations
from dataclasses import dataclass
import base64
import cv2
import numpy as np

from drawcv.core.enums import MaskMapping
from drawcv.core.exceptions import SerializationError, ValidationError



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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation with structured raster payload."""
        success, encoded = cv2.imencode(".png", self.buffer)
        if not success:
            raise SerializationError("Failed to encode Mask buffer to PNG")
        data_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {
            "buffer": {
                "encoding": "png_base64",
                "dtype": "uint8",
                "shape": list(self.buffer.shape),
                "data": data_b64,
            },
            "mapping": self.mapping.value,
            "inverted": bool(self.inverted),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Mask | None:
        """Construct a Mask from a dictionary."""
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValidationError(f"Mask data must be a dict, got {type(data).__name__}")
        if data.get('type') == 'vector_mask':
            from drawcv.effects.vector_mask import VectorMask
            return VectorMask.from_dict(data)

        buf_data = data.get("buffer")
        if not isinstance(buf_data, dict):
            raise SerializationError("Mask 'buffer' must be a structured raster dictionary")
        if buf_data.get("encoding") != "png_base64":
            raise SerializationError(f"Unsupported mask encoding '{buf_data.get('encoding')}'")

        raw_bytes = base64.b64decode(buf_data["data"])
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise SerializationError("Failed to decode Mask PNG buffer")

        expected_shape = tuple(buf_data.get("shape", []))
        if expected_shape and decoded.shape != expected_shape:
            raise SerializationError(
                f"Decoded mask shape {decoded.shape} does not match expected shape {expected_shape}"
            )

        mapping_str = data.get("mapping", MaskMapping.FIT_BOUNDS.value)
        mapping_val = MaskMapping(mapping_str) if isinstance(mapping_str, str) else MaskMapping.FIT_BOUNDS
        inverted_val = bool(data.get("inverted", False))
        return cls(buffer=decoded, mapping=mapping_val, inverted=inverted_val)
