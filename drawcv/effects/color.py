"""Color manipulation post-processing visual effects."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.effects.effect import Effect


@dataclass
class BrightnessContrastEffect(Effect):
    """Adjust image brightness and contrast.

    Attributes:
        brightness: Additive luminance offset in range [-1.0, 1.0]. Defaults to 0.0.
        contrast: Multiplicative contrast factor in range [0.0, 5.0]. Defaults to 1.0.
    """
    brightness: float = 0.0
    contrast: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.brightness, (int, float)) or isinstance(self.brightness, bool):
            raise ValidationError(f"brightness must be numeric, got {type(self.brightness).__name__}")
        if math.isnan(self.brightness) or math.isinf(self.brightness):
            raise ValidationError(f"brightness must be finite, got {self.brightness}")
        if not (-1.0 <= float(self.brightness) <= 1.0):
            raise ValidationError(f"brightness must be in range [-1.0, 1.0], got {self.brightness}")
        self.brightness = float(self.brightness)

        if not isinstance(self.contrast, (int, float)) or isinstance(self.contrast, bool):
            raise ValidationError(f"contrast must be numeric, got {type(self.contrast).__name__}")
        if math.isnan(self.contrast) or math.isinf(self.contrast):
            raise ValidationError(f"contrast must be finite, got {self.contrast}")
        if not (0.0 <= float(self.contrast) <= 5.0):
            raise ValidationError(f"contrast must be in range [0.0, 5.0], got {self.contrast}")
        self.contrast = float(self.contrast)

    @property
    def effect_type(self) -> str:
        return "brightness_contrast"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "brightness_contrast",
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrightnessContrastEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        return cls(
            brightness=data.get("brightness", 0.0),
            contrast=data.get("contrast", 1.0),
        )


@dataclass
class SaturationEffect(Effect):
    """Adjust color saturation using Rec.709 luma coefficients.

    Attributes:
        factor: Saturation multiplier in range [0.0, 5.0].
                0.0 corresponds to complete desaturation, 1.0 is neutral.
    """
    factor: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.factor, (int, float)) or isinstance(self.factor, bool):
            raise ValidationError(f"factor must be numeric, got {type(self.factor).__name__}")
        if math.isnan(self.factor) or math.isinf(self.factor):
            raise ValidationError(f"factor must be finite, got {self.factor}")
        if not (0.0 <= float(self.factor) <= 5.0):
            raise ValidationError(f"factor must be in range [0.0, 5.0], got {self.factor}")
        self.factor = float(self.factor)

    @property
    def effect_type(self) -> str:
        return "saturation"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "saturation",
            "factor": float(self.factor),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SaturationEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        return cls(factor=data.get("factor", 1.0))


@dataclass
class HueShiftEffect(Effect):
    """Shift color hue around the color wheel by a specified angle in degrees.

    Uses canonical W3C hueRotate matrix formulation.
    """
    angle: float = 0.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.angle, (int, float)) or isinstance(self.angle, bool):
            raise ValidationError(f"angle must be numeric, got {type(self.angle).__name__}")
        if math.isnan(self.angle) or math.isinf(self.angle):
            raise ValidationError(f"angle must be finite, got {self.angle}")
        self.angle = float(self.angle) % 360.0

    @property
    def effect_type(self) -> str:
        return "hue_shift"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "hue_shift",
            "angle": float(self.angle),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HueShiftEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        return cls(angle=data.get("angle", 0.0))


@dataclass
class GrayscaleEffect(Effect):
    """Convert colors to grayscale using Rec.709 luma weighting.

    Attributes:
        intensity: Grayscale blend factor in range [0.0, 1.0].
                   1.0 is full grayscale, 0.0 is unaffected color.
    """
    intensity: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.intensity, (int, float)) or isinstance(self.intensity, bool):
            raise ValidationError(f"intensity must be numeric, got {type(self.intensity).__name__}")
        if math.isnan(self.intensity) or math.isinf(self.intensity):
            raise ValidationError(f"intensity must be finite, got {self.intensity}")
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValidationError(f"intensity must be in range [0.0, 1.0], got {self.intensity}")
        self.intensity = float(self.intensity)

    @property
    def effect_type(self) -> str:
        return "grayscale"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "grayscale",
            "intensity": float(self.intensity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GrayscaleEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        return cls(intensity=data.get("intensity", 1.0))


@dataclass
class SepiaEffect(Effect):
    """Apply a warm sepia tone filter.

    Attributes:
        intensity: Sepia blend factor in range [0.0, 1.0].
                   1.0 is full sepia tone, 0.0 is unaffected color.
    """
    intensity: float = 1.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.intensity, (int, float)) or isinstance(self.intensity, bool):
            raise ValidationError(f"intensity must be numeric, got {type(self.intensity).__name__}")
        if math.isnan(self.intensity) or math.isinf(self.intensity):
            raise ValidationError(f"intensity must be finite, got {self.intensity}")
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValidationError(f"intensity must be in range [0.0, 1.0], got {self.intensity}")
        self.intensity = float(self.intensity)

    @property
    def effect_type(self) -> str:
        return "sepia"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sepia",
            "intensity": float(self.intensity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SepiaEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        return cls(intensity=data.get("intensity", 1.0))


def _default_color_matrix() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
    )


@dataclass
class ColorMatrixEffect(Effect):
    """Linear color matrix transformation.

    Transforms the 5D input vector [R, G, B, A, 1]^T by a 4x5 matrix:
        Columns: R coefficient, G coefficient, B coefficient, A coefficient, bias.

    Alpha row (row index 3) must be strictly (0.0, 0.0, 0.0, 1.0, 0.0) for this milestone
    to guarantee strict alpha preservation and prevent premultiplied invariant violations.
    """
    matrix: tuple[tuple[float, ...], ...] = field(default_factory=_default_color_matrix)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.matrix, (tuple, list)):
            raise ValidationError(f"Color matrix must be a sequence of 4 rows, got {type(self.matrix).__name__}")
        if len(self.matrix) != 4:
            raise ValidationError(f"Color matrix must have exactly 4 rows, got {len(self.matrix)}")

        validated_rows = []
        for row_idx, row in enumerate(self.matrix):
            if not isinstance(row, (tuple, list)):
                raise ValidationError(f"Matrix row {row_idx} must be a sequence, got {type(row).__name__}")
            if len(row) != 5:
                raise ValidationError(f"Matrix row {row_idx} must have 5 columns, got {len(row)}")
            row_floats = []
            for col_idx, val in enumerate(row):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValidationError(f"Matrix entry ({row_idx}, {col_idx}) must be numeric, got {type(val).__name__}")
                if math.isnan(val) or math.isinf(val):
                    raise ValidationError(f"Matrix entry ({row_idx}, {col_idx}) must be finite, got {val}")
                row_floats.append(float(val))
            validated_rows.append(tuple(row_floats))

        # Strict alpha preservation validation on row 3
        alpha_row = validated_rows[3]
        expected_alpha_row = (0.0, 0.0, 0.0, 1.0, 0.0)
        if alpha_row != expected_alpha_row:
            raise ValidationError(
                f"Color matrix row 3 must be strictly {expected_alpha_row} for alpha preservation, got {alpha_row}"
            )

        self.matrix = tuple(validated_rows)

    @property
    def effect_type(self) -> str:
        return "color_matrix"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        return input_bounds

    def get_padding(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "color_matrix",
            "matrix": [list(row) for row in self.matrix],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColorMatrixEffect:
        if not isinstance(data, dict):
            raise ValidationError(f"Data must be a dict, got {type(data).__name__}")
        raw_matrix = data.get("matrix", _default_color_matrix())
        return cls(matrix=raw_matrix)
