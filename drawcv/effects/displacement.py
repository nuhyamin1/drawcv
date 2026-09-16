"""Retained displacement map post-processing effect driven by ImagePaint."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.enums import (
    DisplacementChannel,
    ImageInterpolation,
    coerce_displacement_channel,
)
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_real_number
from drawcv.effects.effect import Effect
from drawcv.styles.paint import ImagePaint, paint_from_dict


@dataclass
class DisplacementMapEffect(Effect):
    """Geometric displacement post-processing effect driven by retained ImagePaint.

    Spatially distorts an entity's content buffer according to straight color values
    sampled from an ImagePaint texture:

        source_global_x = X_dst - (2 * cx - 1) * scale_x
        source_global_y = Y_dst - (2 * cy - 1) * scale_y

    Positive displacement shifts content in the positive coordinate direction (right and down).
    Neutral channel value 0.5 produces zero displacement.
    Transparent map regions (alpha <= 1e-6) evaluate to neutral 0.5 for RGB/Luminance channels
    (and 0.0 for the ALPHA channel).

    Attributes:
        map: ImagePaint defining the displacement vector texture in object or world space (must have opacity == 1.0).
        scale_x: Maximum horizontal displacement magnitude in pixels (default: 0.0).
        scale_y: Maximum vertical displacement magnitude in pixels (default: 0.0).
        x_channel: Channel driving horizontal offset (default: DisplacementChannel.RED).
        y_channel: Channel driving vertical offset (default: DisplacementChannel.GREEN).
        interpolation: Resampling filter for entity content deformation (default: ImageInterpolation.LINEAR).
    """
    map: ImagePaint
    scale_x: float = 0.0
    scale_y: float = 0.0
    x_channel: DisplacementChannel = DisplacementChannel.RED
    y_channel: DisplacementChannel = DisplacementChannel.GREEN
    interpolation: ImageInterpolation = ImageInterpolation.LINEAR

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.map, ImagePaint):
            raise ValidationError(f"Displacement map must be an ImagePaint instance, got {type(self.map).__name__}")
        if float(self.map.opacity) != 1.0:
            raise ValidationError(
                f"Displacement map ImagePaint opacity must be 1.0 to preserve displacement data, got {self.map.opacity}"
            )

        self.scale_x = validate_real_number(self.scale_x, "scale_x")
        self.scale_y = validate_real_number(self.scale_y, "scale_y")

        self.x_channel = coerce_displacement_channel(self.x_channel)
        self.y_channel = coerce_displacement_channel(self.y_channel)

        if isinstance(self.interpolation, str):
            try:
                self.interpolation = ImageInterpolation(self.interpolation.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid ImageInterpolation: '{self.interpolation}'")
        elif not isinstance(self.interpolation, ImageInterpolation):
            raise ValidationError(
                f"interpolation must be an ImageInterpolation enum, got {type(self.interpolation).__name__}"
            )

        if self.interpolation == ImageInterpolation.AREA:
            raise ValidationError("ImageInterpolation.AREA is unsupported for displacement remap sampling")

    @property
    def effect_type(self) -> str:
        return "displacement_map"

    def _get_interpolation_support(self) -> float:
        """Return subpixel reconstruction radius for the content deformation filter."""
        if self.interpolation == ImageInterpolation.NEAREST:
            return 0.0
        elif self.interpolation == ImageInterpolation.LINEAR:
            return 1.0
        elif self.interpolation == ImageInterpolation.CUBIC:
            return 2.0
        elif self.interpolation == ImageInterpolation.LANCZOS:
            return 4.0
        return 1.0

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Visual bounds expand by maximum displacement offset plus interpolation footprint."""
        self._validate()
        interp_support = self._get_interpolation_support()
        pad_x = float(math.ceil(abs(self.scale_x))) + interp_support
        pad_y = float(math.ceil(abs(self.scale_y))) + interp_support
        return BoundingBox(
            input_bounds.left - pad_x,
            input_bounds.top - pad_y,
            input_bounds.width + 2.0 * pad_x,
            input_bounds.height + 2.0 * pad_y,
        )

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Support padding covering maximum source lookup distance plus interpolation tail."""
        self._validate()
        interp_support = self._get_interpolation_support()
        pad_x = float(math.ceil(abs(self.scale_x))) + interp_support
        pad_y = float(math.ceil(abs(self.scale_y))) + interp_support
        return (pad_x, pad_x, pad_y, pad_y)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        self._validate()
        return {
            "type": "displacement_map",
            "map": self.map.to_dict(),
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "x_channel": self.x_channel.value,
            "y_channel": self.y_channel.value,
            "interpolation": self.interpolation.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisplacementMapEffect:
        """Construct a DisplacementMapEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"DisplacementMapEffect data must be a dict, got {type(data).__name__}")
        if "map" not in data:
            raise ValidationError("DisplacementMapEffect requires 'map'")
        parsed_paint = paint_from_dict(data["map"])
        if not isinstance(parsed_paint, ImagePaint):
            raise ValidationError(f"Displacement map must deserialize to an ImagePaint, got {type(parsed_paint).__name__}")

        if "interpolation" in data:
            raw_interp = data["interpolation"]
            if isinstance(raw_interp, str):
                try:
                    interp = ImageInterpolation(raw_interp.strip().lower())
                except ValueError:
                    raise ValidationError(f"Invalid ImageInterpolation: '{raw_interp}'")
            elif isinstance(raw_interp, ImageInterpolation):
                interp = raw_interp
            else:
                raise ValidationError(f"Invalid ImageInterpolation: '{raw_interp}'")
        else:
            interp = ImageInterpolation.LINEAR

        return cls(
            map=parsed_paint,
            scale_x=data.get("scale_x", 0.0),
            scale_y=data.get("scale_y", 0.0),
            x_channel=coerce_displacement_channel(data.get("x_channel", DisplacementChannel.RED)),
            y_channel=coerce_displacement_channel(data.get("y_channel", DisplacementChannel.GREEN)),
            interpolation=interp,
        )
