"""Visual post-processing effects and compositing modifiers."""

from drawcv.core.enums import DisplacementChannel, EdgeDetectionMethod
from drawcv.core.exceptions import ValidationError
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import (
    ClipPath,
    ClipRect,
    capture_clip_state,
    clip_from_dict,
    clip_to_dict,
    evaluate_clip_contours,
    evaluate_clip_coverage,
    get_clip_point_mapper,
    is_point_in_clip,
    restore_clip_state,
)
from drawcv.effects.color import (
    BrightnessContrastEffect,
    ColorMatrixEffect,
    GrayscaleEffect,
    HueShiftEffect,
    SaturationEffect,
    SepiaEffect,
)
from drawcv.effects.convolution import ConvolutionEffect
from drawcv.effects.displacement import DisplacementMapEffect
from drawcv.effects.edge import EdgeDetectionEffect
from drawcv.effects.effect import Effect
from drawcv.effects.emboss import EmbossEffect
from drawcv.effects.glow import GlowEffect
from drawcv.effects.mask import Mask
from drawcv.effects.noise import NoiseEffect
from drawcv.effects.shadow import ShadowEffect
from drawcv.effects.sharpen import SharpenEffect


def effect_from_dict(data: dict) -> Effect:
    """Deserialize an Effect from dictionary representation."""
    if not isinstance(data, dict):
        raise ValidationError(f"Effect data must be a dict, got {type(data).__name__}")
    eff_type = data.get("type")
    if not eff_type:
        raise ValidationError("Effect dictionary missing required 'type' attribute")
    from drawcv.effects.registry import get_effect_deserializer
    deserializer = get_effect_deserializer(str(eff_type))
    return deserializer(data)


__all__ = [
    "BlurEffect",
    "BrightnessContrastEffect",
    "ClipPath",
    "ClipRect",
    "ColorMatrixEffect",
    "ConvolutionEffect",
    "DisplacementChannel",
    "DisplacementMapEffect",
    "EdgeDetectionEffect",
    "EdgeDetectionMethod",
    "Effect",
    "EmbossEffect",
    "GlowEffect",
    "GrayscaleEffect",
    "HueShiftEffect",
    "Mask",
    "NoiseEffect",
    "SaturationEffect",
    "SepiaEffect",
    "ShadowEffect",
    "SharpenEffect",
    "capture_clip_state",
    "clip_from_dict",
    "clip_to_dict",
    "effect_from_dict",
    "evaluate_clip_contours",
    "evaluate_clip_coverage",
    "get_clip_point_mapper",
    "is_point_in_clip",
    "restore_clip_state",
]
