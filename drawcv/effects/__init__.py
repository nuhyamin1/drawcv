"""Visual post-processing effects and compositing modifiers."""

from drawcv.core.exceptions import ValidationError
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect, clip_from_dict
from drawcv.effects.color import (
    BrightnessContrastEffect,
    ColorMatrixEffect,
    GrayscaleEffect,
    HueShiftEffect,
    SaturationEffect,
    SepiaEffect,
)
from drawcv.effects.effect import Effect
from drawcv.effects.glow import GlowEffect
from drawcv.effects.mask import Mask
from drawcv.effects.shadow import ShadowEffect


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
    "Effect",
    "GlowEffect",
    "GrayscaleEffect",
    "HueShiftEffect",
    "Mask",
    "SaturationEffect",
    "SepiaEffect",
    "ShadowEffect",
    "clip_from_dict",
    "effect_from_dict",
]

