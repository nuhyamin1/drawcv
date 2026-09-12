"""Visual post-processing effects and compositing modifiers."""

from drawcv.core.exceptions import ValidationError
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect, clip_from_dict
from drawcv.effects.effect import Effect
from drawcv.effects.mask import Mask
from drawcv.effects.shadow import ShadowEffect


def effect_from_dict(data: dict) -> Effect:
    """Deserialize an Effect from dictionary representation."""
    eff_type = data.get("type")
    if eff_type == "blur":
        return BlurEffect.from_dict(data)
    elif eff_type == "shadow":
        return ShadowEffect.from_dict(data)
    raise ValidationError(f"Unknown effect type: '{eff_type}'")


__all__ = [
    "BlurEffect",
    "ClipPath",
    "ClipRect",
    "Effect",
    "Mask",
    "ShadowEffect",
    "clip_from_dict",
    "effect_from_dict",
]

