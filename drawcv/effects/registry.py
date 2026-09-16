"""Internal registry for serializable effect types and deserializers."""

from __future__ import annotations
from typing import Any, Callable, Type

from drawcv.core.exceptions import SerializationError, ValidationError
from drawcv.effects.effect import Effect

_EFFECT_REGISTRY: dict[str, type[Effect]] = {}
_DESERIALIZER_REGISTRY: dict[str, Callable[[dict[str, Any]], Effect]] = {}
_BUILTIN_EFFECTS_REGISTERED: bool = False


def register_effect_type(
    type_name: str,
    cls: type[Effect],
    deserializer: Callable[[dict[str, Any]], Effect] | None = None,
) -> None:
    """Register an effect type identifier with its class and deserializer.

    Args:
        type_name: Canonical string identifier (e.g. 'blur', 'shadow', 'glow').
        cls: Concrete Effect subclass.
        deserializer: Optional custom deserializer callable. If omitted, cls.from_dict is used.
    """
    if not isinstance(type_name, str) or not type_name.strip():
        raise SerializationError("Effect type name must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, Effect):
        raise SerializationError(f"Class {cls} must be a subclass of Effect")

    clean_name = type_name.strip().lower()
    resolved_deserializer = deserializer if deserializer is not None else getattr(cls, "from_dict", None)
    if resolved_deserializer is None or not callable(resolved_deserializer):
        raise SerializationError(f"Deserializer for '{clean_name}' must be callable")

    if clean_name in _EFFECT_REGISTRY:
        existing_cls = _EFFECT_REGISTRY[clean_name]
        if existing_cls is cls:
            # Idempotent re-registration of the exact same class
            return
        raise SerializationError(
            f"Duplicate effect type registration: '{clean_name}' is already registered to {existing_cls}"
        )

    _EFFECT_REGISTRY[clean_name] = cls
    _DESERIALIZER_REGISTRY[clean_name] = resolved_deserializer


def register_builtin_effects() -> None:
    """Register all built-in standard DrawCV effect types idempotently."""
    global _BUILTIN_EFFECTS_REGISTERED
    if _BUILTIN_EFFECTS_REGISTERED:
        return
    _BUILTIN_EFFECTS_REGISTERED = True

    from drawcv.effects.blur import BlurEffect
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
    from drawcv.effects.emboss import EmbossEffect
    from drawcv.effects.glow import GlowEffect
    from drawcv.effects.noise import NoiseEffect
    from drawcv.effects.shadow import ShadowEffect
    from drawcv.effects.sharpen import SharpenEffect

    register_effect_type("blur", BlurEffect)
    register_effect_type("glow", GlowEffect)
    register_effect_type("shadow", ShadowEffect)
    register_effect_type("brightness_contrast", BrightnessContrastEffect)
    register_effect_type("saturation", SaturationEffect)
    register_effect_type("hue_shift", HueShiftEffect)
    register_effect_type("grayscale", GrayscaleEffect)
    register_effect_type("sepia", SepiaEffect)
    register_effect_type("color_matrix", ColorMatrixEffect)
    register_effect_type("convolution", ConvolutionEffect)
    register_effect_type("sharpen", SharpenEffect)
    register_effect_type("emboss", EmbossEffect)
    register_effect_type("edge_detection", EdgeDetectionEffect)
    register_effect_type("noise", NoiseEffect)
    register_effect_type("displacement_map", DisplacementMapEffect)



def get_effect_class(type_name: str) -> type[Effect]:
    """Retrieve the registered Effect class for the given type identifier."""
    register_builtin_effects()
    clean_name = str(type_name).strip().lower()
    if clean_name not in _EFFECT_REGISTRY:
        raise ValidationError(f"Unknown effect type: '{type_name}'")
    return _EFFECT_REGISTRY[clean_name]


def get_effect_deserializer(type_name: str) -> Callable[[dict[str, Any]], Effect]:
    """Retrieve the deserializer function for the given type identifier."""
    register_builtin_effects()
    clean_name = str(type_name).strip().lower()
    if clean_name not in _DESERIALIZER_REGISTRY:
        raise ValidationError(f"Unknown effect type: '{type_name}'")
    return _DESERIALIZER_REGISTRY[clean_name]


def is_effect_type_registered(type_name: str) -> bool:
    """Check if an effect type identifier is registered."""
    register_builtin_effects()
    return str(type_name).strip().lower() in _EFFECT_REGISTRY
