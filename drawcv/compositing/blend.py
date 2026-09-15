"""Pure RGB blend functions operating on normalized float32 color channels."""

from __future__ import annotations
import numpy as np

from drawcv.core.enums import BlendMode


def _blend_normal(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return cs.copy()


def _blend_multiply(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return cb * cs


def _blend_screen(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return cb + cs - cb * cs


def _blend_overlay(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.where(cb <= 0.5, 2.0 * cb * cs, 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))


def _blend_darken(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.minimum(cb, cs)


def _blend_lighten(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.maximum(cb, cs)


def _blend_color_dodge(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    denom = 1.0 - cs
    safe_denom = np.where(denom <= 0.0, 1.0, denom)
    ratio = np.minimum(1.0, cb / safe_denom)
    return np.where(cb <= 0.0, 0.0, np.where(cs >= 1.0, 1.0, ratio))


def _blend_color_burn(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    safe_cs = np.where(cs <= 0.0, 1.0, cs)
    ratio = np.minimum(1.0, (1.0 - cb) / safe_cs)
    return np.where(cb >= 1.0, 1.0, np.where(cs <= 0.0, 0.0, 1.0 - ratio))


def _blend_hard_light(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.where(cs <= 0.5, 2.0 * cb * cs, 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))


def _blend_soft_light(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    d = np.where(cb <= 0.25, ((16.0 * cb - 12.0) * cb + 4.0) * cb, np.sqrt(np.maximum(0.0, cb)))
    lower = cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    upper = cb + (2.0 * cs - 1.0) * (d - cb)
    return np.where(cs <= 0.5, lower, upper)


def _blend_difference(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return np.abs(cb - cs)


def _blend_exclusion(cb: np.ndarray, cs: np.ndarray) -> np.ndarray:
    return cb + cs - 2.0 * cb * cs


_BLEND_FUNCTIONS = {
    BlendMode.NORMAL: _blend_normal,
    BlendMode.MULTIPLY: _blend_multiply,
    BlendMode.SCREEN: _blend_screen,
    BlendMode.OVERLAY: _blend_overlay,
    BlendMode.DARKEN: _blend_darken,
    BlendMode.LIGHTEN: _blend_lighten,
    BlendMode.COLOR_DODGE: _blend_color_dodge,
    BlendMode.COLOR_BURN: _blend_color_burn,
    BlendMode.HARD_LIGHT: _blend_hard_light,
    BlendMode.SOFT_LIGHT: _blend_soft_light,
    BlendMode.DIFFERENCE: _blend_difference,
    BlendMode.EXCLUSION: _blend_exclusion,
}


def blend_rgb(cb: np.ndarray, cs: np.ndarray, mode: BlendMode) -> np.ndarray:
    """Evaluate pure RGB blend function on normalized [0, 1] color arrays."""
    fn = _BLEND_FUNCTIONS.get(mode)
    if fn is None:
        raise ValueError(f"Unsupported blend mode: {mode}")
    result = fn(cb, cs)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
