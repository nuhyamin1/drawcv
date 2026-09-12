"""Standard easing curves and registry for DrawCV animation."""

from __future__ import annotations
import math
from typing import Callable

from drawcv.core.exceptions import ValidationError


def linear(t: float) -> float:
    """Linear progression."""
    return float(t)


# -----------------------------------------------------------------------------
# Quadratic
# -----------------------------------------------------------------------------

def ease_in_quad(t: float) -> float:
    return float(t * t)


def ease_out_quad(t: float) -> float:
    return float(t * (2.0 - t))


def ease_in_out_quad(t: float) -> float:
    if t < 0.5:
        return float(2.0 * t * t)
    return float(-1.0 + (4.0 - 2.0 * t) * t)


# -----------------------------------------------------------------------------
# Cubic
# -----------------------------------------------------------------------------

def ease_in_cubic(t: float) -> float:
    return float(t * t * t)


def ease_out_cubic(t: float) -> float:
    t1 = t - 1.0
    return float(t1 * t1 * t1 + 1.0)


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return float(4.0 * t * t * t)
    t1 = 2.0 * t - 2.0
    return float(0.5 * t1 * t1 * t1 + 1.0)


# -----------------------------------------------------------------------------
# Sine
# -----------------------------------------------------------------------------

def ease_in_sine(t: float) -> float:
    return float(1.0 - math.cos((t * math.pi) / 2.0))


def ease_out_sine(t: float) -> float:
    return float(math.sin((t * math.pi) / 2.0))


def ease_in_out_sine(t: float) -> float:
    return float(-(math.cos(math.pi * t) - 1.0) / 2.0)


# -----------------------------------------------------------------------------
# Exponential
# -----------------------------------------------------------------------------

def ease_in_expo(t: float) -> float:
    if t <= 0.0:
        return 0.0
    return float(2.0 ** (10.0 * (t - 1.0)))


def ease_out_expo(t: float) -> float:
    if t >= 1.0:
        return 1.0
    return float(1.0 - 2.0 ** (-10.0 * t))


def ease_in_out_expo(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if t < 0.5:
        return float((2.0 ** (20.0 * t - 10.0)) / 2.0)
    return float((2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0)


# -----------------------------------------------------------------------------
# Circular
# -----------------------------------------------------------------------------

def ease_in_circ(t: float) -> float:
    return float(1.0 - math.sqrt(max(0.0, 1.0 - t * t)))


def ease_out_circ(t: float) -> float:
    t1 = t - 1.0
    return float(math.sqrt(max(0.0, 1.0 - t1 * t1)))


def ease_in_out_circ(t: float) -> float:
    if t < 0.5:
        return float((1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * t * t))) / 2.0)
    t1 = -2.0 * t + 2.0
    return float((math.sqrt(max(0.0, 1.0 - t1 * t1)) + 1.0) / 2.0)


# -----------------------------------------------------------------------------
# Bounce
# -----------------------------------------------------------------------------

def ease_out_bounce(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75

    if t < 1.0 / d1:
        return float(n1 * t * t)
    elif t < 2.0 / d1:
        t_sub = t - 1.5 / d1
        return float(n1 * t_sub * t_sub + 0.75)
    elif t < 2.5 / d1:
        t_sub = t - 2.25 / d1
        return float(n1 * t_sub * t_sub + 0.9375)
    else:
        t_sub = t - 2.625 / d1
        return float(n1 * t_sub * t_sub + 0.984375)


def ease_in_bounce(t: float) -> float:
    return float(1.0 - ease_out_bounce(1.0 - t))


def ease_in_out_bounce(t: float) -> float:
    if t < 0.5:
        return float((1.0 - ease_out_bounce(1.0 - 2.0 * t)) / 2.0)
    return float((1.0 + ease_out_bounce(2.0 * t - 1.0)) / 2.0)


# -----------------------------------------------------------------------------
# Elastic
# -----------------------------------------------------------------------------

def ease_in_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return float(-(2.0 ** (10.0 * (t - 1.0))) * math.sin((t * 10.0 - 10.75) * c4))


def ease_out_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return float(2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0)


def ease_in_out_elastic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c5 = (2.0 * math.pi) / 4.5
    if t < 0.5:
        return float(-(2.0 ** (20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0)
    return float((2.0 ** (-20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0)


# Aliases
ease_in = ease_in_quad
ease_out = ease_out_quad
ease_in_out = ease_in_out_quad

EASING_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_sine": ease_in_sine,
    "ease_out_sine": ease_out_sine,
    "ease_in_out_sine": ease_in_out_sine,
    "ease_in_expo": ease_in_expo,
    "ease_out_expo": ease_out_expo,
    "ease_in_out_expo": ease_in_out_expo,
    "ease_in_circ": ease_in_circ,
    "ease_out_circ": ease_out_circ,
    "ease_in_out_circ": ease_in_out_circ,
    "ease_in_bounce": ease_in_bounce,
    "ease_out_bounce": ease_out_bounce,
    "ease_in_out_bounce": ease_in_out_bounce,
    "ease_in_elastic": ease_in_elastic,
    "ease_out_elastic": ease_out_elastic,
    "ease_in_out_elastic": ease_in_out_elastic,
}


def get_easing(name_or_func: str | Callable[[float], float]) -> Callable[[float], float]:
    """Resolve an easing function from a canonical name or callable."""
    if callable(name_or_func):
        return name_or_func
    if not isinstance(name_or_func, str):
        raise ValidationError(f"Easing must be string identifier or callable, got {type(name_or_func).__name__}")
    
    clean_name = name_or_func.strip().lower().replace("-", "_")
    if clean_name in EASING_FUNCTIONS:
        return EASING_FUNCTIONS[clean_name]
    raise ValidationError(
        f"Unknown easing curve '{name_or_func}'. Supported: {', '.join(sorted(EASING_FUNCTIONS.keys()))}"
    )
