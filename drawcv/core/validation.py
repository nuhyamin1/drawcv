"""Small atomic mutation helpers for validated retained values."""

from __future__ import annotations
import copy
import math
from typing import Any
import numpy as np
from drawcv.core.exceptions import ValidationError


def validated_setattr(obj, name, value):
    """Validate a candidate on a shallow copy before touching live state."""
    if getattr(obj, "_initialized", False) and not name.startswith("_"):
        candidate = copy.copy(obj)
        object.__setattr__(candidate, "_initialized", False)
        object.__setattr__(candidate, name, value)
        candidate._validate()
        value = getattr(candidate, name, value)
    object.__setattr__(obj, name, value)


def atomic_transforms(method):
    """Rollback compound transform operations, preserving Transform identity."""
    from functools import wraps
    from inspect import signature

    first_parameter = next(iter(signature(method).parameters))

    @wraps(method)
    def wrapped(*args, **kwargs):
        target = args[0] if args else kwargs.get(first_parameter)
        if hasattr(target, "_get_transform_roots"):
            objects = target._get_transform_roots()
        elif isinstance(target, (list, tuple)):
            objects = target
        else:
            objects = [target]
        states = [(obj.transform, obj.transform.__dict__.copy())
                  for obj in objects if hasattr(obj, "transform")]
        try:
            return method(*args, **kwargs)
        except Exception:
            for transform, state in states:
                transform.__dict__.clear()
                transform.__dict__.update(state)
            raise
    return wrapped


def validate_real_number(val: Any, name: str) -> float:
    """Validate that val is a finite real numeric value, rejecting booleans and complex numbers."""
    import math
    import numpy as np
    from drawcv.core.exceptions import ValidationError

    if isinstance(val, (bool, complex, np.complexfloating)):
        raise ValidationError(f"'{name}' must be a real numeric value, got {type(val).__name__}")
    if not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValidationError(f"'{name}' must be numeric, got {type(val).__name__}")
    float_val = float(val)
    if math.isnan(float_val) or math.isinf(float_val):
        raise ValidationError(f"'{name}' must be finite, got {val}")
    return float_val


def validate_integer(val: Any, name: str) -> int:
    """Validate that val is an integer scalar, rejecting booleans, floats, and complex numbers."""
    import numpy as np
    from drawcv.core.exceptions import ValidationError

    if isinstance(val, (bool, complex, np.complexfloating)):
        raise ValidationError(f"'{name}' must be an integer, got {type(val).__name__}")
    if not isinstance(val, (int, np.integer)):
        raise ValidationError(f"'{name}' must be an integer, got {type(val).__name__}")
    return int(val)
