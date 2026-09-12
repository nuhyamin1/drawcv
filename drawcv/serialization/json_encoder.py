"""Strict JSON serializer for DrawCV documents."""

from __future__ import annotations
import json
import math
from typing import Any

from drawcv.core.exceptions import SerializationError


def _validate_json_primitives(obj: Any, path: str = "root") -> None:
    """Recursively verify that an object contains strictly standard JSON-compliant primitives."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise SerializationError(f"Non-finite float value '{obj}' at '{path}' cannot be serialized to standard JSON")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise SerializationError(f"Dictionary key at '{path}' must be a string, got {type(k).__name__}")
            _validate_json_primitives(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _validate_json_primitives(item, f"{path}[{idx}]")
        return

    raise SerializationError(
        f"Object of type '{type(obj).__name__}' at '{path}' is not standard JSON-serializable."
    )


def to_json(data: dict[str, Any], indent: int = 2) -> str:
    """Serialize a plain dictionary to strict, canonical JSON text.
    
    Guarantees:
    - allow_nan=False (strictly rejects NaN, Infinity).
    - sort_keys=True (deterministic bit-for-bit canonical formatting).
    - Plain JSON primitive data only (raises SerializationError on unsupported types).
    """
    _validate_json_primitives(data)
    try:
        return json.dumps(data, indent=indent, allow_nan=False, sort_keys=True)
    except (ValueError, TypeError) as err:
        raise SerializationError(f"JSON serialization error: {err}") from err
