"""JSON parser and document envelope validator for DrawCV."""

from __future__ import annotations
import json
from typing import Any

from drawcv.core.exceptions import InvalidFormatError, SerializationError
from drawcv.serialization.registry import SchemaMigrator


def from_json(text: str, migrate: bool = True) -> dict[str, Any]:
    """Parse JSON text into a validated DrawCV document payload.
    
    Args:
        text: Valid JSON string containing a DrawCV document envelope.
        migrate: Whether to execute forward schema migrations if needed.
        
    Returns:
        Migrated plain dictionary matching current schema version.
    """
    if not isinstance(text, str):
        raise SerializationError(f"Expected JSON string, got {type(text).__name__}")
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise InvalidFormatError(f"Malformed JSON document: {err}") from err

    if not isinstance(data, dict):
        raise InvalidFormatError(f"Root JSON value must be an object, got {type(data).__name__}")

    if migrate:
        return SchemaMigrator.migrate(data)
    else:
        SchemaMigrator.validate_envelope(data)
        return data
