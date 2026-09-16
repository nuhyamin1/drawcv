"""Shared raster image validation and lossless PNG base64 serialization utilities."""

from __future__ import annotations
import base64
from typing import Any
import cv2
import numpy as np

from drawcv.core.exceptions import SerializationError, ValidationError


def validate_raster_image(image: np.ndarray) -> None:
    """Validate that image is a non-empty uint8 NumPy array of 2D or 3D with 1, 3, or 4 channels."""
    if not isinstance(image, np.ndarray):
        raise ValidationError(f"Image must be a NumPy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValidationError(f"Image dtype must be uint8, got {image.dtype}")
    if image.ndim == 2:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValidationError("Image dimensions must be non-empty")
    elif image.ndim == 3:
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValidationError("Image dimensions must be non-empty")
        if image.shape[2] not in (1, 3, 4):
            raise ValidationError(f"Image channels must be 1, 3 (BGR), or 4 (BGRA), got {image.shape[2]}")
    else:
        raise ValidationError(f"Image array must be 2D or 3D, got {image.ndim}D")


def encode_raster_png(image: np.ndarray, error_context: str = "image") -> dict[str, Any]:
    """Losslessly encode a raster image array to a canonical PNG base64 payload dictionary."""
    validate_raster_image(image)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise SerializationError(f"Failed to encode {error_context} buffer to PNG")
    data_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "encoding": "png_base64",
        "dtype": "uint8",
        "shape": list(image.shape),
        "data": data_b64,
    }


def decode_raster_png(payload: dict[str, Any], error_context: str = "image") -> np.ndarray:
    """Decode a canonical PNG base64 payload dictionary into a validated uint8 raster array."""
    if not isinstance(payload, dict):
        raise SerializationError(f"{error_context} 'image' must be a structured raster dictionary")
    if payload.get("encoding") != "png_base64":
        raise SerializationError(f"Unsupported image encoding '{payload.get('encoding')}'")
    if payload.get("dtype") != "uint8":
        raise SerializationError(f"Unsupported image dtype '{payload.get('dtype')}'; expected 'uint8'")
    if "data" not in payload:
        raise SerializationError(f"Missing 'data' in {error_context} raster dictionary")

    try:
        raw_bytes = base64.b64decode(payload["data"])
    except Exception as e:
        raise SerializationError(f"Failed to base64-decode {error_context} payload: {e}") from e

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise SerializationError(f"Failed to decode {error_context} PNG buffer")

    expected_shape = tuple(payload.get("shape", []))
    if expected_shape and len(expected_shape) == 3 and expected_shape[2] == 1 and decoded.ndim == 2:
        decoded = decoded[:, :, None]

    if expected_shape and decoded.shape != expected_shape:
        raise SerializationError(
            f"Decoded image shape {decoded.shape} does not match expected shape {expected_shape}"
        )

    validate_raster_image(decoded)
    return decoded
