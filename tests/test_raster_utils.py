"""Tests for shared raster image validation and PNG base64 serialization."""

import numpy as np
import pytest

from drawcv.core.exceptions import SerializationError, ValidationError
from drawcv.core.raster import (
    decode_raster_png,
    encode_raster_png,
    validate_raster_image,
)


def test_validate_raster_image_valid():
    gray = np.zeros((10, 10), dtype=np.uint8)
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    validate_raster_image(gray)
    validate_raster_image(bgr)
    validate_raster_image(bgra)


@pytest.mark.parametrize("invalid_img", [
    "not_an_array",
    np.zeros((10, 10), dtype=np.float32),
    np.zeros((0, 10), dtype=np.uint8),
    np.zeros((10, 0), dtype=np.uint8),
    np.zeros((10, 10, 2), dtype=np.uint8),
    np.zeros((10, 10, 5), dtype=np.uint8),
    np.zeros((10, 10, 3, 1), dtype=np.uint8),
])
def test_validate_raster_image_invalid(invalid_img):
    with pytest.raises(ValidationError):
        validate_raster_image(invalid_img)


def test_encode_and_decode_raster_png_roundtrip():
    # Deterministic RGB pattern with alpha
    img = np.zeros((20, 30, 4), dtype=np.uint8)
    img[:, :, 0] = 50   # B
    img[:, :, 1] = 100  # G
    img[:, :, 2] = 200  # R
    img[:, :, 3] = 180  # A

    payload = encode_raster_png(img)
    assert payload["encoding"] == "png_base64"
    assert payload["dtype"] == "uint8"
    assert payload["shape"] == [20, 30, 4]
    assert isinstance(payload["data"], str)

    decoded = decode_raster_png(payload)
    assert np.array_equal(decoded, img)


def test_decode_raster_png_shape_mismatch():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    payload = encode_raster_png(img)
    payload["shape"] = [20, 20, 3]  # corrupted expected shape
    with pytest.raises(SerializationError):
        decode_raster_png(payload)


def test_decode_raster_png_corrupted_data():
    payload = {
        "encoding": "png_base64",
        "dtype": "uint8",
        "shape": [10, 10, 3],
        "data": "not_valid_png_base64_data",
    }
    with pytest.raises(SerializationError):
        decode_raster_png(payload)


@pytest.mark.parametrize("shape", [
    (12, 16),        # 2D grayscale
    (12, 16, 1),     # 3D singleton channel
    (12, 16, 3),     # 3D BGR
    (12, 16, 4),     # 3D BGRA
])
def test_raster_channel_shapes_roundtrip(shape):
    # Deterministic test pattern
    np.random.seed(42)
    img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    payload = encode_raster_png(img)
    assert payload["shape"] == list(shape)
    assert payload["dtype"] == "uint8"

    decoded = decode_raster_png(payload)
    assert decoded.shape == shape
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, img)


def test_decode_raster_png_invalid_dtype():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    payload = encode_raster_png(img)
    payload["dtype"] = "float32"
    with pytest.raises(SerializationError, match="Unsupported image dtype"):
        decode_raster_png(payload)
