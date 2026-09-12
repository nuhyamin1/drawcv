"""Unit tests for Color representation, strict validation, and conversions."""

import pytest
from dataclasses import FrozenInstanceError

from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError


def test_color_construction_valid():
    c = Color(10, 20, 30, 0.5)
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 0.5


def test_color_immutability():
    c = Color.red()
    with pytest.raises(FrozenInstanceError):
        c.r = 100  # type: ignore


def test_color_strict_validation():
    # Out of range values must NOT be silently clamped
    with pytest.raises(ValidationError):
        Color(256, 0, 0)
    with pytest.raises(ValidationError):
        Color(-1, 0, 0)
    with pytest.raises(ValidationError):
        Color(0, 0, 0, 1.5)
    with pytest.raises(ValidationError):
        Color(0, 0, 0, -0.1)

    # Type validation
    with pytest.raises(ValidationError):
        Color("255", 0, 0)  # type: ignore
    with pytest.raises(ValidationError):
        Color(True, 0, 0)  # type: ignore
    with pytest.raises(ValidationError):
        Color(255, 0, 0, "1.0")  # type: ignore


def test_color_hex_parsing():
    # 6-digit hex
    c1 = Color.from_hex("#FF8000")
    assert c1.r == 255 and c1.g == 128 and c1.b == 0 and c1.a == 1.0

    # 8-digit hex with alpha
    c2 = Color.from_hex("00FF0080")
    assert c2.r == 0 and c2.g == 255 and c2.b == 0
    assert pytest.approx(c2.a, rel=1e-2) == 128 / 255.0

    # 3-digit short hex
    c3 = Color.from_hex("#F00")
    assert c3 == Color(255, 0, 0, 1.0)

    # 4-digit short hex with alpha
    c4 = Color.from_hex("#F00F")
    assert c4 == Color(255, 0, 0, 1.0)

    # Malformed hex strings
    with pytest.raises(ValidationError):
        Color.from_hex("#XYZ")
    with pytest.raises(ValidationError):
        Color.from_hex("#12345")
    with pytest.raises(ValidationError):
        Color.from_hex(123456)  # type: ignore


def test_color_conversions():
    c = Color(10, 20, 30, 0.5)
    assert c.to_rgb() == (10, 20, 30)
    assert c.to_rgba() == (10, 20, 30, 0.5)
    assert c.to_bgr() == (30, 20, 10)
    assert c.to_bgra() == (30, 20, 10, 0.5)
    assert c.to_hex() == "#0A141E"
    assert c.to_hex(include_alpha=True) == "#0A141E80"


def test_color_named_factories():
    assert Color.black() == Color(0, 0, 0, 1.0)
    assert Color.white() == Color(255, 255, 255, 1.0)
    assert Color.red() == Color(255, 0, 0, 1.0)
    assert Color.green() == Color(0, 255, 0, 1.0)
    assert Color.blue() == Color(0, 0, 255, 1.0)
    assert Color.transparent() == Color(0, 0, 0, 0.0)


def test_color_with_alpha():
    red = Color.red()
    trans_red = red.with_alpha(0.3)
    assert trans_red.r == 255 and trans_red.a == 0.3
    assert red.a == 1.0
