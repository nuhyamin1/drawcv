"""Unit tests for StrokeStyle and FillStyle."""

import pytest

from drawcv.core.color import Color
from drawcv.core.enums import LineType
from drawcv.core.exceptions import ValidationError
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_stroke_style_defaults():
    s = StrokeStyle()
    assert s.color == Color.black()
    assert s.width == 1.0
    assert s.opacity == 1.0
    assert s.line_type == LineType.AA


def test_stroke_style_validation():
    with pytest.raises(ValidationError):
        StrokeStyle(width=0)

    with pytest.raises(ValidationError):
        StrokeStyle(width=-2.0)

    with pytest.raises(ValidationError):
        StrokeStyle(opacity=1.5)

    with pytest.raises(ValidationError):
        StrokeStyle(opacity=-0.1)

    with pytest.raises(ValidationError):
        StrokeStyle(color="red")  # type: ignore


def test_stroke_style_mutation():
    s = StrokeStyle()
    s.width = 5.0
    s.color = Color.red()
    assert s.width == 5.0
    assert s.color == Color.red()

    with pytest.raises(ValidationError):
        s.width = -1.0


def test_fill_style_defaults():
    f = FillStyle()
    assert f.enabled is True
    assert f.color == Color.white()
    assert f.opacity == 1.0


def test_fill_style_validation():
    with pytest.raises(ValidationError):
        FillStyle(opacity=-0.5)

    with pytest.raises(ValidationError):
        FillStyle(opacity=2.0)

    with pytest.raises(ValidationError):
        FillStyle(enabled="yes")  # type: ignore


def test_fill_style_mutation():
    f = FillStyle()
    f.opacity = 0.5
    f.color = Color.blue()
    f.enabled = False
    assert f.opacity == 0.5
    assert f.color == Color.blue()
    assert f.enabled is False

    with pytest.raises(ValidationError):
        f.opacity = 1.2
