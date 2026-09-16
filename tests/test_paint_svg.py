"""Tests for SVG export of advanced paints: native mapping vs raster fallback."""

import numpy as np
import pytest

from drawcv import (
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    ImageInterpolation,
    ImagePaint,
    Layer,
    LinearGradient,
    Line,
    Point,
    RadialGradient,
    Rectangle,
    RenderError,
    Scene,
    StrokeStyle,
    SVGExporter,
    Transform,
)

STOPS = (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue()))


def make_scene_with(drawable):
    scene = Scene(200, 200, background=Color(255, 255, 255, 1.0))
    scene.add(drawable)
    return scene


def test_svg_linear_gradient_spread_modes():
    grad = LinearGradient(Point(0, 0), Point(100, 0), STOPS, spread="reflect")
    rect = Rectangle(position=Point(10, 10), width=80, height=80, fill=FillStyle(paint=grad))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert 'spreadMethod="reflect"' in export.svg
    assert '<linearGradient' in export.svg


def test_svg_radial_gradient_spread_modes():
    grad = RadialGradient(Point(50, 50), 30.0, STOPS, spread="repeat")
    rect = Rectangle(position=Point(10, 10), width=80, height=80, fill=FillStyle(paint=grad))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert 'spreadMethod="repeat"' in export.svg
    assert '<radialGradient' in export.svg


def test_svg_gradient_stroke_native():
    grad = LinearGradient(Point(0, 0), Point(50, 50), STOPS)
    line = Line(start=Point(10, 10), end=Point(90, 90), stroke=StrokeStyle(paint=grad, width=4.0))
    scene = make_scene_with(line)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert 'stroke="url(#drawcv-' in export.svg
    assert '<linearGradient' in export.svg


def test_svg_image_paint_repeat_native():
    img = np.zeros((8, 8, 4), dtype=np.uint8)
    paint = ImagePaint(image=img, repeat="repeat", scale=(2.0, 2.0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=paint))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert '<pattern' in export.svg
    assert '<image' in export.svg
    assert 'data:image/png;base64,' in export.svg


def test_svg_conic_gradient_fallback():
    conic = ConicGradient(Point(50, 50), STOPS)
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=conic))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 1
    assert export.fallbacks[0].reason == "conic gradient paint"
    assert "data-drawcv-raster" in export.svg


def test_svg_image_paint_non_repeat_fallbacks():
    img = np.zeros((4, 4, 4), dtype=np.uint8)

    for mode in ("reflect", "pad", "none"):
        paint = ImagePaint(image=img, repeat=mode)
        rect = Rectangle(position=Point(0, 0), width=50, height=50, fill=FillStyle(paint=paint))
        scene = make_scene_with(rect)

        export = SVGExporter().render(scene)
        assert len(export.fallbacks) == 1
        assert export.fallbacks[0].reason == f"image paint {mode} repeat"


def test_svg_strict_mode_rejects_fallback_paint():
    conic = ConicGradient(Point(50, 50), STOPS)
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=conic))
    scene = make_scene_with(rect)

    with pytest.raises(RenderError) as exc_info:
        SVGExporter(strict=True).render(scene)
    assert "conic gradient paint" in str(exc_info.value)


def test_svg_image_paint_interpolation_modes():
    img = np.zeros((8, 8, 4), dtype=np.uint8)

    # 1. LINEAR is native
    paint_linear = ImagePaint(image=img, repeat="repeat", interpolation=ImageInterpolation.LINEAR)
    scene_linear = make_scene_with(Rectangle(position=Point(0, 0), width=50, height=50, fill=FillStyle(paint=paint_linear)))
    export_linear = SVGExporter().render(scene_linear)
    assert len(export_linear.fallbacks) == 0
    assert "<pattern" in export_linear.svg

    # 2. Unsupported interpolations fall back
    for interp in (ImageInterpolation.NEAREST, ImageInterpolation.CUBIC, ImageInterpolation.LANCZOS):
        paint = ImagePaint(image=img, repeat="repeat", interpolation=interp)
        scene = make_scene_with(Rectangle(position=Point(0, 0), width=50, height=50, fill=FillStyle(paint=paint)))
        export = SVGExporter().render(scene)
        assert len(export.fallbacks) == 1
        assert export.fallbacks[0].reason == f"image paint {interp.value} interpolation"
        assert "data-drawcv-raster" in export.svg


def test_svg_strict_mode_rejects_unsupported_interpolation():
    img = np.zeros((8, 8, 4), dtype=np.uint8)
    paint = ImagePaint(image=img, repeat="repeat", interpolation=ImageInterpolation.NEAREST)
    scene = make_scene_with(Rectangle(position=Point(0, 0), width=50, height=50, fill=FillStyle(paint=paint)))

    with pytest.raises(RenderError) as exc_info:
        SVGExporter(strict=True).render(scene)
    assert "image paint nearest interpolation" in str(exc_info.value)


def test_svg_gradient_transform_native():
    grad = LinearGradient(
        Point(0, 0),
        Point(100, 0),
        STOPS,
        transform=Transform(rotation=45.0, pivot=Point(50, 0)),
    )
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=grad))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert "gradientTransform=\"matrix(" in export.svg


def test_svg_pattern_transform_and_composition():
    img = np.zeros((8, 8, 4), dtype=np.uint8)
    paint = ImagePaint(
        image=img,
        origin=Point(15, 25),
        scale=(2.0, 3.0),
        repeat="repeat",
        interpolation=ImageInterpolation.LINEAR,
        transform=Transform(rotation=30.0),
    )
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=paint))
    scene = make_scene_with(rect)

    export = SVGExporter().render(scene)
    assert len(export.fallbacks) == 0
    assert "<pattern" in export.svg
    assert "x=\"15\"" in export.svg
    assert "y=\"25\"" in export.svg
    assert "width=\"16\"" in export.svg   # 8 * 2.0
    assert "height=\"24\"" in export.svg  # 8 * 3.0
    assert "patternTransform=\"matrix(" in export.svg
    assert "<image" in export.svg
