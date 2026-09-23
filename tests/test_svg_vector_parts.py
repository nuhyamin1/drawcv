"""Arrows and variable-width freehand strokes export as native SVG vectors."""
import itertools

import numpy as np
import pytest

from drawcv import (Arrow, ArrowHeadStyle, Color, FillStyle, FreehandStroke, Label, LinearGradient, GradientStop,
                    OpenCVRenderer, Point, SVGImporter, Scene, StrokePoint, StrokeStyle, Transform)
from drawcv.svg import _vector_parts


def alpha(*drawables):
    scene = Scene(260, 200, background=Color(0, 0, 0, 0))
    for drawable in drawables:
        scene.add(drawable)
    return OpenCVRenderer().render(scene, alpha=True).buffer[..., 3].astype(int)


def iou(a, b):
    a, b = a > 127, b > 127
    return (a & b).sum() / (a | b).sum()


def arrow(head, space="screen", dash=(), transform="none", **kwargs):
    shape = Arrow(start=Point(30, 60), end=Point(200, 120), head_style=head, head_length=30, head_width=26,
                  fill=FillStyle(color=Color(230, 120, 40)),
                  stroke=StrokeStyle(color=Color(20, 60, 120, 0.8), width=5, space=space, dash_array=dash), **kwargs)
    if transform == "rotate":
        shape.transform.rotation = 25
    elif transform == "scale":
        shape.transform.scale_x = shape.transform.scale_y = 1.3
    elif transform == "shear":
        shape.transform = Transform.from_matrix(np.array([[1.2, .3, -10], [0, .8, 20], [0, 0, 1.]]))
    return shape


def pressure_stroke(**kwargs):
    points = [StrokePoint(20 + i * 6, 100 + 30 * np.sin(i / 3), pressure=0.3 + 0.7 * abs(np.sin(i / 5)))
              for i in range(35)]
    return FreehandStroke(points=points, stroke=StrokeStyle(color=Color(40, 40, 160), width=12),
                          variable_width=True, **kwargs)


@pytest.mark.parametrize("head,space,dash,transform", itertools.product(
    [ArrowHeadStyle.TRIANGLE, ArrowHeadStyle.DIAMOND, ArrowHeadStyle.OPEN],
    ["screen", "object"], [(), (10, 6)], ["none", "rotate", "shear"]))
def test_arrow_parts_render_like_the_arrow(head, space, dash, transform):
    shape = arrow(head, space, dash, transform)
    assert np.abs(alpha(shape) - alpha(*_vector_parts(shape))).max() <= 2


@pytest.mark.parametrize("transform", ["none", "rotate", "scale"])
def test_circle_head_parts_match_up_to_the_raster_integer_center(transform):
    shape = arrow(ArrowHeadStyle.CIRCLE, transform=transform)
    assert iou(alpha(shape), alpha(*_vector_parts(shape))) > 0.98


def test_circle_head_under_shear_keeps_reported_fallback():
    scene = Scene(260, 200)
    scene.add(arrow(ArrowHeadStyle.CIRCLE, transform="shear"))
    assert "circle arrowhead" in scene.export_svg().fallbacks[0].reason


def test_pressure_stroke_outline_matches_the_stroke():
    stroke = pressure_stroke()
    # The raster stroke is ~1px wider than its exact outline, costing a few percent.
    assert iou(alpha(stroke), alpha(*_vector_parts(stroke))) > 0.93


@pytest.mark.parametrize("make", [lambda: arrow(ArrowHeadStyle.TRIANGLE, dash=(10, 6), transform="rotate"),
                                  lambda: arrow(ArrowHeadStyle.CIRCLE), pressure_stroke])
def test_strict_export_is_native_deterministic_and_reimports(make):
    scene = Scene(260, 200, background=Color(0, 0, 0, 0))
    shape = make()
    shape.opacity = 0.6
    scene.add(shape)
    svg = scene.to_svg(strict=True)
    assert "<image" not in svg and svg == scene.to_svg(strict=True)
    reimported = OpenCVRenderer().render(SVGImporter().parse(svg).scene, alpha=True).buffer[..., 3]
    assert iou(reimported.astype(int), alpha(shape)) > 0.85
    assert abs(int(reimported.max()) - int(alpha(shape).max())) <= 2  # opacity applied once


def test_gradient_paint_labels_and_paint_order_carry_over():
    stops = (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))
    shape = arrow(ArrowHeadStyle.TRIANGLE, paint_order="stroke")
    shape.stroke.paint = LinearGradient(Point(30, 0), Point(200, 0), stops)
    shape.label = Label("flow", placement="above", fonts=None)
    stroke = pressure_stroke()
    stroke.stroke.paint = LinearGradient(Point(20, 0), Point(220, 0), stops)
    scene = Scene(260, 200)
    scene.add(shape)
    scene.add(stroke)
    result = scene.export_svg()
    assert [f.entity for f in result.fallbacks] == [f"{shape.id}-label"]  # only the Hershey label rasterizes
    assert result.svg.count("<linearGradient") >= 3
    assert 'paint-order="stroke fill markers"' in result.svg


def test_zero_width_arrow_exports_nothing():
    shape = arrow(ArrowHeadStyle.TRIANGLE)
    shape.stroke = None
    assert _vector_parts(shape) == []
    scene = Scene(100, 100)
    scene.add(shape)
    assert "<path" not in scene.to_svg(strict=True)
