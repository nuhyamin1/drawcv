"""Path.from_svg_path / Path.to_svg_path: authoring paths with SVG path data."""
import numpy as np
import pytest

from drawcv import CapStyle, Color, JoinStyle, FillRule, FillStyle, OpenCVRenderer, Path, Scene, StrokeStyle, ValidationError
from drawcv.shapes.path import Close, CubicTo, EllipticalArcTo, LineTo, MoveTo, QuadraticTo
from drawcv.svg_import import SVGImportError

SHAPES = "M10 10 h60 v40 H10 z m80 0 c10 -20 30 -20 40 0 s30 20 40 0 q10 -15 20 0 t20 0 a15 10 20 1 0 30 0"


def render(path):
    scene = Scene(260, 90, background=Color.white())
    scene.add(path)
    return OpenCVRenderer().render(scene).buffer


def test_all_command_families_parse_to_retained_commands():
    path = Path.from_svg_path(SHAPES)
    kinds = [type(c) for sp in path.subpaths for c in sp.commands]
    assert kinds == [MoveTo, LineTo, LineTo, LineTo, Close, MoveTo, CubicTo, CubicTo,
                     QuadraticTo, QuadraticTo, EllipticalArcTo]
    assert path.fill_rule == FillRule.NON_ZERO


def test_relative_and_absolute_forms_are_equivalent():
    relative = Path.from_svg_path("m10 20 l30 0 l0 30 l-30 0 z")
    absolute = Path.from_svg_path("M10,20 L40,20 L40,50 L10,50 Z")
    assert relative.to_svg_path() == absolute.to_svg_path()


def test_round_trip_is_stable_and_excludes_transform():
    path = Path.from_svg_path(SHAPES)
    data = path.to_svg_path()
    assert Path.from_svg_path(data).to_svg_path() == data
    path.transform.rotation = 45
    assert path.to_svg_path() == data


def test_style_keywords_and_fill_rule_names():
    stroke = StrokeStyle(color=Color(20, 40, 60), width=3)
    path = Path.from_svg_path("M0 0 L10 0 L10 10 Z", stroke=stroke, fill=FillStyle(color=Color(200, 0, 0)),
                              name="tri", fill_rule="evenodd")
    assert path.stroke is stroke and path.name == "tri" and path.fill_rule == FillRule.EVEN_ODD
    assert Path.from_svg_path("M0 0", fill_rule=FillRule.EVEN_ODD).fill_rule == FillRule.EVEN_ODD
    assert Path.from_svg_path("").subpaths == []


def test_invalid_input_is_rejected():
    with pytest.raises(SVGImportError):
        Path.from_svg_path("M10 10 L oops")
    with pytest.raises(ValidationError):
        Path.from_svg_path("M0 0", fill_rule="winding")
    with pytest.raises(ValidationError):
        Path.from_svg_path(None)


def test_renders_like_the_svg_importer():
    fill = FillStyle(color=Color(40, 120, 200))
    # Match SVG's stroke defaults so the comparison isolates path geometry.
    stroke = StrokeStyle(color=Color(0, 0, 0), width=2, space="object", cap_style=CapStyle.BUTT,
                         join_style=JoinStyle.MITER, miter_limit=4)
    ours = render(Path.from_svg_path(SHAPES, fill=fill, stroke=stroke))
    imported = Scene.from_svg(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="260" height="90">'
        f'<rect width="260" height="90" fill="#fff"/>'
        f'<path d="{SHAPES}" fill="rgb(40,120,200)" stroke="#000" stroke-width="2"/></svg>')
    assert np.array_equal(ours, OpenCVRenderer().render(imported).buffer)
