"""Retained paint order: normalization, rendering, persistence and SVG interchange."""
import numpy as np
import pytest

from drawcv import (Arrow, ArrowHeadStyle, Circle, Color, FillStyle, Group, Marker, OpenCVRenderer, Path, Point,
                    Rectangle, SVGImporter, Scene, StrokeStyle, ValidationError)

RED, BLUE, GREEN = Color(255, 0, 0), Color(0, 0, 255), Color(0, 160, 0)


def box(order="normal"):
    return Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=RED),
                     stroke=StrokeStyle(color=BLUE, width=20), paint_order=order)


def render(*drawables, alpha=False):
    scene = Scene(100, 100, background=Color(255, 255, 255, 0 if alpha else 1))
    for drawable in drawables:
        scene.add(drawable)
    return OpenCVRenderer().render(scene, alpha=alpha).buffer


def rgb(buffer, x, y):
    return buffer[y, x, :3][::-1].tolist()


@pytest.mark.parametrize("value,expected", [
    ("normal", "normal"), ("fill", "normal"), ("fill stroke markers", "normal"),
    ("stroke", "stroke fill markers"), ("  Stroke,  FILL ", "stroke fill markers"),
    ("markers", "markers fill stroke"), ("fill markers", "fill markers stroke"),
])
def test_values_normalize_like_svg(value, expected):
    assert box(value).paint_order == expected


@pytest.mark.parametrize("value", ["", "stroke stroke", "outline", "normal fill", 3])
def test_invalid_values_are_rejected_without_changing_state(value):
    shape = box("stroke")
    with pytest.raises(ValidationError):
        shape.paint_order = value
    assert shape.paint_order == "stroke fill markers"


def test_stroke_first_puts_the_fill_over_the_stroke_inner_half():
    assert rgb(render(box()), 50, 25) == [0, 0, 255]
    assert rgb(render(box("stroke")), 50, 25) == [255, 0, 0]
    assert rgb(render(box("stroke")), 50, 15) == [0, 0, 255]  # outer half is unaffected


def test_each_filled_shape_honours_the_order():
    circle = Circle(center=Point(50, 50), radius=30, fill=FillStyle(color=RED),
                    stroke=StrokeStyle(color=BLUE, width=20), paint_order="stroke")
    path = Path.from_svg_path("M20 20 H80 V80 H20 Z", fill=FillStyle(color=RED),
                              stroke=StrokeStyle(color=BLUE, width=20), paint_order="stroke")
    assert rgb(render(circle), 50, 25) == [255, 0, 0]
    assert rgb(render(path), 50, 25) == [255, 0, 0]


def test_arrow_head_fill_can_cover_its_outline():
    def arrow(order):
        return Arrow(start=Point(5, 50), end=Point(95, 50), head_style=ArrowHeadStyle.TRIANGLE,
                     head_length=40, head_width=40, fill=FillStyle(color=RED),
                     stroke=StrokeStyle(color=BLUE, width=10), paint_order=order)
    normal, reordered = render(arrow("normal")), render(arrow("stroke"))
    assert not np.array_equal(normal, reordered)
    assert (reordered[..., 2].astype(int) - reordered[..., 0] > 200).sum() > (normal[..., 2].astype(int) - normal[..., 0] > 200).sum()


def marked_path(order):
    dot = Circle(center=Point(0, 0), radius=12, fill=FillStyle(color=GREEN)).to_path()
    path = Path.from_svg_path("M20 20 H80 V80 H20 Z", fill=FillStyle(color=RED),
                              stroke=StrokeStyle(color=BLUE, width=12), paint_order=order)
    path.marker_start = Marker(dot, units="user")
    return path


def test_markers_can_be_painted_first():
    assert rgb(render(marked_path("normal")), 22, 22) == [0, 160, 0]
    assert rgb(render(marked_path("markers")), 22, 22) == [0, 0, 255]


def test_persistence_clone_and_undo_keep_the_order():
    scene = Scene(100, 100)
    shape = box("stroke")
    scene.add(shape)
    loaded = Scene.from_json(scene.to_json())
    assert loaded.get(shape.id).paint_order == "stroke fill markers"
    assert shape.clone().paint_order == "stroke fill markers"
    assert Group(paint_order="stroke").clone().paint_order == "stroke fill markers"
    with scene.edit(shape):
        shape.paint_order = "normal"
    scene.undo()
    assert shape.paint_order == "stroke fill markers"
    assert "paint_order" not in box().to_dict()


@pytest.mark.parametrize("order", ["stroke", "markers", "fill markers", "stroke markers fill"])
def test_svg_export_round_trips_and_matches_rendering(order):
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(marked_path(order))
    svg = scene.to_svg(strict=True)
    reimported = SVGImporter().parse(svg).scene
    assert np.array_equal(OpenCVRenderer().render(reimported, alpha=True).buffer[20:80, 20:80, 3] > 0,
                          OpenCVRenderer().render(scene, alpha=True).buffer[20:80, 20:80, 3] > 0)
    resvg = pytest.importorskip("resvg_py")
    import cv2
    other = cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=svg), np.uint8), cv2.IMREAD_UNCHANGED)
    ours = OpenCVRenderer().render(scene, alpha=True).buffer
    for x, y in ((22, 22), (50, 25), (50, 15), (50, 50)):
        assert np.abs(ours[y, x].astype(int) - other[y, x]).max() <= 2, (x, y)


def test_single_path_export_uses_the_paint_order_attribute():
    scene = Scene(100, 100)
    scene.add(box("stroke"))
    assert 'paint-order="stroke fill markers"' in scene.to_svg()
    scene.add(marked_path("fill markers"))
    assert scene.to_svg().count("<path") >= 3  # split into fill, markers, stroke
