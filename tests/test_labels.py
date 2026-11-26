"""Shape labels: layout against the host box, retained editing, rendering and SVG."""
import json
from pathlib import Path as FilePath

import numpy as np
import pytest

from drawcv import (BoundingBox, Circle, Color, FillStyle, FontAsset, Group, ImageObject, Label, Line,
                    OpenCVRenderer, Point, Rectangle, Scene, StrokeStyle, Text, ValidationError)
from drawcv.label import label_instance
from drawcv.serialization.registry import CURRENT_SCHEMA_VERSION, SchemaMigrator

FONTS = FilePath(__file__).parent / "assets" / "fonts"


def font():
    pytest.importorskip("uharfbuzz")
    pytest.importorskip("freetype")
    pytest.importorskip("icu")
    return FontAsset.from_file(FONTS / "notosans.ttf")


def box_of(text):
    """Layout box of a positioned label Text in host-local coordinates."""
    return text.measure().paragraph_bounds if text.fonts is not None else text.get_geometry_bounds()


def render(scene, alpha=False):
    return OpenCVRenderer().render(scene, alpha=alpha).buffer


def rect(**kwargs):
    return Rectangle(position=Point(40, 30), width=200, height=120, **kwargs)


@pytest.mark.parametrize("align", ["left", "center", "right"])
@pytest.mark.parametrize("vertical_align", ["top", "middle", "bottom"])
def test_inside_alignment_grid_uses_padded_box(align, vertical_align):
    host = rect()
    host.label = Label("Label", align=align, vertical_align=vertical_align, padding=10)
    b = box_of(label_instance(host))
    x = {"left": 50, "center": 140 - b.width / 2, "right": 230 - b.width}[align]
    y = {"top": 40, "middle": 90 - b.height / 2, "bottom": 140 - b.height}[vertical_align]
    assert b.x == pytest.approx(x) and b.y == pytest.approx(y)


def test_default_label_is_centered_on_both_axes_for_font_text():
    host = rect()
    host.label = Label("Centered", fonts=(font(),), font_size=24)
    assert box_of(label_instance(host)).center.x == pytest.approx(140)
    assert box_of(label_instance(host)).center.y == pytest.approx(90)


@pytest.mark.parametrize("placement,edge", [("above", "bottom"), ("below", "top"), ("left", "right"), ("right", "left")])
def test_outside_placements_sit_beside_the_box(placement, edge):
    host = rect()
    host.label = Label("Caption", placement=placement, padding=6)
    b = box_of(label_instance(host))
    expected = {"above": 30 - 6, "below": 150 + 6, "left": 40 - 6, "right": 240 + 6}[placement]
    assert getattr(b, edge) == pytest.approx(expected)
    if placement in ("above", "below"):
        assert b.center.x == pytest.approx(140)
    else:
        assert b.center.y == pytest.approx(90)


def test_outside_placement_clears_the_stroke():
    line = Line(start=Point(20, 100), end=Point(280, 100), stroke=StrokeStyle(color=Color.black(), width=8))
    line.label = Label("above", placement="above", padding=6)
    assert box_of(label_instance(line)).bottom == pytest.approx(100 - 4 - 6)
    line.label.placement = "inside"
    assert box_of(label_instance(line)).center.y == pytest.approx(100)


def test_offset_and_explicit_box():
    host = rect()
    host.label = Label("X", box=BoundingBox(40, 110, 200, 40), offset=(3, -2))
    b = box_of(label_instance(host))
    assert b.center.x == pytest.approx(143) and b.center.y == pytest.approx(128)


def test_label_follows_host_geometry_and_transform():
    host = rect()
    host.label = Label("Move")
    before = box_of(label_instance(host)).center
    host.position = Point(60, 30)
    assert box_of(label_instance(host)).center.x == pytest.approx(before.x + 20)
    host.transform.rotation = 30
    assert np.allclose(label_instance(host).world_matrix, host.world_matrix)


def test_font_text_wraps_only_when_it_overflows():
    F = font()
    host = rect()
    host.label = Label("Short", fonts=(F,), font_size=20, padding=10)
    assert label_instance(host).wrap_width is None
    host.label.text = "A much longer sentence that cannot fit on a single line"
    text = label_instance(host)
    assert text.wrap_width == pytest.approx(180)
    assert len(text.measure().lines) > 1
    host.label.wrap = False
    assert label_instance(host).wrap_width is None


def test_attribute_forwarding_and_validation_keep_previous_values():
    label = Label("Hi", color=Color(10, 20, 30))
    label.color = Color(200, 0, 0)
    label.text = "Changed"
    assert label.content.color == Color(200, 0, 0) and label.content.text == "Changed"
    with pytest.raises(ValidationError):
        label.align = "middle"
    with pytest.raises(ValidationError):
        label.padding = -1
    with pytest.raises(AttributeError):
        label.no_such_option = 1
    assert label.align == "center" and label.padding == 0
    label.vertical_align = "  BOTTOM "
    assert label.vertical_align == "bottom"


def test_invalid_hosts_and_contents_are_rejected():
    with pytest.raises(ValidationError):
        Text("t").label = Label("x")
    with pytest.raises(ValidationError):
        rect().label = "plain string"
    with pytest.raises(ValidationError):
        Label(Text("x", text_origin="baseline"))
    with pytest.raises(ValidationError):
        Label(Text("x"), color=Color.black())


def test_geometry_bounds_exclude_label_but_visual_bounds_include_it():
    host = rect()
    geometry, bounds = host.get_geometry_bounds(), host.get_bounds()
    host.label = Label("Below the box", placement="below", padding=8)
    assert host.get_geometry_bounds() == geometry and host.get_bounds() == bounds
    assert host.get_effect_bounds().bottom > bounds.bottom + 8
    group = Group(children=[host])
    assert group.get_effect_bounds().bottom == pytest.approx(host.get_effect_bounds().bottom)


def test_label_renders_inside_isolated_host_without_cropping():
    scene = Scene(300, 240, background=Color(0, 0, 0, 0))
    host = rect(opacity=0.5)
    host.label = Label("BELOW", placement="below", padding=10, font_scale=1.2, thickness=2)
    scene.add(host)
    alpha = render(scene, alpha=True)[..., 3]
    assert alpha[165:200].max() == 128  # opacity applied once, text not cropped


def test_label_matches_equivalent_standalone_text_pixels():
    host = rect(fill=FillStyle(color=Color(200, 220, 240)))
    host.label = Label("Same", font_scale=1.1, thickness=2)
    labeled = Scene(300, 200, background=Color.white())
    labeled.add(host)
    plain = Scene(300, 200, background=Color.white())
    plain.add(rect(fill=FillStyle(color=Color(200, 220, 240))))
    text = label_instance(host)
    plain.add(Text("Same", position=text.position, alignment=text.alignment, font_scale=1.1, thickness=2))
    assert np.array_equal(render(labeled), render(plain))


def test_group_and_image_labels_render_and_round_trip(tmp_path):
    scene = Scene(300, 200, background=Color.white())
    group = Group(children=[Circle(center=Point(80, 80), radius=40, fill=FillStyle(color=Color(250, 200, 180)))])
    group.label = Label("group", placement="below", padding=4)
    image = ImageObject(np.full((40, 60, 3), 90, np.uint8), position=Point(180, 40))
    image.label = Label("img", placement="below")
    scene.add(group)
    scene.add(image)
    before = render(scene)
    assert (before[125:150, 40:120] < 100).any()
    path = tmp_path / "labels.json"
    scene.save_json(path)
    loaded = Scene.load_json(path)
    assert np.array_equal(render(loaded), before)
    assert group.clone().label.text == "group" and image.clone().label.text == "img"


def test_json_round_trip_clone_and_undo():
    scene = Scene(300, 200, background=Color.white())
    host = rect(fill=FillStyle(color=Color(220, 230, 250)))
    host.label = Label("Saved", align="right", vertical_align="bottom", padding=7, offset=(1, 2),
                       color=Color(120, 0, 0), font_scale=0.9)
    scene.add(host)
    data = json.loads(scene.to_json())
    assert data["version"] == CURRENT_SCHEMA_VERSION == "1.14"
    loaded = Scene.from_dict(data)
    assert np.array_equal(render(loaded), render(scene))
    copy = host.clone()
    copy.label.text = "Copy"
    assert host.label.text == "Saved"
    with scene.edit(host):
        host.label.vertical_align = "top"
    scene.undo()
    assert host.label.vertical_align == "bottom"
    scene.redo()
    assert host.label.vertical_align == "top"
    host.label = None
    assert "label" not in host.to_dict()


def test_unlabeled_json_is_unchanged_and_1_13_migrates():
    assert "label" not in rect().to_dict()
    data = Scene(10, 10).to_dict()
    data["version"] = "1.13"
    assert SchemaMigrator.migrate(data)["version"] == "1.14"


def test_label_properties_animate_observationally():
    scene = Scene(300, 200, background=Color.white())
    host = rect()
    host.label = Label("Anim", vertical_align="top")
    scene.add(host)
    scene.animate(host, "label.padding", 0.0, 40.0, duration=2)
    scene.animate(host, "label.color", Color(0, 0, 0), Color(250, 0, 0), duration=2)
    frame = scene.render_at_time(1.0)
    assert host.label.padding == 0 and host.label.color == Color(0, 0, 0)
    ys = np.nonzero((frame.buffer[..., 2].astype(int) - frame.buffer[..., 1]) > 60)[0]
    assert ys.min() > 30 + 15


def test_progressive_reveal_keeps_label_on_the_complete_shape():
    line = Line(start=Point(20, 100), end=Point(280, 100), stroke=StrokeStyle(color=Color.black(), width=2))
    line.label = Label("mid", placement="above", padding=6, font_scale=0.8, thickness=2)
    full = Scene(300, 200, background=Color.white())
    full.add(line)
    expected = render(full)[40:94]
    line.render_progress = 0.2
    assert np.array_equal(render(full)[40:94], expected)


def test_svg_exports_single_line_font_label_as_native_text():
    F = font()
    scene = Scene(300, 200, background=Color(0, 0, 0, 0))
    host = rect()
    host.transform.rotation = 12
    host.label = Label("Native gy", fonts=(F,), font_family_name="Noto Sans", font_size=26,
                       align="right", vertical_align="bottom", padding=8)
    scene.add(host)
    svg = scene.to_svg(strict=True)
    assert "<text" in svg and "Native gy" in svg
    resvg = pytest.importorskip("resvg_py")
    import cv2
    png = resvg.svg_to_bytes(svg_string=svg, width=300, height=200, skip_system_fonts=True,
                             font_dirs=[str(FONTS.resolve())])
    other = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)[..., 3]
    ours = render(scene, alpha=True)[..., 3]

    def ink_box(mask):
        ys, xs = np.nonzero(mask > 64)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()])

    assert np.abs(ink_box(ours) - ink_box(other)).max() <= 1
    assert other.sum() / ours.sum() == pytest.approx(1, abs=0.03)


def test_svg_reports_hershey_label_fallback_and_strict_rejects_it():
    scene = Scene(300, 200)
    host = rect()
    host.label = Label("raster")
    scene.add(host)
    result = scene.export_svg()
    assert [f.entity for f in result.fallbacks] == [f"{host.id}-label"]
    with pytest.raises(Exception):
        scene.to_svg(strict=True)


def test_wrapped_font_label_exports_all_lines_as_native_text():
    import xml.etree.ElementTree as ET
    host = rect()
    host.label = Label("one two three four five", fonts=(font(),), font_family_name="Noto Sans",
                       font_size=28, padding=20)
    scene = Scene(300, 200)
    scene.add(host)
    lines = label_instance(host).measure().lines
    assert len(lines) > 1
    root = ET.fromstring(scene.to_svg(strict=True))
    assert [s.text for s in root.findall(".//{*}tspan")] == [line.text for line in lines]
