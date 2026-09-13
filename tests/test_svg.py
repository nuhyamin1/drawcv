"""SVG structure, observational export, and independent resvg rendering checks."""
import base64
import os
os.environ.setdefault('PYTHAINLP_READ_ONLY', '1')
os.environ.setdefault('PYTHAINLP_OFFLINE', '1')
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest
from drawcv import (Scene, Color, Point, Rectangle, Circle, Ellipse, RoundedRectangle,
    Line, Polyline, Polygon, Path, BezierCurve, Arc, FreehandStroke, StrokePoint,
    FillStyle, StrokeStyle, LinearGradient, RadialGradient, GradientStop, Group,
    Transform, Text, ClipRect, ClipPath, Mask, BlurEffect, ShadowEffect, SVGExporter,
    RenderError, ValidationError, OpenCVRenderer, CapStyle, JoinStyle, FillRule,
    ImageObject)

NS = {'s': 'http://www.w3.org/2000/svg'}


def scene(*objects):
    s = Scene(240, 160, Color(0, 0, 0, 0))
    for obj in objects:
        s.add(obj)
    return s


def rect(**kwargs):
    return Rectangle(position=Point(30, 25), width=100, height=75,
                     fill=FillStyle(color=Color(220, 30, 40)), **kwargs)


def resvg_render(result, width=240, height=160):
    resvg = pytest.importorskip('resvg_py')
    png = resvg.svg_to_bytes(svg_string=result.svg, width=width, height=height, skip_system_fonts=True)
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


def fallback_pixels(result, width=240, height=160):
    root = ET.fromstring(result.svg)
    image = root.find('.//s:image', NS)
    data = image.get('{http://www.w3.org/1999/xlink}href').split(',', 1)[1]
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_UNCHANGED)
    canvas = np.zeros((height, width, 4), np.uint8)
    x, y = int(image.get('x')), int(image.get('y'))
    canvas[y:y+pixels.shape[0], x:x+pixels.shape[1]] = pixels
    return canvas


def test_export_deterministic_and_observational(tmp_path):
    s = scene(rect(name='A < B & "quoted"'))
    before = s.to_json()
    result = s.export_svg(strict=True)
    assert result == s.export_svg(strict=True)
    assert not result.fallbacks
    assert s.to_json() == before
    path = tmp_path / 'drawing.svg'
    assert s.save_svg(path) == result
    assert path.read_text(encoding='utf-8') == s.to_svg()
    assert ET.fromstring(result.svg).get('viewBox') == '0 0 240 160'


@pytest.mark.parametrize('obj', [
    Line(start=Point(10, 20), end=Point(90, 40)),
    Circle(center=Point(60, 60), radius=30),
    Ellipse(center=Point(60, 60), radius_x=40, radius_y=20),
    RoundedRectangle(x=10, y=20, width=80, height=60, corner_radius=12),
    Polygon(vertices=[Point(10, 10), Point(80, 20), Point(50, 90)]),
    Polyline(points=[Point(10, 10), Point(80, 20), Point(50, 90)]),
    Arc(center=Point(60, 60), radius_x=40, radius_y=25),
    BezierCurve(p0=Point(10, 10), p1=Point(60, 90), p2=Point(100, 20)),
    FreehandStroke(points=[StrokePoint(10, 10), StrokePoint(90, 60)], variable_width=False),
])
def test_geometry_is_editable_vector(obj):
    obj.stroke = StrokeStyle(width=5)
    result = scene(obj).export_svg(strict=True)
    assert not result.fallbacks
    path = ET.fromstring(result.svg).find('.//s:g/s:g/s:path', NS)
    assert path is not None and path.get('d').startswith('M ')
    assert np.any(resvg_render(result)[..., 3])


def test_world_curve_commands_holes_and_clip():
    path = Path(fill=FillStyle(color=Color(200, 0, 0)), fill_rule=FillRule.EVEN_ODD)
    path.move_to(10, 10).line_to(130, 10).quadratic_to(Point(150, 60), Point(130, 110)).line_to(10, 110).close()
    path.move_to(40, 40).line_to(80, 40).line_to(80, 80).line_to(40, 80).close()
    path.transform = Transform(translation_x=10)
    path.clip = ClipRect(0, 0, 200, 150)
    s = scene(path)
    result = s.export_svg(strict=True)
    root = ET.fromstring(result.svg)
    node = root.find('.//s:g/s:g/s:path', NS)
    assert node.get('d').startswith('M 20 10') and 'Q ' in node.get('d')
    assert node.get('fill-rule') == 'evenodd'
    image = resvg_render(result)
    assert image[60, 65, 3] == 0 and image[30, 35, 3] == 255


@pytest.mark.parametrize('paint', [
    LinearGradient(Point(30, 0), Point(130, 0), (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))),
    LinearGradient(Point(30, 0), Point(130, 0), (GradientStop(0, Color(255, 0, 0, 0)), GradientStop(1, Color(0, 0, 255, 1))), space='world'),
    RadialGradient(Point(80, 60), 60, (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))),
])
def test_gradient_independent_samples(paint):
    obj = rect()
    obj.fill = FillStyle(paint=paint)
    s = scene(obj)
    svg = resvg_render(s.export_svg(strict=True))
    reference = OpenCVRenderer().render(s, alpha=True).buffer
    # SVG samples pixel centers at n+0.5; DrawCV samples integer centers.
    # Compare premultiplied channels to avoid RGB instability near zero alpha.
    def pm(p):
        p = p.astype(float)
        p[..., :3] *= p[..., 3:4] / 255
        return p
    assert np.max(np.abs(pm(svg[40:80, 50:110])-pm(reference[40:80, 50:110]))) < 5


@pytest.mark.parametrize('kind', ['text', 'blur', 'shadow', 'mask', 'image', 'pressure'])
def test_fallback_exact_png_and_strict_no_file_write(kind, tmp_path):
    obj = rect()
    if kind == 'text':
        obj = Text('Hello', position=Point(30, 40))
    elif kind == 'blur':
        obj.effects = [BlurEffect(kernel_size=9)]
    elif kind == 'shadow':
        obj.effects = [ShadowEffect()]
    elif kind == 'mask':
        obj.mask = Mask(np.full((160, 240), 150, np.uint8))
    elif kind == 'image':
        obj = ImageObject(np.full((20, 30, 4), (0, 0, 255, 100), np.uint8), position=Point(20, 20))
    else:
        obj = FreehandStroke(points=[StrokePoint(10, 20), StrokePoint(90, 50)],
            stroke=StrokeStyle(width=8), variable_width=True)
    obj.opacity = .6
    s = scene(obj)
    before = s.to_json()
    result = s.export_svg()
    assert len(result.fallbacks) == 1
    np.testing.assert_array_equal(fallback_pixels(result), OpenCVRenderer().render(s, alpha=True).buffer)
    file = tmp_path / 'keep.svg'
    file.write_text('unchanged')
    with pytest.raises(RenderError, match='raster fallback'):
        s.save_svg(file, strict=True)
    assert file.read_text() == 'unchanged' and s.to_json() == before


def test_nested_fallback_does_not_bake_ancestor_opacity():
    child = rect(opacity=.7)
    child.effects = [BlurEffect(kernel_size=5)]
    group = Group(children=[child], opacity=.4)
    s = scene(group)
    s.layers[0].opacity = .5
    result = s.export_svg()
    pixels = fallback_pixels(result)
    assert pixels[50, 50, 3] == round(.7*255)
    root = ET.fromstring(result.svg)
    assert [g.get('opacity') for g in root.findall('.//s:g', NS)] == ['0.5', '0.4', None]


def test_group_and_layer_fallback_isolated_as_unit():
    group = Group(children=[rect(opacity=.8), Circle(center=Point(100, 80), radius=40,
        fill=FillStyle(color=Color(0, 0, 255)))], opacity=.6)
    s = scene(group)
    group.mask = Mask(np.full((160, 240), 200, np.uint8))
    np.testing.assert_array_equal(fallback_pixels(s.export_svg()), OpenCVRenderer().render(s, alpha=True).buffer)
    group.mask = None
    s.layers[0].effects = [BlurEffect(kernel_size=7)]
    s.layers[0].opacity = .5
    result = s.export_svg()
    assert len(result.fallbacks) == 1 and result.fallbacks[0].entity == 'default'
    np.testing.assert_array_equal(fallback_pixels(result), OpenCVRenderer().render(s, alpha=True).buffer)


def test_hidden_fallbacks_skipped_z_order_and_progress():
    hidden = Text('hidden', visible=False)
    line = Line(start=Point(10, 30), end=Point(110, 30), stroke=StrokeStyle(width=4), z_index=-1)
    line.progress = .5
    s = scene(rect(), hidden, line)
    before = s.to_json()
    result = s.export_svg(strict=True)
    paths = ET.fromstring(result.svg).findall('.//s:g/s:g/s:path', NS)
    assert paths[0].get('d') == 'M 10 30 L 60 30'
    assert s.to_json() == before


def test_stroke_units_under_nested_scale():
    line = Line(start=Point(10, 20), end=Point(100, 20), stroke=StrokeStyle(width=6,
        cap_style=CapStyle.BUTT, join_style=JoinStyle.MITER, dash_array=(12, 8), dash_offset=3))
    group = Group(children=[line], transform=Transform(scale_x=2, scale_y=.5, pivot=Point(0, 0)))
    result = scene(group).export_svg(strict=True)
    path = ET.fromstring(result.svg).find('.//s:path', NS)
    assert path.get('d') == 'M 20 10 L 200 10'
    assert path.get('stroke-width') == '6' and path.get('stroke-dasharray') == '12 8'
    image = resvg_render(result)
    assert np.count_nonzero(image[:, 24, 3]) == 6


def test_exporter_validation_and_reuse():
    with pytest.raises(ValidationError):
        SVGExporter(strict=1)
    exporter = SVGExporter(strict=True)
    with pytest.raises(ValidationError):
        exporter.render(None)
    with pytest.raises(RenderError):
        exporter.render(scene(Text('fallback')))
    assert not exporter.render(scene(rect())).fallbacks


@pytest.mark.parametrize('space', ['object', 'world'])
@pytest.mark.parametrize('radial', [False, True])
def test_transformed_gradient_nested_coordinate_contract(space, radial):
    stops = (GradientStop(.15, Color(255, 0, 0, .3)), GradientStop(.85, Color(0, 0, 255, .9)))
    paint = (RadialGradient(Point(80, 60), 90, stops, space) if radial else
             LinearGradient(Point(30, 25), Point(160, 100), stops, space))
    obj = rect()
    obj.fill = FillStyle(paint=paint)
    group = Group(children=[obj], transform=Transform(rotation=12, scale_x=1.1, scale_y=.85))
    s = scene(group)
    result = s.export_svg(strict=True)
    node = ET.fromstring(result.svg).find('.//s:defs/*', NS)
    assert ('gradientTransform' in node.attrib) == (space == 'object')
    image = resvg_render(result).astype(float)
    reference = OpenCVRenderer().render(s, alpha=True).buffer.astype(float)
    for pixels in (image, reference):
        pixels[..., :3] *= pixels[..., 3:4] / 255
    assert np.max(np.abs(image[45:80, 60:105]-reference[45:80, 60:105])) < 6


def test_temporal_svg_restores_identity_and_state_on_failure():
    obj = rect()
    s = scene(obj)
    s.animate(obj, 'opacity', 1, .5, duration=1)
    before = s.to_json()
    result = s.render_at_time(.5, renderer=SVGExporter(strict=True))
    assert 'opacity="0.75"' in result.svg
    assert s.to_json() == before and s.get(obj.id) is obj
    obj.effects = [BlurEffect(kernel_size=5)]
    before = s.to_json()
    with pytest.raises(RenderError):
        s.render_at_time(.5, renderer=SVGExporter(strict=True))
    assert s.to_json() == before and s.get(obj.id) is obj


def test_clip_under_nested_transform_and_duplicate_gradient_stops():
    obj = rect()
    obj.fill = FillStyle(paint=LinearGradient(Point(30, 0), Point(130, 0), (
        GradientStop(.5, Color(255, 0, 0)), GradientStop(.5, Color(0, 0, 255)))))
    obj.clip = ClipPath([Point(20, 20), Point(100, 20), Point(100, 110), Point(20, 110)])
    group = Group(children=[obj], transform=Transform(translation_x=15))
    image = resvg_render(scene(group).export_svg(strict=True))
    np.testing.assert_array_equal(image[50, 65], [0, 0, 255, 255])
    np.testing.assert_array_equal(image[50, 105], [255, 0, 0, 255])
    assert image[50, 125, 3] == 0


def test_transparent_background_external_composites():
    obj = rect()
    obj.fill = FillStyle(color=Color(255, 0, 0, .5))
    s = scene(obj)
    image = resvg_render(s.export_svg(strict=True)).astype(float)
    reference = OpenCVRenderer().render(s, alpha=True).buffer.astype(float)
    for background in [(255, 255, 255), (0, 0, 0), (70, 120, 190)]:
        def flatten(p):
            a = p[..., 3:4] / 255
            return p[..., :3]*a + np.array(background)*(1-a)
        assert np.max(np.abs(flatten(image)[40:80, 40:110]-flatten(reference)[40:80, 40:110])) <= 1


@pytest.mark.parametrize('fallback', [False, True])
def test_native_group_isolation_and_fallback_ancestor_composition(fallback):
    first = rect(opacity=.7)
    if fallback:
        first.effects = [BlurEffect(kernel_size=5)]
    second = Rectangle(position=Point(70, 50), width=90, height=70,
        fill=FillStyle(color=Color(20, 70, 230, .6)))
    s = scene(Group(children=[first, second], opacity=.6))
    s.layers[0].opacity = .5
    image = resvg_render(s.export_svg()).astype(float)
    reference = OpenCVRenderer().render(s, alpha=True).buffer.astype(float)
    for pixels in (image, reference):
        pixels[..., :3] *= pixels[..., 3:4] / 255
    assert np.max(np.abs(image[60:90, 80:115]-reference[60:90, 80:115])) <= 2


def test_json_clone_history_export_equivalence():
    s = scene(rect())
    obj = s.objects[0]
    baseline = s.to_svg()
    assert Scene.from_json(s.to_json()).to_svg() == baseline
    with s.edit(obj):
        obj.fill = FillStyle(color=Color(40, 120, 220))
    changed = s.to_svg()
    assert changed != baseline
    s.undo()
    assert s.to_svg() == baseline
    s.redo()
    assert s.to_svg() == changed
    clone = obj.clone()
    original_path = ET.fromstring(s.to_svg()).find('.//s:g/s:g/s:path', NS)
    clone_path = ET.fromstring(scene(clone).to_svg()).find('.//s:g/s:g/s:path', NS)
    assert original_path.attrib == clone_path.attrib


def test_font_text_fallback_preserves_bidi_and_transform():
    from pathlib import Path as FilePath
    from drawcv import FontAsset
    for dependency in ('icu', 'uharfbuzz', 'freetype', 'regex', 'pythainlp'):
        pytest.importorskip(dependency)
    font = FontAsset.from_file(FilePath(__file__).parent / 'assets/fonts/notosansarabic.ttf')
    obj = Text('ABC (مرحبا) 123', fonts=(font,), font_size=20, position=Point(20, 40),
        transform=Transform(rotation=8), opacity=.6)
    s = scene(obj)
    result = s.export_svg()
    np.testing.assert_array_equal(fallback_pixels(result), OpenCVRenderer().render(s, alpha=True).buffer)
