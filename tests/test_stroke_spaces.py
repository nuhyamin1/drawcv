"""Object-space stroke geometry, interchange and retained-state contracts."""
import cv2
import numpy as np
import pytest

from drawcv import (Color, Group, Line, OpenCVRenderer, Path, Point, Scene, StrokeStyle,
                    SVGImporter, Transform, ValidationError, CapStyle, JoinStyle,
                    FillStyle, LinearGradient, GradientStop, BlurEffect, PathBooleanError)
from drawcv.serialization import CURRENT_SCHEMA_VERSION, SchemaMigrator


def render(node):
    scene = Scene(220, 180, background=Color(0, 0, 0, 0))
    scene.add(node)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def reference(svg):
    resvg = pytest.importorskip('resvg_py')
    return cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=svg), np.uint8), -1)


@pytest.mark.parametrize('space,inside,outside', [('screen', 42, 47), ('object', 52, 59)])
def test_width_is_measured_in_selected_space(space, inside, outside):
    node = Line(start=Point(10, 10), end=Point(50, 10), stroke=StrokeStyle(width=8, space=space,
                cap_style=CapStyle.BUTT), transform=Transform.from_matrix(np.diag([2., 4., 1.])))
    image = render(node)
    assert image[inside, 60, 3] == 255
    assert image[outside, 60, 3] == 0
    assert node.get_bounds().bottom >= inside


@pytest.mark.parametrize('transform', ['matrix(2 .3 .4 1.5 10 10)',
                                      'matrix(-1.8 .2 .3 1.5 160 10)', 'rotate(20 70 70)'])
@pytest.mark.parametrize('cap,join', [('butt', 'miter'), ('square', 'bevel'), ('round', 'round')])
@pytest.mark.parametrize('dash', ['', 'stroke-dasharray="14 10" stroke-dashoffset="3"'])
def test_affine_caps_joins_and_dashes_match_svg(transform, cap, join, dash):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="220" height="180">'
           f'<g transform="{transform}"><path d="M20 20L60 25L45 65" fill="none" '
           f'stroke="red" stroke-width="10" stroke-linecap="{cap}" stroke-linejoin="{join}" {dash}/></g></svg>')
    scene = SVGImporter().parse(svg).scene
    path = scene.find_by_type(Path)[0]
    assert path.stroke.space == 'object'
    assert path.stroke.width == 10
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    expected = reference(svg)
    exported = scene.export_svg(strict=True)
    assert not exported.fallbacks
    np.testing.assert_allclose(reference(exported.svg).astype(int), expected.astype(int), atol=1)
    # Rasterizers differ at edges. Require full coverage in the reference interior
    # and no stray coverage more than two pixels outside its silhouette.
    kernel = np.ones((5, 5), np.uint8)
    interior = cv2.erode((expected[..., 3] == 255).astype(np.uint8), kernel) > 0
    exterior = cv2.dilate((expected[..., 3] > 0).astype(np.uint8), kernel) == 0
    assert interior.any()
    assert np.min(actual[..., 3][interior]) >= 240
    assert not actual[..., 3][exterior].any()


def test_non_scaling_svg_stroke_and_future_edits():
    svg = '<svg width="220" height="180"><path d="M10 20H70" stroke="red" stroke-width="8" vector-effect="non-scaling-stroke" transform="scale(2 3)"/></svg>'
    scene = SVGImporter().parse(svg).scene
    node = scene.find_by_type(Path)[0]
    assert node.stroke.space == 'screen'
    assert node.stroke.width == 8
    node.transform = Transform.from_matrix(np.diag([2., 4., 1.]))
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    assert actual[80, 80, 3] == 255
    assert actual[88, 80, 3] == 0


@pytest.mark.parametrize('paint_space', ['object', 'world'])
def test_paint_mapping_is_independent_of_stroke_space(paint_space):
    paint = LinearGradient(Point(10, 0), Point(100, 0),
                           (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255))),
                           space=paint_space)
    paint.transform = Transform(translation_x=7)
    node = Path(stroke=StrokeStyle(paint=paint, space='object', width=14),
                fill=FillStyle(paint=paint), transform=Transform.from_matrix(np.array([[1.6, .4, 20.], [0., 1.4, 15.], [0., 0., 1.]])))
    node.move_to(10, 20).line_to(70, 20).line_to(70, 70).close()
    scene = Scene(220, 180, background=Color(0, 0, 0, 0)); scene.add(node)
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    svg = scene.to_svg(strict=True)
    expected = reference(svg)
    for x, y in [(80, 43), (100, 55), (120, 90)]:
        assert actual[y, x, 3] == expected[y, x, 3] == 255
        np.testing.assert_allclose(actual[y, x, :3].astype(int), expected[y, x, :3].astype(int), atol=4)
    reloaded = SVGImporter().parse(svg).scene
    np.testing.assert_allclose(OpenCVRenderer().render(reloaded, alpha=True).buffer.astype(int), actual.astype(int), atol=1)


def test_space_validation_copy_history_json_and_old_documents():
    style = StrokeStyle(space='object', width=8)
    with pytest.raises(ValidationError):
        style.space = 'world'
    assert style.space == 'object'
    assert style.copy().space == StrokeStyle.from_dict(style.to_dict()).space == 'object'
    assert StrokeStyle.from_dict({'width': 8}).space == 'screen'
    node = Line(start=Point(10, 20), end=Point(70, 20), stroke=style)
    scene = Scene(100, 100); scene.add(node)
    with scene.edit(node):
        node.stroke.space = 'screen'
    scene.undo(); assert node.stroke.space == 'object'
    scene.redo(); assert node.stroke.space == 'screen'
    scene.undo()
    loaded = Scene.from_json(scene.to_json())
    assert loaded.objects[0].stroke.space == node.clone().stroke.space == 'object'
    legacy = scene.to_dict(); legacy['version'] = legacy['schema_version'] = '1.9'
    del legacy['scene']['layers'][0]['objects'][0]['stroke']['space']
    assert Scene.from_dict(legacy).objects[0].stroke.space == 'screen'
    assert CURRENT_SCHEMA_VERSION == '1.10'
    from drawcv import UnsupportedVersionError
    with pytest.raises(UnsupportedVersionError):
        SchemaMigrator.migrate(scene.to_dict(), target_version='1.9')


def test_nested_bounds_effects_and_progress_restore():
    node = Line(start=Point(15, 20), end=Point(70, 20),
                stroke=StrokeStyle(space='object', width=12), effects=[BlurEffect(kernel_size=5)])
    parent = Group(children=[node], transform=Transform.from_matrix(np.diag([2., 3., 1.])))
    scene = Scene(220, 180, background=Color(0, 0, 0, 0)); scene.add(parent)
    image = OpenCVRenderer().render(scene, alpha=True).buffer
    assert image[73, 70, 3] > 200
    scene.animate(node, 'stroke.width', 12, 20, duration=1)
    authored = scene.to_json()
    frame = scene.render_at_time(.5, alpha=True).buffer
    assert frame[79, 70, 3] > image[79, 70, 3]
    assert scene.to_json() == authored
    node.render_progress = .5
    partial = OpenCVRenderer().render(scene, alpha=True).buffer
    assert partial[60, 50, 3] > 200
    assert partial[60, 160, 3] == 0


def test_boolean_stroke_conversion_is_explicit():
    path = Path(stroke=StrokeStyle(space='object', width=4, dash_array=(3, 2)))
    path.move_to(10, 10).line_to(40, 10).line_to(40, 40).close()
    path.transform = Transform.from_matrix(np.diag([2., 2., 1.]))
    result = path.union(Path())
    assert result.stroke.space == 'screen'
    assert result.stroke.width == 8
    assert result.stroke.dash_array == (6, 4)
    path.transform = Transform.from_matrix(np.diag([2., 3., 1.]))
    before = path.to_dict()
    with pytest.raises(PathBooleanError, match='object-space stroke'):
        path.union(Path())
    assert path.to_dict() == before


@pytest.mark.parametrize('space', ['screen', 'object'])
def test_svg_round_trip_preserves_stroke_space(space):
    node = Line(start=Point(10, 20), end=Point(60, 20), stroke=StrokeStyle(space=space, width=8),
                transform=Transform.from_matrix(np.diag([2., 3., 1.])))
    scene = Scene(220, 180); scene.add(node)
    svg = scene.to_svg(strict=True)
    restored = SVGImporter().parse(svg).scene.find_by_type(Path)[0]
    assert restored.stroke.space == space
    assert restored.stroke.width == 8


@pytest.mark.parametrize('cap', [CapStyle.ROUND, CapStyle.SQUARE, CapStyle.BUTT])
def test_zero_length_transformed_caps(cap):
    node = Line(start=Point(20, 20), end=Point(20, 20),
                stroke=StrokeStyle(space='object', width=10, cap_style=cap),
                transform=Transform.from_matrix(np.diag([2., 3., 1.])))
    image = render(node)
    if cap == CapStyle.BUTT:
        assert not image[..., 3].any()
    else:
        assert image[70, 40, 3] == 255
        assert image[60, 47, 3] == 255
        assert image[80, 40, 3] == 0


def test_singular_stroke_has_no_area():
    node = Line(start=Point(20, 20), end=Point(60, 20),
                stroke=StrokeStyle(space='object', width=10),
                transform=Transform.from_matrix(np.diag([2., 0., 1.])))
    assert not render(node)[..., 3].any()


def test_variable_width_freehand_scales_coverage_and_bounds():
    from drawcv import FreehandStroke, StrokePoint
    node = FreehandStroke(points=[StrokePoint(10, 20, pressure=.2), StrokePoint(60, 20, pressure=1)],
                          stroke=StrokeStyle(space='object', width=10), variable_width=True,
                          width_mode='pressure', min_width=2, max_width=10,
                          transform=Transform.from_matrix(np.diag([2., 3., 1.])))
    image = render(node)
    assert image[70, 110, 3] > 240
    assert image[70, 25, 3] == 0
    bounds = node.get_bounds()
    ys, xs = np.nonzero(image[..., 3] > 250)
    assert xs.min() >= bounds.left - 1 and xs.max() <= bounds.right + 1
    assert ys.min() >= bounds.top - 1 and ys.max() <= bounds.bottom + 1


@pytest.mark.parametrize('as_path', [True, False])
def test_compressed_curve_dashes_measure_local_arc_length(as_path):
    from drawcv import BezierCurve
    style = StrokeStyle(space='object', width=80, dash_array=(40, 30), cap_style=CapStyle.BUTT)
    points = [Point(0, 0), Point(30, 500), Point(70, -500), Point(100, 0)]
    if as_path:
        node = Path(stroke=style)
        node.move_to(points[0]).cubic_to(*points[1:])
    else:
        node = BezierCurve.cubic(*points, stroke=style)
    node.transform = Transform.from_matrix(np.array([[1.5, 0., 30.], [0., .03, 80.], [0., 0., 1.]]))
    scene = Scene(220, 180, background=Color(0, 0, 0, 0)); scene.add(node)
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    expected = reference(scene.to_svg(strict=True))
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode((expected[..., 3] == 255).astype(np.uint8), kernel) > 0
    # Wide local ribbons magnify tangent/flattening differences at their edges;
    # allow three raster pixels there while requiring the interior to be filled.
    outside = cv2.dilate((expected[..., 3] > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    assert inside.any()
    assert np.min(actual[..., 3][inside]) >= 230
    assert not actual[..., 3][outside].any()
