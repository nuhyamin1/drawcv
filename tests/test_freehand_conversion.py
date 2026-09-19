"""Freehand outlines preserve processed geometry and sampled width semantics."""
import copy

import cv2
import numpy as np
import pytest

from drawcv import (FreehandStroke, StrokePoint, StrokeStyle, CapStyle, JoinStyle,
                    Point, Transform, Scene, Color, Path, SVGImporter, OpenCVRenderer)


def render(node):
    scene = Scene(240, 200, background=Color(0, 0, 0, 0))
    scene.add(node)
    return OpenCVRenderer().render(scene, alpha=True).buffer


@pytest.mark.parametrize('mode', ['pressure', 'velocity', 'constant'])
@pytest.mark.parametrize('space', ['object', 'screen'])
@pytest.mark.parametrize('processing', [{}, {'smoothing': 'chaikin'},
    {'interpolation': 'catmull_rom', 'interpolation_samples': 5},
    {'simplification': 'rdp', 'smoothing': 'chaikin', 'interpolation': 'catmull_rom', 'interpolation_samples': 3}])
@pytest.mark.parametrize('dash', [(), (17, 9)])
def test_processed_outline_matches_renderer(mode, space, processing, dash):
    source = FreehandStroke(points=[StrokePoint(15, 25, pressure=.1, velocity=10),
                StrokePoint(40, 45, pressure=.8, velocity=90),
                StrokePoint(70, 20, pressure=1, velocity=40),
                StrokePoint(100, 70, pressure=.2, velocity=60)],
                stroke=StrokeStyle(width=12, space=space, dash_array=dash, dash_offset=3),
                variable_width=True, width_mode=mode, min_width=3, max_width=20,
                velocity_max=100, **processing)
    source.transform = Transform.from_matrix(np.array([[-1.5, .3, 190], [.2, 1.5, 10], [0, 0, 1.]]))
    before = copy.deepcopy(source.to_dict())
    outline = source.stroke_to_path(tolerance=.1)
    assert source.to_dict() == before
    actual, expected = render(outline)[..., 3], render(source)[..., 3]
    kernel = np.ones((5, 5), np.uint8)
    interior = cv2.erode((expected == 255).astype(np.uint8), kernel) > 0
    exterior = cv2.dilate((expected > 0).astype(np.uint8), kernel) == 0
    assert interior.any() and actual[interior].min() >= 240
    assert not actual[exterior].any()


@pytest.mark.parametrize('dash', [(), (20, 10)])
def test_pressure_taper_matches_analytic_trapezoids(dash):
    resvg = pytest.importorskip('resvg_py')
    source = FreehandStroke(points=[StrokePoint(20, 60, pressure=0), StrokePoint(120, 60, pressure=1)],
                stroke=StrokeStyle(width=5, cap_style=CapStyle.BUTT, dash_array=dash),
                variable_width=True, min_width=0, max_width=40)
    outline = source.stroke_to_path()
    scene = Scene(240, 200, background=Color(0, 0, 0, 0)); scene.add(outline)
    runs = [(20, 120)] if not dash else [(20, 40), (50, 70), (80, 100), (110, 120)]
    polygons = []
    for start, end in runs:
        a, b = (start-20)*.2, (end-20)*.2
        polygons.append(f'<polygon points="{start},{60-a} {end},{60-b} {end},{60+b} {start},{60+a}"/>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="200">' + ''.join(polygons) + '</svg>'
    def raster(svg):
        return cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=svg), np.uint8), -1)
    np.testing.assert_allclose(raster(scene.to_svg(strict=True)).astype(int), raster(svg).astype(int), atol=1)


@pytest.mark.parametrize('cap', list(CapStyle))
def test_empty_single_coincident_and_zero_width_samples(cap):
    kwargs = dict(stroke=StrokeStyle(width=10, cap_style=cap), variable_width=True, min_width=0, max_width=20)
    assert not FreehandStroke(points=[], **kwargs).stroke_to_path().subpaths
    assert not FreehandStroke(points=[StrokePoint(40, 40, pressure=0), StrokePoint(80, 40, pressure=0)], **kwargs).stroke_to_path().subpaths
    single = FreehandStroke(points=[StrokePoint(40, 40, pressure=1)], **kwargs)
    assert bool(single.stroke_to_path().subpaths) == (cap != CapStyle.BUTT)
    repeated = FreehandStroke(points=[StrokePoint(40, 40, pressure=1), StrokePoint(40, 40, pressure=0)], **kwargs)
    assert repeated.stroke_to_path().subpaths == single.stroke_to_path().subpaths


def test_roundtrip_boolean_partial_and_constant_width():
    source = FreehandStroke(points=[StrokePoint(20, 60), StrokePoint(120, 60)],
                stroke=StrokeStyle(width=10), variable_width=False, min_width=40, max_width=60)
    source.render_progress = .2
    outline = source.stroke_to_path()
    assert outline.get_bounds().height == pytest.approx(10, abs=.2)
    assert outline.get_bounds().right > 120
    partial = source.slice_at_progress(.5).stroke_to_path()
    assert partial.get_bounds().right < 80
    assert outline.union(partial).subpaths
    scene = Scene(240, 200); scene.add(outline)
    restored = Scene.from_json(scene.to_json()).find_by_type(Path)[0]
    assert restored.subpaths == outline.subpaths
    exported = scene.export_svg(strict=True)
    assert not exported.fallbacks
    assert SVGImporter().parse(exported.svg).scene.find_by_type(Path)


def test_missing_pressure_velocity_and_absent_style():
    for mode in ('pressure', 'velocity'):
        source = FreehandStroke(points=[StrokePoint(20, 60), StrokePoint(120, 60)],
                    stroke=StrokeStyle(width=10, cap_style=CapStyle.BUTT),
                    variable_width=True, width_mode=mode, min_width=0, max_width=20)
        assert source.stroke_to_path().get_bounds().height == pytest.approx(10)
    assert not FreehandStroke(points=[StrokePoint(40, 40)]).stroke_to_path().subpaths
