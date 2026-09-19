import copy
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from drawcv import (Path, Point, Marker, FillStyle, StrokeStyle, Color, Group, Transform,
                    Scene, OpenCVRenderer, SVGImporter, ValidationError, BlurEffect)
from drawcv.markers import marker_instances
from drawcv.effects.clipping import ClipRect


def triangle():
    return Path(fill=FillStyle(color=Color(255, 0, 0))).move_to(0, 0).line_to(-4, -2).line_to(-4, 2).close()


def host():
    return Path(stroke=StrokeStyle(width=4)).move_to(30, 60).line_to(90, 60).line_to(90, 120)


def render(path):
    scene = Scene(180, 180, background=Color(0, 0, 0, 0)); scene.add(path)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def assert_xy(p, x, y):
    np.testing.assert_allclose([p.x, p.y], [x, y], atol=1e-5)


def test_orientations_ref_and_reuse():
    p = host()
    marker = Marker(triangle(), orient='auto-start-reverse')
    p.marker_start = p.marker_mid = p.marker_end = marker
    instances = list(marker_instances(p))
    assert len(instances) == 3
    assert_xy(instances[0].to_world(Point(-4, 0)), 46, 60)
    assert_xy(instances[1].to_world(Point(-4, 0)), 90-16/math.sqrt(2), 60-16/math.sqrt(2))
    assert_xy(instances[2].to_world(Point(-4, 0)), 90, 104)
    marker.orient = 90
    marker.ref = Point(-4, 0)
    marker.units = 'user'
    assert_xy(list(marker_instances(p))[0].to_world(Point(-4, 0)), 30, 60)
    assert marker.path.parent is None


def test_curve_derivatives_zero_segments_closed_and_subpaths():
    p = Path().move_to(0, 0).cubic_to(Point(0, 0), Point(0, 30), Point(50, 30))
    p.line_to(50, 30).line_to(80, 30)
    p.marker_start = p.marker_mid = p.marker_end = Marker(triangle(), units='user')
    items = list(marker_instances(p))
    assert len(items) == 4
    assert_xy(items[0].to_world(Point(-1, 0)), 0, -1)
    p.close()
    assert len(list(marker_instances(p))) == 5
    p.move_to(120, 20).line_to(150, 20)
    assert len(list(marker_instances(p))) == 7


def test_units_bounds_hit_and_parent_transform():
    p = host()
    p.marker_end = Marker(triangle(), units='user', size=10, orient=0)
    assert p.contains_point(Point(60, 130))
    assert p.get_bounds().bottom >= 140
    assert p.get_local_bounds().bottom >= 140
    g = Group(transform=Transform.from_matrix(np.array([[-2., .4, 200], [0, 1.2, 10], [0,0,1.]])))
    g.add(p, preserve_world_transform=False)
    instance = list(marker_instances(p))[0]
    expected = p.to_world(Point(90,120))
    assert_xy(instance.to_world(Point(0,0)), expected.x, expected.y)
    assert g.get_bounds().bottom >= instance.get_bounds().bottom


def test_json_copy_trim_state_and_old_documents():
    p = host(); p.marker_end = Marker(triangle())
    scene = Scene(180, 180); scene.add(p)
    restored = Scene.from_json(scene.to_json()).find_by_type(Path)[0]
    assert restored.marker_end.to_dict() == p.marker_end.to_dict()
    assert restored.marker_end is not p.marker_end
    clone = p.to_path()
    clone.marker_end.size = 3
    assert p.marker_end.size == 1
    state = p._get_shape_state()
    p.marker_end = None
    p._apply_shape_state(state)
    assert isinstance(p.marker_end, Marker)
    assert p.trim(.2, .8).marker_end is not None
    old = p.to_dict()
    for key in ('marker_start', 'marker_mid', 'marker_end'):
        old.pop(key)
    assert Path.from_dict(old).marker_end is None


def test_render_progress_clip_opacity_and_artwork_unchanged():
    p = Path(stroke=StrokeStyle(width=4)).move_to(30,60).line_to(130,60)
    p.marker_end = Marker(triangle(), units='user', size=5)
    before = copy.deepcopy(p.marker_end.to_dict())
    p.render_progress = .5
    p.opacity = .5
    p.clip = ClipRect(0, 0, 75, 180)
    image = render(p)
    assert image[65, 67, 3] in range(125,130)
    assert not image[:, 80:, 3].any()
    assert p.marker_end.to_dict() == before


@pytest.mark.parametrize('orient', ['auto', 'auto-start-reverse', 30])
@pytest.mark.parametrize('transformed', [False, True])
def test_svg_export_matches_native_markers(orient, transformed):
    resvg = pytest.importorskip('resvg_py')
    p = host(); p.stroke.space = 'object'
    p.marker_start = p.marker_mid = p.marker_end = Marker(triangle(), orient=orient)
    matrix = np.array([[-.8, .2, 130], [.1, .8, 10], [0,0,1.]]) if transformed else np.eye(3)
    p.transform = Transform.from_matrix(matrix)
    transform = 'matrix(-.8 .1 .2 .8 130 10)' if transformed else 'matrix(1 0 0 1 0 0)'
    scene = Scene(180,180,background=Color(0,0,0,0)); scene.add(p)
    svg = scene.to_svg(strict=True)
    reference = f'''<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180">
    <defs><marker id="m" orient="{orient}" markerUnits="strokeWidth" overflow="visible">
    <path d="M0 0L-4 -2L-4 2Z" fill="red"/></marker></defs>
    <path transform="{transform}" d="M30 60L90 60L90 120" fill="none" stroke="black" stroke-width="4"
    stroke-linecap="round" stroke-linejoin="round" marker-start="url(#m)" marker-mid="url(#m)" marker-end="url(#m)"/></svg>'''
    def raster(text):
        return cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=text), np.uint8), -1)
    assert np.mean(np.abs(raster(svg).astype(float)-raster(reference).astype(float))) < .1
    imported = SVGImporter().parse(svg).scene
    assert len(imported.find_by_type(Path)) == 4


def test_marker_mutation_validation_and_nested_rejection():
    marker = Marker(triangle())
    with pytest.raises(ValidationError):
        marker.size = 0
    assert marker.size == 1
    with pytest.raises(ValidationError):
        Marker(triangle(), orient=float('nan'))
    with pytest.raises(ValidationError):
        Path(marker_end='bad')
    p = host(); p.marker_end = marker
    marker.path.marker_end = marker
    with pytest.raises(ValidationError, match='Nested'):
        p.get_bounds()
    with pytest.raises(ValidationError, match='Nested'):
        p.to_dict()


def test_bgr_opacity_is_applied_once_to_overlapping_host_and_marker():
    p = Path(stroke=StrokeStyle(color=Color(255, 0, 0), width=8), opacity=.5)
    p.move_to(20,60).line_to(100,60)
    p.marker_end = Marker(triangle(), size=2)
    scene = Scene(180,180,background=Color(255,255,255)); scene.add(p)
    image = OpenCVRenderer().render(scene).buffer
    np.testing.assert_allclose(image[60,80], [127,127,255], atol=1)


def test_parent_opacity_and_marker_effect_bounds():
    p = host(); artwork = triangle()
    artwork.effects = [BlurEffect(kernel_size=5)]
    p.marker_end = Marker(artwork, size=5, orient=0)
    group = Group(children=[p], opacity=.5)
    image = render(group)
    assert image[130,50,3] in range(120,130)
    marker = list(marker_instances(p))[0]
    assert p.get_bounds().bottom >= marker.get_effect_bounds().bottom
    scene = Scene(180,180); scene.add(group)
    assert scene.export_svg().fallbacks
    from drawcv import RenderError
    with pytest.raises(RenderError):
        scene.export_svg(strict=True)


def test_schema_migration_and_geometry_only_operations_exclude_markers():
    from drawcv import SchemaMigrator, CURRENT_SCHEMA_VERSION
    p = host(); p.marker_end = Marker(triangle())
    scene = Scene(180,180); scene.add(p)
    data = scene.to_dict()
    data['version'] = '1.10'
    if 'schema_version' in data:
        data['schema_version'] = '1.10'
    migrated = SchemaMigrator.migrate(data)
    assert CURRENT_SCHEMA_VERSION == '1.13'
    assert Scene.from_dict(migrated).find_by_type(Path)[0].marker_end is not None
    assert p.stroke_to_path().marker_end is None
    assert p.length() == 120
