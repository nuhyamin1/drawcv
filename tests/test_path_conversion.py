"""Exact primitive conversion and curve-preserving interchange."""
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from drawcv import (Arc, ArcClosure, Circle, Ellipse, RoundedRectangle, Rectangle,
                    Line, Polygon, Polyline, BezierCurve, Group, Path, Point, Scene,
                    Transform, StrokeStyle, FillStyle, Color, SVGImporter, ValidationError)
from drawcv.shapes.path import EllipticalArcTo, Close, LineTo
from drawcv.effects.clipping import ClipRect


def curves():
    return [Circle(center=Point(50, 50), radius=25),
            Ellipse(center=Point(50, 50), radius_x=30, radius_y=15),
            RoundedRectangle(x=20, y=25, width=60, height=40, corner_radius=10),
            Arc(center=Point(50, 50), radius_x=30, radius_y=20,
                start_angle=35, sweep_angle=-270, closure=ArcClosure.PIE)]


@pytest.mark.parametrize('node', curves())
@pytest.mark.parametrize('space', ['screen', 'object'])
def test_exact_export_and_roundtrip(node, space):
    node.stroke = StrokeStyle(width=3, space=space, dash_array=(7, 4))
    node.fill = FillStyle(color=Color(100, 160, 220))
    node.clip = ClipRect(10, 10, 120, 100)
    node.transform = Transform.from_matrix(np.array([[-1.3, .4, 145], [.2, 1.2, 10], [0, 0, 1.]]))
    scene = Scene(200, 160); scene.add(node)
    svg = scene.to_svg(strict=True)
    paths = ET.fromstring(svg).findall('.//{http://www.w3.org/2000/svg}path')
    assert any('A ' in p.get('d', '') for p in paths)
    converted = node.to_path()
    other = Scene(200, 160); other.add(converted)
    assert [p.get('d') for p in paths] == [p.get('d') for p in ET.fromstring(other.to_svg(strict=True)).findall('.//{http://www.w3.org/2000/svg}path')]
    reloaded = SVGImporter().parse(svg).scene
    assert any(isinstance(c, EllipticalArcTo) for p in reloaded.find_by_type(Path) for s in p.subpaths for c in s.commands)
    restored = Scene.from_json(other.to_json()).find_by_type(Path)[0]
    assert restored.subpaths == converted.subpaths
    assert restored.stroke.space == space


@pytest.mark.parametrize('sweep', [-360, -270, -180, 0, 90, 270, 360])
@pytest.mark.parametrize('closure', list(ArcClosure))
def test_arc_flags_endpoints_and_closure(sweep, closure):
    source = Arc(center=Point(30, 40), radius_x=20, radius_y=10,
                 start_angle=20, sweep_angle=sweep, closure=closure, fill=FillStyle())
    path = source.to_path()
    commands = path.subpaths[0].commands
    arcs = [c for c in commands if isinstance(c, EllipticalArcTo)]
    assert len(arcs) == (2 if abs(sweep) == 360 else int(sweep != 0))
    assert all(c.sweep == (sweep > 0) for c in arcs)
    assert all(c.large_arc == (180 < abs(sweep) < 360) for c in arcs)
    assert isinstance(commands[-1], Close) == (closure != ArcClosure.OPEN)
    assert (path.fill is None) == (closure == ArcClosure.OPEN)
    if abs(sweep) == 360:
        assert arcs[-1].end == commands[0].point
    if sweep == 0:
        assert isinstance(commands[1], LineTo)


def test_detached_conversion_preserves_local_or_world_pose_and_independence():
    source = curves()[3]
    source.transform = Transform(rotation=27, scale_x=2)
    source.stroke = StrokeStyle(width=5, space='object')
    source.metadata = {'nested': [1]}
    source.clip = ClipRect(0, 0, 90, 90)
    group = Group(transform=Transform(translation_x=100))
    group.add(source, preserve_world_transform=False)
    local = source.to_path()
    world = source.to_path(preserve_world_transform=True)
    assert local.id != source.id != world.id
    assert local.parent is None
    np.testing.assert_allclose(world.world_matrix, source.world_matrix)
    group.add(local, preserve_world_transform=False)
    np.testing.assert_allclose(local.world_matrix, source.world_matrix)
    local.stroke.width = 10
    local.metadata['nested'].append(2)
    assert source.stroke.width == 5 and source.metadata == {'nested': [1]}
    assert local.clip is not source.clip


@pytest.mark.parametrize('node', [Line(), Rectangle(width=20, height=10),
    RoundedRectangle(width=20, height=10, corner_radius=0),
    Polygon(vertices=[Point(0, 0), Point(10, 0), Point(0, 10)]),
    Polyline(points=[Point(0, 0), Point(10, 10)]), BezierCurve(), Path()])
def test_other_vector_shapes(node):
    assert isinstance(node.to_path(), Path)


def test_empty_geometry_and_unsupported_conversion():
    assert not Circle(radius=0).to_path().subpaths
    with pytest.raises(ValidationError, match='not supported'):
        Group().to_path()
    with pytest.raises(ValidationError, match='boolean'):
        Circle().to_path(preserve_world_transform=1)


@pytest.mark.parametrize('node,primitive', [
    (curves()[0], '<circle cx="50" cy="50" r="25"/>'),
    (curves()[1], '<ellipse cx="50" cy="50" rx="30" ry="15"/>'),
    (curves()[2], '<rect x="20" y="25" width="60" height="40" rx="10"/>')])
def test_export_matches_native_svg_primitives_under_affine_transform(node, primitive):
    resvg = pytest.importorskip('resvg_py')
    def raster(svg):
        return cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=svg), np.uint8), -1)
    node.fill = FillStyle(color=Color(255, 0, 0))
    node.transform = Transform.from_matrix(np.array([[-1.3, .4, 145], [.2, 1.2, 10], [0, 0, 1.]]))
    scene = Scene(200, 160, background=Color(0, 0, 0, 0)); scene.add(node)
    expected = raster('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="160">'
                      '<g fill="red" transform="matrix(-1.3 .2 .4 1.2 145 10)">' + primitive + '</g></svg>')
    actual = raster(scene.to_svg(strict=True))
    # Different exact representations can be tessellated differently by resvg.
    assert np.mean(np.abs(actual.astype(float) - expected.astype(float))) < .4
    assert np.max(np.abs(actual[..., 3].astype(int) - expected[..., 3].astype(int))) < 50
