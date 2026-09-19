import math

import numpy as np
import pytest

from drawcv import Path, Point, Circle, Group, Transform, StrokeStyle, FillStyle, Scene, SVGImporter, ValidationError
from drawcv.shapes.path import QuadraticTo, CubicTo, EllipticalArcTo, LineTo, Close, Subpath, MoveTo


def xy(p):
    return [p.x, p.y]


def paths():
    return [Path().move_to(0, 0).line_to(100, 0),
            Path().move_to(0, 0).quadratic_to(Point(0, 100), Point(100, 0)),
            Path().move_to(0, 0).cubic_to(Point(0, 100), Point(100, 100), Point(100, 0)),
            Path().move_to(30, 0).arc_to(30, 15, 20, True, False, Point(-10, 10))]


@pytest.mark.parametrize('source', paths())
@pytest.mark.parametrize('interval', [(0, .4), (.2, .8), (.4, 1)])
def test_trim_retains_curve_type_length_and_endpoints(source, interval):
    before = source.to_dict()
    a, b = interval
    result = source.trim(a, b, tolerance=1e-6)
    assert type(result.subpaths[0].commands[1]) is type(source.subpaths[0].commands[1])
    assert result.length(tolerance=1e-6) == pytest.approx((b-a)*source.length(tolerance=1e-6), abs=2e-5)
    np.testing.assert_allclose(xy(result.point_at(0)), xy(source.point_at(a, tolerance=1e-6)), atol=1e-6)
    np.testing.assert_allclose(xy(result.point_at(1)), xy(source.point_at(b, tolerance=1e-6)), atol=1e-6)
    for fraction in (.15, .5, .85):
        np.testing.assert_allclose(xy(result.point_at(fraction, tolerance=1e-6)),
            xy(source.point_at(a+(b-a)*fraction, tolerance=1e-6)), atol=2e-5)
    assert source.to_dict() == before


@pytest.mark.parametrize('source', paths())
def test_split_conserves_length_and_has_shared_cut(source):
    left, right = source.split_at(.37, tolerance=1e-6)
    assert left.id != right.id != source.id
    assert left.length(tolerance=1e-6)+right.length(tolerance=1e-6) == pytest.approx(source.length(tolerance=1e-6), abs=2e-5)
    np.testing.assert_allclose(xy(left.point_at(1)), xy(right.point_at(0)), atol=1e-8)


def test_disconnected_boundaries_no_bridges_and_subpath_selection():
    source = Path().move_to(0, 0).line_to(10, 0).move_to(100, 100).line_to(110, 100)
    result = source.trim(.25, .75)
    assert len(result.subpaths) == 2
    assert result.length() == pytest.approx(10)
    left, right = source.split_at(.5)
    np.testing.assert_allclose(xy(left.point_at(1)), [10, 0])
    np.testing.assert_allclose(xy(right.point_at(0)), [100, 100])
    assert source.trim(.2, .8, subpath=1).length() == pytest.approx(6)


def test_closed_curves_and_implicit_close():
    source = Circle(center=Point(0, 0), radius=10).to_path()
    assert source.trim(0, 1).subpaths == source.subpaths
    cut = source.trim(.2, .8, tolerance=1e-7)
    assert not cut.subpaths[0].closed
    assert not any(isinstance(c, Close) for c in cut.subpaths[0].commands)
    assert cut.length() == pytest.approx(12*math.pi)
    triangle = Path(subpaths=[Subpath([MoveTo(Point(0, 0)), LineTo(Point(3, 0)), LineTo(Point(3, 4))], closed=True)])
    closing = triangle.trim(7/12, 1)
    assert closing.length() == pytest.approx(5)
    np.testing.assert_allclose(xy(closing.point_at(0)), [3, 4])
    np.testing.assert_allclose(xy(closing.point_at(1)), [0, 0])


def test_whole_inner_contour_keeps_closure():
    source = Path().move_to(0, 0).line_to(10, 0)
    source.move_to(30, 0).line_to(40, 0).line_to(40, 10).line_to(30, 10).close()
    source.move_to(90, 0).line_to(100, 0)
    result = source.trim(1/6, 5/6)
    assert len(result.subpaths) == 1 and result.subpaths[0].closed


def test_world_metric_and_transform_styles_independence():
    source = Path(stroke=StrokeStyle(width=4, space='object'), fill=FillStyle(), metadata={'nested': [1]})
    source.move_to(0, 0).line_to(10, 0).line_to(10, 10)
    parent = Group(transform=Transform.from_matrix(np.array([[2., .3, 10], [0, 3., 20], [0, 0, 1.]])))
    parent.add(source, preserve_world_transform=False)
    result = source.trim(.2, .8, space='world', preserve_world_transform=True)
    assert result.parent is None
    np.testing.assert_allclose(result.world_matrix, source.world_matrix)
    np.testing.assert_allclose(xy(result.point_at(0, space='world')), xy(source.point_at(.2, space='world')), atol=1e-6)
    assert result.length(space='world') == pytest.approx(.6*source.length(space='world'))
    result.stroke.width = 8
    result.metadata['nested'].append(2)
    assert source.stroke.width == 4 and source.metadata == {'nested': [1]}
    local = source.trim(.2, .8)
    parent.add(local, preserve_world_transform=False)
    np.testing.assert_allclose(local.world_matrix, source.world_matrix)


def test_empty_ranges_degenerate_paths_and_endpoint_splits():
    source = paths()[0]
    assert not source.trim(.5, .5).subpaths
    left, right = source.split_at(0)
    assert not left.subpaths and right.subpaths == source.subpaths
    left, right = source.split_at(1)
    assert not right.subpaths and left.subpaths == source.subpaths
    assert not Path().trim(.2, .8).subpaths
    zero = Path().move_to(5, 6).line_to(5, 6)
    assert zero.trim(0, 1).subpaths == zero.subpaths
    assert not zero.trim(.2, .8).subpaths


def test_json_svg_roundtrip_and_large_corrected_arc():
    source = Path().move_to(0, 0).arc_to(10, 5, 35, True, True, Point(100, 50))
    result = source.trim(.1, .9, tolerance=1e-7)
    cmd = result.subpaths[0].commands[1]
    assert isinstance(cmd, EllipticalArcTo) and cmd.radius_x > 10
    scene = Scene(200, 200); scene.add(result)
    assert Scene.from_json(scene.to_json()).find_by_type(Path)[0].subpaths == result.subpaths
    imported = SVGImporter().parse(scene.to_svg(strict=True)).scene.find_by_type(Path)[0]
    assert imported.length() == pytest.approx(result.length(), abs=.001)


@pytest.mark.parametrize('start,end', [(-.1, 1), (0, 1.1), (.8, .2), (True, 1), (0, float('nan'))])
def test_invalid_ranges(start, end):
    with pytest.raises(ValidationError):
        paths()[0].trim(start, end)


def test_invalid_options_and_split_progress():
    source = paths()[0]
    for kwargs in ({'space': 'screen'}, {'tolerance': 0}, {'subpath': 3}, {'preserve_world_transform': 1}):
        with pytest.raises(ValidationError):
            source.trim(0, 1, **kwargs)
    with pytest.raises(ValidationError):
        source.split_at(float('inf'))
