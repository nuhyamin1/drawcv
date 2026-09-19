import math

import numpy as np
import pytest

from drawcv import (Path, Circle, Point, JoinStyle, FillRule, Transform, Group,
                    FillStyle, StrokeStyle, LinearGradient, GradientStop, Color,
                    Scene, SVGImporter, ValidationError)


def box(x=0, y=0, size=100):
    return Path().move_to(x, y).line_to(x+size, y).line_to(x+size, y+size).line_to(x, y+size).close()


def area(path):
    total = 0
    for points in path.flatten_world():
        total += sum(p.x*q.y-q.x*p.y for p, q in zip(points, points[1:]+points[:1])) / 2
    return abs(total)


@pytest.mark.parametrize('distance', [10, -10, 0])
def test_square_miter_has_exact_bounds_and_area(distance):
    source = box()
    result = source.offset(distance, join_style=JoinStyle.MITER)
    b = result.get_geometry_bounds()
    np.testing.assert_allclose([b.x, b.y, b.width, b.height], [-distance, -distance, 100+2*distance, 100+2*distance])
    assert area(result) == pytest.approx((100+2*distance)**2)
    assert result.stroke is None and all(s.closed for s in result.subpaths)


@pytest.mark.parametrize('join,expected', [(JoinStyle.ROUND, 14000+100*math.pi),
    (JoinStyle.BEVEL, 14200), (JoinStyle.MITER, 14400)])
def test_join_areas_and_miter_limit(join, expected):
    result = box().offset(10, join_style=join, tolerance=.01)
    assert area(result) == pytest.approx(expected, abs=1)
    bevel = box().offset(10, join_style=JoinStyle.MITER, miter_limit=1)
    assert area(bevel) == pytest.approx(14200)


def test_holes_contract_and_expand_and_orientation_does_not_matter():
    source = box()
    source.subpaths.extend(box(30, 30, 40).subpaths)
    source.fill_rule = FillRule.EVEN_ODD
    assert area(source.offset(10, join_style=JoinStyle.MITER)) == pytest.approx(14400-400)
    assert area(source.offset(-10, join_style=JoinStyle.MITER)) == pytest.approx(6400-3600)
    reversed_path = Path().move_to(0, 0).line_to(0, 100).line_to(100, 100).line_to(100, 0).close()
    assert area(reversed_path.offset(10, join_style=JoinStyle.MITER)) == pytest.approx(14400)


def test_nonzero_overlap_is_normalized_before_erosion():
    source = box(0, 0, 100)
    source.subpaths.extend(box(50, 0, 100).subpaths)
    source.fill_rule = FillRule.NON_ZERO
    assert area(source.offset(-10, join_style=JoinStyle.MITER)) == pytest.approx(130*80)
    assert len(source.offset(-10).subpaths) == 1


def test_large_erosion_disappears_and_dilation_merges_islands():
    assert not box().offset(-60).subpaths
    source = box(0, 0, 20)
    source.subpaths.extend(box(30, 0, 20).subpaths)
    assert len(source.offset(6).subpaths) == 1
    assert not Path().offset(10).subpaths


@pytest.mark.parametrize('distance', [-8, 8])
def test_circle_against_analytic_area(distance):
    source = Circle(center=Point(50, 50), radius=30).to_path()
    result = source.offset(distance, tolerance=.02)
    assert area(result) == pytest.approx(math.pi*(30+distance)**2, abs=4)


def test_world_and_local_distance_and_paint_mapping():
    source = box()
    source.stroke = StrokeStyle(width=3, space='object')
    source.fill = FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 0),
            (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))))
    parent = Group(transform=Transform(scale_x=2, scale_y=3, pivot=Point(0, 0)))
    parent.add(source, preserve_world_transform=False)
    before = source.to_dict()
    world = source.offset(10, join_style=JoinStyle.MITER)
    local = source.offset(10, space='local', join_style=JoinStyle.MITER)
    assert area(world) == pytest.approx(220*320)
    assert area(local) == pytest.approx(240*360)
    np.testing.assert_allclose(world.world_matrix, np.eye(3))
    np.testing.assert_allclose(local.world_matrix, source.world_matrix)
    np.testing.assert_allclose(world.fill.paint.transform.get_matrix(), source.world_matrix)
    assert world.fill.paint.space == 'world' and local.fill.paint.space == 'object'
    assert source.to_dict() == before and world.parent is None and world.id != source.id


def test_roundtrips_and_boolean_compatibility():
    result = box().offset(10)
    assert result.difference(box()).subpaths
    scene = Scene(200, 200); scene.add(result)
    assert Scene.from_json(scene.to_json()).find_by_type(Path)[0].subpaths == result.subpaths
    imported = SVGImporter().parse(scene.to_svg(strict=True)).scene.find_by_type(Path)[0]
    assert area(imported) == pytest.approx(area(result), abs=.1)


@pytest.mark.parametrize('options', [{'distance': True}, {'distance': float('nan')},
    {'distance': float('inf')}, {'distance': 2, 'tolerance': 0},
    {'distance': 2, 'space': 'screen'}, {'distance': 2, 'join_style': 'unknown'},
    {'distance': 2, 'miter_limit': 0}])
def test_invalid_arguments(options):
    with pytest.raises(ValidationError):
        box().offset(**options)


def test_open_paths_rejected_without_mutation():
    source = Path().move_to(0, 0).line_to(100, 0)
    before = source.to_dict()
    with pytest.raises(ValidationError, match='closed'):
        source.offset(10)
    assert source.to_dict() == before


def test_erosion_splits_narrow_neck_and_resolves_self_intersection():
    source = Path().move_to(0, 0)
    for p in [(40,0), (40,15), (60,15), (60,0), (100,0), (100,40),
              (60,40), (60,25), (40,25), (40,40), (0,40)]:
        source.line_to(*p)
    source.close()
    assert len(source.offset(-6).subpaths) == 2
    bow = Path().move_to(0,0).line_to(50,50).line_to(0,50).line_to(50,0).close()
    assert area(bow.offset(2)) > 1250
    assert area(bow.offset(-2)) < 1250


def test_reflection_singular_transform_and_tolerance_refinement():
    source = box()
    source.transform = Transform.from_matrix(np.diag([-2., 3., 1.]))
    assert area(source.offset(10, join_style=JoinStyle.MITER)) == pytest.approx(220*320)
    source.transform = Transform.from_matrix(np.diag([2., 0., 1.]))
    assert not source.offset(10).subpaths
    circle = Circle(center=Point(50,50), radius=30).to_path()
    coarse = circle.offset(5, tolerance=1)
    fine = circle.offset(5, tolerance=.1)
    assert abs(area(fine)-math.pi*35**2) < abs(area(coarse)-math.pi*35**2)
