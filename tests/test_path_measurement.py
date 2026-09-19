import math

import numpy as np
import pytest

from drawcv import Path, Point, Circle, Group, Transform, ValidationError
from drawcv.shapes.path import Subpath, MoveTo, LineTo


def assert_point(actual, x, y, tolerance=1e-6):
    np.testing.assert_allclose([actual.x, actual.y], [x, y], atol=tolerance)


def test_line_corner_and_closed_distance():
    path = Path().move_to(0, 0).line_to(30, 0).line_to(30, 40)
    assert path.length() == pytest.approx(70)
    assert_point(path.point_at(.5), 30, 5)
    assert_point(path.tangent_at(30/70), 1, 0)
    assert_point(path.tangent_at(1), 0, 1)
    path.close()
    assert path.length() == pytest.approx(120)
    assert_point(path.point_at(1), 0, 0)
    assert_point(path.tangent_at(1), -.6, -.8)


def test_disconnected_subpaths_and_implicit_closure():
    path = Path().move_to(0, 0).line_to(10, 0).move_to(100, 100).line_to(100, 110)
    assert path.length() == pytest.approx(20)
    assert_point(path.point_at(.5), 10, 0)
    assert_point(path.point_at(.75), 100, 105)
    assert_point(path.tangent_at(.5), 1, 0)
    assert_point(path.point_at(.5, subpath=1), 100, 105)
    assert path.length(subpath=1) == pytest.approx(10)
    path.subpaths = [Subpath([MoveTo(Point(0, 0)), LineTo(Point(3, 0)), LineTo(Point(3, 4))], closed=True)]
    assert path.length() == pytest.approx(12)


def test_curves_have_arc_length_not_parameter_progress():
    path = Path().move_to(0, 0).quadratic_to(Point(0, 0), Point(100, 0))
    assert path.length(tolerance=1e-8) == pytest.approx(100)
    assert_point(path.point_at(.25), 25, 0)
    assert_point(path.tangent_at(0), 1, 0)
    cubic = Path().move_to(0, 0).cubic_to(Point(0, 100), Point(100, 100), Point(100, 0))
    assert cubic.length(tolerance=1e-8) == pytest.approx(200)
    assert_point(cubic.point_at(.5), 50, 75)
    assert_point(cubic.tangent_at(.5), 1, 0)


def test_collinear_reversal_and_closed_loop_are_not_flattened_away():
    path = Path().move_to(0, 0).quadratic_to(Point(100, 0), Point(0, 0))
    assert path.length(tolerance=1e-7) == pytest.approx(100, abs=1e-6)
    assert_point(path.point_at(.25, tolerance=1e-7), 25, 0)
    assert_point(path.tangent_at(.75), -1, 0)
    cubic = Path().move_to(0, 0).cubic_to(Point(100, 100), Point(-100, 100), Point(0, 0))
    assert cubic.length() > 200


def test_circular_arcs_and_reverse_sweep():
    path = Circle(center=Point(20, 30), radius=10).to_path()
    assert path.length(tolerance=1e-8) == pytest.approx(20*math.pi)
    assert_point(path.point_at(.25), 20, 40)
    assert_point(path.tangent_at(.25), -1, 0)
    reverse = Path().move_to(10, 0).arc_to(10, 10, 0, False, False, Point(0, -10))
    assert reverse.length() == pytest.approx(5*math.pi)
    assert_point(reverse.tangent_at(0), 0, -1)


def test_world_transform_and_parent_change_distance_and_direction():
    path = Path().move_to(0, 0).line_to(10, 0).line_to(10, 10)
    group = Group(transform=Transform.from_matrix(np.array([[-2., 0, 50], [0, 3, 10], [0, 0, 1.]])))
    group.add(path, preserve_world_transform=False)
    assert path.length() == pytest.approx(20)
    assert path.length(space='world') == pytest.approx(50)
    assert_point(path.point_at(.5, space='world'), 30, 15)
    assert_point(path.tangent_at(0, space='world'), -1, 0)
    path.line_to(20, 10)
    assert path.length(space='world') == pytest.approx(70)


def test_affine_ellipse_length_against_dense_independent_samples():
    path = Circle(center=Point(0, 0), radius=30).to_path()
    matrix = np.array([[2., .7, 10], [.3, .5, 20], [0, 0, 1.]])
    path.transform = Transform.from_matrix(matrix)
    angles = np.linspace(0, 2*math.pi, 100001)
    points = 30*np.column_stack((np.cos(angles), np.sin(angles))) @ matrix[:2, :2].T
    expected = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert path.length(space='world', tolerance=1e-7) == pytest.approx(expected, abs=1e-5)
    derivative = matrix[:2, :2] @ [0., 1.]
    assert_point(path.tangent_at(0, space='world'), *(derivative/np.linalg.norm(derivative)))


def test_degenerate_and_singular_paths():
    empty = Path()
    assert empty.length() == 0
    with pytest.raises(ValidationError, match='empty'):
        empty.point_at(0)
    path = Path().move_to(4, 5).line_to(4, 5)
    assert path.length() == 0
    assert_point(path.point_at(.8), 4, 5)
    with pytest.raises(ValidationError, match='tangent'):
        path.tangent_at(.2)
    path.line_to(8, 10)
    path.transform = Transform.from_matrix(np.diag([0., 0., 1.]))
    assert path.length(space='world') == 0
    with pytest.raises(ValidationError):
        path.tangent_at(0, space='world')


@pytest.mark.parametrize('kwargs', [{'tolerance': 0}, {'tolerance': True}, {'tolerance': float('nan')},
    {'space': 'screen'}, {'subpath': -1}, {'subpath': True}, {'subpath': 1}])
def test_invalid_options(kwargs):
    with pytest.raises(ValidationError):
        Path().move_to(0, 0).line_to(1, 0).length(**kwargs)


@pytest.mark.parametrize('progress', [-.1, 1.1, True, float('nan'), float('inf'), 'half'])
def test_invalid_progress(progress):
    path = Path().move_to(0, 0).line_to(1, 0)
    for query in (path.point_at, path.tangent_at):
        with pytest.raises(ValidationError):
            query(progress)


def test_move_only_endpoints_and_zero_length_arc():
    path = Path().move_to(4, 5).move_to(10, 10).line_to(20, 10).move_to(90, 90)
    assert path.length() == 10
    assert_point(path.point_at(0), 4, 5)
    assert_point(path.point_at(1), 90, 90)
    assert_point(path.tangent_at(0), 1, 0)
    assert_point(path.tangent_at(1), 1, 0)
    arc = Path().move_to(10, 0).arc_to(10, 10, 0, True, True, Point(10, 0))
    assert arc.length() == 0


def test_queries_leave_semantic_state_unchanged():
    path = Circle(center=Point(20, 30), radius=10).to_path()
    before = path.to_dict()
    path.length(); path.point_at(.3); path.tangent_at(.7)
    assert path.to_dict() == before
