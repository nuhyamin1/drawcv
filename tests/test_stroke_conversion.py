import cv2
import numpy as np
import pytest

from drawcv import (Path, Point, Line, Circle, Group, StrokeStyle, CapStyle, JoinStyle,
                    Color, Scene, OpenCVRenderer, Transform, ValidationError,
                    LinearGradient, GradientStop, FillStyle, SVGImporter)


def render(node):
    scene = Scene(220, 180, background=Color(0, 0, 0, 0))
    scene.add(node)
    return OpenCVRenderer().render(scene, alpha=True).buffer


@pytest.mark.parametrize('space', ['screen', 'object'])
@pytest.mark.parametrize('cap', list(CapStyle))
@pytest.mark.parametrize('join', list(JoinStyle))
@pytest.mark.parametrize('dash', [(), (13, 7, 4)])
def test_outline_matches_stroke(space, cap, join, dash):
    node = Path(stroke=StrokeStyle(width=9, space=space, cap_style=cap,
                join_style=join, dash_array=dash, dash_offset=-4, miter_limit=2))
    node.move_to(15, 20).line_to(65, 25).line_to(40, 65)
    node.transform = Transform.from_matrix(np.array([[-1.5, .3, 170], [.2, 1.6, 10], [0, 0, 1.]]))
    outline = node.stroke_to_path(tolerance=.1)
    actual, expected = render(outline)[..., 3], render(node)[..., 3]
    kernel = np.ones((5, 5), np.uint8)
    interior = cv2.erode((expected == 255).astype(np.uint8), kernel) > 0
    exterior = cv2.dilate((expected > 0).astype(np.uint8), kernel) == 0
    assert interior.any() and actual[interior].min() > 240
    assert not actual[exterior].any()
    assert outline.stroke is None and outline.parent is None and outline.id != node.id
    np.testing.assert_array_equal(outline.world_matrix, np.eye(3))


def test_closed_stroke_has_hole_and_overlaps_do_not_cancel():
    node = Circle(center=Point(70, 70), radius=40, stroke=StrokeStyle(width=12))
    outline = node.stroke_to_path()
    assert not outline.contains_point(Point(70, 70))
    assert outline.contains_point(Point(110, 70))
    node = Path(stroke=StrokeStyle(width=10))
    node.move_to(20, 20).line_to(80, 80).line_to(20, 80).line_to(80, 20)
    assert node.stroke_to_path().contains_point(Point(50, 50))


@pytest.mark.parametrize('cap', list(CapStyle))
def test_zero_length_and_empty_strokes(cap):
    node = Line(start=Point(40, 40), end=Point(40, 40), stroke=StrokeStyle(width=10, cap_style=cap))
    outline = node.stroke_to_path()
    assert bool(outline.subpaths) == (cap != CapStyle.BUTT)
    assert not Circle(radius=10).stroke_to_path().subpaths


def test_paint_parent_transform_roundtrips_and_boolean():
    paint = LinearGradient(Point(0, 0), Point(80, 0),
            (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255))))
    node = Line(start=Point(10, 30), end=Point(80, 30),
                stroke=StrokeStyle(width=10, paint=paint, space='object', opacity=.7))
    group = Group(transform=Transform(scale_x=2, scale_y=3, pivot=Point(0, 0)))
    group.add(node, preserve_world_transform=False)
    outline = node.stroke_to_path()
    assert outline.fill.opacity == .7 and outline.fill.paint is not paint
    np.testing.assert_allclose(outline.fill.paint.transform.get_matrix(), node.world_matrix)
    assert outline.fill.paint.space == 'world'
    actual, expected = render(outline), render(group)
    np.testing.assert_allclose(actual[90, 80].astype(int), expected[90, 80].astype(int), atol=2)
    assert outline.union(outline).contains_point(Point(80, 90))
    scene = Scene(220, 180); scene.add(outline)
    restored = Scene.from_json(scene.to_json()).find_by_type(Path)[0]
    assert restored.subpaths == outline.subpaths
    assert SVGImporter().parse(scene.to_svg(strict=True)).scene.find_by_type(Path)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, 'small'])
def test_invalid_tolerance(value):
    with pytest.raises(ValidationError):
        Line().stroke_to_path(tolerance=value)


def test_unsupported_type_and_tolerance_refinement():
    with pytest.raises(ValidationError, match='not supported'):
        Group().stroke_to_path()
    node = Circle(center=Point(50, 50), radius=35, stroke=StrokeStyle(width=10))
    coarse, fine = node.stroke_to_path(tolerance=1), node.stroke_to_path(tolerance=.05)
    assert sum(len(s.commands) for s in fine.subpaths) > sum(len(s.commands) for s in coarse.subpaths)


def test_dash_positions_and_singular_transform():
    node = Line(start=Point(0, 10), end=Point(100, 10),
                stroke=StrokeStyle(width=4, cap_style=CapStyle.BUTT, dash_array=(10, 10)))
    outline = node.stroke_to_path()
    assert len(outline.subpaths) == 5
    coverage = render(outline)[..., 3]
    for x in range(5, 100, 10):
        assert (coverage[10, x] == 255) == (x % 20 == 5)
    node.stroke.space = 'object'
    node.transform = Transform.from_matrix(np.diag([2., 0., 1.]))
    assert not node.stroke_to_path().subpaths


def test_closed_dashed_outline_against_svg_reference():
    resvg = pytest.importorskip('resvg_py')
    node = Path(stroke=StrokeStyle(width=10, space='object', cap_style=CapStyle.SQUARE,
                join_style=JoinStyle.MITER, dash_array=(30, 9), dash_offset=5))
    node.move_to(30, 30).line_to(140, 30).line_to(100, 120).close()
    original = Scene(220, 180, background=Color(0, 0, 0, 0)); original.add(node)
    converted = Scene(220, 180, background=Color(0, 0, 0, 0)); converted.add(node.stroke_to_path(tolerance=.05))
    def alpha(scene):
        return cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=scene.to_svg(strict=True)), np.uint8), -1)[..., 3]
    expected, actual = alpha(original), alpha(converted)
    assert np.mean(np.abs(expected.astype(float) - actual.astype(float))) < .2
