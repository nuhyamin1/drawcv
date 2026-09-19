"""Unit tests for topological fill rule evaluation (EVEN_ODD vs NON_ZERO)."""

import numpy as np
import pytest

from drawcv.core.enums import FillRule
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import evaluate_fill_rule_mask


STAR = [Point(x, y) for x, y in [(50, 5), (76, 86), (7, 36), (93, 36), (24, 86)]]


@pytest.mark.parametrize('supersample', [1, 2, 4])
@pytest.mark.parametrize('reverse', [False, True])
def test_self_intersecting_star_uses_local_winding(supersample, reverse):
    contour = STAR[::-1] if reverse else STAR[:]
    before = contour[:]
    nonzero = evaluate_fill_rule_mask([contour], FillRule.NON_ZERO, 100, 100, supersample)
    evenodd = evaluate_fill_rule_mask([contour], FillRule.EVEN_ODD, 100, 100, supersample)
    assert nonzero[50, 50] == 255  # Winding two, regardless of orientation.
    assert evenodd[50, 50] == 0
    assert nonzero[20, 50] == evenodd[20, 50] == 255
    assert nonzero[95, 50] == evenodd[95, 50] == 0
    assert contour == before


@pytest.mark.parametrize('opposing_copies,expected', [(0, 255), (1, 255), (2, 0)])
def test_repeated_traversal_preserves_winding_magnitude(opposing_copies, expected):
    square = [Point(10, 10), Point(90, 10), Point(90, 90), Point(10, 90)]
    # One contour travels around the square twice. Simplifying each subpath
    # independently would incorrectly cancel both traversals with one reverse loop.
    contours = [square + square] + [square[::-1]] * opposing_copies
    mask = evaluate_fill_rule_mask(contours, FillRule.NON_ZERO, 100, 100)
    assert mask[50, 50] == expected


def test_zero_signed_area_bowtie_keeps_both_lobes():
    bowtie = [Point(10, 10), Point(90, 90), Point(10, 90), Point(90, 10)]
    mask = evaluate_fill_rule_mask([bowtie], FillRule.NON_ZERO, 100, 100)
    assert mask[20, 50] == mask[80, 50] == 255
    assert mask[50, 20] == 0


@pytest.mark.parametrize('contours', [[], [[Point(10, 10)] * 3],
                                      [[Point(10, 10), Point(20, 20), Point(30, 30)]]])
def test_nonzero_empty_and_degenerate_contours(contours):
    assert not evaluate_fill_rule_mask(contours, FillRule.NON_ZERO, 50, 50).any()


@pytest.mark.parametrize('rule,expected', [(FillRule.NON_ZERO, 255), (FillRule.EVEN_ODD, 0)])
def test_star_import_render_clip_hit_test_and_svg_agree(rule, expected):
    from drawcv import Color, FillStyle, OpenCVRenderer, Path, Rectangle, Scene, SVGImporter
    from drawcv.effects.clipping import is_point_in_clip
    import cv2

    rule_name = 'nonzero' if rule == FillRule.NON_ZERO else 'evenodd'
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
           f'<path d="M50 5 L76 86 L7 36 L93 36 L24 86 Z" fill="red" fill-rule="{rule_name}"/></svg>')
    scene = SVGImporter().parse(svg).scene
    path = scene.find_by_type(Path)[0]
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    assert actual[50, 50, 3] == expected
    assert path.contains_point(Point(50, 50)) == bool(expected)

    owner = Rectangle(width=100, height=100, fill=FillStyle(color=Color(255, 0, 0)), stroke=None,
                      clip=path)
    clipped = Scene(100, 100, background=Color(0, 0, 0, 0))
    clipped.add(owner)
    assert OpenCVRenderer().render(clipped, alpha=True).buffer[50, 50, 3] == expected
    assert is_point_in_clip(path, owner, Point(50, 50)) == bool(expected)

    resvg = pytest.importorskip('resvg_py')
    for document in (svg, scene.to_svg(strict=True), clipped.to_svg(strict=True)):
        png = resvg.svg_to_bytes(svg_string=document)
        reference = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        assert reference[50, 50].tolist() == actual[50, 50].tolist()


def test_even_odd_donut_hole():
    # Outer 100x100 square from (50, 50) to (150, 150)
    outer = [Point(50, 50), Point(150, 50), Point(150, 150), Point(50, 150)]
    # Inner 40x40 square from (80, 80) to (120, 120)
    inner = [Point(80, 80), Point(120, 80), Point(120, 120), Point(80, 120)]

    mask = evaluate_fill_rule_mask([outer, inner], FillRule.EVEN_ODD, width=200, height=200, supersample=2)

    # Point in the ring (60, 60): should be fully filled (> 200)
    assert mask[60, 60] > 200
    # Point in the center hole (100, 100): should be empty (0)
    assert mask[100, 100] == 0
    # Point outside the outer ring (20, 20): should be empty (0)
    assert mask[20, 20] == 0


def test_non_zero_opposing_winding_creates_hole():
    # Outer CW square
    outer_cw = [Point(50, 50), Point(150, 50), Point(150, 150), Point(50, 150)]
    # Inner CCW square (reversed winding)
    inner_ccw = [Point(50, 50), Point(50, 150), Point(150, 150), Point(150, 50)]  # CCW
    inner_ccw_hole = [Point(80, 80), Point(80, 120), Point(120, 120), Point(120, 80)]

    mask = evaluate_fill_rule_mask([outer_cw, inner_ccw_hole], FillRule.NON_ZERO, width=200, height=200, supersample=2)

    # Ring should be filled
    assert mask[60, 60] > 200
    # Hole should have net winding 1 - 1 = 0, so it remains empty
    assert mask[100, 100] == 0


def test_non_zero_concordant_winding_fills_interior():
    # Outer CW square
    outer_cw = [Point(50, 50), Point(150, 50), Point(150, 150), Point(50, 150)]
    # Inner CW square (same orientation: +1 + 1 = +2)
    inner_cw = [Point(80, 80), Point(120, 80), Point(120, 120), Point(80, 120)]

    mask = evaluate_fill_rule_mask([outer_cw, inner_cw], FillRule.NON_ZERO, width=200, height=200, supersample=2)

    # Ring should be filled
    assert mask[60, 60] > 200
    # Center hole is inside both concordant contours (+2 != 0), so NON_ZERO considers it filled!
    assert mask[100, 100] > 200
