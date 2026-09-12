"""Unit tests for topological fill rule evaluation (EVEN_ODD vs NON_ZERO)."""

import numpy as np
import pytest

from drawcv.core.enums import FillRule
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import evaluate_fill_rule_mask


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
