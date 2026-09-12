"""Unit tests for Transform affine matrix calculations, inverses, and dynamic pivots."""

import math
import pytest

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform


def test_transform_identity():
    t = Transform()
    assert t.is_identity()
    assert t.is_translation_only()
    m = t.get_matrix()
    inv_m = t.get_inverse_matrix()
    assert math.isclose(m[0, 0], 1.0) and math.isclose(m[1, 1], 1.0)
    assert math.isclose(inv_m[0, 0], 1.0) and math.isclose(inv_m[1, 1], 1.0)


def test_transform_scale_validation():
    with pytest.raises(ValidationError):
        Transform(scale_x=0)

    with pytest.raises(ValidationError):
        Transform(scale_y=-1.5)


def test_transform_dynamic_pivot_vs_explicit_pivot():
    t_dynamic = Transform(rotation=90.0, pivot=None)
    t_explicit = Transform(rotation=90.0, pivot=Point(50, 50))

    # Resolving with default_pivot Point(50, 50)
    p_orig = Point(60, 50)
    p_trans1 = t_dynamic.transform_point(p_orig, default_pivot=Point(50, 50))
    p_trans2 = t_explicit.transform_point(p_orig, default_pivot=Point(0, 0))

    # 10 units right of (50, 50) rotated 90° clockwise -> (50, 60)
    assert math.isclose(p_trans1.x, 50.0, abs_tol=1e-5)
    assert math.isclose(p_trans1.y, 60.0, abs_tol=1e-5)
    assert math.isclose(p_trans2.x, 50.0, abs_tol=1e-5)
    assert math.isclose(p_trans2.y, 60.0, abs_tol=1e-5)


def test_transform_inverse_roundtrip():
    t = Transform(
        translation_x=120.0,
        translation_y=-45.0,
        rotation=37.5,
        scale_x=1.8,
        scale_y=0.6,
        pivot=Point(100, 100)
    )

    pt = Point(145.2, 88.7)
    world_pt = t.transform_point(pt)
    local_roundtrip = t.inverse_transform_point(world_pt)

    assert math.isclose(local_roundtrip.x, pt.x, abs_tol=1e-5)
    assert math.isclose(local_roundtrip.y, pt.y, abs_tol=1e-5)
