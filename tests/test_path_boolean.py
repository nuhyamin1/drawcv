"""Comprehensive test suite for DrawCV true vector path boolean operations."""

import copy
import json
import math
import numpy as np
import pytest
import pathops

from drawcv import (
    Color,
    FillRule,
    FillStyle,
    GradientStop,
    Group,
    LinearGradient,
    MoveTo,
    LineTo,
    OpenCVRenderer,
    Path,
    PathBooleanError,
    PathBooleanOp,
    Point,
    QuadraticTo,
    CubicTo,
    RadialGradient,
    Scene,
    StrokeStyle,
    Subpath,
    SVGExporter,
    Transform,
    ValidationError,
    to_json,
)
from drawcv.core.path_boolean import (
    drawcv_to_pathops,
    pathops_to_drawcv,
    _CONIC_TO_QUAD_TOLERANCE,
)


def make_rect_path(x: float, y: float, w: float, h: float, fill: bool = True) -> Path:
    """Helper creating a closed rectangular Path with default solid fill for interior hit-testing."""
    p = Path()
    if fill:
        p.fill = FillStyle(color=Color.black())
    p.move_to(x, y).line_to(x + w, y).line_to(x + w, y + h).line_to(x, y + h).close()
    return p


# -----------------------------------------------------------------------------
# 1. Simple Overlapping Rectangles
# -----------------------------------------------------------------------------

def test_overlapping_rectangles_boolean_operations():
    # R1: [0, 0] to [100, 100]
    r1 = make_rect_path(0, 0, 100, 100)
    # R2: [50, 50] to [150, 150]
    r2 = make_rect_path(50, 50, 100, 100)

    # Union: covers [0, 150] x [0, 150] except corners (0..50, 100..150) and (100..150, 0..50)
    union_p = r1.union(r2)
    assert union_p.contains_point(Point(25, 25))
    assert union_p.contains_point(Point(75, 75))
    assert union_p.contains_point(Point(125, 125))
    assert not union_p.contains_point(Point(25, 125))
    assert not union_p.contains_point(Point(125, 25))
    assert not union_p.contains_point(Point(160, 160))

    # Intersection: [50, 100] x [50, 100]
    inter_p = r1.intersection(r2)
    assert inter_p.contains_point(Point(75, 75))
    assert not inter_p.contains_point(Point(25, 25))
    assert not inter_p.contains_point(Point(125, 125))

    # Difference: R1 minus R2 -> covers R1 except [50, 100] x [50, 100]
    diff_p = r1.difference(r2)
    assert diff_p.contains_point(Point(25, 25))
    assert diff_p.contains_point(Point(25, 75))
    assert diff_p.contains_point(Point(75, 25))
    assert not diff_p.contains_point(Point(75, 75))
    assert not diff_p.contains_point(Point(125, 125))

    # XOR: Union minus Intersection
    xor_p = r1.xor(r2)
    assert xor_p.contains_point(Point(25, 25))
    assert xor_p.contains_point(Point(125, 125))
    assert not xor_p.contains_point(Point(75, 75))


def test_generic_boolean_method():
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    u1 = r1.boolean(r2, PathBooleanOp.UNION)
    u2 = r1.boolean(r2, "union")
    u3 = r1.boolean(r2, "UNION ")
    assert u1.contains_point(Point(25, 25))
    assert u2.contains_point(Point(25, 25))
    assert u3.contains_point(Point(25, 25))


# -----------------------------------------------------------------------------
# 2. Disjoint Paths
# -----------------------------------------------------------------------------

def test_disjoint_paths():
    r1 = make_rect_path(0, 0, 40, 40)
    r2 = make_rect_path(100, 100, 40, 40)

    # Union: two separate regions
    u = r1.union(r2)
    assert u.contains_point(Point(20, 20))
    assert u.contains_point(Point(120, 120))
    assert not u.contains_point(Point(60, 60))

    # Intersection: empty
    inter = r1.intersection(r2)
    assert len(inter.subpaths) == 0 or not inter.contains_point(Point(20, 20))
    assert not inter.contains_point(Point(120, 120))

    # Difference: original left region
    d = r1.difference(r2)
    assert d.contains_point(Point(20, 20))
    assert not d.contains_point(Point(120, 120))

    # XOR: both regions
    x = r1.xor(r2)
    assert x.contains_point(Point(20, 20))
    assert x.contains_point(Point(120, 120))


# -----------------------------------------------------------------------------
# 3. Identical Paths (Geometric Equivalence)
# -----------------------------------------------------------------------------

def test_identical_paths():
    r1 = make_rect_path(10, 10, 80, 80)
    r2 = make_rect_path(10, 10, 80, 80)

    # A union A = A
    u = r1.union(r2)
    assert u.contains_point(Point(50, 50))
    assert not u.contains_point(Point(5, 5))
    gb = u.get_geometry_bounds()
    assert gb.x == pytest.approx(10, abs=0.1)
    assert gb.y == pytest.approx(10, abs=0.1)
    assert gb.width == pytest.approx(80, abs=0.1)
    assert gb.height == pytest.approx(80, abs=0.1)

    # A intersect A = A
    i = r1.intersection(r2)
    assert i.contains_point(Point(50, 50))
    assert not i.contains_point(Point(5, 5))

    # A diff A = empty
    d = r1.difference(r2)
    assert len(d.subpaths) == 0 or not d.contains_point(Point(50, 50))

    # A xor A = empty
    x = r1.xor(r2)
    assert len(x.subpaths) == 0 or not x.contains_point(Point(50, 50))


# -----------------------------------------------------------------------------
# 4. Contained Shape / Donut Hole
# -----------------------------------------------------------------------------

def test_contained_shape_hole():
    # Outer 100x100 minus inner 50x50
    outer = make_rect_path(0, 0, 100, 100)
    outer.fill = FillStyle(color=Color.red())
    inner = make_rect_path(25, 25, 50, 50)

    donut = outer.difference(inner)

    # Geometry probe
    assert donut.contains_point(Point(10, 10))
    assert not donut.contains_point(Point(50, 50))
    assert not donut.contains_point(Point(120, 120))

    # Render probe on OpenCV Canvas
    scene = Scene(120, 120, background=Color(0, 0, 0, 0))
    scene.add(donut)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=True)
    buf = canvas.to_numpy()

    # Center of hole must be fully transparent (alpha == 0)
    assert buf[50, 50, 3] == 0

    # Ring must be filled with red (alpha > 200)
    assert buf[10, 10, 3] > 200
    assert buf[10, 10, 2] > 200  # Red channel in BGR is index 2

    # Outside must be transparent
    assert buf[115, 115, 3] == 0

    # Serialization roundtrip of hole
    json_data = to_json(donut.to_dict())
    restored = Path.from_dict(json.loads(json_data))
    assert restored.contains_point(Point(10, 10))
    assert not restored.contains_point(Point(50, 50))

    # SVG strict export of hole
    svg_res = SVGExporter(strict=True).render(scene)
    assert "<path" in svg_res.svg
    assert not svg_res.fallbacks  # Pure vector, no raster fallback


# -----------------------------------------------------------------------------
# 5. Touching Geometry (Edges and Vertices)
# -----------------------------------------------------------------------------

def test_touching_edge():
    # Two rectangles sharing edge x=50
    r1 = make_rect_path(0, 0, 50, 50)
    r2 = make_rect_path(50, 0, 50, 50)

    u = r1.union(r2)
    assert u.contains_point(Point(25, 25))
    assert u.contains_point(Point(75, 25))
    gb = u.get_geometry_bounds()
    assert gb.width == pytest.approx(100, abs=0.1)
    assert gb.height == pytest.approx(50, abs=0.1)


def test_touching_vertex():
    # Rectangles touching at corner (50, 50)
    r1 = make_rect_path(0, 0, 50, 50)
    r2 = make_rect_path(50, 50, 50, 50)

    u = r1.union(r2)
    assert u.contains_point(Point(25, 25))
    assert u.contains_point(Point(75, 75))
    assert not u.contains_point(Point(25, 75))
    assert not u.contains_point(Point(75, 25))


# -----------------------------------------------------------------------------
# 6. Bezier Curves (Direct preservation, no flattening)
# -----------------------------------------------------------------------------

def test_adapter_preserves_curves_without_flattening():
    """Verify that DrawCV QuadraticTo and CubicTo map directly to PathOps QUAD and CUBIC."""
    p = Path()
    p.move_to(0, 0)
    p.quadratic_to(Point(20, 40), Point(50, 0))
    p.cubic_to(Point(60, -30), Point(80, 50), Point(100, 0))
    p.close()

    p_ops = drawcv_to_pathops(p)
    verbs = p_ops.verbs
    assert pathops.PathVerb.QUAD in verbs
    assert pathops.PathVerb.CUBIC in verbs
    # Total verbs must be small (not hundreds of LINE verbs from flattening)
    assert len(verbs) <= 6


def test_quadratic_bezier_boolean():
    # Shape with quadratic top edge
    p1 = Path(fill=FillStyle(color=Color.black()))
    p1.move_to(0, 0).quadratic_to(Point(50, -50), Point(100, 0)).line_to(100, 100).line_to(0, 100).close()

    # Cutter on the right half
    cutter = make_rect_path(50, -60, 70, 120)

    diff = p1.difference(cutter)
    # Output must retain vector curves (not flattened to lines)
    has_curve = any(
        isinstance(cmd, (QuadraticTo, CubicTo))
        for sp in diff.subpaths
        for cmd in sp.commands
    )
    assert has_curve
    # Total commands must remain small
    total_cmds = sum(len(sp.commands) for sp in diff.subpaths)
    assert total_cmds < 20


def test_cubic_bezier_boolean():
    # Shape with cubic top edge
    p1 = Path(fill=FillStyle(color=Color.black()))
    p1.move_to(0, 0).cubic_to(Point(25, -60), Point(75, -60), Point(100, 0)).line_to(100, 100).line_to(0, 100).close()

    cutter = make_rect_path(50, -80, 70, 140)

    diff = p1.difference(cutter)
    has_curve = any(
        isinstance(cmd, (QuadraticTo, CubicTo))
        for sp in diff.subpaths
        for cmd in sp.commands
    )
    assert has_curve
    total_cmds = sum(len(sp.commands) for sp in diff.subpaths)
    assert total_cmds < 20


# -----------------------------------------------------------------------------
# 7. Compound Paths
# -----------------------------------------------------------------------------

def test_compound_paths():
    # Path with two separate subpath squares
    comp = Path(fill=FillStyle(color=Color.black()))
    comp.move_to(0, 0).line_to(40, 0).line_to(40, 40).line_to(0, 40).close()
    comp.move_to(60, 0).line_to(100, 0).line_to(100, 40).line_to(60, 40).close()

    # Cutter cutting horizontally through bottom half of both
    cutter = make_rect_path(-10, 20, 130, 30)

    diff = comp.difference(cutter)
    assert diff.contains_point(Point(20, 10))
    assert diff.contains_point(Point(80, 10))
    assert not diff.contains_point(Point(20, 30))
    assert not diff.contains_point(Point(80, 30))


# -----------------------------------------------------------------------------
# 8. Fill Rules & Winding Normalization
# -----------------------------------------------------------------------------

def test_fill_rule_even_odd_and_non_zero():
    # Even-odd donut: outer CW [0..100], inner CW [25..75] (concordant winding)
    donut_eo = Path(fill_rule=FillRule.EVEN_ODD, fill=FillStyle(color=Color.black()))
    donut_eo.move_to(0, 0).line_to(100, 0).line_to(100, 100).line_to(0, 100).close()
    donut_eo.move_to(25, 25).line_to(75, 25).line_to(75, 75).line_to(25, 75).close()

    cutter = make_rect_path(50, 0, 100, 100)
    diff = donut_eo.difference(cutter)

    # Result has normalized NON_ZERO winding
    assert diff.fill_rule == FillRule.NON_ZERO

    # Left half of donut ring remains; center hole remains empty
    assert diff.contains_point(Point(10, 50))
    assert not diff.contains_point(Point(50, 50))
    assert not diff.contains_point(Point(75, 50))


def test_even_odd_donut_union_empty_preserves_hole():
    """CRITICAL: combining an EVEN_ODD donut with an empty operand must preserve its hole

    under NON_ZERO result normalization.
    """
    donut_eo = Path(fill_rule=FillRule.EVEN_ODD, fill=FillStyle(color=Color.black()))
    donut_eo.move_to(0, 0).line_to(100, 0).line_to(100, 100).line_to(0, 100).close()
    donut_eo.move_to(25, 25).line_to(75, 25).line_to(75, 75).line_to(25, 75).close()

    empty = Path()
    res = donut_eo.union(empty)

    assert res.fill_rule == FillRule.NON_ZERO
    assert res.contains_point(Point(10, 10))
    assert not res.contains_point(Point(50, 50))


# -----------------------------------------------------------------------------
# 9. Transforms (World-Space Evaluation)
# -----------------------------------------------------------------------------

def test_operand_transforms():
    r1 = make_rect_path(0, 0, 100, 100)
    # Translate r1 by (100, 100)
    r1.transform.translation_x = 100
    r1.transform.translation_y = 100

    r2 = make_rect_path(150, 150, 100, 100)

    # In world coordinates, r1 is [100, 200] x [100, 200], overlapping r2 [150, 250] x [150, 250]
    inter = r1.intersection(r2)
    assert inter.contains_point(Point(175, 175))
    assert not inter.contains_point(Point(50, 50))

    # Result has identity transform and world-space geometry
    assert inter.transform.is_identity()
    gb = inter.get_geometry_bounds()
    assert gb.x == pytest.approx(150, abs=0.1)
    assert gb.y == pytest.approx(150, abs=0.1)
    assert gb.width == pytest.approx(50, abs=0.1)
    assert gb.height == pytest.approx(50, abs=0.1)


def test_rotation_and_scale_transforms():
    r1 = make_rect_path(-50, -50, 100, 100)
    r1.transform.rotation = 45  # rotated diamond centered at (0, 0)

    r2 = make_rect_path(0, -100, 100, 200)  # right half plane x >= 0

    inter = r1.intersection(r2)
    assert inter.contains_point(Point(20, 0))
    assert not inter.contains_point(Point(-20, 0))


def test_non_uniform_scale_geometry_transform():
    """Verify boolean geometry with explicit non-uniform scale (e.g. scale_x=2, scale_y=0.5)."""
    r1 = make_rect_path(0, 0, 100, 100)
    r1.transform.pivot = Point(0, 0)
    r1.transform.scale_x = 2.0
    r1.transform.scale_y = 0.5
    # r1 in world space is [0, 200] x [0, 50]

    # r2 is [50, 250] x [20, 80]
    r2 = make_rect_path(50, 20, 200, 60)

    # Intersection: [50, 200] x [20, 50]
    inter = r1.intersection(r2)
    gb = inter.get_geometry_bounds()
    assert gb.x == pytest.approx(50.0, abs=0.1)
    assert gb.y == pytest.approx(20.0, abs=0.1)
    assert gb.width == pytest.approx(150.0, abs=0.1)
    assert gb.height == pytest.approx(30.0, abs=0.1)
    assert inter.contains_point(Point(100, 35))
    assert not inter.contains_point(Point(25, 25))
    assert not inter.contains_point(Point(220, 35))

    # Difference: r1 minus r2
    diff = r1.difference(r2)
    assert diff.contains_point(Point(25, 25))
    assert diff.contains_point(Point(100, 10))
    assert not diff.contains_point(Point(100, 35))
    assert not diff.contains_point(Point(220, 25))


# -----------------------------------------------------------------------------
# 10. Parent / Group Transforms
# -----------------------------------------------------------------------------

def test_ancestor_group_transforms():
    r1 = make_rect_path(0, 0, 100, 100)
    grp = Group(children=[r1], transform=Transform(translation_x=200, translation_y=100))

    r2 = make_rect_path(250, 150, 100, 100)

    inter = r1.intersection(r2)
    assert inter.contains_point(Point(275, 175))
    assert inter._parent is None


# -----------------------------------------------------------------------------
# 11. Numerical Stability (Negative & Fractional Coordinates)
# -----------------------------------------------------------------------------

def test_negative_and_fractional_coordinates():
    r1 = make_rect_path(-43.75, -21.125, 60.5, 80.25)
    r2 = make_rect_path(-10.25, 10.5, 50.0, 50.0)

    inter = r1.intersection(r2)
    assert inter.contains_point(Point(0, 20))
    assert not inter.contains_point(Point(-30, -10))


# -----------------------------------------------------------------------------
# 12. All 8 Directional Empty Path Cases & Empty-vs-Empty
# -----------------------------------------------------------------------------

def test_all_empty_path_combinations():
    rect = make_rect_path(10, 10, 50, 50)
    rect.fill = FillStyle(color=Color.blue())
    rect.stroke = StrokeStyle(color=Color.green(), width=3)

    empty = Path()
    empty.fill = FillStyle(color=Color.red())

    # 1. empty UNION rect -> rect world geometry, empty's style
    e_u_r = empty.union(rect)
    assert e_u_r.contains_point(Point(30, 30))
    assert e_u_r.fill.color == Color.red()
    assert e_u_r is not rect and e_u_r is not empty

    # 2. rect UNION empty -> rect world geometry, rect's style
    r_u_e = rect.union(empty)
    assert r_u_e.contains_point(Point(30, 30))
    assert r_u_e.fill.color == Color.blue()

    # 3. empty INTERSECT rect -> empty
    e_i_r = empty.intersection(rect)
    assert len(e_i_r.subpaths) == 0

    # 4. rect INTERSECT empty -> empty
    r_i_e = rect.intersection(empty)
    assert len(r_i_e.subpaths) == 0

    # 5. empty DIFF rect -> empty
    e_d_r = empty.difference(rect)
    assert len(e_d_r.subpaths) == 0

    # 6. rect DIFF empty -> rect world geometry, rect's style
    r_d_e = rect.difference(empty)
    assert r_d_e.contains_point(Point(30, 30))
    assert r_d_e.fill.color == Color.blue()

    # 7. empty XOR rect -> rect world geometry, empty's style
    e_x_r = empty.xor(rect)
    assert e_x_r.contains_point(Point(30, 30))
    assert e_x_r.fill.color == Color.red()

    # 8. rect XOR empty -> rect world geometry, rect's style
    r_x_e = rect.xor(empty)
    assert r_x_e.contains_point(Point(30, 30))
    assert r_x_e.fill.color == Color.blue()

    # Empty vs empty operations
    e1 = Path()
    e2 = Path()
    for op in (e1.union, e1.intersection, e1.difference, e1.xor):
        res = op(e2)
        assert len(res.subpaths) == 0
        assert res is not e1 and res is not e2


# -----------------------------------------------------------------------------
# 13. Input Immutability
# -----------------------------------------------------------------------------

def test_input_immutability():
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(color=Color.red())
    r1.stroke = StrokeStyle(color=Color.black(), width=2)
    r1.transform.translation_x = 20

    r2 = make_rect_path(50, 50, 100, 100)
    r2.transform.rotation = 15

    r1_snap = copy.deepcopy(r1.to_dict())
    r2_snap = copy.deepcopy(r2.to_dict())

    _ = r1.difference(r2)

    assert r1.to_dict() == r1_snap
    assert r2.to_dict() == r2_snap


# -----------------------------------------------------------------------------
# 14. Result Independence
# -----------------------------------------------------------------------------

def test_result_independence():
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    res = r1.union(r2)
    res.subpaths.clear()
    res.transform.translation_x = 999
    res.fill = FillStyle(color=Color.magenta())

    assert len(r1.subpaths) == 1
    assert r1.transform.translation_x == 0


# -----------------------------------------------------------------------------
# 15. Painting, Styling & Exclusion Contracts
# -----------------------------------------------------------------------------

def test_boolean_with_fill_none():
    r1 = make_rect_path(0, 0, 100, 100, fill=False)
    assert r1.fill is None
    r2 = make_rect_path(50, 50, 100, 100, fill=False)
    assert r2.fill is None

    diff = r1.difference(r2)
    assert diff.fill is None
    assert len(diff.subpaths) == 1

    # Geometry is valid: assigning fill to the result allows testing interior
    diff_with_fill = copy.deepcopy(diff)
    diff_with_fill.fill = FillStyle(color=Color.black())
    assert diff_with_fill.contains_point(Point(25, 25))
    assert not diff_with_fill.contains_point(Point(75, 75))


def test_stroke_width_ignored_by_boolean():
    # Two rectangles with huge stroke widths touching at x=100
    r1 = make_rect_path(0, 0, 100, 100)
    r1.stroke = StrokeStyle(color=Color.black(), width=50)

    r2 = make_rect_path(100, 0, 100, 100)
    r2.stroke = StrokeStyle(color=Color.black(), width=50)

    # In boolean geometry, stroke width is ignored. Intersection should be empty!
    inter = r1.intersection(r2)
    assert len(inter.subpaths) == 0 or not inter.contains_point(Point(100, 50))


def test_style_linear_gradient_world_space():
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    grad = LinearGradient(Point(0, 0), Point(100, 0), stops, space="world")
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(paint=grad)

    r2 = make_rect_path(50, 0, 100, 100)
    diff = r1.difference(r2)

    assert diff.fill is not None
    assert isinstance(diff.fill.paint, LinearGradient)
    assert diff.fill.paint.space == "world"
    assert diff.fill.paint.start == Point(0, 0)
    assert diff.fill.paint.end == Point(100, 0)


@pytest.mark.parametrize('kind', ['linear', 'radial'])
@pytest.mark.parametrize('spread', ['repeat', 'reflect'])
@pytest.mark.parametrize('transformed', [False, True])
def test_disjoint_difference_preserves_spread_and_paint_transform(kind, spread, transformed):
    from drawcv.core.paint_sampling import sample_paint
    stops = (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))
    paint = (LinearGradient(Point(10, 10), Point(30, 10), stops, spread=spread)
             if kind == 'linear' else RadialGradient(Point(30, 30), 15, stops, spread=spread))
    if transformed:
        paint.transform = Transform(translation_x=20, translation_y=5, rotation=30,
                                    scale_x=1.2, scale_y=1.2, pivot=Point(0, 0))
    left = make_rect_path(10, 10, 80, 80)
    left.fill = FillStyle(paint=paint, opacity=0.7)
    left.stroke = None
    parent = Group(children=[left], transform=Transform(translation_x=5, translation_y=8))
    source_state = parent.to_dict()
    result = left.difference(make_rect_path(200, 200, 10, 10))
    assert result.fill.paint.spread == spread
    assert result.fill.opacity == 0.7
    np.testing.assert_allclose(
        # Avoid exact repeat seams: equivalent affine evaluations can land on
        # opposite sides of a discontinuity by machine epsilon.
        sample_paint(left.fill.paint, left.world_matrix, 100, 100, origin=(0.37, 0.19)),
        sample_paint(result.fill.paint, result.world_matrix, 100, 100, origin=(0.37, 0.19)), atol=1e-5,
    )
    assert parent.to_dict() == source_state
    assert result.fill.paint is not left.fill.paint
    result.fill.paint.spread = 'pad'
    assert left.fill.paint.spread == spread


def test_style_linear_gradient_object_space_exact_conversion():
    """Regression test: object-space horizontal linear gradient (0,0) -> (100,0)
    with unequal X/Y translations (tx=30, ty=70) becomes exactly (30,70) -> (130,70).
    """
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    grad = LinearGradient(Point(0, 0), Point(100, 0), stops, space="object")
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(paint=grad)
    r1.transform.translation_x = 30
    r1.transform.translation_y = 70

    r2 = make_rect_path(0, 0, 50, 50)
    diff = r1.difference(r2)

    assert diff.fill is not None
    assert isinstance(diff.fill.paint, LinearGradient)
    assert diff.fill.paint.space == "world"
    assert diff.fill.paint.start.x == pytest.approx(30.0, abs=1e-5)
    assert diff.fill.paint.start.y == pytest.approx(70.0, abs=1e-5)
    assert diff.fill.paint.end.x == pytest.approx(130.0, abs=1e-5)
    assert diff.fill.paint.end.y == pytest.approx(70.0, abs=1e-5)


def test_style_linear_gradient_general_affine_conversion():
    """Test linear gradient preservation under general affine transformations:
    - non-uniform scale
    - rotation + translation
    - shear via Transform.from_matrix
    Verify exact world-space endpoints and sampled scalar t values across spaces.
    """
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    r2 = make_rect_path(0, 0, 10, 10)

    # 1. Non-uniform scale with diagonal gradient (0,0) -> (100,100)
    r1_scale = make_rect_path(0, 0, 100, 100)
    r1_scale.fill = FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 100), stops, space="object"))
    r1_scale.transform.pivot = Point(0, 0)
    r1_scale.transform.scale_x = 2.0
    r1_scale.transform.scale_y = 0.5
    res_scale = r1_scale.union(r2)

    assert res_scale.fill is not None
    p0_scale = res_scale.fill.paint.start
    p1_scale = res_scale.fill.paint.end
    assert p0_scale.x == pytest.approx(0.0, abs=1e-4)
    assert p0_scale.y == pytest.approx(0.0, abs=1e-4)
    # A = diag(2, 0.5), A^(-T) = diag(0.5, 2)
    # v = (100, 100), ||v||^2 = 20000
    # g = (50, 200) / 20000 = (0.0025, 0.01)
    # ||g||^2 = 0.00010625
    # w = g / ||g||^2 = (400/17, 1600/17) ~= (23.5294, 94.1176)
    assert p1_scale.x == pytest.approx(400.0 / 17.0, abs=1e-4)
    assert p1_scale.y == pytest.approx(1600.0 / 17.0, abs=1e-4)

    # Verify sampled t values match across object space and baked world space
    for p_obj in [Point(0, 0), Point(50, 50), Point(100, 100), Point(20, 80)]:
        p_w = r1_scale.to_world(p_obj)
        t_obj = (p_obj.x * 100.0 + p_obj.y * 100.0) / (100.0**2 + 100.0**2)
        dx_w = p1_scale.x - p0_scale.x
        dy_w = p1_scale.y - p0_scale.y
        t_w = ((p_w.x - p0_scale.x) * dx_w + (p_w.y - p0_scale.y) * dy_w) / (dx_w**2 + dy_w**2)
        assert t_w == pytest.approx(t_obj, abs=1e-5)

    # 2. Rotation + translation
    r1_rot = make_rect_path(0, 0, 100, 100)
    r1_rot.fill = FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 0), stops, space="object"))
    r1_rot.transform.pivot = Point(0, 0)
    r1_rot.transform.rotation = 90.0
    r1_rot.transform.translation_x = 40.0
    r1_rot.transform.translation_y = 60.0
    res_rot = r1_rot.union(r2)

    assert res_rot.fill is not None
    p0_rot = res_rot.fill.paint.start
    p1_rot = res_rot.fill.paint.end
    assert p0_rot.x == pytest.approx(40.0, abs=1e-4)
    assert p0_rot.y == pytest.approx(60.0, abs=1e-4)
    assert p1_rot.x == pytest.approx(40.0, abs=1e-4)
    assert p1_rot.y == pytest.approx(160.0, abs=1e-4)

    for p_obj in [Point(0, 0), Point(100, 0), Point(50, 50), Point(10, 80)]:
        p_w = r1_rot.to_world(p_obj)
        t_obj = p_obj.x / 100.0
        dx_w = p1_rot.x - p0_rot.x
        dy_w = p1_rot.y - p0_rot.y
        t_w = ((p_w.x - p0_rot.x) * dx_w + (p_w.y - p0_rot.y) * dy_w) / (dx_w**2 + dy_w**2)
        assert t_w == pytest.approx(t_obj, abs=1e-5)

    # 3. Shear via Transform.from_matrix
    r1_shear = make_rect_path(0, 0, 100, 100)
    r1_shear.fill = FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 0), stops, space="object"))
    shear_mat = np.array([
        [1.0, 0.5, 15.0],
        [0.0, 1.0, 25.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    r1_shear.transform = Transform.from_matrix(shear_mat)
    res_shear = r1_shear.union(r2)

    assert res_shear.fill is not None
    p0_shear = res_shear.fill.paint.start
    p1_shear = res_shear.fill.paint.end
    assert p0_shear.x == pytest.approx(15.0, abs=1e-4)
    assert p0_shear.y == pytest.approx(25.0, abs=1e-4)
    assert p1_shear.x == pytest.approx(95.0, abs=1e-4)
    assert p1_shear.y == pytest.approx(-15.0, abs=1e-4)

    for p_obj in [Point(0, 0), Point(100, 0), Point(50, 50), Point(30, 70)]:
        p_w = r1_shear.to_world(p_obj)
        t_obj = p_obj.x / 100.0
        dx_w = p1_shear.x - p0_shear.x
        dy_w = p1_shear.y - p0_shear.y
        t_w = ((p_w.x - p0_shear.x) * dx_w + (p_w.y - p0_shear.y) * dy_w) / (dx_w**2 + dy_w**2)
        assert t_w == pytest.approx(t_obj, abs=1e-5)


def test_style_radial_gradient_similarity_conversion():
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    grad = RadialGradient(Point(50, 50), 40, stops, space="object")
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(paint=grad)
    r1.transform.scale_x = 2.0
    r1.transform.scale_y = 2.0  # Uniform scale: similarity transform

    r2 = make_rect_path(200, 200, 50, 50)
    diff = r1.difference(r2)

    assert diff.fill is not None
    assert isinstance(diff.fill.paint, RadialGradient)
    assert diff.fill.paint.space == "world"
    assert diff.fill.paint.radius == pytest.approx(80, abs=0.1)


def test_style_radial_gradient_reflected_similarity_conversion():
    """Verify that an object-space radial gradient under a reflected similarity transform
    (Euclidean similarity with negative determinant) is preserved as a world-space RadialGradient.
    """
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    grad = RadialGradient(Point(50, 50), 40, stops, space="object")
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(paint=grad)

    reflection = np.array([
        [-2.0, 0.0, 200.0],
        [ 0.0, 2.0,   0.0],
        [ 0.0, 0.0,   1.0],
    ], dtype=float)
    r1.transform = Transform.from_matrix(reflection)

    r2 = make_rect_path(0, 0, 10, 10)
    res = r1.union(r2)

    assert res.fill is not None
    assert isinstance(res.fill.paint, RadialGradient)
    assert res.fill.paint.space == "world"
    # center (50, 50) -> (-2*50 + 200, 2*50) = (100, 100)
    assert res.fill.paint.center.x == pytest.approx(100.0, abs=1e-4)
    assert res.fill.paint.center.y == pytest.approx(100.0, abs=1e-4)
    # radius 40 * 2 = 80
    assert res.fill.paint.radius == pytest.approx(80.0, abs=1e-4)


def test_style_radial_gradient_nonuniform_fallback():
    stops = (GradientStop(0, Color.black()), GradientStop(1, Color.white()))
    grad = RadialGradient(Point(50, 50), 40, stops, space="object")
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(paint=grad)
    r1.transform.scale_x = 2.0
    r1.transform.scale_y = 1.0  # Non-uniform scale: turns circle into ellipse!

    r2 = make_rect_path(200, 200, 50, 50)
    diff = r1.difference(r2)

    # Geometry is returned, but fill falls back to None under geometry-first policy
    assert diff.fill is None
    assert len(diff.subpaths) == 1


# -----------------------------------------------------------------------------
# 16. Detached Drawable State Contract
# -----------------------------------------------------------------------------

def test_detached_state_contract():
    r1 = make_rect_path(0, 0, 100, 100)
    r1.name = "LeftRect"
    r1.opacity = 0.5
    r1.visible = False
    r1.locked = True
    r1.z_index = 42

    r2 = make_rect_path(50, 50, 100, 100)

    res = r1.union(r2)

    assert res.transform.is_identity()
    assert res._parent is None
    assert res._layer is None
    assert res._scene is None
    assert res.clip is None
    assert res.mask is None
    assert res.effects == []
    assert res.opacity == 1.0
    assert res.visible is True
    assert res.locked is False
    assert res.z_index == 0


def test_rendering_state_exclusion_does_not_affect_geometry():
    """Verify that attaching clip, mask, effects, opacity, or visibility to operands
    does NOT alter the resulting boolean geometry.
    """
    from drawcv.effects import BlurEffect, ClipRect, Mask

    r1_clean = make_rect_path(0, 0, 100, 100)
    r2_clean = make_rect_path(50, 50, 100, 100)
    clean_diff = r1_clean.difference(r2_clean)

    # Decorated operands with clips, masks, effects, opacity, visibility
    r1_dec = make_rect_path(0, 0, 100, 100)
    r1_dec.clip = ClipRect(10, 10, 30, 30)
    r1_dec.mask = Mask(np.full((100, 100), 128, np.uint8))
    r1_dec.effects = [BlurEffect(kernel_size=15)]
    r1_dec.opacity = 0.4
    r1_dec.visible = False

    r2_dec = make_rect_path(50, 50, 100, 100)
    r2_dec.clip = ClipRect(0, 0, 20, 20)
    r2_dec.opacity = 0.2

    dec_diff = r1_dec.difference(r2_dec)

    # Resulting geometry must be strictly identical
    assert len(dec_diff.subpaths) == len(clean_diff.subpaths)
    for sp_dec, sp_clean in zip(dec_diff.subpaths, clean_diff.subpaths):
        assert len(sp_dec.commands) == len(sp_clean.commands)
        for cmd_dec, cmd_clean in zip(sp_dec.commands, sp_clean.commands):
            assert type(cmd_dec) is type(cmd_clean)
            if hasattr(cmd_dec, "point"):
                assert cmd_dec.point.x == pytest.approx(cmd_clean.point.x, abs=1e-5)
                assert cmd_dec.point.y == pytest.approx(cmd_clean.point.y, abs=1e-5)

    # Geometry hit-testing is identical
    for pt in [Point(25, 25), Point(75, 25), Point(25, 75), Point(75, 75), Point(125, 125)]:
        assert dec_diff.contains_point(pt) == clean_diff.contains_point(pt)

    # Detached output contracts hold
    assert dec_diff.clip is None
    assert dec_diff.mask is None
    assert dec_diff.effects == []
    assert dec_diff.opacity == 1.0
    assert dec_diff.visible is True


# -----------------------------------------------------------------------------
# 17. Resolution Independence
# -----------------------------------------------------------------------------

def test_resolution_independence():
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    u = r1.union(r2)

    # Boolean result commands are pure vector coordinates, not raster-grid dependent
    cmds_before = [(type(c), getattr(c, "point", None)) for c in u.subpaths[0].commands]

    # Render at two completely different canvas resolutions
    s1 = Scene(100, 100)
    s1.add(u)
    _ = OpenCVRenderer().render(s1)

    s2 = Scene(2000, 2000)
    s2.add(u)
    _ = OpenCVRenderer().render(s2)

    cmds_after = [(type(c), getattr(c, "point", None)) for c in u.subpaths[0].commands]
    assert cmds_before == cmds_after


# -----------------------------------------------------------------------------
# 18. Open Subpath Implicit Fill Closure
# -----------------------------------------------------------------------------

def test_open_subpath_implicit_closure():
    # Open triangle: (0,0) -> (100,0) -> (100,100), not closed
    open_tri = Path(fill=FillStyle(color=Color.black()))
    open_tri.move_to(0, 0).line_to(100, 0).line_to(100, 100)
    assert not open_tri.subpaths[0].closed

    cutter = make_rect_path(50, 0, 100, 100)

    diff = open_tri.difference(cutter)
    # The left half of the triangle (0,0) to (50,0) down to (50,50) should be retained
    assert diff.contains_point(Point(25, 10))

    # Source operand remains open and unmutated
    assert not open_tri.subpaths[0].closed
    assert len(open_tri.subpaths[0].commands) == 3


# -----------------------------------------------------------------------------
# 19. Serialization Round-Trip
# -----------------------------------------------------------------------------

def test_serialization_round_trip():
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    diff = r1.difference(r2)
    json_str = to_json(diff.to_dict())
    restored = Path.from_dict(json.loads(json_str))

    assert restored.contains_point(Point(25, 25))
    assert not restored.contains_point(Point(75, 75))
    assert restored.fill_rule == FillRule.NON_ZERO


# -----------------------------------------------------------------------------
# 20. SVG Strict Export
# -----------------------------------------------------------------------------

def test_svg_strict_export():
    r1 = make_rect_path(0, 0, 100, 100)
    r1.fill = FillStyle(color=Color.blue())
    r2 = make_rect_path(50, 50, 100, 100)

    diff = r1.difference(r2)
    scene = Scene(200, 200)
    scene.add(diff)

    exporter = SVGExporter(strict=True)
    res = exporter.render(scene)
    assert "<path" in res.svg
    assert not res.fallbacks  # Native vector path, not raster fallback!


# -----------------------------------------------------------------------------
# 21. Clone and History Integration
# -----------------------------------------------------------------------------

def test_clone_and_history():
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    diff = r1.difference(r2)
    cloned = diff.clone()

    assert cloned is not diff
    assert cloned.contains_point(Point(25, 25))

    # Test Scene undo / redo with boolean result
    scene = Scene(200, 200)
    scene.add(diff)
    assert scene.get(diff.id) is diff

    assert scene.undo()
    assert diff.id not in scene

    assert scene.redo()
    assert scene.get(diff.id) is diff


# -----------------------------------------------------------------------------
# 22. Error Handling & Validation
# -----------------------------------------------------------------------------

def test_error_handling_invalid_operands():
    r = make_rect_path(0, 0, 100, 100)
    with pytest.raises(ValidationError, match="Expected Path instance"):
        r.union("invalid")

    with pytest.raises(ValidationError, match="Invalid boolean operation"):
        r.boolean(r, "magic_op")


def test_error_handling_non_finite_coordinates():
    with pytest.raises(ValidationError, match="finite numbers"):
        Point(float("nan"), 0.0)

    with pytest.raises(ValidationError, match="finite numbers"):
        Point(0.0, float("inf"))

    r = make_rect_path(0, 0, 100, 100)
    p = Path()
    p.subpaths = [Subpath(commands=[MoveTo(Point(0.0, 0.0))])]
    # Inject non-finite transform parameter
    p.transform.translation_x = 0.0
    object.__setattr__(p.transform, "translation_x", float("nan"))
    with pytest.raises(ValidationError, match="finite"):
        r.union(p)


def test_error_handling_backend_failure(monkeypatch):
    """Verify that backend failures during pathops.op execution are wrapped in PathBooleanError with cause chained."""
    def mock_op(*args, **kwargs):
        raise RuntimeError("Simulated Skia native backend failure")

    monkeypatch.setattr(pathops, "op", mock_op)
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    with pytest.raises(PathBooleanError, match="Path boolean operation 'union' failed: Simulated Skia native backend failure") as exc_info:
        r1.union(r2)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Simulated Skia native backend failure"


def test_error_handling_conversion_backend_failure(monkeypatch):
    """Verify that native backend failures during drawcv_to_pathops conversion are wrapped in PathBooleanError with cause chained."""
    def mock_moveTo(*args, **kwargs):
        raise RuntimeError("Simulated PathOps moveTo failure")

    monkeypatch.setattr(pathops.Path, "moveTo", mock_moveTo)
    r1 = make_rect_path(0, 0, 100, 100)
    r2 = make_rect_path(50, 50, 100, 100)

    with pytest.raises(PathBooleanError, match="Path boolean operation 'union' failed: Simulated PathOps moveTo failure") as exc_info:
        r1.union(r2)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "Simulated PathOps moveTo failure"


def test_error_handling_backend_non_finite_output_surfaces_path_boolean_error(monkeypatch):
    """Verify that non-finite coordinates in backend output surface as PathBooleanError,

    while invalid DrawCV operand inputs raise ValidationError.
    """
    # 1. Invalid DrawCV input raises ValidationError
    r1 = make_rect_path(0, 0, 100, 100)
    with pytest.raises(ValidationError):
        r1.union("not_a_path")

    # 2. Malformed/non-finite backend output raises PathBooleanError
    bad_output = pathops.Path()
    bad_output.moveTo(0.0, 0.0)
    bad_output.lineTo(float("nan"), 10.0)
    bad_output.close()

    monkeypatch.setattr(pathops, "op", lambda *args, **kwargs: bad_output)

    r2 = make_rect_path(50, 50, 100, 100)
    with pytest.raises(PathBooleanError, match="non-finite coordinates"):
        r1.union(r2)
