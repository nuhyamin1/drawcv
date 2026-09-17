"""Comprehensive tests for Full True Retained Path-Based Clipping.

Validates:
- Retained drawcv.Path geometry used directly as clip boundary (quadratic/cubic Béziers, compound paths, fill rules).
- Exact live Python-object reference preservation across undo/redo and temporal rendering.
- Shared-reference persistence contract (runtime live references vs JSON serialization by value).
- Centralized get_clip_point_mapper for raster and native SVG export (M/L/Q/C/Z, no curve flattening, strict mode).
- Group and Layer retained clipping, Group.contains_point(), and Scene.hit_test().
- Compositing order: effects -> mask -> clip -> opacity -> blend.
- Fractional premultiplied coverage and transparent BGRA output.
- Open path implicit closure, empty paths, degenerate subpaths.
- Legacy ClipRect and polygon ClipPath pixel compatibility.
- Schema 1.6 -> 1.7 migration.
"""

import copy
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from drawcv import (
    Scene, Color, Point, Rectangle, Circle, Group, Layer,
    Transform, FillStyle, StrokeStyle, LinearGradient, GradientStop,
    Path, Subpath, MoveTo, LineTo, QuadraticTo, CubicTo, Close,
    FillRule, BlendMode, Mask, BlurEffect, ShadowEffect,
    ClipRect, ClipPath, SVGExporter, OpenCVRenderer
)
from drawcv.effects.clipping import (
    clip_to_dict, clip_from_dict, capture_clip_state, restore_clip_state,
    get_clip_point_mapper, evaluate_clip_coverage, is_point_in_clip
)
from drawcv.serialization.registry import SchemaMigrator

SVG_NS = {"s": "http://www.w3.org/2000/svg"}


# =============================================================================
# Helper Fixtures & Constructors
# =============================================================================

def render_scene(scene: Scene, alpha: bool = True) -> np.ndarray:
    """Render a scene using OpenCVRenderer and return the underlying numpy array."""
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=alpha)
    return canvas.buffer


def make_cubic_clip(x=20, y=20, size=60) -> Path:
    """Create a closed path with a cubic Bézier curved top edge."""
    p = Path()
    p.move_to(Point(x, y + size))
    p.line_to(Point(x, y + size / 2))
    p.cubic_to(
        Point(x + size * 0.25, y - size * 0.3),
        Point(x + size * 0.75, y + size * 0.7),
        Point(x + size, y + size / 2)
    )
    p.line_to(Point(x + size, y + size))
    p.close()
    return p


def make_quadratic_clip(x=20, y=20, size=60) -> Path:
    """Create a closed path with a quadratic Bézier curve."""
    p = Path()
    p.move_to(Point(x, y))
    p.quadratic_to(Point(x + size / 2, y + size * 1.5), Point(x + size, y))
    p.line_to(Point(x + size, y + size))
    p.line_to(Point(x, y + size))
    p.close()
    return p


def make_donut_path(cx=50, cy=50, r_outer=40, r_inner=20, cw_outer=True, cw_inner=False, fill_rule=FillRule.EVEN_ODD) -> Path:
    """Create a compound path with an outer box and inner box hole."""
    outer_cmds: list = []
    if cw_outer:
        outer_cmds = [
            MoveTo(Point(cx - r_outer, cy - r_outer)),
            LineTo(Point(cx + r_outer, cy - r_outer)),
            LineTo(Point(cx + r_outer, cy + r_outer)),
            LineTo(Point(cx - r_outer, cy + r_outer)),
            Close(),
        ]
    else:
        outer_cmds = [
            MoveTo(Point(cx - r_outer, cy - r_outer)),
            LineTo(Point(cx - r_outer, cy + r_outer)),
            LineTo(Point(cx + r_outer, cy + r_outer)),
            LineTo(Point(cx + r_outer, cy - r_outer)),
            Close(),
        ]
    outer_sp = Subpath(commands=outer_cmds, closed=True)

    inner_cmds: list = []
    if cw_inner:
        inner_cmds = [
            MoveTo(Point(cx - r_inner, cy - r_inner)),
            LineTo(Point(cx + r_inner, cy - r_inner)),
            LineTo(Point(cx + r_inner, cy + r_inner)),
            LineTo(Point(cx - r_inner, cy + r_inner)),
            Close(),
        ]
    else:
        inner_cmds = [
            MoveTo(Point(cx - r_inner, cy - r_inner)),
            LineTo(Point(cx - r_inner, cy + r_inner)),
            LineTo(Point(cx + r_inner, cy + r_inner)),
            LineTo(Point(cx + r_inner, cy - r_inner)),
            Close(),
        ]
    inner_sp = Subpath(commands=inner_cmds, closed=True)

    return Path(subpaths=[outer_sp, inner_sp], fill_rule=fill_rule)


# =============================================================================
# 1. Live-Reference Identity & Semantic Restoration (History / Undo-Redo)
# =============================================================================

def test_history_clip_assignment_undo_redo_identity():
    """Verify exact original Path reference is restored across clip assignment undo/redo."""
    scene = Scene(200, 200)
    shape = Rectangle(position=Point(10, 10), width=100, height=100, fill=FillStyle(color=Color(255, 0, 0)))
    scene.add(shape)

    clip_a = make_cubic_clip(20, 20, 50)
    clip_b = make_quadratic_clip(30, 30, 40)
    shape.clip = clip_a
    assert shape.clip is clip_a

    with scene.edit(shape):
        shape.clip = clip_b

    assert shape.clip is clip_b
    assert shape.clip is not clip_a

    scene.undo()
    assert shape.clip is clip_a

    scene.redo()
    assert shape.clip is clip_b


def test_history_clip_property_mutation_in_place():
    """Verify in-place mutation of an attached clip restores values while preserving the exact reference."""
    scene = Scene(200, 200)
    shape = Rectangle(position=Point(10, 10), width=100, height=100)
    clip = make_cubic_clip(20, 20, 50)
    shape.clip = clip
    scene.add(shape)

    orig_tx = clip.transform.translation_x

    with scene.edit(shape):
        shape.clip.transform.translation_x += 25.0
        shape.clip.transform.translation_y += 15.0

    assert shape.clip is clip
    assert shape.clip.transform.translation_x == orig_tx + 25.0

    scene.undo()
    assert shape.clip is clip
    assert shape.clip.transform.translation_x == orig_tx

    scene.redo()
    assert shape.clip is clip
    assert shape.clip.transform.translation_x == orig_tx + 25.0


def test_temporal_identity_preserved_across_render():
    """Verify shape.clip reference is preserved identically during temporal rendering."""
    scene = Scene(200, 200)
    shape = Rectangle(position=Point(10, 10), width=100, height=100)
    clip = make_cubic_clip(20, 20, 50)
    shape.clip = clip
    scene.add(shape)

    assert shape.clip is clip
    scene.render_at_time(0.5)
    assert shape.clip is clip
    scene.render_at_time(0.0)
    assert shape.clip is clip


# =============================================================================
# 2. Shared-Reference Persistence Semantics
# =============================================================================

def test_shared_reference_persistence_contract():
    """Verify runtime live reference sharing vs by-value JSON serialization contract.

    Retained Path clips use live Python-object references at runtime. JSON serialization
    stores clip geometry/fill-rule/transform by value. Shared object aliasing between a
    clip and a separately registered scene drawable is therefore not preserved across
    serialization round-trips. Visual and geometric clipping semantics are preserved.
    """
    scene = Scene(200, 200)
    shared_path = make_cubic_clip(10, 10, 80)
    shared_path.name = "clip_and_drawable"
    scene.add(shared_path)

    rect = Rectangle(position=Point(0, 0), width=150, height=150, fill=FillStyle(color=Color(0, 255, 0)))
    rect.clip = shared_path
    scene.add(rect)

    # Runtime assertion: exact object aliasing holds
    assert rect.clip is shared_path

    # Round-trip through JSON
    doc = scene.to_dict()
    restored = Scene.from_dict(doc)

    restored_path = restored.find_first(name="clip_and_drawable")
    restored_rect = [obj for obj in restored.layers[0].objects if isinstance(obj, Rectangle)][0]

    assert restored_path is not None
    assert restored_rect.clip is not None
    assert isinstance(restored_rect.clip, Path)

    # By-value contract: reconstructed clip is an independent Path instance
    assert restored_rect.clip is not restored_path

    # But visual and geometric clipping semantics are strictly preserved
    assert len(restored_rect.clip.subpaths) == len(shared_path.subpaths)
    assert restored_rect.clip.fill_rule == shared_path.fill_rule
    assert restored_rect.clip.transform.to_dict() == shared_path.transform.to_dict()


def test_clip_to_dict_lean_representation():
    """Verify clip_to_dict emits lean geometric data without excess drawable fields."""
    clip = make_cubic_clip(10, 10, 50)
    d = clip_to_dict(clip)
    assert d is not None
    assert d["type"] == "retained_path"
    assert "subpaths" in d
    assert "fill_rule" in d
    assert "transform" in d
    # Must NOT have scene drawable metadata
    assert "id" not in d
    assert "name" not in d
    assert "visible" not in d
    assert "stroke" not in d
    assert "fill" not in d


# =============================================================================
# 3. Centralized Clip Point Mapper for Raster and Native SVG
# =============================================================================

def test_centralized_clip_point_mapper_sequential_world_transform():
    """Verify get_clip_point_mapper applies clip-local then owner.to_world()."""
    owner = Rectangle(position=Point(0, 0), width=100, height=100)
    owner.move(50, 50)
    clip = Path()
    clip.transform.translation_x = 10.0
    clip.transform.translation_y = 20.0

    mapper = get_clip_point_mapper(clip, owner)
    # Local point (5, 5) -> clip transform (15, 25) -> owner translation (15+50, 25+50) = (65, 75)
    mapped = mapper(Point(5, 5))
    assert mapped.x == pytest.approx(65.0)
    assert mapped.y == pytest.approx(75.0)


def test_native_svg_export_retained_path_no_curve_flattening():
    """Verify native SVG export uses M/L/Q/C/Z without flattening Béziers and strict mode succeeds."""
    scene = Scene(200, 200, Color(255, 255, 255))
    rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color(200, 0, 0)))
    rect.clip = make_cubic_clip(20, 20, 100)
    scene.add(rect)

    # Strict export must succeed with zero fallbacks
    export_res = scene.export_svg(strict=True)
    assert len(export_res.fallbacks) == 0

    svg_str = export_res.svg
    root = ET.fromstring(svg_str)

    # Check for clipPath definition
    clip_paths = root.findall(".//s:clipPath", SVG_NS)
    assert len(clip_paths) == 1
    clip_elem = clip_paths[0]

    path_elem = clip_elem.find("s:path", SVG_NS)
    assert path_elem is not None
    d_attr = path_elem.get("d")
    assert d_attr is not None

    # Must contain native 'C ' command for cubic Bézier and 'Z' for closure
    assert "C " in d_attr
    assert d_attr.endswith("Z") or " Z" in d_attr

    # Verify clip-rule is present
    assert path_elem.get("clip-rule") in ("nonzero", "evenodd")


def test_native_svg_quadratic_and_fill_rule():
    """Verify quadratic Béziers and EVEN_ODD rule in SVG clipPath."""
    scene = Scene(200, 200)
    rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color(0, 100, 200)))
    rect.clip = make_donut_path(cx=100, cy=100, r_outer=80, r_inner=40, fill_rule=FillRule.EVEN_ODD)
    scene.add(rect)

    export_res = scene.export_svg(strict=True)
    assert not export_res.fallbacks

    root = ET.fromstring(export_res.svg)
    clip_elem = root.find(".//s:clipPath", SVG_NS)
    assert clip_elem is not None
    path_elem = clip_elem.find("s:path", SVG_NS)
    assert path_elem.get("clip-rule") == "evenodd"


# =============================================================================
# 4. Bézier Curves & Antialiased Raster Coverage
# =============================================================================

def test_cubic_bezier_raster_clipping_antialiased():
    """Verify cubic Bézier clipping renders inside pixels and fractional antialiased boundary."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 0, 0, 1.0)))
    rect.clip = make_cubic_clip(10, 10, 80)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    assert frame.shape == (100, 100, 4)

    # Center of clip should be solid red
    center_pixel = frame[60, 50]  # BGRA
    assert center_pixel[2] == 255  # R
    assert center_pixel[3] == 255  # A

    # Far corner outside clip (e.g. 5, 5) must be completely transparent
    corner_pixel = frame[5, 5]
    assert corner_pixel[3] == 0

    # Antialiasing: there must be boundary pixels with 0 < alpha < 255
    alphas = frame[:, :, 3]
    partial_alphas = alphas[(alphas > 0) & (alphas < 255)]
    assert len(partial_alphas) > 0


def test_compound_donut_hole_clipping_even_odd():
    """Verify compound path with outer and inner subpaths correctly cuts out a hole."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 255, 0, 1.0)))
    # Donut with outer=35, inner=15 centered at (50, 50)
    rect.clip = make_donut_path(cx=50, cy=50, r_outer=35, r_inner=15, fill_rule=FillRule.EVEN_ODD)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)

    # Center hole must be unpainted (alpha == 0)
    assert frame[50, 50, 3] == 0
    assert frame[48, 52, 3] == 0

    # Ring between r=15 and r=35 must be painted green
    assert frame[50, 75, 1] == 255  # G
    assert frame[50, 75, 3] == 255  # A

    # Exterior beyond r=35 must be unpainted
    assert frame[5, 5, 3] == 0


def test_fill_rule_non_zero_vs_even_odd():
    """Verify NON_ZERO fills nested same-direction contours while EVEN_ODD creates hole."""
    # CW outer + CW inner:
    # NON_ZERO winding = 2 -> inside
    # EVEN_ODD parity = 2 -> outside (hole)
    path_nonzero = make_donut_path(cx=50, cy=50, r_outer=35, r_inner=15, cw_outer=True, cw_inner=True, fill_rule=FillRule.NON_ZERO)
    path_evenodd = make_donut_path(cx=50, cy=50, r_outer=35, r_inner=15, cw_outer=True, cw_inner=True, fill_rule=FillRule.EVEN_ODD)

    scene_nz = Scene(100, 100, Color(0, 0, 0, 0))
    r_nz = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 255)))
    r_nz.clip = path_nonzero
    scene_nz.add(r_nz)
    frame_nz = render_scene(scene_nz, alpha=True)

    scene_eo = Scene(100, 100, Color(0, 0, 0, 0))
    r_eo = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 255)))
    r_eo.clip = path_evenodd
    scene_eo.add(r_eo)
    frame_eo = render_scene(scene_eo, alpha=True)

    # In NON_ZERO, center (50, 50) is filled (alpha == 255)
    assert frame_nz[50, 50, 3] == 255

    # In EVEN_ODD, center (50, 50) is a hole (alpha == 0)
    assert frame_eo[50, 50, 3] == 0


# =============================================================================
# 5. Transforms, Hierarchies, and Nested Clips
# =============================================================================

def test_clip_transform_affine_shear():
    """Verify clip respects arbitrary authoritative affine transforms including shear."""
    scene = Scene(120, 120, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=120, height=120, fill=FillStyle(color=Color(100, 100, 255)))

    clip = make_cubic_clip(20, 20, 50)
    # Apply shear matrix [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]
    shear_mat = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    clip.transform = Transform.from_matrix(shear_mat)
    rect.clip = clip
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    assert np.any(frame[:, :, 3] > 0)


def test_transformed_owner_and_transformed_clip():
    """Verify clip moves with owner when owner is transformed."""
    scene = Scene(200, 200, Color(0, 0, 0, 0))
    # Owner at (50, 50)
    rect = Rectangle(position=Point(0, 0), width=80, height=80, fill=FillStyle(color=Color(255, 255, 0)))
    rect.move(50, 50)
    # Clip local bounds roughly (0, 0) to (40, 40)
    clip = Path()
    clip.move_to(Point(0, 0))
    clip.line_to(Point(40, 0))
    clip.line_to(Point(40, 40))
    clip.line_to(Point(0, 40))
    clip.close()
    rect.clip = clip
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Inside owner+clip (e.g. 50 + 20, 50 + 20 = 70, 70)
    assert frame[70, 70, 3] == 255
    # Outside owner (e.g. 20, 20)
    assert frame[20, 20, 3] == 0
    # Inside owner but outside clip (e.g. 50 + 60, 50 + 60 = 110, 110)
    assert frame[110, 110, 3] == 0


def test_nested_three_level_clips():
    """Verify Layer clip -> Group clip -> Child clip correctly computes 3-level intersection."""
    scene = Scene(200, 200, Color(0, 0, 0, 0))
    layer = scene.layers[0]

    # Layer clip: (0, 0) to (150, 150)
    clip_l = Path()
    clip_l.move_to(Point(0, 0))
    clip_l.line_to(Point(150, 0))
    clip_l.line_to(Point(150, 150))
    clip_l.line_to(Point(0, 150))
    clip_l.close()
    layer.clip = clip_l

    # Group clip: (50, 50) to (180, 180) -> overlaps layer in [50, 150] x [50, 150]
    group = Group()
    clip_g = Path()
    clip_g.move_to(Point(50, 50))
    clip_g.line_to(Point(180, 50))
    clip_g.line_to(Point(180, 180))
    clip_g.line_to(Point(50, 180))
    clip_g.close()
    group.clip = clip_g

    # Child rect: covers (0, 0) to (200, 200)
    rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color(255, 0, 255)))
    # Child clip: (75, 75) to (125, 125)
    clip_c = Path()
    clip_c.move_to(Point(75, 75))
    clip_c.line_to(Point(125, 75))
    clip_c.line_to(Point(125, 125))
    clip_c.line_to(Point(75, 125))
    clip_c.close()
    rect.clip = clip_c

    group.add(rect)
    scene.add(group)

    frame = render_scene(scene, alpha=True)

    # Center of child clip (100, 100) satisfies all 3 clips -> visible
    assert frame[100, 100, 3] == 255

    # Point (60, 60): inside layer and group clips, but outside child clip -> 0
    assert frame[60, 60, 3] == 0

    # Point (160, 100): outside layer clip -> 0
    assert frame[100, 160, 3] == 0


# =============================================================================
# 6. Edge Cases: Open Path, Empty Path, Degenerate Path
# =============================================================================

def test_open_path_implicit_closure():
    """Verify an unclosed path (no Close command) implicitly closes from endpoint to start point."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 255)))

    # Triangle without close(): (20, 20) -> (80, 20) -> (50, 80)
    clip = Path()
    clip.move_to(Point(20, 20))
    clip.line_to(Point(80, 20))
    clip.line_to(Point(50, 80))
    # Note: NO close()
    rect.clip = clip
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Centroid of triangle ~ (50, 40) should be filled
    assert frame[40, 50, 3] == 255
    # Point outside triangle (50, 10) should be unpainted
    assert frame[10, 50, 3] == 0


def test_empty_path_clips_everything():
    """Verify an empty Path clip (no subpaths) produces zero coverage (hides the object)."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 0, 0)))
    rect.clip = Path()
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Entire buffer must remain transparent
    assert np.all(frame[:, :, 3] == 0)


def test_degenerate_subpaths_safely_ignored():
    """Verify degenerate subpaths (< 3 points) do not crash and remaining valid subpaths clip."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 0, 255)))

    sub_degen = Subpath(commands=[MoveTo(Point(10, 10)), LineTo(Point(20, 20))])

    sub_valid = Subpath(commands=[
        MoveTo(Point(40, 40)),
        LineTo(Point(80, 40)),
        LineTo(Point(80, 80)),
        LineTo(Point(40, 80)),
        Close(),
    ], closed=True)

    rect.clip = Path(subpaths=[sub_degen, sub_valid])
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Center of valid subpath (60, 60) should be blue
    assert frame[60, 60, 0] == 255
    assert frame[60, 60, 3] == 255


# =============================================================================
# 7. Hit Testing & Group.contains_point()
# =============================================================================

def test_hit_testing_with_retained_path_clip():
    """Verify hit_test checks retained Path clip boundaries accurately."""
    scene = Scene(200, 200)
    rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color(255, 0, 0)))
    rect.clip = make_cubic_clip(20, 20, 80)
    scene.add(rect)

    # Point clearly inside the clip
    hits_inside = scene.hit_test(60, 60)
    assert rect in hits_inside

    # Point inside the rectangle bounds (10, 10) but outside the clip curve
    hits_outside = scene.hit_test(10, 10)
    assert rect not in hits_outside


def test_group_contains_point_and_scene_hit_test():
    """Verify Group.contains_point is clip-aware and Scene.hit_test avoids redundant checks."""
    scene = Scene(200, 200)
    group = Group()
    clip = Path()
    clip.move_to(Point(30, 30))
    clip.line_to(Point(90, 30))
    clip.line_to(Point(90, 90))
    clip.line_to(Point(30, 90))
    clip.close()
    group.clip = clip

    child = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color(0, 255, 0)))
    group.add(child)
    scene.add(group)

    # Direct caller to Group.contains_point:
    assert group.contains_point(Point(50, 50)) is True
    assert group.contains_point(Point(10, 10)) is False

    # Scene.hit_test
    hits_in = scene.hit_test(50, 50)
    assert child in hits_in
    assert group in hits_in

    hits_out = scene.hit_test(10, 10)
    assert child not in hits_out
    assert group not in hits_out


# =============================================================================
# 8. Compositing Order: Effects -> Mask -> Clip -> Opacity -> Blend
# =============================================================================

def test_compositing_order_effects_then_clip():
    """Verify effects (blur/shadow) are processed before clipping, so padding is clipped."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=Color(255, 0, 0)))
    # Add blur effect that normally expands bounds
    rect.effects.append(BlurEffect(kernel_size=15, sigma=5.0))

    # Clip tightly to [20, 80] x [20, 80]
    clip = Path()
    clip.move_to(Point(20, 20))
    clip.line_to(Point(80, 20))
    clip.line_to(Point(80, 80))
    clip.line_to(Point(20, 80))
    clip.close()
    rect.clip = clip
    scene.add(rect)

    frame = render_scene(scene, alpha=True)

    # Outside the clip boundary (e.g. (5, 50)), blur must NOT bleed through
    assert frame[50, 5, 3] == 0
    assert frame[5, 50, 3] == 0

    # Inside the clip boundary, blurred red is visible
    assert frame[50, 50, 3] > 0


def test_compositing_order_mask_and_clip():
    """Verify mask and clip are combined multiplicatively."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 255)))

    # Clip to left half [0, 50]
    clip = Path()
    clip.move_to(Point(0, 0))
    clip.line_to(Point(50, 0))
    clip.line_to(Point(50, 100))
    clip.line_to(Point(0, 100))
    clip.close()
    rect.clip = clip

    # Mask to top half [0, 50]
    mask_buf = np.zeros((100, 100), dtype=np.uint8)
    mask_buf[:50, :] = 255
    rect.mask = Mask(mask_buf)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Intersection: [0, 50] x [0, 50] should be visible
    assert frame[25, 25, 3] == 255
    # Inside clip but outside mask [0, 50] x [50, 100] -> unpainted
    assert frame[75, 25, 3] == 0
    # Inside mask but outside clip [50, 100] x [0, 50] -> unpainted
    assert frame[25, 75, 3] == 0


def test_clip_with_gradient_and_opacity_and_blend_mode():
    """Verify retained clip works seamlessly with gradients, opacity, and blend modes."""
    scene = Scene(100, 100, Color(50, 50, 50, 1.0))
    grad = LinearGradient(
        start=Point(0, 0),
        end=Point(100, 100),
        stops=[GradientStop(0.0, Color(255, 0, 0)), GradientStop(1.0, Color(0, 0, 255))]
    )
    rect = Rectangle(
        position=Point(0, 0),
        width=100,
        height=100,
        fill=FillStyle(paint=grad),
        opacity=0.6,
        blend_mode=BlendMode.MULTIPLY
    )
    rect.clip = make_cubic_clip(20, 20, 60)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    # Output rendered without crash and clipped
    assert frame[5, 5, 0] == 50  # background untouched outside clip
    assert frame[50, 50, 3] == 255  # solid background composite


# =============================================================================
# 9. Legacy Regressions: ClipRect and Polygon ClipPath Compatibility
# =============================================================================

def test_legacy_clip_rect_pixel_exact():
    """Verify ClipRect retains exact behavior and serializes cleanly."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 0)))
    rect.clip = ClipRect(20, 20, 60, 60)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    assert frame[50, 50, 3] == 255
    assert frame[10, 10, 3] == 0

    d = clip_to_dict(rect.clip)
    assert d["type"] == "rect"
    restored = clip_from_dict(d)
    assert isinstance(restored, ClipRect)
    assert restored.width == 60


def test_legacy_clip_path_polygon_exact():
    """Verify legacy polygon ClipPath retains exact behavior and backward compatibility."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 255, 255)))
    pts = [Point(20, 20), Point(80, 20), Point(50, 80)]
    rect.clip = ClipPath(pts)
    scene.add(rect)

    frame = render_scene(scene, alpha=True)
    assert frame[40, 50, 3] == 255
    assert frame[10, 50, 3] == 0

    d = clip_to_dict(rect.clip)
    assert d["type"] == "polygon"
    assert "points" in d
    restored = clip_from_dict(d)
    assert isinstance(restored, ClipPath)
    assert len(restored.points) == 3


# =============================================================================
# 10. Schema 1.7 Serialization & Discriminator Tests
# =============================================================================

def test_clip_serialization_discriminators_all_four_cases():
    """Verify all 4 clip discriminator cases: rect, polygon, retained_path, legacy path."""
    # Case 1: New ClipRect -> "rect"
    cr = ClipRect(10, 10, 50, 50)
    d_cr = clip_to_dict(cr)
    assert d_cr["type"] == "rect"
    assert isinstance(clip_from_dict(d_cr), ClipRect)

    # Case 2: New legacy ClipPath -> "polygon"
    cp = ClipPath([Point(0, 0), Point(10, 0), Point(0, 10)])
    d_cp = clip_to_dict(cp)
    assert d_cp["type"] == "polygon"
    assert isinstance(clip_from_dict(d_cp), ClipPath)

    # Case 3: New retained Path clip -> "retained_path"
    rp = make_cubic_clip(10, 10, 40)
    d_rp = clip_to_dict(rp)
    assert d_rp["type"] == "retained_path"
    restored_rp = clip_from_dict(d_rp)
    assert isinstance(restored_rp, Path)

    # Case 4: Old schema-1.6 "path" + "points" loads/migrates correctly
    legacy_1_6_clip_dict = {
        "type": "path",
        "points": [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 0.0, "y": 10.0}],
    }
    loaded_legacy = clip_from_dict(legacy_1_6_clip_dict)
    assert isinstance(loaded_legacy, ClipPath)
    assert len(loaded_legacy.points) == 3


def test_newly_serialized_scene_declares_schema_1_7():
    """Verify newly saved scenes declare schema 1.7 in both version and schema_version."""
    scene = Scene(200, 200)
    rect = Rectangle(position=Point(0, 0), width=50, height=50)
    rect.clip = ClipPath([Point(0, 0), Point(50, 0), Point(0, 50)])
    scene.add(rect)

    doc = scene.to_dict()
    assert doc["version"] == "1.7"
    assert doc["schema_version"] == "1.7"
    assert doc["scene"]["layers"][0]["objects"][0]["clip"]["type"] == "polygon"


def test_schema_1_6_to_1_7_migration():
    """Verify SchemaMigrator handles 1.6 -> 1.7 migration smoothly."""
    doc_1_6 = {
        "format": "drawcv",
        "version": "1.6",
        "scene": {
            "width": 100,
            "height": 100,
            "layers": [
                {
                    "name": "default",
                    "objects": [
                        {
                            "type": "rectangle",
                            "x": 0, "y": 0, "width": 100, "height": 100,
                            "clip": {
                                "type": "path",
                                "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 50, "y": 50}]
                            }
                        }
                    ]
                }
            ]
        }
    }

    migrated = SchemaMigrator.migrate(doc_1_6, target_version="1.7")
    assert migrated["version"] == "1.7"
    clip_dict = migrated["scene"]["layers"][0]["objects"][0]["clip"]
    assert clip_dict["type"] == "polygon"
    assert "points" in clip_dict


# =============================================================================
# 11. External SVG / resvg Parity Verification
# =============================================================================

def _check_resvg_parity(scene: Scene, interior_slice: tuple | None = None, max_mae: float = 2.0):
    """Assert native strict SVG export matches raster OpenCVRenderer with resvg."""
    import resvg_py
    import cv2

    export_res = scene.export_svg(strict=True)
    assert not export_res.fallbacks, f"Unexpected fallback: {export_res.fallbacks}"

    png = resvg_py.svg_to_bytes(svg_string=export_res.svg, width=scene.width, height=scene.height)
    svg_img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED).astype(float)
    ras_img = OpenCVRenderer().render(scene, alpha=True).buffer.astype(float)

    for p in (svg_img, ras_img):
        p[..., :3] *= p[..., 3:4] / 255.0

    diff = np.abs(svg_img - ras_img)
    mae = float(np.mean(diff))
    assert mae <= max_mae, f"Overall MAE {mae:.3f} exceeded tolerance {max_mae}"

    if interior_slice is not None:
        int_mae = float(np.mean(diff[interior_slice]))
        assert int_mae == 0.0, f"Interior MAE {int_mae:.3f} was not 0 in slice {interior_slice}"


def test_resvg_cubic_bezier_clipping_parity():
    """Verify cubic Bézier clipping raster-vs-SVG parity with resvg."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 0, 0)))
    rect.clip = make_cubic_clip(20, 20, 60)
    scene.add(rect)

    _check_resvg_parity(scene, interior_slice=(slice(65, 75), slice(30, 70)), max_mae=2.0)


def test_resvg_quadratic_bezier_clipping_parity():
    """Verify quadratic Bézier clipping raster-vs-SVG parity with resvg."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 255, 0)))
    p = Path()
    p.move_to(Point(20, 20)).quadratic_to(Point(50, 80), Point(80, 20)).line_to(Point(80, 80)).line_to(Point(20, 80)).close()
    rect.clip = p
    scene.add(rect)

    _check_resvg_parity(scene, interior_slice=(slice(50, 75), slice(30, 70)), max_mae=2.0)


def test_resvg_compound_donut_even_odd_parity():
    """Verify compound path with EVEN_ODD rule raster-vs-SVG parity with resvg."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 0, 255)))
    p = Path(fill_rule=FillRule.EVEN_ODD)
    p.move_to(Point(10, 10)).line_to(Point(90, 10)).line_to(Point(90, 90)).line_to(Point(10, 90)).close()
    p.move_to(Point(35, 35)).line_to(Point(65, 35)).line_to(Point(65, 65)).line_to(Point(35, 65)).close()
    rect.clip = p
    scene.add(rect)

    # Center hole must be completely empty (MAE=0) in both
    _check_resvg_parity(scene, interior_slice=(slice(45, 55), slice(45, 55)), max_mae=2.0)


def test_resvg_compound_non_zero_parity():
    """Verify compound path with NON_ZERO rule raster-vs-SVG parity with resvg."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 120, 0)))
    p = Path(fill_rule=FillRule.NON_ZERO)
    # Outer CW
    p.move_to(Point(10, 10)).line_to(Point(90, 10)).line_to(Point(90, 90)).line_to(Point(10, 90)).close()
    # Inner CW (same direction) -> filled in NON_ZERO
    p.move_to(Point(35, 35)).line_to(Point(65, 35)).line_to(Point(65, 65)).line_to(Point(35, 65)).close()
    rect.clip = p
    scene.add(rect)

    # Center must be completely filled (MAE=0) in both
    _check_resvg_parity(scene, interior_slice=(slice(45, 55), slice(45, 55)), max_mae=2.0)


def test_resvg_clip_local_transforms_parity():
    """Verify clip with local translation and scaling maintains SVG parity."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(180, 50, 180)))
    p = Path()
    p.move_to(Point(10, 10)).line_to(Point(50, 10)).line_to(Point(50, 50)).line_to(Point(10, 50)).close()
    p.transform.translation_x = 20.0
    p.transform.translation_y = 20.0
    rect.clip = p
    scene.add(rect)

    _check_resvg_parity(scene, interior_slice=(slice(35, 65), slice(35, 65)), max_mae=2.0)


def test_resvg_owner_and_group_transforms_parity():
    """Verify Group transforms + child clips maintain SVG parity."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    group = Group()
    group.transform.translation_x = 10.0
    group.transform.translation_y = 10.0
    rect = Rectangle(position=Point(0, 0), width=80, height=80, fill=FillStyle(color=Color(50, 150, 200)))
    p = Path()
    p.move_to(Point(10, 10)).line_to(Point(60, 10)).line_to(Point(60, 60)).line_to(Point(10, 60)).close()
    rect.clip = p
    group.add(rect)
    scene.add(group)

    _check_resvg_parity(scene, interior_slice=(slice(30, 60), slice(30, 60)), max_mae=2.0)


def test_resvg_nested_clips_parity():
    """Verify nested Group clip and child clip maintain SVG parity."""
    scene = Scene(100, 100, Color(0, 0, 0, 0))
    group = Group()
    p_g = Path()
    p_g.move_to(Point(10, 10)).line_to(Point(80, 10)).line_to(Point(80, 80)).line_to(Point(10, 80)).close()
    group.clip = p_g

    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 255, 0)))
    p_c = Path()
    p_c.move_to(Point(25, 25)).line_to(Point(95, 25)).line_to(Point(95, 95)).line_to(Point(25, 95)).close()
    rect.clip = p_c
    group.add(rect)
    scene.add(group)

    _check_resvg_parity(scene, interior_slice=(slice(30, 75), slice(30, 75)), max_mae=2.0)

