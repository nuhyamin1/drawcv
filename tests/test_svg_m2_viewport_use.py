"""Comprehensive unit and semantic tests for Milestone 2:
- <use> 3-tier hierarchy and x/y transform placement (Correction 1)
- SVG overflow semantics for nested svg, symbol, and root svg (Correction 2)
- Singular affine arcs strict rejection on import and export (Correction 3)
- <use> / <symbol> width-height auto semantics and symbol geometry (Correction 4)
- Style contract: use-level vs referenced non-inherited properties (Correction 5)
- Expansion bounds (max_use_instances, max_expanded_elements) and reference cycle detection
- Non-uniform rounded rectangles with exact EllipticalArcTo
- Retained clip geometries: Circle, Ellipse, RoundedRectangle
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from drawcv import (
    BlendMode,
    Circle,
    ClipPath,
    ClipRect,
    Color,
    Ellipse,
    EllipticalArcTo,
    Group,
    Line,
    Path,
    Point,
    Rectangle,
    RoundedRectangle,
    Scene,
    SVGExporter,
    SVGImportDiagnostic,
    SVGImportError,
    SVGImporter,
    SVGImportLimits,
    Transform,
)
from drawcv.core.exceptions import RenderError


# =============================================================================
# 1. Correction 1: <use> x/y transform placement and matrix order
# =============================================================================

class TestUseTransformPlacement:
    """Validate 3-tier UseHostGroup -> ViewportGroup -> ViewBoxGroup hierarchy and transform order."""

    def test_use_xy_transform_composition_order(self):
        # transform="scale(2, 3)" x="10" y="20"
        # Column-vector convention: T_host = T_use @ T_translate(x, y)
        # For point (0, 0): p' = T_use @ T_xy @ (0, 0, 1)^T = scale(2, 3) @ (10, 20, 1)^T = (20, 60)
        svg = """<svg width="200" height="200">
          <defs>
            <rect id="r" width="30" height="30"/>
          </defs>
          <use href="#r" transform="scale(2, 3)" x="10" y="20"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        assert isinstance(host_group, Group)

        # Check host_group local transform matrix
        M_host = host_group.transform.get_matrix(Point(0, 0))
        # Expected: scale(2, 3) @ translate(10, 20)
        # = [[2, 0, 0], [0, 3, 0], [0, 0, 1]] @ [[1, 0, 10], [0, 1, 20], [0, 0, 1]]
        # = [[2, 0, 20], [0, 3, 60], [0, 0, 1]]
        expected = np.array([
            [2.0, 0.0, 20.0],
            [0.0, 3.0, 60.0],
            [0.0, 0.0, 1.0],
        ])
        assert np.allclose(M_host, expected)

    def test_use_referencing_ordinary_content_creates_no_viewport(self):
        # Referencing <rect> directly does not establish a viewport even if width/height are set on <use>
        svg = """<svg width="200" height="200">
          <defs>
            <rect id="r" x="5" y="5" width="40" height="40"/>
          </defs>
          <use href="#r" x="15" y="25" width="100" height="100"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        assert isinstance(host_group, Group)
        # Only 1 child: the instantiated Rectangle directly
        assert len(host_group.children) == 1
        rect = host_group.children[0]
        assert isinstance(rect, Rectangle)
        assert rect.width == 40.0 and rect.height == 40.0
        # No viewport clip on host_group
        assert host_group.clip is None

    def test_use_referencing_symbol_three_tier_hierarchy(self):
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" viewBox="0 0 50 50" width="100" height="100">
              <rect width="50" height="50" fill="red"/>
            </symbol>
          </defs>
          <use href="#sym" x="20" y="30" width="80" height="80"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        assert isinstance(host_group, Group)
        # T_host translation is (20, 30)
        assert np.isclose(host_group.transform.translation_x, 20.0)
        assert np.isclose(host_group.transform.translation_y, 30.0)

        # Tier 2: ViewportGroup
        assert len(host_group.children) == 1
        viewport_group = host_group.children[0]
        assert isinstance(viewport_group, Group)
        assert np.allclose(viewport_group.transform.get_matrix(Point(0, 0)), np.eye(3))
        # Symbol defaults to hidden overflow -> viewport_clip is ClipRect(0, 0, 80, 80)
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.x == 0.0 and viewport_group.clip.y == 0.0
        assert viewport_group.clip.width == 80.0 and viewport_group.clip.height == 80.0

        # Tier 3: ViewBoxGroup
        assert len(viewport_group.children) == 1
        viewbox_group = viewport_group.children[0]
        assert isinstance(viewbox_group, Group)
        # Scale 80/50 = 1.6
        M_vb = viewbox_group.transform.get_matrix(Point(0, 0))
        assert np.isclose(M_vb[0, 0], 1.6)
        assert np.isclose(M_vb[1, 1], 1.6)


# =============================================================================
# 2. Correction 2: SVG Overflow Semantics
# =============================================================================

class TestSVGOverflowSemantics:
    """Explicit fixtures for all 6 overflow conditions specified by UA rules."""

    def test_nested_svg_default_overflow_is_hidden(self):
        # nested svg -> default hidden -> viewport clip present
        svg = """<svg width="200" height="200">
          <svg x="10" y="10" width="50" height="50">
            <rect width="100" height="100"/>
          </svg>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 50.0 and viewport_group.clip.height == 50.0

    def test_nested_svg_overflow_visible(self):
        # nested svg overflow="visible" -> no viewport clip
        svg = """<svg width="200" height="200">
          <svg x="10" y="10" width="50" height="50" overflow="visible">
            <rect width="100" height="100"/>
          </svg>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert viewport_group.clip is None

    def test_nested_svg_overflow_auto(self):
        # nested svg overflow="auto" -> static renderer: no viewport clip
        svg = """<svg width="200" height="200">
          <svg x="10" y="10" width="50" height="50" overflow="auto">
            <rect width="100" height="100"/>
          </svg>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert viewport_group.clip is None

    def test_nested_svg_overflow_hidden(self):
        # nested svg overflow="hidden" -> explicit hidden -> viewport clip present
        svg = """<svg width="200" height="200">
          <svg x="10" y="10" width="50" height="50" overflow="hidden">
            <rect width="100" height="100"/>
          </svg>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 50.0 and viewport_group.clip.height == 50.0

    def test_symbol_default_overflow_is_hidden(self):
        # symbol -> default hidden -> viewport clip present
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" viewBox="0 0 100 100">
              <rect width="100" height="100"/>
            </symbol>
          </defs>
          <use href="#sym" width="60" height="60"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 60.0 and viewport_group.clip.height == 60.0

    def test_symbol_overflow_visible(self):
        # symbol overflow="visible" -> no viewport clip
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" viewBox="0 0 100 100" overflow="visible">
              <rect width="100" height="100"/>
            </symbol>
          </defs>
          <use href="#sym" width="60" height="60"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        viewport_group = host_group.children[0]
        assert viewport_group.clip is None

    def test_root_svg_overflow_semantics(self):
        # root svg default -> not subject to svg:not(:root) rule -> visible (no root clip)
        svg_default = """<svg width="200" height="200"><rect width="300" height="300"/></svg>"""
        scene_default = SVGImporter().parse(svg_default).scene
        # Top-level objects are added directly to layer without synthetic clip group
        assert len(scene_default.layers[0].objects) == 1
        assert isinstance(scene_default.layers[0].objects[0], Rectangle)

        # root svg overflow="hidden" -> explicit author clip
        svg_hidden = """<svg width="200" height="200" overflow="hidden"><rect width="300" height="300"/></svg>"""
        scene_hidden = SVGImporter().parse(svg_hidden).scene
        root_group = scene_hidden.layers[0].objects[0]
        assert isinstance(root_group, Group)
        assert isinstance(root_group.clip, ClipRect)
        assert root_group.clip.width == 200.0 and root_group.clip.height == 200.0


# =============================================================================
# 3. Correction 3: Singular Affine Arcs Rejection
# =============================================================================

class TestSingularAffineArcsStrictRejection:
    """Validate deterministic rejection of singular transforms on EllipticalArcTo."""

    def test_import_singular_transform_on_arc_rejected(self):
        # scale(0, 1) collapses x dimension -> singular (det = 0)
        svg = """<svg width="100" height="100">
          <g transform="scale(0, 1)">
            <path d="M 10 10 A 20 20 0 0 0 50 50"/>
          </g>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TRANSFORM"

    def test_export_singular_transform_on_arc_rejected(self):
        # Path with EllipticalArcTo transformed by singular matrix raises RenderError
        p = Path()
        p.move_to(Point(10, 10))
        p.arc_to(20, 20, 0.0, False, False, 50, 50)
        p.transform = Transform.from_matrix(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))  # singular
        scene = Scene(width=100, height=100)
        scene.add(p)
        with pytest.raises(RenderError):
            SVGExporter(strict=True).render(scene)


# =============================================================================
# 4. Correction 4: <use> / <symbol> width-height auto semantics and geometry
# =============================================================================

class TestUseSymbolSizingAndGeometry:
    """Validate SVG2 used-value rules for width/height and symbol geometry attributes."""

    def test_explicit_use_dimensions_override_symbol(self):
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" width="50" height="60">
              <rect width="10" height="10"/>
            </symbol>
          </defs>
          <use href="#sym" width="120" height="140"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        viewport_group = scene.objects[0].children[0]
        assert viewport_group.clip.width == 120.0
        assert viewport_group.clip.height == 140.0

    def test_use_auto_dimensions_fallback_to_symbol(self):
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" width="75" height="85">
              <rect width="10" height="10"/>
            </symbol>
          </defs>
          <use href="#sym" width="auto" height="auto"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        viewport_group = scene.objects[0].children[0]
        assert viewport_group.clip.width == 75.0
        assert viewport_group.clip.height == 85.0

    def test_symbol_dimensions_still_auto_defaults_to_100_percent(self):
        # Neither use nor symbol specify width/height -> 100% of current viewport (200x200)
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym">
              <rect width="10" height="10"/>
            </symbol>
          </defs>
          <use href="#sym"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        viewport_group = scene.objects[0].children[0]
        assert viewport_group.clip.width == 200.0
        assert viewport_group.clip.height == 200.0

    def test_symbol_authored_x_and_y_geometry(self):
        # SVG2 geometry attributes on symbol: x="15" y="25"
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" x="15" y="25" width="60" height="60">
              <rect width="10" height="10"/>
            </symbol>
          </defs>
          <use href="#sym"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        viewport_group = scene.objects[0].children[0]
        assert np.isclose(viewport_group.transform.translation_x, 15.0)
        assert np.isclose(viewport_group.transform.translation_y, 25.0)

    def test_percentages_on_use_and_symbol_dimensions(self):
        svg = """<svg width="400" height="300">
          <defs>
            <symbol id="sym" width="50%" height="50%">
              <rect width="10" height="10"/>
            </symbol>
          </defs>
          <use href="#sym" width="auto" height="auto"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        viewport_group = scene.objects[0].children[0]
        # 50% of 400 = 200, 50% of 300 = 150
        assert viewport_group.clip.width == 200.0
        assert viewport_group.clip.height == 150.0


# =============================================================================
# 5. Correction 5: Style Contract (Use-level vs Referenced Non-inherited Properties)
# =============================================================================

class TestStyleContract:
    """Validate separation of use-host styles vs referenced-subtree styles."""

    def test_use_opacity_and_child_opacity_both_preserved(self):
        svg = """<svg width="200" height="200">
          <defs>
            <rect id="r" width="50" height="50" opacity="0.5" fill="red"/>
          </defs>
          <use href="#r" opacity="0.8" style="mix-blend-mode: multiply;"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        assert isinstance(host_group, Group)
        assert host_group.opacity == 0.8
        assert host_group.blend_mode == BlendMode.MULTIPLY

        # Child rectangle preserves its own opacity=0.5
        child = host_group.children[0]
        assert isinstance(child, Rectangle)
        assert child.opacity == 0.5
        assert child.fill.paint == Color(255, 0, 0, 1.0)

    def test_inherited_properties_cascade_from_use_host(self):
        svg = """<svg width="200" height="200">
          <defs>
            <g id="subtree">
              <rect width="50" height="50"/>
            </g>
          </defs>
          <use href="#subtree" fill="blue" stroke="green" stroke-width="4"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        child_group = host_group.children[0]
        rect = child_group.children[0]
        assert rect.fill.paint == Color(0, 0, 255, 1.0)
        assert rect.stroke.paint == Color(0, 128, 0, 1.0)
        assert rect.stroke.width == 4.0

    def test_use_level_clip_path_applies_to_use_host_group(self):
        svg = """<svg width="200" height="200">
          <defs>
            <clipPath id="cp">
              <rect x="0" y="0" width="30" height="30"/>
            </clipPath>
            <rect id="r" width="100" height="100" fill="red"/>
          </defs>
          <use href="#r" x="10" y="10" clip-path="url(#cp)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        host_group = scene.objects[0]
        assert host_group.clip is not None
        assert isinstance(host_group.clip, ClipRect)
        assert host_group.clip.width == 30.0 and host_group.clip.height == 30.0


# =============================================================================
# 6. Resource Limits and Cycle Detection
# =============================================================================

class TestResourceLimitsAndCycles:
    """Validate strict security limits on <use> expansion and reference graphs."""

    def test_direct_circular_reference_rejected(self):
        svg = """<svg width="100" height="100">
          <use id="a" href="#a"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_REFERENCE_CYCLE"

    def test_indirect_circular_reference_rejected(self):
        svg = """<svg width="100" height="100">
          <defs>
            <g id="a"><use href="#b"/></g>
            <g id="b"><use href="#a"/></g>
          </defs>
          <use href="#a"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_REFERENCE_CYCLE"

    def test_max_use_instances_limit_enforced(self):
        svg = """<svg width="100" height="100">
          <defs><rect id="r" width="10" height="10"/></defs>
          <use href="#r"/>
          <use href="#r"/>
          <use href="#r"/>
        </svg>"""
        limits = SVGImportLimits(max_use_instances=2)
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(limits=limits).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"

    def test_max_expanded_elements_limit_enforced(self):
        svg = """<svg width="100" height="100">
          <defs>
            <g id="group">
              <rect width="10" height="10"/>
              <circle r="5"/>
              <line x1="0" y1="0" x2="10" y2="10"/>
            </g>
          </defs>
          <use href="#group"/>
        </svg>"""
        limits = SVGImportLimits(max_expanded_elements=2)
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(limits=limits).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"


# =============================================================================
# 7. Phase 3: Exact Elliptical Shapes & Clipping Enhancements
# =============================================================================

class TestPhase3ExactEllipticalShapesAndClips:
    """Validate non-uniform rounded rects and retained clipping enhancements."""

    def test_non_uniform_rounded_rect_exact_arcs(self):
        svg = """<svg width="200" height="200">
          <rect x="10" y="20" width="100" height="80" rx="25" ry="15" fill="red"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        # Check subpath commands
        sub = p.subpaths[0]
        arcs = [cmd for cmd in sub.commands if isinstance(cmd, EllipticalArcTo)]
        assert len(arcs) == 4
        for arc in arcs:
            assert np.isclose(arc.radius_x, 25.0)
            assert np.isclose(arc.radius_y, 15.0)
            assert arc.x_axis_rotation == 0.0
            assert arc.large_arc is False
            assert arc.sweep is True

        # Strict SVG export produces native SVG 'A' commands with zero fallbacks
        export_svg = scene.export_svg(strict=True).svg
        assert " A 25 15" in export_svg or " A 25.0 15.0" in export_svg or "A 25" in export_svg

    def test_circle_clip_path_import_and_render(self):
        svg = """<svg width="100" height="100">
          <defs>
            <clipPath id="c">
              <circle cx="50" cy="50" r="30"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="blue" clip-path="url(#c)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rect = scene.objects[0]
        assert rect.clip is not None
        # Strict export succeeds without fallback
        export_svg = scene.export_svg(strict=True).svg
        assert "<clipPath" in export_svg
        assert "url(#" in export_svg

    def test_ellipse_clip_path_import_and_render(self):
        svg = """<svg width="100" height="100">
          <defs>
            <clipPath id="c">
              <ellipse cx="50" cy="50" rx="40" ry="25"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="green" clip-path="url(#c)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rect = scene.objects[0]
        assert rect.clip is not None
        export_svg = scene.export_svg(strict=True).svg
        assert "<clipPath" in export_svg

    def test_rounded_rect_clip_path_import_and_render(self):
        svg = """<svg width="100" height="100">
          <defs>
            <clipPath id="c">
              <rect x="10" y="10" width="80" height="80" rx="15" ry="10"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="purple" clip-path="url(#c)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rect = scene.objects[0]
        assert rect.clip is not None
        export_svg = scene.export_svg(strict=True).svg
        assert "<clipPath" in export_svg


# =============================================================================
# 7. Defect Review Regression Tests (Items E, F, G, J, K)
# =============================================================================

class TestDefectReviewRetainedSemanticsAndViewportClips:
    """Targeted regression tests for Defect Review Items E, F, G, J, K."""

    # -------------------------------------------------------------------------
    # Item E: Retained <use> / <symbol> / nested <svg> Semantics
    # -------------------------------------------------------------------------

    def test_use_referencing_svg_with_transform(self):
        # <use href="#svgWithTransform"> preserves separate UseHost and TargetHost transforms
        svg = """<svg width="300" height="300">
          <defs>
            <svg id="svgWithTransform" transform="rotate(45)" width="100" height="100">
              <rect width="100" height="100"/>
            </svg>
          </defs>
          <use href="#svgWithTransform" x="20" y="30"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        assert isinstance(use_host, Group)
        # UseHost has translate(20, 30)
        assert math.isclose(use_host.transform.translation_x, 20.0)
        assert math.isclose(use_host.transform.translation_y, 30.0)

        # TargetHost is child of UseHost
        assert len(use_host.children) == 1
        target_host = use_host.children[0]
        assert isinstance(target_host, Group)
        # TargetHost has rotate(45)
        M_tgt = target_host.transform.get_matrix(Point(0, 0))
        assert math.isclose(M_tgt[0, 0], math.cos(math.radians(45.0)), abs_tol=1e-5)
        assert math.isclose(M_tgt[1, 0], math.sin(math.radians(45.0)), abs_tol=1e-5)

        # ViewportGroup is child of TargetHost with ClipRect(0, 0, 100, 100)
        assert len(target_host.children) == 1
        viewport_group = target_host.children[0]
        assert isinstance(viewport_group, Group)
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 100.0 and viewport_group.clip.height == 100.0

    def test_use_referencing_svg_opacity_and_blend(self):
        # referenced svg opacity & blend mode composed with use opacity & blend mode
        svg = """<svg width="200" height="200">
          <defs>
            <svg id="s" width="100" height="100" opacity="0.6" style="mix-blend-mode: multiply">
              <rect width="100" height="100" fill="red"/>
            </svg>
          </defs>
          <use href="#s" opacity="0.5" style="mix-blend-mode: screen"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        assert isinstance(use_host, Group)
        assert math.isclose(use_host.opacity, 0.5)
        assert use_host.blend_mode == BlendMode.SCREEN

        target_host = use_host.children[0]
        assert isinstance(target_host, Group)
        assert math.isclose(target_host.opacity, 0.6)
        assert target_host.blend_mode == BlendMode.MULTIPLY

    def test_use_level_clip_plus_target_viewport_clip(self):
        # use-level clip on UseHost plus target viewport overflow clip on ViewportGroup
        svg = """<svg width="300" height="300">
          <defs>
            <clipPath id="useClip">
              <rect x="10" y="10" width="80" height="80"/>
            </clipPath>
            <svg id="tgt" width="100" height="100" overflow="hidden">
              <rect width="200" height="200" fill="blue"/>
            </svg>
          </defs>
          <use href="#tgt" clip-path="url(#useClip)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        assert isinstance(use_host, Group)
        # Use-level clip attached to UseHost
        assert use_host.clip is not None
        assert isinstance(use_host.clip, ClipRect)
        assert use_host.clip.width == 80.0 and use_host.clip.height == 80.0

        viewport_group = use_host.children[0]
        assert isinstance(viewport_group, Group)
        # Target viewport overflow clip attached to ViewportGroup
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 100.0 and viewport_group.clip.height == 100.0

    def test_use_referencing_svg_clip_path(self):
        # Referenced svg itself has authored clip-path
        svg = """<svg width="300" height="300">
          <defs>
            <clipPath id="svgClip">
              <circle cx="50" cy="50" r="30"/>
            </clipPath>
            <svg id="svgWithClip" width="100" height="100" clip-path="url(#svgClip)">
              <rect width="100" height="100" fill="green"/>
            </svg>
          </defs>
          <use href="#svgWithClip"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        assert isinstance(use_host, Group)
        target_host = use_host.children[0]
        assert isinstance(target_host, Group)
        # Referenced svg clip-path is attached to TargetHost
        assert target_host.clip is not None
        assert isinstance(target_host.clip, Path)
        viewport_group = target_host.children[0]
        assert isinstance(viewport_group.clip, ClipRect)

    def test_use_overflow_does_not_leak_to_symbol(self):
        # <use overflow="visible"> referencing <symbol> without authored overflow
        # Symbol's UA default overflow is hidden, and use's non-inherited overflow must not leak
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="sym" viewBox="0 0 100 100">
              <rect width="100" height="100"/>
            </symbol>
          </defs>
          <use href="#sym" width="80" height="80" overflow="visible"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        viewport_group = use_host.children[0]
        assert isinstance(viewport_group, Group)
        # ViewportGroup must still have viewport clip because symbol overflow is hidden by default
        assert isinstance(viewport_group.clip, ClipRect)
        assert viewport_group.clip.width == 80.0 and viewport_group.clip.height == 80.0

    # -------------------------------------------------------------------------
    # Item F: Viewport Clipping Coordinate Spaces
    # -------------------------------------------------------------------------

    def test_root_viewport_clip_outside_viewbox_group(self):
        # Root viewport clipping must occur in viewport coordinates, outside viewBox-transformed user-space Group
        svg = """<svg width="200" height="200" viewBox="0 0 100 100" overflow="hidden">
          <rect width="100" height="100" fill="red"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        root_viewport_group = scene.objects[0]
        assert isinstance(root_viewport_group, Group)
        # ClipRect is in root viewport space (200x200), transform is identity
        assert isinstance(root_viewport_group.clip, ClipRect)
        assert root_viewport_group.clip.width == 200.0 and root_viewport_group.clip.height == 200.0
        assert np.allclose(root_viewport_group.transform.get_matrix(Point(0, 0)), np.eye(3))

        # RootViewBoxGroup has M_viewbox transform (scale 2.0) and NO clip
        assert len(root_viewport_group.children) == 1
        viewbox_group = root_viewport_group.children[0]
        assert isinstance(viewbox_group, Group)
        assert viewbox_group.clip is None
        M_vb = viewbox_group.transform.get_matrix(Point(0, 0))
        assert math.isclose(M_vb[0, 0], 2.0)
        assert math.isclose(M_vb[1, 1], 2.0)

    def test_clip_path_units_object_bounding_box_percentages(self):
        # Percentage geometry in objectBoundingBox resolves against normalized [0, 1] space
        svg = """<svg width="500" height="500">
          <defs>
            <clipPath id="obb_rect" clipPathUnits="objectBoundingBox">
              <rect x="20%" y="20%" width="60%" height="60%"/>
            </clipPath>
            <clipPath id="obb_circle" clipPathUnits="objectBoundingBox">
              <circle cx="50%" cy="50%" r="50%"/>
            </clipPath>
            <clipPath id="obb_ellipse" clipPathUnits="objectBoundingBox">
              <ellipse cx="50%" cy="50%" rx="40%" ry="30%"/>
            </clipPath>
            <clipPath id="obb_arc" clipPathUnits="objectBoundingBox">
              <path d="M 0.2 0.5 A 0.3 0.3 0 0 1 0.8 0.5 Z"/>
            </clipPath>
          </defs>
          <rect id="r1" x="100" y="200" width="300" height="400" clip-path="url(#obb_rect)"/>
          <rect id="r2" x="100" y="200" width="300" height="400" clip-path="url(#obb_circle)"/>
          <rect id="r3" x="100" y="200" width="300" height="400" clip-path="url(#obb_ellipse)"/>
          <rect id="r4" x="100" y="200" width="300" height="400" clip-path="url(#obb_arc)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r1, r2, r3, r4 = scene.objects[0], scene.objects[1], scene.objects[2], scene.objects[3]

        assert r1.clip is not None
        assert r2.clip is not None
        assert r3.clip is not None
        assert r4.clip is not None

        # Export succeeds strictly with zero fallback
        exported = scene.export_svg(strict=True).svg
        assert "<clipPath" in exported

    # -------------------------------------------------------------------------
    # Item G: Close Strict-Import -> Strict-Export Arc Holes
    # -------------------------------------------------------------------------

    def test_singular_transform_on_drawable_with_arc_clips_rejected(self):
        # Singular transform on drawable + circle clip -> must be rejected during strict import
        svg_circle = """<svg width="200" height="200">
          <defs>
            <clipPath id="c">
              <circle cx="50" cy="50" r="40"/>
            </clipPath>
          </defs>
          <g transform="scale(0, 1)">
            <rect width="100" height="100" clip-path="url(#c)"/>
          </g>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg_circle)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TRANSFORM"

        # Also for ellipse clip
        svg_ellipse = """<svg width="200" height="200">
          <defs>
            <clipPath id="e">
              <ellipse cx="50" cy="50" rx="40" ry="20"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" transform="scale(1, 0)" clip-path="url(#e)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg_ellipse)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TRANSFORM"

        # Also for rounded rect clip
        svg_rrect = """<svg width="200" height="200">
          <defs>
            <clipPath id="rr">
              <rect width="100" height="100" rx="20" ry="10"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" transform="scale(0, 0)" clip-path="url(#rr)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg_rrect)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TRANSFORM"

    # -------------------------------------------------------------------------
    # Item J: Exact Expansion Counting
    # -------------------------------------------------------------------------

    def test_max_expanded_elements_exact_counting(self):
        # Each retained node materialized during expansion must be counted exactly once
        # <defs><g id="grp"><rect/><circle/><line/></g></defs>
        # Under <use href="#grp"/>:
        # instantiated target has 3 shape children + 1 group = 4 nodes, plus use_host = 5 nodes total.
        svg = """<svg width="200" height="200">
          <defs>
            <g id="grp">
              <rect width="10" height="10"/>
              <circle cx="5" cy="5" r="5"/>
              <line x1="0" y1="0" x2="10" y2="10"/>
            </g>
          </defs>
          <use href="#grp"/>
        </svg>"""
        # Exactly 5 nodes -> budget 5 must succeed
        limits_5 = SVGImportLimits(max_expanded_elements=5)
        scene = SVGImporter(limits=limits_5).parse(svg).scene
        assert len(scene.objects) == 1

        # Budget 4 must fail deterministically
        limits_4 = SVGImportLimits(max_expanded_elements=4)
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(limits=limits_4).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"

    # -------------------------------------------------------------------------
    # Item K: Distinguish Negative from Zero Viewport Dimensions
    # -------------------------------------------------------------------------

    def test_negative_viewport_dimensions_strict_error(self):
        # Negative width or height must raise SVG_INVALID_ATTRIBUTE_VALUE
        # 1. Nested svg
        svg_neg_w = """<svg width="200" height="200"><svg width="-10" height="50"><rect width="10" height="10"/></svg></svg>"""
        with pytest.raises(SVGImportError) as exc:
            SVGImporter().parse(svg_neg_w)
        assert exc.value.diagnostic.code == "SVG_INVALID_ATTRIBUTE_VALUE"

        svg_neg_h = """<svg width="200" height="200"><svg width="50" height="-10"><rect width="10" height="10"/></svg></svg>"""
        with pytest.raises(SVGImportError) as exc:
            SVGImporter().parse(svg_neg_h)
        assert exc.value.diagnostic.code == "SVG_INVALID_ATTRIBUTE_VALUE"

        # 2. Use referencing symbol
        svg_use_neg = """<svg width="200" height="200">
          <defs><symbol id="sym" viewBox="0 0 10 10"><rect width="10" height="10"/></symbol></defs>
          <use href="#sym" width="-5" height="20"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc:
            SVGImporter().parse(svg_use_neg)
        assert exc.value.diagnostic.code == "SVG_INVALID_ATTRIBUTE_VALUE"

    def test_zero_viewport_dimensions_omitted_non_rendering(self):
        # Zero width or height disables rendering as per SVG spec (not an error)
        # 1. Nested svg with width=0
        svg_zero_w = """<svg width="200" height="200"><svg width="0" height="50"><rect width="10" height="10"/></svg></svg>"""
        scene_zero_w = SVGImporter().parse(svg_zero_w).scene
        assert len(scene_zero_w.objects) == 0

        # 2. Use referencing symbol with height=0
        svg_use_zero = """<svg width="200" height="200">
          <defs><symbol id="sym" viewBox="0 0 10 10"><rect width="10" height="10"/></symbol></defs>
          <use href="#sym" width="50" height="0"/>
        </svg>"""
        scene_use_zero = SVGImporter().parse(svg_use_zero).scene
        assert len(scene_use_zero.objects) == 0

    # -------------------------------------------------------------------------
    # Item 2: Fix inherited visibility on container/use wrappers
    # -------------------------------------------------------------------------

    def test_use_visibility_hidden_with_descendant_visible(self):
        # <use visibility="hidden"> referencing <symbol> with an explicitly visible descendant
        svg = """<svg width="200" height="200">
          <defs>
            <symbol id="s" viewBox="0 0 100 100">
              <rect width="100" height="100" fill="red" visibility="hidden"/>
              <circle cx="50" cy="50" r="20" fill="green" visibility="visible"/>
            </symbol>
          </defs>
          <use href="#s" visibility="hidden"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        use_host = scene.objects[0]
        # Container groups stay render-open (visible=True)
        assert use_host.visible is True
        viewport_group = use_host.children[0]
        assert viewport_group.visible is True
        viewbox_group = viewport_group.children[0]
        assert viewbox_group.visible is True

        rect_child = viewbox_group.children[0]
        circle_child = viewbox_group.children[1]
        assert rect_child.visible is False
        assert circle_child.visible is True

        # Render with OpenCVRenderer: green circle renders at (100, 100) (viewBox 50, 50 scaled 2x), red rect does not
        from drawcv import OpenCVRenderer
        img = OpenCVRenderer().render(scene, alpha=True).buffer
        # Center (100, 100) is green [0, 128, 0, 255]
        assert np.array_equal(img[100, 100], [0, 128, 0, 255])
        # Corner (10, 10) is transparent [0, 0, 0, 0] because red rect was hidden
        assert np.array_equal(img[10, 10], [0, 0, 0, 0])

    def test_nested_svg_visibility_hidden_with_descendant_visible(self):
        svg = """<svg width="200" height="200">
          <svg x="10" y="10" width="100" height="100" visibility="hidden">
            <rect width="100" height="100" fill="red"/>
            <rect x="20" y="20" width="40" height="40" fill="blue" visibility="visible"/>
          </svg>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        nested_svg_host = scene.objects[0]
        assert nested_svg_host.visible is True
        viewbox_group = nested_svg_host.children[0].children[0]
        r1, r2 = viewbox_group.children[0], viewbox_group.children[1]
        assert r1.visible is False  # Inherited hidden
        assert r2.visible is True   # Explicitly visible

        from drawcv import OpenCVRenderer
        img = OpenCVRenderer().render(scene, alpha=True).buffer
        # Inside r2 (30 + 10, 30 + 10) = (40, 40) is blue [255, 0, 0, 255]
        assert np.array_equal(img[40, 40], [255, 0, 0, 255])
        # Outside r2 but inside r1 (15, 15) is transparent [0, 0, 0, 0]
        assert np.array_equal(img[15, 15], [0, 0, 0, 0])

    # -------------------------------------------------------------------------
    # Item 3: Implement SVG2 rx / ry auto semantics
    # -------------------------------------------------------------------------

    def test_svg2_rect_rx_ry_auto_semantics(self):
        # 1. rx=20, ry omitted -> both 20
        s1 = SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="20"/></svg>').scene
        assert isinstance(s1.objects[0], RoundedRectangle)
        assert s1.objects[0].corner_radius == 20.0

        # 2. ry=10, rx omitted -> both 10
        s2 = SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" ry="10"/></svg>').scene
        assert isinstance(s2.objects[0], RoundedRectangle)
        assert s2.objects[0].corner_radius == 10.0

        # 3. rx=auto, ry=10 -> both 10
        s3 = SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="auto" ry="10"/></svg>').scene
        assert isinstance(s3.objects[0], RoundedRectangle)
        assert s3.objects[0].corner_radius == 10.0

        # 4. rx=20, ry=auto -> both 20
        s4 = SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="20" ry="auto"/></svg>').scene
        assert isinstance(s4.objects[0], RoundedRectangle)
        assert s4.objects[0].corner_radius == 20.0

        # 5. rx=auto, ry=auto -> both 0 (normal Rectangle)
        s5 = SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="auto" ry="auto"/></svg>').scene
        assert isinstance(s5.objects[0], Rectangle)

    def test_svg2_ellipse_rx_ry_auto_semantics(self):
        # 1. rx=20, ry omitted -> ry=rx=20
        s1 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" rx="20"/></svg>').scene
        assert len(s1.objects) == 1
        assert isinstance(s1.objects[0], Ellipse)
        assert s1.objects[0].radius_x == 20.0 and s1.objects[0].radius_y == 20.0

        # 2. ry=20, rx omitted -> rx=ry=20
        s2 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" ry="20"/></svg>').scene
        assert len(s2.objects) == 1
        assert isinstance(s2.objects[0], Ellipse)
        assert s2.objects[0].radius_x == 20.0 and s2.objects[0].radius_y == 20.0

        # 3. rx=20, ry=auto -> ry=rx=20
        s3 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" rx="20" ry="auto"/></svg>').scene
        assert len(s3.objects) == 1
        assert s3.objects[0].radius_x == 20.0 and s3.objects[0].radius_y == 20.0

        # 4. rx=auto, ry=20 -> rx=ry=20
        s4 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" rx="auto" ry="20"/></svg>').scene
        assert len(s4.objects) == 1
        assert s4.objects[0].radius_x == 20.0 and s4.objects[0].radius_y == 20.0

        # 5. rx=auto, ry=auto -> non-rendering (omitted)
        s5 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" rx="auto" ry="auto"/></svg>').scene
        assert len(s5.objects) == 0

        # 6. both omitted -> non-rendering (omitted)
        s6 = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50"/></svg>').scene
        assert len(s6.objects) == 0

    def test_rect_explicit_zero_radii(self):
        """Verify that explicit zero rx or ry produces square-corner Rectangle."""
        cases = [
            '<svg width="100" height="50"><rect width="100" height="50" rx="0" ry="10"/></svg>',
            '<svg width="100" height="50"><rect width="100" height="50" rx="10" ry="0"/></svg>',
            '<svg width="100" height="50"><rect width="100" height="50" rx="0" ry="0"/></svg>',
            '<svg width="100" height="50"><rect width="100" height="50" rx="0%" ry="10"/></svg>',
        ]
        for svg in cases:
            scene = SVGImporter().parse(svg).scene
            assert len(scene.objects) == 1, f"Failed for {svg}"
            obj = scene.objects[0]
            assert isinstance(obj, Rectangle), f"Expected Rectangle, got {type(obj)} for {svg}"
            assert obj.width == 100.0 and obj.height == 50.0

        # Negative radii must continue to raise strict error
        with pytest.raises(SVGImportError, match="Negative rect rx"):
            SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="-5" ry="10"/></svg>')
        with pytest.raises(SVGImportError, match="Negative rect ry"):
            SVGImporter().parse('<svg width="100" height="50"><rect width="100" height="50" rx="10" ry="-5"/></svg>')

    def test_clip_path_zero_radius_rect(self):
        """Verify that zero-radius rect inside clipPath produces square clip without arcs."""
        # 1. Identity transform -> ClipRect
        svg1 = """<svg width="100" height="100">
          <defs>
            <clipPath id="c1">
              <rect x="10" y="20" width="60" height="40" rx="0" ry="10"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="red" clip-path="url(#c1)"/>
        </svg>"""
        scene1 = SVGImporter().parse(svg1).scene
        assert len(scene1.objects) == 1
        clip1 = scene1.objects[0].clip
        assert isinstance(clip1, ClipRect)
        assert clip1.x == 10.0 and clip1.y == 20.0 and clip1.width == 60.0 and clip1.height == 40.0

        # 2. Transformed rect with zero radius in clipPath -> Path without EllipticalArcTo
        svg2 = """<svg width="100" height="100">
          <defs>
            <clipPath id="c2">
              <rect x="10" y="20" width="60" height="40" rx="10" ry="0" transform="rotate(45)"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="red" clip-path="url(#c2)"/>
        </svg>"""
        scene2 = SVGImporter().parse(svg2).scene
        assert len(scene2.objects) == 1
        clip2 = scene2.objects[0].clip
        assert isinstance(clip2, Path)
        for subpath in clip2.subpaths:
            for cmd in subpath.commands:
                assert not isinstance(cmd, EllipticalArcTo)

    def test_use_referenced_svg_vs_symbol_display_none(self):
        """Verify display:none prunes referenced <svg>, but NOT directly referenced <symbol>."""
        # 1. Referenced <svg display="none"> -> pruned (no rendered objects)
        svg_hidden_svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <svg id="hiddenSvg" width="100" height="100" display="none">
              <rect width="100" height="100" fill="red"/>
            </svg>
          </defs>
          <use href="#hiddenSvg"/>
        </svg>"""
        scene_svg = SVGImporter().parse(svg_hidden_svg).scene
        assert len(scene_svg.objects) == 0

        # 2. Directly referenced <symbol display="none"> -> instantiated with display:inline (rendered)
        svg_hidden_symbol = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <symbol id="hiddenSymbol" display="none" viewBox="0 0 10 10">
              <rect width="10" height="10" fill="green"/>
            </symbol>
          </defs>
          <use href="#hiddenSymbol" width="100" height="100"/>
        </svg>"""
        scene_symbol = SVGImporter().parse(svg_hidden_symbol).scene
        assert len(scene_symbol.objects) == 1
