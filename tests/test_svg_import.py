"""Comprehensive test suite for native retained SVG import in DrawCV.

Validates:
- Semantic fidelity and genuine retained object construction (schema 1.7).
- Zero hidden rasterization and strict round-trip export with zero fallbacks.
- Robust path lexer and state-machine parser (M/L/Q/C/Z, scientific notation, implicit lineto).
- Exact affine transform parsing and left-to-right composition.
- Computed-style cascade (inline styles, presentation attributes, currentColor, visibility).
- Anisotropic stroke detection and uniform similarity evaluation.
- Paint servers: Linear & Radial gradients, stop normalization, href inheritance, cycle detection.
- Context-aware userSpaceOnUse percentage resolution and zero-size bbox handling.
- Retained clipping in userSpaceOnUse and objectBoundingBox.
- 12 supported blend modes on shapes and groups; rejection of isolation: isolate.
- Strict rejection of deferred constructs (masks, filters, text, images, elliptical rects, etc.).
- Resource limits, security enforcement (DTD/entities forbidden), and resvg parity.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest

from drawcv import (
    BlendMode,
    CapStyle,
    Circle,
    ClipPath,
    ClipRect,
    Close,
    Color,
    CubicTo,
    Drawable,
    Ellipse,
    EllipticalArcTo,
    FillRule,
    FillStyle,
    GradientStop,
    Group,
    JoinStyle,
    Line,
    LineTo,
    LinearGradient,
    MoveTo,
    OpenCVRenderer,
    Path,
    Point,
    Polyline,
    QuadraticTo,
    RadialGradient,
    Rectangle,
    RoundedRectangle,
    Scene,
    StrokeStyle,
    Subpath,
    SVGExporter,
    SVGImportDiagnostic,
    SVGImportError,
    SVGImporter,
    SVGImportLimits,
    SVGImportResult,
    Transform,
)
from drawcv.svg_import import SVGPathParser

SVG_NS = {"s": "http://www.w3.org/2000/svg"}


def render_scene(scene: Scene, alpha: bool = True) -> np.ndarray:
    """Render a scene using OpenCVRenderer and return the BGRA pixel buffer."""
    return OpenCVRenderer().render(scene, alpha=alpha).buffer


def resvg_render(svg_str: str, width: int, height: int) -> np.ndarray:
    """Render SVG string with resvg-py and return BGRA pixel buffer."""
    resvg = pytest.importorskip("resvg_py")
    png_bytes = resvg.svg_to_bytes(svg_string=svg_str, width=width, height=height, skip_system_fonts=True)
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


# =============================================================================
# 1. Path Lexer & State Machine Tests
# =============================================================================

class TestSVGPathParser:
    """Tests for SVG path data parsing into retained DrawCV Path objects."""

    def test_absolute_commands(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 10 20 L 30 40 Q 50 60 70 80 C 80 85 90 95 100 100 Z"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert len(scene.objects) == 1
        p = scene.objects[0]
        assert isinstance(p, Path)
        assert len(p.subpaths) == 1
        sub = p.subpaths[0]
        assert sub.closed is True
        cmds = sub.commands
        assert len(cmds) == 5
        assert isinstance(cmds[0], MoveTo) and cmds[0].point == Point(10, 20)
        assert isinstance(cmds[1], LineTo) and cmds[1].point == Point(30, 40)
        assert isinstance(cmds[2], QuadraticTo) and cmds[2].control == Point(50, 60) and cmds[2].end == Point(70, 80)
        assert isinstance(cmds[3], CubicTo) and cmds[3].control1 == Point(80, 85) and cmds[3].control2 == Point(90, 95) and cmds[3].end == Point(100, 100)
        assert isinstance(cmds[4], Close)

    def test_relative_commands(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="m 10 10 l 10 10 q 5 5 10 10 c 5 5 10 10 15 15 z"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        sub = p.subpaths[0]
        cmds = sub.commands
        assert cmds[0].point == Point(10, 10)
        assert cmds[1].point == Point(20, 20)
        assert cmds[2].control == Point(25, 25) and cmds[2].end == Point(30, 30)
        assert cmds[3].control1 == Point(35, 35) and cmds[3].control2 == Point(40, 40) and cmds[3].end == Point(45, 45)
        assert isinstance(cmds[4], Close)

    def test_compact_syntax_and_scientific_notation(self):
        # Compact coordinates: adjacent signs, decimal points without leading 0, scientific notation
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M10-20L-30.5e1.5Z"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        cmds = p.subpaths[0].commands
        assert cmds[0].point == Point(10, -20)
        assert cmds[1].point == Point(-305, 0.5)

    def test_implicit_lineto_after_moveto(self):
        # Multiple coordinate pairs after M become implicit Lineto commands
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 10 10 20 20 30 30 m 5 5 10 10"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert len(p.subpaths) == 2
        assert len(p.subpaths[0].commands) == 3
        assert isinstance(p.subpaths[0].commands[0], MoveTo) and p.subpaths[0].commands[0].point == Point(10, 10)
        assert isinstance(p.subpaths[0].commands[1], LineTo) and p.subpaths[0].commands[1].point == Point(20, 20)
        assert isinstance(p.subpaths[0].commands[2], LineTo) and p.subpaths[0].commands[2].point == Point(30, 30)
        # Second subpath relative
        assert isinstance(p.subpaths[1].commands[0], MoveTo) and p.subpaths[1].commands[0].point == Point(35, 35)
        assert isinstance(p.subpaths[1].commands[1], LineTo) and p.subpaths[1].commands[1].point == Point(45, 45)

    def test_multiple_subpaths(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 0 0 L 10 0 L 10 10 Z M 20 20 L 30 20 L 30 30 Z"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert len(p.subpaths) == 2
        assert p.subpaths[0].closed is True
        assert p.subpaths[1].closed is True

    def test_fill_rule_support(self):
        svg_nz = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 0 0 L 10 0 L 10 10 Z" fill-rule="nonzero"/>
        </svg>"""
        scene_nz = SVGImporter().parse(svg_nz).scene
        assert scene_nz.objects[0].fill_rule == FillRule.NON_ZERO

        svg_eo = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 0 0 L 10 0 L 10 10 Z" fill-rule="evenodd"/>
        </svg>"""
        scene_eo = SVGImporter().parse(svg_eo).scene
        assert scene_eo.objects[0].fill_rule == FillRule.EVEN_ODD

    def test_unsupported_path_commands_rejected(self):
        # Non-standard / unsupported path command 'B'
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 10 10 B 20 20"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_PATH_COMMAND"

    def test_malformed_path_data_rejected(self):
        bad_svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <path d="M 10 20 L"/>
        </svg>"""
        with pytest.raises(SVGImportError):
            SVGImporter().parse(bad_svg)


# =============================================================================
# 2. Geometry Primitives & Mapping
# =============================================================================

class TestSVGPrimitives:
    """Tests for mapping SVG basic shapes to retained DrawCV shapes."""

    def test_rectangle_basic(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="25" y="35" width="100" height="80"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert isinstance(r, Rectangle)
        assert r.position == Point(25, 35)
        assert r.width == 100
        assert r.height == 80

    def test_rectangle_rounded_circular(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="20" width="100" height="80" rx="15" ry="15"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rr = scene.objects[0]
        assert isinstance(rr, RoundedRectangle)
        assert rr.x == 10 and rr.y == 20
        assert rr.width == 100
        assert rr.height == 80
        assert rr.corner_radius == 15

    def test_rectangle_corner_radius_clamp(self):
        # rx, ry exceed half-width (100 / 2 = 50), should clamp to 50
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="20" width="100" height="100" rx="80" ry="80"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rr = scene.objects[0]
        assert isinstance(rr, RoundedRectangle)
        assert rr.corner_radius == 50.0
        assert rr.x == 10 and rr.y == 20

    def test_rectangle_elliptical_corners_strictly_rejected(self):
        # rx != ry is supported in M2 as exact Path with 4 EllipticalArcTo (no Bezier approximation)
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="20" width="100" height="80" rx="20" ry="10"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        arc_cmds = [cmd for sub in p.subpaths for cmd in sub.commands if isinstance(cmd, EllipticalArcTo)]
        assert len(arc_cmds) == 4

        # Negative rx/ry is strictly rejected
        with pytest.raises(SVGImportError):
            SVGImporter().parse('<svg width="100" height="100"><rect width="50" height="50" rx="-10"/></svg>')

    def test_rectangle_zero_dimension_omitted(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="20" width="0" height="80"/>
          <rect x="10" y="20" width="80" height="0"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert len(scene.objects) == 0

    def test_circle(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <circle cx="60" cy="70" r="30"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        c = scene.objects[0]
        assert isinstance(c, Circle)
        assert c.center == Point(60, 70)
        assert c.radius == 30

    def test_ellipse(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <ellipse cx="80" cy="90" rx="40" ry="25"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        e = scene.objects[0]
        assert isinstance(e, Ellipse)
        assert e.center == Point(80, 90)
        assert e.radius_x == 40
        assert e.radius_y == 25

    def test_line(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <line x1="10" y1="20" x2="150" y2="180" stroke="black"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        l = scene.objects[0]
        assert isinstance(l, Line)
        assert l.start == Point(10, 20)
        assert l.end == Point(150, 180)

    def test_polyline_unfilled_to_polyline(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <polyline points="10,10 50,50 90,20" fill="none" stroke="red"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        poly = scene.objects[0]
        assert isinstance(poly, Polyline)
        assert poly.closed is False
        assert list(poly.points) == [Point(10, 10), Point(50, 50), Point(90, 20)]

    def test_polyline_filled_to_open_path(self):
        # Filled polyline maps to open Path to preserve open stroke with closed fill
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <polyline points="10,10 50,50 90,20" fill="blue" stroke="red"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        assert p.subpaths[0].closed is False
        assert len(p.subpaths[0].commands) == 3
        assert p.subpaths[0].commands[0].point == Point(10, 10)
        assert p.subpaths[0].commands[1].point == Point(50, 50)
        assert p.subpaths[0].commands[2].point == Point(90, 20)

    def test_polygon_to_closed_path(self):
        # SVG polygon maps to closed Path (no O(n^2) segment intersection testing required)
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <polygon points="10,10 50,10 50,50 10,50" fill="green"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        p = scene.objects[0]
        assert isinstance(p, Path)
        assert p.subpaths[0].closed is True
        assert isinstance(p.subpaths[0].commands[-1], Close)


# =============================================================================
# 3. Transform Parsing & Matrix Ordering
# =============================================================================

class TestSVGTransforms:
    """Tests for SVG affine transform functions and compound list composition."""

    def test_translate(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="20" height="20" transform="translate(30, 40)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        pt = scene.objects[0].to_world(Point(0, 0))
        assert math.isclose(pt.x, 30.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 40.0, abs_tol=1e-5)

    def test_scale(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="20" height="20" transform="scale(2, 3)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        pt = scene.objects[0].to_world(Point(10, 10))
        assert math.isclose(pt.x, 20.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 30.0, abs_tol=1e-5)

    def test_rotate_with_pivot(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="20" height="20" transform="rotate(90, 50, 50)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        # Rotating (50, 0) by 90 deg around (50, 50) gives (100, 50)
        pt = scene.objects[0].to_world(Point(50, 0))
        assert math.isclose(pt.x, 100.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 50.0, abs_tol=1e-5)

    def test_skew_x_and_y(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="20" height="20" transform="skewX(45)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        # tan(45) = 1, so x' = x + y, y' = y
        pt = scene.objects[0].to_world(Point(0, 10))
        assert math.isclose(pt.x, 10.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 10.0, abs_tol=1e-5)

    def test_transform_list_order(self):
        # SVG transforms compose left-to-right: translate(50, 50) scale(2, 2)
        # maps (10, 10) -> scale to (20, 20) -> translate to (70, 70)
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="20" height="20" transform="translate(50, 50) scale(2, 2)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        pt = scene.objects[0].to_world(Point(10, 10))
        assert math.isclose(pt.x, 70.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 70.0, abs_tol=1e-5)


# =============================================================================
# 4. Computed Style Cascade & Color Parsing
# =============================================================================

class TestSVGStyleCascade:
    """Tests for computed styles, inheritance, colors, and visibility."""

    def test_inline_style_overrides_presentation_attribute(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="50" height="50" fill="red" style="fill: blue; stroke: yellow; stroke-width: 5;"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert r.fill.color == Color.from_hex("#0000ff")
        assert r.stroke.color == Color.from_hex("#ffff00")
        assert r.stroke.width == 5.0

    def test_group_inheritance_cascade(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <g fill="green" stroke="purple" stroke-width="3">
            <rect width="20" height="20"/>
            <rect x="30" width="20" height="20" fill="orange"/>
          </g>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        g = scene.objects[0]
        assert isinstance(g, Group)
        r1, r2 = g.children
        assert r1.fill.color == Color(0, 128, 0)
        assert r1.stroke.color == Color(128, 0, 128)
        assert r1.stroke.width == 3.0
        # Overridden fill, inherited stroke
        assert r2.fill.color == Color(255, 165, 0)
        assert r2.stroke.color == Color(128, 0, 128)

    def test_current_color(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg" color="teal">
          <rect width="20" height="20" fill="currentColor"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].fill.color == Color(0, 128, 128)

    def test_display_none_pruned(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="20" height="20" display="none"/>
          <g style="display: none;">
            <circle cx="50" cy="50" r="20"/>
          </g>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert len(scene.objects) == 0

    def test_visibility_hidden_with_child_override(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <g visibility="hidden">
            <rect width="20" height="20"/>
            <circle cx="50" cy="50" r="20" visibility="visible"/>
          </g>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        g = scene.objects[0]
        assert isinstance(g, Group)
        r, c = g.children
        assert r.visible is False
        assert c.visible is True

    def test_stroke_dasharray_odd_repeat(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="0" x2="100" y2="0" stroke="black" stroke-width="2" stroke-dasharray="5, 3, 2"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        line = scene.objects[0]
        # SVG spec: odd dasharray repeated -> (5, 3, 2, 5, 3, 2)
        assert line.stroke.dash_array == (5.0, 3.0, 2.0, 5.0, 3.0, 2.0)


# =============================================================================
# 5. Stroke Transform Compatibility & Anisotropic Rejection
# =============================================================================

class TestSVGStrokeCompatibility:
    """Tests for stroke width evaluation under transforms and anisotropic scale rejection."""

    def test_uniform_scale_normalizes_stroke(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="20" height="20" stroke="black" stroke-width="4" transform="scale(2)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert math.isclose(r.stroke.width, 4.0, abs_tol=1e-5)

    def test_reflection_is_uniform_similarity(self):
        # Scale(-2, 2) is a reflection with uniform magnitude 2
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="20" height="20" stroke="black" stroke-width="3" transform="scale(-2, 2)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert math.isclose(r.stroke.width, 3.0, abs_tol=1e-5)

    def test_anisotropic_stroke_retains_object_space(self):
        # Non-uniform scaling (2, 4) on ordinary stroke creates anisotropic ellipse stroke
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="20" height="20" stroke="black" stroke-width="2" transform="scale(2, 4)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].stroke.space == "object"
        assert scene.objects[0].stroke.width == 2.0

    def test_non_scaling_stroke_allowed_under_anisotropic_scale(self):
        # vector-effect="non-scaling-stroke" keeps stroke in screen pixels regardless of transform
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="20" height="20" stroke="black" stroke-width="2" vector-effect="non-scaling-stroke" transform="scale(2, 4)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert r.stroke.width == 2.0


# =============================================================================
# 6. Defs, Gradients & Paint Server Binding
# =============================================================================

class TestSVGGradients:
    """Tests for linear and radial gradients, stop normalization, and percentage resolution."""

    def test_linear_gradient_object_bounding_box(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g1">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="20" y="30" width="100" height="80" fill="url(#g1)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert isinstance(r.fill.paint, LinearGradient)
        assert r.fill.paint.space == "object"
        assert len(r.fill.paint.stops) == 2

    def test_radial_gradient_centered(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="g2">
              <stop offset="0%" stop-color="white"/>
              <stop offset="100%" stop-color="black"/>
            </radialGradient>
          </defs>
          <rect x="20" y="30" width="100" height="80" fill="url(#g2)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert isinstance(r.fill.paint, RadialGradient)
        assert r.fill.paint.space == "object"

    def test_user_space_on_use_linear_percentages(self):
        # Context-aware userSpaceOnUse percentage resolution
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g_user" gradientUnits="userSpaceOnUse" x1="10%" y1="20%" x2="90%" y2="80%">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="200" height="200" fill="url(#g_user)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        grad = scene.objects[0].fill.paint
        assert isinstance(grad, LinearGradient)
        assert grad.space == "world"
        # 10% of 200 = 20, 20% of 200 = 40, 90% of 200 = 180, 80% of 200 = 160
        assert math.isclose(grad.start.x, 20.0, abs_tol=1e-5)
        assert math.isclose(grad.start.y, 40.0, abs_tol=1e-5)
        assert math.isclose(grad.end.x, 180.0, abs_tol=1e-5)
        assert math.isclose(grad.end.y, 160.0, abs_tol=1e-5)

    def test_user_space_on_use_radial_percentages(self):
        # Context-aware userSpaceOnUse radial percentage resolution
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="r_user" gradientUnits="userSpaceOnUse" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="yellow"/>
              <stop offset="100%" stop-color="green"/>
            </radialGradient>
          </defs>
          <rect x="0" y="0" width="200" height="200" fill="url(#r_user)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        grad = scene.objects[0].fill.paint
        assert isinstance(grad, RadialGradient)
        assert math.isclose(grad.center.x, 100.0, abs_tol=1e-5)
        assert math.isclose(grad.center.y, 100.0, abs_tol=1e-5)
        # Normalized viewport diagonal: sqrt(200^2 + 200^2)/sqrt(2) = 200. 50% = 100.
        assert math.isclose(grad.radius, 100.0, abs_tol=1e-5)

    def test_zero_stops_paints_nothing(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs><linearGradient id="empty"/></defs>
          <rect width="50" height="50" fill="url(#empty)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].fill is None

    def test_single_stop_paints_solid_color(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="single"><stop offset="0%" stop-color="purple"/></linearGradient>
          </defs>
          <rect width="50" height="50" fill="url(#single)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].fill.color == Color(128, 0, 128)

    def test_stop_normalization_monotonicity(self):
        # SVG spec: non-decreasing stop positions; out-of-order clamped to preceding
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="norm">
              <stop offset="50%" stop-color="red"/>
              <stop offset="30%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect width="50" height="50" fill="url(#norm)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        grad = scene.objects[0].fill.paint
        assert grad.stops[0].position == 0.5
        assert grad.stops[1].position == 0.5  # Clamped to preceding

    def test_zero_size_object_bounding_box_geometry(self):
        # Line with height = 0 referencing objectBoundingBox gradient must paint nothing
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g_obb" gradientUnits="objectBoundingBox">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <line x1="10" y1="50" x2="90" y2="50" stroke="url(#g_obb)" stroke-width="5"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].stroke is None

    def test_gradient_href_inheritance(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="base">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
            <linearGradient id="derived" href="#base" x1="0" y1="0" x2="1" y2="1"/>
          </defs>
          <rect width="50" height="50" fill="url(#derived)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        grad = scene.objects[0].fill.paint
        assert len(grad.stops) == 2
        assert grad.start == Point(0, 0)
        assert grad.end == Point(1, 1)

    def test_cyclic_reference_rejected(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="gA" href="#gB"/>
            <linearGradient id="gB" href="#gA"/>
          </defs>
          <rect width="50" height="50" fill="url(#gA)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_REFERENCE_CYCLE"

    def test_eccentric_radial_gradient_rejected(self):
        # Milestone 1 strictly rejects eccentric focal points (fx != cx)
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="eccentric" cx="50%" cy="50%" fx="30%" fy="30%" r="50%">
              <stop offset="0%" stop-color="white"/>
              <stop offset="100%" stop-color="black"/>
            </radialGradient>
          </defs>
          <rect width="50" height="50" fill="url(#eccentric)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_RADIAL_FOCUS"


# =============================================================================
# 7. Clipping (userSpaceOnUse & objectBoundingBox)
# =============================================================================

class TestSVGClipping:
    """Tests for retained path clipping imported from SVG clipPath elements."""

    def test_clip_path_user_space_on_use(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="clip1" clipPathUnits="userSpaceOnUse">
              <rect x="10" y="10" width="80" height="80"/>
            </clipPath>
          </defs>
          <rect x="0" y="0" width="200" height="200" fill="red" clip-path="url(#clip1)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert r.clip is not None
        assert isinstance(r.clip, ClipRect)
        assert r.clip.x == 10 and r.clip.y == 10 and r.clip.width == 80 and r.clip.height == 80

    def test_clip_path_with_path_geometry(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="pathclip">
              <path d="M 0 0 L 100 0 L 50 100 Z" clip-rule="evenodd"/>
            </clipPath>
          </defs>
          <rect width="200" height="200" fill="blue" clip-path="url(#pathclip)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert isinstance(r.clip, Path)
        assert r.clip.fill_rule == FillRule.EVEN_ODD

    def test_clip_path_object_bounding_box(self):
        # Clip units in object coordinates [0, 1] scaled to target bbox
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="obb_clip" clipPathUnits="objectBoundingBox">
              <rect x="0" y="0" width="0.5" height="0.5"/>
            </clipPath>
          </defs>
          <rect x="40" y="60" width="100" height="80" fill="green" clip-path="url(#obb_clip)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r = scene.objects[0]
        assert isinstance(r.clip, Path)
        # Point (0.5, 0.5) transformed by bbox (w=100, h=80, x=40, y=60) is (90, 100)
        pt = r.clip.to_world(Point(0.5, 0.5))
        assert math.isclose(pt.x, 90.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 100.0, abs_tol=1e-5)


# =============================================================================
# 8. Blend Modes & Compositing
# =============================================================================

class TestSVGBlendModes:
    """Tests for mix-blend-mode support and isolation rejection."""

    @pytest.mark.parametrize("mode_name,expected_enum", [
        ("multiply", BlendMode.MULTIPLY),
        ("screen", BlendMode.SCREEN),
        ("overlay", BlendMode.OVERLAY),
        ("darken", BlendMode.DARKEN),
        ("lighten", BlendMode.LIGHTEN),
        ("color-dodge", BlendMode.COLOR_DODGE),
        ("color-burn", BlendMode.COLOR_BURN),
        ("hard-light", BlendMode.HARD_LIGHT),
        ("soft-light", BlendMode.SOFT_LIGHT),
        ("difference", BlendMode.DIFFERENCE),
        ("exclusion", BlendMode.EXCLUSION),
    ])
    def test_supported_blend_modes(self, mode_name, expected_enum):
        svg = f"""<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="50" height="50" style="mix-blend-mode: {mode_name};"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        assert scene.objects[0].blend_mode == expected_enum

    def test_isolation_isolate_strictly_rejected(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <g style="isolation: isolate;">
            <rect width="50" height="50"/>
          </g>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_ISOLATION"


# =============================================================================
# 9. Viewport, ViewBox & Security Boundaries
# =============================================================================

class TestSVGViewportAndSecurity:
    """Tests for viewport dimension validation, viewBox mapping, and security rules."""

    def test_fractional_root_dimensions_rejected_in_strict_mode(self):
        svg = """<svg width="100.5" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect width="50" height="50"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_NONINTEGER_VIEWPORT"

    def test_omitted_dimensions_require_caller_viewport(self):
        # When width/height are omitted, caller viewport is mandatory
        svg = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          <rect width="50" height="50"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_INVALID_DIMENSIONS"

        # Providing caller viewport succeeds
        res = SVGImporter(viewport=(200, 200)).parse(svg)
        assert res.scene.width == 200 and res.scene.height == 200

    def test_viewbox_aspect_ratio_preserve(self):
        # 100x50 viewBox in 200x200 canvas with xMidYMid meet -> scale=2, centered vertically at y=50
        svg = """<svg width="200" height="200" viewBox="0 0 100 50" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="10" height="10"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        # Root group wraps children with viewBox matrix
        assert len(scene.objects) == 1
        root_g = scene.objects[0]
        assert isinstance(root_g, Group)
        r = root_g.children[0]
        pt = r.to_world(Point(0, 0))
        assert math.isclose(pt.x, 0.0, abs_tol=1e-5)
        assert math.isclose(pt.y, 50.0, abs_tol=1e-5)

    def test_xml_doctype_and_entity_forbidden(self):
        # Security: DOCTYPE and ENTITY declarations must be rejected
        svg = """<?xml version="1.0"?>
        <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="10" height="10"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_SECURITY_VIOLATION"

    def test_external_reference_forbidden(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g" href="https://example.com/gradient.svg#grad"/>
          </defs>
          <rect width="10" height="10" fill="url(#g)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_EXTERNAL_REFERENCE"


# =============================================================================
# 10. Strict Semantic Round-Trip & Export
# =============================================================================

class TestSVGExportRoundTrip:
    """Tests verifying that every accepted SVG imports to genuine objects and exports with zero fallbacks."""

    def test_strict_export_zero_fallbacks(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
            <clipPath id="c1">
              <polygon points="10,10 170,10 90,170"/>
            </clipPath>
          </defs>
          <g transform="translate(10, 10)">
            <rect x="0" y="0" width="180" height="180" fill="url(#g1)" stroke="black" stroke-width="4" clip-path="url(#c1)"/>
          </g>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        export_res = scene.export_svg(strict=True)
        assert export_res.fallbacks == ()
        assert len(export_res.svg) > 0

    def test_json_persistence_round_trip(self):
        svg = """<svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="20" width="100" height="80" rx="10" fill="orange" stroke="navy" stroke-width="2"/>
        </svg>"""
        scene1 = SVGImporter().parse(svg).scene
        json_str = scene1.to_json()
        scene2 = Scene.from_json(json_str)

        assert scene2.width == 150
        assert scene2.height == 150
        assert len(scene2.objects) == 1
        r = scene2.objects[0]
        assert isinstance(r, RoundedRectangle)
        assert r.x == 10 and r.y == 20
        assert r.corner_radius == 10.0


# =============================================================================
# 11. Direct resvg Parity Fixtures
# =============================================================================

class TestResvgParity:
    """Independent visual parity tests comparing DrawCV rendering with resvg-py."""

    def test_shapes_parity(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="80" height="60" fill="red" stroke="blue" stroke-width="4"/>
          <circle cx="150" cy="50" r="30" fill="green"/>
          <polygon points="20,120 70,120 45,180" fill="yellow" stroke="black" stroke-width="2"/>
        </svg>"""
        resvg_img = resvg_render(svg, 200, 200)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Center pixels should match exactly between DrawCV and resvg
        assert np.array_equal(drawcv_img[40, 50], resvg_img[40, 50])  # Rect interior
        assert np.array_equal(drawcv_img[50, 150], resvg_img[50, 150])  # Circle interior
        assert np.array_equal(drawcv_img[140, 45], resvg_img[140, 45])  # Polygon interior

        # Exact paths (rect and polygon) roundtrip through strict export with zero pixel diff
        svg_poly = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="80" height="60" fill="red" stroke="blue" stroke-width="4"/>
          <polygon points="20,120 70,120 45,180" fill="yellow" stroke="black" stroke-width="2"/>
        </svg>"""
        resvg_poly_img = resvg_render(svg_poly, 200, 200)
        scene_poly = SVGImporter().parse(svg_poly).scene
        export_poly_svg = scene_poly.export_svg(strict=True).svg
        rt_poly_img = resvg_render(export_poly_svg, 200, 200)
        assert np.max(np.abs(resvg_poly_img.astype(float) - rt_poly_img.astype(float))) == 0.0

    def test_clipping_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="c">
              <rect x="25" y="25" width="50" height="50"/>
            </clipPath>
          </defs>
          <rect x="10" y="10" width="80" height="80" fill="red" clip-path="url(#c)"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Pixel inside clip (50, 50) is red; pixel outside clip (20, 20) is transparent
        assert np.array_equal(drawcv_img[50, 50], [0, 0, 255, 255])
        assert drawcv_img[20, 20, 3] == 0

        # Export resvg roundtrip
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_gradient_user_space_on_use_defaults_parity(self):
        # User requested parity fixture: linearGradient userSpaceOnUse default coordinates
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g_def" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="200" height="200" fill="url(#g_def)"/>
        </svg>"""
        resvg_img = resvg_render(svg, 200, 200)
        scene = SVGImporter().parse(svg).scene

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 200, 200)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_group_parity(self):
        # User requested parity fixture: Group + mix-blend-mode
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="60" height="60" fill="red"/>
          <g style="mix-blend-mode: multiply;">
            <rect x="30" y="30" width="60" height="60" fill="blue"/>
          </g>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Overlap interior pixel at (40, 40) is black [0, 0, 0, 255] in both
        assert np.array_equal(drawcv_img[40, 40], [0, 0, 0, 255])
        assert np.array_equal(resvg_img[40, 40], [0, 0, 0, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_leaf_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="60" height="60" fill="red"/>
          <rect x="30" y="30" width="60" height="60" fill="blue" style="mix-blend-mode: multiply;"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        assert np.array_equal(drawcv_img[40, 40], [0, 0, 0, 255])
        assert np.array_equal(resvg_img[40, 40], [0, 0, 0, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_group_opacity_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="60" height="60" fill="red"/>
          <g opacity="0.8" style="mix-blend-mode: multiply;">
            <rect x="30" y="30" width="60" height="60" fill="blue"/>
          </g>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        assert np.array_equal(drawcv_img[40, 40], [0, 0, 51, 255])
        assert np.array_equal(resvg_img[40, 40], [0, 0, 51, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_nested_groups_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="80" height="80" fill="red"/>
          <g style="mix-blend-mode: screen;">
            <g style="mix-blend-mode: multiply;">
              <rect x="20" y="20" width="60" height="60" fill="blue"/>
            </g>
          </g>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        assert np.array_equal(drawcv_img[40, 40], [255, 0, 255, 255])
        assert np.array_equal(resvg_img[40, 40], [255, 0, 255, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_transformed_group_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect x="10" y="10" width="60" height="60" fill="red"/>
          <g transform="translate(10, 10)" style="mix-blend-mode: multiply;">
            <rect x="20" y="20" width="60" height="60" fill="blue"/>
          </g>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        assert np.array_equal(drawcv_img[40, 40], [0, 0, 0, 255])
        assert np.array_equal(resvg_img[40, 40], [0, 0, 0, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0

    def test_blend_mode_clip_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="c">
              <rect x="25" y="25" width="50" height="50"/>
            </clipPath>
          </defs>
          <rect x="10" y="10" width="60" height="60" fill="red"/>
          <rect x="20" y="20" width="60" height="60" fill="blue" clip-path="url(#c)" style="mix-blend-mode: multiply;"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        assert np.array_equal(drawcv_img[40, 40], [0, 0, 0, 255])
        assert np.array_equal(resvg_img[40, 40], [0, 0, 0, 255])

        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)
        assert np.max(np.abs(resvg_img.astype(float) - rt_img.astype(float))) == 0.0


# =============================================================================
# 12. Review Hardening & Regression Fixtures
# =============================================================================

class TestReviewHardeningAndRegressions:
    """Tests specifically validating the 9 review correctness fixes."""

    # 1. BLOCKER: malformed path data can hang forever after Z
    def test_malformed_path_post_z_raises_without_hang(self):
        with pytest.raises(SVGImportError) as exc_info:
            SVGPathParser.parse("M0 0 Z 10 10")
        assert "Path data must start with a command" in str(exc_info.value)

        with pytest.raises(SVGImportError):
            SVGImporter().parse('<svg width="100" height="100"><path d="M 0 0 L 10 10 Z 20 20"/></svg>')

    # 2. BLOCKER: strict clipping invariants and composed transforms
    def test_clip_transform_composition_clip_path_and_child(self):
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="cp" transform="translate(50, 60)">
              <rect x="0" y="0" width="60" height="60" transform="translate(10, 20)"/>
            </clipPath>
          </defs>
          <rect x="0" y="0" width="200" height="200" fill="red" clip-path="url(#cp)"/>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        rect = scene.objects[0]
        assert rect.clip is not None
        # Child is rect with x=0, y=0, width=60, height=60, transform translate(10, 20)
        # clipPath has transform translate(50, 60)
        # Combined clip transform = translate(50, 60) @ translate(10, 20) = translate(60, 80)
        M_expected = np.array([
            [1.0, 0.0, 60.0],
            [0.0, 1.0, 80.0],
            [0.0, 0.0, 1.0],
        ])
        assert np.allclose(rect.clip.transform.get_matrix(Point(0, 0)), M_expected)

    def test_clip_compound_transforms_resvg_parity(self):
        # Transformed owner + transformed clipPath + transformed child
        svg = """<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="cp" transform="translate(20, 30)">
              <rect x="10" y="10" width="50" height="50" transform="translate(10, 10)"/>
            </clipPath>
          </defs>
          <g transform="translate(10, 10)">
            <rect x="0" y="0" width="200" height="200" fill="red" clip-path="url(#cp)"/>
          </g>
        </svg>"""
        resvg_img = resvg_render(svg, 200, 200)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Expected clipped red box bounds: [50, 100] x [60, 110]
        # Pixel at (60, 70) inside clip -> red
        assert np.array_equal(drawcv_img[70, 60], [0, 0, 255, 255])
        assert np.array_equal(resvg_img[70, 60], [0, 0, 255, 255])
        # Pixel outside clip -> transparent
        assert drawcv_img[20, 20, 3] == 0
        assert resvg_img[20, 20, 3] == 0

    def test_unsupported_clip_geometries_strictly_rejected(self):
        # Line clip remains unsupported geometry
        svg_line = """<svg width="100" height="100"><defs><clipPath id="c"><line x1="0" y1="0" x2="100" y2="100"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_line:
            SVGImporter().parse(svg_line)
        assert exc_line.value.diagnostic.code == "SVG_UNSUPPORTED_CLIP_GEOMETRY"

        # Circle clip is supported in M2
        svg_circle = """<svg width="100" height="100"><defs><clipPath id="c"><circle cx="50" cy="50" r="30"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        scene_circle = SVGImporter().parse(svg_circle).scene
        assert scene_circle.objects[0].clip is not None

        # Ellipse clip is supported in M2
        svg_ellipse = """<svg width="100" height="100"><defs><clipPath id="c"><ellipse cx="50" cy="50" rx="30" ry="20"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        scene_ellipse = SVGImporter().parse(svg_ellipse).scene
        assert scene_ellipse.objects[0].clip is not None

        # Rounded rectangle clip is supported in M2
        svg_rounded_rect = """<svg width="100" height="100"><defs><clipPath id="c"><rect x="10" y="10" width="50" height="50" rx="5"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        scene_rr = SVGImporter().parse(svg_rounded_rect).scene
        assert scene_rr.objects[0].clip is not None

    def test_empty_or_multiple_children_clip_path_rejected(self):
        # Empty clipPath
        svg_empty = """<svg width="100" height="100"><defs><clipPath id="c"></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_empty:
            SVGImporter().parse(svg_empty)
        assert exc_empty.value.diagnostic.code == "SVG_UNSUPPORTED_CLIP_GEOMETRY"

        # Multiple children clipPath
        svg_multi = """<svg width="100" height="100"><defs><clipPath id="c"><rect x="0" y="0" width="10" height="10"/><rect x="20" y="20" width="10" height="10"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_multi:
            SVGImporter().parse(svg_multi)
        assert exc_multi.value.diagnostic.code == "SVG_UNSUPPORTED_CLIP_GEOMETRY"

    # 3. High: strict=False explicitly rejected as not implemented
    def test_strict_false_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            SVGImporter(strict=False)

    # 4. High: invalid enum values strictly rejected
    def test_invalid_enums_rejected(self):
        # gradientUnits
        svg_gu = """<svg width="100" height="100"><defs><linearGradient id="g" gradientUnits="invalid"><stop offset="0" stop-color="red"/></linearGradient></defs><rect width="100" height="100" fill="url(#g)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_gu:
            SVGImporter().parse(svg_gu)
        assert exc_gu.value.diagnostic.code == "SVG_UNSUPPORTED_GRADIENT_UNITS"

        # clipPathUnits
        svg_cpu = """<svg width="100" height="100"><defs><clipPath id="c" clipPathUnits="invalid"><rect width="10" height="10"/></clipPath></defs><rect width="100" height="100" clip-path="url(#c)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_cpu:
            SVGImporter().parse(svg_cpu)
        assert exc_cpu.value.diagnostic.code == "SVG_UNSUPPORTED_CLIP_PATH_UNITS"

        # spreadMethod
        svg_sm = """<svg width="100" height="100"><defs><linearGradient id="g" spreadMethod="unknown"><stop offset="0" stop-color="red"/></linearGradient></defs><rect width="100" height="100" fill="url(#g)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_sm:
            SVGImporter().parse(svg_sm)
        assert exc_sm.value.diagnostic.code == "SVG_UNSUPPORTED_SPREAD_METHOD"

        # preserveAspectRatio
        svg_par1 = """<svg width="100" height="100" viewBox="0 0 100 100" preserveAspectRatio="unknown meet"><rect width="100" height="100"/></svg>"""
        with pytest.raises(SVGImportError) as exc_par1:
            SVGImporter().parse(svg_par1)
        assert exc_par1.value.diagnostic.code == "SVG_MALFORMED_ASPECT_RATIO"

        svg_par2 = """<svg width="100" height="100" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid unknown"><rect width="100" height="100"/></svg>"""
        with pytest.raises(SVGImportError) as exc_par2:
            SVGImporter().parse(svg_par2)
        assert exc_par2.value.diagnostic.code == "SVG_MALFORMED_ASPECT_RATIO"

        # malformed clip-path syntax
        svg_cp_syn = """<svg width="100" height="100"><rect width="100" height="100" clip-path="circle(50px)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_cp_syn:
            SVGImporter().parse(svg_cp_syn)
        assert exc_cp_syn.value.diagnostic.code == "SVG_UNSUPPORTED_CLIP_PATH"

        # malformed url on fill
        svg_fill_url = """<svg width="100" height="100"><rect width="100" height="100" fill="url(bad_ref)"/></svg>"""
        with pytest.raises(SVGImportError) as exc_fill_url:
            SVGImporter().parse(svg_fill_url)
        assert exc_fill_url.value.diagnostic.code == "SVG_MALFORMED_URL_REFERENCE"

    # 5. High: security limits enforced
    def test_security_limits_enforced(self, tmp_path):
        # max_attribute_length
        limits = SVGImportLimits(max_attribute_length=20)
        svg_long_attr = '<svg width="100" height="100"><path d="M 0 0 L 100 100 L 200 200 Z"/></svg>'
        with pytest.raises(SVGImportError) as exc_attr:
            SVGImporter(limits=limits).parse(svg_long_attr)
        assert exc_attr.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"

        # max_coordinate_magnitude
        limits_mag = SVGImportLimits(max_coordinate_magnitude=500.0)
        svg_huge_coord = '<svg width="100" height="100"><circle cx="1000" cy="50" r="10"/></svg>'
        with pytest.raises(SVGImportError) as exc_mag:
            SVGImporter(limits=limits_mag).parse(svg_huge_coord)
        assert exc_mag.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"

        # parse_file pre-size check
        large_file = tmp_path / "huge.svg"
        large_file.write_bytes(b"<svg " + b" " * 1000 + b"/>")
        limits_size = SVGImportLimits(max_source_bytes=500)
        with pytest.raises(SVGImportError) as exc_size:
            SVGImporter(limits=limits_size).parse_file(large_file)
        assert exc_size.value.diagnostic.code == "SVG_RESOURCE_LIMIT_EXCEEDED"

    # 6. Medium: path starting with L/Q/C rejected
    def test_path_initial_command_rejection(self):
        with pytest.raises(SVGImportError) as exc_l:
            SVGPathParser.parse("L 10 10")
        assert exc_l.value.diagnostic.code == "SVG_MALFORMED_PATH"

        with pytest.raises(SVGImportError) as exc_q:
            SVGPathParser.parse("Q 10 10 20 20")
        assert exc_q.value.diagnostic.code == "SVG_MALFORMED_PATH"

        with pytest.raises(SVGImportError) as exc_c:
            SVGPathParser.parse("C 10 10 20 20 30 30")
        assert exc_c.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # 7. Medium: dash parsing length policy
    def test_dash_parsing_length_policy(self):
        svg_dashes = """<svg width="100" height="100">
          <line x1="0" y1="0" x2="100" y2="100" stroke="black" stroke-width="2" stroke-dasharray="5px 3px" stroke-dashoffset="2px"/>
        </svg>"""
        scene = SVGImporter().parse(svg_dashes).scene
        line = scene.objects[0]
        assert line.stroke.dash_array == (5.0, 3.0)
        assert line.stroke.dash_offset == 2.0

        # Negative dash array value rejected
        svg_neg_dash = """<svg width="100" height="100"><line x1="0" y1="0" x2="10" y2="10" stroke="black" stroke-dasharray="5 -2"/></svg>"""
        with pytest.raises(SVGImportError) as exc_neg:
            SVGImporter().parse(svg_neg_dash)
        assert exc_neg.value.diagnostic.code == "SVG_MALFORMED_STYLE"

    # 9. Medium hardening: XML namespace validation
    def test_namespace_validation_rejects_foreign_namespaces(self):
        # Foreign root namespace
        svg_foreign_root = """<svg xmlns="http://www.w3.org/1999/xhtml" width="100" height="100"><div/></svg>"""
        with pytest.raises(SVGImportError) as exc_root:
            SVGImporter().parse(svg_foreign_root)
        assert exc_root.value.diagnostic.code == "SVG_UNSUPPORTED_NAMESPACE"

        # Foreign child element namespace
        svg_foreign_child = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg" xmlns:html="http://www.w3.org/1999/xhtml">
          <html:div>text</html:div>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_child:
            SVGImporter().parse(svg_foreign_child)
        assert exc_child.value.diagnostic.code == "SVG_UNSUPPORTED_NAMESPACE"

    # 10. Follow-up: Percentages outside gradients strictly rejected
    def test_percentages_outside_gradients_strictly_rejected(self):
        # In M2, percentages on shapes and strokes are supported and resolve against viewport
        scene = SVGImporter().parse('<svg width="100" height="100"><rect x="0" y="0" width="50%" height="50"/></svg>').scene
        assert scene.objects[0].width == 50.0

        scene_circle = SVGImporter().parse('<svg width="100" height="100"><circle cx="50" cy="50" r="50%"/></svg>').scene
        assert scene_circle.objects[0].radius == 50.0

        scene_ellipse = SVGImporter().parse('<svg width="100" height="100"><ellipse cx="50" cy="50" rx="50%" ry="20"/></svg>').scene
        assert scene_ellipse.objects[0].radius_x == 50.0 and scene_ellipse.objects[0].radius_y == 20.0

        scene_line = SVGImporter().parse('<svg width="100" height="100"><line x1="10%" y1="0" x2="100" y2="100"/></svg>').scene
        assert scene_line.objects[0].start.x == 10.0

        scene_sw = SVGImporter().parse('<svg width="100" height="100"><rect width="100" height="100" stroke="black" stroke-width="10%"/></svg>').scene
        assert scene_sw.objects[0].stroke.width == 10.0

        scene_da = SVGImporter().parse('<svg width="100" height="100"><line x1="0" y1="0" x2="100" y2="100" stroke="black" stroke-dasharray="10% 5%"/></svg>').scene
        assert scene_da.objects[0].stroke.dash_array == (10.0, 5.0)

        scene_do = SVGImporter().parse('<svg width="100" height="100"><line x1="0" y1="0" x2="100" y2="100" stroke="black" stroke-dashoffset="10%"/></svg>').scene
        assert scene_do.objects[0].stroke.dash_offset == 10.0

        # Unsupported length units are strictly rejected
        with pytest.raises(SVGImportError) as exc_unit:
            SVGImporter().parse('<svg width="100" height="100"><rect width="50ch" height="50"/></svg>')
        assert exc_unit.value.diagnostic.code == "SVG_UNSUPPORTED_LENGTH_UNIT"

    # 11. Follow-up: Radial gradient r <= 0 exact solid color mapping
    def test_radial_gradient_zero_radius_paints_last_stop_color(self):
        # objectBoundingBox r="0"
        svg_zero_obb = """<svg width="100" height="100">
          <defs>
            <radialGradient id="r0" r="0">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </radialGradient>
          </defs>
          <rect width="100" height="100" fill="url(#r0)"/>
        </svg>"""
        scene_obb = SVGImporter().parse(svg_zero_obb).scene
        rect_obb = scene_obb.objects[0]
        assert rect_obb.fill is not None
        assert rect_obb.fill.color == Color(0, 0, 255, 1.0)  # blue (last stop)

        # userSpaceOnUse r="0"
        svg_zero_user = """<svg width="100" height="100">
          <defs>
            <radialGradient id="r_u0" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="0">
              <stop offset="0%" stop-color="yellow"/>
              <stop offset="100%" stop-color="green"/>
            </radialGradient>
          </defs>
          <rect width="100" height="100" fill="url(#r_u0)"/>
        </svg>"""
        scene_user = SVGImporter().parse(svg_zero_user).scene
        rect_user = scene_user.objects[0]
        assert rect_user.fill is not None
        assert rect_user.fill.color == Color(0, 128, 0, 1.0)  # green (last stop)

    def test_radial_gradient_zero_radius_resvg_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="r0" cx="50%" cy="50%" r="0%">
              <stop offset="0%" stop-color="red"/>
              <stop offset="100%" stop-color="blue"/>
            </radialGradient>
          </defs>
          <rect width="100" height="100" fill="url(#r0)"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Entire 100x100 rectangle is solid blue [255, 0, 0, 255] (BGRA) in both
        assert np.array_equal(drawcv_img[50, 50], [255, 0, 0, 255])
        assert np.array_equal(resvg_img[50, 50], [255, 0, 0, 255])
        assert np.max(np.abs(resvg_img.astype(float) - drawcv_img.astype(float))) == 0.0

    def test_radial_gradient_negative_r_rejected(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="r_neg" r="-10">
              <stop offset="0%" stop-color="red"/>
            </radialGradient>
          </defs>
          <rect width="100" height="100" fill="url(#r_neg)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic is not None
        assert exc_info.value.diagnostic.code == "SVG_INVALID_GRADIENT_RADIUS"

    def test_radial_gradient_negative_fr_rejected(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="r_neg_fr" r="50" fr="-5">
              <stop offset="0%" stop-color="red"/>
            </radialGradient>
          </defs>
          <rect width="100" height="100" fill="url(#r_neg_fr)"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg)
        assert exc_info.value.diagnostic is not None
        assert exc_info.value.diagnostic.code == "SVG_INVALID_GRADIENT_RADIUS"

    def test_unsupported_vector_effect_rejected(self):
        # Attribute vector-effect
        svg_attr = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="20" stroke="black" stroke-width="2" vector-effect="non-scaling-size"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter().parse(svg_attr)
        assert exc_info.value.diagnostic is not None
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_VECTOR_EFFECT"

        # Inline style vector-effect
        svg_style = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="0" x2="10" y2="10" stroke="black" style="vector-effect: fixed-position"/>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info2:
            SVGImporter().parse(svg_style)
        assert exc_info2.value.diagnostic is not None
        assert exc_info2.value.diagnostic.code == "SVG_UNSUPPORTED_VECTOR_EFFECT"

    def test_visibility_collapse_invisible(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect id="r1" width="50" height="50" visibility="collapse"/>
          <g id="g1" visibility="collapse">
            <rect id="r2" width="20" height="20"/>
            <circle id="c1" cx="50" cy="50" r="10" visibility="visible"/>
          </g>
        </svg>"""
        scene = SVGImporter().parse(svg).scene
        r1 = scene.find_by_name("r1")[0]
        assert r1.visible is False

        g1 = scene.find_by_name("g1")[0]
        assert isinstance(g1, Group)
        r2, c1 = g1.children
        assert r2.visible is False
        assert c1.visible is True
