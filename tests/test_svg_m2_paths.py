"""Phase 1A Tests: Retained EllipticalArcTo, SVD affine arc export, analytical bounds,
adaptive world flattening, exact progressive slicing, schema migration, and boolean rejection.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest

from drawcv.core.color import Color
from drawcv.core.enums import FillRule, PathBooleanOp
from drawcv.core.exceptions import PathBooleanError, RenderError, UnsupportedVersionError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import (
    elliptical_arc_extrema_bounds,
    elliptical_arc_length,
    elliptical_arc_split,
    flatten_elliptical_arc,
    svg_arc_to_center_parameterization,
    transform_elliptical_arc,
)
from drawcv.core.path_boolean import drawcv_to_pathops
from drawcv.core.path_processing import slice_path
from drawcv.core.transform import Transform
from drawcv.effects.clipping import ClipPath
from drawcv.scene import Scene
from drawcv.serialization.registry import CURRENT_SCHEMA_VERSION, SchemaMigrator
from drawcv.shapes.path import (
    Close,
    CubicTo,
    EllipticalArcTo,
    LineTo,
    MoveTo,
    Path,
    QuadraticTo,
    Subpath,
    deserialize_subpaths,
    serialize_subpaths,
)
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle
from drawcv.svg import SVGExporter


# =============================================================================
# 1. EllipticalArcTo Invariants & Construction
# =============================================================================

def test_elliptical_arc_to_valid_construction():
    cmd = EllipticalArcTo(
        radius_x=25.0,
        radius_y=15.0,
        x_axis_rotation=45.0,
        large_arc=True,
        sweep=False,
        end=Point(100.0, 50.0),
    )
    assert cmd.radius_x == 25.0
    assert cmd.radius_y == 15.0
    assert cmd.x_axis_rotation == 45.0
    assert cmd.large_arc is True
    assert cmd.sweep is False
    assert cmd.end == Point(100.0, 50.0)


def test_elliptical_arc_to_strictly_positive_radii():
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(radius_x=0.0, radius_y=10.0, x_axis_rotation=0.0, large_arc=False, sweep=False, end=Point(10, 10))
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(radius_x=-5.0, radius_y=10.0, x_axis_rotation=0.0, large_arc=False, sweep=False, end=Point(10, 10))
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(radius_x=10.0, radius_y=0.0, x_axis_rotation=0.0, large_arc=False, sweep=False, end=Point(10, 10))
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(radius_x=10.0, radius_y=-1.0, x_axis_rotation=0.0, large_arc=False, sweep=False, end=Point(10, 10))


def test_elliptical_arc_to_validation_flags_and_point():
    with pytest.raises(ValidationError):
        EllipticalArcTo(radius_x=10.0, radius_y=10.0, x_axis_rotation=0.0, large_arc="true", sweep=False, end=Point(10, 10))  # type: ignore
    with pytest.raises(ValidationError):
        EllipticalArcTo(radius_x=10.0, radius_y=10.0, x_axis_rotation=float("nan"), large_arc=False, sweep=False, end=Point(10, 10))
    with pytest.raises(ValidationError):
        EllipticalArcTo(radius_x=10.0, radius_y=10.0, x_axis_rotation=0.0, large_arc=False, sweep=False, end=(10, 10))  # type: ignore


# =============================================================================
# 2. Serialization & Deserialization (Schema 1.8 Canonical Discriminator)
# =============================================================================

def test_elliptical_arc_to_serialization_canonical_discriminator():
    sp = Subpath(
        commands=[
            MoveTo(Point(0.0, 0.0)),
            EllipticalArcTo(25.0, 15.0, 30.0, True, False, Point(50.0, 20.0)),
        ],
        closed=False,
    )
    serialized = serialize_subpaths([sp])
    assert len(serialized) == 1
    cmds = serialized[0]["commands"]
    assert len(cmds) == 2
    arc_data = cmds[1]
    assert arc_data["type"] == "elliptical_arc_to"
    assert arc_data["radius_x"] == 25.0
    assert arc_data["radius_y"] == 15.0
    assert arc_data["x_axis_rotation"] == 30.0
    assert arc_data["large_arc"] is True
    assert arc_data["sweep"] is False
    assert arc_data["end"] == {"x": 50.0, "y": 20.0}

    # Deserialization round-trip
    deserialized = deserialize_subpaths(serialized)
    assert len(deserialized) == 1
    cmd = deserialized[0].commands[1]
    assert isinstance(cmd, EllipticalArcTo)
    assert cmd.radius_x == 25.0
    assert cmd.radius_y == 15.0
    assert cmd.x_axis_rotation == 30.0
    assert cmd.large_arc is True
    assert cmd.sweep is False
    assert cmd.end == Point(50.0, 20.0)


def test_elliptical_arc_to_rejects_non_canonical_discriminators():
    for bad_type in ["arc_to", "ellipticalArcTo", "arc", "EllipticalArcTo"]:
        bad_serialized = [
            {
                "closed": False,
                "commands": [
                    {"type": "move_to", "point": {"x": 0.0, "y": 0.0}},
                    {
                        "type": bad_type,
                        "radius_x": 10.0,
                        "radius_y": 10.0,
                        "x_axis_rotation": 0.0,
                        "large_arc": False,
                        "sweep": False,
                        "end": {"x": 20.0, "y": 0.0},
                    },
                ],
            }
        ]
        with pytest.raises(ValidationError):
            deserialize_subpaths(bad_serialized)


def test_schema_1_7_to_1_8_migration():
    doc_1_7 = {
        "format": "drawcv",
        "version": "1.7",
        "scene": {
            "width": 100,
            "height": 100,
            "layers": [],
        },
    }
    migrated = SchemaMigrator.migrate(doc_1_7, target_version="1.8")
    assert migrated["version"] == "1.8"


# =============================================================================
# 3. Path Fluent API & World Flattening
# =============================================================================

def test_path_fluent_arc_to():
    p = Path().move_to(10, 20).arc_to(25, 25, 0, False, True, 60, 20)
    assert len(p.subpaths) == 1
    assert len(p.subpaths[0].commands) == 2
    cmd = p.subpaths[0].commands[1]
    assert isinstance(cmd, EllipticalArcTo)
    assert cmd.radius_x == 25
    assert cmd.radius_y == 25
    assert cmd.end == Point(60, 20)


def test_path_flatten_world_elliptical_arc():
    p = Path().move_to(0, 0).arc_to(50, 50, 0, False, True, 100, 0)
    pts = p.flatten_world(tolerance=0.25)
    assert len(pts) == 1
    contour = pts[0]
    assert len(contour) > 10  # adaptively subdivided
    assert contour[0] == Point(0, 0)
    assert math.isclose(contour[-1].x, 100.0, abs_tol=1e-5)
    assert math.isclose(contour[-1].y, 0.0, abs_tol=1e-5)

    # For a circular arc of radius 50 centered at (50, 0) in lower half:
    # All points should satisfy (x - 50)^2 + y^2 approx 50^2
    for pt in contour:
        dist_from_center = math.hypot(pt.x - 50.0, pt.y)
        assert math.isclose(dist_from_center, 50.0, abs_tol=0.5)


# =============================================================================
# 4. Analytical Extrema Bounds
# =============================================================================

def test_analytical_extrema_bounds_accuracy():
    # Semicircle from (0, 0) to (100, 0), radius 50, sweep=1
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)
    rx, ry = 50.0, 50.0
    x, y, w, h = elliptical_arc_extrema_bounds(p1, p2, rx, ry, 0.0, False, True)
    # The bottom point of sweep=1 is at y=50
    assert math.isclose(x, 0.0, abs_tol=1e-5)
    assert math.isclose(y, -50.0, abs_tol=1e-5)
    assert math.isclose(w, 100.0, abs_tol=1e-5)
    assert math.isclose(h, 50.0, abs_tol=1e-5)

    # Same arc with Path.get_geometry_bounds()
    p = Path().move_to(0, 0).arc_to(50, 50, 0, False, True, 100, 0)
    bbox = p.get_geometry_bounds()
    assert math.isclose(bbox.x, 0.0, abs_tol=1e-5)
    assert math.isclose(bbox.y, -50.0, abs_tol=1e-5)
    assert math.isclose(bbox.width, 100.0, abs_tol=1e-5)
    assert math.isclose(bbox.height, 50.0, abs_tol=1e-5)


# =============================================================================
# 5. Exact Progressive Slicing (`slice_path`)
# =============================================================================

def test_slice_path_retains_elliptical_arc_to():
    p = Path().move_to(0, 0).arc_to(50, 50, 0, False, True, 100, 0)
    # Slicing at 50% should produce an EllipticalArcTo to (50, 50)
    sliced = slice_path(p, 0.5)
    assert len(sliced.subpaths) == 1
    cmds = sliced.subpaths[0].commands
    assert len(cmds) == 2
    assert isinstance(cmds[0], MoveTo)
    assert isinstance(cmds[1], EllipticalArcTo)
    # Endpoint should be at bottom of circle: (50, -50)
    assert math.isclose(cmds[1].end.x, 50.0, abs_tol=1.0)
    assert math.isclose(cmds[1].end.y, -50.0, abs_tol=1.0)
    # Radii should be preserved exactly
    assert cmds[1].radius_x == 50.0
    assert cmds[1].radius_y == 50.0


def test_slice_path_boundaries():
    p = Path().move_to(0, 0).arc_to(50, 50, 0, False, True, 100, 0)
    empty = slice_path(p, 0.0)
    assert len(empty.subpaths[0].commands) == 1  # only MoveTo

    full = slice_path(p, 1.0)
    assert full is p


# =============================================================================
# 6. PathOps Boolean Rejection
# =============================================================================

def test_path_boolean_rejects_elliptical_arc_to():
    p1 = Path().move_to(0, 0).arc_to(50, 50, 0, False, True, 100, 0).close()
    p2 = Path().move_to(0, 0).line_to(100, 0).line_to(100, 100).close()
    with pytest.raises(PathBooleanError, match="EllipticalArcTo"):
        p1.boolean(p2, PathBooleanOp.UNION)
    with pytest.raises(PathBooleanError, match="EllipticalArcTo"):
        drawcv_to_pathops(p1)


# =============================================================================
# 7. SVD Affine Arc Transformation & Strict SVG Export
# =============================================================================

def test_transform_elliptical_arc_rigid():
    # Translation + 90 deg rotation
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)
    # 90 deg rotation: (x, y) -> (-y, x)
    M = np.array([
        [0.0, -1.0, 10.0],
        [1.0, 0.0, 20.0],
        [0.0, 0.0, 1.0],
    ])
    p1_w, p2_w, rx_w, ry_w, phi_w, large_w, sweep_w = transform_elliptical_arc(
        p1, p2, 50.0, 30.0, 0.0, False, True, M
    )
    assert math.isclose(p1_w.x, 10.0, abs_tol=1e-5)
    assert math.isclose(p1_w.y, 20.0, abs_tol=1e-5)
    assert math.isclose(p2_w.x, 10.0, abs_tol=1e-5)
    assert math.isclose(p2_w.y, 120.0, abs_tol=1e-5)
    assert math.isclose(rx_w, 50.0, abs_tol=1e-4)
    assert math.isclose(ry_w, 30.0, abs_tol=1e-4)
    assert math.isclose(phi_w, 90.0, abs_tol=1e-4)
    assert large_w is False
    assert sweep_w is True  # det(M) = +1 -> sweep preserved


def test_transform_elliptical_arc_reflection():
    # Reflection across Y axis: x -> -x
    M = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)
    p1_w, p2_w, rx_w, ry_w, phi_w, large_w, sweep_w = transform_elliptical_arc(
        p1, p2, 50.0, 30.0, 0.0, False, True, M
    )
    # det(M) = -1 -> sweep inverted
    assert sweep_w is False


def test_transform_elliptical_arc_singular_rejection():
    # Singular matrix (rank 1)
    M_singular = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValidationError, match="Singular affine"):
        transform_elliptical_arc(Point(0, 0), Point(10, 10), 5, 5, 0, False, False, M_singular)


from unittest.mock import patch

def test_svg_export_native_arc_command():
    scene = Scene(200, 200)
    p = Path(stroke=StrokeStyle(color=Color(0, 0, 0), width=2.0))
    p.move_to(20, 20).arc_to(30, 15, 45, True, False, 120, 80)
    scene.add(p)

    svg_str = SVGExporter(strict=True).render(scene).svg
    assert "<path" in svg_str
    # Must contain native 'A' command
    assert " A " in svg_str
    assert "120" in svg_str
    assert "80" in svg_str


def test_svg_export_singular_arc_raises_render_error():
    scene = Scene(200, 200)
    p = Path(stroke=StrokeStyle(color=Color(0, 0, 0), width=2.0))
    p.move_to(20, 20).arc_to(30, 15, 0, False, False, 100, 20)
    scene.add(p)

    # Singular matrix (collapse x axis)
    singular_m = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with patch.object(Path, "world_matrix", new_callable=lambda: singular_m):
        with pytest.raises(RenderError, match="singular"):
            SVGExporter(strict=True).render(scene)


# =============================================================================
# 8. Phase 1B: SVG Path Parser (H, V, S, T, A) Tests
# =============================================================================

from drawcv.svg_import import SVGPathParser, SVGImportError


def test_path_parser_horizontal_vertical():
    # M 10 20 H 50 V 60 h -10 v -20
    p = SVGPathParser.parse("M 10 20 H 50 V 60 h -10 v -20")
    cmds = p.subpaths[0].commands
    assert len(cmds) == 5
    assert isinstance(cmds[0], MoveTo) and cmds[0].point == Point(10, 20)
    assert isinstance(cmds[1], LineTo) and cmds[1].point == Point(50, 20)
    assert isinstance(cmds[2], LineTo) and cmds[2].point == Point(50, 60)
    assert isinstance(cmds[3], LineTo) and cmds[3].point == Point(40, 60)
    assert isinstance(cmds[4], LineTo) and cmds[4].point == Point(40, 40)


def test_path_parser_repeated_coordinates():
    # H with multiple x values, V with multiple y values
    p = SVGPathParser.parse("M 0 0 H 10 20 30 V 40 50")
    cmds = p.subpaths[0].commands
    assert len(cmds) == 6
    assert cmds[1].point == Point(10, 0)
    assert cmds[2].point == Point(20, 0)
    assert cmds[3].point == Point(30, 0)
    assert cmds[4].point == Point(30, 40)
    assert cmds[5].point == Point(30, 50)


def test_path_parser_smooth_cubic_reflection():
    # C 10 20 30 40 50 50 S 70 80 90 90
    # For C, last control point c2 = (30, 40), end = (50, 50)
    # For S, reflected control point c1 = 2 * (50, 50) - (30, 40) = (70, 60)
    p = SVGPathParser.parse("M 0 0 C 10 20 30 40 50 50 S 70 80 90 90")
    cmds = p.subpaths[0].commands
    assert len(cmds) == 3
    s_cmd = cmds[2]
    assert isinstance(s_cmd, CubicTo)
    assert s_cmd.control1 == Point(70, 60)
    assert s_cmd.control2 == Point(70, 80)
    assert s_cmd.end == Point(90, 90)

    # S without preceding C/S has c1 = current_point
    p2 = SVGPathParser.parse("M 10 10 S 30 40 50 50")
    s_cmd2 = p2.subpaths[0].commands[1]
    assert s_cmd2.control1 == Point(10, 10)


def test_path_parser_smooth_quadratic_reflection():
    # Q 20 40 50 50 T 90 90
    # For Q, ctrl = (20, 40), end = (50, 50)
    # For T, reflected ctrl = 2 * (50, 50) - (20, 40) = (80, 60)
    p = SVGPathParser.parse("M 0 0 Q 20 40 50 50 T 90 90")
    cmds = p.subpaths[0].commands
    assert len(cmds) == 3
    t_cmd = cmds[2]
    assert isinstance(t_cmd, QuadraticTo)
    assert t_cmd.control == Point(80, 60)
    assert t_cmd.end == Point(90, 90)

    # T without preceding Q/T has ctrl = current_point
    p2 = SVGPathParser.parse("M 10 10 T 50 50")
    t_cmd2 = p2.subpaths[0].commands[1]
    assert t_cmd2.control == Point(10, 10)


def test_path_parser_arc_compact_syntax():
    # Compact flags without spaces: A25 25 0 0150 100
    p = SVGPathParser.parse("M 0 0 A25 25 0 0150 100")
    cmds = p.subpaths[0].commands
    assert len(cmds) == 2
    arc = cmds[1]
    assert isinstance(arc, EllipticalArcTo)
    assert arc.radius_x == 25.0
    assert arc.radius_y == 25.0
    assert arc.x_axis_rotation == 0.0
    assert arc.large_arc is False
    assert arc.sweep is True
    assert arc.end == Point(50, 100)

    # Compact negative coordinate after flag: A25 25 0 10-50 100
    p2 = SVGPathParser.parse("M 0 0 A25 25 0 10-50 100")
    arc2 = p2.subpaths[0].commands[1]
    assert arc2.large_arc is True
    assert arc2.sweep is False
    assert arc2.end == Point(-50, 100)


def test_path_parser_arc_canonicalization():
    # Negative radii dropped to positive
    p = SVGPathParser.parse("M 0 0 A -30 -40 0 0 1 50 50")
    arc = p.subpaths[0].commands[1]
    assert arc.radius_x == 30.0
    assert arc.radius_y == 40.0

    # Zero radius degenerates to LineTo
    p_zero = SVGPathParser.parse("M 0 0 A 0 20 0 0 1 50 50")
    assert isinstance(p_zero.subpaths[0].commands[1], LineTo)
    assert p_zero.subpaths[0].commands[1].point == Point(50, 50)

    # Coincident endpoints: omitted
    p_coincident = SVGPathParser.parse("M 50 50 A 25 25 0 0 1 50 50")
    assert len(p_coincident.subpaths[0].commands) == 1  # only MoveTo


def test_path_parser_errors():
    # Bad flag
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 0 0 A 25 25 0 2 1 50 50")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # Incomplete coordinates
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 0 0 H")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # Unknown command
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 0 0 X 10 10")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"


# =============================================================================
# 9. Defect Review Regression Tests (Items A, B, C, D, H, I)
# =============================================================================

def test_elliptical_arc_length_starting_angle_dependency():
    """Item A: Verify elliptical arc length depends on starting angle for non-circular ellipses."""
    # Ellipse with rx=50, ry=20 (a=50, b=20)
    # Arc 1: theta1 = 0 -> theta1 + dtheta = pi/2 (span pi/2, from (50, 0) to (0, 20))
    p1_a = Point(50.0, 0.0)
    p2_a = Point(0.0, 20.0)
    L1 = elliptical_arc_length(p1_a, p2_a, 50.0, 20.0, 0.0, large_arc=False, sweep=True)

    # Arc 2: span pi/2 centered at top vertex (theta1 = pi/4 -> 3*pi/4)
    # x1 = 50*cos(pi/4), y1 = 20*sin(pi/4)
    # x2 = 50*cos(3*pi/4), y2 = 20*sin(3*pi/4)
    p1_b = Point(50.0 * math.cos(math.pi / 4.0), 20.0 * math.sin(math.pi / 4.0))
    p2_b = Point(50.0 * math.cos(3.0 * math.pi / 4.0), 20.0 * math.sin(3.0 * math.pi / 4.0))
    L2 = elliptical_arc_length(p1_b, p2_b, 50.0, 20.0, 0.0, large_arc=False, sweep=True)

    # Both arcs have identical rx=50, ry=20 and identical span=pi/2
    # But their lengths MUST differ because arc length depends on theta1
    assert not math.isclose(L1, L2, rel_tol=0.05)

    # Validate against reference numerical integration using 64-point Gauss-Legendre
    nodes, weights = np.polynomial.legendre.leggauss(64)
    span = math.pi / 2.0
    t_vals = 0.5 * span * (nodes + 1.0)
    w_vals = 0.5 * span * weights

    # Ref 1: theta in [0, pi/2]
    ang1 = t_vals
    int1 = np.sqrt((50.0 * np.sin(ang1)) ** 2 + (20.0 * np.cos(ang1)) ** 2)
    ref_L1 = float(np.sum(int1 * w_vals))
    assert math.isclose(L1, ref_L1, rel_tol=1e-5)

    # Ref 2: theta in [pi/4, 3*pi/4]
    ang2 = (math.pi / 4.0) + t_vals
    int2 = np.sqrt((50.0 * np.sin(ang2)) ** 2 + (20.0 * np.cos(ang2)) ** 2)
    ref_L2 = float(np.sum(int2 * w_vals))
    assert math.isclose(L2, ref_L2, rel_tol=1e-5)


def test_elliptical_arc_split_with_undersized_radii_corrected():
    """Item A: Sliced arc with undersized radii undergoing SVG radius correction lies on the exact corrected ellipse."""
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)
    # Radii 20 and 10 are too small for chord distance 100
    # Correction scales them by factor Lambda = 50 / 20 = 2.5 -> eff_rx=50, eff_ry=25
    cut_pt, s_rx, s_ry, s_phi, s_large, s_sweep = elliptical_arc_split(
        p1, p2, 20.0, 10.0, 0.0, large_arc=False, sweep=True, fraction=0.5
    )

    # Must return corrected radii, not original 20 and 10
    assert math.isclose(s_rx, 50.0, abs_tol=1e-4)
    assert math.isclose(s_ry, 25.0, abs_tol=1e-4)
    # Cut point at 50% must be at the bottom vertex of the corrected ellipse: (50, -25)
    assert math.isclose(cut_pt.x, 50.0, abs_tol=1e-3)
    assert math.isclose(cut_pt.y, -25.0, abs_tol=1e-3)

    # Center parameterization of sliced arc must match center of corrected ellipse
    c_slice, sl_rx, sl_ry, _, _, _ = svg_arc_to_center_parameterization(
        p1, cut_pt, s_rx, s_ry, s_phi, s_large, s_sweep
    )
    c_orig, orig_rx, orig_ry, _, _, _ = svg_arc_to_center_parameterization(
        p1, p2, 50.0, 25.0, 0.0, False, True
    )
    assert math.isclose(c_slice.x, c_orig.x, abs_tol=1e-3)
    assert math.isclose(c_slice.y, c_orig.y, abs_tol=1e-3)
    assert math.isclose(sl_rx, orig_rx, abs_tol=1e-3)
    assert math.isclose(sl_ry, orig_ry, abs_tol=1e-3)


def test_schema_1_8_forward_and_backward_policies():
    """Item B: Verify Schema 1.8 forward migration, roundtrip, downgrade rejection, and unknown schema policy."""
    # 1. Old 1.7 document loads/migrates to 1.8
    doc_1_7 = {
        "format": "drawcv",
        "version": "1.7",
        "scene": {
            "width": 100,
            "height": 100,
            "layers": [],
        },
    }
    migrated = SchemaMigrator.migrate(doc_1_7, target_version="1.8")
    assert migrated["version"] == "1.8"

    # 2. New arc document serializes as CURRENT_SCHEMA_VERSION (1.9)
    scene = Scene(200, 200)
    p = Path().move_to(0, 0).arc_to(30, 20, 0, False, True, 50, 50)
    scene.add(p)
    doc_1_9 = scene.to_dict()
    assert doc_1_9["version"] == CURRENT_SCHEMA_VERSION
    assert doc_1_9["schema_version"] == CURRENT_SCHEMA_VERSION

    # 3. JSON round-trips normally
    json_str = json.dumps(doc_1_9)
    restored = Scene.from_dict(json.loads(json_str))
    assert restored.to_dict()["version"] == CURRENT_SCHEMA_VERSION
    assert isinstance(restored.objects[0].subpaths[0].commands[1], EllipticalArcTo)

    # 4. Do not pretend a newer document is an older version (downgrade rejection)
    doc_1_8 = SchemaMigrator.migrate(doc_1_7, target_version="1.8")
    with pytest.raises(UnsupportedVersionError):
        SchemaMigrator.migrate(doc_1_8, target_version="1.7")
    with pytest.raises(UnsupportedVersionError):
        SchemaMigrator.migrate(doc_1_9, target_version="1.8")

    # 5. Unknown/future schema rejected according to existing policy
    doc_future = {
        "format": "drawcv",
        "version": "99.0",
        "scene": {"width": 100, "height": 100},
    }
    with pytest.raises(UnsupportedVersionError):
        SchemaMigrator.migrate(doc_future)


def test_flatten_elliptical_arc_mapped_space_adaptive():
    """Item C: World-space adaptive subdivision under scale, non-uniform scale, rotation, and shear."""
    # Arc from (0, 0) to (100, 0), rx=50, ry=30
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)

    # 1. Large scale (50x magnification)
    M_large = np.array([
        [50.0, 0.0, 100.0],
        [0.0, 50.0, 200.0],
        [0.0, 0.0, 1.0],
    ])
    map_fn_large = lambda p: Point(M_large[0, 0] * p.x + M_large[0, 2], M_large[1, 1] * p.y + M_large[1, 2])
    pts_large = flatten_elliptical_arc(p1, p2, 50.0, 30.0, 0.0, False, True, tolerance=0.5, map_point=map_fn_large)
    # Scaled up 50x requires more segments to satisfy 0.5px world tolerance
    assert len(pts_large) > 30
    assert math.isclose(pts_large[0].x, 100.0, abs_tol=1e-5)
    assert math.isclose(pts_large[-1].x, 5100.0, abs_tol=1e-5)

    # 2. Non-uniform scale (scale_x=10.0, scale_y=0.1)
    M_nonuniform = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])
    map_fn_nu = lambda p: Point(10.0 * p.x, 0.1 * p.y)
    pts_nu = flatten_elliptical_arc(p1, p2, 50.0, 30.0, 0.0, False, True, tolerance=0.5, map_point=map_fn_nu)
    assert len(pts_nu) >= 4
    assert math.isclose(pts_nu[0].x, 0.0, abs_tol=1e-5)
    assert math.isclose(pts_nu[-1].x, 1000.0, abs_tol=1e-5)

    # 3. Rotation (45 deg) and Shear
    theta = math.radians(45.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Affine matrix with rotation + shear: [cos, -sin, 0] @ [1, 0.5, 0; 0, 1, 0]
    A_shear = np.array([
        [cos_t, -sin_t + 0.5 * cos_t, 50.0],
        [sin_t, cos_t + 0.5 * sin_t, 50.0],
        [0.0, 0.0, 1.0],
    ])
    map_fn_shear = lambda p: Point(A_shear[0, 0] * p.x + A_shear[0, 1] * p.y + A_shear[0, 2],
                                   A_shear[1, 0] * p.x + A_shear[1, 1] * p.y + A_shear[1, 2])
    pts_shear = flatten_elliptical_arc(p1, p2, 50.0, 30.0, 0.0, False, True, tolerance=0.5, map_point=map_fn_shear)
    assert len(pts_shear) >= 8

    # Verify segment budget prevents unbounded recursion
    assert len(pts_large) <= 4096
    assert len(pts_shear) <= 4096


def test_elliptical_arc_to_strict_validation():
    """Item D: Strict EllipticalArcTo parameter typing and deserialization validation."""
    # Reject NaN and Inf
    with pytest.raises(ValidationError):
        EllipticalArcTo(float("nan"), 10.0, 0.0, False, True, Point(10, 10))
    with pytest.raises(ValidationError):
        EllipticalArcTo(10.0, float("inf"), 0.0, False, True, Point(10, 10))
    with pytest.raises(ValidationError):
        EllipticalArcTo(10.0, 10.0, float("nan"), False, True, Point(10, 10))

    # Reject non-positive radii
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(0.0, 10.0, 0.0, False, True, Point(10, 10))
    with pytest.raises(ValidationError, match="strictly positive"):
        EllipticalArcTo(10.0, -1.0, 0.0, False, True, Point(10, 10))

    # Reject non-bool flags (no coercion)
    with pytest.raises(ValidationError, match="actual boolean"):
        EllipticalArcTo(10.0, 10.0, 0.0, 1, True, Point(10, 10))  # type: ignore
    with pytest.raises(ValidationError, match="actual boolean"):
        EllipticalArcTo(10.0, 10.0, 0.0, False, 0, Point(10, 10))  # type: ignore

    # Deserialization rejects non-bool flags
    bad_doc = [
        {
            "closed": False,
            "commands": [
                {
                    "type": "elliptical_arc_to",
                    "radius_x": 10.0,
                    "radius_y": 10.0,
                    "x_axis_rotation": 0.0,
                    "large_arc": 1,
                    "sweep": True,
                    "end": {"x": 10.0, "y": 10.0},
                }
            ],
        }
    ]
    with pytest.raises(ValidationError, match="actual boolean"):
        deserialize_subpaths(bad_doc)

    # Deserialization rejects missing required fields
    incomplete_doc = [
        {
            "closed": False,
            "commands": [
                {
                    "type": "elliptical_arc_to",
                    "radius_x": 10.0,
                    # missing radius_y
                    "x_axis_rotation": 0.0,
                    "large_arc": False,
                    "sweep": True,
                    "end": {"x": 10.0, "y": 10.0},
                }
            ],
        }
    ]
    with pytest.raises(ValidationError, match="Missing required field"):
        deserialize_subpaths(incomplete_doc)


def test_path_numeric_grammar_positive():
    """Item H: Positive lexer fixtures for trailing decimals and scientific notation."""
    # 1. Trailing decimals: 1., -2.
    p1 = SVGPathParser.parse("M 1. 2. L -3. 4.")
    assert p1.subpaths[0].commands[0].point == Point(1.0, 2.0)
    assert p1.subpaths[0].commands[1].point == Point(-3.0, 4.0)

    # 2. Trailing decimal with exponent: 1.e2, -2.e-1
    p2 = SVGPathParser.parse("M 1.e2 2.e-1")
    assert p2.subpaths[0].commands[0].point == Point(100.0, 0.2)

    # 3. Concatenated decimal forms without whitespace: 1.2.3 -> (1.2, 0.3)
    p3 = SVGPathParser.parse("M 1.2.3")
    assert math.isclose(p3.subpaths[0].commands[0].point.x, 1.2)
    assert math.isclose(p3.subpaths[0].commands[0].point.y, 0.3)

    # 4. Sign-separated coordinate concatenation: 10-20 -> (10.0, -20.0)
    p4 = SVGPathParser.parse("M 10-20")
    assert p4.subpaths[0].commands[0].point == Point(10.0, -20.0)

    # 5. Negative trailing decimals: -2. -3.
    p5 = SVGPathParser.parse("M -2. -3.")
    assert p5.subpaths[0].commands[0].point == Point(-2.0, -3.0)


def test_path_numeric_grammar_negative():
    """Item H: Negative lexer fixtures for malformed commas and numbers."""
    # 1. Comma immediately following command
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M,10 10")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M, 10 10")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10 10 L,20 20")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10 10 Z,")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10 10 Z ,")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # 2. Consecutive commas
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10,,20")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10, ,20")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # 3. Trailing comma at end of path
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10 10,")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # 4. Comma before command
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M 10 10, L 20 20")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"

    # 5. Lone decimal point
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("M . 10")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"


def test_transform_elliptical_arc_no_silent_radii_clamp():
    """Item I: Non-singular transform preserves exact derived singular values without artificial clamping."""
    # Scale by 1e-4 (small but representable and well-conditioned non-singular transform)
    M_small = np.array([
        [1e-4, 0.0, 0.0],
        [0.0, 1e-4, 0.0],
        [0.0, 0.0, 1.0],
    ])
    _, _, rx_w, ry_w, _, _, _ = transform_elliptical_arc(
        Point(0, 0), Point(10, 0), 50.0, 30.0, 0.0, False, True, M_small
    )
    # rx_w must be 50 * 1e-4 = 0.005, not clamped to 1e-9
    assert math.isclose(rx_w, 0.005, rel_tol=1e-5)
    assert math.isclose(ry_w, 0.003, rel_tol=1e-5)


def test_svg_closepath_continuation_semantics():
    """Item 1: SVG closepath continuation semantics when non-M commands follow Z."""
    from drawcv.shapes.path import Close, LineTo, MoveTo, QuadraticTo, CubicTo

    # 1. M0 0 L10 0 Z L20 20
    # First subpath: M 0 0, L 10 0, Close (closed=True)
    # Second subpath: starts at initial point of previous subpath (0, 0), then L 20 20
    p1 = SVGPathParser.parse("M 0 0 L 10 0 Z L 20 20")
    assert len(p1.subpaths) == 2
    sp1, sp2 = p1.subpaths[0], p1.subpaths[1]
    assert sp1.closed is True
    assert len(sp1.commands) == 3
    assert sp1.commands[0] == MoveTo(Point(0.0, 0.0))
    assert sp1.commands[1] == LineTo(Point(10.0, 0.0))
    assert isinstance(sp1.commands[2], Close)

    assert sp2.closed is False
    assert len(sp2.commands) == 2
    assert sp2.commands[0] == MoveTo(Point(0.0, 0.0))
    assert sp2.commands[1] == LineTo(Point(20.0, 20.0))

    # 2. M0 0 L10 0 Z l20 20 (relative line)
    # Starts at (0, 0), l 20 20 moves to (0+20, 0+20) = (20, 20)
    p2 = SVGPathParser.parse("M 0 0 L 10 0 Z l 20 20")
    assert len(p2.subpaths) == 2
    assert p2.subpaths[1].commands[0] == MoveTo(Point(0.0, 0.0))
    assert p2.subpaths[1].commands[1] == LineTo(Point(20.0, 20.0))

    # 3. M0 0 Q... Z C...
    p3 = SVGPathParser.parse("M 0 0 Q 5 10 10 0 Z C 15 -10 20 -10 25 0")
    assert len(p3.subpaths) == 2
    assert p3.subpaths[0].closed is True
    assert p3.subpaths[1].commands[0] == MoveTo(Point(0.0, 0.0))
    assert isinstance(p3.subpaths[1].commands[1], CubicTo)
    assert p3.subpaths[1].commands[1].end == Point(25.0, 0.0)

    # 4. M0 0 A... Z A...
    p4 = SVGPathParser.parse("M 0 0 A 10 10 0 0 1 20 0 Z A 15 15 0 0 0 30 0")
    assert len(p4.subpaths) == 2
    assert p4.subpaths[0].closed is True
    assert p4.subpaths[1].commands[0] == MoveTo(Point(0.0, 0.0))
    assert isinstance(p4.subpaths[1].commands[1], EllipticalArcTo)
    assert p4.subpaths[1].commands[1].end == Point(30.0, 0.0)

    # 5. Z followed by M: normal new subpath at new coordinates
    p5 = SVGPathParser.parse("M 0 0 L 10 0 Z M 50 50 L 60 60")
    assert len(p5.subpaths) == 2
    assert p5.subpaths[0].closed is True
    assert p5.subpaths[1].commands[0] == MoveTo(Point(50.0, 50.0))
    assert p5.subpaths[1].commands[1] == LineTo(Point(60.0, 60.0))

    # 6. Non-M without any prior subpath still raises SVGImportError
    with pytest.raises(SVGImportError) as exc:
        SVGPathParser.parse("L 10 10")
    assert exc.value.diagnostic.code == "SVG_MALFORMED_PATH"


def test_path_fluent_arc_to_strict_validation():
    """Item 4: Path.arc_to preserves strict validation without silent type coercion."""
    p = Path()

    # Boolean flags must be actual bools
    with pytest.raises(ValidationError, match="actual boolean"):
        p.arc_to(10.0, 10.0, 0.0, "false", True, 20.0, 20.0)  # type: ignore

    with pytest.raises(ValidationError, match="actual boolean"):
        p.arc_to(10.0, 10.0, 0.0, False, 1, 20.0, 20.0)  # type: ignore

    # Radii must not be boolean
    with pytest.raises(ValidationError, match="must be numeric"):
        p.arc_to(True, 10.0, 0.0, False, True, 20.0, 20.0)  # type: ignore

    with pytest.raises(ValidationError, match="must be numeric"):
        p.arc_to(10.0, False, 0.0, False, True, 20.0, 20.0)  # type: ignore

    with pytest.raises(ValidationError, match="must be numeric"):
        p.arc_to(10.0, 10.0, True, False, True, 20.0, 20.0)  # type: ignore

    # x_axis_rotation is normalized to [0, 360)
    p_valid = Path()
    p_valid.arc_to(10.0, 10.0, 450.0, False, True, 20.0, 20.0)
    arc_cmd = p_valid.subpaths[0].commands[1]
    assert isinstance(arc_cmd, EllipticalArcTo)
    assert math.isclose(arc_cmd.x_axis_rotation, 90.0)


def test_adaptive_arc_flattening_per_segment_verification():
    """Item 7: Verify every emitted segment satisfies true mapped arc midpoint vs chord midpoint <= tolerance."""
    p1 = Point(0.0, 0.0)
    p2 = Point(100.0, 0.0)
    rx, ry = 50.0, 30.0
    phi_deg = 0.0
    large_arc, sweep = False, True
    tol = 0.5

    center, eff_rx, eff_ry, phi_rad, theta1, dtheta = svg_arc_to_center_parameterization(
        p1, p2, rx, ry, phi_deg, large_arc, sweep
    )
    cos_phi = math.cos(phi_rad)
    sin_phi = math.sin(phi_rad)

    def eval_local(t: float) -> Point:
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = center.x + eff_rx * cos_phi * cos_t - eff_ry * sin_phi * sin_t
        y = center.y + eff_rx * sin_phi * cos_t + eff_ry * cos_phi * sin_t
        return Point(x, y)

    def collect_leaf_segments(t_a: float, t_b: float, map_fn, depth: int = 0) -> list[tuple[float, float, float]]:
        t_m = 0.5 * (t_a + t_b)
        pt_a_m = map_fn(eval_local(t_a))
        pt_b_m = map_fn(eval_local(t_b))
        pt_m_m = map_fn(eval_local(t_m))
        chord_mid_x = 0.5 * (pt_a_m.x + pt_b_m.x)
        chord_mid_y = 0.5 * (pt_a_m.y + pt_b_m.y)
        dist = math.hypot(pt_m_m.x - chord_mid_x, pt_m_m.y - chord_mid_y)
        if dist <= tol or depth >= 12:
            return [(t_a, t_b, dist)]
        return collect_leaf_segments(t_a, t_m, map_fn, depth + 1) + collect_leaf_segments(t_m, t_b, map_fn, depth + 1)

    # 1. Large scale (50x)
    map_large = lambda p: Point(50.0 * p.x + 100.0, 50.0 * p.y + 200.0)
    num_initial = max(2, int(math.ceil(abs(dtheta) / (math.pi / 2.0))))
    for i in range(num_initial):
        t_a = theta1 + (i / num_initial) * dtheta
        t_b = theta1 + ((i + 1) / num_initial) * dtheta
        segments = collect_leaf_segments(t_a, t_b, map_large)
        for _, _, dist in segments:
            assert dist <= tol

    # 2. Non-uniform scale (10x, 0.1x)
    map_nu = lambda p: Point(10.0 * p.x, 0.1 * p.y)
    for i in range(num_initial):
        t_a = theta1 + (i / num_initial) * dtheta
        t_b = theta1 + ((i + 1) / num_initial) * dtheta
        segments = collect_leaf_segments(t_a, t_b, map_nu)
        for _, _, dist in segments:
            assert dist <= tol

    # 3. Rotation (45 deg) + Shear
    theta = math.radians(45.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    A_shear = np.array([
        [cos_t, -sin_t + 0.5 * cos_t, 50.0],
        [sin_t, cos_t + 0.5 * sin_t, 50.0],
        [0.0, 0.0, 1.0],
    ])
    map_shear = lambda p: Point(A_shear[0, 0] * p.x + A_shear[0, 1] * p.y + A_shear[0, 2],
                                A_shear[1, 0] * p.x + A_shear[1, 1] * p.y + A_shear[1, 2])
    for i in range(num_initial):
        t_a = theta1 + (i / num_initial) * dtheta
        t_b = theta1 + ((i + 1) / num_initial) * dtheta
        segments = collect_leaf_segments(t_a, t_b, map_shear)
        for _, _, dist in segments:
            assert dist <= tol
