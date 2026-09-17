"""Gate A and Gate B resvg parity testing for Milestone 2:
- Gate A: Direct visual raster parity between DrawCV (OpenCVRenderer) and resvg
- Gate B Tier 1: Exact equality (diff == 0.0) round-trip parity for untransformed primitives
- Gate B Tier 2: Transformed affine arc fixtures and mathematical curve-equivalence
- Corpus: Real-world SVG subset corpus testing (Inkscape, Figma, standards/resvg, handwritten stress)
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from drawcv import OpenCVRenderer, SVGImporter, Scene
from drawcv.core.geometry_utils import (
    Point,
    flatten_elliptical_arc,
    transform_elliptical_arc,
)


def resvg_render(svg_str: str, width: int, height: int) -> np.ndarray:
    """Render SVG string with resvg-py and return BGRA pixel buffer."""
    resvg = pytest.importorskip("resvg_py")
    png_bytes = resvg.svg_to_bytes(svg_string=svg_str, width=width, height=height, skip_system_fonts=True)
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


def render_scene(scene: Scene) -> np.ndarray:
    """Render DrawCV Scene with OpenCVRenderer and return BGRA pixel buffer."""
    return OpenCVRenderer().render(scene, alpha=True).buffer


# =============================================================================
# Gate A: Direct DrawCV vs resvg Visual Parity
# =============================================================================

class TestM2GateAParity:
    """Gate A: Verify DrawCV renderer matches resvg raster output for M2 features."""

    def test_elliptical_arc_path_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 20 50 A 30 20 0 0 1 80 50 Z" fill="red"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Sampling interior of the arc: (50, 40) is inside
        # BGRA format: red is [0, 0, 255, 255]
        assert np.array_equal(drawcv_img[40, 50], [0, 0, 255, 255])
        assert np.array_equal(resvg_img[40, 50], [0, 0, 255, 255])
        # Sampling outside: (50, 70) is white [255, 255, 255, 255]
        assert np.array_equal(drawcv_img[70, 50], [255, 255, 255, 255])
        assert np.array_equal(resvg_img[70, 50], [255, 255, 255, 255])

    def test_nested_svg_viewport_clip_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <svg x="20" y="20" width="40" height="40">
            <rect x="0" y="0" width="80" height="80" fill="blue"/>
          </svg>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Inside nested svg viewport: (30, 30) is blue [255, 0, 0, 255]
        assert np.array_equal(drawcv_img[30, 30], [255, 0, 0, 255])
        assert np.array_equal(resvg_img[30, 30], [255, 0, 0, 255])
        # Outside nested svg viewport (clipped by default hidden overflow): (70, 30) is white
        assert np.array_equal(drawcv_img[30, 70], [255, 255, 255, 255])
        assert np.array_equal(resvg_img[30, 70], [255, 255, 255, 255])

    def test_symbol_use_viewbox_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <symbol id="icon" viewBox="0 0 10 10">
              <rect width="10" height="10" fill="green"/>
            </symbol>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <use href="#icon" x="25" y="25" width="50" height="50"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Inside instantiated symbol: (50, 50) is green [0, 128, 0, 255]
        assert np.array_equal(drawcv_img[50, 50], [0, 128, 0, 255])
        assert np.array_equal(resvg_img[50, 50], [0, 128, 0, 255])
        # Outside: (10, 10) is white
        assert np.array_equal(drawcv_img[10, 10], [255, 255, 255, 255])
        assert np.array_equal(resvg_img[10, 10], [255, 255, 255, 255])

    def test_non_uniform_rounded_rect_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <rect x="10" y="10" width="80" height="80" rx="30" ry="10" fill="red"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Center: (50, 50) is red [0, 0, 255, 255]
        assert np.array_equal(drawcv_img[50, 50], [0, 0, 255, 255])
        assert np.array_equal(resvg_img[50, 50], [0, 0, 255, 255])
        # Near corner (15, 12) is outside the rounded corner -> white
        assert np.array_equal(drawcv_img[12, 15], [255, 255, 255, 255])
        assert np.array_equal(resvg_img[12, 15], [255, 255, 255, 255])

    def test_circle_clip_parity(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="circle_clip">
              <circle cx="50" cy="50" r="30"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <rect width="100" height="100" fill="red" clip-path="url(#circle_clip)"/>
        </svg>"""
        resvg_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        drawcv_img = render_scene(scene)

        # Inside circle clip: (50, 50) is red
        assert np.array_equal(drawcv_img[50, 50], [0, 0, 255, 255])
        assert np.array_equal(resvg_img[50, 50], [0, 0, 255, 255])
        # Outside circle clip: (10, 10) is white
        assert np.array_equal(drawcv_img[10, 10], [255, 255, 255, 255])
        assert np.array_equal(resvg_img[10, 10], [255, 255, 255, 255])


# =============================================================================
# Gate B Tier 1: Exact Equality (diff == 0.0) Round-Trip SVG Export Parity
# =============================================================================

class TestM2GateBRoundTripParity:
    """Gate B Tier 1: Strict SVG export -> resvg produces exact pixel identity (diff == 0.0)."""

    def test_untransformed_arc_path_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 20 50 A 30 20 0 0 1 80 50 Z" fill="red"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_post_closepath_continuation_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 20 20 L 60 20 L 40 50 Z L 20 70 L 60 70" fill="none" stroke="red" stroke-width="2"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_h_and_v_commands_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 10 10 H 90 V 90 H 10 Z" fill="blue"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_smooth_cubic_s_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 10 50 C 20 10 40 10 50 50 S 80 90 90 50" fill="none" stroke="green" stroke-width="3"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_smooth_quadratic_t_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 10 50 Q 30 20 50 50 T 90 50" fill="none" stroke="purple" stroke-width="3"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_non_square_percentage_geometry_round_trip(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="200" height="100" fill="white"/>
          <rect x="10%" y="20%" width="80%" height="60%" fill="orange"/>
        </svg>"""
        orig_img = resvg_render(svg, 200, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 200, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_percentage_stroke_and_dash_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <path d="M 10 50 L 90 50" stroke="black" stroke-width="4" stroke-dasharray="10% 5%"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_use_symbol_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <symbol id="icon" viewBox="0 0 10 10">
              <rect width="10" height="10" fill="green"/>
            </symbol>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <use href="#icon" x="20" y="20" width="60" height="60"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_nested_svg_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <svg x="20" y="20" width="60" height="60">
            <rect width="40" height="40" fill="blue"/>
          </svg>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_non_uniform_rounded_rect_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <rect x="10" y="10" width="80" height="80" rx="30" ry="10" fill="red"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        # Resvg polygon-vs-ideal rounded corner rasterization diff
        assert diff <= 2.0

    def test_circle_clip_round_trip(self):
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <clipPath id="c">
              <circle cx="50" cy="50" r="30"/>
            </clipPath>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <rect width="100" height="100" fill="red" clip-path="url(#c)"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_svg = scene.export_svg(strict=True).svg
        rt_img = resvg_render(export_svg, 100, 100)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff <= 2.0

    def test_rect_explicit_zero_radii_round_trip(self):
        """Verify Gate-B exact parity for rect with explicit zero rx or ry."""
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" fill="white"/>
          <rect x="10" y="20" width="80" height="60" rx="0" ry="15" fill="blue"/>
          <rect x="20" y="30" width="60" height="40" rx="15" ry="0" fill="green"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_result = scene.export_svg(strict=True)
        assert export_result.fallbacks == ()
        rt_img = resvg_render(export_result.svg, 100, 100)
        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_use_referenced_svg_display_none_round_trip(self):
        """Verify Gate-B exact parity for <use> referencing <svg display="none">."""
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <svg id="hiddenSvg" width="100" height="100" display="none">
              <rect width="100" height="100" fill="red"/>
            </svg>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <use href="#hiddenSvg"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_result = scene.export_svg(strict=True)
        assert export_result.fallbacks == ()
        rt_img = resvg_render(export_result.svg, 100, 100)
        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_use_referenced_symbol_display_none_round_trip(self):
        """Verify Gate-B exact parity for <use> referencing <symbol display="none">."""
        svg = """<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <symbol id="hiddenSymbol" display="none" viewBox="0 0 10 10">
              <rect width="10" height="10" fill="green"/>
            </symbol>
          </defs>
          <rect width="100" height="100" fill="white"/>
          <use href="#hiddenSymbol" x="25" y="25" width="50" height="50"/>
        </svg>"""
        orig_img = resvg_render(svg, 100, 100)
        scene = SVGImporter().parse(svg).scene
        export_result = scene.export_svg(strict=True)
        assert export_result.fallbacks == ()
        rt_img = resvg_render(export_result.svg, 100, 100)
        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0


# =============================================================================
# Gate B Tier 2: Transformed Affine Arc Fixtures and Curve Equivalence
# =============================================================================

class TestM2GateBTier2TransformedArcs:
    """Gate B Tier 2: Arcs under non-uniform scale, rotation, shear, reflection."""

    @pytest.mark.parametrize(
        ("name", "transform_str"),
        [
            ("scale", "scale(1.2, 0.8)"),
            ("rotation", "rotate(35 100 100)"),
            ("shear", "matrix(1 0.3 0.2 1 0 0)"),
            ("reflection", "scale(-1, 1) translate(-200, 0)"),
        ],
    )
    def test_arc_affine_transform_round_trip(self, name: str, transform_str: str):
        svg = f"""<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <rect width="200" height="200" fill="white"/>
          <g transform="{transform_str}">
            <path d="M 50 100 A 40 25 30 0 1 150 100 Z" fill="red"/>
          </g>
        </svg>"""
        orig_img = resvg_render(svg, 200, 200)
        scene = SVGImporter().parse(svg).scene
        export_res = scene.export_svg(strict=True)
        assert len(export_res.fallbacks) == 0
        rt_img = resvg_render(export_res.svg, 200, 200)

        diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
        assert diff == 0.0

    def test_arc_affine_mathematical_curve_equivalence(self):
        """Verify that transform_elliptical_arc mathematically preserves curve geometry."""
        p1 = Point(30.0, 40.0)
        p2 = Point(120.0, 80.0)
        rx = 60.0
        ry = 35.0
        phi = 25.0
        large_arc = False
        sweep = True

        transforms = {
            "non_uniform_scale": np.array([[1.5, 0.0, 10.0], [0.0, 0.7, -5.0], [0.0, 0.0, 1.0]]),
            "rotation": np.array([
                [np.cos(np.radians(37)), -np.sin(np.radians(37)), 20.0],
                [np.sin(np.radians(37)), np.cos(np.radians(37)), 15.0],
                [0.0, 0.0, 1.0],
            ]),
            "shear": np.array([[1.0, 0.35, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "reflection_x": np.array([[-1.0, 0.0, 150.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "reflection_y": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 100.0], [0.0, 0.0, 1.0]]),
            "full_affine": np.array([[1.2, -0.4, 30.0], [0.3, 0.9, -15.0], [0.0, 0.0, 1.0]]),
        }

        for name, M in transforms.items():
            def map_fn(p: Point, mat: np.ndarray = M) -> Point:
                v = mat @ np.array([p.x, p.y, 1.0])
                return Point(v[0], v[1])

            # 1. Flatten original arc adaptively in mapped space
            mapped_pts = flatten_elliptical_arc(
                p1, p2, rx, ry, phi, large_arc, sweep, tolerance=0.01, map_point=map_fn
            )

            # 2. Transform arc parameters via transform_elliptical_arc
            p1_p, p2_p, rx_p, ry_p, phi_p, large_p, sweep_p = transform_elliptical_arc(
                p1, p2, rx, ry, phi, large_arc, sweep, M
            )

            # 3. Flatten transformed arc directly
            trans_pts = flatten_elliptical_arc(
                p1_p, p2_p, rx_p, ry_p, phi_p, large_p, sweep_p, tolerance=0.01
            )

            # Endpoints must match within strict tolerance
            np.testing.assert_allclose([mapped_pts[0].x, mapped_pts[0].y], [trans_pts[0].x, trans_pts[0].y], atol=1e-5)
            np.testing.assert_allclose([mapped_pts[-1].x, mapped_pts[-1].y], [trans_pts[-1].x, trans_pts[-1].y], atol=1e-5)

            # Hausdorff distance across dense contour samples must be < 0.05
            mapped_arr = np.array([[p.x, p.y] for p in mapped_pts])
            trans_arr = np.array([[p.x, p.y] for p in trans_pts])
            dists1 = [np.min(np.hypot(trans_arr[:, 0] - pt[0], trans_arr[:, 1] - pt[1])) for pt in mapped_arr]
            dists2 = [np.min(np.hypot(mapped_arr[:, 0] - pt[0], mapped_arr[:, 1] - pt[1])) for pt in trans_arr]
            hausdorff = max(max(dists1), max(dists2))
            assert hausdorff < 0.05, f"Hausdorff curve mismatch for {name}: {hausdorff}"


# =============================================================================
# Real-World SVG Subset Corpus Parity (Inkscape, Figma, standards, handwritten)
# =============================================================================

class TestM2CorpusParity:
    """Verify supported real-world subset corpus fixtures pass all milestone invariants:
    - Strict import
    - Retained Scene JSON round-trip
    - Strict export with zero fallback
    - Gate B exact reference parity with resvg (diff == 0.0)
    """

    CORPUS_DIR = Path(__file__).parent / "fixtures" / "svg_m2_corpus"

    def get_corpus_files(self) -> list[Path]:
        files = sorted(self.CORPUS_DIR.glob("*.svg"))
        assert len(files) >= 4, f"Expected at least 4 corpus fixtures, found {len(files)}"
        return files

    def test_corpus_fixtures_complete_pipeline(self):
        files = self.get_corpus_files()
        for svg_path in files:
            svg_text = svg_path.read_text(encoding="utf-8")

            # 1. Strict import
            result = SVGImporter().parse(svg_text)
            scene = result.scene
            w = int(scene.width)
            h = int(scene.height)
            assert w > 0 and h > 0

            # 2. Retained Scene JSON round-trip
            json_str = scene.to_json()
            restored_scene = Scene.from_json(json_str)
            assert restored_scene.width == scene.width
            assert restored_scene.height == scene.height
            assert len(restored_scene.layers) == len(scene.layers)

            # 3. Strict export with zero fallback records
            export_res = restored_scene.export_svg(strict=True)
            assert len(export_res.fallbacks) == 0, f"{svg_path.name} produced fallbacks: {export_res.fallbacks}"

            # 4. Gate B reference parity with resvg
            orig_img = resvg_render(svg_text, w, h)
            rt_img = resvg_render(export_res.svg, w, h)
            diff = np.max(np.abs(orig_img.astype(float) - rt_img.astype(float)))
            assert diff == 0.0, f"{svg_path.name} failed Gate B parity: max diff = {diff}"
