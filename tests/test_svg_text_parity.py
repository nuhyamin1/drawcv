"""Gate parity testing for Milestone M1 SVG text interchange:
- Verifies deterministic corpus SVGs render in DrawCV (OpenCVRenderer) and resvg
- Verifies strict round-trip native export for all corpus files
- Verifies raster metrics and visual consistency across the full round-trip gate:
  original SVG -> SVGImporter -> retained DrawCV -> SVGExporter(strict=True) -> exported SVG -> resvg
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

pytest.importorskip("uharfbuzz")
pytest.importorskip("freetype")
pytest.importorskip("icu")
pytest.importorskip("regex")

from drawcv import (
    DictFontResolver,
    FontAsset,
    OpenCVRenderer,
    SVGExporter,
    SVGImporter,
    Scene,
    Text,
)

FONTS_DIR = Path(__file__).parent / "assets" / "fonts"
CORPUS_DIR = Path(__file__).parent / "fixtures" / "svg_text_corpus"

CORPUS_FILES = [
    ("01_simple_text.svg", 300, 150),
    ("02_multirun_tspan.svg", 400, 150),
    ("03_arabic_rtl.svg", 400, 150),
    ("04_scalar_offsets.svg", 300, 150),
    ("05_anchors.svg", 300, 200),
    ("06_nested_tspan.svg", 400, 150),
    ("07_fill_opacity.svg", 400, 150),
    ("08_fill_none.svg", 400, 150),
    ("09_bold_italic.svg", 400, 150),
    ("10_arabic_styled_spans.svg", 400, 150),
    ("11_rtl_anchors.svg", 400, 200),
    ("12_transformed_text.svg", 400, 300),
    ("13_tspan_absolute_xy.svg", 400, 200),
]


def get_corpus_resolver() -> DictFontResolver:
    noto_sans = FontAsset.from_file(FONTS_DIR / "notosans.ttf")
    noto_arabic = FontAsset.from_file(FONTS_DIR / "notosansarabic.ttf")
    resolver = DictFontResolver(default_fonts=(noto_sans,))
    resolver.register("Noto Sans", (noto_sans,))
    resolver.register("Noto Sans", (noto_sans,), weight="bold")
    resolver.register("Noto Sans", (noto_sans,), style="italic")
    resolver.register("Noto Sans Arabic", (noto_arabic,))
    resolver.register("Noto Sans Arabic", (noto_arabic,), weight="bold")
    return resolver


def resvg_render(svg_str: str, width: int, height: int) -> np.ndarray:
    resvg = pytest.importorskip("resvg_py")
    png_bytes = resvg.svg_to_bytes(
        svg_string=svg_str,
        width=width,
        height=height,
        skip_system_fonts=True,
        font_dirs=[str(FONTS_DIR.resolve())],
    )
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)


def render_scene(scene: Scene) -> np.ndarray:
    return OpenCVRenderer().render(scene, alpha=True).buffer


class TestSVGTextCorpusParity:
    """Test corpus files against resvg and verify strict round-trip export."""

    @pytest.mark.parametrize("filename,width,height", CORPUS_FILES)
    def test_corpus_strict_round_trip(self, filename: str, width: int, height: int):
        svg_path = CORPUS_DIR / filename
        svg_content = svg_path.read_text(encoding="utf-8")

        resolver = get_corpus_resolver()
        importer = SVGImporter(strict=True, font_resolver=resolver)
        result = importer.parse(svg_content)
        scene = result.scene

        # Strict export must have zero raster fallbacks
        exporter = SVGExporter(strict=True)
        export_res = exporter.render(scene)
        assert len(export_res.fallbacks) == 0
        assert "<text" in export_res.svg

        # Re-import exported SVG to ensure bidirectional fidelity
        re_imported = importer.parse(export_res.svg)
        assert len(re_imported.scene.find_by_type(Text)) == len(scene.find_by_type(Text))

    @pytest.mark.parametrize("filename,width,height", CORPUS_FILES)
    def test_corpus_raster_parity_with_resvg(self, filename: str, width: int, height: int):
        svg_path = CORPUS_DIR / filename
        svg_content = svg_path.read_text(encoding="utf-8")

        resolver = get_corpus_resolver()
        importer = SVGImporter(strict=True, font_resolver=resolver)
        scene = importer.parse(svg_content).scene

        # Render retained DrawCV scene
        drawcv_img = render_scene(scene)

        # Export to native strict SVG and render in resvg
        exporter = SVGExporter(strict=True)
        export_res = exporter.render(scene)
        assert len(export_res.fallbacks) == 0

        resvg_orig = resvg_render(svg_content, width, height)
        resvg_exp = resvg_render(export_res.svg, width, height)

        alpha_orig = resvg_orig[..., 3]
        alpha_exp = resvg_exp[..., 3]
        alpha_d = drawcv_img[..., 3]

        count_orig = np.count_nonzero(alpha_orig)
        count_exp = np.count_nonzero(alpha_exp)
        count_d = np.count_nonzero(alpha_d)

        assert count_orig > 0, f"Original resvg rendered zero text pixels for {filename}"
        assert count_exp > 0, f"Exported resvg rendered zero text pixels for {filename}"
        assert count_d > 0, f"DrawCV rendered zero text pixels for {filename}"

        # ---------------------------------------------------------------------
        # 1. Gate parity: exported SVG rendered by resvg vs original resvg render
        # ---------------------------------------------------------------------
        # Nonzero alpha count must match within 5%
        assert abs(count_exp - count_orig) / count_orig <= 0.05, (
            f"Coverage difference between original and exported resvg for {filename}: "
            f"orig={count_orig}, exp={count_exp}"
        )

        yy_o, xx_o = np.nonzero(alpha_orig)
        yy_e, xx_e = np.nonzero(alpha_exp)

        # Ink bounding boxes must match within 2 pixels
        bbox_o = np.array([xx_o.min(), yy_o.min(), xx_o.max(), yy_o.max()])
        bbox_e = np.array([xx_e.min(), yy_e.min(), xx_e.max(), yy_e.max()])
        assert np.max(np.abs(bbox_o - bbox_e)) <= 2.0, (
            f"Ink bbox mismatch between original and exported resvg for {filename}: "
            f"orig={bbox_o.tolist()}, exp={bbox_e.tolist()}"
        )

        # Centers of mass must match within 1 pixel
        cx_o, cy_o = float(np.mean(xx_o)), float(np.mean(yy_o))
        cx_e, cy_e = float(np.mean(xx_e)), float(np.mean(yy_e))
        assert abs(cx_o - cx_e) <= 1.0, f"X center mismatch between resvg renders for {filename}"
        assert abs(cy_o - cy_e) <= 1.0, f"Y center mismatch between resvg renders for {filename}"

        # Mask IoU between original and exported resvg must be >= 0.95
        mask_o = alpha_orig > 0
        mask_e = alpha_exp > 0
        resvg_iou = np.count_nonzero(mask_o & mask_e) / np.count_nonzero(mask_o | mask_e)
        assert resvg_iou >= 0.95, f"Mask IoU between resvg renders for {filename}: {resvg_iou:.3f} < 0.95"

        # Pixel value checks: Alpha MAE / max error, and BGRA MAE over union ink mask
        union_mask = mask_o | mask_e
        if np.any(union_mask):
            alpha_diff = np.abs(alpha_orig.astype(np.float32) - alpha_exp.astype(np.float32))
            alpha_mae = float(np.mean(alpha_diff[union_mask]))
            alpha_max = float(np.max(alpha_diff[union_mask]))
            assert alpha_mae <= 1.0, f"Alpha MAE between resvg renders for {filename}: {alpha_mae:.3f} > 1.0"
            assert alpha_max <= 5.0, f"Alpha max error between resvg renders for {filename}: {alpha_max:.1f} > 5.0"

            bgra_diff = np.abs(resvg_orig.astype(np.float32) - resvg_exp.astype(np.float32))
            bgra_mae = float(np.mean(bgra_diff[union_mask]))
            assert bgra_mae <= 1.0, f"BGRA MAE between resvg renders for {filename}: {bgra_mae:.3f} > 1.0"

        # ---------------------------------------------------------------------
        # 2. Parity: DrawCV renderer vs original resvg render
        # ---------------------------------------------------------------------
        # Alpha coverage ratio within [0.65, 1.50]
        cov_ratio = count_d / count_orig
        assert 0.65 <= cov_ratio <= 1.50, (
            f"Coverage ratio out of bounds for {filename}: {cov_ratio:.2f}"
        )

        yy_d, xx_d = np.nonzero(alpha_d)
        cx_d, cy_d = float(np.mean(xx_d)), float(np.mean(yy_d))

        # Centers of mass must match within 8 pixels
        assert abs(cx_d - cx_o) <= 8.0, (
            f"X center mismatch for {filename}: drawcv={cx_d:.1f}, resvg={cx_o:.1f}"
        )
        assert abs(cy_d - cy_o) <= 8.0, (
            f"Y center mismatch for {filename}: drawcv={cy_d:.1f}, resvg={cy_o:.1f}"
        )

        # Ink bounding boxes must match within 8 pixels
        bbox_d = np.array([xx_d.min(), yy_d.min(), xx_d.max(), yy_d.max()])
        assert np.max(np.abs(bbox_d - bbox_o)) <= 8.0, (
            f"Ink bbox mismatch between DrawCV and resvg for {filename}: "
            f"drawcv={bbox_d.tolist()}, resvg={bbox_o.tolist()}"
        )

        # Non-empty overlap metric
        mask_d = alpha_d > 0
        overlap_ratio = np.count_nonzero(mask_d & mask_o) / min(count_d, count_orig)
        assert overlap_ratio >= 0.50, f"Low mask overlap for {filename}: {overlap_ratio:.2f}"
