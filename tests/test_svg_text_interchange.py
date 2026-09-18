"""Tests for DrawCV SVG Text Interchange (M1):
- Parsing <text> and <tspan>
- Whitespace normalization across nodes (xml:space="default" vs "preserve")
- Strict diagnostic rejection of unsupported text features
- Permissive font substitution vs strict rejection
- Native SVG text export and strict round-trip
"""

from pathlib import Path
import pytest

from drawcv import (
    Color,
    DictFontResolver,
    FontAsset,
    Group,
    OpenCVRenderer,
    Point,
    RenderError,
    SVGExporter,
    SVGImporter,
    Scene,
    Text,
    TextAnchor,
    TextRun,
    Transform,
    ValidationError,
)
from drawcv.svg_import import SVGImportError
from tests.test_svg_text_parity import resvg_render

FONTS_DIR = Path(__file__).parent / "assets" / "fonts"


def get_test_resolver() -> DictFontResolver:
    noto_sans = FontAsset.from_file(FONTS_DIR / "notosans.ttf")
    noto_arabic = FontAsset.from_file(FONTS_DIR / "notosansarabic.ttf")
    resolver = DictFontResolver(default_fonts=(noto_sans,))
    resolver.register("Noto Sans", (noto_sans,))
    resolver.register("Noto Sans", (noto_sans,), weight="bold")
    resolver.register("Noto Sans", (noto_sans,), style="italic")
    resolver.register("Noto Sans", (noto_sans,), weight="bold", style="italic")
    resolver.register("Noto Sans Arabic", (noto_arabic,))
    resolver.register("Noto Sans Arabic", (noto_arabic,), weight="bold")
    resolver.register("Noto Sans Arabic", (noto_arabic,), style="italic")
    resolver.register("Noto Sans Arabic", (noto_arabic,), weight="bold", style="italic")
    return resolver


class TestSVGTextImport:
    """Test importing SVG <text> and <tspan> elements."""

    def test_import_simple_text(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="25" y="60" font-family="Noto Sans" font-size="24" fill="#ff0000">Hello World</text>
        </svg>"""
        resolver = get_test_resolver()
        result = SVGImporter(font_resolver=resolver).parse(svg)
        scene = result.scene
        texts = [obj for layer in scene.layers for obj in layer.objects if isinstance(obj, Text)]
        assert len(texts) == 1
        t = texts[0]
        assert t.text == "Hello World"
        assert t.position.x == 25.0
        assert t.position.y == 60.0
        assert t.font_size == 24.0
        assert t.font_family_name == "Noto Sans"
        assert t.color == Color(255, 0, 0, 1.0)
        assert t.text_origin == "baseline"
        assert t.text_anchor == TextAnchor.START
        assert t.is_font_substituted is False

    def test_import_multirun_tspan(self):
        svg = """<svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="50" font-family="Noto Sans" font-size="20" fill="black"><tspan fill="red">Red</tspan><tspan fill="blue" dx="5">Blue</tspan></text>
        </svg>"""
        resolver = get_test_resolver()
        result = SVGImporter(font_resolver=resolver).parse(svg)
        scene = result.scene
        t = [obj for layer in scene.layers for obj in layer.objects if isinstance(obj, Text)][0]
        assert t.runs is not None
        assert len(t.runs) == 2
        assert t.runs[0].text == "Red"
        assert t.runs[0].fill == Color(255, 0, 0, 1.0)
        assert t.runs[1].text == "Blue"
        assert t.runs[1].fill == Color(0, 0, 255, 1.0)
        assert t.runs[1].dx == 5.0

    def test_whitespace_normalization_default(self):
        # In default mode: newlines removed, tabs->spaces, collapsed across elements, stripped ends
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30" font-family="Noto Sans">
                Hello
                <tspan> World </tspan>
            </text>
        </svg>"""
        resolver = get_test_resolver()
        t = [obj for layer in SVGImporter(font_resolver=resolver).parse(svg).scene.layers for obj in layer.objects if isinstance(obj, Text)][0]
        assert t.runs is not None
        # "Hello" and "World" should have a single space between them across node boundaries
        combined = "".join(r.text for r in t.runs)
        assert combined == "Hello World"

    def test_whitespace_normalization_preserve(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30" font-family="Noto Sans" xml:space="preserve">  A   B  </text>
        </svg>"""
        resolver = get_test_resolver()
        t = [obj for layer in SVGImporter(font_resolver=resolver).parse(svg).scene.layers for obj in layer.objects if isinstance(obj, Text)][0]
        assert t.text == "  A   B  "


class TestSVGTextRejections:
    """Test strict diagnostic rejection of unsupported text features."""

    def test_reject_textpath(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text><textPath href="#p">Path text</textPath></text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXTPATH"

    def test_reject_stroke(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="20" stroke="black" stroke-width="1">Stroked</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXT_STROKE"

    def test_reject_paint_server_fill(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <defs><linearGradient id="g"/></defs>
            <text x="10" y="20" fill="url(#g)">Gradient</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXT_PAINT_SERVER"

    def test_reject_tspan_opacity(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="20"><tspan opacity="0.5">Half</tspan></text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXT_OPACITY"

    def test_reject_coordinate_list(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10 20 30" y="20">List</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXT_COORDINATE_LIST"

    def test_reject_baseline_properties(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="20" dominant-baseline="middle">Shifted</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_BASELINE_PROPERTY"

    def test_reject_unsupported_css_whitespace(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="20" style="white-space: pre-wrap;">Pre</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_CSS_PROPERTY"

    def test_reject_unresolved_font_in_strict(self):
        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="20" font-family="NonExistentFont">Missing</text>
        </svg>"""
        resolver = get_test_resolver()
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True, font_resolver=resolver).parse(svg)
        assert exc_info.value.diagnostic.code == "SVG_UNRESOLVED_FONT"


class TestSVGTextExport:
    """Test native SVG text export, strict round-trip, and fallback checks."""

    def test_strict_native_export(self):
        resolver = get_test_resolver()
        font = resolver.resolve("Noto Sans").fonts
        t = Text(
            "Native Export",
            position=Point(50, 80),
            color=Color(51, 102, 153),
            font_size=28.0,
            font_family_name="Noto Sans",
            fonts=font,
            text_origin="baseline",
            text_anchor=TextAnchor.MIDDLE,
        )
        scene = Scene(width=300, height=200)
        scene.add(t)

        # Strict export must succeed with 0 fallbacks
        result = SVGExporter(strict=True).render(scene)
        assert len(result.fallbacks) == 0
        assert "<text" in result.svg
        assert 'font-family="Noto Sans"' in result.svg
        assert 'font-size="28"' in result.svg
        assert 'text-anchor="middle"' in result.svg

    def test_strict_round_trip(self):
        svg_in = """<svg width="300" height="150" viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg">
  <text x="30" y="80" font-family="Noto Sans" font-size="32" fill="#ff0000">Hello World</text>
</svg>"""
        resolver = get_test_resolver()
        importer = SVGImporter(strict=True, font_resolver=resolver)
        scene = importer.parse(svg_in).scene

        exporter = SVGExporter(strict=True)
        export_out = exporter.render(scene)
        assert len(export_out.fallbacks) == 0
        assert 'fill="rgb(255,0,0)"' in export_out.svg
        assert "Hello World" in export_out.svg

    def test_substituted_font_strict_rejection(self):
        svg_in = """<svg width="300" height="150" xmlns="http://www.w3.org/2000/svg">
  <text x="30" y="80" font-family="UnknownFont" font-size="32">Substituted</text>
</svg>"""
        resolver = get_test_resolver()
        # Permissive font import allows substitution
        scene = SVGImporter(strict_fonts=False, font_resolver=resolver).parse(svg_in).scene

        # But strict export must reject it with RenderError
        with pytest.raises(RenderError, match="font substituted"):
            SVGExporter(strict=True).render(scene)

        # Non-strict export falls back to raster
        out = SVGExporter(strict=False).render(scene)
        assert len(out.fallbacks) == 1
        assert out.fallbacks[0].reason == "font substituted"

    def test_font_weight_and_style_retained_and_exported(self):
        noto = FontAsset.from_file(FONTS_DIR / "notosans.ttf")
        resolver = DictFontResolver(default_fonts=(noto,))
        resolver.register("Noto Sans", noto, weight="normal", style="normal")
        resolver.register("Noto Sans", noto, weight="bold", style="normal")
        resolver.register("Noto Sans", noto, weight="normal", style="italic")
        resolver.register("Noto Sans", noto, weight="bold", style="italic")

        svg = """<svg width="300" height="150" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-weight="bold" font-style="italic">Styled</text>
        </svg>"""
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.font_weight == "bold"
        assert t.font_style == "italic"

        export_res = SVGExporter(strict=True).render(scene)
        assert 'font-weight="bold"' in export_res.svg
        assert 'font-style="italic"' in export_res.svg

    def test_export_run_direction_override(self):
        noto_sans = FontAsset.from_file(FONTS_DIR / "notosans.ttf")
        noto_arabic = FontAsset.from_file(FONTS_DIR / "notosansarabic.ttf")
        resolver = DictFontResolver(default_fonts=(noto_sans,))
        resolver.register("Noto Sans", noto_sans)
        resolver.register("Noto Sans Arabic", noto_arabic)

        svg = """<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" direction="ltr">LTR <tspan x="100" direction="rtl" font-family="Noto Sans Arabic">سلام</tspan></text>
        </svg>"""
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.direction == "ltr"
        assert t.runs is not None
        rtl_run = [r for r in t.runs if r.direction == "rtl"]
        assert len(rtl_run) == 1

        export_res = SVGExporter(strict=True).render(scene)
        assert 'direction="rtl"' in export_res.svg


class TestSVGTextFidelityAndSemantics:
    """Test detailed semantics: fill-opacity, fill=none, nested tspans, and deferred feature rejections."""

    def test_fill_opacity_not_squared_vs_resvg(self):
        resvg = pytest.importorskip("resvg_py")
        import cv2
        import numpy as np

        svg = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="24" fill="#000000" fill-opacity="0.5">Half</text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.fill_opacity == 0.5

        drawcv_img = OpenCVRenderer().render(scene, alpha=True).buffer
        png_bytes = resvg.svg_to_bytes(
            svg_string=svg, width=200, height=100, skip_system_fonts=True,
            font_dirs=[str(FONTS_DIR.resolve())]
        )
        resvg_img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        # Both DrawCV and resvg must render maximum alpha near 128 (0.5 * 255), not squared 64 (0.25 * 255)
        d_max_alpha = int(drawcv_img[..., 3].max())
        r_max_alpha = int(resvg_img[..., 3].max())
        assert abs(d_max_alpha - 128) <= 2, f"DrawCV squared or corrupted fill opacity: max={d_max_alpha}"
        assert abs(r_max_alpha - 128) <= 2, f"resvg max alpha mismatch: max={r_max_alpha}"
        assert abs(d_max_alpha - r_max_alpha) <= 2

    def test_tspan_fill_opacity_override(self):
        svg = """<svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="24" fill-opacity="0.5">
                Root <tspan fill-opacity="0.8">Tspan</tspan>
            </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.fill_opacity == 0.5
        assert t.runs is not None
        # Root run has None (inherited), tspan run has 0.8
        tspan_run = [r for r in t.runs if r.text == "Tspan"][0]
        assert tspan_run.fill_opacity == 0.8

        layout = t._font_layout()
        assert layout.styled_runs is not None
        override_sruns = [sr for sr in layout.styled_runs if sr.fill_opacity == 0.8]
        assert len(override_sruns) == 1
        root_sruns = [sr for sr in layout.styled_runs if sr.fill_opacity == 0.5]
        assert len(root_sruns) >= 1

    def test_fill_none_semantics(self):
        import numpy as np
        resolver = get_test_resolver()

        # Root fill="none" produces no glyph fill
        svg_none = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="24" fill="none">Invisible</text>
        </svg>"""
        scene_none = SVGImporter(strict=True, font_resolver=resolver).parse(svg_none).scene
        t_none = scene_none.find_by_type(Text)[0]
        assert t_none.fill_none is True
        img_none = OpenCVRenderer().render(scene_none, alpha=True).buffer
        assert np.count_nonzero(img_none[..., 3]) == 0

        # Export writes fill="none"
        exp_none = SVGExporter(strict=True).render(scene_none).svg
        assert 'fill="none"' in exp_none

        # Tspan fill="none" produces no fill for that run
        svg_tspan_none = """<svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="24" fill="red">
                Visible <tspan fill="none">NoFill</tspan> End
            </text>
        </svg>"""
        scene_tspan = SVGImporter(strict=True, font_resolver=resolver).parse(svg_tspan_none).scene
        t_tspan = scene_tspan.find_by_type(Text)[0]
        nofill_run = [r for r in t_tspan.runs if r.text == "NoFill"][0]
        assert nofill_run.fill_none is True

        exp_tspan = SVGExporter(strict=True).render(scene_tspan).svg
        assert '<tspan fill="none">NoFill</tspan>' in exp_tspan

    def test_nested_tspan_mixed_content_recursion(self):
        svg = """<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="60" font-family="Noto Sans" font-size="24" fill="#111111">
                Level0 <tspan fill="blue">Level1 <tspan fill="green">Level2</tspan> Tail1</tspan> Tail0
            </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.runs is not None

        run_texts = [r.text for r in t.runs]
        assert "Level0 " in run_texts
        assert "Level1 " in run_texts
        assert "Level2" in run_texts
        assert " Tail1" in run_texts
        assert " Tail0" in run_texts

        level2_run = [r for r in t.runs if r.text == "Level2"][0]
        assert level2_run.fill == Color(0, 128, 0, 1.0)  # green

        tail1_run = [r for r in t.runs if r.text == " Tail1"][0]
        assert tail1_run.fill == Color(0, 0, 255, 1.0)  # inherited from Level1 (blue)

        # Full round trip
        exp = SVGExporter(strict=True).render(scene)
        assert len(exp.fallbacks) == 0

    def test_reject_deferred_features(self):
        resolver = get_test_resolver()

        deferred_cases = [
            ('<text x="10" y="20" rotate="30">Rot</text>', "SVG_UNSUPPORTED_TEXT_ROTATE"),
            ('<text x="10" y="20"><tspan rotate="30">Rot</tspan></text>', "SVG_UNSUPPORTED_TEXT_ROTATE"),
            ('<text x="10" y="20" textLength="100">Len</text>', "SVG_UNSUPPORTED_TEXT_LENGTH"),
            ('<text x="10" y="20"><tspan textLength="100">Len</tspan></text>', "SVG_UNSUPPORTED_TEXT_LENGTH"),
            ('<text x="10" y="20" lengthAdjust="spacing">Adj</text>', "SVG_UNSUPPORTED_TEXT_LENGTH_ADJUST"),
            ('<text x="10" y="20" letter-spacing="2px">Letter</text>', "SVG_UNSUPPORTED_LETTER_SPACING"),
            ('<text x="10" y="20" word-spacing="4px">Word</text>', "SVG_UNSUPPORTED_WORD_SPACING"),
            ('<text x="10" y="20" text-decoration="underline">Deco</text>', "SVG_UNSUPPORTED_TEXT_DECORATION"),
            ('<text x="10" y="20" writing-mode="vertical-rl">Vert</text>', "SVG_UNSUPPORTED_WRITING_MODE"),
            ('<text x="10" y="20"><tspan style="opacity:0.5">CSS Opacity</tspan></text>', "SVG_UNSUPPORTED_TEXT_OPACITY"),
            ('<text x="10" y="20" alignment-baseline="middle">Base</text>', "SVG_UNSUPPORTED_BASELINE_PROPERTY"),
        ]

        for svg_frag, expected_code in deferred_cases:
            svg = f'<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">{svg_frag}</svg>'
            with pytest.raises(SVGImportError) as exc_info:
                SVGImporter(strict=True, font_resolver=resolver).parse(svg)
            assert exc_info.value.diagnostic.code == expected_code, f"Failed on {svg_frag}"

    def test_deduplicate_embedded_fonts_in_runs(self):
        svg = """<svg width="300" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="24" fill="red">
                Run1 <tspan fill="blue">Run2</tspan> Run3
            </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.runs is not None
        # Inherited runs must store run.fonts = None to avoid multiplying font bytes
        for r in t.runs:
            assert r.fonts is None


class TestStrictFidelityGapsM1:
    """Regression tests for strict-fidelity gaps:
    1. Inherited nested tspan paint (fill, fill="none", fill-opacity, font-size, font-family, font-weight, font-style, direction)
    2. Nested tspan positioning propagation to first addressable descendant
    3. xml:space="preserve" strict export & CSS white-space rejection
    4. Run Color alpha export & non-inheritance of root alpha
    5. Per-run direction chunking and invalid direction rejection
    6. Font-size zero/negative/overlarge rejection
    7. Nested-group transform, clip, and opacity integration
    """

    def test_inherited_nested_tspan_paint(self):
        svg = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20" font-weight="normal" font-style="normal" fill="red" fill-opacity="1.0" direction="ltr">
                <tspan fill="blue" fill-opacity="0.4" font-size="28" font-weight="bold" font-style="italic">
                    <tspan>InnerBlue</tspan>
                </tspan>
                <tspan fill="none">
                    <tspan>InnerNone</tspan>
                </tspan>
            </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.color == Color(255, 0, 0, 1.0)
        assert t.font_size == 20.0
        assert t.fill_opacity == 1.0
        assert t.runs is not None

        blue_run = [r for r in t.runs if r.text == "InnerBlue"][0]
        assert blue_run.fill == Color(0, 0, 255, 1.0)
        assert blue_run.fill_opacity == 0.4
        assert blue_run.font_size == 28.0
        assert str(blue_run.font_weight) in ("bold", "700")
        assert blue_run.font_style == "italic"

        none_run = [r for r in t.runs if r.text == "InnerNone"][0]
        assert none_run.fill_none is True

        # Export must emit the inherited overrides on tspans
        exp = SVGExporter(strict=True).render(scene)
        assert len(exp.fallbacks) == 0
        assert 'fill="rgb(0,0,255)"' in exp.svg
        assert 'fill-opacity="0.4"' in exp.svg
        assert 'font-size="28"' in exp.svg
        assert 'font-weight="bold"' in exp.svg
        assert 'font-style="italic"' in exp.svg
        assert 'fill="none"' in exp.svg

    def test_nested_tspan_positioning_propagation(self):
        svg = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="40" font-family="Noto Sans" font-size="20">
                <tspan x="150">
                    <tspan>Nested</tspan>
                </tspan>
            </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.runs is not None
        nested_run = [r for r in t.runs if r.text == "Nested"][0]
        assert nested_run.x == 150.0

        # Also test outer dx/dy propagation when inner does not specify them
        svg_outer_dx = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="40" font-family="Noto Sans" font-size="20">
                <tspan y="80" dx="10" dy="5">
                    <tspan>OuterShifted</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_outer_dx = SVGImporter(strict=True, font_resolver=resolver).parse(svg_outer_dx).scene
        t_outer_dx = scene_outer_dx.find_by_type(Text)[0]
        outer_shifted_run = [r for r in t_outer_dx.runs if r.text == "OuterShifted"][0]
        assert outer_shifted_run.y == 80.0
        assert outer_shifted_run.dx == 10.0
        assert outer_shifted_run.dy == 5.0

        # When inner specifies dx/dy, it overrides outer pending dx/dy
        svg_offsets = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="40" font-family="Noto Sans" font-size="20">
                <tspan y="80" dx="10" dy="5">
                    <tspan dx="15" dy="25">Shifted</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_offsets = SVGImporter(strict=True, font_resolver=resolver).parse(svg_offsets).scene
        t_offsets = scene_offsets.find_by_type(Text)[0]
        shifted_run = [r for r in t_offsets.runs if r.text == "Shifted"][0]
        assert shifted_run.y == 80.0
        assert shifted_run.dx == 15.0
        assert shifted_run.dy == 25.0

        # Parity test vs resvg
        exp_offsets = SVGExporter(strict=True).render(scene_offsets).svg
        orig_img = resvg_render(svg_offsets, 400, 200)
        exp_img = resvg_render(exp_offsets, 400, 200)
        import numpy as np
        mask = (orig_img[..., 3] > 0) | (exp_img[..., 3] > 0)
        assert np.any(mask)
        alpha_mae = float(np.mean(np.abs(orig_img[..., 3].astype(float) - exp_img[..., 3].astype(float))[mask]))
        assert alpha_mae <= 1.0

    def test_xml_space_preserve_and_css_rejection(self):
        svg = """<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20" xml:space="preserve">  Leading   Middle <tspan>  Inner   </tspan> Trailing  </text>
        </svg>"""
        resolver = get_test_resolver()
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.xml_space == "preserve"
        assert t.runs is not None
        all_text = "".join(r.text for r in t.runs)
        assert "  Leading   Middle " in all_text
        assert "  Inner   " in all_text
        assert " Trailing  " in all_text

        exp = SVGExporter(strict=True).render(scene)
        assert len(exp.fallbacks) == 0
        assert 'xml:space="preserve"' in exp.svg

        # Strict resvg parity
        orig_img = resvg_render(svg, 400, 150)
        exp_img = resvg_render(exp.svg, 400, 150)
        import numpy as np
        mask = (orig_img[..., 3] > 0) | (exp_img[..., 3] > 0)
        assert np.any(mask)
        alpha_mae = float(np.mean(np.abs(orig_img[..., 3].astype(float) - exp_img[..., 3].astype(float))[mask]))
        assert alpha_mae <= 1.0

        # Reject CSS white-space: pre
        svg_pre = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="30" style="white-space: pre">Preformatted</text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True, font_resolver=resolver).parse(svg_pre)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_CSS_PROPERTY"

    def test_run_alpha_export_fidelity(self):
        resolver = get_test_resolver()
        test_fonts = resolver.resolve("Noto Sans").fonts

        # Case 1: Run with Color alpha
        text_obj = Text(
            position=Point(20, 50),
            text_origin="baseline",
            fonts=test_fonts,
            font_size=24,
            font_family_name="Noto Sans",
            color=Color.black(),
            fill_opacity=1.0,
            runs=(
                TextRun("Half", fill=Color(255, 0, 0, 0.5), fill_opacity=None),
            ),
        )
        scene = Scene(width=200, height=100)
        scene.add(text_obj)
        exp_svg = SVGExporter(strict=True).render(scene).svg
        assert 'fill="rgb(255,0,0)"' in exp_svg
        assert 'fill-opacity="0.5"' in exp_svg

        # Case 2: Root with alpha 0.5, child with opaque red fill
        text_obj2 = Text(
            position=Point(20, 50),
            text_origin="baseline",
            fonts=test_fonts,
            font_size=24,
            font_family_name="Noto Sans",
            color=Color(0, 0, 0, 0.5),
            fill_opacity=1.0,
            runs=(
                TextRun("Opaque", fill=Color(255, 0, 0, 1.0), fill_opacity=None),
            ),
        )
        scene2 = Scene(width=200, height=100)
        scene2.add(text_obj2)
        exp_svg2 = SVGExporter(strict=True).render(scene2).svg
        assert 'fill-opacity="0.5"' in exp_svg2
        assert '<tspan fill="rgb(255,0,0)" fill-opacity="1">Opaque</tspan>' in exp_svg2

        # Verify resvg renders child at full opacity
        resvg_img = resvg_render(exp_svg2, 200, 100)
        max_alpha = int(resvg_img[..., 3].max())
        assert max_alpha >= 250

    def test_font_size_strict_validation(self):
        resolver = get_test_resolver()
        # Negative length
        with pytest.raises(SVGImportError) as exc1:
            SVGImporter(strict=True, font_resolver=resolver).parse(
                '<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg"><text font-size="-10px">Neg</text></svg>'
            )
        assert exc1.value.diagnostic.code == "SVG_MALFORMED_LENGTH"

        # Zero font size
        with pytest.raises(SVGImportError) as exc2:
            SVGImporter(strict=True, font_resolver=resolver).parse(
                '<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg"><text font-size="0">Zero</text></svg>'
            )
        assert exc2.value.diagnostic.code == "SVG_UNSUPPORTED_FONT_SIZE"

        # Overlarge font size
        with pytest.raises(SVGImportError) as exc3:
            SVGImporter(strict=True, font_resolver=resolver).parse(
                '<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg"><text font-size="5000">Big</text></svg>'
            )
        assert exc3.value.diagnostic.code == "SVG_UNSUPPORTED_FONT_SIZE"

    def test_direction_chunk_and_resvg_parity(self):
        resolver = get_test_resolver()
        # Per-tspan direction without chunk boundary must be rejected
        svg_nochunk = """<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50"><tspan direction="rtl">NoChunk</tspan></text>
        </svg>"""
        with pytest.raises(SVGImportError) as exc_info:
            SVGImporter(strict=True, font_resolver=resolver).parse(svg_nochunk)
        assert exc_info.value.diagnostic.code == "SVG_UNSUPPORTED_TEXT_DIRECTION"

        # Mixed direction with chunk boundary
        svg_mixed = """<svg width="400" height="150" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20" direction="ltr">English <tspan x="150" direction="rtl" font-family="Noto Sans Arabic">سلام</tspan></text>
        </svg>"""
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg_mixed).scene
        t = scene.find_by_type(Text)[0]
        assert t.runs is not None
        assert any(r.direction == "rtl" for r in t.runs)

        exp = SVGExporter(strict=True).render(scene)
        assert len(exp.fallbacks) == 0
        assert 'direction="rtl"' in exp.svg

        orig_img = resvg_render(svg_mixed, 400, 150)
        exp_img = resvg_render(exp.svg, 400, 150)
        import numpy as np
        mask = (orig_img[..., 3] > 0) | (exp_img[..., 3] > 0)
        assert np.any(mask)
        alpha_mae = float(np.mean(np.abs(orig_img[..., 3].astype(float) - exp_img[..., 3].astype(float))[mask]))
        assert alpha_mae <= 1.0

        # DrawCV renderer vs resvg overlap
        drawcv_img = OpenCVRenderer().render(scene, alpha=True).buffer
        d_mask = drawcv_img[..., 3] > 0
        o_mask = orig_img[..., 3] > 0
        assert np.count_nonzero(d_mask & o_mask) > 0

    def test_group_transform_clip_opacity_integration(self):
        resolver = get_test_resolver()
        import numpy as np

        # 1. Group transform -> Text transform
        svg_group_tf = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <g transform="translate(30, 20) rotate(15)">
                <text x="10" y="30" font-family="Noto Sans" font-size="24" transform="translate(5, 5)">Transformed</text>
            </g>
        </svg>"""
        scene_tf = SVGImporter(strict=True, font_resolver=resolver).parse(svg_group_tf).scene
        exp_tf = SVGExporter(strict=True).render(scene_tf).svg
        orig_img_tf = resvg_render(svg_group_tf, 400, 200)
        exp_img_tf = resvg_render(exp_tf, 400, 200)
        mask_tf = (orig_img_tf[..., 3] > 0) | (exp_img_tf[..., 3] > 0)
        assert np.any(mask_tf)
        alpha_mae_tf = float(np.mean(np.abs(orig_img_tf[..., 3].astype(float) - exp_img_tf[..., 3].astype(float))[mask_tf]))
        assert alpha_mae_tf <= 1.0

        # 2. Text + ClipPath
        svg_clip = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <clipPath id="cp">
                    <rect x="0" y="0" width="80" height="80"/>
                </clipPath>
            </defs>
            <text x="20" y="50" font-family="Noto Sans" font-size="28" clip-path="url(#cp)">ClippedText</text>
        </svg>"""
        scene_clip = SVGImporter(strict=True, font_resolver=resolver).parse(svg_clip).scene
        exp_clip = SVGExporter(strict=True).render(scene_clip).svg
        orig_img_clip = resvg_render(svg_clip, 400, 200)
        exp_img_clip = resvg_render(exp_clip, 400, 200)
        mask_clip = (orig_img_clip[..., 3] > 0) | (exp_img_clip[..., 3] > 0)
        assert np.any(mask_clip)
        alpha_mae_clip = float(np.mean(np.abs(orig_img_clip[..., 3].astype(float) - exp_img_clip[..., 3].astype(float))[mask_clip]))
        assert alpha_mae_clip <= 1.0

        # 3. Text + drawable opacity
        svg_opa = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="28" opacity="0.4">OpacityText</text>
        </svg>"""
        scene_opa = SVGImporter(strict=True, font_resolver=resolver).parse(svg_opa).scene
        exp_opa = SVGExporter(strict=True).render(scene_opa).svg
        orig_img_opa = resvg_render(svg_opa, 400, 200)
        exp_img_opa = resvg_render(exp_opa, 400, 200)
        mask_opa = (orig_img_opa[..., 3] > 0) | (exp_img_opa[..., 3] > 0)
        assert np.any(mask_opa)
        alpha_mae_opa = float(np.mean(np.abs(orig_img_opa[..., 3].astype(float) - exp_img_opa[..., 3].astype(float))[mask_opa]))
        assert alpha_mae_opa <= 1.0
        assert abs(int(orig_img_opa[..., 3].max()) - 102) <= 2

    def test_nested_tspan_explicit_zero_vs_omitted_dx_dy(self):
        resolver = get_test_resolver()
        import numpy as np

        # Outer nonzero dx/dy + inner explicit zero dx/dy
        svg_zero = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                <tspan dx="30" dy="15">
                    <tspan dx="0" dy="0">InnerZero</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_zero = SVGImporter(strict=True, font_resolver=resolver).parse(svg_zero).scene
        t_zero = scene_zero.find_by_type(Text)[0]
        assert t_zero.runs is not None
        run_zero = [r for r in t_zero.runs if "InnerZero" in r.text][0]
        assert run_zero.dx == 0.0
        assert run_zero.dy == 0.0

        exp_zero = SVGExporter(strict=True).render(scene_zero)
        assert len(exp_zero.fallbacks) == 0
        orig_img_zero = resvg_render(svg_zero, 400, 200)
        exp_img_zero = resvg_render(exp_zero.svg, 400, 200)
        mask_zero = (orig_img_zero[..., 3] > 0) | (exp_img_zero[..., 3] > 0)
        assert np.any(mask_zero)
        alpha_mae_zero = float(np.mean(np.abs(orig_img_zero[..., 3].astype(float) - exp_img_zero[..., 3].astype(float))[mask_zero]))
        assert alpha_mae_zero <= 1.0

        # Outer nonzero dx/dy + inner omitted dx/dy
        svg_omitted = """<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                <tspan dx="30" dy="15">
                    <tspan>InnerInherited</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_omitted = SVGImporter(strict=True, font_resolver=resolver).parse(svg_omitted).scene
        t_omitted = scene_omitted.find_by_type(Text)[0]
        assert t_omitted.runs is not None
        run_omitted = [r for r in t_omitted.runs if "InnerInherited" in r.text][0]
        assert run_omitted.dx == 30.0
        assert run_omitted.dy == 15.0

        exp_omitted = SVGExporter(strict=True).render(scene_omitted)
        assert len(exp_omitted.fallbacks) == 0
        orig_img_omitted = resvg_render(svg_omitted, 400, 200)
        exp_img_omitted = resvg_render(exp_omitted.svg, 400, 200)
        mask_omitted = (orig_img_omitted[..., 3] > 0) | (exp_img_omitted[..., 3] > 0)
        assert np.any(mask_omitted)
        alpha_mae_omitted = float(np.mean(np.abs(orig_img_omitted[..., 3].astype(float) - exp_img_omitted[..., 3].astype(float))[mask_omitted]))
        assert alpha_mae_omitted <= 1.0

        # Displacements must produce distinct rendering coordinates in both resvg and DrawCV
        ys_z, xs_z = np.where(orig_img_zero[..., 3] > 0)
        ys_o, xs_o = np.where(orig_img_omitted[..., 3] > 0)
        assert xs_o.min() > xs_z.min() + 15
        assert ys_o.min() > ys_z.min() + 8

    def test_rtl_positioned_tspan_followed_by_ltr_tail_parity(self):
        resolver = get_test_resolver()
        import numpy as np

        svg = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" direction="ltr" font-family="Noto Sans" font-size="20">Start <tspan x="180" direction="rtl" font-family="Noto Sans Arabic">سلام</tspan> Tail</text>
        </svg>"""
        scene = SVGImporter(strict=True, font_resolver=resolver).parse(svg).scene
        t = scene.find_by_type(Text)[0]
        assert t.runs is not None
        assert len(t.runs) == 3
        assert t.runs[0].text == "Start "
        assert t.runs[1].text == "سلام"
        assert t.runs[1].x == 180.0
        assert t.runs[1].direction == "rtl"
        assert t.runs[2].text == " Tail"
        assert t.runs[2].x is None
        assert t.runs[2].direction is None

        # Strict SVG export
        exp = SVGExporter(strict=True).render(scene)
        assert len(exp.fallbacks) == 0

        # resvg render of original and exported
        orig_img = resvg_render(svg, 400, 100)
        exp_img = resvg_render(exp.svg, 400, 100)
        mask = (orig_img[..., 3] > 0) | (exp_img[..., 3] > 0)
        assert np.any(mask)
        alpha_mae = float(np.mean(np.abs(orig_img[..., 3].astype(float) - exp_img[..., 3].astype(float))[mask]))
        assert alpha_mae <= 1.0

        # DrawCV render vs resvg comparison
        drawcv_img = OpenCVRenderer().render(scene, alpha=True).buffer
        r_alpha = orig_img[..., 3]
        d_alpha = drawcv_img[..., 3]

        # In both resvg and DrawCV:
        # Start is around xs < 100
        # Salam is positioned at 180..210
        # Tail is positioned AFTER Salam (> 205), NOT before Salam
        r_ys_tail, r_xs_tail = np.where((r_alpha > 0) & (np.arange(400)[None, :] >= 205))
        d_ys_tail, d_xs_tail = np.where((d_alpha > 0) & (np.arange(400)[None, :] >= 205))
        assert len(r_xs_tail) > 0
        assert len(d_xs_tail) > 0
        assert abs(d_xs_tail.min() - r_xs_tail.min()) <= 5

        # Check overlap
        overlap = np.count_nonzero((r_alpha > 0) & (d_alpha > 0))
        assert overlap > 0

    def test_strict_export_font_and_direction_eligibility(self):
        noto_sans = FontAsset.from_file(FONTS_DIR / "notosans.ttf")
        noto_arabic = FontAsset.from_file(FONTS_DIR / "notosansarabic.ttf")

        # 1. Semantic bold override without run font asset
        t_bold_no_font = Text(
            runs=(
                TextRun("Regular"),
                TextRun("Bold", font_weight="bold", fonts=None),
            ),
            fonts=(noto_sans,),
            font_family_name="Noto Sans",
            font_size=20,
            direction="ltr",
            text_origin="baseline",
        )
        scene1 = Scene(width=200, height=100)
        scene1.add(t_bold_no_font)
        with pytest.raises(RenderError, match="semantic font override without run font asset"):
            SVGExporter(strict=True).render(scene1)
        # Strict=False must fall back to raster
        exp1 = SVGExporter(strict=False).render(scene1)
        assert len(exp1.fallbacks) == 1
        assert "<image" in exp1.svg

        # 2. Run font asset override without SVG semantic descriptor
        t_font_no_desc = Text(
            runs=(
                TextRun("Regular"),
                TextRun("Other", fonts=(noto_arabic,)),
            ),
            fonts=(noto_sans,),
            font_family_name="Noto Sans",
            font_size=20,
            direction="ltr",
            text_origin="baseline",
        )
        scene2 = Scene(width=200, height=100)
        scene2.add(t_font_no_desc)
        with pytest.raises(RenderError, match="run font asset override without SVG semantic descriptor"):
            SVGExporter(strict=True).render(scene2)
        exp2 = SVGExporter(strict=False).render(scene2)
        assert len(exp2.fallbacks) == 1
        assert "<image" in exp2.svg

        # 3. Root direction="auto" with RTL text
        t_rtl_auto = Text(
            "سلام",
            fonts=(noto_arabic,),
            font_family_name="Noto Sans Arabic",
            font_size=20,
            direction="auto",
            text_origin="baseline",
        )
        scene3 = Scene(width=200, height=100)
        scene3.add(t_rtl_auto)
        with pytest.raises(RenderError, match="text direction auto with RTL text"):
            SVGExporter(strict=True).render(scene3)
        exp3 = SVGExporter(strict=False).render(scene3)
        assert len(exp3.fallbacks) == 1
        assert "<image" in exp3.svg

        # 4. Run direction="auto"
        t_run_auto = Text(
            runs=(
                TextRun("Hello"),
                TextRun("World", direction="auto"),
            ),
            fonts=(noto_sans,),
            font_family_name="Noto Sans",
            font_size=20,
            direction="ltr",
            text_origin="baseline",
        )
        scene4 = Scene(width=200, height=100)
        scene4.add(t_run_auto)
        with pytest.raises(RenderError, match="run direction auto"):
            SVGExporter(strict=True).render(scene4)
        exp4 = SVGExporter(strict=False).render(scene4)
        assert len(exp4.fallbacks) == 1
        assert "<image" in exp4.svg

    def test_empty_tspan_positioning_scope(self):
        resolver = get_test_resolver()
        import numpy as np

        # 1. Empty positioned tspan with dx followed by parent tail
        svg_empty_dx = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                A<tspan dx="100"></tspan>B
            </text>
        </svg>"""
        scene_dx = SVGImporter(strict=True, font_resolver=resolver).parse(svg_empty_dx).scene
        t_dx = scene_dx.find_by_type(Text)[0]
        assert t_dx.runs is not None
        run_b_dx = [r for r in t_dx.runs if r.text == "B"][0]
        assert run_b_dx.dx == 0.0

        exp_dx = SVGExporter(strict=True).render(scene_dx)
        assert len(exp_dx.fallbacks) == 0
        orig_img_dx = resvg_render(svg_empty_dx, 400, 100)
        exp_img_dx = resvg_render(exp_dx.svg, 400, 100)
        mask_dx = (orig_img_dx[..., 3] > 0) | (exp_img_dx[..., 3] > 0)
        assert np.any(mask_dx)
        alpha_mae_dx = float(np.mean(np.abs(orig_img_dx[..., 3].astype(float) - exp_img_dx[..., 3].astype(float))[mask_dx]))
        assert alpha_mae_dx <= 1.0

        # 2. Empty tspan with x/y followed by parent tail
        svg_empty_xy = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                A<tspan x="200" y="80"></tspan>B
            </text>
        </svg>"""
        scene_xy = SVGImporter(strict=True, font_resolver=resolver).parse(svg_empty_xy).scene
        t_xy = scene_xy.find_by_type(Text)[0]
        assert t_xy.runs is not None
        run_b_xy = [r for r in t_xy.runs if r.text == "B"][0]
        assert run_b_xy.x is None
        assert run_b_xy.y is None

        exp_xy = SVGExporter(strict=True).render(scene_xy)
        assert len(exp_xy.fallbacks) == 0
        orig_img_xy = resvg_render(svg_empty_xy, 400, 100)
        exp_img_xy = resvg_render(exp_xy.svg, 400, 100)
        mask_xy = (orig_img_xy[..., 3] > 0) | (exp_img_xy[..., 3] > 0)
        assert np.any(mask_xy)
        alpha_mae_xy = float(np.mean(np.abs(orig_img_xy[..., 3].astype(float) - exp_img_xy[..., 3].astype(float))[mask_xy]))
        assert alpha_mae_xy <= 1.0

        # 3. Nested descendant propagation still working
        svg_nested_prop = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                <tspan dx="30">
                    <tspan>Nested</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_nested = SVGImporter(strict=True, font_resolver=resolver).parse(svg_nested_prop).scene
        t_nested = scene_nested.find_by_type(Text)[0]
        assert t_nested.runs is not None
        assert t_nested.runs[0].dx == 30.0

        exp_nested = SVGExporter(strict=True).render(scene_nested)
        assert len(exp_nested.fallbacks) == 0
        orig_img_nested = resvg_render(svg_nested_prop, 400, 100)
        exp_img_nested = resvg_render(exp_nested.svg, 400, 100)
        mask_nested = (orig_img_nested[..., 3] > 0) | (exp_img_nested[..., 3] > 0)
        assert np.any(mask_nested)
        alpha_mae_nested = float(np.mean(np.abs(orig_img_nested[..., 3].astype(float) - exp_img_nested[..., 3].astype(float))[mask_nested]))
        assert alpha_mae_nested <= 1.0

        # 4. Nested descendant with explicit dx=0 override
        svg_nested_zero = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">
                <tspan dx="30">
                    <tspan dx="0">Nested</tspan>
                </tspan>
            </text>
        </svg>"""
        scene_zero = SVGImporter(strict=True, font_resolver=resolver).parse(svg_nested_zero).scene
        t_zero = scene_zero.find_by_type(Text)[0]
        assert t_zero.runs is not None
        assert t_zero.runs[0].dx == 0.0

        exp_zero = SVGExporter(strict=True).render(scene_zero)
        assert len(exp_zero.fallbacks) == 0
        orig_img_zero = resvg_render(svg_nested_zero, 400, 100)
        exp_img_zero = resvg_render(exp_zero.svg, 400, 100)
        mask_zero = (orig_img_zero[..., 3] > 0) | (exp_img_zero[..., 3] > 0)
        assert np.any(mask_zero)
        alpha_mae_zero = float(np.mean(np.abs(orig_img_zero[..., 3].astype(float) - exp_img_zero[..., 3].astype(float))[mask_zero]))
        assert alpha_mae_zero <= 1.0

        # 5. Empty nested x override: ancestor pending x retained for sibling
        svg_nested_x_empty = '<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg"><text x="20" y="50" font-family="Noto Sans" font-size="20"><tspan x="100"><tspan x="200"></tspan><tspan>Text</tspan></tspan></text></svg>'
        scene_nxe = SVGImporter(strict=True, font_resolver=resolver).parse(svg_nested_x_empty).scene
        t_nxe = scene_nxe.find_by_type(Text)[0]
        assert t_nxe.runs is not None
        run_nxe = [r for r in t_nxe.runs if r.text == "Text"][0]
        assert run_nxe.x == 100.0

        exp_nxe = SVGExporter(strict=True).render(scene_nxe)
        assert len(exp_nxe.fallbacks) == 0
        orig_img_nxe = resvg_render(svg_nested_x_empty, 400, 100)
        exp_img_nxe = resvg_render(exp_nxe.svg, 400, 100)
        mask_nxe = (orig_img_nxe[..., 3] > 0) | (exp_img_nxe[..., 3] > 0)
        assert np.any(mask_nxe)
        alpha_mae_nxe = float(np.mean(np.abs(orig_img_nxe[..., 3].astype(float) - exp_img_nxe[..., 3].astype(float))[mask_nxe]))
        assert alpha_mae_nxe <= 1.0

        # 6. Empty nested dx override: ancestor pending dx retained for sibling
        svg_nested_dx_empty = '<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg"><text x="20" y="50" font-family="Noto Sans" font-size="20"><tspan dx="30"><tspan dx="80"></tspan><tspan>Text</tspan></tspan></text></svg>'
        scene_ndxe = SVGImporter(strict=True, font_resolver=resolver).parse(svg_nested_dx_empty).scene
        t_ndxe = scene_ndxe.find_by_type(Text)[0]
        assert t_ndxe.runs is not None
        run_ndxe = [r for r in t_ndxe.runs if r.text == "Text"][0]
        assert run_ndxe.dx == 30.0

        exp_ndxe = SVGExporter(strict=True).render(scene_ndxe)
        assert len(exp_ndxe.fallbacks) == 0
        orig_img_ndxe = resvg_render(svg_nested_dx_empty, 400, 100)
        exp_img_ndxe = resvg_render(exp_ndxe.svg, 400, 100)
        mask_ndxe = (orig_img_ndxe[..., 3] > 0) | (exp_img_ndxe[..., 3] > 0)
        assert np.any(mask_ndxe)
        alpha_mae_ndxe = float(np.mean(np.abs(orig_img_ndxe[..., 3].astype(float) - exp_img_ndxe[..., 3].astype(float))[mask_ndxe]))
        assert alpha_mae_ndxe <= 1.0

        # 7. Non-empty nested override: ancestor value consumed by first addressable descendant
        svg_nested_x_consumed = '<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg"><text x="20" y="50" font-family="Noto Sans" font-size="20"><tspan x="100"><tspan x="200">A</tspan><tspan>B</tspan></tspan></text></svg>'
        scene_nxc = SVGImporter(strict=True, font_resolver=resolver).parse(svg_nested_x_consumed).scene
        t_nxc = scene_nxc.find_by_type(Text)[0]
        assert t_nxc.runs is not None
        assert len(t_nxc.runs) == 2
        run_a = [r for r in t_nxc.runs if r.text == "A"][0]
        run_b = [r for r in t_nxc.runs if r.text == "B"][0]
        assert run_a.x == 200.0
        assert run_b.x is None

        exp_nxc = SVGExporter(strict=True).render(scene_nxc)
        assert len(exp_nxc.fallbacks) == 0
        orig_img_nxc = resvg_render(svg_nested_x_consumed, 400, 100)
        exp_img_nxc = resvg_render(exp_nxc.svg, 400, 100)
        mask_nxc = (orig_img_nxc[..., 3] > 0) | (exp_img_nxc[..., 3] > 0)
        assert np.any(mask_nxc)
        alpha_mae_nxc = float(np.mean(np.abs(orig_img_nxc[..., 3].astype(float) - exp_img_nxc[..., 3].astype(float))[mask_nxc]))
        assert alpha_mae_nxc <= 1.0

    def test_mixed_xml_space_trailing_processing(self):
        resolver = get_test_resolver()
        import numpy as np

        # 1. <text>A <tspan xml:space="preserve">B </tspan></text>
        svg_mixed1 = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20">A <tspan xml:space="preserve">B </tspan></text>
        </svg>"""
        scene1 = SVGImporter(strict=True, font_resolver=resolver).parse(svg_mixed1).scene
        t1 = scene1.find_by_type(Text)[0]
        assert t1.text == "A B "

        exp1 = SVGExporter(strict=True).render(scene1)
        assert len(exp1.fallbacks) == 0
        orig_img1 = resvg_render(svg_mixed1, 400, 100)
        exp_img1 = resvg_render(exp1.svg, 400, 100)
        mask1 = (orig_img1[..., 3] > 0) | (exp_img1[..., 3] > 0)
        assert np.any(mask1)
        alpha_mae1 = float(np.mean(np.abs(orig_img1[..., 3].astype(float) - exp_img1[..., 3].astype(float))[mask1]))
        assert alpha_mae1 <= 1.0

        # 2. <text xml:space="preserve">A <tspan xml:space="default"> B </tspan>C </text>
        svg_mixed2 = """<svg width="400" height="100" xmlns="http://www.w3.org/2000/svg">
            <text x="20" y="50" font-family="Noto Sans" font-size="20" xml:space="preserve">A <tspan xml:space="default"> B </tspan>C </text>
        </svg>"""
        scene2 = SVGImporter(strict=True, font_resolver=resolver).parse(svg_mixed2).scene
        t2 = scene2.find_by_type(Text)[0]
        assert t2.text == "A B C "

        exp2 = SVGExporter(strict=True).render(scene2)
        assert len(exp2.fallbacks) == 0
        orig_img2 = resvg_render(svg_mixed2, 400, 100)
        exp_img2 = resvg_render(exp2.svg, 400, 100)
        mask2 = (orig_img2[..., 3] > 0) | (exp_img2[..., 3] > 0)
        assert np.any(mask2)
        alpha_mae2 = float(np.mean(np.abs(orig_img2[..., 3].astype(float) - exp_img2[..., 3].astype(float))[mask2]))
        assert alpha_mae2 <= 1.0



