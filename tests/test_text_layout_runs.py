import math
import numpy as np
import pytest

pytest.importorskip("uharfbuzz")
pytest.importorskip("freetype")
pytest.importorskip("icu")
pytest.importorskip("regex")

from drawcv.canvas import Canvas
from drawcv.core.color import Color
from drawcv.core.geometry import Point
from drawcv.renderer import OpenCVRenderer
from drawcv.shapes.text import Text, TextAnchor, TextRun
from drawcv.typography.assets import FontAsset
from drawcv.typography.layout import (
    ShapedGlyph,
    _shape_cached,
    layout_text_runs,
)


@pytest.fixture
def noto_sans():
    return FontAsset.from_file("tests/assets/fonts/notosans.ttf")


@pytest.fixture
def noto_arabic():
    return FontAsset.from_file("tests/assets/fonts/notosansarabic.ttf")


def test_shaped_glyph_unpacking_and_attributes():
    glyph = ShapedGlyph(gid=42, x=10.5, y=-5.0, cluster_start=1, cluster_end=3, x_advance=15.0)
    # 3-element unpacking for backward compatibility
    gid, x, y = glyph
    assert gid == 42
    assert x == 10.5
    assert y == -5.0
    # Rich cluster and advance metadata
    assert glyph.cluster_start == 1
    assert glyph.cluster_end == 3
    assert glyph.x_advance == 15.0


def test_harfbuzz_cluster_preservation_latin_ligature(noto_sans):
    # "office": "ffi" forms a ligature covering indices [1, 4)
    glyphs, advance = _shape_cached("office", noto_sans, 32 * 64, "Latn", 0)
    assert len(glyphs) > 0
    # First glyph is 'o' (index 0)
    assert glyphs[0].cluster_start == 0
    assert glyphs[0].cluster_end == 1
    # Second glyph is 'ffi' ligature (indices 1..3)
    assert glyphs[1].cluster_start == 1
    assert glyphs[1].cluster_end == 4


def test_harfbuzz_cluster_preservation_combining_mark(noto_sans):
    # 'e' (U+0065) + combining acute (U+0301)
    text = "e\u0301"
    glyphs, advance = _shape_cached(text, noto_sans, 32 * 64, "Latn", 0)
    assert len(glyphs) == 1
    # Cluster covers both input codepoints
    assert glyphs[0].cluster_start == 0
    assert glyphs[0].cluster_end == 2


def test_arabic_contextual_shaping_and_rtl_clusters(noto_arabic):
    # 'سلام' (salam: sin, lam, alif, mim)
    text = "سلام"
    glyphs, advance = _shape_cached(text, noto_arabic, 32 * 64, "Arab", 1)
    assert len(glyphs) > 0
    # In RTL, clusters are in descending visual order: mim (3), alif (2), lam (1), sin (0)
    clusters = [g.cluster_start for g in glyphs]
    assert clusters == sorted(clusters, reverse=True)
    # Every glyph has valid cluster interval
    for g in glyphs:
        assert g.cluster_start < g.cluster_end
        assert g.cluster_end <= len(text)


def test_layout_text_runs_coalescing(noto_sans):
    # Adjacent runs with identical styles and no position resets should coalesce
    r1 = TextRun("Hello ", font_size=20)
    r2 = TextRun("World", font_size=20)
    layout = layout_text_runs(
        runs=(r1, r2),
        default_fonts=(noto_sans,),
        default_font_size=20,
        text_origin="baseline",
    )
    assert layout.lines[0].text == "Hello World"
    assert layout.styled_runs is None  # no custom paint needed


def test_layout_text_runs_scalar_dx_dy(noto_sans):
    # Scalar dx/dy adjusts position without breaking text chunk
    r1 = TextRun("A", font_size=20)
    r2 = TextRun("B", font_size=20, dx=10.0, dy=-2.0)
    layout = layout_text_runs(
        runs=(r1, r2),
        default_fonts=(noto_sans,),
        default_font_size=20,
        text_origin="baseline",
    )
    assert len(layout.lines) == 1
    assert layout.lines[0].text == "AB"


def test_layout_text_runs_absolute_chunk_boundary(noto_sans):
    # Absolute x/y starts a new text chunk
    r1 = TextRun("First", font_size=20, x=50, y=50)
    r2 = TextRun("Second", font_size=20, x=150, y=100)
    layout = layout_text_runs(
        runs=(r1, r2),
        default_fonts=(noto_sans,),
        default_font_size=20,
        text_origin="baseline",
    )
    assert len(layout.lines) == 2
    assert layout.lines[0].text == "First"
    assert layout.lines[1].text == "Second"
    assert layout.lines[0].baseline == 50.0
    assert layout.lines[1].baseline == 100.0


def test_layout_text_runs_text_anchor_displacement(noto_sans):
    # Middle anchor centers the chunk around the anchor position
    r = TextRun("Centered", font_size=20, text_anchor=TextAnchor.MIDDLE)
    layout = layout_text_runs(
        runs=(r,),
        default_fonts=(noto_sans,),
        default_font_size=20,
        text_origin="baseline",
        text_anchor="middle",
        base_x=100.0,
        base_y=50.0,
    )
    # Ink bounding box should be centered around x = 0 (relative to base_x = 100)
    center_x = layout.ink.x + layout.ink.width / 2.0
    assert abs(center_x) < 5.0  # close to 0 in local space relative to base_x


def test_multi_run_rendering_different_colors(noto_sans):
    # Render Text with red and blue runs on a canvas
    red = Color(255, 0, 0)
    blue = Color(0, 0, 255)
    r1 = TextRun("RED", fill=red, font_size=24)
    r2 = TextRun("BLUE", fill=blue, font_size=24, dx=10.0)

    text_obj = Text(
        runs=(r1, r2),
        fonts=(noto_sans,),
        position=Point(50, 50),
        text_origin="baseline",
    )

    canvas = Canvas(300, 100)
    renderer = OpenCVRenderer()
    renderer.render_drawable(text_obj, canvas)

    # Convert canvas buffer (BGR) to check colors
    # Canvas buffer is in BGR format
    bgr = canvas.buffer
    # Has red pixels (B=0, R>100)
    has_red = np.any((bgr[:, :, 2] > 150) & (bgr[:, :, 0] < 50))
    # Has blue pixels (B>100, R=0)
    has_blue = np.any((bgr[:, :, 0] > 150) & (bgr[:, :, 2] < 50))

    assert has_red, "Canvas should contain red pixels from run 1"
    assert has_blue, "Canvas should contain blue pixels from run 2"
