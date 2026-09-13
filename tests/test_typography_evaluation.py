"""Optional feasibility regressions. Install examples/typography-requirements.txt."""
import os
os.environ.setdefault("PYTHAINLP_READ_ONLY", "1")
os.environ.setdefault("PYTHAINLP_OFFLINE", "1")
import hashlib
import json
import numpy as np
import pytest

pytest.importorskip("uharfbuzz")
pytest.importorskip("freetype")
pytest.importorskip("regex")
pytest.importorskip("pythainlp")
from examples._typography_backend import (FONT_DIR, shape_run, rasterize, fallback_runs,
    wrap_thai, line_advance, draw_line, MissingGlyphError)

LATIN = FONT_DIR / "notosans.ttf"
THAI = FONT_DIR / "notosansthai.ttf"
ARABIC = FONT_DIR / "notosansarabic.ttf"
SAMPLES = [("AVATAR office", LATIN), ("น้ำ กุ้ง กิ๊ง ปู่ ผู้", THAI),
           ("السلام عليكم", ARABIC), ("Ág j office", FONT_DIR / "sourcesans3.otf")]
PARAGRAPH = "ภาษาไทยต้องตัดคำอย่างถูกต้อง เพื่อให้อ่านข้อความได้ง่ายขึ้น"


def test_font_sources_are_pinned_and_unchanged():
    for entry in json.loads((FONT_DIR / "manifest.json").read_text(encoding="utf-8")):
        assert hashlib.sha256((FONT_DIR / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
        assert "/main/" not in entry["source"]


@pytest.mark.parametrize("text,path", SAMPLES)
def test_unhinted_outline_metrics_enclose_visible_ink(text, path):
    run = shape_run(text, path)
    raster = rasterize(run)
    ys, xs = np.where(raster.mask > 0)
    ink = (xs.min()+raster.left, ys.min()+raster.top,
           xs.max()+1+raster.left, ys.max()+1+raster.top)
    # Different outline scaling/coverage rounding may differ by one pixel.
    assert np.max(np.abs(np.array(ink)-np.array(run.outline_bounds))) <= 1.1
    assert all(g["gid"] > 0 for g in run.glyphs)


@pytest.mark.parametrize("text,path", SAMPLES)
def test_glyph_selection_and_positions_against_qt(text, path):
    pytest.importorskip("PySide6.QtGui")
    from examples._typography_qt import layout_reference
    run = shape_run(text, path)
    reference = layout_reference(text, [path])
    qt_run = reference["runs"][0]
    glyphs, positions = qt_run["glyphs"], qt_run["positions"]
    if path == ARABIC:
        glyphs, positions = glyphs[::-1], positions[::-1]
    assert glyphs == [g["gid"] for g in run.glyphs]
    assert abs(reference["lines"][0]["width"]-run.advance) < .5
    qt_xy = np.array(positions)
    qt_xy[:, 1] -= reference["lines"][0]["baseline"]
    assert np.max(np.abs(qt_xy-np.array([(g["x"], g["y"]) for g in run.glyphs]))) < .5


def test_ligature_and_arabic_joining_are_not_per_character_rendering():
    assert len(shape_run("office", LATIN).glyphs) < len("office")
    joined = [g["gid"] for g in shape_run("سلام", ARABIC).glyphs]
    isolated = [g["gid"] for c in "سلام" for g in shape_run(c, ARABIC).glyphs]
    assert sorted(joined) != sorted(isolated)


def test_fallback_preserves_combining_cluster_and_uses_explicit_fonts():
    runs = fallback_runs("DrawCV กิ๊ง 123", [LATIN, THAI])
    assert "".join(r.text for r in runs) == "DrawCV กิ๊ง 123"
    assert {r.font_path for r in runs} == {LATIN, THAI}
    assert any("กิ๊ง" in r.text and r.font_path == THAI for r in runs)
    with pytest.raises(MissingGlyphError):
        fallback_runs("กิ๊ง", [LATIN])
    with pytest.raises(MissingGlyphError):
        fallback_runs("\U0010ffff", [LATIN, THAI])
    with pytest.raises(ValueError, match="bidi"):
        fallback_runs("English العربية", [LATIN, ARABIC])


@pytest.mark.parametrize("width", [200, 300, 420])
def test_thai_wrapping_preserves_words_and_fits(width):
    from pythainlp.tokenize import word_tokenize
    lines = wrap_thai(PARAGRAPH, [LATIN, THAI], 32, width)
    assert len(lines) > 1
    assert "".join(lines).replace(" ", "") == PARAGRAPH.replace(" ", "")
    words = word_tokenize(PARAGRAPH, engine="newmm", keep_whitespace=True)
    boundaries = np.cumsum([len(w) for w in words])
    offset = 0
    for line in lines:
        start = PARAGRAPH.index(line, offset)
        offset = start+len(line)
        assert offset in boundaries
        assert line_advance(line, [LATIN, THAI], 32) <= width


def test_explicit_blank_lines_and_overlong_word_policy():
    assert wrap_thai("Hello\n\nWorld", [LATIN], 24, 300) == ["Hello", "", "World"]
    with pytest.raises(ValueError, match="word"):
        wrap_thai("ภาษาไทย", [THAI], 36, 2)
    assert shape_run("", LATIN).advance == 0
    assert not rasterize(shape_run("   ", LATIN)).mask.any()
    assert shape_run("A ", LATIN).advance > shape_run("A", LATIN).advance


@pytest.mark.parametrize("alignment", ["left", "center", "right"])
def test_qt_multiline_alignment_and_line_spacing(alignment):
    pytest.importorskip("PySide6.QtGui")
    from examples._typography_qt import layout_reference
    q = layout_reference("AVATAR\nHi", [LATIN], 36, 300, alignment, spacing=1.5)
    assert len(q["lines"]) == 2
    first, second = q["lines"]
    assert abs(second["baseline"]-first["baseline"]-first["height"]*1.5) < .1
    if alignment == "left":
        assert first["x"] == second["x"] == 0
    elif alignment == "center":
        assert abs(first["x"]+first["width"]/2 - (second["x"]+second["width"]/2)) < 1
    else:
        assert abs(first["x"]+first["width"] - (second["x"]+second["width"])) < 1


def test_explicit_thai_breaks_fit_qt_reference():
    pytest.importorskip("PySide6.QtGui")
    from examples._typography_qt import layout_reference
    lines = wrap_thai(PARAGRAPH, [THAI], 32, 300)
    q = layout_reference("\n".join(lines), [THAI], 32, 300)
    assert len(q["lines"]) == len(lines)
    assert all(line["width"] <= 300 for line in q["lines"])


def test_qt_bidi_and_font_fallback_are_separate_from_single_font_shaping():
    pytest.importorskip("PySide6.QtGui")
    from examples._typography_qt import layout_reference
    q = layout_reference("DrawCV العربية 123", [LATIN, ARABIC], allow_fallback=True)
    assert {r["rtl"] for r in q["runs"]} == {False, True}
    assert all(0 not in r["glyphs"] for r in q["runs"])
    assert {r["family"] for r in q["runs"]} <= {"Noto Sans", "Noto Sans Arabic"}
    no_fallback = layout_reference("กิ๊ง", [LATIN], allow_fallback=False)
    assert any(0 in r["glyphs"] for r in no_fallback["runs"])


def test_raster_text_through_existing_scene_alpha_effects_and_png(tmp_path):
    import cv2
    from drawcv import (Scene, OpenCVRenderer, ImageObject, Point, Group, Transform,
                        Color, BlurEffect, ClipRect)
    mask = np.zeros((120, 380), np.uint8)
    draw_line(mask, "DrawCV กิ๊ง 123", [LATIN, THAI], 36, 20, 70)
    image = np.zeros((*mask.shape, 4), np.uint8)
    image[..., 2], image[..., 3] = 255, mask
    scene = Scene(500, 240, background=Color(0, 0, 0, 0))
    obj = ImageObject(image, position=Point(25, 40), width=380, height=120,
                      transform=Transform(rotation=8), effects=[BlurEffect(kernel_size=5, sigma=1)],
                      clip=ClipRect(0, 0, 450, 200))
    scene.add(Group(children=[obj], opacity=.6))
    canvas = OpenCVRenderer().render(scene, alpha=True)
    canvas.save(tmp_path / "text.png")
    decoded = cv2.imread(str(tmp_path / "text.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, canvas.buffer)
    visible = decoded[..., 3] > 0
    assert visible.any() and np.all(decoded[visible, :3] == (0, 0, 255))
    a = decoded[..., 3:4]/255
    for bg in [Color.white(), Color.black(), Color(60, 170, 120)]:
        external = np.rint(decoded[..., :3]*a + np.array(bg.to_bgr())*(1-a))
        scene.background = bg
        reference = OpenCVRenderer().render(scene, alpha=True).buffer[..., :3]
        assert np.max(np.abs(reference.astype(float)-external)) <= 1
