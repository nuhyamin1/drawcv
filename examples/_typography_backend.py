"""Milestone 4 feasibility code, NOT a public DrawCV text API.

Explicit-font horizontal runs; LTR Latin/Thai fallback and dictionary wrapping.
Standalone RTL runs use HarfBuzz directly. Mixed bidi paragraphs are deliberately
left to the Qt reference rather than implementing a partial Unicode bidi engine.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import unicodedata
import numpy as np
import uharfbuzz as hb
import freetype
import regex

FONT_DIR = Path(__file__).resolve().parents[1] / "tests/assets/fonts"


class MissingGlyphError(ValueError):
    pass


@dataclass
class ShapedRun:
    text: str
    font_path: Path
    size: int
    glyphs: list[dict]
    advance: float
    outline_bounds: tuple[float, float, float, float] | None


@dataclass
class Raster:
    mask: np.ndarray
    left: int
    top: int
    advance: float


def load_font(path, size):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer pixel size")
    font = hb.Font(hb.Face(Path(path).read_bytes()))
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)
    return font


def shape_run(text, path, size=36, *, direction=None, language=None):
    if "\n" in text:
        raise ValueError("shape_run accepts one line")
    font = load_font(path, size)
    if not text:
        return ShapedRun(text, Path(path), size, [], 0.0, None)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    if direction:
        buf.direction = direction
    if language:
        buf.language = language
    hb.shape(font, buf)
    glyphs, boxes = [], []
    pen = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        if info.codepoint == 0:
            raise MissingGlyphError(f"Missing glyph at character {info.cluster} in {Path(path).name}")
        x, y = pen + pos.x_offset/64, -pos.y_offset/64
        glyphs.append(dict(gid=info.codepoint, cluster=info.cluster, x=x, y=y,
                           advance=pos.x_advance/64))
        ext = font.get_glyph_extents(info.codepoint)
        if ext and ext.width and ext.height:
            left, top = x+ext.x_bearing/64, y-ext.y_bearing/64
            boxes.append((left, top, left+ext.width/64, top-ext.height/64))
        pen += pos.x_advance/64
    bounds = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else None
    return ShapedRun(text, Path(path), size, glyphs, pen, bounds)


def rasterize(run):
    """Render positioned glyph IDs with subpixel origins and grayscale coverage."""
    face = freetype.Face(str(run.font_path))
    face.set_pixel_sizes(0, run.size)
    tiles = []
    for glyph in run.glyphs:
        x, y = glyph["x"], glyph["y"]
        ix, iy = math.floor(x), math.floor(y)
        face.set_transform(freetype.Matrix(0x10000, 0, 0, 0x10000),
                           freetype.Vector(round((x-ix)*64), -round((y-iy)*64)))
        face.load_glyph(glyph["gid"], freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
        bitmap = face.glyph.bitmap
        if not bitmap.width or not bitmap.rows:
            continue
        if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
            raise ValueError("The fixture only supports grayscale outline fonts")
        tile = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, abs(bitmap.pitch))[:, :bitmap.width]
        if bitmap.pitch < 0:
            tile = tile[::-1]
        tiles.append((ix+face.glyph.bitmap_left, iy-face.glyph.bitmap_top, tile))
    if not tiles:
        return Raster(np.zeros((1, 1), np.uint8), 0, 0, run.advance)
    left = min(x for x, y, tile in tiles)
    top = min(y for x, y, tile in tiles)
    right = max(x+tile.shape[1] for x, y, tile in tiles)
    bottom = max(y+tile.shape[0] for x, y, tile in tiles)
    mask = np.zeros((bottom-top, right-left), dtype=np.float32)
    for x, y, tile in tiles:
        target = mask[y-top:y-top+tile.shape[0], x-left:x-left+tile.shape[1]]
        coverage = tile.astype(np.float32)/255
        target[:] = coverage + target*(1-coverage)
    return Raster(np.rint(mask*255).astype(np.uint8), left, top, run.advance)


def fallback_runs(text, paths, size=36):
    """Whole-grapheme coverage selection, coalesced to preserve shaping context.

    Limited to LTR Latin/Thai samples. This is not a general fallback/bidi engine.
    """
    if any(unicodedata.bidirectional(c) in ("R", "AL", "AN") for c in text):
        raise ValueError("LTR fixture cannot itemize bidi text; use the paragraph reference")
    fonts = [load_font(p, size) for p in paths]
    spans = []
    for cluster in regex.findall(r"\X", text):
        chosen = next((i for i, f in enumerate(fonts)
                       if all(f.get_nominal_glyph(ord(c)) for c in cluster)), None)
        if chosen is None:
            raise MissingGlyphError("No supplied font covers grapheme " + ascii(cluster))
        if spans and spans[-1][0] == chosen:
            spans[-1] = (chosen, spans[-1][1]+cluster)
        else:
            spans.append((chosen, cluster))
    return [shape_run(s, paths[i], size) for i, s in spans]


def line_advance(text, paths, size=36):
    return sum(run.advance for run in fallback_runs(text, paths, size))


def wrap_thai(text, paths, size=36, width=320):
    """Greedy newmm word boundaries, reshaping each candidate line.

    Break whitespace is trimmed. Overlong words raise rather than splitting a
    Thai cluster. Explicit empty paragraphs are retained.
    """
    from pythainlp.tokenize import word_tokenize
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be positive and finite")
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for token in word_tokenize(paragraph, engine="newmm", keep_whitespace=True):
            candidate = current+token
            if line_advance(candidate.rstrip(), paths, size) <= width:
                current = candidate
                continue
            if current.strip():
                lines.append(current.rstrip())
            current = token.lstrip()
            if line_advance(current.rstrip(), paths, size) > width:
                raise ValueError("A word exceeds the wrap width")
        lines.append(current.rstrip())
    return lines


def draw_line(mask, text, paths, size, x, baseline):
    """Place LTR fallback runs onto a generously sized evaluation mask."""
    pen = x
    for run in fallback_runs(text, paths, size):
        raster = rasterize(run)
        left, top = round(pen)+raster.left, round(baseline)+raster.top
        h, w = raster.mask.shape
        if left < 0 or top < 0 or left+w > mask.shape[1] or top+h > mask.shape[0]:
            raise ValueError("Evaluation canvas clips text")
        target = mask[top:top+h, left:left+w]
        a = raster.mask.astype(float)/255
        target[:] = np.rint(raster.mask + target*(1-a)).astype(np.uint8)
        pen += run.advance
    return pen-x
