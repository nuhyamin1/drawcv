"""Optional horizontal paragraph layout; native engines are loaded lazily."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from drawcv.typography.assets import FontAsset
import math
import threading
import unicodedata
import numpy as np
from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from .metrics import TextLineMetrics

try:
    import uharfbuzz as hb
    import freetype
    import regex
    import icu
except ImportError as exc:
    raise ValidationError('Font text requires the optional dependencies: pip install "pydrawcv[typography]"') from exc

_lock = threading.RLock()
_paragraphs = OrderedDict()
_paragraph_bytes = 0
_hits = 0
_MASK_BUDGET = 32 * 1024 * 1024
_MAX_PIXELS = 16 * 1024 * 1024


from typing import Any


@dataclass(frozen=True)
class ShapedGlyph:
    """Represents a shaped glyph with source cluster interval and advance geometry."""
    gid: int
    x: float
    y: float
    cluster_start: int = 0
    cluster_end: int = 0
    x_advance: float = 0.0
    y_advance: float = 0.0

    def __iter__(self):
        yield self.gid
        yield self.x
        yield self.y


@dataclass(frozen=True)
class StyledRunLayout:
    """Isolated raster mask and positioning for a styled text run."""
    mask: np.ndarray
    left: int
    top: int
    color: Any | None = None
    fill_none: bool = False
    fill_opacity: float = 1.0


@dataclass(frozen=True)
class Layout:
    width: float
    height: float
    ascent: float
    descent: float
    lines: tuple
    ink: BoundingBox | None
    mask: np.ndarray
    left: int
    top: int
    styled_runs: tuple[StyledRunLayout, ...] | None = None


@lru_cache(maxsize=16)
def _faces(asset, size):
    try:
        font = hb.Font(hb.Face(asset.data))
        font.scale = (size, size)
        hb.ot_font_set_funcs(font)
        face = freetype.Face.from_bytes(asset.data)
        face.set_char_size(0, size, 72, 72)
        return font, face
    except Exception as exc:
        raise ValidationError(f"Cannot load font: {asset.name}") from exc


@lru_cache(maxsize=256)
def _shape_cached(text, asset, size, script, level):
    font, _ = _faces(asset, size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.direction = "rtl" if level % 2 else "ltr"
    buf.script = script
    buf.language = {"Latn": "en", "Thai": "th", "Arab": "ar"}[script]
    hb.shape(font, buf)
    pen, glyphs = 0., []
    infos = buf.glyph_infos or []
    positions = buf.glyph_positions or []
    n_chars = len(text)

    clusters = [info.cluster for info in infos]
    distinct_clusters = sorted(set(clusters))
    cluster_ranges = {}
    for i, c in enumerate(distinct_clusters):
        c_end = distinct_clusters[i + 1] if i + 1 < len(distinct_clusters) else n_chars
        cluster_ranges[c] = (c, c_end)

    for info, pos in zip(infos, positions):
        if info.codepoint == 0:
            raise ValidationError(f"Missing shaped glyph in {asset.name}: {text!r}")
        c_start, c_end = cluster_ranges.get(info.cluster, (info.cluster, info.cluster + 1))
        glyphs.append(ShapedGlyph(
            gid=info.codepoint,
            x=pen + pos.x_offset / 64,
            y=-pos.y_offset / 64,
            cluster_start=c_start,
            cluster_end=c_end,
            x_advance=pos.x_advance / 64,
            y_advance=pos.y_advance / 64,
        ))
        pen += pos.x_advance / 64
    return tuple(glyphs), pen


def _shape_subrange(chunk_codepoints, item_offset, item_length, asset, size, script, level):
    font, _ = _faces(asset, size)
    buf = hb.Buffer()
    buf.add_codepoints(chunk_codepoints, item_offset=item_offset, item_length=item_length)
    buf.direction = "rtl" if level % 2 else "ltr"
    buf.script = script
    buf.language = {"Latn": "en", "Thai": "th", "Arab": "ar"}[script]
    hb.shape(font, buf)
    pen, glyphs = 0., []
    infos = buf.glyph_infos or []
    positions = buf.glyph_positions or []
    subrange_end = item_offset + item_length

    clusters = [info.cluster for info in infos]
    distinct_clusters = sorted(set(clusters))
    cluster_ranges = {}
    for i, c in enumerate(distinct_clusters):
        c_end = distinct_clusters[i + 1] if i + 1 < len(distinct_clusters) else subrange_end
        cluster_ranges[c] = (c, c_end)

    for info, pos in zip(infos, positions):
        if info.codepoint == 0:
            sub_str = "".join(chr(c) for c in chunk_codepoints[item_offset:subrange_end])
            raise ValidationError(f"Missing shaped glyph in {asset.name}: {sub_str!r}")
        c_start, c_end = cluster_ranges.get(info.cluster, (info.cluster, info.cluster + 1))
        if c_start < item_offset or c_end > subrange_end:
            raise ValidationError("Shaped cluster spans across style boundary")
        glyphs.append(ShapedGlyph(
            gid=info.codepoint,
            x=pen + pos.x_offset / 64,
            y=-pos.y_offset / 64,
            cluster_start=c_start,
            cluster_end=c_end,
            x_advance=pos.x_advance / 64,
            y_advance=pos.y_advance / 64,
        ))
        pen += pos.x_advance / 64
    return tuple(glyphs), pen


def _shape(text, asset, size, script, level):
    function = _shape_cached if len(text) <= 256 else _shape_cached.__wrapped__
    return function(text, asset, size, script, level)


@lru_cache(maxsize=512)
def _glyph_cached(asset, size, gid, fx, fy):
    _, face = _faces(asset, size)
    face.set_transform(freetype.Matrix(65536, 0, 0, 65536), freetype.Vector(fx, -fy))
    face.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
    bitmap = face.glyph.bitmap
    if not bitmap.rows or not bitmap.width:
        return None
    if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
        raise ValidationError("Font text supports grayscale outlines, not color or bitmap glyphs")
    if bitmap.rows * abs(bitmap.pitch) > _MAX_PIXELS:
        raise ValidationError("Glyph raster exceeds the 16M pixel limit")
    tile = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, abs(bitmap.pitch))[:, :bitmap.width].copy()
    if bitmap.pitch < 0:
        tile = tile[::-1].copy()
    ys, xs = np.where(tile > 0)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    tile = tile[y0:y1, x0:x1].copy()
    tile.flags.writeable = False
    return face.glyph.bitmap_left+int(x0), -face.glyph.bitmap_top+int(y0), tile


def _glyph(asset, size, gid, x, y):
    # Large sizes bypass the glyph cache, limiting retained bitmap memory.
    ix, iy = math.floor(x), math.floor(y)
    fn = _glyph_cached if size <= 128*64 else _glyph_cached.__wrapped__
    result = fn(asset, size, gid, round((x-ix)*64), round((y-iy)*64))
    if result is None:
        return None
    left, top, tile = result
    return ix+left, iy+top, tile


def _script(c):
    for script in ("Latn", "Thai", "Arab"):
        if regex.fullmatch(fr"\p{{Script={script}}}", c):
            return script
    if regex.fullmatch(r"[\p{Script=Common}\p{Script=Inherited}]", c):
        if regex.fullmatch(r"\p{Extended_Pictographic}|\p{Regional_Indicator}", c) or '\ufe00' <= c <= '\ufe0f':
            raise ValidationError("Emoji and variation selectors are not supported")
        return None
    raise ValidationError(f"Unsupported text script at U+{ord(c):04X}")


def _resolve(text, direction):
    for c in text:
        if unicodedata.category(c) in ("Cc", "Cf", "Cs", "Cn", "Zl", "Zp"):
            raise ValidationError(f"Unsupported text/control character U+{ord(c):04X}")
    source = icu.UnicodeString(text)
    paragraph = icu.Bidi()
    paragraph.setPara(source, icu.Bidi.DEFAULT_LTR if direction == 'auto' else int(direction == 'rtl'))
    levels = paragraph.getLevels() if text else ()
    chars, unit = [], 0
    for i, c in enumerate(text):
        chars.append(dict(ch=c, index=i, unit=unit, level=levels[unit], script=_script(c)))
        unit += len(c.encode('utf-16-le'))//2
    storage = dict(chars=chars, paragraph=paragraph, source=source, units=unit)
    # Common/inherited characters adopt an adjacent script within a level run.
    chars = storage['chars']
    future, following_scripts = {}, [None]*len(chars)
    for i in range(len(chars)-1, -1, -1):
        char = chars[i]
        if char['script']:
            future[char['level']] = char['script']
        following_scripts[i] = future.get(char['level'])
    previous = {}
    for i, char in enumerate(chars):
        if char['script'] is None:
            char['script'] = previous.get(char['level']) or following_scripts[i] or 'Latn'
        previous[char['level']] = char['script']
    return storage


def _runs(storage, start, end, fonts, size):
    chars = [dict(c) for c in storage['chars'][start:end]]
    if not chars:
        return [], 0.
    start_unit = chars[0]['unit']
    end_unit = storage['chars'][end]['unit'] if end < len(storage['chars']) else storage['units']
    line = storage['paragraph'].setLine(start_unit, end_unit)
    levels = line.getLevels()
    unit_ranks = {logical: visual for visual, logical in enumerate(line.getVisualMap())}
    ranks = {}
    for char in chars:
        char['level'] = levels[char['unit']-start_unit]
        ranks[char['index']] = unit_ranks[char['unit']-start_unit]
    logical = chars
    spans = []
    for char in logical:
        key = char['script'], char['level']
        if not spans or spans[-1][0] != key:
            spans.append((key, [char]))
        else:
            spans[-1][1].append(char)
    spans.sort(key=lambda span: min(ranks[c['index']] for c in span[1]))
    result, pen = [], 0.
    for (script, level), span in spans:
        text = ''.join(c['ch'] for c in span)
        asset = next((a for a in fonts if all(_faces(a, size)[0].get_nominal_glyph(ord(c)) for c in text)), None)
        if asset is None:
            raise ValidationError(f"No supplied font covers the complete {script} run: {text!r}")
        glyphs, advance = _shape(text, asset, size, script, level)
        result.append((asset, pen, glyphs))
        pen += advance
    return result, pen


def _breaks(text):
    # Only the bundled dictionary is used; no corpus/model downloads.
    if any('\u0e00' <= c <= '\u0e7f' for c in text):
        import os
        os.environ.setdefault('PYTHAINLP_READ_ONLY', '1')
        os.environ.setdefault('PYTHAINLP_OFFLINE', '1')
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError as exc:
            raise ValidationError('Thai wrapping requires pydrawcv[typography]') from exc
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=True)
    else:
        tokens = regex.findall(r' +|[^ ]+', text)
    ends, pos = [], 0
    for token in tokens:
        pos += len(token)
        # A no-break space glues both adjacent tokens, including Thai tokens.
        if pos == len(text) or (text[pos-1] != '\u00a0' and text[pos] != '\u00a0'):
            ends.append(pos)
    return ends


def _lines(text, direction, width, fonts, size):
    output = []
    for paragraph in text.replace('\r\n', '\n').split('\n'):
        storage = _resolve(paragraph, direction)
        if width is None or not paragraph:
            runs, advance = _runs(storage, 0, len(paragraph), fonts, size)
            output.append((paragraph, runs, advance))
            continue
        start = 0
        ends = _breaks(paragraph)
        while start < len(paragraph):
            choice = None
            for end in ends:
                if end <= start:
                    continue
                trimmed = len(paragraph[:end].rstrip(' '))
                runs, advance = _runs(storage, start, max(start, trimmed), fonts, size)
                if advance > width:
                    break
                choice = (end, trimmed, runs, advance)
            if choice is None:
                raise ValidationError('A text word exceeds wrap_width; enlarge the box or insert a line break')
            end, trimmed, runs, advance = choice
            output.append((paragraph[start:trimmed], runs, advance))
            start = end
            while start < len(paragraph) and paragraph[start] == ' ':
                start += 1
    return output


def _union(tiles):
    if not tiles:
        return None
    left = min(t[0] for t in tiles)
    top = min(t[1] for t in tiles)
    right = max(t[0]+t[2].shape[1] for t in tiles)
    bottom = max(t[1]+t[2].shape[0] for t in tiles)
    return BoundingBox(left, top, right-left, bottom-top)


def _make_layout(text, fonts, size, width, alignment, direction, spacing):
    lines = _lines(text, direction, width, fonts, size)
    extents = [_faces(a, size)[0].get_font_extents('ltr') for a in fonts]
    ascent = max(e.ascender/64 for e in extents)
    descent = max(-e.descender/64 for e in extents)
    # Include actual outline overhang vertically in the common line metrics.
    for _, runs, _ in lines:
        for asset, _, glyphs in runs:
            font = _faces(asset, size)[0]
            for gid, x, y in glyphs:
                e = font.get_glyph_extents(gid)
                if e:
                    ascent = max(ascent, e.y_bearing/64-y)
                    descent = max(descent, y-(e.y_bearing+e.height)/64)
    ascent, descent = math.ceil(ascent), math.ceil(descent)
    line_height = ascent+descent
    paragraph_width = width if width is not None else max((line[2] for line in lines), default=0)
    height = line_height+(len(lines)-1)*line_height*spacing
    if text.strip() and paragraph_width*height > _MAX_PIXELS:
        raise ValidationError('Text layout exceeds the 16M pixel limit')
    all_tiles, metrics = [], []
    for index, (content, runs, advance) in enumerate(lines):
        baseline = ascent+index*line_height*spacing
        offset = {'left': 0, 'center': (paragraph_width-advance)/2, 'right': paragraph_width-advance}[alignment]
        tiles = []
        for asset, pen, glyphs in runs:
            for gid, x, y in glyphs:
                tile = _glyph(asset, size, gid, x+pen+offset, y+baseline)
                if tile is not None:
                    tiles.append(tile)
        all_tiles.extend(tiles)
        metrics.append(TextLineMetrics(content, advance, baseline,
            BoundingBox(0, baseline-ascent, paragraph_width, line_height), _union(tiles)))
    ink = _union(all_tiles)
    if ink and ink.width*ink.height > _MAX_PIXELS:
        raise ValidationError('Text raster exceeds the 16M pixel limit')
    left, top = (int(ink.x), int(ink.y)) if ink else (0, 0)
    mask = np.zeros((int(ink.height), int(ink.width)) if ink else (1, 1), np.uint8)
    for x, y, tile in all_tiles:
        target = mask[y-top:y-top+tile.shape[0], x-left:x-left+tile.shape[1]]
        a = tile.astype(np.float32)/255
        target[:] = np.rint(tile+target*(1-a)).clip(0, 255).astype(np.uint8)
    # A glyph bitmap may contain a blank border; report actual nonzero pixels.
    ys, xs = np.where(mask > 0)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        mask = mask[y0:y1, x0:x1].copy()
        left, top = left+int(x0), top+int(y0)
        ink = BoundingBox(left, top, mask.shape[1], mask.shape[0])
    else:
        ink = None
    mask.flags.writeable = False
    return Layout(paragraph_width, height,
                  ascent, descent, tuple(metrics), ink, mask, left, top)


def layout_text(text, fonts, font_size, width, alignment, direction, spacing):
    global _paragraph_bytes, _hits
    if len(text) > 20000:
        raise ValidationError('Font text is limited to 20,000 characters per object')
    key = text, fonts, max(1, round(font_size*64)), width, alignment, direction, spacing
    with _lock:
        if key in _paragraphs:
            _hits += 1
            _paragraphs.move_to_end(key)
            return _paragraphs[key]
        result = _make_layout(*key)
        while _paragraphs and (_paragraph_bytes+result.mask.nbytes > _MASK_BUDGET or len(_paragraphs) >= 32):
            _, old = _paragraphs.popitem(last=False)
            _paragraph_bytes -= old.mask.nbytes
        _paragraphs[key] = result
        _paragraph_bytes += result.mask.nbytes
        return result


def clear_text_cache():
    global _paragraph_bytes, _hits
    with _lock:
        _paragraphs.clear()
        _paragraph_bytes = _hits = 0
        for cache in (_faces, _shape_cached, _glyph_cached):
            cache.cache_clear()


def text_cache_info():
    with _lock:
        return dict(paragraphs=len(_paragraphs), mask_bytes=_paragraph_bytes, hits=_hits,
                    fonts=_faces.cache_info().currsize, runs=_shape_cached.cache_info().currsize,
                    glyphs=_glyph_cached.cache_info().currsize)


def _make_layout_runs(
    runs: tuple[Any, ...],
    default_fonts: tuple[FontAsset, ...] | None,
    default_font_size: float,
    wrap_width: float | None = None,
    alignment: str = "left",
    direction: str = "auto",
    line_spacing: float = 1.2,
    text_origin: str = "top_left",
    text_anchor: str | None = None,
    base_x: float = 0.0,
    base_y: float = 0.0,
    root_fill_opacity: float = 1.0,
    root_fill_none: bool = False,
) -> Layout:
    if not runs:
        return Layout(0.0, 0.0, 0.0, 0.0, (), None, np.zeros((1, 1), np.uint8), 0, 0)

    total_chars = sum(len(r.text) for r in runs)
    if total_chars > 20000:
        raise ValidationError('Font text is limited to 20,000 characters per object')

    def _can_coalesce(a, b):
        return (
            b.x is None and b.y is None and b.dx == 0.0 and b.dy == 0.0
            and a.fonts == b.fonts
            and a.font_size == b.font_size
            and a.fill == b.fill
            and a.fill_none == b.fill_none
            and a.fill_opacity == b.fill_opacity
            and a.text_anchor == b.text_anchor
            and a.direction == b.direction
            and a.xml_space == b.xml_space
            and a.font_family_name == b.font_family_name
            and a.font_weight == b.font_weight
            and a.font_style == b.font_style
            and a.is_font_substituted == b.is_font_substituted
        )

    from drawcv.shapes.text import TextRun
    coalesced: list[TextRun] = []
    for r in runs:
        if not r.text:
            continue
        if coalesced and _can_coalesce(coalesced[-1], r):
            prev = coalesced[-1]
            coalesced[-1] = TextRun(
                text=prev.text + r.text,
                fonts=prev.fonts,
                font_size=prev.font_size,
                font_family_name=prev.font_family_name,
                font_weight=prev.font_weight,
                font_style=prev.font_style,
                fill=prev.fill,
                fill_none=prev.fill_none,
                fill_opacity=prev.fill_opacity,
                x=prev.x,
                y=prev.y,
                dx=prev.dx,
                dy=prev.dy,
                text_anchor=prev.text_anchor,
                direction=prev.direction,
                xml_space=prev.xml_space,
                is_font_substituted=prev.is_font_substituted,
            )
        else:
            coalesced.append(r)

    if not coalesced:
        return Layout(0.0, 0.0, 0.0, 0.0, (), None, np.zeros((1, 1), np.uint8), 0, 0)

    chunks: list[list[TextRun]] = []
    cur_chunk = []
    for r in coalesced:
        has_new_pos = (r.x is not None or r.y is not None)
        if has_new_pos and cur_chunk:
            chunks.append(cur_chunk)
            cur_chunk = [r]
        else:
            cur_chunk.append(r)
    if cur_chunk:
        chunks.append(cur_chunk)

    all_fonts_set = set(default_fonts or ())
    for r in coalesced:
        if r.fonts:
            all_fonts_set.update(r.fonts)
    if not all_fonts_set:
        raise ValidationError("No fonts provided for text runs")
    all_fonts = tuple(all_fonts_set)
    base_font_size_int = max(1, round(default_font_size * 64))
    extents = [_faces(a, base_font_size_int)[0].get_font_extents('ltr') for a in all_fonts]
    ascent = max((e.ascender / 64 for e in extents), default=math.ceil(default_font_size * 0.8))
    descent = max((-e.descender / 64 for e in extents), default=math.ceil(default_font_size * 0.2))

    all_tiles = []
    run_tiles: dict[int, list[tuple]] = {i: [] for i in range(len(coalesced))}
    ctp_x, ctp_y = 0.0, 0.0
    run_counter = 0

    lines_metrics = []

    for chunk in chunks:
        r0 = chunk[0]
        chunk_x = (r0.x - base_x) if r0.x is not None else ctp_x
        chunk_y = (r0.y - base_y) if r0.y is not None else ctp_y
        chunk_x += r0.dx
        chunk_y += r0.dy

        chunk_anchor = r0.text_anchor.value if r0.text_anchor is not None else (text_anchor if text_anchor is not None else "start")
        chunk_dir = r0.direction if r0.direction is not None else direction

        chunk_text = "".join(r.text for r in chunk)

        # Partition chunk into direction segments of contiguous runs with the same effective direction
        dir_segments: list[list[tuple[int, TextRun]]] = []
        cur_seg: list[tuple[int, TextRun]] = []
        cur_eff_dir = None
        for i_cr, cr in enumerate(chunk):
            cr_eff_dir = cr.direction if cr.direction is not None else direction
            if cur_eff_dir is not None and cr_eff_dir != cur_eff_dir:
                dir_segments.append(cur_seg)
                cur_seg = [(i_cr, cr)]
                cur_eff_dir = cr_eff_dir
            else:
                cur_seg.append((i_cr, cr))
                cur_eff_dir = cr_eff_dir
        if cur_seg:
            dir_segments.append(cur_seg)

        chunk_positioned_glyphs = []
        pen_x, pen_y = 0.0, 0.0
        applied_run_offsets = set()

        for seg_runs in dir_segments:
            seg_dir = seg_runs[0][1].direction if seg_runs[0][1].direction is not None else direction
            seg_text = "".join(cr.text for _, cr in seg_runs)
            seg_codepoints = [ord(c) for c in seg_text]

            storage = _resolve(seg_text, seg_dir)
            chars = storage['chars']

            char_to_chunk_run = []
            for i_cr, cr in seg_runs:
                for _ in range(len(cr.text)):
                    char_to_chunk_run.append(i_cr)

            line = storage['paragraph'].setLine(0, storage['units'])
            levels = line.getLevels()
            unit_ranks = {logical: visual for visual, logical in enumerate(line.getVisualMap())}
            ranks = {}
            for char in chars:
                char['level'] = levels[char['unit']]
                ranks[char['index']] = unit_ranks[char['unit']]

            subspans = []
            for char in chars:
                cr_idx = char_to_chunk_run[char['index']]
                key = (char['script'], char['level'], cr_idx)
                if not subspans or subspans[-1]['key'] != key:
                    subspans.append({'key': key, 'script': char['script'], 'level': char['level'],
                                     'cr_idx': cr_idx, 'chars': [char]})
                else:
                    subspans[-1]['chars'].append(char)

            subspans.sort(key=lambda s: min(ranks[c['index']] for c in s['chars']))

            for subspan in subspans:
                cr_idx = subspan['cr_idx']
                run_obj = chunk[cr_idx]
                global_idx = run_counter + cr_idx

                if cr_idx > 0 and cr_idx not in applied_run_offsets:
                    pen_x += run_obj.dx
                    pen_y += run_obj.dy
                    applied_run_offsets.add(cr_idx)

                run_fonts = run_obj.fonts or default_fonts or all_fonts
                run_size = run_obj.font_size or default_font_size
                run_size_int = max(1, round(run_size * 64))

                span_text = "".join(c['ch'] for c in subspan['chars'])
                asset = next((a for a in run_fonts if all(_faces(a, run_size_int)[0].get_nominal_glyph(ord(ch)) for ch in span_text)), None)
                if asset is None:
                    raise ValidationError(f"No supplied font covers the run: {span_text!r}")

                item_offset = subspan['chars'][0]['index']
                item_length = len(subspan['chars'])

                if len(seg_runs) == 1:
                    glyphs, advance = _shape(span_text, asset, run_size_int, subspan['script'], subspan['level'])
                else:
                    glyphs, advance = _shape_subrange(seg_codepoints, item_offset, item_length,
                                                      asset, run_size_int, subspan['script'], subspan['level'])

                for g in glyphs:
                    chunk_positioned_glyphs.append((asset, run_size_int, g.gid, pen_x + g.x, pen_y + g.y, global_idx))
                pen_x += advance

        chunk_advance = pen_x


        if chunk_anchor == "middle":
            delta_x = -chunk_advance / 2.0
        elif chunk_anchor == "end":
            delta_x = -chunk_advance
        else:  # "start"
            delta_x = 0.0

        base_line_y = chunk_y if text_origin == "baseline" else (ascent + chunk_y)

        chunk_tiles = []
        for asset, size_int, gid, gx, gy, g_run_idx in chunk_positioned_glyphs:
            final_x = chunk_x + delta_x + gx
            final_y = base_line_y + gy
            tile = _glyph(asset, size_int, gid, final_x, final_y)
            if tile is not None:
                all_tiles.append(tile)
                chunk_tiles.append(tile)
                run_tiles[g_run_idx].append(tile)

        chunk_ink = _union(chunk_tiles)
        lines_metrics.append(TextLineMetrics(
            chunk_text, chunk_advance, base_line_y,
            BoundingBox(chunk_x + delta_x, base_line_y - ascent, chunk_advance, ascent + descent),
            chunk_ink
        ))

        ctp_x = chunk_x + delta_x + chunk_advance
        ctp_y = chunk_y
        run_counter += len(chunk)

    ink = _union(all_tiles)
    if ink and ink.width * ink.height > _MAX_PIXELS:
        raise ValidationError('Text raster exceeds the 16M pixel limit')

    left, top = (int(ink.x), int(ink.y)) if ink else (0, 0)
    mask = np.zeros((int(ink.height), int(ink.width)) if ink else (1, 1), np.uint8)
    for x, y, tile in all_tiles:
        target = mask[y - top:y - top + tile.shape[0], x - left:x - left + tile.shape[1]]
        a = tile.astype(np.float32) / 255
        target[:] = np.rint(tile + target * (1 - a)).clip(0, 255).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        mask = mask[y0:y1, x0:x1].copy()
        left, top = left + int(x0), top + int(y0)
        ink = BoundingBox(left, top, mask.shape[1], mask.shape[0])
    else:
        ink = None
    mask.flags.writeable = False

    has_styled_runs = any(
        r.fill is not None
        or r.fill_none
        or (r.fill_opacity is not None and not math.isclose(r.fill_opacity, root_fill_opacity))
        for r in coalesced
    )
    styled_runs = None
    if has_styled_runs:
        sruns = []
        for i_run, r in enumerate(coalesced):
            r_tiles = run_tiles[i_run]
            if not r_tiles:
                continue
            r_ink = _union(r_tiles)
            if not r_ink:
                continue
            r_left, r_top = int(r_ink.x), int(r_ink.y)
            r_mask = np.zeros((int(r_ink.height), int(r_ink.width)), np.uint8)
            for x, y, tile in r_tiles:
                target = r_mask[y - r_top:y - r_top + tile.shape[0], x - r_left:x - r_left + tile.shape[1]]
                a = tile.astype(np.float32) / 255
                target[:] = np.rint(tile + target * (1 - a)).clip(0, 255).astype(np.uint8)
            r_mask.flags.writeable = False
            eff_fill_none = r.fill_none or (r.fill is None and root_fill_none)
            eff_fill_opacity = r.fill_opacity if r.fill_opacity is not None else root_fill_opacity
            sruns.append(StyledRunLayout(
                mask=r_mask,
                left=r_left,
                top=r_top,
                color=r.fill,
                fill_none=eff_fill_none,
                fill_opacity=eff_fill_opacity,
            ))
        styled_runs = tuple(sruns)

    total_width = ink.width if ink else max((lm.advance_width for lm in lines_metrics), default=0.0)
    total_height = ink.height if ink else (ascent + descent)

    return Layout(
        width=total_width,
        height=total_height,
        ascent=ascent,
        descent=descent,
        lines=tuple(lines_metrics),
        ink=ink,
        mask=mask,
        left=left,
        top=top,
        styled_runs=styled_runs,
    )


def layout_text_runs(
    runs: tuple[Any, ...],
    default_fonts: tuple[FontAsset, ...] | None,
    default_font_size: float,
    wrap_width: float | None = None,
    alignment: str = "left",
    direction: str = "auto",
    line_spacing: float = 1.2,
    text_origin: str = "top_left",
    text_anchor: str | None = None,
    base_x: float = 0.0,
    base_y: float = 0.0,
    root_fill_opacity: float = 1.0,
    root_fill_none: bool = False,
) -> Layout:
    return _make_layout_runs(
        runs,
        default_fonts,
        default_font_size,
        wrap_width,
        alignment,
        direction,
        line_spacing,
        text_origin,
        text_anchor,
        base_x,
        base_y,
        root_fill_opacity=root_fill_opacity,
        root_fill_none=root_fill_none,
    )
