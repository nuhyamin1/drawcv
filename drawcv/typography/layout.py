"""Optional horizontal paragraph layout; native engines are loaded lazily."""
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
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
    for info, pos in zip(buf.glyph_infos or [], buf.glyph_positions or []):
        if info.codepoint == 0:
            raise ValidationError(f"Missing shaped glyph in {asset.name}: {text!r}")
        glyphs.append((info.codepoint, pen+pos.x_offset/64, -pos.y_offset/64))
        pen += pos.x_advance/64
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
