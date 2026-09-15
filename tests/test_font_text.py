"""Retained typography contracts; optional native dependencies required."""
import json
import os
os.environ.setdefault('PYTHAINLP_READ_ONLY', '1')
os.environ.setdefault('PYTHAINLP_OFFLINE', '1')
import numpy as np
import pytest
pytest.importorskip('uharfbuzz')
pytest.importorskip('freetype')
pytest.importorskip('icu')
pytest.importorskip('regex')
from drawcv import (FontAsset, Text, Point, Color, Scene, OpenCVRenderer, Group,
    Transform, TextAlignment, FillStyle, ClipRect, Mask, MaskMapping, BlurEffect,
    ShadowEffect, ValidationError, Canvas, CURRENT_SCHEMA_VERSION)
from drawcv.typography.layout import clear_text_cache, text_cache_info, _resolve, _runs
from pathlib import Path

ROOT = Path(__file__).parent / 'assets/fonts'
LATIN = FontAsset.from_file(ROOT / 'notosans.ttf')
THAI = FontAsset.from_file(ROOT / 'notosansthai.ttf')
ARABIC = FontAsset.from_file(ROOT / 'notosansarabic.ttf')
OTF = FontAsset.from_file(ROOT / 'sourcesans3.otf')


def scene_with(obj):
    s = Scene(700, 350, background=Color(0, 0, 0, 0))
    s.add(obj)
    return s


def render(obj):
    return OpenCVRenderer().render(scene_with(obj.clone()), alpha=True).buffer


@pytest.mark.parametrize('content,font', [('AVATAR j Ág ', LATIN), ('น้ำ กุ้ง กิ๊ง ปู่ ผู้', THAI),
                                        ('السلام عليكم', ARABIC), ('j Ág office', OTF)])
def test_measured_ink_is_actual_coverage(content, font):
    t = Text(content, fonts=[font], font_size=36, position=Point(40, 50))
    m = t.measure()
    data = render(t)
    y, x = np.where(data[..., 3] > 0)
    assert (x.min(), y.min(), x.max()+1, y.max()+1) == (m.ink_bounds.left, m.ink_bounds.top, m.ink_bounds.right, m.ink_bounds.bottom)
    assert m.lines[0].ink_bounds == m.ink_bounds
    assert t.contains_point(Point(int(x.min()), int(y.min())))


@pytest.mark.parametrize('alignment', list(TextAlignment))
def test_paragraph_anchor_alignment_advance_and_hit_box(alignment):
    t = Text('A \nlonger', fonts=[LATIN], wrap_width=250, alignment=alignment,
             position=Point(300, 50), padding=10)
    m = t.measure()
    assert m.layout_bounds.width == 250 and m.paragraph_bounds.width == 270
    assert m.advance_width < 250
    if alignment == TextAlignment.LEFT:
        assert m.paragraph_bounds.left == 300
    elif alignment == TextAlignment.CENTER:
        assert m.paragraph_bounds.center.x == 300
    else:
        assert m.paragraph_bounds.right == 300
    assert t.contains_point(m.paragraph_bounds.bottom_right)
    assert not t.contains_point(Point(m.paragraph_bounds.right+20, 50))
    assert m.lines[0].baseline < m.lines[1].baseline


def test_mixed_font_lines_share_metrics_and_baselines():
    t = Text('English\nกิ๊ง ไทย\nالعربية 123', fonts=[LATIN, THAI, ARABIC], line_spacing=1.5)
    m = t.measure()
    assert len({l.line_box.height for l in m.lines}) == 1
    h = m.lines[0].line_box.height
    assert np.allclose(np.diff([l.baseline for l in m.lines]), 1.5*h)
    for line in m.lines:
        assert line.line_box.top <= line.ink_bounds.top
        assert line.line_box.bottom >= line.ink_bounds.bottom


@pytest.mark.parametrize('content,font', [('DrawCV العربية 123', ARABIC), ('العربية DrawCV 123', ARABIC),
    ('ABC (العربية) 123', ARABIC), ('العربية [ABC 123] العربية', ARABIC), ('ภาษาไทย DrawCV กิ๊ง 123', THAI)])
def test_mixed_script_bidi_positions_match_qt(content, font):
    pytest.importorskip('PySide6.QtGui')
    from examples._typography_qt import layout_reference
    q = layout_reference(content, [ROOT/font.name], 36, 900)
    runs, advance = _runs(_resolve(content, 'auto'), 0, len(content), (font,), 36*64)
    ours = sorted((gid, x+pen, y) for asset, pen, glyphs in runs for gid, x, y in glyphs)
    reference = sorted((gid, x, y-q['lines'][0]['baseline'])
                       for run in q['runs'] for gid, (x,y) in zip(run['glyphs'],run['positions']))
    assert [g[0] for g in ours] == [g[0] for g in reference]
    assert np.max(np.abs(np.array(ours)[:, 1:]-np.array(reference)[:, 1:])) < .5
    assert abs(advance-q['lines'][0]['width']) < .5


def test_fallback_order_and_missing_run_error():
    t = Text('ABC ภาษาไทย ABC', fonts=[LATIN, THAI])
    data = render(t)
    assert data[..., 3].any()
    runs, _ = _runs(_resolve(t.text, 'auto'), 0, len(t.text), t.fonts, 32*64)
    assert [a for a, pen, gs in runs] == [LATIN, THAI, LATIN]
    t.fonts = [THAI, LATIN]
    runs, _ = _runs(_resolve(t.text, 'auto'), 0, len(t.text), t.fonts, 32*64)
    assert all(a == THAI for a, pen, gs in runs)
    t.fonts = [LATIN]
    with pytest.raises(ValidationError, match='covers'):
        t.measure()


@pytest.mark.parametrize('content', ['中文', '😀', 'A\tB', 'A\u2067B\u2069', 'ا\u200dب', 'A\ufe0f'])
def test_unsupported_inputs_fail_explicitly(content):
    with pytest.raises(ValidationError):
        Text(content, fonts=[LATIN, THAI, ARABIC]).measure()


def test_thai_wrap_crlf_blank_lines_and_overflow():
    pytest.importorskip('pythainlp')
    t = Text('ภาษาไทยต้องตัดคำอย่างถูกต้อง เพื่อให้อ่านข้อความได้ง่ายขึ้น\r\n\nABC',
             fonts=[LATIN, THAI], font_size=32, wrap_width=300)
    m = t.measure()
    assert all(l.advance_width <= 300 for l in m.lines)
    assert m.lines[-2].text == '' and m.lines[-1].text == 'ABC'
    assert ''.join(l.text for l in m.lines[:-2]).replace(' ','') == t.text.split('\r')[0].replace(' ','')
    t.wrap_width = 1
    with pytest.raises(ValidationError, match='word'):
        t.measure()


@pytest.mark.parametrize('field,value', [('font_size', 0), ('font_size', float('nan')), ('fonts', []),
                                      ('wrap_width', -1), ('line_spacing', .9), ('direction','sideways')])
def test_invalid_font_option_assignment_is_atomic(field, value):
    t = Text('ABC', fonts=[LATIN])
    old = t.to_dict()
    with pytest.raises(ValidationError):
        setattr(t, field, value)
    assert t.to_dict() == old


def test_cache_reuse_invalidation_and_readonly_measurements():
    clear_text_cache()
    t = Text('DrawCV กิ๊ง', fonts=[LATIN, THAI])
    a = t.measure()
    first = text_cache_info()
    t.color = Color.red()
    t.position = Point(30, 30)
    b = t.measure()
    second = text_cache_info()
    assert second['paragraphs'] == first['paragraphs'] == 1 and second['hits'] > first['hits']
    assert a.advance_width == b.advance_width and b.ink_bounds.x == a.ink_bounds.x+30
    t.font_size = 48
    assert t.measure().advance_width > a.advance_width
    t.text = 'changed'
    assert t.measure().lines[0].text == 'changed'
    assert text_cache_info()['paragraphs'] == 3
    assert not t._font_layout().mask.flags.writeable
    for i in range(35):
        t.text = f'cache {i}'
        t.measure()
    assert text_cache_info()['paragraphs'] == 32
    assert text_cache_info()['mask_bytes'] <= 32*1024*1024


def test_font_json_is_portable_after_source_removal(tmp_path):
    path = tmp_path / 'font.otf'
    path.write_bytes(OTF.data)
    t = Text('Portable Ág', fonts=[FontAsset.from_file(path)])
    s = scene_with(t)
    original = render(t)
    encoded = s.to_json()
    path.unlink()
    loaded = Scene.from_json(encoded)
    assert loaded.to_dict()['version'] == CURRENT_SCHEMA_VERSION
    assert np.array_equal(original, OpenCVRenderer().render(loaded, alpha=True).buffer)
    font = t.fonts[0].to_dict()
    font['sha256'] = '0'*64
    with pytest.raises(ValidationError):
        FontAsset.from_dict(font)


def test_clone_history_temporal_restore_and_mode_switch():
    t = Text('original', fonts=[OTF], font_size=24, wrap_width=300)
    s = scene_with(t)
    original = render(t)
    clone = t.clone()
    assert clone.fonts == t.fonts and clone.id != t.id
    with s.edit(t):
        t.text = 'changed'
        t.font_size = 40
        t.direction = 'rtl'
    edited = render(t)
    assert s.undo() and np.array_equal(render(t), original)
    assert s.redo() and np.array_equal(render(t), edited)
    with s.edit(t):
        t.fonts = None
    assert s.undo() and t.fonts == (OTF,)
    s.animate(t, 'font_size', 20, 50, duration=1)
    saved = s.to_json()
    a = s.render_at_time(.2, alpha=True).buffer
    b = s.render_at_time(.8, alpha=True).buffer
    assert not np.array_equal(a,b)
    assert s.to_json() == saved and s.get(t.id) is t
    assert np.array_equal(a, Scene.from_json(saved).render_at_time(.2, alpha=True).buffer)
    s.animate(t, 'wrap_width', 300, .1, duration=1)
    saved = s.to_json()
    with pytest.raises(ValidationError):
        s.render_at_time(1, alpha=True)
    assert s.to_json() == saved and s.get(t.id) is t


def test_transforms_masks_effects_export_and_direct_canvas(tmp_path):
    import cv2
    t = Text('น้ำ กุ้ง 123', fonts=[THAI], position=Point(60,60), color=Color(255,0,0,.7),
             background_fill=FillStyle(color=Color(255,0,0,.1)), padding=10,
             transform=Transform(rotation=-12, scale_x=1.2),
             effects=[BlurEffect(kernel_size=5, sigma=1), ShadowEffect(color=Color(255,0,0,.5))],
             clip=ClipRect(0,0,600,300), mask=Mask(np.full((350,700),128,np.uint8), mapping=MaskMapping.ABSOLUTE))
    g = Group(children=[t], opacity=.6, transform=Transform(translation_x=40,translation_y=40))
    s = scene_with(g)
    renderer = OpenCVRenderer()
    canvas = renderer.render(s, alpha=True)
    canvas.save(tmp_path/'text.png')
    decoded = cv2.imread(str(tmp_path/'text.png'),cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded,canvas.buffer)
    visible = decoded[...,3] > 0
    assert visible.any() and np.all(decoded[visible,:3] == (0,0,255))
    alpha = decoded[...,3:4]/255
    for bg in [Color.white(),Color.black(),Color(70,180,120)]:
        expected = np.rint(decoded[...,:3]*alpha+np.array(bg.to_bgr())*(1-alpha))
        s.background = bg
        opaque = renderer.render(s).buffer
        assert np.abs(expected-opaque.astype(float)).max() <= 1
        target = Canvas(700,350)
        target.clear(bg)
        renderer.render_drawable(g,target)
        assert np.array_equal(target.buffer,opaque)


@pytest.mark.parametrize('matrix', [np.array([[1, .3, 90],[.2,1,40],[0,0,1.]]),
    np.array([[0,-1,300],[1.3,0,20],[0,0,1.]]), np.array([[.3,0,80],[0,.6,80],[0,0,1.]])])
def test_transformed_text_matches_warp_of_measured_mask(matrix):
    import cv2
    t = Text('j น้ำ Ág', fonts=[LATIN,THAI], font_size=40, position=Point(20,30))
    local = render(t)[...,3]
    t.transform = Transform.from_matrix(matrix)
    actual = render(t)[...,3]
    expected = cv2.warpAffine(local, matrix[:2], (700,350), flags=cv2.INTER_LINEAR)
    assert np.array_equal(actual,expected)
    y,x = np.where(actual>0)
    b=t.get_bounds()
    assert x.min() >= b.left-1 and x.max() <= b.right+1
    assert y.min() >= b.top-1 and y.max() <= b.bottom+1


def test_empty_space_metrics_and_inactive_font_options_roundtrip():
    empty=Text('',fonts=[LATIN])
    assert empty.measure().ink_bounds is None
    assert empty.measure().layout_bounds.width == 0
    assert empty.measure().layout_bounds.height > 0
    spaced=Text('A ',fonts=[LATIN])
    assert spaced.measure().advance_width > Text('A',fonts=[LATIN]).measure().advance_width
    spaced.fonts=None
    spaced.font_size=48
    loaded=Text.from_dict(spaced.to_dict())
    assert loaded.fonts is None and loaded.font_size == 48
    with pytest.raises(ValidationError, match='word'):
        Text('A\u00a0B',fonts=[LATIN],font_size=36,wrap_width=30).measure()
    with pytest.raises(ValidationError, match='pixel limit'):
        Text('Too large',fonts=[LATIN],font_size=4096).measure()


def test_optional_engines_are_not_needed_for_hershey():
    import subprocess,sys
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'uharfbuzz','freetype','icu','regex','pythainlp'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0,Block())
from drawcv import *
s=Scene(100,100); s.add(Text('old API'))
assert OpenCVRenderer().render(s).buffer.shape == (100,100,3)
t=Text('font API',fonts=[FontAsset.from_file('tests/assets/fonts/notosans.ttf')])
try:
    t.measure()
except ValidationError as e:
    assert 'typography' in str(e)
else:
    raise AssertionError('missing dependency should be reported')
'''
    code = 'import sys; sys.path = ' + repr(sys.path) + '\n' + code
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT.parents[2],capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
