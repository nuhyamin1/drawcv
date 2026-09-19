"""Review probes for 0.9.1, updated with the strict paint-order rejection contract.

Run explicitly: python -m pytest -q docs/audits/svg_review_repros.py
Requires the project's dev/typography dependencies and resvg-py.
Kept outside tests/ so a review does not change the normal regression suite.
"""
from pathlib import Path as FilePath

import cv2
import numpy as np
import pytest

from drawcv import (
    Color, DictFontResolver, FillStyle, FontAsset, GradientStop, LinearGradient,
    OpenCVRenderer, Path, Point, SVGImporter, Scene, Transform,
)


def reference(svg, font_dirs=()):
    import resvg_py
    png = resvg_py.svg_to_bytes(
        svg_string=svg, skip_system_fonts=True, font_dirs=list(font_dirs),
    )
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize('root_attrs,body,point', [
    ('', '<path d="M50 5 L76 86 L7 36 L93 36 L24 86 Z" fill="red" fill-rule="nonzero"/>', (50, 50)),
    ('opacity="0.5"', '<rect width="100" height="100" fill="red"/>', (50, 50)),
    ('transform="translate(40 40)"', '<rect width="20" height="20" fill="red"/>', (10, 10)),
], ids=['nonzero-star', 'root-opacity', 'root-transform'])
def test_import_preserves_interior_pixels(root_attrs, body, point):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" {root_attrs}>{body}</svg>'
    scene = SVGImporter().parse(svg).scene
    scene.to_svg(strict=True)  # All examples also pass strict export.
    actual = OpenCVRenderer().render(scene, alpha=True).buffer
    expected = reference(svg)
    x, y = point
    assert actual[y, x].tolist() == expected[y, x].tolist()


def test_unsupported_paint_order_is_rejected():
    from drawcv import SVGImportError
    svg = ('<svg width="100" height="100"><rect x="20" y="20" width="60" height="60" '
           'fill="red" stroke="blue" stroke-width="20" paint-order="stroke fill"/></svg>')
    with pytest.raises(SVGImportError) as error:
        SVGImporter().parse(svg)
    assert error.value.diagnostic.code == 'SVG_UNSUPPORTED_PAINT_ORDER'


def test_transformed_text_keeps_clip_position_on_export():
    fonts = FilePath(__file__).resolve().parents[2] / 'tests' / 'assets' / 'fonts'
    asset = FontAsset.from_file(fonts / 'notosans.ttf')
    resolver = DictFontResolver(default_fonts=(asset,))
    resolver.register('Noto Sans', (asset,))
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="240" height="100">
      <defs><clipPath id="c"><rect width="45" height="100"/></clipPath></defs>
      <text transform="translate(80 0)" clip-path="url(#c)" x="0" y="60"
            font-family="Noto Sans" font-size="40">HELLO</text></svg>'''
    scene = SVGImporter(font_resolver=resolver).parse(svg).scene
    original = reference(svg, [str(fonts)])
    exported = reference(scene.to_svg(strict=True), [str(fonts)])
    original_x = np.nonzero(original[..., 3])[1]
    exported_x = np.nonzero(exported[..., 3])[1]
    assert len(original_x) and len(exported_x)
    assert abs(int(original_x.min()) - int(exported_x.min())) <= 1


@pytest.mark.parametrize('variant', ['repeat', 'paint-transform'])
def test_disjoint_difference_preserves_gradient(variant):
    stops = (GradientStop(0, Color(255, 0, 0)), GradientStop(1, Color(0, 0, 255)))
    paint = LinearGradient(Point(10, 0), Point(30, 0), stops,
                           spread='repeat' if variant == 'repeat' else 'pad')
    if variant == 'paint-transform':
        paint.transform = Transform.from_matrix(np.array([[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]]))
    left = Path(fill=FillStyle(paint=paint), stroke=None)
    left.move_to(10, 10).line_to(90, 10).line_to(90, 90).line_to(10, 90).close()
    other = Path(stroke=None)
    other.move_to(110, 110).line_to(120, 110).line_to(120, 120).line_to(110, 120).close()
    result = left.difference(other)
    images = []
    for node in (left, result):
        scene = Scene(100, 100, background=Color(0, 0, 0, 0))
        scene.add(node)
        images.append(OpenCVRenderer().render(scene, alpha=True).buffer)
    assert images[0][35, 35].tolist() == images[1][35, 35].tolist()
