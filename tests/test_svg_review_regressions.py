"""Appearance regressions discovered in the 0.9.1 SVG fidelity review."""
import cv2
import numpy as np
import pytest

from drawcv import OpenCVRenderer, SVGImporter, Scene


def reference(svg, **kwargs):
    resvg = pytest.importorskip('resvg_py')
    png = resvg.svg_to_bytes(svg_string=svg, **kwargs)
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize('attrs,body,points', [
    ('opacity="0.5"', '<rect width="70" height="100" fill="red"/><rect x="30" width="70" height="100" fill="blue"/>', [(10, 50), (50, 50), (90, 50)]),
    ('transform="translate(40 40)"', '<rect width="20" height="20" fill="red"/>', [(10, 10), (50, 50)]),
    ('transform="scale(2)"', '<path d="M10 20H40" stroke="red" stroke-width="10"/>', [(40, 47), (40, 54)]),
    ('display="none"', '<rect width="100" height="100" fill="red"/>', [(50, 50)]),
    ('viewBox="0 0 50 50" transform="translate(10 0)" clip-path="url(#c)" opacity="0.5"', '<defs><clipPath id="c"><rect width="20" height="50"/></clipPath></defs><rect width="50" height="50" fill="red"/>', [(30, 50), (70, 50)]),
], ids=['root-opacity-overlap', 'root-translation',
        'root-scaled-stroke', 'root-display', 'root-clip-viewbox'])
def test_root_properties_preserve_appearance(attrs, body, points):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" {attrs}>{body}</svg>'
    scene = SVGImporter().parse(svg).scene
    expected = reference(svg)
    frames = [OpenCVRenderer().render(scene, alpha=True).buffer,
              OpenCVRenderer().render(Scene.from_json(scene.to_json()), alpha=True).buffer,
              reference(scene.to_svg(strict=True))]
    for frame in frames:
        for x, y in points:
            assert np.max(np.abs(frame[y, x].astype(int) - expected[y, x].astype(int))) <= 1


def test_root_viewbox_transform_order():
    # SVG2 coords section 8.6 and Chrome: root_transform @ viewBox.
    # resvg 0.5.0 reverses this order on the root, so use equivalent groups
    # for the raster oracle instead of adopting that engine's discrepancy.
    from drawcv import Point, Rectangle
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
           'viewBox="0 0 50 50" transform="translate(20 10)">'
           '<rect x="5" y="5" width="10" height="10" fill="red"/></svg>')
    expected_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
                    '<g transform="translate(20 10)"><g transform="scale(2)">'
                    '<rect x="5" y="5" width="10" height="10" fill="red"/></g></g></svg>')
    scene = SVGImporter().parse(svg).scene
    assert scene.find_by_type(Rectangle)[0].to_world(Point(5, 5)) == Point(30, 20)
    expected = reference(expected_svg)
    for image in (OpenCVRenderer().render(scene, alpha=True).buffer, reference(scene.to_svg(strict=True))):
        for x, y in [(15, 15), (40, 30), (60, 40)]:
            assert image[y, x].tolist() == expected[y, x].tolist()


@pytest.mark.parametrize('transform', ['translate(80 0)', 'translate(80 0) rotate(12)',
                                      'translate(80 0) scale(1.5 0.8)'])
@pytest.mark.parametrize('clip_geometry', ['<rect width="45" height="100"/>',
                                         '<path d="M0 0H45V100H0Z"/>'])
def test_transformed_text_clip_export(transform, clip_geometry):
    pytest.importorskip("uharfbuzz")
    pytest.importorskip("freetype")
    pytest.importorskip("icu")
    pytest.importorskip("regex")
    from pathlib import Path
    from drawcv import DictFontResolver, FontAsset
    fonts = Path(__file__).parent / 'assets' / 'fonts'
    asset = FontAsset.from_file(fonts / 'notosans.ttf')
    resolver = DictFontResolver(default_fonts=(asset,))
    resolver.register('Noto Sans', (asset,))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="150">'
           f'<defs><clipPath id="c">{clip_geometry}</clipPath></defs>'
           '<g transform="translate(10 5)" opacity="0.7">'
           f'<text transform="{transform}" clip-path="url(#c)" x="0" y="60" '
           'font-family="Noto Sans" font-size="40">HELLO</text></g></svg>')
    scene = SVGImporter(font_resolver=resolver).parse(svg).scene
    options = dict(skip_system_fonts=True, font_dirs=[str(fonts.resolve())])
    original = reference(svg, **options)
    exported = reference(scene.to_svg(strict=True), **options)
    assert np.count_nonzero(original[..., 3]) > 100
    assert np.max(np.abs(original.astype(int) - exported.astype(int))) <= 1


@pytest.mark.parametrize('attribute', ['paint-order="stroke fill"',
                                      'style="paint-order:stroke fill"',
                                      'paint-order="normal" style="paint-order:stroke"'])
def test_unsupported_paint_order_is_explicit(attribute):
    from drawcv import SVGImportError
    svg = f'<svg width="100" height="100"><rect width="80" height="80" {attribute}/></svg>'
    with pytest.raises(SVGImportError) as error:
        SVGImporter().parse(svg)
    assert error.value.diagnostic.code == 'SVG_UNSUPPORTED_PAINT_ORDER'


@pytest.mark.parametrize('value', ['normal', 'fill', 'fill stroke', 'fill stroke markers'])
def test_normal_paint_order_is_accepted(value):
    svg = f'<svg width="100" height="100"><rect width="80" height="80" paint-order="{value}"/></svg>'
    assert SVGImporter().parse(svg).scene.to_svg(strict=True)
