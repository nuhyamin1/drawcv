"""Effect stacks export as SVG filters that compute the raster effects."""
import base64
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from drawcv import (BlurEffect, BlurType, BrightnessContrastEffect, Circle, ClipRect, Color, ColorMatrixEffect,
                    ConvolutionEffect, EmbossEffect, FillStyle, GlowEffect, GradientStop, GrayscaleEffect, Group,
                    HueShiftEffect, LinearGradient, NoiseEffect, OpenCVRenderer, Point, Rectangle,
                    SaturationEffect, Scene, SepiaEffect, ShadowEffect, SharpenEffect, StrokeStyle)
from drawcv.svg_filters import build_filter

W, H = 220, 160


def shape(effects=()):
    paint = LinearGradient(Point(60, 0), Point(150, 0),
                           (GradientStop(0, Color(230, 60, 40)), GradientStop(1, Color(40, 120, 220, 0.6))))
    return Rectangle(position=Point(60, 40), width=90, height=70, fill=FillStyle(paint=paint),
                     stroke=StrokeStyle(color=Color(20, 20, 20), width=4), effects=list(effects))


def render(drawable):
    scene = Scene(W, H, background=Color(0, 0, 0, 0))
    scene.add(drawable)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def filter_difference(effects):
    """Apply the exported filter in resvg to DrawCV's own unfiltered pixels.

    Both sides then start from identical input, so only the filter math is compared.
    """
    resvg = pytest.importorskip("resvg_py")
    base = render(shape())
    root = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": str(W), "height": str(H)})
    b = shape(effects).get_effect_bounds()
    x1, y1 = max(0, b.left - 3), max(0, b.top - 3)
    build_filter(ET.SubElement(root, "defs"), "f", effects, (x1, y1, min(W, b.right + 3) - x1, min(H, b.bottom + 3) - y1))
    group = ET.SubElement(root, "g", {"filter": "url(#f)"})
    png = base64.b64encode(cv2.imencode(".png", base)[1]).decode()
    ET.SubElement(group, "image", {"width": str(W), "height": str(H), "href": "data:image/png;base64," + png})
    other = cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=ET.tostring(root, encoding="unicode")), np.uint8),
                         cv2.IMREAD_UNCHANGED).astype(float)
    ours = render(shape(effects)).astype(float)
    # Premultiplied comparison: fully transparent pixels carry no color.
    color = np.abs(ours[..., :3] * ours[..., 3:] - other[..., :3] * other[..., 3:]).max(axis=2) / 255
    return np.maximum(color, np.abs(ours[..., 3] - other[..., 3]))


@pytest.mark.parametrize("effects,limit", [
    ([BlurEffect(kernel_size=21)], 3),
    ([BlurEffect(kernel_size=9, sigma=6)], 3),  # truncated kernel -> explicit weights
    ([BlurEffect(kernel_size=7, blur_type=BlurType.BOX)], 3),
    ([ShadowEffect()], 3),
    ([ShadowEffect(offset_x=6, offset_y=4, blur_radius=4, color=Color(0, 0, 120, 0.7), opacity=.8)], 10),
    ([GlowEffect(blur_radius=6, color=Color(255, 200, 0), opacity=.9)], 4),
    ([BrightnessContrastEffect(brightness=.1, contrast=1.4)], 4),
    ([SaturationEffect(factor=1.8)], 4),
    ([GrayscaleEffect(intensity=.7)], 3),
    ([SepiaEffect(intensity=1)], 3),
    ([HueShiftEffect(angle=120)], 4),
    ([ColorMatrixEffect(matrix=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (1, 0, 0, 0, .1), (0, 0, 0, 1, 0)))], 3),
    ([ConvolutionEffect(kernel=[[1, 2, 0], [0, 1, 0], [0, 0, 0]], factor=0.25)], 3),  # orientation-sensitive
    ([GlowEffect(blur_radius=5), SepiaEffect(intensity=.5), ShadowEffect(blur_radius=3)], 5),
], ids=["gaussian", "gaussian-truncated", "box", "shadow", "shadow-offset", "glow", "brightness-contrast",
        "saturation", "grayscale", "sepia", "hue", "color-matrix", "convolution", "chain"])
def test_filters_match_raster_effects_on_identical_input(effects, limit):
    difference = filter_difference(effects)
    assert difference.max() <= limit
    assert difference.mean() < 0.6


def test_sharpening_kernel_differs_only_at_faint_edges():
    difference = filter_difference([ConvolutionEffect(kernel=[[0, -1, 0], [-1, 5, -1], [0, -1, 0]])])
    assert (difference > 8).sum() <= 5 and difference.mean() < 0.6


@pytest.mark.parametrize("effect,reason", [
    (ConvolutionEffect(kernel=[[1]], bias=0.2), "effect convolution with bias"),
    (NoiseEffect(), "effect noise"), (SharpenEffect(), "effect sharpen"), (EmbossEffect(), "effect emboss"),
])
def test_effects_without_an_equivalent_keep_reported_fallbacks(effect, reason):
    scene = Scene(W, H)
    scene.add(shape([BlurEffect(), effect]))
    assert [f.reason for f in scene.export_svg().fallbacks] == [reason]


def test_strict_export_places_filter_before_clip_and_opacity():
    scene = Scene(W, H)
    drawable = shape([ShadowEffect(offset_x=30, offset_y=30, blur_radius=6)])
    drawable.opacity = 0.5
    drawable.clip = ClipRect(40, 20, 150, 100)
    scene.add(drawable)
    svg = scene.to_svg(strict=True)
    group = next(g for g in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}g")
                 if g.get("data-drawcv-id") == drawable.id)
    assert group.get("filter") and group.get("clip-path") and group.get("opacity") == "0.5"
    region = ET.fromstring(svg).find(".//{http://www.w3.org/2000/svg}filter")
    assert float(region.get("x")) >= 0 and float(region.get("y")) + float(region.get("height")) <= H
    # The shadow reaches beyond the default 10% filter region, so an explicit one is required.
    assert float(region.get("y")) + float(region.get("height")) > 110 + 30


def test_group_effects_and_no_op_stacks_stay_visible():
    scene = Scene(W, H, background=Color(0, 0, 0, 0))
    group = Group(children=[Circle(center=Point(100, 80), radius=30, fill=FillStyle(color=Color(0, 90, 200)))],
                  effects=[GlowEffect(blur_radius=0), GrayscaleEffect(intensity=0)])
    scene.add(group)
    svg = scene.to_svg(strict=True)
    assert "<feOffset" in svg  # pass-through keeps the group from vanishing
    resvg = pytest.importorskip("resvg_py")
    other = cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=svg), np.uint8), cv2.IMREAD_UNCHANGED)
    assert other[80, 100, 3] == 255


def test_export_is_deterministic():
    scene = Scene(W, H)
    scene.add(shape([GlowEffect(), ShadowEffect(), BlurEffect(kernel_size=5, sigma=4)]))
    assert scene.to_svg(strict=True) == scene.to_svg(strict=True)
