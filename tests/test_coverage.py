"""Area-exact coverage: pixel k is the square [k, k + 1], as in SVG."""
import itertools
import math

import numpy as np
import pytest

from drawcv import (CapStyle, Circle, Color, FillStyle, JoinStyle, Line, OpenCVRenderer, Path, Point, Polygon,
                    Rectangle, Scene, StrokeStyle)
from drawcv.core import coverage, stroking
from drawcv.core.coverage import coverage_mask
from drawcv.core.stroking import _exact_stroke_mask


def area(mask):
    return mask.astype(float).sum() / 255


def alpha(drawable, width=220, height=160):
    scene = Scene(width, height, background=Color(0, 0, 0, 0))
    scene.add(drawable)
    return OpenCVRenderer().render(scene, alpha=True).buffer[..., 3]


@pytest.mark.parametrize("contour,expected", [
    ([(30.25, 40.6), (130.25, 40.6), (130.25, 100.6), (30.25, 100.6)], 6000),
    ([(20, 20), (180, 40), (60, 170)], 11600),
    ([(10, 10), (10.25, 10), (10.25, 10.25), (10, 10.25)], 0.0625),
    ([(-50, -50), (50, -50), (50, 50), (-50, 50)], 2500),  # clipped to the canvas
])
def test_area_is_exact_in_either_orientation(contour, expected):
    for points in (contour, contour[::-1]):
        assert area(coverage_mask([points], 200, 200)) == pytest.approx(expected, rel=1e-3, abs=0.01)


def test_whole_pixel_edges_are_crisp():
    mask = coverage_mask([[(0, 0), (40, 0), (40, 30), (0, 30)]], 50, 40)
    assert mask[:30, :40].min() == 255 and mask[30:].max() == 0 and mask[:, 40:].max() == 0
    half = coverage_mask([[(10.5, 0), (20, 0), (20, 5), (10.5, 5)]], 30, 10)
    assert half[2, 10] == 128 and half[2, 11] == 255


def test_axis_aligned_fast_path_matches_general_rasterizer(monkeypatch):
    rng = np.random.default_rng(7)
    rectangles = []
    for _ in range(40):
        x, y = rng.uniform(-10, 90, 2)
        w, h = rng.uniform(0.2, 40, 2)
        rectangles.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    fast = [coverage_mask([r], 100, 80) for r in rectangles]
    monkeypatch.setattr(coverage, "_axis_aligned_rectangle", lambda points: False)
    for rectangle, expected in zip(rectangles, fast):
        general = coverage_mask([rectangle], 100, 80, subrows=64)
        assert np.abs(general.astype(int) - expected).max() <= 2


def test_fill_rules_and_unions():
    star = [(100 + 80 * math.cos(math.radians(-90 + 144 * i)), 100 + 80 * math.sin(math.radians(-90 + 144 * i)))
            for i in range(5)]
    assert coverage_mask([star], 200, 200)[100, 100] == 255
    assert coverage_mask([star], 200, 200, even_odd=True)[100, 100] == 0
    square = [(40, 40), (100, 40), (100, 100), (40, 100)]
    neighbour = [(100, 40), (160, 40), (160, 100), (100, 100)]
    joined = coverage_mask([square, neighbour], 200, 200)
    assert joined[70, 95:106].min() == 255  # no seam where pieces meet
    assert area(coverage_mask([square, square], 200, 200)) == pytest.approx(3600)  # no double counting


def test_non_finite_contours_are_skipped_whole():
    triangle = [(20, 20), (180, 40), (60, 170)]
    assert area(coverage_mask([[(1, 1), (np.nan, 5), (9, 9)], triangle], 200, 200)) == pytest.approx(11600, rel=1e-3)


@pytest.mark.parametrize("drawable,expected", [
    (Line(start=Point(20.3, 50), end=Point(180.3, 50), stroke=StrokeStyle(width=2, cap_style=CapStyle.BUTT)), 320),
    (Line(start=Point(20, 20), end=Point(150, 120), stroke=StrokeStyle(width=6, cap_style=CapStyle.BUTT)),
     math.hypot(130, 100) * 6),
    (Rectangle(position=Point(30, 40), width=100, height=60,
               stroke=StrokeStyle(width=4, join_style=JoinStyle.MITER)), 104 * 64 - 96 * 56),
    (Rectangle(position=Point(30.25, 40.6), width=100, height=60, fill=FillStyle(color=Color.black())), 6000),
    (Polygon(vertices=[Point(20, 20), Point(180, 40), Point(60, 150)], fill=FillStyle(color=Color.black())), 10000),
    (Path.from_svg_path("M30.25 40.6 h100 v60 h-100 z", fill=FillStyle(color=Color.black())), 6000),
    (Circle(center=Point(100, 80), radius=40, stroke=StrokeStyle(width=3)), math.pi * (41.5 ** 2 - 38.5 ** 2)),
])
def test_rendered_shapes_cover_their_geometric_area(drawable, expected):
    # OpenCV's antialiased fillPoly added ~0.7 px outside every edge (a 2 px line covered 3.35 px).
    assert area(alpha(drawable)) == pytest.approx(expected, rel=0.004)


def reference_stroke(width, height, contours, style, transform):
    """Pieces from the callback tessellator, rasterized one polygon each."""
    pieces = []
    scale = float(np.linalg.norm(transform[:2, :2], ord=2)) if transform is not None else 1.0

    def polygon(points):
        points = np.asarray(points, float)
        if transform is not None:
            points = points @ transform[:2, :2].T + transform[:2, 2]
        signed = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
        if signed:
            pieces.append(points if signed > 0 else points[::-1])

    def disk(p, radius):
        if radius > 0:
            world = max(radius * scale, 0.125)
            angle = 2 * math.acos(min(1, max(-1, 1 - 0.125 / world)))
            polygon(p + radius * stroking._unit_circle(max(12, math.ceil(2 * math.pi / max(angle, 1e-6)))))

    stroking._stroke_geometry(contours, style, polygon, disk)
    return coverage_mask(pieces, width, height)


@pytest.mark.parametrize("small", [True, False], ids=["per-piece", "vectorized"])
def test_both_stroke_builders_match_the_tessellator(small, monkeypatch):
    monkeypatch.setattr(stroking, "_SMALL_STROKE", 10 ** 9 if small else 0)
    rng = np.random.default_rng(3)
    zigzag = [(20 + i * 14, 60 + 30 * (-1) ** i + rng.uniform(-5, 5), 8 + 4 * math.sin(i)) for i in range(14)]
    ring = [(100 + 50 * math.cos(t), 70 + 40 * math.sin(t), 6) for t in np.linspace(0, 2 * math.pi, 9, endpoint=False)]
    shear = np.array([[1.1, 0.3, 5], [0.1, 0.9, -3], [0, 0, 1.]])
    for joins, caps, dashes, contours, transform in itertools.product(
            JoinStyle, CapStyle, [(), (15, 7)], [[(zigzag, False)], [(ring, True)], [([(40, 40, 10)], False)]],
            [None, shear]):
        style = StrokeStyle(width=8, join_style=joins, cap_style=caps, dash_array=dashes, miter_limit=3)
        ours = _exact_stroke_mask(220, 150, contours, style, transform)
        expected = reference_stroke(220, 150, contours, style, transform)
        assert np.abs(ours.astype(int) - expected).max() <= 1, (joins, caps, dashes, transform)


def test_stroke_widths_match_resvg():
    resvg = pytest.importorskip("resvg_py")
    import cv2
    for width in (1, 2, 5, 12):
        scene = Scene(100, 60, background=Color(0, 0, 0, 0))
        scene.add(Line(start=Point(10, 30), end=Point(90, 30),
                       stroke=StrokeStyle(width=width, cap_style=CapStyle.BUTT)))
        ours = OpenCVRenderer().render(scene, alpha=True).buffer[:, 50, 3]
        other = cv2.imdecode(np.frombuffer(resvg.svg_to_bytes(svg_string=scene.to_svg(strict=True)), np.uint8),
                             cv2.IMREAD_UNCHANGED)[:, 50, 3]
        assert ours.sum() / 255 == pytest.approx(width, abs=0.01)
        assert np.abs(ours.astype(int) - other).max() <= 1
