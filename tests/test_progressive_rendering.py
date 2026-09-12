"""Tests for arc-length progressive path slicing and rendering."""

import pytest
import numpy as np

from drawcv.canvas import Canvas
from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, ArrowHeadStyle
from drawcv.core.geometry import Point, StrokePoint
from drawcv.renderer import OpenCVRenderer
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.line import Line
from drawcv.shapes.path import Path
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_line_progressive_slicing():
    line = Line(start=Point(0, 0), end=Point(100, 0), stroke=StrokeStyle(color=Color.black(), width=2))
    assert line.supports_progressive_rendering is True

    half_line = line.slice_at_progress(0.5)
    assert half_line.render_progress == 1.0  # recursion guard
    assert half_line.start.x == 0.0
    assert half_line.end.x == 50.0
    assert half_line.end.y == 0.0

    # Test rendering at 0.5 without recursion errors
    canvas = Canvas(200, 200)
    canvas.clear(Color.white())
    line.render_progress = 0.5
    renderer = OpenCVRenderer()
    renderer.render_drawable(line, canvas)
    # Right half (x > 50 at y=0) should be untouched white
    assert np.all(canvas.buffer[0, 75] == [255, 255, 255])
    # Left half (x < 50 at y=0) should have drawn black stroke
    assert np.all(canvas.buffer[0, 25] == [0, 0, 0])


def test_progressive_rendering_recursion_guard():
    line = Line(start=Point(0, 0), end=Point(100, 0), stroke=StrokeStyle(color=Color.black(), width=2))
    line.render_progress = 0.5

    original_slice = line.slice_at_progress
    call_count = 0

    def slice_spy(p):
        nonlocal call_count
        call_count += 1
        return original_slice(p)

    line.slice_at_progress = slice_spy

    canvas = Canvas(200, 200)
    renderer = OpenCVRenderer()
    renderer.render_drawable(line, canvas)

    # Invariant: exactly one geometric slice occurs, preventing recursion
    assert call_count == 1


def test_polyline_euclidean_vs_index():
    # Segment 1: (0, 0) to (10, 0) -> length 10
    # Segment 2: (10, 0) to (100, 0) -> length 90
    # Total length = 100.
    # At 50% arc length, distance = 50.
    # This must be at x = 50 on segment 2, NOT halfway along index (which would be at vertex 1)!
    poly = Polyline(points=[Point(0, 0), Point(10, 0), Point(100, 0)])
    sliced = poly.slice_at_progress(0.5)
    assert len(sliced.points) == 3
    assert sliced.points[0].x == 0.0
    assert sliced.points[1].x == 10.0
    assert pytest.approx(sliced.points[2].x) == 50.0
    assert sliced.render_progress == 1.0


def test_arrow_progressive_reveal():
    arrow = Arrow(
        start=Point(0, 0),
        end=Point(100, 0),
        head_style=ArrowHeadStyle.TRIANGLE,
        fill=FillStyle(color=Color.red()),
    )
    # At 50%: shaft is half length, fill is None (revealed outline)
    sliced = arrow.slice_at_progress(0.5)
    assert sliced.end.x == 50.0
    assert sliced.fill is None

    # At 100%: full length, arrowhead restored, fill restored
    full = arrow.slice_at_progress(1.0)
    assert full.end.x == 100.0
    assert full.head_style == ArrowHeadStyle.TRIANGLE
    assert full.fill is not None


def test_bezier_progressive_slicing():
    bezier = BezierCurve(
        p0=Point(0, 0),
        p1=Point(0, 100),
        p2=Point(100, 100),
        p3=Point(100, 0),
        stroke=StrokeStyle(color=Color.blue(), width=2),
    )
    sliced = bezier.slice_at_progress(0.5)
    assert sliced.render_progress == 1.0
    assert sliced.p0 == Point(0, 0)
    # The end of the sliced curve (p3) should be near the midpoint of the curve (x ~ 50)
    assert pytest.approx(sliced.p3.x, abs=2.0) == 50.0


def test_path_progressive_slicing_and_fill_suppression():
    path = Path(
        stroke=StrokeStyle(color=Color.black(), width=2),
        fill=FillStyle(color=Color.green()),
    )
    path.move_to(0, 0).line_to(100, 0).line_to(100, 100).close()

    # Sliced at 50%
    sliced = path.slice_at_progress(0.5)
    assert sliced.render_progress == 1.0
    assert sliced.stroke is not None
    assert sliced.fill is None  # Stroke-first reveal: fill suppressed when progress < 1.0

    # Sliced at 100%
    full = path.slice_at_progress(1.0)
    assert full.fill is not None  # Fill revealed at 1.0


def test_freehand_stroke_metadata_interpolation():
    p1 = StrokePoint(0, 0, pressure=0.2, timestamp=1.0, velocity=10.0)
    p2 = StrokePoint(100, 0, pressure=0.8, timestamp=2.0, velocity=20.0)
    stroke = FreehandStroke(points=[p1, p2])

    sliced = stroke.slice_at_progress(0.5)
    assert len(sliced.points) == 2
    cut = sliced.points[1]
    assert cut.x == 50.0
    assert pytest.approx(cut.pressure) == 0.5
    assert pytest.approx(cut.timestamp) == 1.5
    assert pytest.approx(cut.velocity) == 15.0


def test_arc_progressive_sweep_reveal():
    arc = Arc(
        center=Point(100, 100),
        radius_x=50,
        radius_y=50,
        start_angle=0,
        sweep_angle=180,
        closure=ArcClosure.PIE,
        fill=FillStyle(color=Color.yellow()),
    )
    sliced = arc.slice_at_progress(0.5)
    assert pytest.approx(sliced.sweep_angle) == 90.0
    assert sliced.closure == ArcClosure.OPEN  # open outline during reveal
    assert sliced.fill is None               # fill suppressed during reveal


def test_unsupported_progressive_shapes_render_safely():
    # Shapes without slice_at_progress should not raise AttributeError when progress < 1.0
    rect = Rectangle(position=Point(10, 10), width=50, height=50)
    circle = Circle(center=Point(50, 50), radius=30)

    assert rect.supports_progressive_rendering is False
    assert circle.supports_progressive_rendering is False

    rect.render_progress = 0.5
    circle.render_progress = 0.5

    canvas = Canvas(100, 100)
    renderer = OpenCVRenderer()
    # Rendering should succeed without throwing
    renderer.render_drawable(rect, canvas)
    renderer.render_drawable(circle, canvas)


def test_drawable_progress_property_clamping():
    line = Line(start=Point(0, 0), end=Point(10, 10))
    line.progress = 0.75
    assert line.progress == 0.75
    assert line.render_progress == 0.75

    # Clamp overshooting
    line.progress = 1.5
    assert line.progress == 1.0
    line.progress = -0.5
    assert line.progress == 0.0
