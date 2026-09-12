"""Integration tests for rendering Phase 3 shapes with OpenCVRenderer."""

import numpy as np
import pytest

from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, ArrowHeadStyle, FillRule
from drawcv.core.geometry import Point
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.path import Path
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_render_all_phase3_shapes():
    scene = Scene(width=800, height=600, background=Color.WHITE)

    # Ellipse
    scene.add(Ellipse(
        center=Point(100, 100), radius_x=60, radius_y=30,
        fill=FillStyle(color=Color(255, 0, 0, 0.8)),
        stroke=StrokeStyle(color=Color.BLACK, width=2.0)
    ))

    # Polygon
    scene.add(Polygon(
        vertices=[Point(250, 50), Point(350, 80), Point(300, 150), Point(220, 120)],
        fill=FillStyle(color=Color(0, 200, 0, 0.7)),
        stroke=StrokeStyle(color=Color.BLACK, width=3.0)
    ))

    # Polyline
    scene.add(Polyline(
        points=[Point(400, 50), Point(450, 100), Point(420, 150), Point(480, 200)],
        closed=False,
        stroke=StrokeStyle(color=Color(0, 0, 255), width=4.0)
    ))

    # RoundedRectangle
    scene.add(RoundedRectangle(
        x=550, y=50, width=120, height=80, corner_radius=20,
        fill=FillStyle(color=Color(255, 165, 0)),
        stroke=StrokeStyle(color=Color(100, 50, 0), width=2.5)
    ))

    # Arc (Pie sector)
    scene.add(Arc(
        center=Point(100, 300), radius_x=60, radius_y=60,
        start_angle=30.0, sweep_angle=120.0, closure=ArcClosure.PIE,
        fill=FillStyle(color=Color(200, 0, 200, 0.8)),
        stroke=StrokeStyle(color=Color.BLACK, width=2.0)
    ))

    # Arrow
    scene.add(Arrow(
        start=Point(220, 300), end=Point(350, 300),
        head_length=25.0, head_width=16.0, head_style=ArrowHeadStyle.TRIANGLE,
        stroke=StrokeStyle(color=Color(0, 100, 200), width=3.0),
        fill=FillStyle(color=Color(0, 100, 200))
    ))

    # BezierCurve
    scene.add(BezierCurve.cubic(
        p0=Point(400, 350), p1=Point(450, 250), p2=Point(500, 450), p3=Point(550, 350),
        stroke=StrokeStyle(color=Color(255, 20, 147), width=4.0)
    ))

    # Path (Donut with EVEN_ODD)
    path = Path(
        fill_rule=FillRule.EVEN_ODD,
        fill=FillStyle(color=Color(0, 128, 128, 0.9)),
        stroke=StrokeStyle(color=Color.BLACK, width=2.0)
    )
    # Outer square
    path.move_to(620, 250).line_to(720, 250).line_to(720, 350).line_to(620, 350).close()
    # Inner hole
    path.move_to(650, 280).line_to(690, 280).line_to(690, 320).line_to(650, 320).close()
    scene.add(path)

    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    assert canvas.width == 800
    assert canvas.height == 600
    assert canvas.buffer.shape == (600, 800, 3)

    # Verify background is not completely blank white everywhere
    # Some pixels must be non-white
    is_not_white = np.any(canvas.buffer != 255)
    assert is_not_white is np.True_
