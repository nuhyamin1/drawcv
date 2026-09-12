"""Unit tests for OpenCVRenderer, alpha compositing, and retained-mode rasterization."""

import numpy as np
import pytest

from drawcv.core.color import Color
from drawcv.core.exceptions import RenderError
from drawcv.core.geometry import Point
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_renderer_canvas_dimensions_and_background():
    scene = Scene(width=300, height=200, background=Color(100, 150, 200))
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    assert canvas.width == 300
    assert canvas.height == 200
    buf = canvas.to_numpy()
    assert buf.shape == (200, 300, 3)
    # Check background BGR
    assert np.all(buf[50, 50] == [200, 150, 100])


def test_renderer_line_drawing():
    scene = Scene(width=100, height=100, background=Color.white())
    line = Line(
        start=Point(10, 50),
        end=Point(90, 50),
        stroke=StrokeStyle(color=Color.black(), width=4)
    )
    scene.add(line)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    # Center of line should be black [0, 0, 0]
    assert np.array_equal(buf[50, 50], [0, 0, 0])
    # Outside line should still be white [255, 255, 255]
    assert np.array_equal(buf[10, 10], [255, 255, 255])


def test_renderer_rectangle_drawing():
    scene = Scene(width=100, height=100, background=Color.white())
    rect = Rectangle(
        position=Point(20, 20),
        width=60,
        height=60,
        fill=FillStyle(color=Color.blue())  # BGR: [255, 0, 0]
    )
    scene.add(rect)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    # Inside rectangle should be blue [255, 0, 0]
    assert np.array_equal(buf[50, 50], [255, 0, 0])
    # Outside should be white [255, 255, 255]
    assert np.array_equal(buf[5, 5], [255, 255, 255])


def test_renderer_circle_drawing():
    scene = Scene(width=100, height=100, background=Color.white())
    circle = Circle(
        center=Point(50, 50),
        radius=25,
        fill=FillStyle(color=Color.red())  # BGR: [0, 0, 255]
    )
    scene.add(circle)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    # Center of circle should be red [0, 0, 255]
    assert np.array_equal(buf[50, 50], [0, 0, 255])
    # Outside should be white
    assert np.array_equal(buf[10, 10], [255, 255, 255])


def test_renderer_visibility():
    scene = Scene(width=100, height=100, background=Color.white())
    rect = Rectangle(
        position=Point(20, 20),
        width=60,
        height=60,
        fill=FillStyle(color=Color.black()),
        visible=False
    )
    scene.add(rect)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    # Since rectangle is invisible, entire canvas remains white
    assert np.all(buf == 255)


def test_alpha_compositing_mathematical_precision():
    # White background: [255, 255, 255]
    scene = Scene(width=100, height=100, background=Color.white())
    
    # Red fill: BGR [0, 0, 255]
    # Effective alpha = color.a (1.0) * fill.opacity (0.5) * drawable.opacity (1.0) = 0.5
    # Expected interior BGR:
    # dst = 255 * (1 - 0.5) + color * 0.5 = 127.5 -> rounded to 127 or 128
    rect = Rectangle(
        position=Point(10, 10),
        width=80,
        height=80,
        fill=FillStyle(color=Color.red(), opacity=0.5)
    )
    scene.add(rect)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    sample_pixel = buf[50, 50]
    b, g, r = int(sample_pixel[0]), int(sample_pixel[1]), int(sample_pixel[2])
    # Blue: 127-128, Green: 127-128, Red: 255
    assert abs(b - 128) <= 1
    assert abs(g - 128) <= 1
    assert r == 255


def test_effective_alpha_cascade():
    # Test color.a * fill.opacity * drawable.opacity
    scene = Scene(width=100, height=100, background=Color.white())
    rect = Rectangle(
        position=Point(10, 10),
        width=80,
        height=80,
        opacity=0.8,
        fill=FillStyle(
            color=Color(255, 0, 0, a=0.75),
            opacity=0.5
        )
    )
    # Effective alpha = 0.75 * 0.5 * 0.8 = 0.30
    scene.add(rect)
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()

    # Blue channel: 255 * 0.70 + 0 * 0.30 = 178.5 -> ~178 or 179
    sample_pixel = buf[50, 50]
    b, g, r = int(sample_pixel[0]), int(sample_pixel[1]), int(sample_pixel[2])
    assert abs(b - 179) <= 1
    assert abs(g - 179) <= 1
    assert r == 255


def test_z_order_and_stable_ordering():
    scene = Scene(width=100, height=100, background=Color.white())
    # Object A: Blue rectangle
    r1 = Rectangle(position=Point(20, 20), width=50, height=50, fill=FillStyle(color=Color.blue()), z_index=0)
    # Object B: Red rectangle overlapping Object A
    r2 = Rectangle(position=Point(40, 40), width=50, height=50, fill=FillStyle(color=Color.red()), z_index=0)

    scene.add(r1)
    scene.add(r2)

    renderer = OpenCVRenderer()
    # By default, insertion order preserves r2 above r1 at overlap point (45, 45)
    canvas = renderer.render(scene)
    buf = canvas.to_numpy()
    assert np.array_equal(buf[45, 45], [0, 0, 255])  # Red

    # Move r1 to front -> r1 should now appear on top of r2
    scene.move_to_front(r1.id)
    canvas2 = renderer.render(scene)
    buf2 = canvas2.to_numpy()
    assert np.array_equal(buf2[45, 45], [255, 0, 0])  # Blue


def test_retained_mode_state_mutation():
    scene = Scene(width=100, height=100, background=Color.white())
    circle = Circle(
        center=Point(30, 30),
        radius=15,
        fill=FillStyle(color=Color.blue())
    )
    scene.add(circle)

    renderer = OpenCVRenderer()
    canvas1 = renderer.render(scene)
    buf1 = canvas1.to_numpy()
    assert np.array_equal(buf1[30, 30], [255, 0, 0])  # Blue at (30, 30)

    # Mutate the retained object in-place
    circle.center = Point(70, 70)
    circle.fill.color = Color.red()

    # Re-rendering must show updated state without residual artifacts at old location
    canvas2 = renderer.render(scene)
    buf2 = canvas2.to_numpy()
    assert np.array_equal(buf2[70, 70], [0, 0, 255])  # Red at (70, 70)
    assert np.array_equal(buf2[30, 30], [255, 255, 255])  # Restored to white at old position


def test_transform_rotation_and_scale_rendering():
    scene = Scene(100, 100)
    circle = Circle(center=Point(50, 50), radius=20)
    circle.transform.rotation = 45.0
    scene.add(circle)

    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    assert canvas.width == 100 and canvas.height == 100

    # Non-positive scaling must be rejected by Transform validation
    with pytest.raises(Exception):
        circle.transform.scale_x = 0.0
