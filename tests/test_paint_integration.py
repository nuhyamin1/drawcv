"""Integration tests for advanced paints across shapes, strokes, effects, and compositing."""

import copy
import numpy as np
import pytest

from drawcv import (
    Arc,
    ArcClosure,
    Arrow,
    ArrowHeadStyle,
    BezierCurve,
    BlendMode,
    BlurEffect,
    Circle,
    ClipRect,
    Color,
    ConicGradient,
    FillStyle,
    FreehandStroke,
    GradientStop,
    Group,
    ImageInterpolation,
    ImagePaint,
    Layer,
    LinearGradient,
    Line,
    Mask,
    MaskMapping,
    OpenCVRenderer,
    Path,
    Point,
    Polygon,
    Polyline,
    RadialGradient,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShadowEffect,
    StrokePoint,
    StrokeStyle,
    Transform,
)

STOPS_RGB = (
    GradientStop(0.0, Color.red()),
    GradientStop(0.5, Color.green()),
    GradientStop(1.0, Color.blue()),
)


def sample_texture(w=16, h=16):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = [int((x / w) * 255), int((y / h) * 255), 128, 255]
    return arr


def test_dual_gradient_fill_and_stroke_on_shape():
    fill_grad = LinearGradient(Point(0, 0), Point(100, 100), STOPS_RGB)
    stroke_grad = RadialGradient(Point(50, 50), 40.0, (GradientStop(0.0, Color.yellow()), GradientStop(1.0, Color.red())))

    rect = Rectangle(
        position=Point(10, 10),
        width=80,
        height=80,
        fill=FillStyle(paint=fill_grad),
        stroke=StrokeStyle(paint=stroke_grad, width=6.0),
    )
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(rect)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    assert np.any(buf[..., 3] > 0)
    # Center should be mostly fill gradient green/midpoint
    assert buf[50, 50, 3] == 255


def test_image_paint_fill_with_stroke_and_blur():
    tex = sample_texture(10, 10)
    img_paint = ImagePaint(image=tex, scale=(5.0, 5.0), repeat="repeat")
    rect = RoundedRectangle(
        x=10,
        y=10,
        width=80,
        height=80,
        corner_radius=15.0,
        fill=FillStyle(paint=img_paint),
        stroke=StrokeStyle(color=Color.black(), width=2.0),
        effects=[BlurEffect(kernel_size=5, sigma=1.5)],
    )

    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(rect)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    assert np.any(buf[..., 3] > 0)
    # Blurred edges should have fractional alpha
    assert np.any((buf[..., 3] > 0) & (buf[..., 3] < 255))


def test_conic_gradient_with_shadow_effect():
    conic = ConicGradient(Point(50, 50), STOPS_RGB)
    circle = Circle(
        center=Point(50, 50),
        radius=30.0,
        fill=FillStyle(paint=conic),
        effects=[ShadowEffect(offset_x=10.0, offset_y=10.0, blur_radius=5.0, color=Color(0, 0, 0, 0.5))],
    )

    scene = Scene(120, 120, background=Color(255, 255, 255, 1.0))
    scene.add(circle)

    buf = OpenCVRenderer().render(scene).buffer
    # Shadow region should darken background
    shadow_sample = buf[90, 90]
    assert shadow_sample[0] < 255 or shadow_sample[1] < 255 or shadow_sample[2] < 255


def test_gradient_stroke_under_clipping():
    grad = LinearGradient(Point(0, 50), Point(100, 50), STOPS_RGB)
    line = Line(
        start=Point(0, 50),
        end=Point(100, 50),
        stroke=StrokeStyle(paint=grad, width=10.0),
        clip=ClipRect(x=20, y=40, width=60, height=20),
    )

    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(line)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    # Outside clip: (x=10, y=50) should be transparent
    assert buf[50, 10, 3] == 0
    # Inside clip: (x=50, y=50) should be opaque
    assert buf[50, 50, 3] == 255
    # Outside clip: (x=90, y=50) should be transparent
    assert buf[50, 90, 3] == 0


def test_painted_group_with_blend_mode():
    grad1 = LinearGradient(Point(0, 0), Point(100, 0), (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())))
    rect1 = Rectangle(position=Point(10, 10), width=80, height=80, fill=FillStyle(paint=grad1))

    grad2 = LinearGradient(Point(0, 0), Point(0, 100), (GradientStop(0.0, Color.green()), GradientStop(1.0, Color.yellow())))
    rect2 = Rectangle(
        position=Point(20, 20),
        width=80,
        height=80,
        fill=FillStyle(paint=grad2),
        blend_mode=BlendMode.MULTIPLY,
    )

    group = Group(children=[rect1, rect2])
    scene = Scene(120, 120, background=Color(255, 255, 255, 1.0))
    scene.add(group)

    buf = OpenCVRenderer().render(scene).buffer
    assert buf.shape == (120, 120, 3)
    # Overlap area has multiplied colors
    assert tuple(buf[50, 50]) != (255, 255, 255)


def test_paint_undo_and_redo_fill_and_stroke():
    scene = Scene(100, 100)
    rect = Rectangle(position=Point(10, 10), width=50, height=50, fill=FillStyle(color=Color.red()))
    scene.add(rect)

    # 1. Fill edit undo/redo
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[:, :] = [0, 255, 0, 255]
    img_paint = ImagePaint(image=img)

    with scene.edit(rect):
        rect.fill = FillStyle(paint=img_paint)

    assert isinstance(rect.fill.paint, ImagePaint)
    scene.undo()
    assert isinstance(rect.fill.paint, Color)
    assert rect.fill.paint == Color.red()
    scene.redo()
    assert isinstance(rect.fill.paint, ImagePaint)

    # 2. Stroke edit undo/redo
    grad = LinearGradient(Point(0, 0), Point(50, 0), STOPS_RGB)
    with scene.edit(rect):
        rect.stroke = StrokeStyle(paint=grad, width=4.0)

    assert isinstance(rect.stroke.paint, LinearGradient)
    scene.undo()
    assert rect.stroke is None or isinstance(rect.stroke.paint, Color)
    scene.redo()
    assert isinstance(rect.stroke.paint, LinearGradient)


def test_cloning_with_image_paint_independent_raster():
    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]
    paint = ImagePaint(image=img, scale=(2.0, 2.0))
    rect = Rectangle(position=Point(0, 0), width=40, height=40, fill=FillStyle(paint=paint))

    # Test Drawable clone
    cloned_rect = rect.clone()
    assert cloned_rect.fill.paint.image is not rect.fill.paint.image
    np.testing.assert_array_equal(cloned_rect.fill.paint.image, rect.fill.paint.image)
    cloned_rect.fill.paint.image[0, 0] = [99, 99, 99]
    assert rect.fill.paint.image[0, 0, 0] == 10

    # Test Scene deepcopy
    scene = Scene(100, 100)
    scene.add(rect)
    cloned_scene = copy.deepcopy(scene)
    scene_rect = cloned_scene.layers[0].objects[0]
    assert scene_rect.fill.paint.image is not rect.fill.paint.image
    scene_rect.fill.paint.image[0, 0] = [77, 77, 77]
    assert rect.fill.paint.image[0, 0, 0] == 10


def test_paint_with_mask():
    grad = LinearGradient(
        Point(0, 0),
        Point(100, 0),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )
    mask_buf = np.zeros((100, 100), dtype=np.uint8)
    mask_buf[30:70, 30:70] = 255
    mask = Mask(mask_buf, mapping=MaskMapping.ABSOLUTE)

    rect = Rectangle(
        position=Point(0, 0),
        width=100,
        height=100,
        fill=FillStyle(paint=grad),
        mask=mask,
    )
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(rect)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    # Inside mask: (50, 50) is fully opaque
    assert buf[50, 50, 3] == 255
    # Outside mask: (10, 10) is fully transparent
    assert buf[10, 10, 3] == 0


def test_paint_inside_layer_opacity():
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    layer = scene.create_layer("PaintLayer", opacity=0.5)
    grad = LinearGradient(
        Point(0, 0),
        Point(100, 0),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )
    rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(paint=grad))
    layer.add(rect)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    # Pixel (50, 50) modulated by layer opacity 0.5 -> alpha ~ 128
    assert np.isclose(buf[50, 50, 3], 128, atol=2)


def test_arrow_gradient_paint_shaft_and_arrowhead():
    arrow = Arrow(
        start=Point(10, 50),
        end=Point(90, 50),
        head_length=20.0,
        head_width=20.0,
        head_style=ArrowHeadStyle.TRIANGLE,
        stroke=StrokeStyle(
            paint=LinearGradient(
                Point(10, 50),
                Point(90, 50),
                (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
            ),
            width=6.0,
        ),
        fill=FillStyle(
            paint=LinearGradient(
                Point(70, 50),
                Point(90, 50),
                (GradientStop(0.0, Color.green()), GradientStop(1.0, Color.yellow())),
            ),
        ),
    )
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(arrow)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    # Shaft pixel near start (x=20, y=50): Red-dominated stroke
    shaft_px = buf[50, 20]
    assert shaft_px[3] == 255
    assert shaft_px[2] > 200  # Red channel
    assert shaft_px[0] < 60   # Blue channel

    # Arrowhead center (x=80, y=50): Arrowhead fill (Green/Yellow)
    head_px = buf[50, 80]
    assert head_px[3] == 255
    assert head_px[1] > 150   # Green channel


def test_bezier_and_path_gradient_strokes():
    grad = LinearGradient(
        Point(10, 50),
        Point(90, 50),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )

    # 1. BezierCurve
    bezier = BezierCurve(
        p0=Point(10, 50),
        p1=Point(35, 20),
        p2=Point(65, 80),
        p3=Point(90, 50),
        stroke=StrokeStyle(paint=grad, width=6.0),
    )
    scene_bezier = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_bezier.add(bezier)
    buf_b = OpenCVRenderer().render(scene_bezier, alpha=True).buffer
    assert buf_b[50, 12, 3] > 0
    assert buf_b[50, 12, 2] > buf_b[50, 12, 0]  # Red > Blue at start
    assert buf_b[50, 88, 3] > 0
    assert buf_b[50, 88, 0] > buf_b[50, 88, 2]  # Blue > Red at end

    # 2. Path
    path = Path(stroke=StrokeStyle(paint=grad, width=6.0))
    path.move_to(Point(10, 50))
    path.line_to(Point(90, 50))
    scene_path = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_path.add(path)
    buf_p = OpenCVRenderer().render(scene_path, alpha=True).buffer
    assert buf_p[50, 12, 3] > 0
    assert buf_p[50, 12, 2] > buf_p[50, 12, 0]  # Red > Blue at start
    assert buf_p[50, 88, 3] > 0
    assert buf_p[50, 88, 0] > buf_p[50, 88, 2]  # Blue > Red at end


def test_polyline_and_polygon_stroke_paint():
    grad = LinearGradient(
        Point(10, 80),
        Point(90, 80),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )

    # 1. Polyline
    poly = Polyline(
        points=[Point(10, 80), Point(50, 20), Point(90, 80)],
        stroke=StrokeStyle(paint=grad, width=6.0),
    )
    scene_poly = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_poly.add(poly)

    buf_poly = OpenCVRenderer().render(scene_poly, alpha=True).buffer
    assert buf_poly[78, 12, 3] > 0
    assert buf_poly[78, 12, 2] > buf_poly[78, 12, 0]  # Red > Blue at start
    assert buf_poly[78, 88, 3] > 0
    assert buf_poly[78, 88, 0] > buf_poly[78, 88, 2]  # Blue > Red at end

    # 2. Polygon
    pgon = Polygon(
        vertices=[Point(10, 80), Point(50, 10), Point(90, 80)],
        stroke=StrokeStyle(paint=grad, width=6.0),
    )
    scene_pgon = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_pgon.add(pgon)

    buf_pgon = OpenCVRenderer().render(scene_pgon, alpha=True).buffer
    # Left bottom corner near (12, 78) is Red
    assert buf_pgon[78, 12, 3] > 0
    assert buf_pgon[78, 12, 2] > 200 and buf_pgon[78, 12, 0] < 50
    # Right bottom corner near (88, 78) is Blue
    assert buf_pgon[78, 88, 3] > 0
    assert buf_pgon[78, 88, 0] > 200 and buf_pgon[78, 88, 2] < 50


def test_arc_gradient_stroke():
    grad_arc = LinearGradient(
        Point(10, 50),
        Point(90, 50),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )
    arc = Arc(
        center=Point(50, 50),
        radius_x=40.0,
        radius_y=40.0,
        start_angle=180.0,
        sweep_angle=180.0,
        closure=ArcClosure.OPEN,
        stroke=StrokeStyle(paint=grad_arc, width=6.0),
    )
    scene_arc = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_arc.add(arc)

    buf_arc = OpenCVRenderer().render(scene_arc, alpha=True).buffer
    # Arc start at (10, 50) is pure Red
    p_start = buf_arc[50, 10]
    assert p_start[3] > 0
    assert p_start[2] > 200 and p_start[0] < 50  # Red > 200, Blue < 50

    # Arc end at (90, 50) is pure Blue
    p_end = buf_arc[50, 90]
    assert p_end[3] > 0
    assert p_end[0] > 200 and p_end[2] < 50      # Blue > 200, Red < 50


def test_freehand_stroke_paint():
    pts = [
        StrokePoint(10, 50, pressure=1.0, timestamp=0.0),
        StrokePoint(50, 50, pressure=1.0, timestamp=0.1),
        StrokePoint(90, 50, pressure=1.0, timestamp=0.2),
    ]
    grad = LinearGradient(
        Point(10, 50),
        Point(90, 50),
        (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
    )
    fh = FreehandStroke(points=pts, stroke=StrokeStyle(paint=grad, width=6.0))
    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(fh)

    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    assert buf[50, 12, 3] > 0
    assert buf[50, 12, 2] > buf[50, 12, 0]  # Red > Blue at start
    assert buf[50, 88, 3] > 0
    assert buf[50, 88, 0] > buf[50, 88, 2]  # Blue > Red at end


def test_object_versus_world_space_stroke_paint_nested_transforms():
    stops = (GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue()))

    # Line from (0, 0) to (50, 0) rotated by 90 degrees under nested groups
    # Object space: gradient rotates with the entity, so at (0, 40) it is Blue
    grad_obj = LinearGradient(Point(0, 0), Point(50, 0), stops, space="object")
    line_obj = Line(start=Point(0, 0), end=Point(50, 0), stroke=StrokeStyle(paint=grad_obj, width=6.0))
    g1_obj = Group(children=[line_obj], transform=Transform(rotation=45.0, pivot=Point(0, 0)))
    g2_obj = Group(children=[g1_obj], transform=Transform(rotation=45.0, pivot=Point(0, 0)))
    scene_obj = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_obj.add(g2_obj)
    buf_obj = OpenCVRenderer().render(scene_obj, alpha=True).buffer

    # World space: gradient remains in world canvas coords, so along x=0 it evaluates to Red
    grad_world = LinearGradient(Point(0, 0), Point(50, 0), stops, space="world")
    line_world = Line(start=Point(0, 0), end=Point(50, 0), stroke=StrokeStyle(paint=grad_world, width=6.0))
    g1_world = Group(children=[line_world], transform=Transform(rotation=45.0, pivot=Point(0, 0)))
    g2_world = Group(children=[g1_world], transform=Transform(rotation=45.0, pivot=Point(0, 0)))
    scene_world = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene_world.add(g2_world)
    buf_world = OpenCVRenderer().render(scene_world, alpha=True).buffer

    p_obj = buf_obj[40, 0]
    p_world = buf_world[40, 0]
    assert p_obj[0] > p_obj[2]  # Object space has rotated Blue along world +Y
    assert p_world[2] > p_world[0]  # World space stays fixed at Red along world X=0
