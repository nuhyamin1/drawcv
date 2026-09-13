"""Compare regional composition against the pre-optimization full-canvas equations."""
import cv2
import numpy as np
import pytest
from drawcv import (Scene, Point, Color, FillStyle, StrokeStyle, GradientStop,
    LinearGradient, RadialGradient, Rectangle, Circle, Path, Group, Transform,
    Mask, ClipRect, BlurEffect, ShadowEffect, OpenCVRenderer, Line, Polyline,
    StrokePoint, FreehandStroke, CapStyle, JoinStyle, LineType)
from drawcv.core.bounds import BoundingBox
from drawcv.core.paint_sampling import sample_gradient


class FullCanvasRenderer(OpenCVRenderer):
    """Frozen pre-5B gradient equations; full bounds for unbounded mask blending."""
    def _fill_mask(self, drawable, canvas, mask):
        fill = drawable.fill
        if isinstance(fill.paint, Color):
            return super()._fill_mask(drawable, canvas, mask)
        opacity = fill.opacity * self._get_render_opacity(drawable)
        source = sample_gradient(fill.paint, drawable.world_matrix, canvas.width, canvas.height)
        source *= (mask.astype(np.float32) / 255 * opacity)[..., None]
        a = source[..., 3:4]
        if getattr(canvas, 'is_isolated', False):
            canvas.buffer[..., :3] = source[..., :3] + canvas.buffer[..., :3]*(1-a)
            canvas.buffer[..., 3:4] = a + canvas.buffer[..., 3:4]*(1-a)
        else:
            canvas.buffer[:] = np.rint(source[..., :3] + canvas.buffer*(1-a)).clip(0, 255).astype(np.uint8)

    def _composite_mask(self, canvas, mask, color_bgr, alpha, bbox=None):
        if bbox is None:
            bbox = BoundingBox(0, 0, canvas.width, canvas.height)
        return super()._composite_mask(canvas, mask, color_bgr, alpha, bbox)

    def _world_points(self, drawable, points):
        return [drawable.to_world(Point(p.x, p.y)) for p in points]


STOPS = (GradientStop(.1, Color(250, 10, 20, 0)), GradientStop(.5, Color(20, 240, 80, .7)),
         GradientStop(.5, Color(10, 50, 250, .2)), GradientStop(.9, Color(230, 90, 60)))


def assert_equivalent(scene, *, alpha=True):
    before = scene.to_json()
    actual = OpenCVRenderer().render(scene, alpha=alpha).buffer
    expected = FullCanvasRenderer().render(scene, alpha=alpha).buffer
    np.testing.assert_array_equal(actual, expected)
    assert scene.to_json() == before
    return actual


@pytest.mark.parametrize('space', ['object', 'world'])
@pytest.mark.parametrize('radial', [False, True])
@pytest.mark.parametrize('alpha', [False, True])
def test_gradient_regions_preserve_global_coordinates_effects_and_edges(space, radial, alpha):
    scene = Scene(167, 113, background=Color(30, 40, 50, .3))
    for i, (x, y) in enumerate([(-20, 15), (50, 40), (155, 100), (400, 300)]):
        paint = (RadialGradient(Point(x+15, y+10), 37, STOPS, space) if radial else
                 LinearGradient(Point(x, y), Point(x+47, y+36), STOPS, space))
        obj = Rectangle(position=Point(x, y), width=39, height=29,
            fill=FillStyle(paint=paint), stroke=StrokeStyle(width=3, color=Color(0, 0, 0, .4)),
            transform=Transform(rotation=13, scale_x=1.2, scale_y=.8), opacity=.7)
        if i == 1:
            obj.mask = Mask(np.arange(256, dtype=np.uint8).reshape(16, 16))
            obj.clip = ClipRect(40, 30, 65, 70)
            obj.effects = [BlurEffect(kernel_size=7), ShadowEffect(blur_radius=3)]
        scene.add(Group(children=[obj], opacity=.8, transform=Transform(translation_x=2)))
    assert_equivalent(scene, alpha=alpha)


@pytest.mark.parametrize('line_type', list(LineType))
@pytest.mark.parametrize('cap', list(CapStyle))
def test_stroke_regions_preserve_caps_joins_dashes_and_canvas_clipping(line_type, cap):
    scene = Scene(143, 101, Color(0, 0, 0, 0))
    for i, join in enumerate(JoinStyle):
        scene.add(Polyline(points=[Point(-12, i*33+8), Point(55, i*33-5), Point(61, i*33+25), Point(158, i*33+10)],
            stroke=StrokeStyle(width=9, cap_style=cap, join_style=join, miter_limit=8,
                dash_array=(11, 3, 2), dash_offset=-7, line_type=line_type, color=Color(240, 70, 20, .6))))
    scene.add(Line(start=Point(125, 90), end=Point(125, 90), stroke=StrokeStyle(width=11, cap_style=cap)))
    assert_equivalent(scene)
    assert_equivalent(scene, alpha=False)


def test_compound_gradient_holes_and_empty_masks():
    scene = Scene(150, 120, Color(0, 0, 0, 0))
    path = Path(fill=FillStyle(paint=LinearGradient(Point(0, 0), Point(150, 90), STOPS)),
                stroke=StrokeStyle(width=4, color=Color(40, 90, 180, .4)))
    path.move_to(30, 20).line_to(120, 20).line_to(120, 100).line_to(30, 100).close()
    path.move_to(50, 40).line_to(100, 40).line_to(100, 80).line_to(50, 80).close()
    scene.add(path)
    assert_equivalent(scene)
    path.progress = 0
    assert not assert_equivalent(scene)[..., 3].any()


def test_regional_render_recomputes_after_edits_and_temporal_sampling():
    scene = Scene(180, 100, Color(0, 0, 0, 0))
    obj = Circle(center=Point(30, 50), radius=15,
        fill=FillStyle(paint=RadialGradient(Point(30, 50), 30, STOPS)))
    scene.add(obj)
    scene.animate(obj, 'transform.translation_x', 0, 110, duration=1)
    before = scene.to_json()
    for t in (0, .3, 1, .3):
        np.testing.assert_array_equal(scene.render_at_time(t, OpenCVRenderer(), alpha=True).buffer,
            scene.render_at_time(t, FullCanvasRenderer(), alpha=True).buffer)
    assert scene.to_json() == before
    with scene.edit(obj):
        obj.radius = 30
        obj.fill.paint.center = Point(45, 65)
    assert_equivalent(scene)
    scene.undo()
    assert_equivalent(scene)
    scene.redo()
    assert_equivalent(scene)


@pytest.mark.parametrize('alpha', [False, True])
def test_freehand_transform_resolution_keeps_nested_arithmetic_and_progress(alpha):
    points = [StrokePoint(20+i*.7, 40+20*np.sin(i*.1), pressure=(i%13)/13) for i in range(120)]
    ink = FreehandStroke(points=points, variable_width=True, width_mode='pressure',
        min_width=1, max_width=9, stroke=StrokeStyle(width=5, dash_array=(5, 3)),
        transform=Transform(rotation=17, scale_x=1.3, scale_y=.6))
    group = Group(children=[ink], transform=Transform(rotation=-11, translation_x=4, translation_y=3))
    scene = Scene(190, 125, Color(0, 0, 0, 0))
    scene.add(Group(children=[group], transform=Transform(scale_x=.85, scale_y=1.1)))
    for progress in (1, .4, .8):
        ink.progress = progress
        assert_equivalent(scene, alpha=alpha)
    ink.points.append(StrokePoint(175, 95, pressure=.9))
    assert_equivalent(scene, alpha=alpha)


@pytest.mark.parametrize('custom_transform', [False, True])
def test_freehand_custom_coordinate_mapping_is_respected(custom_transform):
    class OffsetInk(FreehandStroke):
        def to_world(self, point):
            world = super().to_world(point)
            return Point(world.x+7, world.y-3)
    class OffsetTransform(Transform):
        def transform_point(self, point, default_pivot=None):
            world = super().transform_point(point, default_pivot)
            return Point(world.x+7, world.y-3)
    cls = FreehandStroke if custom_transform else OffsetInk
    ink = cls(points=[StrokePoint(10, 40), StrokePoint(60, 50)], stroke=StrokeStyle(width=4),
              transform=OffsetTransform() if custom_transform else Transform())
    scene = Scene(100, 80, Color(0, 0, 0, 0))
    scene.add(ink)
    # Avoid serialization of an intentionally unregistered extension type.
    np.testing.assert_array_equal(OpenCVRenderer().render(scene, alpha=True).buffer,
                                  FullCanvasRenderer().render(scene, alpha=True).buffer)


def test_small_gradient_avoids_canvas_sized_sampling(monkeypatch):
    import drawcv.renderer as module
    original = module.sample_gradient
    sampled = []
    def track(paint, matrix, width, height, **kwargs):
        sampled.append((width, height))
        return original(paint, matrix, width, height, **kwargs)
    monkeypatch.setattr(module, 'sample_gradient', track)
    scene = Scene(800, 600, Color(0, 0, 0, 0))
    scene.add(Rectangle(position=Point(400, 300), width=15, height=12,
        fill=FillStyle(paint=LinearGradient(Point(400, 300), Point(415, 312), STOPS))))
    pixels = OpenCVRenderer().render(scene, alpha=True).buffer
    assert pixels[305, 405, 3] > 0
    assert len(sampled) == 1 and sampled[0][0]*sampled[0][1] < 400


def test_explicit_pivot_coordinate_mapping_does_not_need_geometry(monkeypatch):
    obj = Rectangle(position=Point(10, 20), width=60, height=30,
        transform=Transform(pivot=Point(0, 0), rotation=20, scale_x=1.2))
    group = Group(children=[obj], transform=Transform(pivot=Point(4, 5), rotation=-13))
    expected_matrix = group.transform.get_matrix() @ obj.transform.get_matrix()
    expected_point = group.transform.transform_point(obj.transform.transform_point(Point(33, 47)))
    def unused():
        raise AssertionError('Explicit pivot should not calculate an unused geometry center')
    monkeypatch.setattr(obj, 'get_geometry_bounds', unused)
    monkeypatch.setattr(group, 'get_geometry_bounds', unused)
    assert obj.to_world(Point(33, 47)) == expected_point
    np.testing.assert_array_equal(obj.world_matrix, expected_matrix)
