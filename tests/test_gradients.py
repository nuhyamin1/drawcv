"""Gradient contract tests using analytic pixel expectations."""
import cv2
import numpy as np
import pytest
from drawcv import (Scene, OpenCVRenderer, Point, Color, FillStyle, Rectangle,
    Group, Transform, GradientStop, LinearGradient, RadialGradient, ValidationError,
    Mask, MaskMapping, ClipRect, BlurEffect, ShadowEffect, Path, Circle)
from drawcv import Ellipse, RoundedRectangle, Polygon, Arc, ArcClosure, Arrow, ArrowHeadStyle, StrokeStyle, Text, Canvas

STOPS = (GradientStop(0, Color.red()), GradientStop(1, Color.blue()))


def linear(space="object", stops=STOPS):
    return LinearGradient(Point(20, 20), Point(100, 20), stops, space)


def shape(paint):
    return Rectangle(position=Point(0, 0), width=140, height=100, fill=FillStyle(paint=paint))


def scene_of(obj):
    scene = Scene(240, 180, background=Color(0, 0, 0, 0))
    scene.add(obj)
    return scene


def pixels(obj):
    return OpenCVRenderer().render(scene_of(obj), alpha=True).buffer


def test_linear_endpoints_midpoint_and_padding():
    data = pixels(shape(linear()))
    assert tuple(data[40, 10]) == (0, 0, 255, 255)
    assert tuple(data[40, 20]) == (0, 0, 255, 255)
    assert tuple(data[40, 60]) == (128, 0, 128, 255)
    assert tuple(data[40, 120]) == (255, 0, 0, 255)


def test_duplicate_stops_last_wins_and_missing_endpoints_pad():
    stops = (GradientStop(.25, Color.red()), GradientStop(.5, Color.red()),
             GradientStop(.5, Color.blue()), GradientStop(.75, Color.blue()))
    data = pixels(shape(linear(stops=stops)))
    assert tuple(data[40, 30]) == (0, 0, 255, 255)
    assert tuple(data[40, 59]) == (0, 0, 255, 255)
    assert tuple(data[40, 60]) == (255, 0, 0, 255)
    assert tuple(data[40, 120]) == (255, 0, 0, 255)


def test_straight_channel_interpolation_before_premultiplication():
    data = pixels(shape(linear(stops=(GradientStop(0, Color.red()),
                                    GradientStop(1, Color(0, 0, 255, 0))))))
    assert tuple(data[40, 60]) == (128, 0, 128, 128)
    assert tuple(data[40, 120]) == (0, 0, 0, 0)


@pytest.mark.parametrize("space,expected", [("object", (0, 0, 255, 255)), ("world", (128, 0, 128, 255))])
def test_translation_distinguishes_object_and_world(space, expected):
    obj = shape(linear(space))
    obj.move(40, 0)
    assert tuple(pixels(obj)[40, 60]) == expected


def test_nested_affine_matrix_maps_gradient_in_local_space():
    obj = shape(linear())
    obj.transform = Transform(translation_x=10)
    group = Group(children=[obj], transform=Transform.from_matrix(
        np.array([[0, -1, 160], [2, .5, 0], [0, 0, 1.]])))
    # Local (60,40) -> child (70,40) -> world (120,160).
    assert tuple(pixels(group)[160, 120]) == (128, 0, 128, 255)


def test_radial_center_radius_padding_and_nonuniform_scale():
    paint = RadialGradient(Point(40, 40), 40, STOPS)
    obj = shape(paint)
    obj.transform = Transform(scale_x=2, pivot=Point(0, 0))
    data = pixels(obj)
    assert tuple(data[40, 80]) == (0, 0, 255, 255)
    assert tuple(data[60, 80]) == tuple(data[40, 120]) == (128, 0, 128, 255)
    assert tuple(data[40, 180]) == (255, 0, 0, 255)


def test_nested_opacity_mask_clip_and_gradient_alpha():
    obj = shape(linear(stops=(GradientStop(0, Color(255, 0, 0, .5)), GradientStop(1, Color(0, 0, 255, .5)))))
    obj.opacity = .5
    obj.fill.opacity = .5
    obj.mask = Mask(np.full((180, 240), 128, np.uint8), mapping=MaskMapping.ABSOLUTE)
    obj.clip = ClipRect(0, 0, 80, 100)
    group = Group(children=[obj], opacity=.5)
    data = pixels(group)
    assert tuple(data[40, 60, [1, 3]]) == (0, 8)
    assert np.all(np.abs(data[40, 60, [0, 2]].astype(float) - 127.5) <= .5)
    assert np.all(data[40, 90] == 0)


def test_png_blur_and_external_backgrounds(tmp_path):
    paint = RadialGradient(Point(60, 50), 40, (GradientStop(0, Color.red()), GradientStop(1, Color(255, 0, 0, 0))))
    obj = shape(paint)
    obj.effects = [BlurEffect(kernel_size=9, sigma=2)]
    scene = scene_of(Group(children=[obj], opacity=.6))
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=True)
    path = tmp_path / "gradient.png"
    canvas.save(path)
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, canvas.buffer)
    visible = decoded[..., 3] > 0
    assert np.all(decoded[visible, :3] == (0, 0, 255))
    for bg in [Color.white(), Color.black(), Color(90, 60, 170)]:
        a = decoded[..., 3:4].astype(float)/255
        external = np.rint(decoded[..., :3]*a + np.array(bg.to_bgr())*(1-a)).astype(np.uint8)
        scene.background = bg
        target = renderer.render(scene, alpha=True).buffer[..., :3]
        assert np.abs(external.astype(int)-target.astype(int)).max() <= 1


def test_transparent_gradient_controls_shadow_silhouette():
    obj = Rectangle(position=Point(0, 0), width=100, height=40,
        fill=FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 0),
            (GradientStop(0, Color.red()), GradientStop(1, Color(255, 0, 0, 0))))),
        effects=[ShadowEffect(offset_x=0, offset_y=60, blur_radius=0, color=Color.blue())])
    data = pixels(obj)
    assert tuple(data[80, 50]) == (255, 0, 0, 128)
    assert data[80, 80, 3] < data[80, 20, 3]


def test_compound_hole_and_progressive_fill_suppression():
    obj = Path(fill=FillStyle(paint=linear()))
    obj.move_to(0, 0).line_to(130, 0).line_to(130, 100).line_to(0, 100).close()
    obj.move_to(40, 30).line_to(80, 30).line_to(80, 70).line_to(40, 70).close()
    assert pixels(obj)[50, 60, 3] == 0
    assert pixels(obj)[20, 60, 3] == 255
    obj.progress = .5
    assert np.all(pixels(obj) == 0)


@pytest.mark.parametrize("field,bad", [("start", Point(100, 20)), ("space", "screen"),
    ("stops", ()), ("stops", tuple(reversed(STOPS))), ("stops", (Color.red(), Color.blue()))])
def test_invalid_gradient_mutation_is_atomic(field, bad):
    paint = linear()
    before = paint.to_dict()
    with pytest.raises(ValidationError):
        setattr(paint, field, bad)
    assert paint.to_dict() == before


@pytest.mark.parametrize("bad", [-1, 0, float("nan"), float("inf"), True])
def test_radial_radius_validation(bad):
    paint = RadialGradient(Point(40, 40), 20, STOPS)
    with pytest.raises(ValidationError):
        paint.radius = bad
    assert paint.radius == 20


@pytest.mark.parametrize("position", [-.1, 1.1, float("nan"), True])
def test_invalid_stop_positions_rejected(position):
    with pytest.raises(ValidationError):
        GradientStop(position, Color.red())


def test_fill_color_compatibility_and_explicit_paint_switch():
    fill = FillStyle(False, Color.red(), .5)
    assert fill.to_dict() == {"enabled": False, "color": Color.red().to_dict(), "opacity": .5}
    fill.paint = linear()
    with pytest.raises(ValidationError):
        _ = fill.color
    fill.color = Color.blue()
    assert fill.paint == Color.blue()
    with pytest.raises(ValidationError):
        FillStyle(color=Color.red(), paint=linear())
    with pytest.raises(ValidationError):
        FillStyle(color=linear())
    with pytest.raises(ValidationError):
        fill.paint = "linear"
    assert fill.color == Color.blue()


@pytest.mark.parametrize("paint", [linear(), RadialGradient(Point(50, 50), 40, STOPS, "world")])
def test_persistence_clone_history_and_animation(paint):
    obj = shape(paint)
    scene = scene_of(obj)
    renderer = OpenCVRenderer()
    original = renderer.render(scene, alpha=True).buffer
    copied = obj.clone()
    assert copied.fill.paint == obj.fill.paint and copied.fill.paint is not obj.fill.paint
    loaded = Scene.from_json(scene.to_json())
    assert scene.to_dict()["version"] == "1.4"
    assert np.array_equal(original, renderer.render(loaded, alpha=True).buffer)
    with scene.edit(obj):
        obj.fill.paint.stops = (GradientStop(0, Color.green()), GradientStop(1, Color.white()))
    edited = renderer.render(scene, alpha=True).buffer
    assert scene.undo() and np.array_equal(original, renderer.render(scene, alpha=True).buffer)
    assert scene.redo() and np.array_equal(edited, renderer.render(scene, alpha=True).buffer)
    if isinstance(paint, LinearGradient):
        scene.animate(obj, "fill.paint.end", Point(100, 20), Point(180, 20), duration=1)
    else:
        scene.animate(obj, "fill.paint.radius", 20, 70, duration=1)
        scene.animate(obj, "fill.paint.center.x", 50, 80, duration=1)
    saved = scene.to_json()
    a = scene.render_at_time(.25, alpha=True).buffer
    b = scene.render_at_time(.75, alpha=True).buffer
    assert not np.array_equal(a, b)
    assert np.array_equal(a, Scene.from_json(saved).render_at_time(.25, alpha=True).buffer)
    assert scene.to_json() == saved and scene.get(obj.id) is obj
    with pytest.raises(ValidationError):
        scene.animate(obj, "fill.paint.stops", STOPS, STOPS)


@pytest.mark.parametrize("kind", ["circle", "ellipse", "rounded", "polygon", "arc", "triangle", "diamond", "circlehead", "text"])
def test_gradient_coverage_matches_solid_fill(kind):
    fill = FillStyle(color=Color.red())
    if kind == "circle":
        obj = Circle(center=Point(60, 50), radius=40, fill=fill)
    elif kind == "ellipse":
        obj = Ellipse(center=Point(60, 50), radius_x=50, radius_y=30, fill=fill)
    elif kind == "rounded":
        obj = RoundedRectangle(x=10, y=10, width=100, height=80, corner_radius=20, fill=fill)
    elif kind == "polygon":
        obj = Polygon(vertices=[Point(10, 10), Point(110, 50), Point(10, 90)], fill=fill)
    elif kind == "arc":
        obj = Arc(center=Point(60, 50), radius_x=50, radius_y=40, sweep_angle=270, closure=ArcClosure.PIE, fill=fill)
    elif kind == "text":
        obj = Text("Paint", position=Point(15, 50), color=Color(0, 0, 0, 0), background_fill=fill, padding=10,
                   transform=Transform(rotation=10))
        fill = obj.background_fill
    else:
        style = {"triangle": ArrowHeadStyle.TRIANGLE, "diamond": ArrowHeadStyle.DIAMOND, "circlehead": ArrowHeadStyle.CIRCLE}[kind]
        obj = Arrow(start=Point(10, 50), end=Point(110, 50), head_length=40, head_width=50,
                    head_style=style, fill=fill, stroke=StrokeStyle(color=Color.black()))
    solid = pixels(obj)
    fill.paint = linear()
    gradient = pixels(obj)
    assert np.count_nonzero(gradient[..., 3]) > 50
    assert np.array_equal(solid[..., 3], gradient[..., 3])
    assert not np.array_equal(solid[..., :3], gradient[..., :3])


@pytest.mark.parametrize("alpha", [False, True])
def test_render_drawable_and_bgr_scene_match_opaque_bgra(alpha):
    obj = shape(linear(stops=(GradientStop(0, Color(255, 0, 0, .7)), GradientStop(1, Color(0, 0, 255, .2)))))
    scene = scene_of(obj)
    scene.background = Color(80, 150, 30)
    renderer = OpenCVRenderer()
    reference = renderer.render(scene, alpha=True).buffer
    actual = renderer.render(scene, alpha=alpha).buffer
    canvas = Canvas(240, 180, alpha=alpha)
    canvas.clear(scene.background)
    renderer.render_drawable(obj, canvas)
    assert np.array_equal(actual, reference if alpha else reference[..., :3])
    assert np.array_equal(canvas.buffer, actual)


def test_schema_12_solid_scene_migration():
    scene = scene_of(Rectangle(width=40, height=40, fill=FillStyle(color=Color.red())))
    old = scene.to_dict()
    old["version"] = "1.2"
    loaded = Scene.from_dict(old)
    assert loaded.to_dict()["version"] == "1.4"
    assert np.array_equal(OpenCVRenderer().render(scene).buffer, OpenCVRenderer().render(loaded).buffer)


def test_invalid_animated_radius_restores_scene():
    obj = shape(RadialGradient(Point(50, 50), 40, STOPS))
    scene = scene_of(obj)
    scene.animate(obj, "fill.paint.radius", 40, -40, duration=1)
    before = scene.to_json()
    with pytest.raises(ValidationError):
        scene.render_at_time(.75, alpha=True)
    assert scene.to_json() == before
    assert scene.get(obj.id) is obj
