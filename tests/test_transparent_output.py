"""Output-alpha contracts checked independently of renderer compositing helpers."""
import cv2
import numpy as np
import pytest

from drawcv import (Canvas, Scene, OpenCVRenderer, Color, Point, Rectangle, Circle,
                    FillStyle, StrokeStyle, Group, Transform, ImageObject,
                    ImageInterpolation, Mask, MaskMapping, ClipRect, ClipPath,
                    BlurEffect, BlurType, ShadowEffect, Text, ValidationError,
                    VideoRenderer, Line)


def scene_with(*objects, background=Color(0, 0, 0, 0)):
    scene = Scene(100, 100, background=background)
    for obj in objects:
        scene.add(obj)
    return scene


def rect(color=Color.red(), **kwargs):
    return Rectangle(position=Point(20, 20), width=60, height=60,
                     fill=FillStyle(color=color), **kwargs)


def render(scene):
    return OpenCVRenderer().render(scene, alpha=True)


def external_over(bgra, background_bgr):
    a = bgra[..., 3:4].astype(float) / 255
    return np.rint(bgra[..., :3] * a + np.array(background_bgr) * (1 - a)).astype(np.uint8)


def test_canvas_explicit_mode_copy_clear_and_flatten():
    assert Canvas(4, 3).buffer.shape == (3, 4, 3)
    canvas = Canvas(4, 3, alpha=True)
    assert canvas.has_alpha and np.all(canvas.buffer == 0)
    canvas.clear(Color(255, 0, 0, .5))
    assert tuple(canvas.buffer[0, 0]) == (0, 0, 255, 128)
    copied = canvas.to_numpy()
    copied[:] = 0
    assert canvas.buffer[0, 0, 3] == 128
    assert np.array_equal(canvas.flatten(Color.white()).buffer, external_over(canvas.buffer, (255, 255, 255)))
    assert not canvas.flatten(Color.black()).has_alpha
    canvas.clear(Color(255, 0, 0, 0))
    assert np.all(canvas.buffer == 0)
    with pytest.raises(ValidationError):
        canvas.flatten(Color(0, 0, 0, .5))


@pytest.mark.parametrize("alpha,shape", [(False, (3, 4, 4)), (True, (3, 4, 3))])
def test_buffer_channel_mismatch_rejected(alpha, shape):
    with pytest.raises(ValidationError):
        Canvas(4, 3, np.zeros(shape, np.uint8), alpha=alpha)


@pytest.mark.parametrize("invalid", [None, 1, "BGRA"])
def test_output_mode_requires_boolean(invalid):
    with pytest.raises(ValidationError):
        Canvas(4, 4, alpha=invalid)
    with pytest.raises(ValidationError):
        OpenCVRenderer().render(scene_with(), alpha=invalid)
    with pytest.raises(ValidationError):
        scene_with().render_at_time(0, alpha=invalid)


def test_background_alpha_is_honored_only_when_requested():
    scene = scene_with(background=Color(120, 60, 20, .25))
    assert tuple(render(scene).buffer[0, 0]) == (20, 60, 120, 64)
    assert tuple(OpenCVRenderer().render(scene).buffer[0, 0]) == (20, 60, 120)
    scene.background = Color(255, 80, 40, 0)
    assert np.all(render(scene).buffer == 0)


def test_straight_output_and_independent_source_over_equation():
    # Red .5 over blue .25: a=.625, straight RGB=(204,0,51).
    scene = scene_with(rect(Color(255, 0, 0, .5)), background=Color(0, 0, 255, .25))
    assert tuple(render(scene).buffer[50, 50]) == (51, 0, 204, 159)


def test_group_opacity_applies_once_after_child_overlap():
    a = rect()
    b = rect(Color.blue())
    b.move(20, 0)
    group = Group(children=[a, b], opacity=.5)
    pixels = render(scene_with(group)).buffer
    assert tuple(pixels[50, 30]) == (0, 0, 255, 128)
    assert tuple(pixels[50, 50]) == (255, 0, 0, 128)


def test_object_opacity_applies_once_to_overlapping_fill_and_stroke():
    obj = rect(opacity=.5, stroke=StrokeStyle(color=Color.blue(), width=12))
    pixels = render(scene_with(obj)).buffer
    assert pixels[50, 22, 3] == pixels[50, 50, 3] == 128
    assert tuple(pixels[50, 22, :3]) == (255, 0, 0)


def test_nested_layer_group_object_style_color_mask_opacities():
    obj = rect(Color(255, 0, 0, .5), opacity=.5)
    obj.fill.opacity = .5
    obj.mask = Mask(np.full((100, 100), 128, np.uint8), mapping=MaskMapping.ABSOLUTE)
    obj.clip = ClipRect(20, 20, 35, 60)
    group = Group(children=[obj], opacity=.5)
    scene = scene_with()
    layer = scene.create_layer("art", opacity=.8)
    layer.add(group)
    pixels = render(scene).buffer
    expected_alpha = round(255 * .5 * .5 * .5 * (128/255) * .5 * .8)
    assert tuple(pixels[40, 40]) == (0, 0, 255, expected_alpha)
    assert np.all(pixels[40, 70] == 0)


@pytest.mark.parametrize("interpolation", list(ImageInterpolation))
def test_hidden_rgb_cannot_leak_through_resize_warp_or_blur(interpolation):
    source = np.zeros((12, 12, 4), np.uint8)
    source[3:9, 3:9] = (0, 0, 255, 128)
    dirty = source.copy()
    dirty[dirty[..., 3] == 0, :3] = (250, 255, 30)
    def draw(image):
        obj = ImageObject(image, position=Point(20, 20), width=57, height=49,
                          interpolation=interpolation, opacity=.7,
                          transform=Transform(rotation=17), effects=[BlurEffect(kernel_size=7, sigma=1.2)])
        return render(scene_with(Group(children=[obj], opacity=.6))).buffer
    clean_pixels, dirty_pixels = draw(source), draw(dirty)
    assert np.array_equal(clean_pixels, dirty_pixels)
    painted = dirty_pixels[..., 3] > 0
    assert painted.any()
    assert np.all(dirty_pixels[painted, :3] == (0, 0, 255))
    assert np.all(dirty_pixels[~painted, :3] == 0)


def test_image_alpha_and_object_opacity_are_each_applied_once():
    image = np.full((20, 20, 4), (40, 80, 200, 128), np.uint8)
    obj = ImageObject(image, position=Point(20, 20), opacity=.5)
    output = render(scene_with(Group(children=[obj], opacity=.5))).buffer
    assert tuple(output[30, 30]) == (40, 80, 200, 32)


@pytest.mark.parametrize("channels", [0, 1, 3])
def test_grayscale_and_bgr_images_become_opaque_before_object_opacity(channels):
    shape = (12, 12) if channels == 0 else (12, 12, channels)
    obj = ImageObject(np.full(shape, 90, np.uint8), position=Point(20, 20),
                      width=35, height=35, opacity=.5)
    assert tuple(render(scene_with(obj)).buffer[30, 30]) == (90, 90, 90, 128)


@pytest.mark.parametrize("blur_type", list(BlurType))
def test_blurred_red_png_has_no_fringe_over_any_background(tmp_path, blur_type):
    obj = rect(Color(255, 0, 0, .5), transform=Transform(rotation=23),
               effects=[BlurEffect(kernel_size=15, sigma=2, blur_type=blur_type)])
    scene = scene_with(Group(children=[obj], opacity=.6))
    canvas = render(scene)
    path = tmp_path / "red.png"
    canvas.save(path)
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(decoded, canvas.buffer)
    alpha = decoded[..., 3]
    edge = (alpha > 0) & (alpha < 70)
    assert edge.any()
    assert np.all(decoded[edge, :3] == (0, 0, 255))
    for bgr in [(255, 255, 255), (0, 0, 0), (35, 90, 160)]:
        external = external_over(decoded, bgr)
        scene.background = Color.from_bgr(*bgr)
        opaque_render = render(scene).buffer[..., :3]
        assert np.abs(external.astype(int)-opaque_render.astype(int)).max() <= 1


def test_colored_shadow_preserves_alpha_and_hue():
    obj = Rectangle(position=Point(15, 20), width=20, height=20,
                    fill=FillStyle(color=Color(255, 0, 0, .5)), opacity=.5,
                    effects=[ShadowEffect(offset_x=35, offset_y=0, blur_radius=1,
                                          color=Color(0, 0, 255, .8), opacity=.5)])
    pixels = render(scene_with(obj)).buffer
    assert tuple(pixels[30, 25]) == (0, 0, 255, 64)
    assert tuple(pixels[30, 60, :3]) == (255, 0, 0)
    assert abs(int(pixels[30, 60, 3]) - 25.5) <= .5


def test_blur_uses_transparent_canvas_border_instead_of_reflection():
    obj = Rectangle(position=Point(0, 0), width=50, height=50,
                    fill=FillStyle(color=Color.red()),
                    effects=[BlurEffect(kernel_size=5, blur_type=BlurType.BOX)])
    pixels = render(scene_with(obj)).buffer
    assert pixels[0, 0, 3] == round(255 * 9/25)
    assert tuple(pixels[0, 0, :3]) == (0, 0, 255)
    assert pixels[10, 10, 3] == 255


@pytest.mark.parametrize("inverted", [False, True])
def test_absolute_mask_coverage_and_inversion(inverted):
    mask = np.tile(np.arange(100, dtype=np.uint8)*2, (100, 1))
    obj = rect(mask=Mask(mask, mapping=MaskMapping.ABSOLUTE, inverted=inverted))
    pixels = render(scene_with(obj)).buffer
    assert pixels[50, 30, 3] == (195 if inverted else 60)


def test_fit_bounds_mask_is_cropped_not_stretched_offscreen():
    mask = np.tile(np.arange(100, dtype=np.uint8)*2, (60, 1))
    obj = Rectangle(position=Point(-50, 20), width=100, height=60,
                    fill=FillStyle(color=Color.red()), mask=Mask(mask))
    pixels = render(scene_with(obj)).buffer
    assert pixels[40, 10, 3] == 120


@pytest.mark.parametrize("clip", [ClipRect(0, 0, 50, 100),
                                  ClipPath([Point(0, 0), Point(50, 0), Point(50, 100), Point(0, 100)])])
def test_layer_clip_clears_alpha_without_transform_method(clip):
    scene = scene_with()
    layer = scene.create_layer("clipped")
    layer.clip = clip
    layer.add(rect())
    pixels = render(scene).buffer
    assert pixels[50, 30, 3] == 255
    assert np.all(pixels[50, 70] == 0)


def test_transformed_clip_applies_after_blur():
    obj = rect(clip=ClipRect(20, 20, 30, 60), transform=Transform(translation_x=10),
               effects=[BlurEffect(kernel_size=7)])
    pixels = render(scene_with(obj)).buffer
    assert pixels[50, 40, 3] == 255
    assert np.all(pixels[50, 70] == 0)


def test_text_alpha_and_ancestor_transform():
    obj = Text("M", position=Point(10, 10), color=Color(255, 0, 0, .5),
               font_scale=1, thickness=3, opacity=.5)
    original = render(scene_with(obj)).buffer
    assert original[..., 3].max() == 64
    clone = obj.clone()
    pixels = render(scene_with(Group(children=[clone], opacity=.5,
                    transform=Transform(translation_x=30)))).buffer
    assert pixels[..., 3].max() == 32
    assert np.abs(pixels[:, 40:70, 3].astype(float) - original[:, 10:40, 3] / 2).max() <= 1


def test_render_drawable_to_existing_straight_bgra_canvas():
    canvas = Canvas(100, 100, alpha=True)
    canvas.clear(Color(0, 0, 255, 128/255))
    reference = canvas.buffer.copy()
    obj = rect(Color(255, 0, 0, 128/255))
    OpenCVRenderer().render_drawable(obj, canvas)
    assert tuple(canvas.buffer[50, 50]) == (85, 0, 170, 192)
    assert np.array_equal(canvas.buffer[0, 0], reference[0, 0])


def test_render_drawable_attached_to_parent_preserves_inherited_opacity():
    obj = rect(opacity=.5)
    group = Group(children=[obj], opacity=.5)
    canvas = Canvas(100, 100, alpha=True)
    OpenCVRenderer().render_drawable(obj, canvas)
    assert tuple(canvas.buffer[50, 50]) == (0, 0, 255, 64)
    assert obj.parent is group


def test_text_glyphs_and_plate_share_parent_affine_transform():
    transform = Transform(rotation=20, translation_x=20, pivot=Point(0, 0))
    obj = Text("Hi", position=Point(10, 10), thickness=2,
               background_fill=FillStyle(color=Color(255, 0, 0, .5)),
               padding=4, background_radius=4, opacity=.5)
    own = obj.clone()
    own.transform = transform.copy()
    parent = Group(children=[obj], transform=transform)
    assert np.array_equal(render(scene_with(own)).buffer, render(scene_with(parent)).buffer)


def test_cropped_image_keeps_source_alpha_and_source_bytes():
    from drawcv import BoundingBox
    image = np.full((20, 20, 4), (0, 255, 0, 0), np.uint8)
    image[5:15, 5:15] = (20, 60, 220, 128)
    original = image.copy()
    obj = ImageObject(image, crop=BoundingBox(5, 5, 10, 10),
                      position=Point(20, 20), width=40, height=40)
    pixels = render(scene_with(obj)).buffer
    assert tuple(pixels[40, 40]) == (20, 60, 220, 128)
    assert np.array_equal(image, original)


def test_failed_alpha_animation_render_restores_authored_state():
    scene = scene_with(rect())
    obj = scene.objects[0]
    scene.animate(obj, "opacity", .1, .9, duration=1)
    before = scene.to_json()
    class FailingRenderer(OpenCVRenderer):
        def _render_single_primitive(self, drawable, canvas):
            raise RuntimeError("test raster failure")
    renderer = FailingRenderer()
    with pytest.raises(RuntimeError):
        scene.render_at_time(.5, renderer, alpha=True)
    assert scene.to_json() == before
    assert scene.get(obj.id) is obj
    assert renderer._current_isolating_ancestor is None


def test_alpha_export_refuses_silent_loss(tmp_path):
    canvas = Canvas(5, 5, alpha=True)
    with pytest.raises(ValidationError):
        canvas.save(tmp_path / "bad.jpg")
    assert not (tmp_path / "bad.jpg").exists()
    canvas.flatten(Color.white()).save(tmp_path / "good.jpg")
    with pytest.raises(ValidationError):
        VideoRenderer.render_image_sequence(scene_with(), tmp_path / "bad", pattern="%d.jpg", alpha=True)


def test_animation_png_sequence_and_serialization_are_observational(tmp_path):
    obj = Line(start=Point(10, 40), end=Point(90, 40), stroke=StrokeStyle(color=Color.red(), width=12))
    scene = scene_with(obj)
    scene.animate(obj, "opacity", .2, .8, duration=1)
    before = scene.to_json()
    counts = scene.history.undo_count, scene.history.redo_count
    a = scene.render_at_time(.25, alpha=True).buffer
    scene.render_at_time(.75, alpha=True)
    assert np.array_equal(a, scene.render_at_time(.25, alpha=True).buffer)
    assert scene.get(obj.id) is obj
    paths = VideoRenderer.render_image_sequence(scene, tmp_path, duration=.5, fps=4, alpha=True)
    assert len(paths) == 2
    assert cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED).shape == (100, 100, 4)
    loaded = Scene.from_json(before)
    assert np.array_equal(a, loaded.render_at_time(.25, alpha=True).buffer)
    assert scene.to_json() == before
    assert (scene.history.undo_count, scene.history.redo_count) == counts


def test_bgr_defaults_and_custom_renderer_signature_still_work():
    scene = scene_with(rect(Color(255, 0, 0, .5)))
    renderer = OpenCVRenderer()
    assert np.array_equal(renderer.render(scene).buffer, renderer.render(scene, alpha=False).buffer)
    assert renderer.render(scene).buffer.shape == (100, 100, 3)
    class ExistingRenderer:
        def render(self, scene):
            return "legacy renderer"
    assert scene.render_at_time(0, ExistingRenderer()) == "legacy renderer"
    assert all(not frame.has_alpha for frame in VideoRenderer.render_frames(scene, duration=.1, fps=10))


def test_full_alpha_example_roundtrip_clone_and_history():
    from examples.transparent_output import build_scene
    scene = build_scene()
    original = render(scene).buffer.copy()
    saved = scene.to_json()
    loaded = Scene.from_json(saved)
    assert np.array_equal(original, render(loaded).buffer)
    group, image = scene.objects[:2]
    cloned_scene = Scene(scene.width, scene.height, background=scene.background)
    for obj in scene.objects:
        cloned_scene.add(obj.clone())
    assert np.array_equal(original, render(cloned_scene).buffer)
    with scene.edit(group, image):
        group.opacity = .25
        image.opacity = .2
    edited = render(scene).buffer.copy()
    assert not np.array_equal(edited, original)
    assert scene.undo()
    assert np.array_equal(original, render(scene).buffer)
    assert scene.get(image.id) is image and scene.get(group.id) is group
    assert scene.redo()
    assert np.array_equal(edited, render(scene).buffer)
