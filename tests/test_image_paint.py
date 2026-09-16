"""Tests for ImagePaint construction, repeat modes, coordinate composition, and compounding transforms."""

import numpy as np
import pytest

from drawcv import (
    Color,
    FillStyle,
    Group,
    ImageInterpolation,
    ImagePaint,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    StrokeStyle,
    Transform,
    ValidationError,
)
from drawcv.core.alpha import unpremultiply
from drawcv.core.paint_sampling import sample_image


def sample_2x2_texture():
    # 2x2 texture with distinctive colors:
    # (0,0): Red (0,0,255), (1,0): Green (0,255,0)
    # (0,1): Blue (255,0,0), (1,1): Yellow (0,255,255)
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[0, 0] = [0, 0, 255, 255]      # Top-left: Red
    img[0, 1] = [0, 255, 0, 255]      # Top-right: Green
    img[1, 0] = [255, 0, 0, 255]      # Bottom-left: Blue
    img[1, 1] = [0, 255, 255, 255]    # Bottom-right: Yellow
    return img


def render_paint(paint, width=100, height=100):
    rect = Rectangle(position=Point(0, 0), width=width, height=height, fill=FillStyle(paint=paint))
    scene = Scene(width, height, background=Color(0, 0, 0, 0))
    scene.add(rect)
    return OpenCVRenderer().render(scene, alpha=True).buffer


def test_image_paint_basic_and_nearest():
    img = sample_2x2_texture()
    paint = ImagePaint(
        image=img,
        scale=(10.0, 10.0),
        interpolation=ImageInterpolation.NEAREST,
        repeat="repeat",
    )
    buf = render_paint(paint)

    # Within (0,0) to (10,10): texel (0,0) = Red
    assert tuple(buf[5, 5]) == (0, 0, 255, 255)
    # Within (10,0) to (20,10): texel (1,0) = Green
    assert tuple(buf[5, 15]) == (0, 255, 0, 255)
    # Within (0,10) to (10,20): texel (0,1) = Blue
    assert tuple(buf[15, 5]) == (255, 0, 0, 255)
    # Within (10,10) to (20,20): texel (1,1) = Yellow
    assert tuple(buf[15, 15]) == (0, 255, 255, 255)

    # Repeating period: at x=25, y=5 -> repeats to x=5 -> Red
    assert tuple(buf[5, 25]) == (0, 0, 255, 255)


def test_image_paint_repeat_modes():
    img = sample_2x2_texture()

    # 1. repeat="none": coordinates outside (0, 0) to (scale*W, scale*H) are transparent
    paint_none = ImagePaint(
        image=img,
        scale=(10.0, 10.0),
        repeat="none",
        interpolation=ImageInterpolation.NEAREST,
    )
    buf_none = render_paint(paint_none)
    assert tuple(buf_none[5, 5]) == (0, 0, 255, 255)       # Inside (0,0)-(20,20)
    assert tuple(buf_none[50, 50]) == (0, 0, 0, 0)         # Outside -> transparent

    # 2. repeat="pad": coordinates outside are clamped to border texels
    paint_pad = ImagePaint(
        image=img,
        scale=(10.0, 10.0),
        repeat="pad",
        interpolation=ImageInterpolation.NEAREST,
    )
    buf_pad = render_paint(paint_pad)
    assert tuple(buf_pad[5, 5]) == (0, 0, 255, 255)
    # At (50, 50), x >= 20 and y >= 20 -> clamps to bottom-right texel (1,1) = Yellow
    assert tuple(buf_pad[50, 50]) == (0, 255, 255, 255)

    # 3. repeat="reflect": reflects across boundaries
    paint_reflect = ImagePaint(
        image=img,
        scale=(10.0, 10.0),
        repeat="reflect",
        interpolation=ImageInterpolation.NEAREST,
    )
    buf_ref = render_paint(paint_reflect)
    # Cycle 0: [Red, Green]
    assert tuple(buf_ref[5, 5]) == (0, 0, 255, 255)
    assert tuple(buf_ref[5, 15]) == (0, 255, 0, 255)
    # Cycle 1 (reflected): [Green, Red]
    assert tuple(buf_ref[5, 25]) == (0, 255, 0, 255)
    assert tuple(buf_ref[5, 35]) == (0, 0, 255, 255)


def test_image_paint_compounding_scale():
    # Verify that paint scale and paint Transform scale compound predictably
    img = sample_2x2_texture()

    # Image scale is 10x, and Transform scale is 2x -> compounded scale is 20x per texel
    paint = ImagePaint(
        image=img,
        scale=(10.0, 10.0),
        transform=Transform(scale_x=2.0, scale_y=2.0),
        interpolation=ImageInterpolation.NEAREST,
        repeat="repeat",
    )
    buf = render_paint(paint)

    # With total 20x scale, texel (0,0) spans [0, 20] x [0, 20]
    assert tuple(buf[10, 10]) == (0, 0, 255, 255)          # Still Red at (10, 10)
    assert tuple(buf[10, 25]) == (0, 255, 0, 255)          # Green at (25, 10)


def test_image_paint_origin_and_rotation():
    img = sample_2x2_texture()

    # Shift origin to (20, 20)
    paint_shifted = ImagePaint(
        image=img,
        origin=Point(20, 20),
        scale=(10.0, 10.0),
        interpolation=ImageInterpolation.NEAREST,
        repeat="none",
    )
    buf = render_paint(paint_shifted)
    assert tuple(buf[10, 10]) == (0, 0, 0, 0)              # (10, 10) is before origin -> transparent
    assert tuple(buf[25, 25]) == (0, 0, 255, 255)          # (25, 25) is inside texel (0,0) -> Red


def test_image_paint_nested_entity_transforms():
    img = sample_2x2_texture()
    paint = ImagePaint(image=img, scale=(10.0, 10.0), space="object")
    rect = Rectangle(position=Point(0, 0), width=40, height=40, fill=FillStyle(paint=paint))
    rect.transform = Transform(translation_x=10, translation_y=10)
    group = Group(children=[rect], transform=Transform(rotation=45, pivot=Point(30, 30)))

    scene = Scene(100, 100, background=Color(0, 0, 0, 0))
    scene.add(group)
    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    assert np.count_nonzero(buf[..., 3]) > 0


def test_image_paint_copy_immutability():
    original_arr = sample_2x2_texture()
    paint = ImagePaint(image=original_arr)

    # Mutating original numpy array must not mutate the paint's internal buffer
    original_arr[0, 0] = [123, 123, 123, 123]
    assert paint.image[0, 0, 0] != 123

    # Cloned paint must have an independent image buffer
    cloned = paint.copy()
    cloned.image[0, 0] = [77, 77, 77, 77]
    assert paint.image[0, 0, 0] != 77


def test_image_paint_serialization_roundtrip():
    img = sample_2x2_texture()
    paint = ImagePaint(
        image=img,
        origin=Point(5, 10),
        scale=(2.5, 3.5),
        repeat="reflect",
        space="world",
        opacity=0.75,
        transform=Transform(rotation=30),
    )
    data = paint.to_dict()
    assert data["type"] == "image"
    assert data["origin"] == {"x": 5.0, "y": 10.0}
    assert data["scale"] == [2.5, 3.5]
    assert data["repeat"] == "reflect"
    assert data["space"] == "world"
    assert data["opacity"] == 0.75
    assert data["transform"]["rotation"] == 30.0

    restored = ImagePaint.from_dict(data)
    assert restored.origin == paint.origin
    assert restored.scale == paint.scale
    assert restored.repeat == paint.repeat
    assert restored.space == paint.space
    assert restored.opacity == paint.opacity
    assert restored.transform.rotation == 30.0
    assert np.array_equal(restored.image, paint.image)


def test_image_paint_exact_1to1_nearest_and_linear():
    tex = np.array([
        [[10, 20, 30, 255], [40, 50, 60, 255]],
        [[70, 80, 90, 255], [100, 110, 120, 255]],
    ], dtype=np.uint8)

    for interp in (ImageInterpolation.NEAREST, ImageInterpolation.LINEAR):
        paint = ImagePaint(
            image=tex,
            scale=(1.0, 1.0),
            origin=Point(0.0, 0.0),
            interpolation=interp,
            repeat="repeat",
        )
        sampled = sample_image(paint, np.eye(3), 2, 2)
        unpm = unpremultiply(sampled)
        np.testing.assert_array_equal(unpm, tex)

        rect = Rectangle(position=Point(0, 0), width=2, height=2, fill=FillStyle(paint=paint))
        scene = Scene(2, 2, background=Color(0, 0, 0, 0))
        scene.add(rect)
        buf = OpenCVRenderer().render(scene, alpha=True).buffer
        np.testing.assert_array_equal(buf, tex)


def test_image_paint_scaled_linear_boundary():
    tex = np.array([
        [[0, 0, 0, 255], [100, 100, 100, 255]]
    ], dtype=np.uint8)

    paint = ImagePaint(
        image=tex,
        scale=(2.0, 1.0),
        interpolation=ImageInterpolation.LINEAR,
        repeat="pad",
    )
    sampled = sample_image(paint, np.eye(3), 4, 1)
    vals = sampled[0, :, 0]

    assert np.isclose(vals[0], 0.0, atol=1e-3)
    assert np.isclose(vals[1], 25.0, atol=1e-3)
    assert np.isclose(vals[2], 75.0, atol=1e-3)
    assert np.isclose(vals[3], 100.0, atol=1e-3)


def test_image_paint_scale_and_origin_validation():
    img = sample_2x2_texture()

    # Zero scale
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(0.0, 1.0))
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(1.0, 0.0))

    # Negative scale
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(-1.0, 1.0))
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(1.0, -2.0))

    # NaN / Inf scale
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(float("nan"), 1.0))
    with pytest.raises(ValidationError, match="scale factors must be positive finite"):
        ImagePaint(image=img, scale=(1.0, float("inf")))

    # Malformed scale sequences
    with pytest.raises(ValidationError, match="scale must have exactly 2 elements"):
        ImagePaint(image=img, scale=(1.0,))
    with pytest.raises(ValidationError, match="scale must have exactly 2 elements"):
        ImagePaint(image=img, scale=[1.0, 2.0, 3.0])
    with pytest.raises(ValidationError, match="scale cannot be a boolean"):
        ImagePaint(image=img, scale=True)
    with pytest.raises(ValidationError, match="scale factors cannot be booleans"):
        ImagePaint(image=img, scale=(True, 1.0))
    with pytest.raises(ValidationError, match="scale factors must be numeric"):
        ImagePaint(image=img, scale=("abc", 1.0))

    # Malformed origin sequences
    with pytest.raises(ValidationError, match="origin sequence must have exactly 2 elements"):
        ImagePaint(image=img, origin=(1.0,))
    with pytest.raises(ValidationError, match="origin sequence must have exactly 2 elements"):
        ImagePaint(image=img, origin=[1.0, 2.0, 3.0])
    with pytest.raises(ValidationError, match="origin cannot be a boolean"):
        ImagePaint(image=img, origin=False)
    with pytest.raises(ValidationError, match="origin coordinates cannot be booleans"):
        ImagePaint(image=img, origin=(True, 0.0))
    with pytest.raises(ValidationError, match="origin must be a Point or 2-element sequence"):
        ImagePaint(image=img, origin="bad_origin")
    with pytest.raises(ValidationError, match="coordinates must be finite"):
        ImagePaint(image=img, origin=(float("nan"), 0.0))
    with pytest.raises(ValidationError, match="coordinates must be finite"):
        ImagePaint(image=img, origin=(float("inf"), 0.0))


def test_image_paint_safe_equality():
    img = sample_2x2_texture()
    p1 = ImagePaint(image=img, scale=(2.0, 2.0))
    p2 = p1.copy()

    assert p1 == p2

    img_diff = img.copy()
    img_diff[0, 0] = [12, 34, 56, 255]
    p_diff_img = ImagePaint(image=img_diff, scale=(2.0, 2.0))
    assert p1 != p_diff_img

    p_diff_scale = ImagePaint(image=img, scale=(3.0, 2.0))
    assert p1 != p_diff_scale

    assert p1 != "string"
    assert p1 != 123
    assert p1 is not None
    assert p1 != None

    fill1 = FillStyle(paint=p1)
    fill2 = FillStyle(paint=p2)
    assert fill1 == fill2
    assert fill1 != FillStyle(paint=p_diff_img)

    stroke1 = StrokeStyle(paint=p1)
    stroke2 = StrokeStyle(paint=p2)
    assert stroke1 == stroke2
    assert stroke1 != StrokeStyle(paint=p_diff_img)
    assert fill1 != stroke1


def test_image_paint_copy_does_not_serialize(monkeypatch):
    img = sample_2x2_texture()
    p = ImagePaint(image=img, scale=(2.0, 3.0))

    import drawcv.core.raster as raster_mod
    encode_called = False
    original_encode = raster_mod.encode_raster_png

    def tracking_encode(*args, **kwargs):
        nonlocal encode_called
        encode_called = True
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(raster_mod, "encode_raster_png", tracking_encode)

    p_copied = p.copy()
    assert not encode_called, "ImagePaint.copy() should not encode/serialize image to PNG!"
    assert p_copied.image is not p.image
    np.testing.assert_array_equal(p_copied.image, p.image)

    p_copied.image[0, 0] = [88, 88, 88, 88]
    assert not np.array_equal(p_copied.image, p.image)


def test_image_paint_alpha_and_premultiplied_filtering():
    tex = np.zeros((2, 2, 4), dtype=np.uint8)
    tex[0, 0] = [0, 0, 255, 255]
    tex[0, 1] = [0, 255, 0, 0]
    tex[1, 0] = [255, 0, 0, 128]
    tex[1, 1] = [255, 255, 255, 255]

    paint = ImagePaint(image=tex, scale=(1.0, 1.0), interpolation=ImageInterpolation.NEAREST)
    sampled = sample_image(paint, np.eye(3), 2, 2)
    assert sampled[0, 0, 3] == 1.0
    assert sampled[0, 1, 3] == 0.0
    assert np.isclose(sampled[1, 0, 3], 128.0 / 255.0, atol=0.01)
    assert sampled[1, 1, 3] == 1.0

    paint_half = ImagePaint(image=tex, scale=(1.0, 1.0), opacity=0.5, interpolation=ImageInterpolation.NEAREST)
    sampled_half = sample_image(paint_half, np.eye(3), 2, 2)
    assert np.isclose(sampled_half[0, 0, 3], 0.5, atol=0.01)
    assert sampled_half[0, 1, 3] == 0.0

    rect = Rectangle(position=Point(0, 0), width=2, height=2, fill=FillStyle(paint=paint_half, opacity=0.5))
    scene = Scene(2, 2, background=Color(0, 0, 0, 0))
    scene.add(rect)
    buf = OpenCVRenderer().render(scene, alpha=True).buffer
    assert np.isclose(buf[0, 0, 3], 64, atol=2)

    paint_lin = ImagePaint(image=tex, scale=(2.0, 1.0), interpolation=ImageInterpolation.LINEAR, repeat="pad")
    sampled_lin = sample_image(paint_lin, np.eye(3), 4, 1)
    unpm = unpremultiply(sampled_lin)
    assert np.isclose(unpm[0, 1, 2], 255, atol=2), "Red channel should remain un-darkened under premultiplied filtering"

    paint_none = ImagePaint(image=tex, scale=(1.0, 1.0), interpolation=ImageInterpolation.LINEAR, repeat="none")
    sampled_none = sample_image(paint_none, np.eye(3), 6, 6)
    assert np.all(sampled_none[4:, 4:, 3] == 0.0)
    assert sampled_none[0, 0, 3] > sampled_none[0, 1, 3]


def test_image_paint_area_interpolation_rejected():
    img = sample_2x2_texture()
    with pytest.raises(ValidationError, match="ImageInterpolation.AREA is unsupported for spatial ImagePaint"):
        ImagePaint(image=img, interpolation=ImageInterpolation.AREA)
    with pytest.raises(ValidationError, match="ImageInterpolation.AREA is unsupported for spatial ImagePaint"):
        ImagePaint(image=img, interpolation="area")


def test_image_paint_all_supported_interpolations_smoke():
    img = sample_2x2_texture()
    for interp in (
        ImageInterpolation.NEAREST,
        ImageInterpolation.LINEAR,
        ImageInterpolation.CUBIC,
        ImageInterpolation.LANCZOS,
    ):
        paint = ImagePaint(image=img, scale=(2.0, 2.0), interpolation=interp, repeat="repeat")
        sampled = sample_image(paint, np.eye(3), 10, 10)
        assert sampled.shape == (10, 10, 4)
        assert np.any(sampled[..., 3] > 0)
        unpm = unpremultiply(sampled)
        assert unpm.shape == (10, 10, 4)

        rect = Rectangle(position=Point(0, 0), width=10, height=10, fill=FillStyle(paint=paint))
        scene = Scene(10, 10, background=Color(0, 0, 0, 0))
        scene.add(rect)
        buf = OpenCVRenderer().render(scene, alpha=True).buffer
        assert buf.shape == (10, 10, 4)
        assert np.any(buf[..., 3] > 0)
