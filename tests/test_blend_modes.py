"""Comprehensive unit, integration, regression, and contract tests for DrawCV blend modes."""

from __future__ import annotations
import json
import math
import numpy as np
import pytest

from drawcv import (
    BlendMode,
    Canvas,
    Circle,
    Color,
    FillStyle,
    ClipPath,
    Group,
    Layer,
    Line,
    LinearGradient,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    ShadowEffect,
    StrokeStyle,
    ValidationError,
    CURRENT_SCHEMA_VERSION,
)
from drawcv.core.enums import coerce_blend_mode
from drawcv.compositing.blend import blend_rgb
from drawcv.compositing.compositor import composite_blend
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipRect
from drawcv.effects.mask import Mask
from drawcv.renderer import _IsolatedSurface


# =============================================================================
# 1. Enums and Coercion Tests
# =============================================================================

class TestBlendModeEnumsAndCoercion:
    def test_blend_mode_members(self):
        expected_modes = {
            "normal", "multiply", "screen", "overlay", "darken", "lighten",
            "color_dodge", "color_burn", "hard_light", "soft_light",
            "difference", "exclusion"
        }
        actual_modes = {m.value for m in BlendMode}
        assert actual_modes == expected_modes

    def test_top_level_export(self):
        import drawcv
        assert hasattr(drawcv, "BlendMode")
        assert "BlendMode" in drawcv.__all__
        # coerce_blend_mode must remain internal
        assert not hasattr(drawcv, "coerce_blend_mode")
        assert "coerce_blend_mode" not in drawcv.__all__

    def test_internal_coerce_blend_mode(self):
        # Enum passthrough
        assert coerce_blend_mode(BlendMode.MULTIPLY) is BlendMode.MULTIPLY

        # Case-insensitive string normalization
        assert coerce_blend_mode("multiply") is BlendMode.MULTIPLY
        assert coerce_blend_mode("MULTIPLY") is BlendMode.MULTIPLY
        assert coerce_blend_mode("Color_Dodge") is BlendMode.COLOR_DODGE
        assert coerce_blend_mode("color-dodge") is BlendMode.COLOR_DODGE
        assert coerce_blend_mode(" hard_light ") is BlendMode.HARD_LIGHT

        # Invalid strings and types raise ValidationError
        with pytest.raises(ValidationError, match=r"(?i)blendmode"):
            coerce_blend_mode("invalid_mode")
        with pytest.raises(ValidationError, match=r"(?i)blend_mode"):
            coerce_blend_mode(123)
        with pytest.raises(ValidationError, match=r"(?i)blend_mode"):
            coerce_blend_mode(None)

    def test_drawable_and_layer_property_coercion(self):
        rect = Rectangle(width=100, height=100)
        assert rect.blend_mode == BlendMode.NORMAL

        rect.blend_mode = "screen"
        assert rect.blend_mode == BlendMode.SCREEN

        rect.blend_mode = BlendMode.OVERLAY
        assert rect.blend_mode == BlendMode.OVERLAY

        with pytest.raises(ValidationError):
            rect.blend_mode = "not_a_mode"

        layer = Layer("test_layer", blend_mode="darken")
        assert layer.blend_mode == BlendMode.DARKEN

        layer.blend_mode = BlendMode.LIGHTEN
        assert layer.blend_mode == BlendMode.LIGHTEN

        with pytest.raises(ValidationError):
            layer.blend_mode = "unknown"

        group = Group()
        assert group.blend_mode == BlendMode.NORMAL
        group.blend_mode = "color_burn"
        assert group.blend_mode == BlendMode.COLOR_BURN


# =============================================================================
# 2. Pure RGB Blend Mathematical Truth Tables
# =============================================================================

class TestBlendMathTruthTables:
    @pytest.mark.parametrize("mode, cb, cs, expected", [
        (BlendMode.NORMAL, 0.4, 0.7, 0.7),
        (BlendMode.MULTIPLY, 0.4, 0.7, 0.28),
        (BlendMode.SCREEN, 0.4, 0.7, 1.0 - (1.0 - 0.4) * (1.0 - 0.7)),  # 0.82
        (BlendMode.OVERLAY, 0.4, 0.7, 2.0 * 0.4 * 0.7),  # 0.56 (since cb <= 0.5)
        (BlendMode.OVERLAY, 0.8, 0.7, 1.0 - 2.0 * (1.0 - 0.8) * (1.0 - 0.7)),  # 0.88 (cb > 0.5)
        (BlendMode.DARKEN, 0.4, 0.7, 0.4),
        (BlendMode.LIGHTEN, 0.4, 0.7, 0.7),
        (BlendMode.COLOR_DODGE, 0.4, 0.2, min(1.0, 0.4 / (1.0 - 0.2))),  # 0.5
        (BlendMode.COLOR_BURN, 0.4, 0.7, 1.0 - min(1.0, (1.0 - 0.4) / 0.7)),
        (BlendMode.HARD_LIGHT, 0.4, 0.2, 2.0 * 0.4 * 0.2),  # 0.16 (since cs <= 0.5)
        (BlendMode.HARD_LIGHT, 0.4, 0.7, 1.0 - 2.0 * (1.0 - 0.4) * (1.0 - 0.7)),  # 0.64 (cs > 0.5)
        (BlendMode.DIFFERENCE, 0.4, 0.7, abs(0.4 - 0.7)),  # 0.3
        (BlendMode.EXCLUSION, 0.4, 0.7, 0.4 + 0.7 - 2.0 * 0.4 * 0.7),  # 0.54
    ])
    def test_analytic_truth_table(self, mode, cb, cs, expected):
        cb_arr = np.array([[[cb, cb, cb]]], dtype=np.float32)
        cs_arr = np.array([[[cs, cs, cs]]], dtype=np.float32)
        result = blend_rgb(cb_arr, cs_arr, mode)
        np.testing.assert_allclose(result[0, 0, 0], expected, atol=1e-5)

    def test_soft_light_branches(self):
        # cs <= 0.5 branch
        cb_arr = np.array([[[0.4, 0.4, 0.4]]], dtype=np.float32)
        cs_arr = np.array([[[0.3, 0.3, 0.3]]], dtype=np.float32)
        res1 = blend_rgb(cb_arr, cs_arr, BlendMode.SOFT_LIGHT)
        # cb - (1 - 2*cs) * cb * (1 - cb) = 0.4 - 0.4 * 0.4 * 0.6 = 0.304
        np.testing.assert_allclose(res1[0, 0, 0], 0.304, atol=1e-5)

        # cs > 0.5, cb <= 0.25 branch: D(cb) = ((16*cb - 12)*cb + 4)*cb
        cb_low = np.array([[[0.2, 0.2, 0.2]]], dtype=np.float32)
        cs_high = np.array([[[0.8, 0.8, 0.8]]], dtype=np.float32)
        res2 = blend_rgb(cb_low, cs_high, BlendMode.SOFT_LIGHT)
        d_val = ((16.0 * 0.2 - 12.0) * 0.2 + 4.0) * 0.2
        exp2 = 0.2 + (2.0 * 0.8 - 1.0) * (d_val - 0.2)
        np.testing.assert_allclose(res2[0, 0, 0], exp2, atol=1e-5)

        # cs > 0.5, cb > 0.25 branch: D(cb) = sqrt(cb)
        cb_med = np.array([[[0.49, 0.49, 0.49]]], dtype=np.float32)
        res3 = blend_rgb(cb_med, cs_high, BlendMode.SOFT_LIGHT)
        exp3 = 0.49 + (2.0 * 0.8 - 1.0) * (0.7 - 0.49)
        np.testing.assert_allclose(res3[0, 0, 0], exp3, atol=1e-5)

    def test_color_dodge_and_burn_singularities(self):
        # Dodge: cs = 1.0 should yield 1.0 (or 0.0 if cb == 0.0)
        cb_zero = np.zeros((1, 1, 3), dtype=np.float32)
        cb_half = np.full((1, 1, 3), 0.5, dtype=np.float32)
        cs_one = np.ones((1, 1, 3), dtype=np.float32)

        res_dodge_zero = blend_rgb(cb_zero, cs_one, BlendMode.COLOR_DODGE)
        assert res_dodge_zero[0, 0, 0] == 0.0
        assert np.all(np.isfinite(res_dodge_zero))

        res_dodge_half = blend_rgb(cb_half, cs_one, BlendMode.COLOR_DODGE)
        assert res_dodge_half[0, 0, 0] == 1.0
        assert np.all(np.isfinite(res_dodge_half))

        # Burn: cs = 0.0 should yield 0.0 (or 1.0 if cb == 1.0)
        cb_one = np.ones((1, 1, 3), dtype=np.float32)
        cs_zero = np.zeros((1, 1, 3), dtype=np.float32)

        res_burn_one = blend_rgb(cb_one, cs_zero, BlendMode.COLOR_BURN)
        assert res_burn_one[0, 0, 0] == 1.0
        assert np.all(np.isfinite(res_burn_one))

        res_burn_half = blend_rgb(cb_half, cs_zero, BlendMode.COLOR_BURN)
        assert res_burn_half[0, 0, 0] == 0.0
        assert np.all(np.isfinite(res_burn_half))


# =============================================================================
# 3. Alpha-Aware Compositing Interactions
# =============================================================================

class TestCompositingAlphaInteractions:
    def test_alpha_boundaries_isolated_surface(self):
        dst = np.zeros((10, 10, 4), dtype=np.float32)
        src = np.zeros((10, 10, 4), dtype=np.float32)

        # Transparent src into transparent dst -> all zero
        composite_blend(dst, src, (0, 0, 10, 10), BlendMode.MULTIPLY, is_destination_isolated=True)
        assert np.all(dst == 0.0)

        # Opaque src into transparent dst -> exact src
        src[:, :, :3] = 128.0
        src[:, :, 3] = 1.0  # premultiplied
        composite_blend(dst, src, (0, 0, 10, 10), BlendMode.MULTIPLY, is_destination_isolated=True)
        np.testing.assert_allclose(dst[:, :, :3], 128.0, atol=1e-4)
        np.testing.assert_allclose(dst[:, :, 3], 1.0, atol=1e-4)

        # Transparent src into non-transparent dst -> dst unchanged
        src_zero = np.zeros((10, 10, 4), dtype=np.float32)
        dst_copy = dst.copy()
        composite_blend(dst, src_zero, (0, 0, 10, 10), BlendMode.MULTIPLY, is_destination_isolated=True)
        np.testing.assert_allclose(dst, dst_copy)

    def test_tiny_nonzero_alpha_no_nan(self):
        dst = np.full((10, 10, 4), 1e-6, dtype=np.float32)
        src = np.full((10, 10, 4), 1e-6, dtype=np.float32)
        for mode in BlendMode:
            composite_blend(dst, src, (0, 0, 10, 10), mode, is_destination_isolated=True)
            assert np.all(np.isfinite(dst))
            assert np.all(dst >= 0.0)

    def test_partial_alpha_w3c_formula_isolated(self):
        # Backdrop: color (200, 100, 50), alpha 0.6 -> premultiplied = (120, 60, 30, 0.6)
        dst = np.zeros((1, 1, 4), dtype=np.float32)
        dst[0, 0, :3] = np.array([200.0, 100.0, 50.0]) * 0.6
        dst[0, 0, 3] = 0.6

        # Source: color (50, 150, 250), alpha 0.4 -> premultiplied = (20, 60, 100, 0.4)
        src = np.zeros((1, 1, 4), dtype=np.float32)
        src[0, 0, :3] = np.array([50.0, 150.0, 250.0]) * 0.4
        src[0, 0, 3] = 0.4

        # Using Multiply mode
        cb = np.array([200.0, 100.0, 50.0]) / 255.0
        cs = np.array([50.0, 150.0, 250.0]) / 255.0
        b_val = cb * cs
        b_255 = b_val * 255.0

        # W3C formula: co_pm = (1 - ab) * cs_pm + (1 - as) * cb_pm + as * ab * B(cb, cs)
        expected_pm_rgb = (1.0 - 0.6) * src[0, 0, :3] + (1.0 - 0.4) * dst[0, 0, :3] + (0.4 * 0.6) * b_255
        expected_alpha = 0.4 + 0.6 * (1.0 - 0.4)

        composite_blend(dst, src, (0, 0, 1, 1), BlendMode.MULTIPLY, is_destination_isolated=True)

        np.testing.assert_allclose(dst[0, 0, :3], expected_pm_rgb, atol=1e-4)
        np.testing.assert_allclose(dst[0, 0, 3], expected_alpha, atol=1e-4)

    def test_canvas_bgr_opaque_backdrop(self):
        # BGR canvas: uint8 [100, 150, 200]
        dst = np.array([[[100, 150, 200]]], dtype=np.uint8)

        # Source: color (50, 80, 120), alpha 0.5 -> premultiplied = (25, 40, 60, 0.5)
        src = np.zeros((1, 1, 4), dtype=np.float32)
        src[0, 0, :3] = [25.0, 40.0, 60.0]
        src[0, 0, 3] = 0.5

        # Backdrop cb
        cb = np.array([100.0, 150.0, 200.0]) / 255.0
        cs = np.array([50.0, 80.0, 120.0]) / 255.0
        b_255 = (cb * cs) * 255.0
        expected = np.clip(np.round(np.array([100.0, 150.0, 200.0]) * 0.5 + b_255 * 0.5), 0, 255).astype(np.uint8)

        composite_blend(dst, src, (0, 0, 1, 1), BlendMode.MULTIPLY, is_destination_isolated=False)
        np.testing.assert_array_equal(dst[0, 0], expected)


# =============================================================================
# 4. Exact NORMAL Path Preservation (Bit-for-bit)
# =============================================================================

class TestExactNormalPreservation:
    def test_bit_for_bit_normal_isolated(self):
        np.random.seed(42)
        dst = np.random.uniform(0, 255, (50, 50, 4)).astype(np.float32)
        dst[:, :, 3] = np.random.uniform(0, 1, (50, 50))
        dst[:, :, :3] *= dst[:, :, 3:4]  # valid premultiplied

        src = np.random.uniform(0, 255, (50, 50, 4)).astype(np.float32)
        src[:, :, 3] = np.random.uniform(0, 1, (50, 50))
        src[:, :, :3] *= src[:, :, 3:4]

        # Compute legacy inline formula
        expected_dst = dst.copy()
        sub_src = src[10:40, 10:40]
        src_rgb = sub_src[:, :, :3]
        src_a = sub_src[:, :, 3:4]
        sub_dst = expected_dst[10:40, 10:40]
        dst_rgb = sub_dst[:, :, :3]
        dst_a = sub_dst[:, :, 3:4]
        sub_dst[:, :, :3] = src_rgb + dst_rgb * (1.0 - src_a)
        sub_dst[:, :, 3] = (src_a + dst_a * (1.0 - src_a))[:, :, 0]

        # Call composite_blend with BlendMode.NORMAL
        actual_dst = dst.copy()
        composite_blend(actual_dst, src, (10, 10, 40, 40), BlendMode.NORMAL, is_destination_isolated=True)

        assert np.array_equal(actual_dst, expected_dst)

    def test_bit_for_bit_normal_canvas_bgr(self):
        np.random.seed(42)
        dst = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        src = np.random.uniform(0, 255, (50, 50, 4)).astype(np.float32)
        src[:, :, 3] = np.random.uniform(0, 1, (50, 50))
        src[:, :, :3] *= src[:, :, 3:4]

        # Compute legacy inline formula
        expected_dst = dst.copy()
        sub_src = src[10:40, 10:40]
        src_rgb = sub_src[:, :, :3]
        src_a = sub_src[:, :, 3:4]
        sub_dst = expected_dst[10:40, 10:40].astype(np.float32)
        blended = sub_dst * (1.0 - src_a) + src_rgb
        expected_dst[10:40, 10:40] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

        # Call composite_blend with BlendMode.NORMAL
        actual_dst = dst.copy()
        composite_blend(actual_dst, src, (10, 10, 40, 40), BlendMode.NORMAL, is_destination_isolated=False)

        assert np.array_equal(actual_dst, expected_dst)


# =============================================================================
# 5. Local Isolation vs Global Alpha Pipeline
# =============================================================================

class TestIsolationSemantics:
    def test_blend_mode_does_not_force_global_alpha_pipeline(self):
        scene = Scene(200, 200, background=Color.white())
        rect = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        rect.blend_mode = BlendMode.MULTIPLY
        scene.add(rect)

        renderer = OpenCVRenderer()
        # Scene has no alpha background, no transparent shapes, so _requires_alpha_pipeline must remain False!
        assert not renderer._requires_alpha_pipeline(scene)

    def test_blend_mode_forces_local_isolated_compositing(self):
        renderer = OpenCVRenderer()
        rect = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        assert not renderer._needs_isolated_compositing(rect)

        rect.blend_mode = BlendMode.MULTIPLY
        assert renderer._needs_isolated_compositing(rect)

        rect.blend_mode = BlendMode.NORMAL
        assert not renderer._needs_isolated_compositing(rect)


# =============================================================================
# 6. Effects, Mask, Clip, and Opacity Interactions
# =============================================================================

class TestPipelineInteractions:
    def test_blend_mode_with_opacity(self):
        # Draw backdrop: red rectangle
        scene = Scene(100, 100, background=Color.white())
        r1 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(255, 0, 0)))
        # Overlay blue with opacity 0.5 and MULTIPLY
        r2 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color(0, 0, 255)))
        r2.blend_mode = BlendMode.MULTIPLY
        r2.opacity = 0.5
        scene.add(r1)
        scene.add(r2)

        renderer = OpenCVRenderer()
        rendered = renderer.render(scene).buffer

        # Red = BGR (0, 0, 255)
        # Multiply pure red and pure blue = black (0, 0, 0)
        # With opacity 0.5 on blue: blended = 0.5 * red + 0.5 * black = (0, 0, 128)
        pixel = rendered[50, 50]
        np.testing.assert_allclose(pixel, [0, 0, 128], atol=2)

    def test_blend_mode_with_clip(self):
        scene = Scene(100, 100, background=Color.white())
        # Base rectangle
        r1 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        # Clipped shape with multiply
        r2 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.blue()))
        r2.blend_mode = BlendMode.MULTIPLY
        r2.clip = ClipRect(x=0, y=0, width=50, height=100)
        scene.add(r1)
        scene.add(r2)

        rendered = OpenCVRenderer().render(scene).buffer
        # Left half should be blended (black), right half unchanged (red)
        assert np.array_equal(rendered[50, 25], [0, 0, 0])
        assert np.array_equal(rendered[50, 75], [0, 0, 255])

    def test_blend_mode_with_mask(self):
        scene = Scene(100, 100, background=Color.white())
        r1 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))

        # Mask buffer: half 255, half 0
        mask_buf = np.zeros((100, 100), dtype=np.uint8)
        mask_buf[:, :50] = 255
        mask = Mask(mask_buf)

        r2 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.blue()))
        r2.blend_mode = BlendMode.MULTIPLY
        r2.mask = mask
        scene.add(r1)
        scene.add(r2)

        rendered = OpenCVRenderer().render(scene).buffer
        assert np.array_equal(rendered[50, 25], [0, 0, 0])
        assert np.array_equal(rendered[50, 75], [0, 0, 255])

    def test_blend_mode_with_blur(self):
        scene = Scene(100, 100, background=Color.white())
        r1 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.white()))
        c = Circle(center=Point(50, 50), radius=20, fill=FillStyle(color=Color.black()))
        c.blend_mode = BlendMode.MULTIPLY
        c.effects.append(BlurEffect(kernel_size=5))
        scene.add(r1)
        scene.add(c)

        rendered = OpenCVRenderer().render(scene).buffer
        # Blurred circle edges should have a smooth gradient from black towards white
        center_val = rendered[50, 50, 0]
        mid_val = rendered[50, 68, 0]
        outer_val = rendered[50, 90, 0]
        assert center_val < mid_val < outer_val

    def test_blend_mode_with_shadow(self):
        scene = Scene(120, 120, background=Color.white())
        c = Circle(center=Point(50, 50), radius=25, fill=FillStyle(color=Color.red()))
        c.blend_mode = BlendMode.MULTIPLY
        c.effects.append(ShadowEffect(offset_x=10.0, offset_y=10.0, blur_radius=3.0))
        scene.add(c)
        buf = OpenCVRenderer().render(scene).buffer
        # Shadow should darken area around (75, 75) outside the circle
        shadow_pixel = buf[75, 75]
        white_pixel = buf[115, 115]
        assert np.all(shadow_pixel < white_pixel)

    def test_blend_mode_with_clip_path(self):
        scene = Scene(100, 100, background=Color.white())
        r1 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        r2 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.blue()))
        r2.blend_mode = BlendMode.MULTIPLY
        r2.clip = ClipPath([Point(0, 0), Point(50, 0), Point(50, 100), Point(0, 100)])
        scene.add(r1)
        scene.add(r2)
        buf = OpenCVRenderer().render(scene).buffer
        # Inside clip (x=25): red * blue = black
        assert np.array_equal(buf[50, 25], [0, 0, 0])
        # Outside clip (x=75): red
        assert np.array_equal(buf[50, 75], [0, 0, 255])


# =============================================================================
# 7. Groups and Layers
# =============================================================================

class TestGroupAndLayerCompositing:
    def test_group_blend_mode_and_clone(self):
        group = Group(blend_mode=BlendMode.SCREEN)
        assert group.blend_mode == BlendMode.SCREEN

        c1 = Circle(center=Point(30, 30), radius=20, fill=FillStyle(color=Color.red()))
        group.add(c1)

        cloned = group.clone()
        assert cloned.blend_mode == BlendMode.SCREEN
        assert len(cloned.children) == 1

    def test_group_isolated_render_with_blend_mode(self):
        scene = Scene(100, 100, background=Color.black())
        group = Group(blend_mode=BlendMode.SCREEN)
        r1 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.red()))
        r2 = Rectangle(position=Point(30, 10), width=40, height=40, fill=FillStyle(color=Color.green()))
        group.add(r1)
        group.add(r2)
        scene.add(group)

        rendered = OpenCVRenderer().render(scene).buffer
        # In intersection (x=35, y=20), r2 is over r1 with NORMAL inside group
        # Red is (0, 0, 255), Green is (0, 255, 0)
        # Inside group: r2 covers r1, so pixel is green
        # Then group composites over black with SCREEN -> still green
        assert np.array_equal(rendered[20, 35], [0, 255, 0])

    def test_layer_blend_mode_and_roundtrip(self):
        scene = Scene(100, 100)
        layer = scene.create_layer("overlay_layer", blend_mode=BlendMode.OVERLAY)
        assert layer.blend_mode == BlendMode.OVERLAY

        # Serialize layer
        d = layer.to_dict()
        assert d["blend_mode"] == "overlay"

        # Deserialize layer
        loaded = Layer.from_dict(d)
        assert loaded.blend_mode == BlendMode.OVERLAY
        assert loaded.name == "overlay_layer"

    def test_drawable_blend_mode_undo_redo(self):
        scene = Scene(100, 100)
        rect = Rectangle(width=50, height=50, id="r1")
        scene.add(rect)
        assert rect.blend_mode == BlendMode.NORMAL
        with scene.edit(rect):
            rect.blend_mode = BlendMode.MULTIPLY
        assert rect.blend_mode == BlendMode.MULTIPLY
        assert scene.undo()
        assert rect.blend_mode == BlendMode.NORMAL
        assert scene.redo()
        assert rect.blend_mode == BlendMode.MULTIPLY

    def test_group_blend_mode_undo_redo(self):
        scene = Scene(100, 100)
        group = Group(id="g1")
        scene.add(group)
        assert group.blend_mode == BlendMode.NORMAL
        with scene.edit(group):
            group.blend_mode = BlendMode.SCREEN
        assert group.blend_mode == BlendMode.SCREEN
        assert scene.undo()
        assert group.blend_mode == BlendMode.NORMAL
        assert scene.redo()
        assert group.blend_mode == BlendMode.SCREEN

    def test_child_non_normal_blend_mode_inside_normal_group(self):
        scene = Scene(100, 100, background=Color.white())
        group = Group(blend_mode=BlendMode.NORMAL)
        r1 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color.red()))
        r2 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color.blue()), blend_mode=BlendMode.MULTIPLY)
        group.add(r1)
        group.add(r2)
        scene.add(group)
        buf = OpenCVRenderer().render(scene).buffer
        # Child 2 blends with multiply over child 1 -> black
        assert np.array_equal(buf[50, 50], [0, 0, 0])

    def test_child_blend_mode_inside_group_with_non_normal_blend_mode(self):
        scene = Scene(100, 100, background=Color.black())
        group = Group(blend_mode=BlendMode.SCREEN)
        r1 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color.red()))
        r2 = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color.blue()), blend_mode=BlendMode.MULTIPLY)
        group.add(r1)
        group.add(r2)
        scene.add(group)
        buf = OpenCVRenderer().render(scene).buffer
        # Inside group: red * blue = black
        # Group SCREEN over black: black SCREEN black = black
        assert np.array_equal(buf[50, 50], [0, 0, 0])


# =============================================================================
# 8. Serialization, Round-Trip, and Migration
# =============================================================================

class TestSerializationAndMigration:
    def test_scene_round_trip_with_blend_modes(self):
        scene = Scene(200, 200, background=Color(20, 30, 40))
        layer = scene.create_layer("effect_layer", blend_mode=BlendMode.MULTIPLY)
        rect = Rectangle(width=50, height=50, fill=FillStyle(color=Color.red()), blend_mode=BlendMode.COLOR_DODGE)
        layer.add(rect)

        json_str = scene.to_json()
        restored = Scene.from_json(json_str)

        assert restored.to_dict()["version"] == CURRENT_SCHEMA_VERSION
        assert len(restored.layers) == 2  # default layer + effect_layer
        restored_layer = restored.get_layer("effect_layer")
        assert restored_layer.blend_mode == BlendMode.MULTIPLY
        assert restored_layer.objects[0].blend_mode == BlendMode.COLOR_DODGE

    def test_schema_1_4_to_1_5_migration(self):
        legacy_doc = {
            "format": "drawcv",
            "version": "1.4",
            "scene": {
                "width": 100,
                "height": 100,
                "background": {"r": 0, "g": 0, "b": 0, "a": 1.0},
                "layers": [
                    {
                        "name": "default",
                        "visible": True,
                        "locked": False,
                        "opacity": 1.0,
                        "objects": [
                            {
                                "type": "rectangle",
                                "id": "r1",
                                "name": "Rect",
                                "visible": True,
                                "locked": False,
                                "opacity": 1.0,
                                "z_index": 0,
                                "tags": [],
                                "metadata": {},
                                "transform": {
                                    "translation_x": 0.0,
                                    "translation_y": 0.0,
                                    "rotation": 0.0,
                                    "scale_x": 1.0,
                                    "scale_y": 1.0,
                                    "pivot": None,
                                },
                                "clip": None,
                                "mask": None,
                                "effects": [],
                                "position": {"x": 0.0, "y": 0.0},
                                "width": 50.0,
                                "height": 50.0,
                                "stroke": None,
                                "fill": None,
                            }
                        ],
                    }
                ],
            },
        }

        loaded = Scene.from_dict(legacy_doc)
        assert loaded.to_dict()["version"] == CURRENT_SCHEMA_VERSION
        assert loaded.layers[0].blend_mode == BlendMode.NORMAL
        assert loaded.layers[0].objects[0].blend_mode == BlendMode.NORMAL

    def test_invalid_blend_mode_from_json_raises_validation_error(self):
        bad_doc = {
            "format": "drawcv",
            "version": "1.5",
            "scene": {
                "width": 100,
                "height": 100,
                "background": {"r": 0, "g": 0, "b": 0, "a": 1.0},
                "layers": [
                    {
                        "name": "default",
                        "visible": True,
                        "locked": False,
                        "opacity": 1.0,
                        "blend_mode": "invalid_mode_name",
                        "objects": [],
                    }
                ],
            },
        }
        with pytest.raises(ValidationError):
            Scene.from_dict(bad_doc)


# =============================================================================
# 9. Animation Rejection Regression Test
# =============================================================================

class TestAnimationRejection:
    def test_blend_mode_animation_rejected_with_validation_error(self):
        scene = Scene(200, 200)
        rect = Rectangle(width=50, height=50)
        scene.add(rect)

        with pytest.raises(ValidationError, match="discrete property 'blend_mode'"):
            scene.animate(rect, "blend_mode", BlendMode.NORMAL, BlendMode.MULTIPLY, duration=1.0)

        # Also test layer blend_mode rejection
        with pytest.raises(ValidationError, match="discrete property 'blend_mode'"):
            scene.animate(scene.layers[0], "blend_mode", BlendMode.NORMAL, BlendMode.SCREEN, duration=1.0)

    def test_cannot_enter_numeric_interpolation(self):
        from drawcv.animation.track import AnimationTrack
        rect = Rectangle(width=50, height=50)
        with pytest.raises(ValidationError, match="discrete property 'blend_mode'"):
            AnimationTrack(target_id=rect.id, property_path="blend_mode", start_value="normal", end_value="multiply")


# =============================================================================
# 10. SVG Export
# =============================================================================

class TestSvgExport:
    def test_svg_vector_blend_mode_style(self):
        scene = Scene(200, 200, background=Color.white())
        rect = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        rect.blend_mode = BlendMode.MULTIPLY
        scene.add(rect)

        result = scene.export_svg()
        assert 'mix-blend-mode: multiply;' in result.svg
        # No fallback because rect is vector-supported
        assert len(result.fallbacks) == 0

    def test_svg_raster_fallback_blend_mode_style(self):
        scene = Scene(200, 200, background=Color.white())
        rect = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        rect.blend_mode = BlendMode.COLOR_DODGE
        # Adding an effect triggers SVG raster fallback
        rect.effects.append(BlurEffect(kernel_size=5))
        scene.add(rect)

        result = scene.export_svg()
        assert 'mix-blend-mode: color-dodge;' in result.svg
        assert len(result.fallbacks) == 1
        assert result.fallbacks[0].reason == "effects"

    def test_svg_resvg_comparison_if_available(self):
        resvg = pytest.importorskip("resvg_py")
        scene = Scene(100, 100, background=Color.white())
        r1 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()))
        r2 = Rectangle(width=100, height=100, fill=FillStyle(color=Color.blue()))
        r2.blend_mode = BlendMode.MULTIPLY
        scene.add(r1)
        scene.add(r2)

        export = scene.export_svg()
        png_bytes = resvg.svg_to_bytes(svg_string=export.svg, width=100, height=100, skip_system_fonts=True)
        import cv2
        resvg_pixels = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        drawcv_pixels = OpenCVRenderer().render(scene).buffer

        # Compare central pixel: Red * Blue = Black in both
        np.testing.assert_allclose(resvg_pixels[50, 50], drawcv_pixels[50, 50], atol=5)

    def test_svg_blend_plus_mask_raster_fallback(self):
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(width=100, height=100, fill=FillStyle(color=Color.red()), blend_mode=BlendMode.MULTIPLY)
        mask_buf = np.zeros((100, 100), dtype=np.uint8)
        mask_buf[:, :50] = 255
        rect.mask = Mask(mask_buf)
        scene.add(rect)
        result = scene.export_svg()
        assert 'mix-blend-mode: multiply;' in result.svg
        assert len(result.fallbacks) == 1
        assert result.fallbacks[0].reason == "mask"
        assert 'data-drawcv-raster="mask"' in result.svg

    def test_svg_group_blend_plus_child_blend(self):
        scene = Scene(100, 100, background=Color.white())
        group = Group(blend_mode=BlendMode.SCREEN)
        rect = Rectangle(width=50, height=50, fill=FillStyle(color=Color.blue()), blend_mode=BlendMode.MULTIPLY)
        group.add(rect)
        scene.add(group)
        result = scene.export_svg()
        assert 'mix-blend-mode: screen;' in result.svg
        assert 'mix-blend-mode: multiply;' in result.svg


# =============================================================================
# 11. Positional Backwards Compatibility Tests
# =============================================================================

class TestPositionalBackwardsCompatibility:
    def test_layer_positional_signature_unchanged(self):
        # Pre-existing positional signature: Layer(name, visible, locked, opacity, z_order)
        layer = Layer("my_layer", False, True, 0.75, 42)
        assert layer.name == "my_layer"
        assert layer.visible is False
        assert layer.locked is True
        assert layer.opacity == 0.75
        assert layer.z_order == 42
        assert layer.blend_mode == BlendMode.NORMAL

    def test_scene_create_layer_positional_signature_unchanged(self):
        scene = Scene(100, 100)
        # Pre-existing positional signature: create_layer(name, z_order, visible, locked, opacity)
        layer = scene.create_layer("my_layer", 15, False, True, 0.6)
        assert layer.name == "my_layer"
        assert layer.z_order == 15
        assert layer.visible is False
        assert layer.locked is True
        assert layer.opacity == 0.6
        assert layer.blend_mode == BlendMode.NORMAL

    def test_group_positional_signature_unchanged(self):
        c1 = Circle(center=Point(10, 10), radius=5)
        group = Group([c1])
        assert len(group.children) == 1
        assert group.blend_mode == BlendMode.NORMAL

    def test_shape_dataclass_positional_signatures_unchanged(self):
        # Positional args on Rectangle:
        # Arg 0: supports_progressive_rendering (bool)
        # Arg 1: id (str)
        # Arg 2: name (str)
        # Arg 3: visible (bool)
        # Arg 4: locked (bool)
        # Arg 5: opacity (float)
        # Arg 6: z_index (int)
        # If blend_mode were not kw_only, Arg 6 would have shifted to blend_mode!
        r = Rectangle(False, "my_rect", "RectName", True, False, 0.75, 12)
        assert r.id == "my_rect"
        assert r.name == "RectName"
        assert r.opacity == 0.75
        assert r.z_index == 12
        assert r.blend_mode == BlendMode.NORMAL

        c = Circle(False, "my_circle", "CircleName", True, False, 0.5, 9)
        assert c.id == "my_circle"
        assert c.name == "CircleName"
        assert c.opacity == 0.5
        assert c.z_index == 9
        assert c.blend_mode == BlendMode.NORMAL

        line = Line(False, "my_line", "LineName", True, False, 0.9, 3)
        assert line.id == "my_line"
        assert line.name == "LineName"
        assert line.opacity == 0.9
        assert line.z_index == 3
        assert line.blend_mode == BlendMode.NORMAL

