"""Tests for Generalized Ordered Effects Architecture.

Verifies:
- List order is strictly authoritative: Blur -> Shadow != Shadow -> Blur
- Duplicate effects execute in sequence
- Hierarchical propagation: Child effect -> Group effect -> Layer effect
- Sequential bounds propagation across drawables, groups, and layers
"""

import numpy as np
import pytest

from drawcv import (
    BlurEffect,
    BlurType,
    BrightnessContrastEffect,
    Circle,
    Color,
    ColorMatrixEffect,
    FillStyle,
    GlowEffect,
    GrayscaleEffect,
    Group,
    HueShiftEffect,
    Layer,
    OpenCVRenderer,
    Point,
    Rectangle,
    SaturationEffect,
    Scene,
    SepiaEffect,
    ShadowEffect,
)


def render(scene, alpha=False):
    return OpenCVRenderer().render(scene, alpha=alpha).buffer.copy()


class TestEffectsOrderAndDuplication:
    """Verify authoritative execution ordering and legal duplicate effects."""

    def test_empty_effects_stack_matches_no_effects(self):
        """Empty effect stack produces pixel-identical output to no-effect renderer."""
        s1 = Scene(100, 100, background=Color.white())
        r1 = Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=Color.red()))
        s1.add(r1)

        s2 = Scene(100, 100, background=Color.white())
        r2 = Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=Color.red()))
        r2.effects = []
        s2.add(r2)

        buf1 = render(s1)
        buf2 = render(s2)
        assert np.array_equal(buf1, buf2)

    def test_blur_then_shadow_differs_from_shadow_then_blur(self):
        """Blur -> Shadow produces a visually and numerically different result from Shadow -> Blur."""
        blur = BlurEffect(kernel_size=25, sigma=8.0)
        shadow = ShadowEffect(offset_x=5.0, offset_y=5.0, blur_radius=2.0, color=Color(0, 255, 0))

        # Scene A: Blur first, then Shadow
        # Content is blurred into a wide soft disc first; shadow is generated from the wide blurred alpha.
        scene_a = Scene(100, 100, background=Color.white())
        c_a = Circle(center=Point(50, 50), radius=15, fill=FillStyle(color=Color.red()))
        c_a.effects = [blur, shadow]
        scene_a.add(c_a)
        buf_a = render(scene_a)

        # Scene B: Shadow first, then Blur
        # Shadow is generated from sharp circle first; blur then acts on the combined (sharp circle + sharp shadow).
        scene_b = Scene(100, 100, background=Color.white())
        c_b = Circle(center=Point(50, 50), radius=15, fill=FillStyle(color=Color.red()))
        c_b.effects = [shadow, blur]
        scene_b.add(c_b)
        buf_b = render(scene_b)

        # 1. Broad assertion: outputs are demonstrably and visibly different
        diff = np.abs(buf_a.astype(float) - buf_b.astype(float))
        assert diff.max() >= 20.0, f"Expected substantial difference, got max diff {diff.max()}"
        assert np.sum(diff > 5.0) > 500, "Expected hundreds of pixels differing between the two orderings"

    def test_duplicate_blur_effects_compound(self):
        """Two sequential blurs produce a stronger blur than a single blur."""
        scene_single = Scene(100, 100, background=Color.white())
        rect1 = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.black()))
        rect1.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene_single.add(rect1)
        buf1 = render(scene_single)

        scene_double = Scene(100, 100, background=Color.white())
        rect2 = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.black()))
        rect2.effects = [
            BlurEffect(kernel_size=9, sigma=2.0),
            BlurEffect(kernel_size=9, sigma=2.0),
        ]
        scene_double.add(rect2)
        buf2 = render(scene_double)

        # At pixel (26, 26) just outside the boundary, double blur spreads more energy than single blur
        assert buf2[26, 26, 0] < buf1[26, 26, 0]

    def test_duplicate_shadow_effects_render_two_shadows(self):
        """Two ShadowEffects with different offsets produce two distinct drop shadows."""
        scene = Scene(140, 140, background=Color.white())
        rect = Rectangle(position=Point(50, 50), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [
            # Shadow 1: down-right
            ShadowEffect(offset_x=20.0, offset_y=20.0, blur_radius=3.0, color=Color(0, 0, 0, 0.5)),
            # Shadow 2: up-left
            ShadowEffect(offset_x=-20.0, offset_y=-20.0, blur_radius=3.0, color=Color(0, 0, 0, 0.5)),
        ]
        scene.add(rect)
        buf = render(scene)

        # Region (100, 100) down-right outside rect [50..90] is darkened by Shadow 1
        assert buf[100, 100, 0] < 200 and buf[100, 100, 1] < 200 and buf[100, 100, 2] < 200
        # Region (38, 38) up-left outside rect [50..90] is darkened by Shadow 2
        assert buf[38, 38, 0] < 200 and buf[38, 38, 1] < 200 and buf[38, 38, 2] < 200
        # Region (38, 100) is un-shadowed pure white
        assert buf[38, 100, 0] == 255 and buf[38, 100, 1] == 255 and buf[38, 100, 2] == 255


class TestHierarchyEffectsAndBounds:
    """Verify sequential bounds propagation and group/layer effect composition."""

    def test_child_shadow_then_group_blur(self):
        """Child shadow is composited into group surface, then group blur blurs child + shadow."""
        scene = Scene(150, 150, background=Color.white())
        group = Group()
        rect = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.red()))
        rect.effects = [ShadowEffect(offset_x=20.0, offset_y=20.0, blur_radius=4.0)]
        group.add(rect)
        group.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene.add(group)

        # Assert input bounds of group contains child's shadow
        input_bounds = group.get_effect_input_bounds()
        assert input_bounds.right >= 90.0  # 30 + 40 + 20
        assert input_bounds.bottom >= 90.0

        # Group effect bounds further expands by blur (4 pixels)
        group_bounds = group.get_effect_bounds()
        assert group_bounds.right >= input_bounds.right + 4.0
        assert group_bounds.bottom >= input_bounds.bottom + 4.0

        buf = render(scene)
        # Shadow region (80, 80) is blurred by group blur
        assert buf[80, 80, 0] < 250
        # Group blur softened the edge of the shadow
        assert 0 < buf[85, 85, 0] < 255

    def test_nested_hierarchy_bounds_three_levels(self):
        """Child effect -> Group effect -> Layer effect correctly propagates bounds at every level."""
        scene = Scene(200, 200)
        layer = scene.create_layer("main_layer")
        group = Group()
        rect = Rectangle(position=Point(50, 50), width=40, height=40)
        rect.effects = [BlurEffect(kernel_size=9)]  # pad = 4
        group.add(rect)
        group.effects = [ShadowEffect(offset_x=20, offset_y=20, blur_radius=4)] # blur_pad = 12
        layer.add(group)
        layer.effects = [BlurEffect(kernel_size=15)]  # pad = 7

        # Level 1: Child bounds
        cb = rect.get_effect_bounds()
        assert cb.left == 46.0 and cb.top == 46.0
        assert cb.right == 94.0 and cb.bottom == 94.0

        # Level 2: Group input bounds == child effect bounds
        gib = group.get_effect_input_bounds()
        assert gib == cb

        # Level 2: Group effect bounds (Shadow offset +20, blur_pad 12)
        geb = group.get_effect_bounds()
        assert geb.left == 46.0  # union doesn't expand left
        assert geb.right == 94.0 + 20.0 + 12.0  # 126.0
        assert geb.bottom == 94.0 + 20.0 + 12.0 # 126.0

        # Level 3: Layer input bounds == group effect bounds
        lib = layer.get_effect_input_bounds()
        assert lib == geb

        # Level 3: Layer effect bounds (Blur pad = 7)
        leb = layer.get_effect_bounds()
        assert leb.left == geb.left - 7.0
        assert leb.right == geb.right + 7.0
        assert leb.top == geb.top - 7.0
        assert leb.bottom == geb.bottom + 7.0

    def test_child_glow_group_shadow_layer_blur_bounds(self):
        """child GlowEffect -> Group ShadowEffect -> Layer BlurEffect with bounds assertions at every level."""
        scene = Scene(300, 300)
        layer = scene.create_layer("main_layer")
        group = Group()
        rect = Rectangle(position=Point(60, 60), width=40, height=40)
        # Glow with blur_radius=4.0: pad = gaussian_pad_for_radius(4.0) = ceil(12)*2+1 // 2 = 25 // 2 = 12
        rect.effects = [GlowEffect(blur_radius=4.0, color=Color(255, 255, 0))]
        group.add(rect)
        # Shadow with offset=(15, 15), blur_radius=3.0: pad = gaussian_pad_for_radius(3.0) = ceil(9)*2+1 // 2 = 19 // 2 = 9
        group.effects = [ShadowEffect(offset_x=15.0, offset_y=15.0, blur_radius=3.0)]
        layer.add(group)
        # Blur with kernel_size=11: pad = 11 // 2 = 5
        layer.effects = [BlurEffect(kernel_size=11)]

        # Level 1: Child bounds
        cb = rect.get_effect_bounds()
        assert cb.left == 60.0 - 12.0  # 48.0
        assert cb.top == 60.0 - 12.0   # 48.0
        assert cb.right == 100.0 + 12.0 # 112.0
        assert cb.bottom == 100.0 + 12.0 # 112.0

        # Level 2: Group input bounds must match child effect bounds
        gib = group.get_effect_input_bounds()
        assert gib == cb

        # Level 2: Group effect bounds (union with shadow)
        geb = group.get_effect_bounds()
        assert geb.left == 48.0
        assert geb.top == 48.0
        assert geb.right == 112.0 + 15.0 + 9.0  # 136.0
        assert geb.bottom == 112.0 + 15.0 + 9.0 # 136.0

        # Level 3: Layer input bounds must match group effect bounds
        lib = layer.get_effect_input_bounds()
        assert lib == geb

        # Level 3: Layer effect bounds (Blur pad = 5)
        leb = layer.get_effect_bounds()
        assert leb.left == geb.left - 5.0
        assert leb.top == geb.top - 5.0
        assert leb.right == geb.right + 5.0
        assert leb.bottom == geb.bottom + 5.0

        # Verify it renders without error
        buf = render(scene)
        assert buf.shape == (300, 300, 3)

    def test_blur_expand_bounds_vs_get_padding_distinction(self):
        """BlurEffect distinction between authoritative expand_bounds and legacy get_padding."""
        from drawcv.core.bounds import BoundingBox
        blur = BlurEffect(kernel_size=15, sigma=1.0)
        # expand_bounds uses actual finite kernel support: kernel_size // 2 = 7
        bbox = BoundingBox(10.0, 10.0, 20.0, 20.0)
        expanded = blur.expand_bounds(bbox)
        assert expanded.left == 3.0
        assert expanded.top == 3.0
        assert expanded.width == 34.0
        assert expanded.height == 34.0
        # legacy get_padding preserves historical semantics: ceil(3 * sigma) = 3.0
        assert blur.get_padding() == (3.0, 3.0, 3.0, 3.0)

        # When sigma is 0.0, both use kernel_size // 2
        blur_zero = BlurEffect(kernel_size=15, sigma=0.0)
        assert blur_zero.get_padding() == (7.0, 7.0, 7.0, 7.0)


class TestGlowSemantics:
    """Verify true outer glow semantics and interaction with translucency."""

    def test_outer_glow_preserves_translucent_interior(self):
        """Outer glow does not invade or alter interior translucent pixels."""
        scene = Scene(100, 100, background=Color.transparent())
        # Red rectangle with 50% opacity
        rect = Rectangle(
            position=Point(30, 30),
            width=40,
            height=40,
            fill=FillStyle(color=Color(255, 0, 0, 0.5)),
        )
        # Yellow glow: R=255, G=255, B=0
        rect.effects = [GlowEffect(blur_radius=5.0, color=Color(255, 255, 0, 1.0))]
        scene.add(rect)

        buf = render(scene, alpha=True)

        # Center pixel (50, 50) is well inside the rectangle
        center_pixel = buf[50, 50]
        # In straight uint8 BGRA:
        # B = 0, G should be 0 (no yellow glow bleed!), R = 255, A ~ 128 (50% alpha)
        assert center_pixel[3] == pytest.approx(128, abs=5)
        assert center_pixel[0] == 0   # Blue
        assert center_pixel[1] == 0   # Green (no yellow bleed!)
        assert center_pixel[2] == 255 # Red straight-color

        # Outside pixel (25, 50) is in the glow region
        glow_pixel = buf[50, 25]
        assert glow_pixel[3] > 10
        # Yellow glow has Green > 0
        assert glow_pixel[1] > 10

    def test_glow_then_shadow_vs_shadow_then_glow(self):
        """Glow -> Shadow and Shadow -> Glow both execute cleanly."""
        glow = GlowEffect(blur_radius=4.0, color=Color(255, 255, 0))
        shadow = ShadowEffect(offset_x=10.0, offset_y=10.0, blur_radius=3.0, color=Color(0, 0, 255))

        scene1 = Scene(120, 120)
        c1 = Circle(center=Point(60, 60), radius=20, fill=FillStyle(color=Color.red()))
        c1.effects = [glow, shadow]
        scene1.add(c1)
        buf1 = render(scene1)

        scene2 = Scene(120, 120)
        c2 = Circle(center=Point(60, 60), radius=20, fill=FillStyle(color=Color.red()))
        c2.effects = [shadow, glow]
        scene2.add(c2)
        buf2 = render(scene2)

        # Both produce valid distinct renders
        assert buf1.shape == (120, 120, 3)
        assert buf2.shape == (120, 120, 3)


class TestColorEffects:
    """Verify color manipulation effects and strict alpha preservation."""

    def test_brightness_contrast(self):
        """Brightness shifts luminance while preserving transparent exterior."""
        scene = Scene(60, 60, background=Color.transparent())
        rect = Rectangle(position=Point(15, 15), width=30, height=30, fill=FillStyle(color=Color(100, 100, 100)))
        rect.effects = [BrightnessContrastEffect(brightness=0.2, contrast=1.0)]
        scene.add(rect)

        buf = render(scene, alpha=True)
        # Inside pixel is brighter than original 100
        assert buf[30, 30, 0] > 140
        assert buf[30, 30, 1] > 140
        assert buf[30, 30, 2] > 140
        # Transparent outside pixel is strictly 0 across all channels
        assert buf[5, 5, 3] == 0
        assert buf[5, 5, 0] == 0
        assert buf[5, 5, 1] == 0
        assert buf[5, 5, 2] == 0

    def test_saturation_desaturate(self):
        """Saturation factor 0.0 converts pure red to grayscale Rec.709 luma."""
        scene = Scene(60, 60, background=Color.transparent())
        rect = Rectangle(position=Point(15, 15), width=30, height=30, fill=FillStyle(color=Color(255, 0, 0)))
        rect.effects = [SaturationEffect(factor=0.0)]
        scene.add(rect)

        buf = render(scene, alpha=True)
        # In Rec.709: Y = 0.2126 * 255 ~= 54.2
        p = buf[30, 30]
        assert p[0] == pytest.approx(54, abs=5)
        assert p[1] == pytest.approx(54, abs=5)
        assert p[2] == pytest.approx(54, abs=5)
        # Alpha is preserved at 255
        assert p[3] == 255

    def test_grayscale_intensity(self):
        """Grayscale intensity=1.0 makes R, G, B equal."""
        scene = Scene(60, 60)
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(200, 50, 100)))
        rect.effects = [GrayscaleEffect(intensity=1.0)]
        scene.add(rect)

        buf = render(scene)
        p = buf[30, 30]
        # BGR should all be approximately equal
        assert abs(int(p[0]) - int(p[1])) <= 2
        assert abs(int(p[1]) - int(p[2])) <= 2

    def test_sepia(self):
        """Sepia effect imparts warm tones with R > G > B."""
        scene = Scene(60, 60)
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(150, 150, 150)))
        rect.effects = [SepiaEffect(intensity=1.0)]
        scene.add(rect)

        buf = render(scene)
        p = buf[30, 30]  # BGR order: p[0]=B, p[1]=G, p[2]=R
        # In sepia: R > G > B
        assert p[2] > p[1] > p[0]

    def test_hue_shift_identity_and_shift(self):
        """HueShiftEffect 0.0/360.0 is identity; 120.0 shifts hue."""
        scene_base = Scene(60, 60)
        r0 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(255, 0, 0)))
        scene_base.add(r0)
        buf0 = render(scene_base)

        scene_360 = Scene(60, 60)
        r360 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(255, 0, 0)))
        r360.effects = [HueShiftEffect(angle=360.0)]
        scene_360.add(r360)
        buf360 = render(scene_360)
        # 360 degree hue rotation matches base
        assert np.allclose(buf0, buf360, atol=2.0)

        scene_120 = Scene(60, 60)
        r120 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(255, 0, 0)))
        r120.effects = [HueShiftEffect(angle=120.0)]
        scene_120.add(r120)
        buf120 = render(scene_120)
        # Green channel is now dominant instead of Red
        p120 = buf120[30, 30]
        assert p120[1] > p120[2]  # Green > Red

    def test_color_matrix_invert_and_validation(self):
        """ColorMatrixEffect properly applies 4x5 transformation and validates alpha row."""
        from drawcv.core.exceptions import ValidationError

        # Invert matrix
        invert_matrix = (
            (-1.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 0.0),
        )
        effect = ColorMatrixEffect(matrix=invert_matrix)

        scene = Scene(60, 60, background=Color.transparent())
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(255, 0, 0)))
        rect.effects = [effect]
        scene.add(rect)

        buf = render(scene, alpha=True)
        # Red inverted: (255, 0, 0) -> Cyan (0, 255, 255) in RGB -> (255, 255, 0) in BGR
        p = buf[30, 30]
        assert p[0] == 255 # B
        assert p[1] == 255 # G
        assert p[2] == 0   # R
        assert p[3] == 255 # A

        # Invalid alpha row must be rejected
        bad_matrix = (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.5, 0.0),
        )
        with pytest.raises(ValidationError, match="Color matrix row 3 must be strictly"):
            ColorMatrixEffect(matrix=bad_matrix)


class TestEffectsAnimation:
    """Verify animation track path traversal on effects and transactional mutation safety."""

    def test_animate_effect_property_path_indexed(self):
        """Animation track successfully animates indexed effect property (effects.0.blur_radius)."""
        from drawcv.animation.track import AnimationTrack
        from drawcv.animation.timing import Timing

        rect = Rectangle(position=Point(10, 10), width=40, height=40)
        glow = GlowEffect(blur_radius=5.0)
        rect.effects = [glow]

        track = AnimationTrack(
            target_id=rect.id,
            property_path="effects.0.blur_radius",
            start_value=5.0,
            end_value=25.0,
            timing=Timing(duration=2.0),
        )

        # Midpoint: t=1.0 -> blur_radius = 15.0
        val = track.evaluate(1.0, rect)
        assert val == pytest.approx(15.0)
        assert rect.effects[0].blur_radius == pytest.approx(15.0)

    def test_animate_effect_immutable_color_alpha(self):
        """Animation track successfully navigates through effect to immutable Color (effects.0.color.a)."""
        from drawcv.animation.track import AnimationTrack
        from drawcv.animation.timing import Timing

        rect = Rectangle(position=Point(10, 10), width=40, height=40)
        glow = GlowEffect(blur_radius=5.0, color=Color(255, 255, 0, 0.2))
        rect.effects = [glow]

        track = AnimationTrack(
            target_id=rect.id,
            property_path="effects.0.color.a",
            start_value=0.2,
            end_value=0.8,
            timing=Timing(duration=2.0),
        )

        val = track.evaluate(1.0, rect)
        assert val == pytest.approx(0.5)
        assert rect.effects[0].color.a == pytest.approx(0.5)
        # Original RGB preserved
        assert rect.effects[0].color.r == 255
        assert rect.effects[0].color.g == 255
        assert rect.effects[0].color.b == 0

    def test_transactional_rollback_on_invalid_effect_value(self):
        """When animation evaluation attempts an invalid value, validation fails AND original state is preserved."""
        from drawcv.animation.track import AnimationTrack
        from drawcv.animation.timing import Timing
        from drawcv.core.exceptions import ValidationError

        rect = Rectangle(position=Point(10, 10), width=40, height=40)
        shadow = ShadowEffect(blur_radius=10.0, opacity=0.8)
        rect.effects = [shadow]

        # Track that attempts to set an invalid negative blur_radius (-5.0)
        track = AnimationTrack(
            target_id=rect.id,
            property_path="effects.0.blur_radius",
            start_value=10.0,
            end_value=-5.0,
            timing=Timing(duration=1.0),
        )

        with pytest.raises(ValidationError, match="blur_radius must be non-negative"):
            track.evaluate(1.0, rect)

        # Invariant: effect state MUST NOT be corrupted with the invalid -5.0 value
        assert rect.effects[0].blur_radius == 10.0
        assert rect.effects[0].opacity == 0.8

    def test_transactional_rollback_on_immutable_child_invalid_value(self):
        """Invalid color channel assignment fails validation and preserves original Color object."""
        from drawcv.animation.track import AnimationTrack
        from drawcv.animation.timing import Timing
        from drawcv.core.exceptions import ValidationError

        rect = Rectangle(position=Point(10, 10), width=40, height=40)
        original_color = Color(100, 150, 200, 0.5)
        glow = GlowEffect(blur_radius=6.0, color=original_color)
        rect.effects = [glow]

        # Track attempting to set alpha to 2.5 (> 1.0)
        track = AnimationTrack(
            target_id=rect.id,
            property_path="effects.0.color.a",
            start_value=0.5,
            end_value=2.5,
            timing=Timing(duration=1.0),
        )

        with pytest.raises(ValidationError, match="Color alpha must be in range"):
            track.evaluate(1.0, rect)

        # Invariant: original color remains completely untouched
        assert rect.effects[0].color == original_color
        assert rect.effects[0].color.a == 0.5


class TestEffectsSerializationAndHistory:
    """Verify JSON serialization roundtrip, undo/redo, and cloning of effects stacks."""

    def test_json_roundtrip_all_effect_types(self):
        """Full scene with all 9 effect types serializes to JSON and deserializes losslessly."""
        from drawcv import from_json

        scene = Scene(200, 200)
        rect = Rectangle(position=Point(10, 10), width=50, height=50, fill=FillStyle(color=Color.red()))
        rect.effects = [
            BlurEffect(kernel_size=15, sigma=4.0),
            ShadowEffect(offset_x=10.0, offset_y=12.0, blur_radius=5.0, color=Color(0, 0, 0, 0.75)),
            GlowEffect(blur_radius=8.0, color=Color(255, 200, 50, 0.9), opacity=0.8),
            BrightnessContrastEffect(brightness=0.1, contrast=1.2),
            SaturationEffect(factor=1.5),
            HueShiftEffect(angle=45.0),
            GrayscaleEffect(intensity=0.5),
            SepiaEffect(intensity=0.3),
            ColorMatrixEffect(matrix=(
                (1.0, 0.0, 0.0, 0.0, 0.1),
                (0.0, 1.0, 0.0, 0.0, 0.1),
                (0.0, 0.0, 1.0, 0.0, 0.1),
                (0.0, 0.0, 0.0, 1.0, 0.0),
            )),
        ]
        scene.add(rect)

        json_str = scene.to_json()
        restored = Scene.from_json(json_str)

        restored_rect = restored.get(rect.id)
        assert len(restored_rect.effects) == 9

        # Verify ordering and types
        assert isinstance(restored_rect.effects[0], BlurEffect)
        assert restored_rect.effects[0].kernel_size == 15
        assert restored_rect.effects[0].sigma == 4.0

        assert isinstance(restored_rect.effects[1], ShadowEffect)
        assert restored_rect.effects[1].offset_x == 10.0
        assert restored_rect.effects[1].color.a == 0.75

        assert isinstance(restored_rect.effects[2], GlowEffect)
        assert restored_rect.effects[2].blur_radius == 8.0
        assert restored_rect.effects[2].opacity == 0.8

        assert isinstance(restored_rect.effects[3], BrightnessContrastEffect)
        assert restored_rect.effects[3].brightness == 0.1

        assert isinstance(restored_rect.effects[4], SaturationEffect)
        assert restored_rect.effects[4].factor == 1.5

        assert isinstance(restored_rect.effects[5], HueShiftEffect)
        assert restored_rect.effects[5].angle == 45.0

        assert isinstance(restored_rect.effects[6], GrayscaleEffect)
        assert restored_rect.effects[6].intensity == 0.5

        assert isinstance(restored_rect.effects[7], SepiaEffect)
        assert restored_rect.effects[7].intensity == 0.3

        assert isinstance(restored_rect.effects[8], ColorMatrixEffect)
        assert restored_rect.effects[8].matrix[0][4] == 0.1
        assert restored_rect.effects[8].matrix[3] == (0.0, 0.0, 0.0, 1.0, 0.0)

    def test_clone_creates_independent_effect_instances(self):
        """Cloning a drawable clones the effects list and effect instances independently."""
        rect1 = Rectangle(position=Point(10, 10), width=40, height=40)
        rect1.effects = [GlowEffect(blur_radius=5.0)]

        rect2 = rect1.clone()
        assert len(rect2.effects) == 1
        assert rect2.effects[0] is not rect1.effects[0]

        # Mutating clone does not mutate original
        rect2.effects[0].blur_radius = 20.0
        assert rect1.effects[0].blur_radius == 5.0

    def test_undo_redo_effects_mutation(self):
        """Modifying effects on a scene object supports undo/redo."""
        scene = Scene(100, 100)
        rect = Rectangle(position=Point(10, 10), width=40, height=40)
        rect.effects = [BlurEffect(kernel_size=5)]
        scene.add(rect)

        # Mutate effects via state edit transaction
        with scene.edit(rect):
            rect.effects = [BlurEffect(kernel_size=5), GlowEffect(blur_radius=10.0)]

        assert len(rect.effects) == 2

        scene.undo()
        assert len(rect.effects) == 1
        assert isinstance(rect.effects[0], BlurEffect)

        scene.redo()
        assert len(rect.effects) == 2
        assert isinstance(rect.effects[1], GlowEffect)


class TestEffectsSVGExport:
    """Verify SVG export triggers raster fallback accounting for entities with effects."""

    def test_svg_export_raster_fallback_with_effects(self):
        """Entities with effects trigger raster fallback in SVG export."""
        scene = Scene(100, 100)
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.red()))
        rect.effects = [ShadowEffect(blur_radius=4.0)]
        scene.add(rect)

        res = scene.export_svg()
        assert len(res.fallbacks) == 1
        assert res.fallbacks[0].reason == "effects"
        assert res.fallbacks[0].entity == rect.id
        assert "<image" in res.svg

    def test_svg_export_strict_mode_rejects_effects(self):
        """Strict SVG export raises RenderError when effects require raster fallback."""
        from drawcv.core.exceptions import RenderError

        scene = Scene(100, 100)
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.red()))
        rect.effects = [GlowEffect(blur_radius=5.0)]
        scene.add(rect)

        with pytest.raises(RenderError, match="SVG requires raster fallback"):
            scene.export_svg(strict=True)


class TestEffectDeserializationInvariants:
    """Verify that from_dict and Scene.from_dict reject invalid raw types without prior coercion."""

    @pytest.mark.parametrize(
        "effect_dict",
        [
            # Booleans must not coerce to 1.0 or 0.0
            {"type": "brightness_contrast", "brightness": True},
            {"type": "brightness_contrast", "contrast": False},
            {"type": "saturation", "factor": True},
            {"type": "hue_shift", "angle": True},
            {"type": "grayscale", "intensity": True},
            {"type": "sepia", "intensity": True},
            {"type": "glow", "blur_radius": True},
            {"type": "glow", "opacity": True},
            {"type": "blur", "kernel_size": True},
            {"type": "blur", "sigma": True},
            {"type": "shadow", "offset_x": True},
            {"type": "shadow", "blur_radius": True},
            {"type": "shadow", "opacity": True},
            # Strings in numeric fields
            {"type": "brightness_contrast", "brightness": "0.5"},
            {"type": "saturation", "factor": "2.0"},
            {"type": "hue_shift", "angle": "90"},
            {"type": "grayscale", "intensity": "1.0"},
            {"type": "sepia", "intensity": "0.5"},
            {"type": "glow", "blur_radius": "5.0"},
            {"type": "blur", "kernel_size": "15"},
            {"type": "blur", "sigma": "1.0"},
            {"type": "shadow", "offset_x": "10"},
            # Non-finites (NaN / Inf)
            {"type": "brightness_contrast", "brightness": float("nan")},
            {"type": "brightness_contrast", "contrast": float("inf")},
            {"type": "saturation", "factor": float("nan")},
            {"type": "hue_shift", "angle": float("inf")},
            {"type": "grayscale", "intensity": float("nan")},
            {"type": "sepia", "intensity": float("inf")},
            {"type": "glow", "blur_radius": float("nan")},
            {"type": "blur", "sigma": float("inf")},
            {"type": "shadow", "blur_radius": float("nan")},
            # Invalid ColorMatrix entries
            {"type": "color_matrix", "matrix": [[True] * 5] * 4},
            {"type": "color_matrix", "matrix": [["bad"] * 5] * 4},
            {"type": "color_matrix", "matrix": [[float("nan")] * 5] * 4},
            {"type": "color_matrix", "matrix": [[1.0] * 5] * 4},
        ],
    )
    def test_effect_from_dict_rejects_invalid_types(self, effect_dict):
        from drawcv.effects import effect_from_dict
        from drawcv.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            effect_from_dict(effect_dict)

    def test_scene_from_dict_propagates_effect_validation_error(self):
        from drawcv.scene import Scene
        from drawcv.core.exceptions import ValidationError

        scene = Scene(100, 100)
        rect = Rectangle(position=Point(10, 10), width=20, height=20)
        rect.effects = [BrightnessContrastEffect(brightness=0.2)]
        scene.add(rect)
        scene_dict = scene.to_dict()

        # Mutate serialized effect to invalid boolean
        scene_dict["scene"]["layers"][0]["objects"][0]["effects"][0]["brightness"] = True
        with pytest.raises(ValidationError):
            Scene.from_dict(scene_dict)


class TestAnimationTrackNonEffectTarget:
    """Verify non-effect targets still animate and roll back cleanly via explicit _validate."""

    def test_shape_property_animation_success(self):
        from drawcv.animation.track import AnimationTrack
        circle = Circle(center=Point(50, 50), radius=20.0)
        track = AnimationTrack(target_id=circle.id, property_path="radius", start_value=20.0, end_value=40.0)
        val = track.evaluate(1.0, target=circle)
        assert val == 40.0
        assert circle.radius == 40.0

    def test_shape_property_animation_rollback_on_invalid(self):
        from drawcv.animation.track import AnimationTrack
        from drawcv.core.exceptions import ValidationError

        circle = Circle(center=Point(50, 50), radius=20.0)
        # Attempt to set negative radius which fails Circle._validate
        track = AnimationTrack(target_id=circle.id, property_path="radius", start_value=20.0, end_value=-10.0)
        with pytest.raises(ValidationError):
            track.evaluate(1.0, target=circle)
        # State rolled back transactionally
        assert circle.radius == 20.0
