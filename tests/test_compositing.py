"""Unit tests for Phase 6: Alpha compositing, isolated buffers, effects, clipping, masks, ImageObject, and Text."""

import math
import cv2
import numpy as np
import pytest

from drawcv import (
    BlurEffect,
    BlurType,
    BoundingBox,
    Canvas,
    ClipPath,
    ClipRect,
    Color,
    FillStyle,
    FontFamily,
    Group,
    ImageInterpolation,
    ImageObject,
    Layer,
    Mask,
    MaskMapping,
    OpenCVRenderer,
    Point,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShadowEffect,
    StrokeStyle,
    Text,
    TextAlignment,
    ValidationError,
)


class TestIsolatedOpacityCompositing:
    """Verify single-application opacity boundary and elimination of intersection darkening."""

    def test_nested_isolated_opacity_exact_multiplication(self):
        """Layer(0.8) -> Group(0.5) -> Child(0.5) must yield exactly 0.20 net opacity on white."""
        scene = Scene(width=100, height=100, background=Color.white())
        layer = scene.create_layer("test_layer", opacity=0.8)

        # Isolated group
        group = Group(opacity=0.5)
        rect = Rectangle(
            position=Point(10, 10),
            width=80,
            height=80,
            fill=FillStyle(color=Color.black()),
            stroke=None,
            opacity=0.5,
        )
        group.add(rect)
        layer.add(group)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Expected: net alpha = 0.8 * 0.5 * 0.5 = 0.20
        # Background is white (255, 255, 255), rect is black (0, 0, 0)
        # Result = 255 * (1 - 0.20) + 0 * 0.20 = 204.0
        center_pixel = canvas.buffer[50, 50]
        for c in range(3):
            assert abs(int(center_pixel[c]) - 204) <= 2, f"Channel {c} was {center_pixel[c]}, expected ~204"

    def test_elimination_of_overlapping_geometry_darkening(self):
        """Two overlapping opaque shapes in a Group(opacity=0.5) must NOT multiply opacities in intersection."""
        scene = Scene(width=120, height=120, background=Color.white())

        group = Group(opacity=0.5)
        # Shape A
        r1 = Rectangle(position=Point(10, 10), width=60, height=60, fill=FillStyle(color=Color.black()), stroke=None)
        # Shape B overlapping A
        r2 = Rectangle(position=Point(30, 30), width=60, height=60, fill=FillStyle(color=Color.black()), stroke=None)
        group.add(r1)
        group.add(r2)
        scene.add(group)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Non-overlapping part of r1 (e.g. at 20, 20)
        p_single = canvas.buffer[20, 20]
        # Overlapping part of r1 and r2 (at 40, 40)
        p_overlap = canvas.buffer[40, 40]

        # In an isolated group, the union of r1 and r2 is solid black inside the group buffer.
        # When composited at group opacity 0.5 over white, BOTH single and overlap regions
        # must have identical pixel intensity ~ 128 (255 * 0.5).
        # Without isolated compositing, overlap would be 255 * (1 - 0.5) * (1 - 0.5) = 64 (darker).
        assert abs(int(p_single[0]) - 128) <= 2
        assert abs(int(p_overlap[0]) - 128) <= 2
        assert abs(int(p_single[0]) - int(p_overlap[0])) <= 1, (
            f"Overlap darkened: single={p_single[0]}, overlap={p_overlap[0]}"
        )


class TestEffectsPipeline:
    """Verify Drop Shadows, Blur, and Asymmetric Padding."""

    def test_shadow_asymmetric_bounds_expansion(self):
        """Negative and signed shadow offsets expand bounds correctly without clipping."""
        rect = Rectangle(position=Point(100, 100), width=50, height=50)
        shadow = ShadowEffect(offset_x=-30.0, offset_y=40.0, blur_radius=10.0)
        rect.effects.append(shadow)

        eb = rect.get_effect_bounds()
        assert eb.left <= 40.0
        assert eb.right >= 180.0
        assert eb.top <= 70.0
        assert eb.bottom >= 220.0

    def test_shadow_rendered_behind_base(self):
        """Shadow renders behind base object without obscuring the base colors."""
        scene = Scene(width=200, height=200, background=Color.white())
        rect = Rectangle(
            position=Point(50, 50),
            width=50,
            height=50,
            fill=FillStyle(color=Color(255, 0, 0)), # Red
            stroke=None,
        )
        rect.effects.append(ShadowEffect(offset_x=30, offset_y=30, blur_radius=5, color=Color(0, 0, 0, 0.5)))
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Center of red square (75, 75) should be solid red (BGR: 0, 0, 255)
        center_px = canvas.buffer[75, 75]
        assert center_px[2] > 200 # Red channel high
        assert center_px[0] < 50  # Blue channel low

        # Shadow area at (110, 110) should have darkened pixels (not pure white)
        shadow_px = canvas.buffer[110, 110]
        assert shadow_px[0] < 240 and shadow_px[1] < 240 and shadow_px[2] < 240

    def test_blur_effect_gaussian_and_box(self):
        """Content blur blurs the entity in premultiplied space."""
        scene = Scene(width=100, height=100, background=Color.white())
        rect = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.black()), stroke=None)
        rect.effects.append(BlurEffect(kernel_size=15, sigma=5.0, blur_type=BlurType.GAUSSIAN))
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Edges should be smoothly blurred; outside original boundary (e.g. at 25, 25) should not be pure white
        blur_edge = canvas.buffer[25, 25]
        assert blur_edge[0] < 255, "Blur effect did not expand beyond geometry boundary"


class TestClipping:
    """Verify ClipRect and ClipPath stenciling."""

    def test_clip_rect_hard_boundary(self):
        """ClipRect strictly discards all pixels outside the rectangular region."""
        scene = Scene(width=200, height=200, background=Color.white())
        rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color.black()), stroke=None)
        rect.clip = ClipRect(x=50, y=50, width=50, height=50)
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Inside clip: black
        assert canvas.buffer[75, 75][0] == 0
        # Outside clip: white
        assert canvas.buffer[25, 25][0] == 255
        assert canvas.buffer[150, 150][0] == 255

    def test_clip_path_polygon_boundary(self):
        """ClipPath clips geometry to an arbitrary polygon contour."""
        scene = Scene(width=200, height=200, background=Color.white())
        rect = Rectangle(position=Point(0, 0), width=200, height=200, fill=FillStyle(color=Color.black()), stroke=None)
        rect.clip = ClipPath(points=[Point(100, 20), Point(180, 180), Point(20, 180)])
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Centroid of triangle (100, 126) should be black
        assert canvas.buffer[126, 100][0] == 0
        # Top-left corner (10, 10) should remain white
        assert canvas.buffer[10, 10][0] == 255

    def test_clip_applies_after_blur(self):
        """Clip stencils the output so blurred pixels do not spill outside the clip."""
        scene = Scene(width=200, height=200, background=Color.white())
        rect = Rectangle(position=Point(50, 50), width=100, height=100, fill=FillStyle(color=Color.black()), stroke=None)
        rect.effects.append(BlurEffect(kernel_size=25, sigma=8.0))
        # Clip strictly to [50, 50, 100, 100]
        rect.clip = ClipRect(x=50, y=50, width=100, height=100)
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Pixel just outside the clip boundary (e.g. 45, 100) must be pure white (255)
        outside_pixel = canvas.buffer[100, 45]
        assert outside_pixel[0] == 255 and outside_pixel[1] == 255 and outside_pixel[2] == 255


class TestMasking:
    """Verify Grayscale Masks and 4-channel premultiplied scaling."""

    def test_mask_premultiplied_scaling_no_bright_fringing(self):
        """Mask modulates premultiplied RGB and Alpha together, preserving C <= a."""
        # 100x100 half-transparent mask (value 128)
        mask_buf = np.full((100, 100), 128, dtype=np.uint8)
        mask = Mask(buffer=mask_buf, mapping=MaskMapping.FIT_BOUNDS)

        scene = Scene(width=200, height=200, background=Color.white())
        rect = Rectangle(
            position=Point(50, 50),
            width=100,
            height=100,
            fill=FillStyle(color=Color(255, 0, 0)), # Red (r=255, g=0, b=0)
            stroke=None,
        )
        rect.mask = mask
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Coverage factor is 128/255 ~= 0.502
        # White background (255, 255, 255), red fill (0, 0, 255)
        # Blended Blue/Green: 255 * (1 - 0.502) = ~127
        # Blended Red: 255 * (1 - 0.502) + 255 * 0.502 = 255
        pixel = canvas.buffer[100, 100]
        assert abs(int(pixel[0]) - 127) <= 2 # Blue
        assert abs(int(pixel[1]) - 127) <= 2 # Green
        assert pixel[2] == 255              # Red

    def test_mask_fit_bounds_uses_pre_effect_bounds(self):
        """Mask FIT_BOUNDS aligns with pre-effect bounds even when a shadow is attached."""
        # Half white (255), half black (0) vertical mask
        mask_buf = np.zeros((100, 100), dtype=np.uint8)
        mask_buf[:, :50] = 255 # Left half visible, right half hidden
        mask = Mask(buffer=mask_buf, mapping=MaskMapping.FIT_BOUNDS)

        scene = Scene(width=300, height=300, background=Color.white())
        rect = Rectangle(
            position=Point(100, 100),
            width=100,
            height=100,
            fill=FillStyle(color=Color.black()),
            stroke=None,
        )
        # Add shadow that expands bounds to the right
        rect.effects.append(ShadowEffect(offset_x=40, offset_y=0, blur_radius=5))
        rect.mask = mask
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Pre-effect bounds are [100, 100, 100, 100].
        # Left half (x in [100, 150]) must be visible (black)
        assert canvas.buffer[150, 120][0] < 50
        # Right half of shape (x in [150, 200]) must be masked out (white)
        assert canvas.buffer[150, 180][0] > 200


class TestImageObject:
    """Verify ImageObject scaling, cropping, rotation, and alpha channels."""

    def test_image_bgr_and_scaling(self):
        """Test BGR image scaling to custom display dimensions."""
        img_mat = np.zeros((50, 50, 3), dtype=np.uint8)
        img_mat[:, :] = [255, 0, 0] # Solid Blue

        img_obj = ImageObject(image=img_mat, position=Point(20, 20), width=100, height=80)
        assert img_obj.display_width == 100.0
        assert img_obj.display_height == 80.0
        assert img_obj.source_width == 50
        assert img_obj.source_height == 50

        scene = Scene(width=200, height=200, background=Color.white())
        scene.add(img_obj)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Scaled area [20..120, 20..100] should be solid blue
        assert np.array_equal(canvas.buffer[50, 50], [255, 0, 0])
        # Outside image should remain white
        assert np.array_equal(canvas.buffer[150, 150], [255, 255, 255])

    def test_image_cropping(self):
        """Test sub-rectangle cropping."""
        img_mat = np.zeros((100, 100, 3), dtype=np.uint8)
        img_mat[:50, :] = [0, 255, 0]   # Top half Green
        img_mat[50:, :] = [0, 0, 255]   # Bottom half Red

        # Crop bottom half only
        crop_box = BoundingBox(0, 50, 100, 50)
        img_obj = ImageObject(image=img_mat, position=Point(0, 0), crop=crop_box)
        assert img_obj.display_width == 100.0
        assert img_obj.display_height == 50.0

        scene = Scene(width=100, height=100, background=Color.white())
        scene.add(img_obj)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # (25, 25) should be Red, not Green
        assert np.array_equal(canvas.buffer[25, 25], [0, 0, 255])

    def test_image_bgra_source_transparency(self):
        """4-channel BGRA image alpha channel is respected during rendering."""
        img_mat = np.zeros((50, 50, 4), dtype=np.uint8)
        img_mat[:, :] = [0, 0, 0, 128] # Semi-transparent black

        img_obj = ImageObject(image=img_mat, position=Point(0, 0))
        scene = Scene(width=50, height=50, background=Color.white())
        scene.add(img_obj)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Alpha 128 over white (255) yields ~ 128
        assert abs(int(canvas.buffer[25, 25][0]) - 128) <= 2


class TestTextDrawable:
    """Verify Text metrics, alignment, background plate, and hit-testing."""

    def test_text_measuring_and_alignments(self):
        """Text measures dimensions and positions bounding box based on alignment."""
        t_left = Text("Hello", position=Point(100, 100), alignment=TextAlignment.LEFT)
        t_center = Text("Hello", position=Point(100, 100), alignment=TextAlignment.CENTER)
        t_right = Text("Hello", position=Point(100, 100), alignment=TextAlignment.RIGHT)

        b_left = t_left.get_geometry_bounds()
        b_center = t_center.get_geometry_bounds()
        b_right = t_right.get_geometry_bounds()

        assert b_left.left == 100.0
        assert abs((b_center.left + b_center.width / 2.0) - 100.0) < 1e-4
        assert abs(b_right.right - 100.0) < 1e-4

    def test_text_background_plate_rendering(self):
        """Background plate renders with fill and padding."""
        text = Text(
            "DrawCV",
            position=Point(50, 50),
            color=Color.white(),
            background_fill=Color(0, 128, 255),
            background_radius=8.0,
            padding=10.0,
        )
        scene = Scene(width=200, height=200, background=Color.white())
        scene.add(text)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Inside padding area (e.g. top-left of plate at (55, 55)) should be plate color (BGR: 255, 128, 0)
        plate_px = canvas.buffer[55, 55]
        assert np.array_equal(plate_px, [255, 128, 0])

    def test_text_anchors_and_hit_testing(self):
        """Text anchors and hit testing function correctly."""
        text = Text("Sample", position=Point(40, 40), padding=5.0)
        b = text.get_bounds()

        assert text.contains_point(b.center)
        assert not text.contains_point(Point(b.left - 10, b.top - 10))
        assert text.anchor("top_left") == b.top_left
        assert text.anchor("center") == b.center


class TestLayerCompositing:
    """Verify Layer-level opacity, clipping, and effects."""

    def test_layer_level_effects_and_opacity(self):
        """Layer with opacity and shadow applies correctly to all contained objects."""
        scene = Scene(width=200, height=200, background=Color.white())
        layer = scene.create_layer("fx_layer", opacity=0.8)
        rect = Rectangle(position=Point(40, 40), width=60, height=60, fill=FillStyle(color=Color.black()), stroke=None)
        layer.add(rect)

        layer.effects.append(ShadowEffect(offset_x=20, offset_y=20, blur_radius=5))

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)

        # Object should have 0.8 opacity on white -> ~ 51
        obj_px = canvas.buffer[70, 70]
        assert abs(int(obj_px[0]) - 51) <= 2

        # Shadow area should exist
        sh_px = canvas.buffer[115, 115]
        assert sh_px[0] < 255


class TestPhase6ValidationsAndEdgeCases:
    """Verify input validation, cloning, rotated typography/images, and mask variations."""

    def test_validation_errors(self):
        """Invalid parameters raise ValidationError."""
        with pytest.raises(ValidationError):
            BlurEffect(kernel_size=4) # even integer
        with pytest.raises(ValidationError):
            BlurEffect(sigma=-1.0)
        with pytest.raises(ValidationError):
            ShadowEffect(opacity=1.5)
        with pytest.raises(ValidationError):
            ClipRect(0, 0, -10, 20)
        with pytest.raises(ValidationError):
            ClipPath([Point(0, 0), Point(10, 10)]) # fewer than 3 points
        with pytest.raises(ValidationError):
            Mask(np.zeros((10, 10, 3), dtype=np.uint8)) # not 2D
        with pytest.raises(ValidationError):
            ImageObject(image=np.zeros((10, 10), dtype=np.float32)) # not uint8
        with pytest.raises(ValidationError):
            Text(text="", font_scale=-0.5)

    def test_mask_inverted_and_absolute(self):
        """Inverted mask and ABSOLUTE mapping mode."""
        buf = np.zeros((50, 50), dtype=np.uint8)
        buf[:25, :] = 255
        mask = Mask(buffer=buf, mapping=MaskMapping.ABSOLUTE, inverted=True)

        scene = Scene(width=100, height=100, background=Color.white())
        rect = Rectangle(position=Point(0, 0), width=100, height=100, fill=FillStyle(color=Color.black()), stroke=None)
        rect.mask = mask
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)
        # Top 25 pixels were 255 in buffer, so inverted = 0 (hidden -> white)
        assert canvas.buffer[10, 10][0] == 255
        # Bottom area was 0 in buffer, so inverted = 1 (visible -> black)
        assert canvas.buffer[40, 40][0] == 0

    def test_box_blur(self):
        """Box blur renders correctly."""
        scene = Scene(width=100, height=100, background=Color.white())
        rect = Rectangle(position=Point(40, 40), width=20, height=20, fill=FillStyle(color=Color.black()), stroke=None)
        rect.effects.append(BlurEffect(kernel_size=9, blur_type=BlurType.BOX))
        scene.add(rect)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)
        assert canvas.buffer[38, 38][0] < 255 # Blurred edge

    def test_rotated_text_and_image(self):
        """Affine transformation on Text and ImageObject renders without errors."""
        scene = Scene(width=200, height=200, background=Color.white())

        # Rotated text
        t = Text("Rotated", position=Point(50, 50), color=Color.blue())
        t.transform.rotation = 45.0
        scene.add(t)

        # Rotated image
        mat = np.zeros((30, 30, 3), dtype=np.uint8)
        mat[:, :] = [0, 255, 0] # Green
        img = ImageObject(mat, position=Point(100, 100))
        img.transform.rotation = 30.0
        scene.add(img)

        renderer = OpenCVRenderer()
        canvas = renderer.render(scene)
        assert canvas.buffer is not None

    def test_cloning(self):
        """Clone creates independent instances with deep-copied effects and properties."""
        img_mat = np.zeros((20, 20, 3), dtype=np.uint8)
        img = ImageObject(img_mat, position=Point(10, 10), effects=[BlurEffect(kernel_size=5)])
        c_img = img.clone()
        assert c_img.id != img.id
        assert c_img.effects is not img.effects
        assert len(c_img.effects) == 1

        t = Text("Hello", position=Point(5, 5), background_fill=FillStyle(color=Color.red()))
        c_t = t.clone()
        assert c_t.id != t.id
        assert c_t.background_fill is not t.background_fill
        assert c_t.background_fill.color.r == 255
