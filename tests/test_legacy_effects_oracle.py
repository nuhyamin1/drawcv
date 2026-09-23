"""Authoritative legacy regression tests and canvas-boundary policy tests.

This test module verifies:
1. Byte-exact regression locks for representative legacy effect scenes positioned
   away from canvas boundaries (see TestLegacyRegressionReferences for their history).
2. The intentional Milestone 6 border compatibility policy: spatial effects now consistently
   use cv2.BORDER_CONSTANT (transparent zero) in both BGR and BGRA output, rather than
   the legacy BGR-specific cv2.BORDER_DEFAULT (reflected border).
"""

import hashlib
import numpy as np
import pytest

from drawcv import (
    BlurEffect,
    Circle,
    Color,
    FillStyle,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    ShadowEffect,
    Transform,
)


def render_scene(scene: Scene, alpha: bool = False) -> np.ndarray:
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=alpha)
    return canvas.buffer.copy()


def buffer_hash(buf: np.ndarray) -> str:
    return hashlib.sha256(buf.tobytes()).hexdigest()


class TestLegacyRegressionReferences:
    """Strict byte-identical regression checks for representative effect scenes.

    The hashes were first established against the pre-refactor renderer, proving
    the generalized effects executor byte-identical to historical rendering away
    from canvas boundaries. They were re-established once when shape coverage
    became area-exact (pixel k spans [k, k + 1]); the effect processors were
    unchanged by that switch, only the rasterized shapes they receive.
    """

    def test_single_blur_away_from_canvas_boundary(self):
        """Single Gaussian blur on an opaque rectangle away from canvas edges."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(25, 25), width=50, height=50, fill=FillStyle(color=Color.red()))
        rect.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "5d1d839e728347e8ebb90d42ac68e9ab07f7308bf539acc29a9497a71a4441c7"
        assert buffer_hash(buf) == expected_sha256

    def test_single_shadow_positive_offset(self):
        """Single drop shadow with positive (+8, +6) offset."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [ShadowEffect(offset_x=8, offset_y=6, blur_radius=5, color=Color(0, 0, 0, 0.5))]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "38de8e9712de9711ea6c332fae0ffccaac8c0b409fc3df0368e20c2ca69174a5"
        assert buffer_hash(buf) == expected_sha256

    def test_single_shadow_negative_offset(self):
        """Single drop shadow with negative (-6, -8) offset."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.green()))
        rect.effects = [ShadowEffect(offset_x=-6, offset_y=-8, blur_radius=5, color=Color(0, 0, 0, 0.6))]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "ffc45640c5921fb32c62fd60cc415a70a916b8a6cbdbb712586f4eea969d4100"
        assert buffer_hash(buf) == expected_sha256

    def test_translucent_blur(self):
        """Translucent rectangle (opacity=0.6) with Gaussian blur."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=Color.red()), opacity=0.6)
        rect.effects = [BlurEffect(kernel_size=11, sigma=2.5)]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "9a13b7d8f3dd4cd4aed9fba0c70113bd981b97be685205a91c759234e926fe07"
        assert buffer_hash(buf) == expected_sha256

    def test_transparent_bgra_blur(self):
        """Transparent BGRA canvas with blurred Circle."""
        scene = Scene(80, 80, background=Color.transparent())
        c = Circle(center=Point(40, 40), radius=20, fill=FillStyle(color=Color.blue()))
        c.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene.add(c)

        buf = render_scene(scene, alpha=True)
        expected_sha256 = "5a808b3cfbb059d9d9d062a9298935047b879f9ddbd43c89414111dcf6d55ae8"
        assert buffer_hash(buf) == expected_sha256

    def test_transformed_object_shadow(self):
        """Transformed (rotated 45 deg + scaled) rectangle with drop shadow."""
        scene = Scene(120, 120, background=Color.white())
        rect = Rectangle(
            position=Point(30, 30),
            width=40,
            height=40,
            fill=FillStyle(color=Color.red()),
            transform=Transform(rotation=45.0, scale_x=1.2, scale_y=0.8),
        )
        rect.effects = [ShadowEffect(offset_x=5, offset_y=5, blur_radius=7, color=Color(0, 0, 0, 0.5))]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "efcde7324a45965391e50ade52676e78645b4197b13c15a0a36f2e95bf140dce"
        assert buffer_hash(buf) == expected_sha256


class TestCanvasBoundaryPolicy:
    """Tests for the intentional Milestone 6 transparent-zero border policy.

    In Milestone 6, spatial effects consistently treat pixels beyond the isolated
    surface or canvas boundary as transparent zero (cv2.BORDER_CONSTANT) in both BGR
    and BGRA modes.

    Compatibility note:
    In the legacy pre-milestone renderer, BGR rendering used cv2.BORDER_DEFAULT
    (reflected border), while BGRA rendering used cv2.BORDER_CONSTANT. Under the new
    standardized semantics, objects touching the canvas edge fade cleanly into the canvas
    background rather than reflecting interior pixels at the edge.
    """

    def test_canvas_boundary_transparent_zero_semantic(self):
        """Verify that an object touching the canvas boundary fades into canvas with transparent-zero border."""
        scene = Scene(80, 80, background=Color.white())
        rect = Rectangle(position=Point(0, 0), width=30, height=30, fill=FillStyle(color=Color.black()))
        rect.effects = [BlurEffect(kernel_size=9, sigma=1.5)]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        assert buf.shape == (80, 80, 3)
        # Deep interior of the rectangle remains solid black
        assert buf[15, 15, 0] == 0
        # Canvas corner (0, 0) with transparent BORDER_CONSTANT fades toward the white canvas
        assert 0 < buf[0, 0, 0] < 255
