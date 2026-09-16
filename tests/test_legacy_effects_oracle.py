"""Authoritative legacy regression tests and canvas-boundary policy tests.

This test module verifies:
1. Strict numerical equivalence (byte-exact identical rendered buffers) between the
   generalized effects architecture and the pre-milestone HEAD baseline across
   representative legacy scenes positioned away from canvas boundaries.
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
    """Strict byte-identical regression checks against pre-milestone HEAD output.

    Each reference hash was established by running the exact pre-refactor renderer
    (extracted from git HEAD:drawcv/renderer.py) on identical scene configurations.
    Away from canvas boundaries, the generalized effects executor produces 100%
    byte-identical pixel output (max_diff == 0) to historical rendering.
    """

    def test_single_blur_away_from_canvas_boundary(self):
        """Single Gaussian blur on an opaque rectangle away from canvas edges."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(25, 25), width=50, height=50, fill=FillStyle(color=Color.red()))
        rect.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "10f2bc1e06eb8b55d978cc56b09a5c708dff8c1b39c3334352a4cd727c062c51"
        assert buffer_hash(buf) == expected_sha256

    def test_single_shadow_positive_offset(self):
        """Single drop shadow with positive (+8, +6) offset."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [ShadowEffect(offset_x=8, offset_y=6, blur_radius=5, color=Color(0, 0, 0, 0.5))]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "9ecfac41b85443439747c21276bd92deed59218350b4e37713484496e5e09eef"
        assert buffer_hash(buf) == expected_sha256

    def test_single_shadow_negative_offset(self):
        """Single drop shadow with negative (-6, -8) offset."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(30, 30), width=40, height=40, fill=FillStyle(color=Color.green()))
        rect.effects = [ShadowEffect(offset_x=-6, offset_y=-8, blur_radius=5, color=Color(0, 0, 0, 0.6))]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "f51ee373c645906d438045dd83cd32ddedefdefffb61f4d06bba3379fa7929b2"
        assert buffer_hash(buf) == expected_sha256

    def test_translucent_blur(self):
        """Translucent rectangle (opacity=0.6) with Gaussian blur."""
        scene = Scene(100, 100, background=Color.white())
        rect = Rectangle(position=Point(20, 20), width=60, height=60, fill=FillStyle(color=Color.red()), opacity=0.6)
        rect.effects = [BlurEffect(kernel_size=11, sigma=2.5)]
        scene.add(rect)

        buf = render_scene(scene, alpha=False)
        expected_sha256 = "0ddb1b2b5d6a72f280ca8f85d058e3947874fc40449f060c2ba8835a878ea4fc"
        assert buffer_hash(buf) == expected_sha256

    def test_transparent_bgra_blur(self):
        """Transparent BGRA canvas with blurred Circle."""
        scene = Scene(80, 80, background=Color.transparent())
        c = Circle(center=Point(40, 40), radius=20, fill=FillStyle(color=Color.blue()))
        c.effects = [BlurEffect(kernel_size=9, sigma=2.0)]
        scene.add(c)

        buf = render_scene(scene, alpha=True)
        expected_sha256 = "613ec98b85e5eb9a0a5bf9c30e85e7392d860a50b97177ad4d35dca3386d7936"
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
        expected_sha256 = "23d751813e864d2b0ab7549b14b5fb4f3b65b3baf3f1d2debc603238f85ecdc8"
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
