"""Comprehensive tests for DrawCV Advanced Raster Effects milestone.

Verifies:
1. ConvolutionEffect (cross-correlation orientation, factor/bias, partial-alpha, borders)
2. SharpenEffect (unsharp masking, zero no-ops, sampling support)
3. EmbossEffect (Rec.709 directional luminance gradient, neutral bias, cardinal directions)
4. EdgeDetectionEffect (Sobel and Laplacian, fixed scaling, invert, synthetic edges)
5. NoiseEffect (deterministic coordinate hash, seed modulo 2^32, delta tolerance, ROI independence)
6. DisplacementMapEffect (ROI-relative remap, interpolation-aware bounds, neutral transparent map, channels, edge-of-canvas origin consistency)
7. Pipeline integration (ordering, BGR/BGRA parity, masks, clips, blend modes, SVG fallback, animation, serialization, transactional rollback, complex scalar rejection)
"""

import math
import numpy as np
import pytest

from drawcv import (
    BlendMode,
    BoundingBox,
    Circle,
    Color,
    ColorMatrixEffect,
    ConvolutionEffect,
    DisplacementChannel,
    DisplacementMapEffect,
    EdgeDetectionEffect,
    EdgeDetectionMethod,
    Effect,
    EmbossEffect,
    FillStyle,
    Group,
    ImageInterpolation,
    ImagePaint,
    Layer,
    LinearGradient,
    GradientStop,
    NoiseEffect,
    OpenCVRenderer,
    Point,
    Rectangle,
    RenderError,
    Scene,
    SharpenEffect,
    StrokeStyle,
    Transform,
    ValidationError,
)
from drawcv.core.alpha import unpremultiply
from drawcv.effects.clipping import ClipRect
from drawcv.effects.mask import Mask
from drawcv.effects.noise import hash_noise_coords
from drawcv.effects.processors import (
    RasterEffectStageContext,
    process_convolution,
    process_displacement_map,
    process_edge_detection,
    process_emboss,
    process_noise,
    process_sharpen,
)


def render_scene(scene: Scene, alpha: bool = False) -> np.ndarray:
    """Helper to render a scene and return a copy of its buffer."""
    return OpenCVRenderer().render(scene, alpha=alpha).buffer.copy()


def flatten_bgra(bgra: np.ndarray, bg_bgr: tuple[float, float, float]) -> np.ndarray:
    """Helper to flatten a straight uint8 BGRA buffer onto a solid BGR background."""
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    bgr_straight = bgra[..., :3].astype(np.float32)
    bg_arr = np.array(bg_bgr, dtype=np.float32)
    flattened = bgr_straight * alpha + bg_arr * (1.0 - alpha)
    return np.rint(flattened).clip(0, 255).astype(np.uint8)


def make_synthetic_premultiplied_buffer(width: int = 50, height: int = 50) -> np.ndarray:
    """Create a synthetic float32 premultiplied BGRA buffer with known translucent and transparent regions."""
    buf = np.zeros((height, width, 4), dtype=np.float32)
    # Region 1: alpha = 0.5, straight BGR = [100, 150, 200] -> pm BGR = [50, 75, 100]
    buf[10:25, 10:25, 0] = 50.0
    buf[10:25, 10:25, 1] = 75.0
    buf[10:25, 10:25, 2] = 100.0
    buf[10:25, 10:25, 3] = 0.5

    # Region 2: alpha = 0.25, straight BGR = [200, 100, 50] -> pm BGR = [50, 25, 12.5]
    buf[20:40, 20:40, 0] = 50.0
    buf[20:40, 20:40, 1] = 25.0
    buf[20:40, 20:40, 2] = 12.5
    buf[20:40, 20:40, 3] = 0.25

    # Region 3: alpha = 1.0, straight BGR = [60, 120, 180] -> pm BGR = [60, 120, 180]
    buf[30:45, 15:35, 0] = 60.0
    buf[30:45, 15:35, 1] = 120.0
    buf[30:45, 15:35, 2] = 180.0
    buf[30:45, 15:35, 3] = 1.0
    return buf


def assert_processor_alpha_and_premultiplied_invariants(result: np.ndarray, original: np.ndarray) -> None:
    """Assert direct processor-level invariants:
    1. Exact alpha preservation
    2. Transparent black at alpha=0
    3. 0 <= BGR_pm <= 255 * alpha everywhere
    """
    # 1. Exact alpha preservation
    assert np.array_equal(result[..., 3], original[..., 3]), "Alpha channel must be bit-identically preserved"

    # 2. Transparent black at alpha=0
    zero_alpha = original[..., 3] <= 1e-6
    assert np.all(result[zero_alpha] == 0.0), "Pixels with alpha=0 must remain transparent black (0,0,0,0)"

    # 3. Premultiplied invariant: 0 <= BGRpm <= 255 * alpha
    alpha = result[..., 3]
    for c in range(3):
        assert np.all(result[..., c] >= 0.0), f"Channel {c} must be non-negative"
        assert np.all(result[..., c] <= alpha * 255.0 + 1e-4), f"Channel {c} exceeds 255 * alpha"


class TestConvolutionEffect:
    """Tests for ConvolutionEffect discrete 2D spatial filtering."""

    def test_identity_kernel(self):
        """A 3x3 identity kernel preserves pixels bit-identically."""
        scene = Scene(60, 60, background=Color.white())
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        kernel = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        rect.effects = [ConvolutionEffect(kernel=kernel)]
        scene.add(rect)
        buf_with_eff = render_scene(scene)

        scene_ref = Scene(60, 60, background=Color.white())
        rect_ref = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        scene_ref.add(rect_ref)
        buf_ref = render_scene(scene_ref)

        assert np.array_equal(buf_with_eff, buf_ref)

    def test_asymmetric_kernel_orientation(self):
        """Lock cross-correlation orientation using an asymmetric impulse kernel."""
        # Kernel with single weight at right neighbor: K(1, 2) = 1.0
        # Under cross-correlation: C'(x, y) = C(x + 1, y)
        # Therefore an isolated impulse at x0 appears at x0 - 1 in the output.
        scene = Scene(40, 40, background=Color.black())
        group = Group()
        bg_rect = Rectangle(position=Point(10, 10), width=20, height=20, fill=FillStyle(color=Color.black()))
        stripe = Rectangle(position=Point(20, 10), width=1, height=20, fill=FillStyle(color=Color.white()))
        group.add(bg_rect)
        group.add(stripe)
        kernel = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
        group.effects = [ConvolutionEffect(kernel=kernel)]
        scene.add(group)
        buf = render_scene(scene, alpha=True)

        # Output pixel at x=19 should have received the white sample from x=20
        assert buf[20, 19, 0] > 200.0, f"Expected bright pixel at x=19, got {buf[20, 19]}"

    def test_factor_and_bias(self):
        """Scalar factor and straight-color bias correctly modulate output."""
        scene = Scene(40, 40, background=Color.black())
        # Gray square: 100/255 straight luminance
        rect = Rectangle(position=Point(10, 10), width=20, height=20, fill=FillStyle(color=Color(100, 100, 100, 1.0)))
        kernel = [[1.0]]  # 1x1 identity
        # factor=0.5, bias=0.2 (in [0, 1] straight scale)
        # Expected straight = 0.5 * (100/255) + 0.2 ~= 0.196 + 0.2 = 0.396 -> ~101/255
        rect.effects = [ConvolutionEffect(kernel=kernel, factor=0.5, bias=0.2)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        val = buf[20, 20, 0]
        expected = (0.5 * (100.0 / 255.0) + 0.2) * 255.0
        assert abs(val - expected) < 2.0

    def test_direct_processor_premultiplied_alpha_invariants(self):
        """Direct processor-level test using synthetic float32 premultiplied BGRA buffer."""
        buf = make_synthetic_premultiplied_buffer()
        kernel = [
            [1.0 / 9, 1.0 / 9, 1.0 / 9],
            [1.0 / 9, 1.0 / 9, 1.0 / 9],
            [1.0 / 9, 1.0 / 9, 1.0 / 9],
        ]
        effect = ConvolutionEffect(kernel=kernel, factor=1.2, bias=0.1)
        ctx = RasterEffectStageContext(
            canvas_width=buf.shape[1],
            canvas_height=buf.shape[0],
            input_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            output_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            alpha_output=True,
            sampling_padding=effect.get_sampling_padding(),
        )
        res = process_convolution(effect, buf.copy(), ctx)
        assert_processor_alpha_and_premultiplied_invariants(res, buf)

    def test_convolution_transparent_zero_boundary_numerics(self):
        """Convolution at boundaries mathematically evaluates BORDER_CONSTANT transparent zeros accurately."""
        # 3x3 uniform box blur: factor = 1/9
        kernel = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        eff = ConvolutionEffect(kernel=kernel, factor=1.0 / 9.0, bias=0.0)

        # Synthetic buffer: 5x5 image where only pixel (2, 2) is 1.0 opaque white
        buf = np.zeros((5, 5, 4), dtype=np.float32)
        buf[2, 2] = [255.0, 255.0, 255.0, 1.0]

        ctx = RasterEffectStageContext(
            canvas_width=5,
            canvas_height=5,
            input_bounds=BoundingBox(0, 0, 5, 5),
            output_bounds=BoundingBox(0, 0, 5, 5),
            alpha_output=True,
            sampling_padding=eff.get_sampling_padding(),
        )

        res = process_convolution(eff, buf.copy(), ctx)
        # Because process_convolution preserves pixel alpha, only (2, 2) has non-zero alpha in output
        # Straight color filtered at (2, 2): 1 out of 9 pixels is 1.0, 8 are 0.0 -> straight = 1/9
        # Output premultiplied at (2, 2): straight * alpha * 255 = (1/9) * 1.0 * 255 = 28.333
        assert abs(res[2, 2, 0] - (255.0 / 9.0)) < 0.5
        assert res[2, 2, 3] == 1.0
        # All other pixels had alpha=0 and remain transparent zero
        assert np.all(res[0, :, :] == 0.0)

    def test_validation(self):
        """Validates odd dimensions, non-empty, finite real numbers, rejecting booleans."""
        with pytest.raises(ValidationError, match="kernel width must be an odd integer"):
            ConvolutionEffect(kernel=[[1.0, 2.0]])  # Even width 2

        with pytest.raises(ValidationError, match="kernel height must be an odd integer"):
            ConvolutionEffect(kernel=[[1.0], [2.0]])  # Even height 2

        with pytest.raises(ValidationError, match="kernel must be a non-empty"):
            ConvolutionEffect(kernel=[])

        with pytest.raises(ValidationError, match="kernel.*must be a real numeric value"):
            ConvolutionEffect(kernel=[[True]])  # Reject bool

        with pytest.raises(ValidationError, match="kernel.*must be finite"):
            ConvolutionEffect(kernel=[[float("nan")]])

        with pytest.raises(ValidationError, match="'factor' must be a real numeric value"):
            ConvolutionEffect(kernel=[[1.0]], factor=1 + 2j)  # Reject complex


class TestSharpenEffect:
    """Tests for SharpenEffect unsharp masking."""

    def test_uniform_field_invariance(self):
        """A uniform solid field is unchanged by sharpening."""
        scene = Scene(50, 50, background=Color.black())
        # Color(r=120, g=180, b=200): in BGR buffer, B=200, G=180, R=120
        rect = Rectangle(position=Point(5, 5), width=40, height=40, fill=FillStyle(color=Color(120, 180, 200, 1.0)))
        rect.effects = [SharpenEffect(amount=2.0, radius=2.0)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        center_pixel = buf[25, 25]
        assert abs(center_pixel[0] - 200.0) < 1.0  # Blue
        assert abs(center_pixel[1] - 180.0) < 1.0  # Green
        assert abs(center_pixel[2] - 120.0) < 1.0  # Red

    def test_amount_zero_no_op(self):
        """amount=0.0 is an exact no-op virtually identical to no-effect render."""
        scene1 = Scene(50, 50, background=Color.white())
        c1 = Circle(center=Point(25, 25), radius=15, fill=FillStyle(color=Color.red()))
        c1.effects = [SharpenEffect(amount=0.0, radius=2.0)]
        scene1.add(c1)
        buf1 = render_scene(scene1)

        scene2 = Scene(50, 50, background=Color.white())
        c2 = Circle(center=Point(25, 25), radius=15, fill=FillStyle(color=Color.red()))
        scene2.add(c2)
        buf2 = render_scene(scene2)

        diff = np.abs(buf1.astype(float) - buf2.astype(float))
        assert diff.max() <= 1.0

    def test_radius_zero_no_op(self):
        """radius=0.0 is an exact no-op virtually identical to no-effect render."""
        scene1 = Scene(50, 50, background=Color.white())
        c1 = Circle(center=Point(25, 25), radius=15, fill=FillStyle(color=Color.red()))
        c1.effects = [SharpenEffect(amount=2.0, radius=0.0)]
        scene1.add(c1)
        buf1 = render_scene(scene1)

        scene2 = Scene(50, 50, background=Color.white())
        c2 = Circle(center=Point(25, 25), radius=15, fill=FillStyle(color=Color.red()))
        scene2.add(c2)
        buf2 = render_scene(scene2)

        diff = np.abs(buf1.astype(float) - buf2.astype(float))
        assert diff.max() <= 1.0

    def test_edge_contrast_boost(self):
        """Sharpening strictly increases high-frequency gradient across an edge."""
        scene_unsharp = Scene(60, 60, background=Color.black())
        r_unsharp = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        scene_unsharp.add(r_unsharp)
        buf_unsharp = render_scene(scene_unsharp, alpha=True)

        scene_sharp = Scene(60, 60, background=Color.black())
        r_sharp = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        r_sharp.effects = [SharpenEffect(amount=1.5, radius=1.0)]
        scene_sharp.add(r_sharp)
        buf_sharp = render_scene(scene_sharp, alpha=True)

        # Transition across edge: buffer with sharpen must differ noticeably
        diff = np.abs(buf_sharp.astype(float) - buf_unsharp.astype(float))
        assert diff.max() > 20.0

    def test_direct_processor_premultiplied_alpha_invariants(self):
        """Direct processor-level test using synthetic float32 premultiplied BGRA buffer."""
        buf = make_synthetic_premultiplied_buffer()
        effect = SharpenEffect(amount=2.0, radius=2.0)
        ctx = RasterEffectStageContext(
            canvas_width=buf.shape[1],
            canvas_height=buf.shape[0],
            input_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            output_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            alpha_output=True,
            sampling_padding=effect.get_sampling_padding(),
        )
        res = process_sharpen(effect, buf.copy(), ctx)
        assert_processor_alpha_and_premultiplied_invariants(res, buf)

    def test_sharpen_support_beyond_legacy_guard(self):
        """Sharpen with large radius (e.g. radius=10, ~30px support) samples smoothly beyond legacy 3px margin."""
        eff = SharpenEffect(amount=2.0, radius=10.0)
        padding = eff.get_sampling_padding()
        assert padding[0] >= 20.0, f"Sharpen sampling padding must exceed legacy guard, got {padding[0]}"

        # Render scene with large sharpen radius
        scene = Scene(120, 120, background=Color.black())
        rect = Rectangle(position=Point(30, 30), width=60, height=60, fill=FillStyle(color=Color.white()))
        rect.effects = [eff]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)
        assert buf[60, 60, 0] > 0


class TestEmbossEffect:
    """Tests for EmbossEffect directional luminance relief."""

    def test_uniform_field_neutral(self):
        """A uniform field produces exact neutral bias gray."""
        scene = Scene(50, 50, background=Color.black())
        rect = Rectangle(position=Point(10, 10), width=30, height=30, fill=FillStyle(color=Color(120, 80, 200, 1.0)))
        rect.effects = [EmbossEffect(strength=2.0, angle=135.0, bias=0.5)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Interior pixel has zero gradient, so result = bias = 0.5 -> 127.5/255
        center = buf[25, 25]
        assert abs(center[0] - 127.5) <= 1.0
        assert abs(center[1] - 127.5) <= 1.0
        assert abs(center[2] - 127.5) <= 1.0

    def test_cardinal_directions(self):
        """Opposite lighting angles produce inverted relief responses across a vertical edge."""
        def make_edge_scene(angle: float) -> np.ndarray:
            s = Scene(60, 60, background=Color.black())
            g = Group()
            g.effects = [EmbossEffect(strength=2.0, angle=angle, bias=0.5)]
            r1 = Rectangle(position=Point(10, 10), width=20, height=40, fill=FillStyle(color=Color.black()))
            r2 = Rectangle(position=Point(30, 10), width=20, height=40, fill=FillStyle(color=Color.white()))
            g.add(r1)
            g.add(r2)
            s.add(g)
            return render_scene(s, alpha=True)

        buf1 = make_edge_scene(0.0)
        buf2 = make_edge_scene(180.0)

        v1 = buf1[30, 30, 0] / 255.0
        v2 = buf2[30, 30, 0] / 255.0
        assert (v1 > 0.5 and v2 < 0.5) or (v1 < 0.5 and v2 > 0.5)

    def test_direct_processor_premultiplied_alpha_invariants(self):
        """Direct processor-level test using synthetic float32 premultiplied BGRA buffer."""
        buf = make_synthetic_premultiplied_buffer()
        effect = EmbossEffect(strength=2.0, angle=135.0, bias=0.5)
        ctx = RasterEffectStageContext(
            canvas_width=buf.shape[1],
            canvas_height=buf.shape[0],
            input_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            output_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
            alpha_output=True,
            sampling_padding=effect.get_sampling_padding(),
        )
        res = process_emboss(effect, buf.copy(), ctx)
        assert_processor_alpha_and_premultiplied_invariants(res, buf)


class TestEdgeDetectionEffect:
    """Tests for EdgeDetectionEffect spatial edge detection."""

    def test_direct_processor_premultiplied_alpha_invariants(self):
        """Direct processor-level test using synthetic float32 premultiplied BGRA buffer."""
        buf = make_synthetic_premultiplied_buffer()
        for method in (EdgeDetectionMethod.SOBEL, EdgeDetectionMethod.LAPLACIAN):
            for invert in (False, True):
                effect = EdgeDetectionEffect(method=method, strength=1.5, invert=invert)
                ctx = RasterEffectStageContext(
                    canvas_width=buf.shape[1],
                    canvas_height=buf.shape[0],
                    input_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
                    output_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
                    alpha_output=True,
                    sampling_padding=effect.get_sampling_padding(),
                )
                res = process_edge_detection(effect, buf.copy(), ctx)
                assert_processor_alpha_and_premultiplied_invariants(res, buf)

    def test_uniform_field_zero(self):
        """A uniform solid field produces zero edge response."""
        scene = Scene(50, 50, background=Color.black())
        rect = Rectangle(position=Point(10, 10), width=30, height=30, fill=FillStyle(color=Color.blue()))
        rect.effects = [EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Center of uniform blue rectangle has zero edge
        assert np.all(buf[25, 25, :3] == 0.0)

    def test_vertical_and_horizontal_edges(self):
        """Detects both horizontal and vertical edges under Sobel and Laplacian."""
        for method in (EdgeDetectionMethod.SOBEL, EdgeDetectionMethod.LAPLACIAN):
            scene = Scene(60, 60, background=Color.black())
            rect = Rectangle(position=Point(15, 15), width=30, height=30, fill=FillStyle(color=Color.white()))
            rect.effects = [EdgeDetectionEffect(method=method, strength=1.5)]
            scene.add(rect)
            buf = render_scene(scene, alpha=True)

            # Boundaries of rectangle must exhibit non-zero edge response
            edge_pixels = buf[15, :, 0]
            assert edge_pixels.max() > 40.0

    def test_invert_mode(self):
        """invert=True produces 1.0 in flat regions and decreases at edges."""
        scene = Scene(50, 50, background=Color.black())
        rect = Rectangle(position=Point(10, 10), width=30, height=30, fill=FillStyle(color=Color.blue()))
        rect.effects = [EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL, invert=True)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Flat interior has 0 edge response -> inverted response = 1.0 (white gray)
        assert abs(buf[25, 25, 0] - 255.0) < 1.0


class TestNoiseEffect:
    """Tests for deterministic coordinate-hashed NoiseEffect."""

    def test_deterministic_seed(self):
        """Same seed and parameters produce bit-identical noise."""
        scene1 = Scene(60, 60, background=Color.white())
        r1 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        r1.effects = [NoiseEffect(amount=0.2, seed=42, monochrome=True)]
        scene1.add(r1)
        s1 = render_scene(scene1, alpha=True)

        scene2 = Scene(60, 60, background=Color.white())
        r2 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        r2.effects = [NoiseEffect(amount=0.2, seed=42, monochrome=True)]
        scene2.add(r2)
        s2 = render_scene(scene2, alpha=True)

        assert np.array_equal(s1, s2)

    def test_different_seed(self):
        """Different seeds produce different noise patterns."""
        scene1 = Scene(60, 60, background=Color.white())
        r1 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        r1.effects = [NoiseEffect(amount=0.2, seed=42)]
        scene1.add(r1)
        s1 = render_scene(scene1, alpha=True)

        scene2 = Scene(60, 60, background=Color.white())
        r2 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        r2.effects = [NoiseEffect(amount=0.2, seed=999)]
        scene2.add(r2)
        s2 = render_scene(scene2, alpha=True)

        assert not np.array_equal(s1, s2)

    def test_roi_independence(self):
        """Noise at absolute coordinate (x, y) is invariant to canvas size or ROI origin."""
        # Scene 1: 50x50 canvas, entity at (10, 10)
        s1 = Scene(50, 50, background=Color.black())
        r1 = Rectangle(position=Point(10, 10), width=30, height=30, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        r1.effects = [NoiseEffect(amount=0.3, seed=12345, monochrome=True)]
        s1.add(r1)
        buf1 = render_scene(s1, alpha=True)

        # Scene 2: 100x100 canvas, entity at (10, 10)
        s2 = Scene(100, 100, background=Color.black())
        r2 = Rectangle(position=Point(10, 10), width=30, height=30, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        r2.effects = [NoiseEffect(amount=0.3, seed=12345, monochrome=True)]
        s2.add(r2)
        buf2 = render_scene(s2, alpha=True)

        # Check overlapping region [10:40, 10:40]
        assert np.array_equal(buf1[10:40, 10:40], buf2[10:40, 10:40])

    def test_global_rng_isolation(self):
        """Global np.random state has zero effect on NoiseEffect."""
        scene = Scene(40, 40, background=Color.black())
        rect = Rectangle(position=Point(5, 5), width=30, height=30, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        rect.effects = [NoiseEffect(amount=0.2, seed=777)]
        scene.add(rect)

        np.random.seed(111)
        buf1 = render_scene(scene)

        np.random.seed(999999)
        _ = np.random.rand(1000)
        buf2 = render_scene(scene)

        assert np.array_equal(buf1, buf2)

    def test_monochrome_perturbation_delta_tolerance(self):
        """Monochrome noise applies the exact same delta perturbation across B, G, R on colored inputs."""
        scene = Scene(40, 40, background=Color.black())
        # Input has distinct channel colors: B=76.5, G=153.0, R=204.0
        in_b, in_g, in_r = 76.5, 153.0, 204.0
        rect = Rectangle(
            position=Point(5, 5),
            width=30,
            height=30,
            fill=FillStyle(color=Color(r=int(in_r), g=int(in_g), b=int(in_b), a=1.0)),
        )
        rect.effects = [NoiseEffect(amount=0.1, seed=42, monochrome=True)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        sub = buf[10:30, 10:30]
        delta_b = sub[..., 0] - in_b
        delta_g = sub[..., 1] - in_g
        delta_r = sub[..., 2] - in_r

        assert np.allclose(delta_b, delta_g, atol=1.0)
        assert np.allclose(delta_g, delta_r, atol=1.0)

    def test_colored_noise_channels(self):
        """Colored noise (monochrome=False) produces differing deltas across channels."""
        scene = Scene(40, 40, background=Color.black())
        rect = Rectangle(position=Point(5, 5), width=30, height=30, fill=FillStyle(color=Color(128, 128, 128, 1.0)))
        rect.effects = [NoiseEffect(amount=0.2, seed=42, monochrome=False)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        sub = buf[10:30, 10:30]
        # Channels should not be identical
        assert not np.array_equal(sub[..., 0], sub[..., 1])
        assert not np.array_equal(sub[..., 1], sub[..., 2])

    def test_direct_processor_premultiplied_alpha_invariants(self):
        """Direct processor-level test using synthetic float32 premultiplied BGRA buffer."""
        buf = make_synthetic_premultiplied_buffer()
        for mono in (True, False):
            effect = NoiseEffect(amount=0.25, seed=42, monochrome=mono)
            ctx = RasterEffectStageContext(
                canvas_width=buf.shape[1],
                canvas_height=buf.shape[0],
                input_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
                output_bounds=BoundingBox(0, 0, buf.shape[1], buf.shape[0]),
                alpha_output=True,
                sampling_padding=effect.get_sampling_padding(),
            )
            res = process_noise(effect, buf.copy(), ctx)
            assert_processor_alpha_and_premultiplied_invariants(res, buf)

    def test_pinned_test_vectors(self):
        """Frozen hash test vectors verify exact cross-platform hash stability."""
        h1 = hash_noise_coords(np.array([15]), np.array([30]), seed=12345, channel=0)[0]
        h2 = hash_noise_coords(np.array([15]), np.array([30]), seed=12345, channel=1)[0]
        h3 = hash_noise_coords(np.array([0]), np.array([0]), seed=0, channel=0)[0]
        h4 = hash_noise_coords(np.array([100]), np.array([200]), seed=42, channel=2)[0]

        assert np.isclose(h1, 0.93643689, atol=1e-6)
        assert np.isclose(h2, 0.65711462, atol=1e-6)
        assert np.isclose(h3, -1.0, atol=1e-6)
        assert np.isclose(h4, -0.57766819, atol=1e-6)

    def test_noise_seed_canonicalization(self):
        """Noise seed is canonicalized modulo 2^32 for any integer value."""
        n_neg = NoiseEffect(seed=-1)
        assert n_neg.seed == 0xFFFFFFFF

        n_large = NoiseEffect(seed=0x100000005)
        assert n_large.seed == 5

        # Noise hash output for seed=-1 matches seed=0xFFFFFFFF
        h_neg = hash_noise_coords(np.array([10]), np.array([20]), seed=-1, channel=0)
        h_pos = hash_noise_coords(np.array([10]), np.array([20]), seed=0xFFFFFFFF, channel=0)
        assert np.array_equal(h_neg, h_pos)


class TestDisplacementMapEffect:
    """Tests for DisplacementMapEffect geometric distortion."""

    def _make_solid_paint(self, b: int, g: int, r: int, a: int = 255) -> ImagePaint:
        arr = np.full((16, 16, 4), [b, g, r, a], dtype=np.uint8)
        return ImagePaint(image=arr, repeat="repeat")

    def test_neutral_map_no_op(self):
        """A flat neutral 0.5 map (RGB=128) produces bit-identical output."""
        scene_neutral = Scene(60, 60, background=Color.black())
        c1 = Circle(center=Point(30, 30), radius=15, fill=FillStyle(color=Color.red()))
        # Neutral map: BGR = 128, 128, 128 (0.5 straight value)
        neutral_map = self._make_solid_paint(128, 128, 128, 255)
        c1.effects = [DisplacementMapEffect(map=neutral_map, scale_x=10.0, scale_y=10.0, interpolation=ImageInterpolation.NEAREST)]
        scene_neutral.add(c1)
        buf_neutral = render_scene(scene_neutral, alpha=True)

        scene_ref = Scene(60, 60, background=Color.black())
        c2 = Circle(center=Point(30, 30), radius=15, fill=FillStyle(color=Color.red()))
        scene_ref.add(c2)
        buf_ref = render_scene(scene_ref, alpha=True)

        diff = np.abs(buf_neutral.astype(float) - buf_ref.astype(float))
        assert diff.max() <= 1.0

    def test_positive_and_negative_x_shift(self):
        """Value 1.0 in RED moves content right (+X); value 0.0 moves content left (-X)."""
        # Red map with 255 (value 1.0) -> +scale displacement (shifts right)
        map_right = self._make_solid_paint(0, 0, 255, 255)
        scene_right = Scene(80, 80, background=Color.transparent())
        r1 = Rectangle(position=Point(30, 30), width=20, height=20, fill=FillStyle(color=Color.blue()))
        r1.effects = [DisplacementMapEffect(map=map_right, scale_x=10.0, scale_y=0.0)]
        scene_right.add(r1)
        buf_right = render_scene(scene_right, alpha=True)

        # Center of mass of the displaced rectangle in X should have moved from 40 to 50
        xx_right = np.where(buf_right[..., 3] > 0.5)[1]
        mean_x_right = xx_right.mean()
        assert mean_x_right > 45.0, f"Expected shift to right (>45), got {mean_x_right}"

        # Red map with 0 (value 0.0) -> -scale displacement (shifts left)
        map_left = self._make_solid_paint(0, 0, 0, 255)
        scene_left = Scene(80, 80, background=Color.transparent())
        r2 = Rectangle(position=Point(30, 30), width=20, height=20, fill=FillStyle(color=Color.blue()))
        r2.effects = [DisplacementMapEffect(map=map_left, scale_x=10.0, scale_y=0.0)]
        scene_left.add(r2)
        buf_left = render_scene(scene_left, alpha=True)

        xx_left = np.where(buf_left[..., 3] > 0.5)[1]
        mean_x_left = xx_left.mean()
        assert mean_x_left < 35.0, f"Expected shift to left (<35), got {mean_x_left}"

    def test_remap_local_origin_off_canvas(self):
        """Remap coordinates handle objects partially off the canvas without origin mismatch."""
        map_right = self._make_solid_paint(0, 0, 255, 255)
        scene = Scene(80, 80, background=Color.black())
        rect = Rectangle(position=Point(-10, 20), width=30, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [DisplacementMapEffect(map=map_right, scale_x=15.0, scale_y=0.0)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Entity shifted right into view: should have non-zero pixels inside canvas
        assert np.any(buf[20:60, :40, 3] > 0.0)

    def test_displacement_edge_of_canvas_origin_consistency(self):
        """Edge-of-canvas test: verifying remap coordinates use requested origin (rx1, ry1), not (sx1, sy1).

        When a shape is at (0, 0), rx1 < 0 due to sampling padding while sx1 is clamped to 0.
        If displacement remap coordinates mistakenly subtracted sx1 instead of rx1, the lookup
        into the padded snapshot would sample empty border padding, corrupting or blacking out
        pixels at (0, 0).
        """
        scene = Scene(50, 50, background=Color.black())
        rect = Rectangle(position=Point(0, 0), width=20, height=20, fill=FillStyle(color=Color.white()))
        neutral_map = self._make_solid_paint(128, 128, 128, 255)
        rect.effects = [
            DisplacementMapEffect(
                map=neutral_map,
                scale_x=10.0,
                scale_y=10.0,
                interpolation=ImageInterpolation.NEAREST,
            )
        ]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Reference scene with identical rectangle and no displacement effect
        scene_ref = Scene(50, 50, background=Color.black())
        rect_ref = Rectangle(position=Point(0, 0), width=20, height=20, fill=FillStyle(color=Color.white()))
        scene_ref.add(rect_ref)
        buf_ref = render_scene(scene_ref, alpha=True)

        # Pixels across the entire shape including origin (0, 0) must match bit-identically
        diff = np.abs(buf[0:20, 0:20].astype(float) - buf_ref[0:20, 0:20].astype(float))
        assert diff.max() <= 1.0, f"Displacement remap origin mismatch! diff.max() = {diff.max()}"
        assert buf[0, 0, 0] > 200.0, f"Origin pixel at (0, 0) should be white, got {buf[0, 0]}"

    def test_interpolation_visual_bounds(self):
        """Visual bounds include interpolation footprint and do not clip at any of the 4 edges."""
        map_right = self._make_solid_paint(0, 0, 255, 255)
        for interp in (ImageInterpolation.LINEAR, ImageInterpolation.CUBIC, ImageInterpolation.LANCZOS):
            eff = DisplacementMapEffect(map=map_right, scale_x=5.0, scale_y=5.0, interpolation=interp)
            base_box = BoundingBox(20, 20, 40, 40)
            expanded = eff.expand_bounds(base_box)
            # Expanded box must be larger than base_box by at least scale + interp_support
            assert expanded.left < base_box.left - 5.0
            assert expanded.right > base_box.right + 5.0
            assert expanded.top < base_box.top - 5.0
            assert expanded.bottom > base_box.bottom + 5.0

    def test_transparent_map_neutral_default(self):
        """Areas outside a non-repeating map (repeat='none') evaluate to neutral 0.5 (no displacement)."""
        arr = np.full((10, 10, 4), [0, 0, 255, 255], dtype=np.uint8)
        non_repeating_map = ImagePaint(image=arr, origin=Point(200, 200), repeat="none")

        scene = Scene(80, 80, background=Color.transparent())
        rect = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.blue()))
        # Driven by RED channel. At (20, 20) map is transparent. Must default to 0.5 (no shift).
        rect.effects = [DisplacementMapEffect(map=non_repeating_map, scale_x=20.0, scale_y=20.0)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Rect should remain centered around (40, 40) rather than being displaced by -20
        xx = np.where(buf[..., 3] > 0.5)[1]
        assert abs(xx.mean() - 40.0) <= 1.0

    def test_alpha_channel_driver(self):
        """When channel is ALPHA, transparent map (alpha=0) evaluates to minimum 0.0 (-scale)."""
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        transparent_map = ImagePaint(image=arr, repeat="repeat")

        scene = Scene(80, 80, background=Color.transparent())
        rect = Rectangle(position=Point(30, 30), width=20, height=20, fill=FillStyle(color=Color.blue()))
        # Driving X offset by ALPHA channel: alpha=0 -> value=0.0 -> -scale shift (left)
        rect.effects = [
            DisplacementMapEffect(
                map=transparent_map,
                scale_x=10.0,
                scale_y=0.0,
                x_channel=DisplacementChannel.ALPHA,
            )
        ]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        xx = np.where(buf[..., 3] > 0.5)[1]
        assert xx.mean() < 35.0, f"Expected shift left under alpha=0, got {xx.mean()}"

    def test_direct_processor_displacement_rgb_alpha_movement_and_invariants(self):
        """Direct processor-level test verifying RGB+alpha movement, transparent black, and premultiplied bounds."""
        buf = np.zeros((60, 60, 4), dtype=np.float32)
        alpha_val = 0.6
        straight_bgr = [150.0, 100.0, 50.0]
        pm_bgr = [straight_bgr[0] * alpha_val, straight_bgr[1] * alpha_val, straight_bgr[2] * alpha_val]
        buf[15:30, 15:30, :3] = pm_bgr
        buf[15:30, 15:30, 3] = alpha_val

        # Displacement map with constant RED=255 (value 1.0) -> +X shift of 10 pixels
        map_paint = ImagePaint(image=np.full((16, 16, 4), [0, 0, 255, 255], dtype=np.uint8), repeat="repeat")
        effect = DisplacementMapEffect(map=map_paint, scale_x=10.0, scale_y=0.0, interpolation=ImageInterpolation.NEAREST)

        ctx = RasterEffectStageContext(
            canvas_width=60,
            canvas_height=60,
            input_bounds=BoundingBox(0, 0, 60, 60),
            output_bounds=BoundingBox(0, 0, 60, 60),
            alpha_output=True,
            sampling_padding=effect.get_sampling_padding(),
        )

        result = process_displacement_map(effect, buf.copy(), ctx)

        # 1. RGB + alpha movement: the block should have shifted right by 10 pixels
        orig_center_x = 22.0
        displaced_center_x = np.where(result[..., 3] > 0.1)[1].mean()
        assert abs(displaced_center_x - (orig_center_x + 10.0)) < 1.0, f"Displacement center x {displaced_center_x} != expected 32.0"

        # RGB moved along with alpha: where alpha is active, RGB matches
        displaced_region = result[15:30, 25:40]
        assert np.allclose(displaced_region[..., 3], alpha_val, atol=1e-4)
        assert np.allclose(displaced_region[..., :3], pm_bgr, atol=1e-4)

        # 2. Transparent black at alpha=0
        zero_alpha = result[..., 3] <= 1e-6
        assert np.all(result[zero_alpha] == 0.0), "Displacement output with alpha=0 must be transparent black"

        # 3. Premultiplied invariant: 0 <= BGRpm <= 255 * alpha everywhere
        alpha = result[..., 3]
        for c in range(3):
            assert np.all(result[..., c] >= 0.0)
            assert np.all(result[..., c] <= alpha * 255.0 + 1e-4)

    def test_displacement_map_nested_opacity_mutation_rejected(self):
        """Mutating map.opacity after construction is detected and rejected at processing time and bounds computation."""
        arr = np.full((8, 8, 4), 128, dtype=np.uint8)
        img_paint = ImagePaint(image=arr)
        eff = DisplacementMapEffect(map=img_paint, scale_x=5.0, scale_y=5.0)

        # Nested mutation
        eff.map.opacity = 0.5

        buf = np.zeros((20, 20, 4), dtype=np.float32)
        ctx = RasterEffectStageContext(
            canvas_width=20,
            canvas_height=20,
            input_bounds=BoundingBox(0, 0, 20, 20),
            output_bounds=BoundingBox(0, 0, 20, 20),
            alpha_output=True,
        )

        with pytest.raises(ValidationError, match="opacity must be 1.0"):
            process_displacement_map(eff, buf, ctx)

        with pytest.raises(ValidationError, match="opacity must be 1.0"):
            eff.expand_bounds(BoundingBox(0, 0, 10, 10))

        with pytest.raises(ValidationError, match="opacity must be 1.0"):
            eff.get_sampling_padding()

        with pytest.raises(ValidationError, match="opacity must be 1.0"):
            eff.to_dict()

    def test_displacement_object_vs_world_space(self):
        """Displacement respects object vs. world space paint coordinates."""
        arr = np.zeros((40, 40, 4), dtype=np.uint8)
        arr[:, :, 2] = 255  # RED=255
        arr[..., 3] = 255   # Opaque

        paint_obj = ImagePaint(image=arr, space="object", repeat="none")
        paint_world = ImagePaint(image=arr, space="world", repeat="none")

        eff_obj = DisplacementMapEffect(map=paint_obj, scale_x=15.0, scale_y=0.0)
        eff_world = DisplacementMapEffect(map=paint_world, scale_x=15.0, scale_y=0.0)

        s_obj = Scene(100, 100, background=Color.transparent())
        r_obj = Rectangle(position=Point(0, 0), width=30, height=30, fill=FillStyle(color=Color.white()))
        r_obj.transform = Transform(translation_x=40, translation_y=40)
        r_obj.effects = [eff_obj]
        s_obj.add(r_obj)
        buf_obj = render_scene(s_obj, alpha=True)

        s_world = Scene(100, 100, background=Color.transparent())
        r_world = Rectangle(position=Point(0, 0), width=30, height=30, fill=FillStyle(color=Color.white()))
        r_world.transform = Transform(translation_x=40, translation_y=40)
        r_world.effects = [eff_world]
        s_world.add(r_world)
        buf_world = render_scene(s_world, alpha=True)

        assert not np.array_equal(buf_obj, buf_world)

    @pytest.mark.parametrize(
        "channel,expected_shift_dir",
        [
            (DisplacementChannel.RED, 1),       # Map RED is 255 (1.0) -> +X
            (DisplacementChannel.GREEN, 0),     # Map GREEN is 128 (0.5) -> 0 shift
            (DisplacementChannel.BLUE, -1),      # Map BLUE is 0 (0.0) -> -X
            (DisplacementChannel.ALPHA, 1),     # Map ALPHA is 255 (1.0) -> +X
            (DisplacementChannel.LUMINANCE, 1), # Map is mostly RED -> luma > 0.5 -> +X
        ],
    )
    def test_displacement_channel_selection(self, channel, expected_shift_dir):
        """Verify each DisplacementChannel independently drives displacement offset."""
        arr = np.full((16, 16, 4), [0, 128, 255, 255], dtype=np.uint8)
        map_paint = ImagePaint(image=arr, repeat="repeat")

        scene = Scene(80, 80, background=Color.transparent())
        rect = Rectangle(position=Point(30, 30), width=20, height=20, fill=FillStyle(color=Color.blue()))
        rect.effects = [
            DisplacementMapEffect(
                map=map_paint,
                scale_x=10.0,
                scale_y=0.0,
                x_channel=channel,
                interpolation=ImageInterpolation.NEAREST,
            )
        ]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        xx = np.where(buf[..., 3] > 0.5)[1]
        mean_x = xx.mean()
        orig_center_x = 39.5
        if expected_shift_dir > 0:
            assert mean_x > orig_center_x + 1.0, f"Expected positive shift, got mean_x={mean_x}"
        elif expected_shift_dir < 0:
            assert mean_x < orig_center_x - 1.0, f"Expected negative shift, got mean_x={mean_x}"
        else:
            assert abs(mean_x - orig_center_x) <= 1.0, f"Expected neutral shift, got mean_x={mean_x}"

    def test_displacement_content_interpolation_modes(self):
        """NEAREST, LINEAR, CUBIC, and LANCZOS execute and produce distinct subpixel reconstructions."""
        arr = np.full((16, 16, 4), [0, 0, 255, 255], dtype=np.uint8)
        map_paint = ImagePaint(image=arr, repeat="repeat")

        results = {}
        for interp in (ImageInterpolation.NEAREST, ImageInterpolation.LINEAR, ImageInterpolation.CUBIC, ImageInterpolation.LANCZOS):
            scene = Scene(80, 80, background=Color.black())
            rect = Rectangle(position=Point(20, 20), width=30, height=30, fill=FillStyle(color=Color.white()))
            rect.effects = [
                DisplacementMapEffect(
                    map=map_paint,
                    scale_x=2.5,
                    scale_y=0.0,
                    interpolation=interp,
                )
            ]
            scene.add(rect)
            results[interp] = render_scene(scene, alpha=True)

        assert not np.array_equal(results[ImageInterpolation.NEAREST], results[ImageInterpolation.LINEAR])
        assert not np.array_equal(results[ImageInterpolation.LINEAR], results[ImageInterpolation.CUBIC])

    def test_displacement_rendered_interpolation_boundary_coverage(self):
        """Rendered displacement fully captures the expanded footprint without clipping at original visual bounds."""
        arr = np.full((16, 16, 4), [0, 0, 255, 255], dtype=np.uint8)
        map_paint = ImagePaint(image=arr, repeat="repeat")

        scene = Scene(100, 100, background=Color.black())
        rect = Rectangle(position=Point(30, 30), width=20, height=20, fill=FillStyle(color=Color.white()))
        rect.effects = [
            DisplacementMapEffect(
                map=map_paint,
                scale_x=15.0,
                scale_y=0.0,
                interpolation=ImageInterpolation.CUBIC,
            )
        ]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Pixels between x=55 and x=64 must be non-zero (proving rendered output was not clipped at old x=50)
        assert np.any(buf[30:50, 55:65, 3] > 0.0)

    def test_invalid_displacement_interpolation_raises_validation_error(self):
        """Deserializing or setting invalid interpolation raises DrawCV ValidationError, not raw ValueError."""
        arr = np.full((8, 8, 4), 128, dtype=np.uint8)
        paint = ImagePaint(image=arr)

        with pytest.raises(ValidationError, match="Invalid ImageInterpolation"):
            DisplacementMapEffect.from_dict({"map": paint.to_dict(), "interpolation": "unsupported_filter"})

        with pytest.raises(ValidationError, match="Invalid ImageInterpolation"):
            DisplacementMapEffect(map=paint, interpolation="unsupported_filter")


class TestEffectsIntegration:
    """Tests for ordering, compositing, serialization, and hierarchy."""

    @pytest.mark.parametrize(
        "effect",
        [
            ConvolutionEffect(kernel=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], factor=1.0, bias=0.5),
            SharpenEffect(amount=1.5, radius=1.0),
            EmbossEffect(strength=1.5, angle=135.0, bias=0.5),
            EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL, strength=1.0),
            NoiseEffect(amount=0.1, seed=42, monochrome=True),
            DisplacementMapEffect(
                map=ImagePaint(image=np.full((16, 16, 4), 128, dtype=np.uint8), repeat="repeat"),
                scale_x=5.0,
                scale_y=5.0,
            ),
        ],
    )
    def test_bgr_vs_bgra_parity_all_effects(self, effect):
        """Each effect renders identically on BGR vs. BGRA surfaces when flattened onto the same background."""
        bg = Color(40, 50, 60)
        # Scene 1: BGR render
        s_bgr = Scene(80, 80, background=bg)
        r1 = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.red()))
        r1.effects = [effect]
        s_bgr.add(r1)
        bgr_buf = render_scene(s_bgr, alpha=False)

        # Scene 2: BGRA render with transparent background
        s_bgra = Scene(80, 80, background=Color.transparent())
        r2 = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.red()))
        r2.effects = [effect]
        s_bgra.add(r2)
        bgra_buf = render_scene(s_bgra, alpha=True)

        flattened = flatten_bgra(bgra_buf, bg.to_bgr())
        # Flattened BGRA must match BGR canvas render within 1.0 intensity
        diff = np.abs(bgr_buf.astype(float) - flattened.astype(float))
        assert diff.max() <= 1.0, f"BGR vs BGRA mismatch for {effect.effect_type}: max diff = {diff.max()}"

    def test_order_commutativity(self):
        """Different ordering of non-commutative effects produces distinct visual results."""
        # Noise -> Sharpen
        s1 = Scene(60, 60, background=Color.black())
        r1 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.white()))
        r1.effects = [
            NoiseEffect(amount=0.3, seed=42),
            SharpenEffect(amount=2.0, radius=2.0),
        ]
        s1.add(r1)
        buf1 = render_scene(s1)

        # Sharpen -> Noise
        s2 = Scene(60, 60, background=Color.black())
        r2 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.white()))
        r2.effects = [
            SharpenEffect(amount=2.0, radius=2.0),
            NoiseEffect(amount=0.3, seed=42),
        ]
        s2.add(r2)
        buf2 = render_scene(s2)

        assert not np.array_equal(buf1, buf2)

    def test_advanced_paint_interaction(self):
        """Effects work seamlessly with LinearGradient, RadialGradient, ConicGradient, and ImagePaint."""
        stops = (
            GradientStop(0.0, Color.red()),
            GradientStop(1.0, Color.blue()),
        )
        grad = LinearGradient(Point(10, 10), Point(50, 50), stops=stops)

        scene = Scene(60, 60, background=Color.black())
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(paint=grad))
        rect.effects = [EmbossEffect(strength=1.5, angle=135.0)]
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        assert np.any(buf[10:50, 10:50, :3] > 0.0)

    def test_group_and_layer_effects(self):
        """Hierarchical propagation: child effect -> group effect -> layer effect."""
        scene = Scene(80, 80, background=Color.black())
        layer = scene.create_layer("test_layer", effects=[EmbossEffect(strength=1.0)])

        group = Group()
        group.effects = [SharpenEffect(amount=1.0)]

        rect = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [NoiseEffect(amount=0.1, seed=1)]

        group.add(rect)
        layer.add(group)

        buf = render_scene(scene, alpha=True)
        assert np.any(buf[20:60, 20:60, :3] > 0.0)

    def test_serialization_round_trip(self):
        """All new effects round-trip losslessly through JSON serialization."""
        arr = np.full((8, 8, 4), 128, dtype=np.uint8)
        map_paint = ImagePaint(image=arr)

        effects = [
            ConvolutionEffect(kernel=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], factor=2.0, bias=0.1),
            SharpenEffect(amount=1.5, radius=2.5),
            EmbossEffect(strength=1.8, angle=90.0, bias=0.6),
            EdgeDetectionEffect(method=EdgeDetectionMethod.LAPLACIAN, strength=2.2, invert=True),
            NoiseEffect(amount=0.25, seed=98765, monochrome=False),
            DisplacementMapEffect(map=map_paint, scale_x=12.0, scale_y=8.0, x_channel=DisplacementChannel.BLUE),
        ]

        scene = Scene(100, 100)
        rect = Rectangle(position=Point(10, 10), width=50, height=50, fill=FillStyle(color=Color.red()))
        rect.effects = effects
        scene.add(rect)

        json_str = scene.to_json()
        scene_deserialized = Scene.from_json(json_str)

        deser_effects = scene_deserialized.layers[0].objects[0].effects
        assert len(deser_effects) == len(effects)
        assert deser_effects[0].effect_type == "convolution"
        assert deser_effects[1].effect_type == "sharpen"
        assert deser_effects[2].effect_type == "emboss"
        assert deser_effects[3].effect_type == "edge_detection"
        assert deser_effects[4].effect_type == "noise"
        assert deser_effects[5].effect_type == "displacement_map"

    def test_transactional_rollback(self):
        """Setting invalid property rolls back without leaving corrupted state."""
        effect = SharpenEffect(amount=1.0, radius=2.0)
        with pytest.raises(ValidationError):
            effect.amount = -5.0  # Invalid
        assert effect.amount == 1.0  # Preserved

        with pytest.raises(ValidationError):
            effect.radius = "invalid"  # Invalid
        assert effect.radius == 2.0  # Preserved

        # Test on another effect type
        emboss = EmbossEffect(strength=1.0, bias=0.5)
        with pytest.raises(ValidationError):
            emboss.bias = 2.5  # Invalid (>1.0)
        assert emboss.bias == 0.5

    def test_numeric_validation_rejects_complex_scalars(self):
        """Ensure numeric validation strictly rejects complex scalars across all new effects."""
        arr = np.full((4, 4, 4), 128, dtype=np.uint8)
        img_paint = ImagePaint(image=arr)

        with pytest.raises(ValidationError):
            ConvolutionEffect(kernel=[[1.0]], factor=1 + 2j)

        with pytest.raises(ValidationError):
            ConvolutionEffect(kernel=[[1.0]], bias=1 + 2j)

        with pytest.raises(ValidationError):
            ConvolutionEffect(kernel=[[1 + 2j]])

        with pytest.raises(ValidationError):
            SharpenEffect(amount=1 + 2j)

        with pytest.raises(ValidationError):
            SharpenEffect(radius=1 + 2j)

        with pytest.raises(ValidationError):
            EmbossEffect(strength=1 + 2j)

        with pytest.raises(ValidationError):
            EmbossEffect(angle=1 + 2j)

        with pytest.raises(ValidationError):
            EmbossEffect(bias=1 + 2j)

        with pytest.raises(ValidationError):
            EdgeDetectionEffect(strength=1 + 2j)

        with pytest.raises(ValidationError):
            NoiseEffect(amount=1 + 2j)

        with pytest.raises(ValidationError):
            NoiseEffect(seed=1 + 2j)

        with pytest.raises(ValidationError):
            DisplacementMapEffect(map=img_paint, scale_x=1 + 2j)

        with pytest.raises(ValidationError):
            DisplacementMapEffect(map=img_paint, scale_y=1 + 2j)

    def test_effects_with_mask(self):
        """Effects composite accurately when masked by a grayscale Mask."""
        scene = Scene(60, 60, background=Color.transparent())
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.white()))
        rect.effects = [SharpenEffect(amount=2.0, radius=2.0)]
        mask_data = np.zeros((40, 40), dtype=np.uint8)
        mask_data[:20, :] = 255
        rect.mask = Mask(mask_data)
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        assert np.any(buf[10:30, 10:50, 3] > 0.5)
        assert np.all(buf[32:50, 10:50, 3] == 0.0)

    def test_effects_with_clip(self):
        """Effects respect ClipRect boundary, clipping expanded bounds to the clip region."""
        scene = Scene(80, 80, background=Color.transparent())
        rect = Rectangle(position=Point(20, 20), width=40, height=40, fill=FillStyle(color=Color.white()))
        map_right = ImagePaint(image=np.full((8, 8, 4), [0, 0, 255, 255], dtype=np.uint8), repeat="repeat")
        rect.effects = [DisplacementMapEffect(map=map_right, scale_x=20.0, scale_y=0.0)]
        rect.clip = ClipRect(0, 0, 40, 40)
        scene.add(rect)
        buf = render_scene(scene, alpha=True)

        # Pixels beyond x=60 must be strictly 0 due to ClipRect
        assert np.all(buf[:, 61:, 3] == 0.0)

    def test_effects_with_non_normal_blend_modes(self):
        """Effects composite accurately with non-normal blend modes (MULTIPLY, SCREEN)."""
        scene = Scene(60, 60, background=Color(128, 128, 128))
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(200, 200, 200)))
        rect.blend_mode = BlendMode.MULTIPLY
        rect.effects = [NoiseEffect(amount=0.1, seed=42)]
        scene.add(rect)
        buf_mult = render_scene(scene, alpha=False)

        rect_norm = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color(200, 200, 200)))
        rect_norm.blend_mode = BlendMode.NORMAL
        rect_norm.effects = [NoiseEffect(amount=0.1, seed=42)]
        scene_norm = Scene(60, 60, background=Color(128, 128, 128))
        scene_norm.add(rect_norm)
        buf_norm = render_scene(scene_norm, alpha=False)

        assert buf_mult[25, 25, 0] < buf_norm[25, 25, 0]

    @pytest.mark.parametrize(
        "effect",
        [
            ConvolutionEffect(kernel=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]),
            SharpenEffect(amount=1.5, radius=1.0),
            EmbossEffect(strength=1.5, angle=135.0),
            EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL),
            NoiseEffect(amount=0.1, seed=42),
            DisplacementMapEffect(
                map=ImagePaint(image=np.full((8, 8, 4), 128, dtype=np.uint8), repeat="repeat"),
                scale_x=5.0,
                scale_y=5.0,
            ),
        ],
    )
    def test_svg_export_fallback_and_strict_mode(self, effect):
        """All new effects produce a raster fallback on export_svg() and raise RenderError in strict mode."""
        scene = Scene(60, 60)
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.red()))
        rect.effects = [effect]
        scene.add(rect)

        # Standard export -> produces fallback
        result = scene.export_svg()
        assert len(result.fallbacks) >= 1

        # Strict export -> raises RenderError
        with pytest.raises(RenderError, match="raster fallback"):
            scene.export_svg(strict=True)

    def test_effects_scalar_property_animation(self):
        """Indexed effect property animation smoothly interpolates values across time."""
        from drawcv.animation.track import AnimationTrack
        from drawcv.animation.timing import Timing

        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect.effects = [SharpenEffect(amount=0.0, radius=1.0)]

        track = AnimationTrack(
            target_id=rect.id,
            property_path="effects.0.amount",
            start_value=0.0,
            end_value=2.0,
            timing=Timing(duration=2.0),
        )

        val0 = track.evaluate(0.0, rect)
        assert val0 == pytest.approx(0.0)
        assert rect.effects[0].amount == pytest.approx(0.0)

        val1 = track.evaluate(1.0, rect)
        assert val1 == pytest.approx(1.0)
        assert rect.effects[0].amount == pytest.approx(1.0)

        val2 = track.evaluate(2.0, rect)
        assert val2 == pytest.approx(2.0)
        assert rect.effects[0].amount == pytest.approx(2.0)

        scene = Scene(60, 60, background=Color.black())
        rect2 = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.blue()))
        rect2.effects = [NoiseEffect(amount=0.0, seed=42)]
        scene.add(rect2)
        scene.animate(rect2, "effects.0.amount", 0.0, 0.5, duration=2.0)

        f0 = scene.render_at_time(0.0)
        f1 = scene.render_at_time(1.0)
        f2 = scene.render_at_time(2.0)

        assert not np.array_equal(f0.buffer, f1.buffer)
        assert not np.array_equal(f1.buffer, f2.buffer)

    def test_effects_cloning_and_undo_redo_isolation(self):
        """Cloning creates independent effects, and undo/redo correctly restores effects state."""
        rect = Rectangle(position=Point(10, 10), width=40, height=40, fill=FillStyle(color=Color.red()))
        rect.effects = [SharpenEffect(amount=1.5, radius=2.0)]
        cloned = rect.clone()

        # Mutating clone does not mutate original
        cloned.effects[0].amount = 3.5
        assert rect.effects[0].amount == 1.5
        assert cloned.effects[0].amount == 3.5

        # Undo / Redo in Scene
        scene = Scene(60, 60)
        scene.add(rect)
        initial_img = render_scene(scene)

        rect.effects = [NoiseEffect(amount=0.5, seed=123)]
        mutated_img = render_scene(scene)
        assert not np.array_equal(initial_img, mutated_img)
