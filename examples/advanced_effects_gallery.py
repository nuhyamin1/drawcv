"""Advanced Raster Effects visual gallery for DrawCV.

Demonstrates all 6 advanced post-processing effects:
1. Discrete 2D Convolution (custom sharpening & edge kernels)
2. Unsharp Masking Sharpening (high-frequency gradient enhancement)
3. Directional Luminance Relief Embossing (135-deg light source)
4. Sobel / Laplacian Edge Detection (synthetic & geometric contours)
5. Deterministic Film Grain / Noise (monochrome & color perturbations)
6. Retained Displacement Mapping (ripple/distortion driven by ImagePaint)

Run:
    python -m examples.advanced_effects_gallery
"""

from pathlib import Path
import cv2
import numpy as np

from drawcv import (
    Circle,
    Color,
    ConvolutionEffect,
    DisplacementChannel,
    DisplacementMapEffect,
    EdgeDetectionEffect,
    EdgeDetectionMethod,
    EmbossEffect,
    FillStyle,
    GradientStop,
    Group,
    ImageInterpolation,
    ImagePaint,
    LinearGradient,
    NoiseEffect,
    OpenCVRenderer,
    Point,
    RadialGradient,
    Rectangle,
    RoundedRectangle,
    Scene,
    SharpenEffect,
    StrokeStyle,
    Transform,
)


def make_ripple_texture(w: int = 128, h: int = 128) -> np.ndarray:
    """Generate high-contrast ripple displacement texture."""
    ys, xs = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    ripple = np.sin(r * 0.25) * 0.5 + 0.5
    disp_u8 = (ripple * 255.0).astype(np.uint8)
    return np.stack([disp_u8, disp_u8, disp_u8, np.full((h, w), 255, dtype=np.uint8)], axis=-1)


def build_card_content(x: float, y: float, w: float, h: float) -> Group:
    """Create rich multi-element graphic content to showcase raster effects."""
    content = Group()

    # Backdrop inside card
    bg = Rectangle(
        position=Point(x + 10, y + 10),
        width=w - 20,
        height=h - 20,
        fill=FillStyle(color=Color(35, 42, 58)),
    )
    content.add(bg)

    # Gradient circle
    stops = (
        GradientStop(0.0, Color(255, 95, 109)),
        GradientStop(1.0, Color(255, 195, 113)),
    )
    grad = RadialGradient(Point(x + w * 0.35, y + h * 0.5), w * 0.3, stops)
    circle = Circle(
        center=Point(x + w * 0.35, y + h * 0.5),
        radius=w * 0.24,
        fill=FillStyle(paint=grad),
    )
    content.add(circle)

    # Overlapping diamond / rotated rectangle
    grad2 = LinearGradient(
        Point(x + w * 0.5, y + h * 0.2),
        Point(x + w * 0.8, y + h * 0.8),
        (GradientStop(0.0, Color(79, 172, 254)), GradientStop(1.0, Color(0, 242, 254))),
    )
    diamond = Rectangle(
        position=Point(x + w * 0.45, y + h * 0.25),
        width=w * 0.35,
        height=h * 0.5,
        fill=FillStyle(paint=grad2),
        stroke=StrokeStyle(color=Color(255, 255, 255, 0.8), width=3.0),
    )
    content.add(diamond)

    return content


def create_gallery_scene() -> Scene:
    scene = Scene(1280, 880, background=Color(18, 22, 30))

    # Title card background
    header = Rectangle(
        position=Point(40, 30),
        width=1200,
        height=60,
        fill=FillStyle(color=Color(26, 32, 44)),
        stroke=StrokeStyle(color=Color(60, 70, 95), width=1.5),
    )
    scene.add(header)

    panel_w = 373
    panel_h = 340
    start_x = 40
    start_y = 120
    gap_x = 40
    gap_y = 40

    ripple_tex = make_ripple_texture(128, 128)
    ripple_paint = ImagePaint(image=ripple_tex, repeat="repeat")

    # 6 Panels:
    # Row 1: Convolution, Sharpen, Emboss
    # Row 2: Edge Detection, Noise, Displacement Mapping

    panel_configs = [
        # (title, effect_list, col, row)
        (
            "1. Convolution (3x3 Edge Kernel)",
            [
                ConvolutionEffect(
                    kernel=[
                        [0.0, 1.0, 0.0],
                        [1.0, -4.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ],
                    factor=1.5,
                    bias=0.3,
                )
            ],
            0,
            0,
        ),
        (
            "2. Sharpen (Unsharp Mask)",
            [SharpenEffect(amount=2.5, radius=2.5)],
            1,
            0,
        ),
        (
            "3. Emboss (135° Lighting)",
            [EmbossEffect(strength=2.2, angle=135.0, bias=0.5)],
            2,
            0,
        ),
        (
            "4. Edge Detection (Sobel)",
            [EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL, strength=2.0)],
            0,
            1,
        ),
        (
            "5. Deterministic Grain / Noise",
            [NoiseEffect(amount=0.22, seed=98765, monochrome=False)],
            1,
            1,
        ),
        (
            "6. Displacement Mapping (Ripples)",
            [
                DisplacementMapEffect(
                    map=ripple_paint,
                    scale_x=16.0,
                    scale_y=16.0,
                    x_channel=DisplacementChannel.RED,
                    y_channel=DisplacementChannel.GREEN,
                    interpolation=ImageInterpolation.CUBIC,
                )
            ],
            2,
            1,
        ),
    ]

    for title, effects, col, row in panel_configs:
        px = start_x + col * (panel_w + gap_x)
        py = start_y + row * (panel_h + gap_y)

        # Panel frame
        frame = Rectangle(
            position=Point(px, py),
            width=panel_w,
            height=panel_h,
            fill=FillStyle(color=Color(24, 29, 41)),
            stroke=StrokeStyle(color=Color(50, 60, 80), width=1.5),
        )
        scene.add(frame)

        # Content with effect
        content = build_card_content(px + 10, py + 10, panel_w - 20, panel_h - 20)
        content.effects = effects
        scene.add(content)

    return scene


def main():
    scene = create_gallery_scene()
    renderer = OpenCVRenderer()
    print("Rendering Advanced Raster Effects Gallery...")
    rendered = renderer.render(scene, alpha=False)

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "advanced_effects_gallery.png"

    cv2.imwrite(str(out_path), rendered.buffer)
    print(f"Gallery successfully saved to: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
