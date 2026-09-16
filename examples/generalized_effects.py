"""Generalized Ordered Effects Architecture gallery. Run: python -m examples.generalized_effects."""

from pathlib import Path
import cv2
import numpy as np

from drawcv import (
    BlurEffect,
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
    RoundedRectangle,
    SaturationEffect,
    Scene,
    SepiaEffect,
    ShadowEffect,
    StrokeStyle,
)


def build_scene() -> Scene:
    # 4 columns x 2 rows grid: 8 cards
    # Canvas: 1100 x 600
    scene = Scene(1100, 600, background=Color(24, 26, 32))

    cards_meta = [
        ("Blur -> Shadow", "Blur executed before drop shadow"),
        ("Shadow -> Blur", "Shadow executed before spatial blur"),
        ("Dual Directional Shadows", "Two duplicate ShadowEffects (+ / -)"),
        ("True Outer Glow", "Preserves translucent interior"),
        ("Hue Shift (120 deg)", "Canonical W3C hueRotate matrix"),
        ("Sepia & Saturation", "Warm tone & saturation boost"),
        ("ColorMatrix Invert", "4x5 affine color space transform"),
        ("Hierarchical Effects", "Child Glow -> Group Shadow"),
    ]

    cell_w, cell_h = 240, 240
    pad_x, pad_y = 28, 40

    for idx, (title, subtitle) in enumerate(cards_meta):
        col = idx % 4
        row = idx // 4
        ox = pad_x + col * (cell_w + 24)
        oy = pad_y + row * (cell_h + 30)

        # Base card background
        card = RoundedRectangle(
            x=ox,
            y=oy,
            width=cell_w,
            height=cell_h,
            corner_radius=12.0,
            fill=FillStyle(color=Color(36, 40, 50)),
            stroke=StrokeStyle(width=1.5, color=Color(60, 66, 82)),
        )
        scene.add(card)

        cx = ox + cell_w // 2
        cy = oy + cell_h // 2 + 10

        if idx == 0:
            # Blur -> Shadow
            disc = Circle(center=Point(cx, cy), radius=35, fill=FillStyle(color=Color(255, 60, 60)))
            disc.effects = [
                BlurEffect(kernel_size=25, sigma=8.0),
                ShadowEffect(offset_x=8.0, offset_y=8.0, blur_radius=3.0, color=Color(0, 255, 0, 0.8)),
            ]
            scene.add(disc)

        elif idx == 1:
            # Shadow -> Blur
            disc = Circle(center=Point(cx, cy), radius=35, fill=FillStyle(color=Color(255, 60, 60)))
            disc.effects = [
                ShadowEffect(offset_x=8.0, offset_y=8.0, blur_radius=3.0, color=Color(0, 255, 0, 0.8)),
                BlurEffect(kernel_size=25, sigma=8.0),
            ]
            scene.add(disc)

        elif idx == 2:
            # Dual Shadows
            rect = RoundedRectangle(
                x=cx - 30, y=cy - 30, width=60, height=60, corner_radius=10.0,
                fill=FillStyle(color=Color(70, 140, 250)),
            )
            rect.effects = [
                ShadowEffect(offset_x=14.0, offset_y=14.0, blur_radius=4.0, color=Color(0, 0, 0, 0.7)),
                ShadowEffect(offset_x=-14.0, offset_y=-14.0, blur_radius=4.0, color=Color(0, 200, 255, 0.5)),
            ]
            scene.add(rect)

        elif idx == 3:
            # True Outer Glow with translucent interior
            disc = Circle(
                center=Point(cx, cy),
                radius=36,
                fill=FillStyle(color=Color(255, 50, 120, 0.5)),
            )
            disc.effects = [
                GlowEffect(blur_radius=10.0, color=Color(255, 230, 0, 1.0)),
            ]
            scene.add(disc)

        elif idx == 4:
            # Hue Shift
            rect = RoundedRectangle(
                x=cx - 32, y=cy - 32, width=64, height=64, corner_radius=12.0,
                fill=FillStyle(color=Color(255, 50, 50)),
            )
            rect.effects = [
                HueShiftEffect(angle=120.0),
            ]
            scene.add(rect)

        elif idx == 5:
            # Sepia & Saturation
            rect = RoundedRectangle(
                x=cx - 32, y=cy - 32, width=64, height=64, corner_radius=12.0,
                fill=FillStyle(color=Color(120, 140, 220)),
            )
            rect.effects = [
                SepiaEffect(intensity=0.8),
                SaturationEffect(factor=1.5),
            ]
            scene.add(rect)

        elif idx == 6:
            # ColorMatrix Invert
            invert_matrix = (
                (-1.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, -1.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0, 0.0),
            )
            disc = Circle(center=Point(cx, cy), radius=35, fill=FillStyle(color=Color(255, 180, 40)))
            disc.effects = [
                ColorMatrixEffect(matrix=invert_matrix),
            ]
            scene.add(disc)

        elif idx == 7:
            # Hierarchical effects
            group = Group()
            group.effects = [ShadowEffect(offset_x=10.0, offset_y=10.0, blur_radius=4.0, color=Color(0, 0, 0, 0.7))]
            child = RoundedRectangle(
                x=cx - 30, y=cy - 30, width=60, height=60, corner_radius=10.0,
                fill=FillStyle(color=Color(50, 220, 160)),
                effects=[GlowEffect(blur_radius=8.0, color=Color(255, 255, 100, 0.9))],
            )
            group.add(child)
            scene.add(group)

    return scene


def main() -> None:
    scene = build_scene()
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generalized_effects.png"
    cv2.imwrite(str(out_path), canvas.buffer)
    print(f"Generated generalized effects gallery: {out_path}")


if __name__ == "__main__":
    main()
