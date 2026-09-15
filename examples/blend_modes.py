"""Retained blend modes gallery. Run: python -m examples.blend_modes."""

from pathlib import Path
import cv2
import numpy as np

from drawcv import (
    BlendMode,
    Circle,
    Color,
    FillStyle,
    GradientStop,
    Group,
    LinearGradient,
    OpenCVRenderer,
    Point,
    RadialGradient,
    Rectangle,
    RoundedRectangle,
    Scene,
    StrokeStyle,
    Transform,
)


MODES = [
    (BlendMode.NORMAL, "NORMAL"),
    (BlendMode.MULTIPLY, "MULTIPLY"),
    (BlendMode.SCREEN, "SCREEN"),
    (BlendMode.OVERLAY, "OVERLAY"),
    (BlendMode.DARKEN, "DARKEN"),
    (BlendMode.LIGHTEN, "LIGHTEN"),
    (BlendMode.COLOR_DODGE, "COLOR DODGE"),
    (BlendMode.COLOR_BURN, "COLOR BURN"),
    (BlendMode.HARD_LIGHT, "HARD LIGHT"),
    (BlendMode.SOFT_LIGHT, "SOFT LIGHT"),
    (BlendMode.DIFFERENCE, "DIFFERENCE"),
    (BlendMode.EXCLUSION, "EXCLUSION"),
]


def build_scene() -> Scene:
    # 4 columns x 3 rows grid
    # Canvas size: 1040 x 780
    scene = Scene(1040, 780, background=Color(24, 26, 32))

    cell_w, cell_h = 240, 220
    pad_x, pad_y = 20, 25

    for idx, (mode, label) in enumerate(MODES):
        col = idx % 4
        row = idx // 4
        ox = pad_x + col * (cell_w + 15)
        oy = pad_y + row * (cell_h + 25)

        # Card container
        card = RoundedRectangle(
            x=ox,
            y=oy,
            width=cell_w,
            height=cell_h,
            corner_radius=14,
            fill=FillStyle(color=Color(34, 38, 48)),
            stroke=StrokeStyle(color=Color(55, 62, 78), width=1.5),
        )
        scene.add(card)

        # Backdrop inside card: colorful gradient rectangle
        bg_stops = (
            GradientStop(0.0, Color(240, 80, 50)),
            GradientStop(0.5, Color(220, 50, 180)),
            GradientStop(1.0, Color(40, 120, 240)),
        )
        backdrop = RoundedRectangle(
            x=ox + 15,
            y=oy + 15,
            width=cell_w - 30,
            height=cell_h - 60,
            corner_radius=10,
            fill=FillStyle(paint=LinearGradient(Point(ox + 15, oy + 15), Point(ox + cell_w - 15, oy + cell_h - 45), bg_stops)),
        )
        scene.add(backdrop)

        # Secondary backdrop accent circle
        accent = Circle(
            center=Point(ox + 65, oy + 85),
            radius=42,
            fill=FillStyle(color=Color(255, 210, 40, 0.9)),
        )
        scene.add(accent)

        # Blend foreground element: overlapping circle with radial gradient and target blend_mode
        fg_stops = (
            GradientStop(0.0, Color(80, 240, 220, 1.0)),
            GradientStop(0.7, Color(40, 180, 250, 0.9)),
            GradientStop(1.0, Color(10, 70, 200, 0.8)),
        )
        blended_shape = Circle(
            center=Point(ox + cell_w - 75, oy + 85),
            radius=46,
            fill=FillStyle(paint=RadialGradient(Point(ox + cell_w - 85, oy + 75), 52, fg_stops, space="world")),
            blend_mode=mode,
        )
        scene.add(blended_shape)

    return scene


def main():
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)

    scene = build_scene()
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    out_path = output / "blend_modes.png"
    canvas.save(out_path)
    scene.save_json(output / "blend_modes.json")
    scene.save_svg(output / "blend_modes.svg")

    # Generate labeled preview
    img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    cell_w, cell_h = 240, 220
    pad_x, pad_y = 20, 25

    for idx, (mode, label) in enumerate(MODES):
        col = idx % 4
        row = idx // 4
        ox = pad_x + col * (cell_w + 15)
        oy = pad_y + row * (cell_h + 25)

        # Bottom label bar on each card
        cv2.putText(
            img,
            label,
            (ox + 20, oy + cell_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 220, 240),
            1,
            cv2.LINE_AA,
        )

    preview_path = output / "blend_modes_preview.png"
    cv2.imwrite(str(preview_path), img)
    print(f"Gallery images written to {out_path} and {preview_path}")


if __name__ == "__main__":
    main()
