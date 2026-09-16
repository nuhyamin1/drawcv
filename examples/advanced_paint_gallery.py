"""Advanced Paint System visual gallery.

Demonstrates:
- Linear Gradient spread modes (pad, repeat, reflect)
- Radial Gradient spread modes (repeat, reflect)
- Conic Gradient full turn color sweeps and start angles
- Paintable Strokes with gradients and transforms
- ImagePaint bitmap tiling, repeat modes, and affine transforms

Run: python -m examples.advanced_paint_gallery
"""

from pathlib import Path
import cv2
import numpy as np

from drawcv import (
    Circle,
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    Group,
    ImageInterpolation,
    ImagePaint,
    Line,
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


def make_test_texture(w: int = 24, h: int = 24) -> np.ndarray:
    """Create an attractive 4-quadrant geometric tile texture."""
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    hw, hh = w // 2, h // 2
    # Top-left: vibrant cyan
    tile[:hh, :hw] = [230, 200, 30, 255]
    # Top-right: coral orange
    tile[:hh, hw:] = [50, 120, 240, 255]
    # Bottom-left: purple
    tile[hh:, :hw] = [200, 50, 150, 255]
    # Bottom-right: golden yellow
    tile[hh:, hw:] = [30, 215, 255, 255]
    # Center dot: white
    cv2.circle(tile, (hw, hh), min(hw, hh) // 3, (255, 255, 255, 255), -1)
    return tile


def build_gallery_scene() -> Scene:
    width, height = 1100, 800
    scene = Scene(width, height, background=Color(20, 22, 28, 1.0))

    stops_fire = (
        GradientStop(0.0, Color(40, 60, 240)),     # Orange/Red
        GradientStop(0.5, Color(30, 180, 255)),    # Yellow/Gold
        GradientStop(1.0, Color(240, 80, 50)),     # Blue/Violet
    )

    stops_conic = (
        GradientStop(0.0, Color(240, 50, 50)),
        GradientStop(0.25, Color(50, 220, 100)),
        GradientStop(0.5, Color(50, 150, 240)),
        GradientStop(0.75, Color(220, 200, 50)),
        GradientStop(1.0, Color(240, 50, 50)),
    )

    card_w, card_h = 230, 150
    margin_x, margin_y = 35, 45
    spacing_x, spacing_y = 265, 190

    # -------------------------------------------------------------------------
    # Row 0: Linear Gradients (pad, repeat, reflect) + Radial Repeat
    # -------------------------------------------------------------------------
    # Col 0: Linear Pad
    x0, y0 = margin_x, margin_y
    lin_pad = LinearGradient(Point(x0 + 40, y0), Point(x0 + 190, y0), stops_fire, spread="pad")
    scene.add(RoundedRectangle(x=x0, y=y0, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=lin_pad)))

    # Col 1: Linear Repeat
    x1 = margin_x + spacing_x
    lin_rep = LinearGradient(Point(x1 + 20, y0), Point(x1 + 90, y0), stops_fire, spread="repeat")
    scene.add(RoundedRectangle(x=x1, y=y0, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=lin_rep)))

    # Col 2: Linear Reflect
    x2 = margin_x + spacing_x * 2
    lin_ref = LinearGradient(Point(x2 + 20, y0), Point(x2 + 90, y0), stops_fire, spread="reflect")
    scene.add(RoundedRectangle(x=x2, y=y0, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=lin_ref)))

    # Col 3: Radial Repeat
    x3 = margin_x + spacing_x * 3
    rad_rep = RadialGradient(Point(x3 + card_w // 2, y0 + card_h // 2), 35.0, stops_fire, spread="repeat")
    scene.add(RoundedRectangle(x=x3, y=y0, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=rad_rep)))

    # -------------------------------------------------------------------------
    # Row 1: Radial Reflect, Conic Standard, Conic Rotated, Conic + Radial Ring
    # -------------------------------------------------------------------------
    y1 = margin_y + spacing_y
    # Col 0: Radial Reflect
    rad_ref = RadialGradient(Point(x0 + card_w // 2, y1 + card_h // 2), 40.0, stops_fire, spread="reflect")
    scene.add(RoundedRectangle(x=x0, y=y1, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=rad_ref)))

    # Col 1: Conic Gradient 360 Sweep
    conic_std = ConicGradient(Point(x1 + card_w // 2, y1 + card_h // 2), stops_conic)
    scene.add(RoundedRectangle(x=x1, y=y1, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=conic_std)))

    # Col 2: Conic Gradient with start_angle=135 and local transform scale
    conic_rot = ConicGradient(
        Point(x2 + card_w // 2, y1 + card_h // 2),
        stops_conic,
        start_angle=135.0,
        transform=Transform(scale_x=1.2, scale_y=0.8, pivot=Point(x2 + card_w // 2, y1 + card_h // 2)),
    )
    scene.add(RoundedRectangle(x=x2, y=y1, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=conic_rot)))

    # Col 3: Conic Gradient Disk in dark container
    scene.add(RoundedRectangle(x=x3, y=y1, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(color=Color(32, 35, 44))))
    conic_circle = Circle(
        center=Point(x3 + card_w // 2, y1 + card_h // 2),
        radius=55.0,
        fill=FillStyle(paint=ConicGradient(Point(x3 + card_w // 2, y1 + card_h // 2), stops_fire)),
    )
    scene.add(conic_circle)

    # -------------------------------------------------------------------------
    # Row 2: Paintable Strokes
    # -------------------------------------------------------------------------
    y2 = margin_y + spacing_y * 2
    # Col 0: Gradient Stroke on Rectangle
    stroke_lin = LinearGradient(Point(x0, y2), Point(x0 + card_w, y2 + card_h), stops_conic)
    scene.add(RoundedRectangle(x=x0 + 10, y=y2 + 10, width=card_w - 20, height=card_h - 20, corner_radius=14,
                               fill=FillStyle(color=Color(28, 30, 38)),
                               stroke=StrokeStyle(paint=stroke_lin, width=8.0)))

    # Col 1: Radial Gradient Stroke on Concentric Circles
    scene.add(RoundedRectangle(x=x1, y=y2, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(color=Color(28, 30, 38))))
    c_center = Point(x1 + card_w // 2, y2 + card_h // 2)
    rad_stroke = RadialGradient(c_center, 60.0, stops_fire, spread="reflect")
    scene.add(Circle(center=c_center, radius=50.0,
                     fill=None,
                     stroke=StrokeStyle(paint=rad_stroke, width=10.0)))
    scene.add(Circle(center=c_center, radius=25.0,
                     fill=None,
                     stroke=StrokeStyle(paint=rad_stroke, width=6.0)))

    # Col 2: Conic Stroke on Circle
    scene.add(RoundedRectangle(x=x2, y=y2, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(color=Color(28, 30, 38))))
    c_center2 = Point(x2 + card_w // 2, y2 + card_h // 2)
    conic_stroke = ConicGradient(c_center2, stops_conic)
    scene.add(Circle(center=c_center2, radius=48.0,
                     fill=FillStyle(color=Color(15, 17, 22)),
                     stroke=StrokeStyle(paint=conic_stroke, width=12.0)))

    # Col 3: Multi-segment gradient strokes
    scene.add(RoundedRectangle(x=x3, y=y2, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(color=Color(28, 30, 38))))
    line_grad = LinearGradient(Point(x3 + 20, y2 + 20), Point(x3 + card_w - 20, y2 + card_h - 20), stops_fire)
    scene.add(Line(start=Point(x3 + 30, y2 + 30), end=Point(x3 + card_w - 30, y2 + card_h - 30),
                   stroke=StrokeStyle(paint=line_grad, width=8.0)))
    scene.add(Line(start=Point(x3 + 30, y2 + card_h - 30), end=Point(x3 + card_w - 30, y2 + 30),
                   stroke=StrokeStyle(paint=line_grad, width=8.0)))

    # -------------------------------------------------------------------------
    # Row 3: ImagePaint Tiling & Transforms
    # -------------------------------------------------------------------------
    y3 = margin_y + spacing_y * 3
    texture = make_test_texture(24, 24)

    # Col 0: ImagePaint Repeat Tiling
    img_rep = ImagePaint(image=texture, scale=(1.2, 1.2), repeat="repeat")
    scene.add(RoundedRectangle(x=x0, y=y3, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=img_rep),
                               stroke=StrokeStyle(color=Color(60, 65, 80), width=2.0)))

    # Col 1: ImagePaint Reflect Tiling
    img_ref = ImagePaint(image=texture, scale=(1.5, 1.5), repeat="reflect")
    scene.add(RoundedRectangle(x=x1, y=y3, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=img_ref),
                               stroke=StrokeStyle(color=Color(60, 65, 80), width=2.0)))

    # Col 2: ImagePaint with Transform (Rotation 30 deg + Scaling)
    img_tf = ImagePaint(
        image=texture,
        scale=(1.2, 1.2),
        repeat="repeat",
        transform=Transform(rotation=30.0, pivot=Point(x2 + card_w // 2, y3 + card_h // 2)),
    )
    scene.add(RoundedRectangle(x=x2, y=y3, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=img_tf),
                               stroke=StrokeStyle(color=Color(60, 65, 80), width=2.0)))

    # Col 3: ImagePaint repeat="none" centered icon
    img_none = ImagePaint(
        image=texture,
        origin=Point(x3 + card_w // 2 - 36, y3 + card_h // 2 - 36),
        scale=(3.0, 3.0),
        repeat="none",
        interpolation=ImageInterpolation.NEAREST,
    )
    scene.add(RoundedRectangle(x=x3, y=y3, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(color=Color(32, 35, 44))))
    scene.add(RoundedRectangle(x=x3, y=y3, width=card_w, height=card_h, corner_radius=16,
                               fill=FillStyle(paint=img_none),
                               stroke=StrokeStyle(color=Color(60, 65, 80), width=2.0)))

    return scene


def main():
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = build_gallery_scene()
    canvas = OpenCVRenderer().render(scene, alpha=False)

    out_path = output_dir / "advanced_paint_gallery.png"
    canvas.save(out_path)

    # Add descriptive typography labels on top of the rendered image
    bgr = cv2.imread(str(out_path))
    labels = [
        # Row 0
        ("LINEAR PAD", (45, 35)),
        ("LINEAR REPEAT", (310, 35)),
        ("LINEAR REFLECT", (575, 35)),
        ("RADIAL REPEAT", (840, 35)),
        # Row 1
        ("RADIAL REFLECT", (45, 225)),
        ("CONIC 360 SWEEP", (310, 225)),
        ("CONIC TRANSFORMED", (575, 225)),
        ("CONIC DISK", (840, 225)),
        # Row 2
        ("GRADIENT STROKE RECT", (45, 415)),
        ("RADIAL STROKE CIRCLE", (310, 415)),
        ("CONIC STROKE RING", (575, 415)),
        ("DIAGONAL STROKE LINES", (840, 415)),
        # Row 3
        ("IMAGE REPEAT TILING", (45, 605)),
        ("IMAGE REFLECT TILING", (310, 605)),
        ("IMAGE ROTATED TILE", (575, 605)),
        ("IMAGE REPEAT NONE", (840, 605)),
    ]
    for text, pos in labels:
        cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 200), 1, cv2.LINE_AA)

    preview_path = output_dir / "advanced_paint_gallery_preview.png"
    cv2.imwrite(str(preview_path), bgr)
    print(f"Gallery written to: {out_path}")
    print(f"Preview with labels written to: {preview_path}")


if __name__ == "__main__":
    main()
