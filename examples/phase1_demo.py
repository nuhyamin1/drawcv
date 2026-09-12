"""Phase 1 Visual Demonstration Script for DrawCV.

Demonstrates:
- Canvas creation and white background
- High-quality anti-aliased lines with various stroke widths and colors
- Filled and stroked rectangles
- Filled and stroked circles
- Semi-transparent shapes showcasing mathematically correct alpha compositing
- Overlapping shapes demonstrating z-order control
- Visibility toggle (hidden shape)
- Retained-mode manipulation in place before rendering
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawcv import (
    Circle,
    Color,
    FillStyle,
    Line,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    StrokeStyle,
)


def main():
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase1_demo.png"

    print("Creating DrawCV scene (1200x800)...")
    scene = Scene(width=1200, height=800, background=Color.white())

    # 1. Anti-aliased lines with varying stroke widths and colors
    print("Adding anti-aliased line demonstration...")
    for i, (color, width) in enumerate([
        (Color.from_hex("#333333"), 1.0),
        (Color.from_hex("#4A90E2"), 3.0),
        (Color.from_hex("#50E3C2"), 6.0),
        (Color.from_hex("#F5A623"), 10.0),
        (Color.from_hex("#E02020"), 16.0),
    ]):
        y = 80 + i * 35
        line = Line(
            start=Point(100, y),
            end=Point(450, y),
            stroke=StrokeStyle(color=color, width=width)
        )
        scene.add(line)

    # 2. Diagonal crossing line
    diag_line = Line(
        start=Point(100, 260),
        end=Point(450, 80),
        stroke=StrokeStyle(color=Color.from_hex("#9013FE"), width=4.0)
    )
    scene.add(diag_line)

    # 3. Solid filled rectangle with crisp stroke
    print("Adding rectangle and circle entities...")
    rect_solid = Rectangle(
        position=Point(550, 80),
        width=260,
        height=180,
        stroke=StrokeStyle(color=Color.from_hex("#1A365D"), width=5.0),
        fill=FillStyle(color=Color.from_hex("#63B3ED"))
    )
    scene.add(rect_solid)

    # 4. Overlapping semi-transparent circles showcasing alpha compositing
    # Circle 1: Red translucent fill (effective alpha = 0.6)
    c1 = Circle(
        center=Point(960, 160),
        radius=90,
        stroke=StrokeStyle(color=Color.from_hex("#C53030"), width=4.0),
        fill=FillStyle(color=Color.from_hex("#FC8181"), opacity=0.6),
        z_index=1
    )
    # Circle 2: Blue translucent fill (effective alpha = 0.6), overlapping c1
    c2 = Circle(
        center=Point(1040, 220),
        radius=90,
        stroke=StrokeStyle(color=Color.from_hex("#2B6CB0"), width=4.0),
        fill=FillStyle(color=Color.from_hex("#63B3ED"), opacity=0.6),
        z_index=2
    )
    scene.add(c1)
    scene.add(c2)

    # 5. Complex overlapping compositing section
    # Background card
    card = Rectangle(
        position=Point(100, 360),
        width=480,
        height=360,
        stroke=StrokeStyle(color=Color.from_hex("#CBD5E0"), width=2.0),
        fill=FillStyle(color=Color.from_hex("#F7FAFC"))
    )
    scene.add(card)

    # Three overlapping translucent colored lenses inside the card
    lens_colors = [
        ("#FF416C", Point(280, 500)),  # Magenta-red
        ("#8A2387", Point(380, 500)),  # Purple
        ("#F8B195", Point(330, 580)),  # Peach
    ]
    for hex_color, center in lens_colors:
        lens = Circle(
            center=center,
            radius=80,
            stroke=StrokeStyle(color=Color.from_hex("#2D3748"), width=3.0, opacity=0.8),
            fill=FillStyle(color=Color.from_hex(hex_color), opacity=0.55),
            z_index=5
        )
        scene.add(lens)

    # 6. Retained-mode object manipulation:
    # We create an object, add it, and then modify its geometry and appearance
    print("Demonstrating retained-mode in-place mutation...")
    manipulated_rect = Rectangle(
        position=Point(650, 400),
        width=100,
        height=100,
        stroke=StrokeStyle(color=Color.black(), width=2.0),
        fill=FillStyle(color=Color.from_hex("#CBD5E0"))
    )
    scene.add(manipulated_rect)

    # In-place mutations on the retained object
    manipulated_rect.move(50, 50)               # Translated from (650, 400) to (700, 450)
    manipulated_rect.width = 380                # Resized width
    manipulated_rect.height = 220               # Resized height
    manipulated_rect.fill.color = Color.from_hex("#38A169")  # Changed color to emerald green
    manipulated_rect.fill.opacity = 0.75        # Translucent fill
    manipulated_rect.stroke.color = Color.from_hex("#22543D")
    manipulated_rect.stroke.width = 6.0

    # Retained circle placed over manipulated_rect to prove z-order
    star_circle = Circle(
        center=manipulated_rect.center,
        radius=55,
        stroke=StrokeStyle(color=Color.white(), width=5.0),
        fill=FillStyle(color=Color.from_hex("#D69E2E"), opacity=0.9),
        z_index=10
    )
    scene.add(star_circle)

    # 7. Hidden object (should NOT appear on canvas)
    hidden_rect = Rectangle(
        position=Point(700, 450),
        width=200,
        height=200,
        fill=FillStyle(color=Color.black()),
        visible=False
    )
    scene.add(hidden_rect)

    # 8. Render using OpenCVRenderer
    print("Rendering scene with OpenCVRenderer...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    # 9. Save output
    canvas.save(output_file)
    print(f"Phase 1 demonstration image successfully saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
