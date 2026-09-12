"""Demonstration of retained-mode graphics lifecycle in DrawCV.

Visualizes 3 sequential states of the EXACT same scene and objects:
- State 1: Initial small blue circle
- State 2: The circle is translated and enlarged in-place (no pixel erasing needed)
- State 3: The circle is re-colored, re-sized, and re-styled with a new background shape

Proves that the scene model is authoritative and the raster is purely derived.
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawcv import (
    Circle,
    Color,
    FillStyle,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    StrokeStyle,
)


def main():
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = OpenCVRenderer()
    scene = Scene(width=600, height=400, background=Color.white())

    # Create persistent circle
    circle = Circle(
        center=Point(150, 200),
        radius=50,
        stroke=StrokeStyle(color=Color.from_hex("#1A365D"), width=3.0),
        fill=FillStyle(color=Color.from_hex("#3182CE"))
    )
    scene.add(circle)

    # State 1 Render
    canvas1 = renderer.render(scene)
    s1_path = output_dir / "retained_state_1.png"
    canvas1.save(s1_path)
    print(f"Rendered State 1 -> {s1_path}")

    # State 2: Modify circle in-place (move right, enlarge radius)
    circle.move(150, 0)       # center is now (300, 200)
    circle.radius = 80
    canvas2 = renderer.render(scene)
    s2_path = output_dir / "retained_state_2.png"
    canvas2.save(s2_path)
    print(f"Rendered State 2 -> {s2_path}")

    # State 3: Re-style circle and insert background rectangle behind it
    circle.move(100, 0)       # center is now (400, 200)
    circle.radius = 110
    circle.fill.color = Color.from_hex("#E53E3E")      # Red fill
    circle.stroke.color = Color.from_hex("#38A169")    # Green stroke
    circle.stroke.width = 8.0
    circle.z_index = 2

    # Add a background rectangle at lower z_index
    bg_rect = Rectangle(
        position=Point(50, 50),
        width=500,
        height=300,
        fill=FillStyle(color=Color.from_hex("#ECC94B"), opacity=0.4),
        stroke=StrokeStyle(color=Color.from_hex("#D69E2E"), width=2.0),
        z_index=1
    )
    scene.add(bg_rect)

    canvas3 = renderer.render(scene)
    s3_path = output_dir / "retained_state_3.png"
    canvas3.save(s3_path)
    print(f"Rendered State 3 -> {s3_path}")


if __name__ == "__main__":
    main()
