"""Phase 2 Visual Demonstration Script for DrawCV.

Demonstrates:
- Affine transformation rendering: arbitrary rotation and non-uniform scaling
- Non-scaling screen stroke rule (stroke remains constant screen pixels under scale/rotation)
- Transformed circles rasterized as rotated ellipses
- Geometric connector lines pinned cleanly to transformed shape anchors
- Visual overlay of tight analytical world bounding boxes (get_bounds())
- Hit-testing indicator showing topmost selected object
"""

import sys
from pathlib import Path

# Ensure root directory is on sys.path
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
    output_file = output_dir / "phase2_demo.png"

    print("Creating DrawCV Phase 2 scene (1200x800)...")
    scene = Scene(width=1200, height=800, background=Color.white())

    # 1. Rotated Rectangles with non-scaling stroke
    print("Adding rotated rectangles...")
    angles = [15.0, 45.0, 75.0]
    colors = ["#4A90E2", "#50E3C2", "#9013FE"]
    for i, (angle, col) in enumerate(zip(angles, colors)):
        rect = Rectangle(
            position=Point(150 + i * 140, 180),
            width=100,
            height=60,
            stroke=StrokeStyle(color=Color.from_hex("#1A365D"), width=3.0),
            fill=FillStyle(color=Color.from_hex(col), opacity=0.7)
        )
        rect.rotate(angle)
        scene.add(rect)

        # Draw computed tight world AABB in faint red to verify analytical bounds
        w_bounds = rect.get_bounds()
        bbox_visual = Rectangle(
            position=w_bounds.top_left,
            width=w_bounds.width,
            height=w_bounds.height,
            stroke=StrokeStyle(color=Color.from_hex("#E53E3E"), width=1.0),
            fill=None,
            z_index=-1
        )
        scene.add(bbox_visual)

    # 2. Transformed Ellipses (non-uniformly scaled circles with rotation)
    print("Adding transformed rotated ellipses...")
    ellipse1 = Circle(
        center=Point(800, 200),
        radius=60,
        stroke=StrokeStyle(color=Color.from_hex("#C53030"), width=4.0),
        fill=FillStyle(color=Color.from_hex("#FEB2B2"), opacity=0.6)
    )
    ellipse1.scale(1.8, 0.7)  # Stretched horizontally
    ellipse1.rotate(35.0)     # Rotated 35°
    scene.add(ellipse1)

    # Overlay analytical ellipse bounds in faint red
    e_bounds = ellipse1.get_bounds()
    e_box = Rectangle(
        position=e_bounds.top_left,
        width=e_bounds.width,
        height=e_bounds.height,
        stroke=StrokeStyle(color=Color.from_hex("#E53E3E"), width=1.0),
        fill=None,
        z_index=-1
    )
    scene.add(e_box)

    # 3. Geometric Anchors & Connector Demonstration
    print("Demonstrating geometric anchor connections...")
    # Base card box rotated 20°
    source_box = Rectangle(
        position=Point(150, 480),
        width=160,
        height=90,
        stroke=StrokeStyle(color=Color.from_hex("#2D3748"), width=3.0),
        fill=FillStyle(color=Color.from_hex("#ED8936"), opacity=0.8)
    )
    source_box.rotate(20.0)
    scene.add(source_box)

    # Target circle rotated & scaled
    target_node = Circle(
        center=Point(550, 520),
        radius=50,
        stroke=StrokeStyle(color=Color.from_hex("#2D3748"), width=3.0),
        fill=FillStyle(color=Color.from_hex("#38A169"), opacity=0.8)
    )
    target_node.scale(1.3, 0.9)
    target_node.rotate(-15.0)
    scene.add(target_node)

    # Connector line attached from rotated top-right corner to circle circumference anchor
    start_pt = source_box.anchor("top_right")
    end_pt = target_node.anchor_at_angle(180)  # Left edge of circle circumference

    connector = Line(
        start=start_pt,
        end=end_pt,
        stroke=StrokeStyle(color=Color.from_hex("#2B6CB0"), width=4.0)
    )
    scene.add(connector)

    # Small pin markers at connection points
    pin1 = Circle(center=start_pt, radius=5, fill=FillStyle(color=Color.black()), z_index=10)
    pin2 = Circle(center=end_pt, radius=5, fill=FillStyle(color=Color.black()), z_index=10)
    scene.add(pin1)
    scene.add(pin2)

    # 4. Hit-Testing Demonstration
    print("Demonstrating hit-testing query...")
    # Deep cloned shapes with distinct z-indexes
    hub = Circle(
        center=Point(900, 520),
        radius=70,
        stroke=StrokeStyle(color=Color.from_hex("#319795"), width=4.0),
        fill=FillStyle(color=Color.from_hex("#B2F5EA"), opacity=0.7),
        z_index=2
    )
    hub_clone = hub.clone()
    hub_clone.move(40, 30)
    hub_clone.fill.color = Color.from_hex("#D69E2E")
    hub_clone.fill.opacity = 0.5
    hub_clone.z_index = 3

    scene.add(hub)
    scene.add(hub_clone)

    # Perform hit-test at cursor point (920, 530)
    test_cursor = Point(920, 530)
    top_hit = scene.hit_test_top(test_cursor.x, test_cursor.y)

    # Draw cursor crosshair indicator
    cursor_h = Line(
        start=Point(test_cursor.x - 12, test_cursor.y),
        end=Point(test_cursor.x + 12, test_cursor.y),
        stroke=StrokeStyle(color=Color.red(), width=2.0),
        z_index=20
    )
    cursor_v = Line(
        start=Point(test_cursor.x, test_cursor.y - 12),
        end=Point(test_cursor.x, test_cursor.y + 12),
        stroke=StrokeStyle(color=Color.red(), width=2.0),
        z_index=20
    )
    scene.add(cursor_h)
    scene.add(cursor_v)

    print(f"Hit test at {test_cursor}: Hit Topmost Object ID = {top_hit.id if top_hit else 'None'}")

    # Render scene
    print("Rendering scene with OpenCVRenderer...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    canvas.save(output_file)
    print(f"Phase 2 demonstration image saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
