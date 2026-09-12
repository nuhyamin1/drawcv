"""Phase 5 Visual Demonstration Script for DrawCV — Freehand Engine Showcase.

Demonstrates:
1. Modular Path Processing Pipeline:
   - Raw sampled noisy points vs RDP simplification vs Chaikin smoothing vs Catmull-Rom spline.
   - Visualizing raw control points and refined contours.
2. Stylus Pressure Sensitivity & Calligraphy:
   - Dynamic variable width ribbons mapping pressure to width (w_min=2.0 to w_max=18.0).
   - Expressive calligraphic flourishes and loops.
3. Physical Velocity Dynamics (Fountain Pen Taper):
   - Fast stroke flick tapering based on velocity v = delta_d / delta_t.
   - Real-time simulation of pen dynamics.
4. Scene Graph & Hierarchical Transformations:
   - Freehand strokes combined inside Groups and Layers with geometric primitives.
   - Hierarchical rotation, scaling, and relative positioning.
"""

import math
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawcv import (
    Canvas,
    Circle,
    Color,
    FillStyle,
    FreehandStroke,
    Group,
    Layer,
    Line,
    OpenCVRenderer,
    Point,
    Rectangle,
    RoundedRectangle,
    Scene,
    StrokePoint,
    StrokeStyle,
    align_centers,
    place_below,
    place_right_of,
)


def create_card_panel(
    scene: Scene,
    layer_name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    tag: str,
) -> tuple[RoundedRectangle, float, float]:
    """Create a stylized dark card container with title header."""
    card = RoundedRectangle(
        x=x,
        y=y,
        width=w,
        height=h,
        corner_radius=12.0,
        fill=FillStyle(color=Color.from_hex("#131B2E"), opacity=0.85),
        stroke=StrokeStyle(color=Color.from_hex("#334155"), width=1.5),
        tags={tag, "card"},
    )
    scene.add(card, layer=layer_name)

    # Header accent bar
    header_bar = RoundedRectangle(
        x=x + 16,
        y=y + 16,
        width=6,
        height=28,
        corner_radius=3.0,
        fill=FillStyle(color=Color.from_hex("#38BDF8")),
    )
    scene.add(header_bar, layer=layer_name)

    # Return card and usable inner content origin (content_x, content_y)
    return card, x + 32, y + 60


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase5_demo.png"

    print("Initializing Phase 5 Demo Scene (1600x1000)...")
    scene = Scene(width=1600, height=1000, background=Color.from_hex("#0B0F19"))

    # Register structured layers
    scene.create_layer("grid", z_order=-10)
    scene.create_layer("panels", z_order=0)
    scene.create_layer("strokes", z_order=10)
    scene.create_layer("overlays", z_order=20)

    # -------------------------------------------------------------------------
    # 1. Subtle Background Coordinate Grid
    # -------------------------------------------------------------------------
    grid_color = Color.from_hex("#1E293B")
    for gx in range(0, 1601, 80):
        scene.add(
            Line(start=Point(gx, 0), end=Point(gx, 1000), stroke=StrokeStyle(color=grid_color, width=1.0, opacity=0.4)),
            layer="grid",
        )
    for gy in range(0, 1001, 80):
        scene.add(
            Line(start=Point(0, gy), end=Point(1600, gy), stroke=StrokeStyle(color=grid_color, width=1.0, opacity=0.4)),
            layer="grid",
        )

    # Title Banner Panel
    banner = RoundedRectangle(
        x=40,
        y=20,
        width=1520,
        height=75,
        corner_radius=10.0,
        fill=FillStyle(color=Color.from_hex("#131B2E"), opacity=0.9),
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=1.5),
    )
    scene.add(banner, layer="panels")

    # Banner accent indicator badge
    badge = RoundedRectangle(
        x=60,
        y=38,
        width=120,
        height=38,
        corner_radius=6.0,
        fill=FillStyle(color=Color.from_hex("#0284C7")),
    )
    scene.add(badge, layer="panels")

    # =========================================================================
    # Panel 1: Top-Left — Modular Processing Algorithms
    # =========================================================================
    card1, c1_x, c1_y = create_card_panel(
        scene, "panels", 40, 115, 730, 410,
        "MODULAR PATH PROCESSING",
        "Raw Samples -> RDP Simplification -> Chaikin Smoothing -> Catmull-Rom Spline",
        "panel_algorithms"
    )

    # Generate a complex hand-drawn S-curve with natural noise
    base_points: list[StrokePoint] = []
    num_samples = 28
    for i in range(num_samples):
        t = i / (num_samples - 1)
        px = c1_x + 30 + t * 620
        # Double sinusoidal wave with high-frequency noise
        py = c1_y + 110 + math.sin(t * math.pi * 2.5) * 60 + math.sin(i * 1.7) * 9.0
        base_points.append(StrokePoint(px, py, pressure=0.5))

    # A. Raw Noisy Stroke (White, thin with red vertices)
    raw_stroke = FreehandStroke(
        points=[p.translate(0, -50) for p in base_points],
        stroke=StrokeStyle(color=Color.from_hex("#94A3B8"), width=1.5),
    )
    scene.add(raw_stroke, layer="strokes")
    # Draw vertex dots on raw stroke
    for pt in raw_stroke.points:
        scene.add(
            Circle(center=pt.to_point(), radius=2.5, fill=FillStyle(color=Color.from_hex("#F43F5E"))),
            layer="overlays",
        )

    # B. RDP Simplified Stroke (Amber, epsilon=6.0)
    rdp_stroke = FreehandStroke(
        points=[p.translate(0, 30) for p in base_points],
        stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=2.5),
        simplification="rdp",
        simplification_tolerance=8.0,
    )
    scene.add(rdp_stroke, layer="strokes")
    # Draw simplified vertices
    for pt in rdp_stroke.get_processed_points():
        scene.add(
            Circle(center=pt.to_point(), radius=3.5, fill=FillStyle(color=Color.from_hex("#F59E0B"))),
            layer="overlays",
        )

    # C. Chaikin Smoothed Stroke (Emerald, 2 iterations)
    chaikin_stroke = FreehandStroke(
        points=[p.translate(0, 110) for p in base_points],
        stroke=StrokeStyle(color=Color.from_hex("#10B981"), width=3.0),
        smoothing="chaikin",
        smoothing_iterations=2,
    )
    scene.add(chaikin_stroke, layer="strokes")

    # D. Catmull-Rom Centripetal Spline (Cyan, C1 continuous)
    catmull_stroke = FreehandStroke(
        points=[p.translate(0, 190) for p in base_points],
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=3.5),
        interpolation="catmull_rom",
        interpolation_samples=8,
    )
    scene.add(catmull_stroke, layer="strokes")

    # =========================================================================
    # Panel 2: Top-Right — Stylus Pressure Dynamics & Calligraphy
    # =========================================================================
    card2, c2_x, c2_y = create_card_panel(
        scene, "panels", 830, 115, 730, 410,
        "PRESSURE DYNAMICS & CALLIGRAPHY",
        "Variable Width Tapering (w_min=2.0, w_max=22.0) with Round Joints",
        "panel_calligraphy"
    )

    # Create dynamic calligraphic ribbon loops with modulating pressure
    calligraphy_pts: list[StrokePoint] = []
    num_cal = 70
    for i in range(num_cal):
        t = i / (num_cal - 1)
        angle = t * math.pi * 5.0
        # Lissajous figure / lemniscate flourish
        rx = c2_x + 330 + math.cos(angle) * 260 * (1.0 - t * 0.25)
        ry = c2_y + 160 + math.sin(angle * 1.6) * 110 * (1.0 - t * 0.15)
        # Dynamic pressure: heavy downstrokes, thin hairlines
        p_val = 0.5 + 0.5 * math.sin(angle - math.pi / 4.0)
        p_val = max(0.05, min(0.98, p_val))
        calligraphy_pts.append(StrokePoint(rx, ry, pressure=p_val))

    calligraphy_stroke = FreehandStroke(
        points=calligraphy_pts,
        stroke=StrokeStyle(color=Color.from_hex("#EC4899"), width=8.0),
        smoothing="chaikin",
        smoothing_iterations=2,
        interpolation="catmull_rom",
        interpolation_samples=6,
        variable_width=True,
        width_mode="pressure",
        min_width=2.5,
        max_width=20.0,
    )
    scene.add(calligraphy_stroke, layer="strokes")

    # Second complementary gold flourishing ribbon
    gold_pts: list[StrokePoint] = []
    for i in range(50):
        t = i / 49.0
        gx = c2_x + 80 + t * 520
        gy = c2_y + 300 + math.sin(t * math.pi * 3.5) * 35
        # Pressure increases from start to mid, tapers at end
        p_val = math.sin(t * math.pi) ** 1.5
        gold_pts.append(StrokePoint(gx, gy, pressure=p_val))

    gold_stroke = FreehandStroke(
        points=gold_pts,
        stroke=StrokeStyle(color=Color.from_hex("#FBBF24"), width=6.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="pressure",
        min_width=1.5,
        max_width=16.0,
    )
    scene.add(gold_stroke, layer="strokes")

    # =========================================================================
    # Panel 3: Bottom-Left — Physical Velocity Dynamics (Fountain Pen Taper)
    # =========================================================================
    card3, c3_x, c3_y = create_card_panel(
        scene, "panels", 40, 555, 730, 405,
        "VELOCITY DYNAMICS (FOUNTAIN PEN TAPER)",
        "v = delta_d / delta_t: Rapid flicks taper to thin hairlines, slow strokes remain bold",
        "panel_velocity"
    )

    # Simulate three fountain pen gestures: slow, medium, rapid flick
    # 1. Deliberate slow arc (v = 80 px/s -> thick)
    slow_pts: list[StrokePoint] = []
    for i in range(30):
        t = i / 29.0
        px = c3_x + 50 + t * 580
        py = c3_y + 60 + math.sin(t * math.pi) * 30
        slow_pts.append(StrokePoint(px, py, velocity=60.0))

    slow_stroke = FreehandStroke(
        points=slow_pts,
        stroke=StrokeStyle(color=Color.from_hex("#60A5FA"), width=6.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="velocity",
        min_width=2.0,
        max_width=16.0,
        velocity_min=0.0,
        velocity_max=800.0,
    )
    scene.add(slow_stroke, layer="strokes")

    # 2. Accelerating stroke (slow start, rapid flick finish)
    accel_pts: list[StrokePoint] = []
    for i in range(40):
        t = i / 39.0
        px = c3_x + 50 + t * 580
        py = c3_y + 160 + math.cos(t * math.pi * 1.5) * 40
        # Velocity starts at 50 px/s, accelerates exponentially to 750 px/s
        vel = 50.0 + (t ** 2.5) * 700.0
        accel_pts.append(StrokePoint(px, py, velocity=vel))

    accel_stroke = FreehandStroke(
        points=accel_pts,
        stroke=StrokeStyle(color=Color.from_hex("#A855F7"), width=6.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="velocity",
        min_width=1.5,
        max_width=18.0,
        velocity_min=0.0,
        velocity_max=800.0,
    )
    scene.add(accel_stroke, layer="strokes")

    # 3. Dynamic signatures with real-time velocity calculation via timestamps
    sig_stroke = FreehandStroke(
        stroke=StrokeStyle(color=Color.from_hex("#34D399"), width=5.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="velocity",
        min_width=2.0,
        max_width=14.0,
        velocity_min=0.0,
        velocity_max=1000.0,
    )
    # Simulate drawing with timestamps: quick loops followed by rapid flourish
    cur_t = 0.0
    for i in range(50):
        progress = i / 49.0
        sx = c3_x + 50 + progress * 580
        sy = c3_y + 260 + math.sin(progress * math.pi * 6.0) * (25.0 * math.sin(progress * math.pi))
        # Accelerate timestamp intervals
        dt = max(0.005, 0.05 * (1.0 - progress * 0.8))
        cur_t += dt
        sig_stroke.add_point((sx, sy), timestamp=cur_t)
    scene.add(sig_stroke, layer="strokes")

    # =========================================================================
    # Panel 4: Bottom-Right — Scene Hierarchy & Spatial Transformations
    # =========================================================================
    card4, c4_x, c4_y = create_card_panel(
        scene, "panels", 830, 555, 730, 405,
        "HIERARCHICAL SCENE & TRANSFORMS",
        "Freehand strokes grouped with geometric shapes, rotated, and scaled in scene graph",
        "panel_hierarchy"
    )

    # Build an artistic hand-drawn crest badge assembly inside a Group
    crest_group = Group(id="freehand_crest")

    # Geometric backing plate
    backing_shield = RoundedRectangle(
        x=-110, y=-110, width=220, height=220, corner_radius=24.0,
        fill=FillStyle(color=Color.from_hex("#1E293B"), opacity=0.9),
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=2.5),
    )
    crest_group.add(backing_shield)

    inner_ring = Circle(
        center=Point(0, 0),
        radius=90.0,
        stroke=StrokeStyle(color=Color.from_hex("#64748B"), width=1.5),
    )
    crest_group.add(inner_ring)

    # Spiral freehand stroke drawn at local center (0, 0)
    spiral_pts: list[StrokePoint] = []
    for i in range(80):
        theta = (i / 79.0) * math.pi * 6.0
        r = 10.0 + theta * 11.0
        sp_x = math.cos(theta) * r
        sp_y = math.sin(theta) * r
        p_val = 0.2 + 0.8 * (i / 79.0)
        spiral_pts.append(StrokePoint(sp_x, sp_y, pressure=p_val))

    spiral_stroke = FreehandStroke(
        id="crest_spiral",
        points=spiral_pts,
        stroke=StrokeStyle(color=Color.from_hex("#F43F5E"), width=4.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="pressure",
        min_width=2.0,
        max_width=12.0,
    )
    crest_group.add(spiral_stroke)

    # Center jewel
    center_jewel = Circle(
        center=Point(0, 0),
        radius=14.0,
        fill=FillStyle(color=Color.from_hex("#38BDF8")),
        stroke=StrokeStyle(color=Color.white(), width=2.0),
    )
    crest_group.add(center_jewel)

    # Position Instance 1: Original orientation
    crest_group.move(c4_x + 180, c4_y + 170)
    scene.add(crest_group, layer="strokes")

    # Instance 2: Cloned, Rotated 45 degrees, and scaled 0.8x
    crest_clone = crest_group.clone()
    crest_clone.id = "crest_clone"
    crest_clone.rotate(35.0)
    crest_clone.scale(0.8)
    crest_clone.move(290, 0)
    scene.add(crest_clone, layer="strokes")

    # =========================================================================
    # 5. Render Scene to Canvas & Export
    # =========================================================================
    print("Rendering high-resolution Phase 5 Canvas...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    print(f"Saving rendered showcase to {output_file}...")
    canvas.save(str(output_file))
    print("Phase 5 Demo rendered successfully!")


if __name__ == "__main__":
    main()
