"""Phase 4 Visual Demonstration Script for DrawCV — Scene System Showcase.

Demonstrates:
- Multi-Layer Architecture:
  * 'grid' layer (z_order=-10): Subtle isometric/Cartesian technical grid
  * 'schematics' layer (z_order=0): Core structural objects and assemblies
  * 'annotations' layer (z_order=10): Engineering dimensions, arrows, and leader lines
  * 'overlay' layer (z_order=20): Glassmorphism status panels, badges, and focus rings
- Semantic Groups (Hierarchical Transforms):
  * Multi-part articulated robotic arm assembly with nested wrist/end-effector groups
  * Rigid group rotation and translation preserving all internal child geometries
- Declarative Relative Positioning Utilities:
  * align_left, align_top, align_center_x, align_center_y, align_centers
  * place_below(..., gap=20), place_right_of(..., gap=20)
  * distribute_horizontally, distribute_vertically
- Logical Selection & Batch Manipulation:
  * Multi-object rigid constellation transformation around collective center
- Recursive Tag & Metadata Queries
"""

import math
from pathlib import Path
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawcv import (
    Arc,
    ArcClosure,
    Arrow,
    ArrowHeadStyle,
    Circle,
    Color,
    Ellipse,
    FillStyle,
    Group,
    Line,
    OpenCVRenderer,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    RoundedRectangle,
    Scene,
    StrokeStyle,
    align_bottom,
    align_center_x,
    align_center_y,
    align_centers,
    align_left,
    align_right,
    align_top,
    distribute_horizontally,
    distribute_vertically,
    place_above,
    place_below,
    place_left_of,
    place_right_of,
)


def main():
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase4_demo.png"

    print("Building DrawCV Phase 4 Scene (1600x1000)...")
    # Modern dark canvas (Slate 950: #0B0F19)
    scene = Scene(width=1600, height=1000, background=Color.from_hex("#0B0F19"))

    # =========================================================================
    # 1. Multi-Layer Setup
    # =========================================================================
    grid_layer = scene.create_layer("grid", z_order=-10, opacity=0.35)
    schematics_layer = scene.create_layer("schematics", z_order=0)
    annotations_layer = scene.create_layer("annotations", z_order=10)
    overlay_layer = scene.create_layer("overlay", z_order=20)

    # Populate Grid Layer: Engineering Cartesian Grid
    for x in range(0, 1600, 50):
        is_major = (x % 200 == 0)
        grid_line = Line(
            start=Point(x, 0),
            end=Point(x, 1000),
            stroke=StrokeStyle(
                color=Color.from_hex("#38BDF8" if is_major else "#1E293B"),
                width=1.5 if is_major else 0.75,
            )
        )
        scene.add(grid_line, layer="grid")

    for y in range(0, 1000, 50):
        is_major = (y % 200 == 0)
        grid_line = Line(
            start=Point(0, y),
            end=Point(1600, y),
            stroke=StrokeStyle(
                color=Color.from_hex("#38BDF8" if is_major else "#1E293B"),
                width=1.5 if is_major else 0.75,
            )
        )
        scene.add(grid_line, layer="grid")

    # =========================================================================
    # Section Card Helper (Glassmorphism containers)
    # =========================================================================
    def add_section_card(x: float, y: float, w: float, h: float, title: str):
        card = RoundedRectangle(
            x=x, y=y, width=w, height=h, corner_radius=12.0,
            fill=FillStyle(color=Color.from_hex("#111827"), opacity=0.85),
            stroke=StrokeStyle(color=Color.from_hex("#374151"), width=1.5)
        )
        scene.add(card, layer="schematics")

        # Decorative title bar accent
        accent_bar = Rectangle(
            position=Point(x + 16, y + 16), width=4.0, height=20.0,
            fill=FillStyle(color=Color.from_hex("#38BDF8")),
            stroke=None
        )
        scene.add(accent_bar, layer="overlay")
        return card

    # =========================================================================
    # 2. Semantic Hierarchical Groups: Articulated Robotic Mechanism
    # =========================================================================
    add_section_card(50, 40, 720, 440, "HIERARCHICAL GROUPS: ARTICULATED MECHANISM")

    # Base turntable
    base_plate = RoundedRectangle(
        x=160, y=360, width=180, height=36, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#1E293B")),
        stroke=StrokeStyle(color=Color.from_hex("#64748B"), width=2.0)
    )
    base_joint = Circle(
        center=Point(250, 360), radius=28,
        fill=FillStyle(color=Color.from_hex("#3B82F6"), opacity=0.6),
        stroke=StrokeStyle(color=Color.from_hex("#93C5FD"), width=2.5)
    )

    # Lower arm boom
    lower_arm = Polygon(
        vertices=[
            Point(240, 360), Point(260, 360),
            Point(275, 220), Point(225, 220)
        ],
        fill=FillStyle(color=Color.from_hex("#0284C7"), opacity=0.8),
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=2.0)
    )

    # Elbow joint
    elbow_joint = Circle(
        center=Point(250, 220), radius=22,
        fill=FillStyle(color=Color.from_hex("#F59E0B"), opacity=0.7),
        stroke=StrokeStyle(color=Color.from_hex("#FCD34D"), width=2.0)
    )

    # Forearm assembly (Sub-group)
    forearm_boom = RoundedRectangle(
        x=238, y=100, width=24, height=120, corner_radius=6.0,
        fill=FillStyle(color=Color.from_hex("#0D9488"), opacity=0.85),
        stroke=StrokeStyle(color=Color.from_hex("#2DD4BF"), width=2.0)
    )
    wrist_joint = Circle(
        center=Point(250, 100), radius=16,
        fill=FillStyle(color=Color.from_hex("#EC4899"), opacity=0.8),
        stroke=StrokeStyle(color=Color.from_hex("#F472B6"), width=2.0)
    )

    # Gripper pads (Nested End-Effector group)
    gripper_left = Polygon(
        vertices=[Point(236, 100), Point(220, 60), Point(228, 55), Point(242, 90)],
        fill=FillStyle(color=Color.from_hex("#E11D48")),
        stroke=StrokeStyle(color=Color.WHITE, width=1.5)
    )
    gripper_right = Polygon(
        vertices=[Point(264, 100), Point(280, 60), Point(272, 55), Point(258, 90)],
        fill=FillStyle(color=Color.from_hex("#E11D48")),
        stroke=StrokeStyle(color=Color.WHITE, width=1.5)
    )
    end_effector = Group([gripper_left, gripper_right], name="end_effector")

    # Assemble nested forearm group
    forearm_group = Group([forearm_boom, wrist_joint, end_effector], name="forearm_assembly")

    # Rotate forearm group around elbow joint
    forearm_group.rotate(32.0, pivot=Point(250, 220))

    # Assemble full mechanism group
    robot_assembly = Group(
        [base_plate, base_joint, lower_arm, elbow_joint, forearm_group],
        name="robotic_arm"
    )

    # Position and add mechanism to schematics layer
    robot_assembly.move(100, 20)
    scene.add(robot_assembly, layer="schematics")

    # World-space bounding box of the entire group rendered in annotations layer
    assembly_bounds = robot_assembly.get_bounds()
    bbox_vis = Rectangle(
        position=assembly_bounds.top_left,
        width=assembly_bounds.width,
        height=assembly_bounds.height,
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=1.0)
    )
    scene.add(bbox_vis, layer="annotations")

    # Group center anchor marker
    center_marker = Circle(
        center=robot_assembly.anchor("center"), radius=5.0,
        fill=FillStyle(color=Color.from_hex("#F43F5E")),
        stroke=StrokeStyle(color=Color.WHITE, width=1.5)
    )
    scene.add(center_marker, layer="overlay")

    # Dimension leader arrow to group anchor
    arrow_to_arm = Arrow(
        start=Point(620, 120),
        end=robot_assembly.anchor("top_right"),
        head_style=ArrowHeadStyle.TRIANGLE,
        stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=2.0)
    )
    scene.add(arrow_to_arm, layer="annotations")

    # =========================================================================
    # 3. Declarative Layout via Relative Positioning Utilities
    # =========================================================================
    add_section_card(810, 40, 740, 440, "DECLARATIVE LAYOUT & POSITIONING UTILITIES")

    # Card 1: Master Control Block
    card_master = RoundedRectangle(
        x=850, y=90, width=200, height=75, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#1E1B4B")),
        stroke=StrokeStyle(color=Color.from_hex("#6366F1"), width=2.0)
    )
    scene.add(card_master, layer="schematics")

    # Card 2: placed directly below Card 1 with default gap=20.0, aligned center
    card_slave_1 = RoundedRectangle(
        x=0, y=0, width=200, height=75, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#064E3B")),
        stroke=StrokeStyle(color=Color.from_hex("#10B981"), width=2.0)
    )
    place_below(card_slave_1, card_master, gap=20.0, align="center")
    scene.add(card_slave_1, layer="schematics")

    # Card 3: placed directly below Card 2 with default gap=20.0, aligned center
    card_slave_2 = RoundedRectangle(
        x=0, y=0, width=200, height=75, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#701A75")),
        stroke=StrokeStyle(color=Color.from_hex("#D946EF"), width=2.0)
    )
    place_below(card_slave_2, card_slave_1, gap=20.0, align="center")
    scene.add(card_slave_2, layer="schematics")

    # Connecting bus arrow placed to the right of the stack
    bus_line = Arrow(
        start=Point(1070, 127),
        end=Point(1070, 317),
        head_style=ArrowHeadStyle.DIAMOND,
        stroke=StrokeStyle(color=Color.from_hex("#A855F7"), width=3.0)
    )
    scene.add(bus_line, layer="annotations")

    # Horizontal telemetry indicators distributed evenly across right wing
    telemetry_cards = [
        RoundedRectangle(
            x=0, y=0, width=110, height=60, corner_radius=6.0,
            fill=FillStyle(color=Color.from_hex("#1F2937")),
            stroke=StrokeStyle(color=Color.from_hex("#4B5563"), width=1.5)
        )
        for _ in range(3)
    ]
    # Position initial card relative to card_master
    place_right_of(telemetry_cards[0], card_master, gap=50.0, align="top")
    for c in telemetry_cards[1:]:
        align_top(c, telemetry_cards[0])
    # Distribute the 3 cards horizontally with 18px spacing
    distribute_horizontally(telemetry_cards, spacing=18.0)
    for c in telemetry_cards:
        scene.add(c, layer="schematics")

    # Circular gauge gauges placed below telemetry cards
    gauges = [
        Circle(center=Point(0, 0), radius=26, fill=FillStyle(color=Color.from_hex("#0369A1"), opacity=0.6), stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=2.0)),
        Circle(center=Point(0, 0), radius=26, fill=FillStyle(color=Color.from_hex("#B45309"), opacity=0.6), stroke=StrokeStyle(color=Color.from_hex("#FBBF24"), width=2.0)),
        Circle(center=Point(0, 0), radius=26, fill=FillStyle(color=Color.from_hex("#BE185D"), opacity=0.6), stroke=StrokeStyle(color=Color.from_hex("#F472B6"), width=2.0)),
    ]
    for i, gauge in enumerate(gauges):
        align_center_x(gauge, telemetry_cards[i])
        place_below(gauge, telemetry_cards[i], gap=35.0)
        scene.add(gauge, layer="schematics")

    # =========================================================================
    # 4. Multi-Object Selection & Constellation Transformation
    # =========================================================================
    add_section_card(50, 520, 720, 440, "SELECTION: RIGID CONSTELLATION TRANSFORMS")

    # Create 3 constellation satellites
    sat1 = Rectangle(position=Point(150, 680), width=60, height=60, fill=FillStyle(color=Color.from_hex("#3B82F6"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=2.0), id="sat1")
    sat2 = Circle(center=Point(320, 620), radius=35, fill=FillStyle(color=Color.from_hex("#EC4899"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=2.0), id="sat2")
    sat3 = RoundedRectangle(x=280, y=780, width=70, height=50, corner_radius=10.0, fill=FillStyle(color=Color.from_hex("#10B981"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=2.0), id="sat3")

    scene.add(sat1, layer="schematics")
    scene.add(sat2, layer="schematics")
    scene.add(sat3, layer="schematics")

    # Initial ghost bounds in dashed line
    sel_initial = scene.select([sat1, sat2, sat3])
    init_b = sel_initial.bounds
    ghost_bbox = Rectangle(
        position=init_b.top_left, width=init_b.width, height=init_b.height,
        stroke=StrokeStyle(color=Color.from_hex("#475569"), width=1.0)
    )
    scene.add(ghost_bbox, layer="annotations")

    # Perform coordinated multi-object rigid rotation & move via Selection!
    sel_initial.rotate(35.0)
    sel_initial.move(140, 20)

    # Highlight transformed selection collective bounds in bright amber
    trans_b = sel_initial.bounds
    active_bbox = Rectangle(
        position=trans_b.top_left, width=trans_b.width, height=trans_b.height,
        stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=1.5)
    )
    scene.add(active_bbox, layer="overlay")

    # Center of constellation marker
    sel_center = Circle(
        center=trans_b.center, radius=6.0,
        fill=FillStyle(color=Color.from_hex("#EF4444")),
        stroke=StrokeStyle(color=Color.WHITE, width=2.0)
    )
    scene.add(sel_center, layer="overlay")

    # =========================================================================
    # 5. Layer Ordering & Tag / Metadata Overlay
    # =========================================================================
    add_section_card(810, 520, 740, 440, "LAYER STACKING, TAGS & RECURSIVE QUERIES")

    # Overlapping colored layer discs demonstrating ascending z_order stacking
    disc_bg = Circle(center=Point(1000, 720), radius=90, fill=FillStyle(color=Color.from_hex("#3B82F6"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=3.0), id="disc_bg")
    disc_mid = Circle(center=Point(1090, 720), radius=90, fill=FillStyle(color=Color.from_hex("#8B5CF6"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=3.0), id="disc_mid")
    disc_fg = Circle(center=Point(1180, 720), radius=90, fill=FillStyle(color=Color.from_hex("#EC4899"), opacity=0.8), stroke=StrokeStyle(color=Color.WHITE, width=3.0), id="disc_fg")

    # Add tags and metadata
    disc_bg.add_tag("sensor").set_metadata("priority", 1)
    disc_mid.add_tag("sensor").set_metadata("priority", 2)
    disc_fg.add_tag("actuator").set_metadata("priority", 3)

    scene.add(disc_bg, layer="schematics")
    scene.add(disc_mid, layer="schematics")
    scene.add(disc_fg, layer="schematics")

    # Decorative callout badges in overlay layer
    badge1 = RoundedRectangle(
        x=1320, y=600, width=180, height=45, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#1E293B"), opacity=0.9),
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=1.5)
    )
    badge2 = RoundedRectangle(
        x=1320, y=0, width=180, height=45, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#1E293B"), opacity=0.9),
        stroke=StrokeStyle(color=Color.from_hex("#10B981"), width=1.5)
    )
    place_below(badge2, badge1, gap=15.0)

    badge3 = RoundedRectangle(
        x=1320, y=0, width=180, height=45, corner_radius=8.0,
        fill=FillStyle(color=Color.from_hex("#1E293B"), opacity=0.9),
        stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=1.5)
    )
    place_below(badge3, badge2, gap=15.0)

    scene.add(badge1, layer="overlay")
    scene.add(badge2, layer="overlay")
    scene.add(badge3, layer="overlay")

    # Connect sensors to badges via leader lines in annotations layer
    scene.add(Line(start=disc_bg.anchor("top"), end=badge1.anchor("left"), stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=1.5)), layer="annotations")
    scene.add(Line(start=disc_mid.anchor("top"), end=badge2.anchor("left"), stroke=StrokeStyle(color=Color.from_hex("#10B981"), width=1.5)), layer="annotations")
    scene.add(Line(start=disc_fg.anchor("top"), end=badge3.anchor("left"), stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=1.5)), layer="annotations")

    # Query demonstration
    sensors = scene.find_by_tag("sensor")
    print(f"Total objects in scene: {len(scene)}")
    print(f"Total layers: {[l.name for l in scene.layers]}")
    print(f"Sensor tag query returned: {[s.id for s in sensors]}")

    # =========================================================================
    # 6. Render & Export Canvas
    # =========================================================================
    print("Rendering scene with OpenCVRenderer...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    print(f"Exporting rendered image to {output_file}...")
    canvas.save(str(output_file))
    print("Phase 4 Demo successfully created and exported!")


if __name__ == "__main__":
    main()
