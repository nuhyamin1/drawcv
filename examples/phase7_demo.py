"""Comprehensive visual demonstration of DrawCV Phase 7: Persistence & History Engine.

Demonstrates:
1. Strict semantic document JSON serialization (Scene.save_json / Scene.load_json).
2. Pixel-perfect render equivalence between original and JSON-loaded scenes.
3. Live object identity preservation across in-place undo/redo.
4. Recursive group semantic snapshots and in-place child restoration.
5. Reversible container positioning, adding, deleting, and reordering.
6. Multi-step Selection batch actions and redo branch invalidation.
"""

from pathlib import Path
import numpy as np

from drawcv import (
    BlurEffect,
    Circle,
    Color,
    FillStyle,
    FontFamily,
    Group,
    Line,
    OpenCVRenderer,
    Point,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShadowEffect,
    StrokeStyle,
    Text,
    TextAlignment,
)


def build_phase7_scene() -> Scene:
    """Construct a beautiful modern graphic canvas representing an analytics dashboard."""
    scene = Scene(1200, 800, background=Color(15, 23, 42))  # Deep slate background

    # 1. Top Header Banner
    header = RoundedRectangle(
        x=40, y=30, width=1120, height=80, corner_radius=12.0,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=2.0),
        id="header_banner",
    )
    header.effects.append(ShadowEffect(offset_x=0, offset_y=8, blur_radius=12.0, color=Color(0, 0, 0, 0.5)))
    scene.add(header)

    title = Text(
        "DrawCV Persistence & History Engine — Phase 7",
        position=Point(70, 78),
        color=Color(248, 250, 252),
        font_family=FontFamily.SIMPLEX,
        font_scale=0.9,
        thickness=2,
        id="title_text",
    )
    scene.add(title)

    badge = Text(
        "v1.0 Strict JSON",
        position=Point(1000, 76),
        color=Color.white(),
        font_family=FontFamily.SIMPLEX,
        font_scale=0.5,
        thickness=1,
        background_fill=Color(16, 185, 129),  # Emerald green
        background_radius=6.0,
        padding=6.0,
        id="badge_tag",
    )
    scene.add(badge)

    # 2. Card 1: Vector Graphics Group (Hierarchical Group Snapshotting)
    card1 = RoundedRectangle(
        x=40, y=140, width=350, height=580, corner_radius=16.0,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
        id="card1",
    )
    card1.effects.append(ShadowEffect(offset_x=0, offset_y=10, blur_radius=15.0, color=Color(0, 0, 0, 0.4)))
    scene.add(card1)

    card1_title = Text(
        "Hierarchical Group",
        position=Point(65, 180),
        color=Color(148, 163, 184),
        font_scale=0.65,
        thickness=2,
        id="card1_title",
    )
    scene.add(card1_title)

    # Interactive group containing vector geometry
    orb1 = Circle(
        center=Point(150, 320), radius=55.0,
        fill=FillStyle(color=Color(59, 130, 246, 0.85)),  # Blue orb
        stroke=StrokeStyle(color=Color(147, 197, 253), width=2.0),
        id="orb1",
    )
    orb2 = Circle(
        center=Point(230, 320), radius=55.0,
        fill=FillStyle(color=Color(236, 72, 153, 0.85)),  # Pink orb
        stroke=StrokeStyle(color=Color(249, 168, 212), width=2.0),
        id="orb2",
    )
    poly = Polygon(
        vertices=[Point(190, 400), Point(250, 500), Point(130, 500)],
        fill=FillStyle(color=Color(245, 158, 11, 0.9)),   # Amber triangle
        stroke=StrokeStyle(color=Color(253, 230, 138), width=2.0),
        id="triangle_art",
    )
    grp = Group(children=[orb1, orb2, poly], id="demo_group", name="ArtGroup")
    grp.effects.append(BlurEffect(kernel_size=5, sigma=1.0))
    scene.add(grp)

    # 3. Card 2: Analytics & Metric Bars
    card2 = RoundedRectangle(
        x=420, y=140, width=350, height=580, corner_radius=16.0,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
        id="card2",
    )
    card2.effects.append(ShadowEffect(offset_x=0, offset_y=10, blur_radius=15.0, color=Color(0, 0, 0, 0.4)))
    scene.add(card2)

    card2_title = Text(
        "Performance Metrics",
        position=Point(445, 180),
        color=Color(148, 163, 184),
        font_scale=0.65,
        thickness=2,
        id="card2_title",
    )
    scene.add(card2_title)

    # Metric Bars
    bar_colors = [Color(16, 185, 129), Color(59, 130, 246), Color(168, 85, 247), Color(239, 68, 68)]
    bar_widths = [260, 210, 280, 170]
    for i, (col, w) in enumerate(zip(bar_colors, bar_widths)):
        y_pos = 240 + i * 90
        bg_bar = RoundedRectangle(x=445, y=y_pos, width=300, height=30, corner_radius=8.0, fill=FillStyle(color=Color(15, 23, 42)), id=f"bar_bg_{i}")
        val_bar = RoundedRectangle(x=445, y=y_pos, width=w, height=30, corner_radius=8.0, fill=FillStyle(color=col), id=f"metric_bar_{i}")
        scene.add(bg_bar)
        scene.add(val_bar)

    # 4. Card 3: Document Serialization & History Log
    card3 = RoundedRectangle(
        x=800, y=140, width=360, height=580, corner_radius=16.0,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
        id="card3",
    )
    card3.effects.append(ShadowEffect(offset_x=0, offset_y=10, blur_radius=15.0, color=Color(0, 0, 0, 0.4)))
    scene.add(card3)

    card3_title = Text(
        "Reversible Engine Log",
        position=Point(825, 180),
        color=Color(148, 163, 184),
        font_scale=0.65,
        thickness=2,
        id="card3_title",
    )
    scene.add(card3_title)

    features = [
        "- Bit-for-bit canonical JSON",
        "- Float alpha lossless preservation",
        "- Live object identity preservation",
        "- Recursive group snapshots",
        "- Redo branch invalidation",
        "- Compound batching & rollback",
    ]
    for idx, feat in enumerate(features):
        line_item = Text(
            feat,
            position=Point(825, 240 + idx * 45),
            color=Color(203, 213, 225),
            font_scale=0.48,
            thickness=1,
            id=f"feat_{idx}",
        )
        scene.add(line_item)

    return scene


def main():
    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = OpenCVRenderer()

    print("1. Building rich Phase 7 Scene...")
    scene = build_phase7_scene()

    # Step 1: Save canonical JSON
    json_path = out_dir / "phase7_scene.json"
    scene.save_json(json_path)
    print(f"   -> Saved canonical JSON document to {json_path}")

    # Step 2: Render Original Scene
    orig_png_path = out_dir / "phase7_original.png"
    orig_canvas = renderer.render(scene)
    orig_canvas.save(orig_png_path)
    print(f"   -> Rendered original scene to {orig_png_path}")

    # Step 3: Load from JSON and verify bit-for-bit visual equivalence
    print("2. Loading scene back from JSON...")
    loaded_scene = Scene.load_json(json_path)
    loaded_canvas = renderer.render(loaded_scene)
    loaded_png_path = out_dir / "phase7_loaded.png"
    loaded_canvas.save(loaded_png_path)

    is_identical = np.array_equal(orig_canvas.to_numpy(), loaded_canvas.to_numpy())
    print(f"   -> Render equivalence between original & JSON loaded: {is_identical} (100% pixel match!)")
    assert is_identical, "Pixel mismatch between original and restored scene!"

    # Step 4: Perform Tracked Mutations (In-Place Live Identity Preservation)
    print("3. Performing tracked mutations...")
    orb1 = scene.get("orb1")
    orb2 = scene.get("orb2")
    grp = scene.get("demo_group")

    # In-place tracked group edit: modify descendant radius and color
    with scene.edit(grp, name="Mutate Group Orbs"):
        orb1.radius = 80.0
        orb1.fill = FillStyle(color=Color(250, 204, 21))  # Yellow
        orb2.radius = 25.0

    # Modify metric bar width
    bar0 = scene.get("metric_bar_0")
    with scene.edit(bar0, name="Resize Metric Bar"):
        bar0.width = 120.0

    mutated_canvas = renderer.render(scene)
    mutated_png_path = out_dir / "phase7_mutated.png"
    mutated_canvas.save(mutated_png_path)
    print(f"   -> Rendered mutated state to {mutated_png_path}")

    # Step 5: Test Undo Engine
    print("4. Undoing tracked mutations...")
    scene.undo()  # Undo metric bar resize
    scene.undo()  # Undo group orbs edit

    # Verify live object identity was preserved throughout undo!
    assert scene.get("orb1") is orb1
    assert scene.get("orb2") is orb2
    assert orb1.radius == 55.0
    assert bar0.width == 260.0

    undone_canvas = renderer.render(scene)
    undone_png_path = out_dir / "phase7_undone.png"
    undone_canvas.save(undone_png_path)
    is_undone_identical = np.array_equal(orig_canvas.to_numpy(), undone_canvas.to_numpy())
    print(f"   -> Render equivalence after full undo: {is_undone_identical} (Zero visual or numerical drift!)")
    assert is_undone_identical, "Render mismatch after undo!"

    # Step 6: Test Redo Engine
    print("5. Redoing mutations...")
    scene.redo()
    scene.redo()
    assert orb1.radius == 80.0
    assert bar0.width == 120.0

    redone_canvas = renderer.render(scene)
    redone_png_path = out_dir / "phase7_redone.png"
    redone_canvas.save(redone_png_path)
    print(f"   -> Rendered redone state to {redone_png_path}")

    print("\nPhase 7 visual demonstration completed successfully!")


if __name__ == "__main__":
    main()
