"""Demonstration gallery for DrawCV true vector path boolean operations.

Run: py -3.12 examples/path_boolean_operations.py
"""

from pathlib import Path as FilePath
import cv2
import numpy as np

from drawcv import (
    Color,
    FillStyle,
    OpenCVRenderer,
    Path,
    Point,
    Scene,
    StrokeStyle,
    SVGExporter,
    Text,
)


def make_circle_path(cx: float, cy: float, r: float) -> Path:
    """Create a circular Path approximated with 4 cubic Bezier segments."""
    # Standard cubic bezier circle approximation factor: kappa = 4/3 * (sqrt(2) - 1) ~ 0.5522847
    k = 0.5522847498 * r
    p = Path()
    p.move_to(cx, cy - r)
    p.cubic_to(Point(cx + k, cy - r), Point(cx + r, cy - k), Point(cx + r, cy))
    p.cubic_to(Point(cx + r, cy + k), Point(cx + k, cy + r), Point(cx, cy + r))
    p.cubic_to(Point(cx - k, cy + r), Point(cx - r, cy + k), Point(cx - r, cy))
    p.cubic_to(Point(cx - r, cy - k), Point(cx - k, cy - r), Point(cx, cy - r))
    p.close()
    return p


def make_rounded_rect_path(x: float, y: float, w: float, h: float, r: float) -> Path:
    """Create a rounded rectangle as a semantic vector Path."""
    p = Path()
    p.move_to(x + r, y)
    p.line_to(x + w - r, y)
    p.quadratic_to(Point(x + w, y), Point(x + w, y + r))
    p.line_to(x + w, y + h - r)
    p.quadratic_to(Point(x + w, y + h), Point(x + w - r, y + h))
    p.line_to(x + r, y + h)
    p.quadratic_to(Point(x, y + h), Point(x, y + h - r))
    p.line_to(x, y + r)
    p.quadratic_to(Point(x, y), Point(x + r, y))
    p.close()
    return p


def make_star_path(cx: float, cy: float, r_outer: float, r_inner: float, points: int = 5) -> Path:
    """Create a star polygon Path."""
    p = Path()
    angle_step = np.pi / points
    # Start pointing up
    start_angle = -np.pi / 2
    for i in range(2 * points):
        r = r_outer if i % 2 == 0 else r_inner
        ang = start_angle + i * angle_step
        x = cx + r * np.cos(ang)
        y = cy + r * np.sin(ang)
        if i == 0:
            p.move_to(x, y)
        else:
            p.line_to(x, y)
    p.close()
    return p


def build_gallery_scene() -> Scene:
    scene = Scene(1020, 680, background=Color(20, 24, 32))

    coral = Color(245, 95, 75)
    cyan = Color(50, 185, 230)
    purple = Color(165, 90, 240)
    gold = Color(250, 200, 60)
    white_stroke = StrokeStyle(color=Color(240, 245, 255), width=2.5)

    # Header
    scene.add(Text(
        "DrawCV Vector Path Boolean Operations",
        position=Point(30, 42),
        font_scale=0.75,
        color=Color(255, 255, 255),
    ))
    scene.add(Text(
        "True retained 2D constructive vector geometry powered by Skia PathOps backend",
        position=Point(30, 68),
        font_scale=0.42,
        color=Color(150, 165, 185),
    ))

    # Grid layout: 4 columns in row 1, 3 columns in row 2
    # Row 1: Source Shapes, Union, Intersection, Difference
    # Card 1: Two Source Shapes
    c1_a = make_circle_path(110, 175, 48)
    c1_a.fill = FillStyle(color=coral, opacity=0.85)
    c1_a.stroke = white_stroke
    c1_b = make_circle_path(165, 175, 48)
    c1_b.fill = FillStyle(color=cyan, opacity=0.85)
    c1_b.stroke = white_stroke
    scene.add(c1_a)
    scene.add(c1_b)
    scene.add(Text("1. SOURCE OPERANDS", position=Point(55, 260), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Two overlapping circles", position=Point(55, 278), font_scale=0.35, color=Color(130, 145, 165)))

    # Card 2: Union
    c2_a = make_circle_path(355, 175, 48)
    c2_a.fill = FillStyle(color=coral)
    c2_a.stroke = white_stroke
    c2_b = make_circle_path(410, 175, 48)
    c2_b.fill = FillStyle(color=cyan)
    c2_union = c2_a.union(c2_b)
    scene.add(c2_union)
    scene.add(Text("2. UNION (A | B)", position=Point(315, 260), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Merged boundary contour", position=Point(315, 278), font_scale=0.35, color=Color(130, 145, 165)))

    # Card 3: Intersection
    c3_a = make_circle_path(600, 175, 48)
    c3_a.fill = FillStyle(color=purple)
    c3_a.stroke = white_stroke
    c3_b = make_circle_path(655, 175, 48)
    c3_b.fill = FillStyle(color=cyan)
    c3_inter = c3_a.intersection(c3_b)
    scene.add(c3_inter)
    scene.add(Text("3. INTERSECTION (A & B)", position=Point(550, 260), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Overlapping lens region", position=Point(550, 278), font_scale=0.35, color=Color(130, 145, 165)))

    # Card 4: Difference
    c4_a = make_circle_path(845, 175, 48)
    c4_a.fill = FillStyle(color=coral)
    c4_a.stroke = white_stroke
    c4_b = make_circle_path(900, 175, 48)
    c4_b.fill = FillStyle(color=cyan)
    c4_diff = c4_a.difference(c4_b)
    scene.add(c4_diff)
    scene.add(Text("4. DIFFERENCE (A - B)", position=Point(795, 260), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Crescent cutout (A minus B)", position=Point(795, 278), font_scale=0.35, color=Color(130, 145, 165)))

    # Row 2: XOR, Punched Hole Donut, Bezier Curve Boolean
    # Card 5: XOR
    c5_a = make_circle_path(125, 440, 50)
    c5_a.fill = FillStyle(color=gold)
    c5_a.stroke = white_stroke
    c5_b = make_circle_path(185, 440, 50)
    c5_b.fill = FillStyle(color=cyan)
    c5_xor = c5_a.xor(c5_b)
    scene.add(c5_xor)
    scene.add(Text("5. XOR / EXCLUSIVE-OR", position=Point(75, 535), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Disjoint crescent wings", position=Point(75, 553), font_scale=0.35, color=Color(130, 145, 165)))

    # Card 6: Punched Hole Donut (Compound Vector Hole)
    c6_body = make_rounded_rect_path(420, 375, 130, 130, 24)
    c6_body.fill = FillStyle(color=Color(35, 195, 140))
    c6_body.stroke = white_stroke
    c6_hole = make_star_path(485, 440, 42, 20, points=6)
    c6_punched = c6_body.difference(c6_hole)
    scene.add(c6_punched)
    scene.add(Text("6. PUNCHED VECTOR HOLE", position=Point(400, 535), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("True star hole in rounded box", position=Point(400, 553), font_scale=0.35, color=Color(130, 145, 165)))

    # Card 7: Bezier Curve Boolean
    # Construct a smooth heart/droplet curve
    c7_curve = Path(fill=FillStyle(color=Color(240, 85, 150)), stroke=white_stroke)
    c7_curve.move_to(800, 410)
    c7_curve.cubic_to(Point(800, 360), Point(860, 360), Point(860, 410))
    c7_curve.cubic_to(Point(860, 460), Point(800, 490), Point(800, 510))
    c7_curve.cubic_to(Point(800, 490), Point(740, 460), Point(740, 410))
    c7_curve.cubic_to(Point(740, 360), Point(800, 360), Point(800, 410))
    c7_curve.close()

    # Cutter wave across the bottom
    c7_cutter = make_circle_path(800, 490, 48)
    c7_cutout = c7_curve.difference(c7_cutter)
    scene.add(c7_cutout)
    scene.add(Text("7. BEZIER CURVE CUTOUT", position=Point(725, 535), font_scale=0.42, color=Color(210, 220, 235)))
    scene.add(Text("Preserves smooth cubic curves", position=Point(725, 553), font_scale=0.35, color=Color(130, 145, 165)))

    # Footer
    scene.add(Text(
        "Editable retained paths. Exported natively to SVG vector <path> and JSON 1.4 without raster masks.",
        position=Point(30, 645),
        font_scale=0.38,
        color=Color(140, 155, 175),
    ))

    return scene


def main():
    output_dir = FilePath(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = build_gallery_scene()

    # 1. Render PNG with OpenCVRenderer
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=True)
    png_path = output_dir / "path_boolean_operations.png"
    canvas.save(png_path)
    print(f"Rendered PNG saved to: {png_path}")

    # 2. SVG export of gallery (Text entities use standard reported SVG raster fallbacks)
    exporter = SVGExporter(strict=False)
    svg_res = exporter.render(scene)
    svg_path = output_dir / "path_boolean_operations.svg"
    svg_path.write_text(svg_res.svg, encoding="utf-8")
    print(f"Gallery SVG saved to: {svg_path} (fallbacks: {len(svg_res.fallbacks)})")

    # Export pure vector boolean paths with strict=True to demonstrate 100% native vector SVG
    strict_scene = Scene(1020, 680, background=Color(20, 24, 32))
    for obj in scene.objects:
        if isinstance(obj, Path):
            strict_scene.add(obj.clone())
    strict_res = SVGExporter(strict=True).render(strict_scene)
    strict_path = output_dir / "path_boolean_strict.svg"
    strict_path.write_text(strict_res.svg, encoding="utf-8")
    print(f"Strict vector SVG saved to: {strict_path} (fallbacks: {len(strict_res.fallbacks)})")

    # 3. JSON serialization
    json_path = output_dir / "path_boolean_operations.json"
    scene.save_json(json_path)
    print(f"Scene JSON saved to: {json_path}")

    # 4. Checkerboard background preview for transparency verification
    decoded = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if decoded is not None and decoded.shape[2] == 4:
        alpha = decoded[..., 3:4].astype(float) / 255.0
        h, w = decoded.shape[:2]
        yy, xx = np.indices((h, w))
        checker = np.repeat(np.where(((xx // 20 + yy // 20) % 2)[..., None], 36, 26), 3, axis=2)
        preview = np.rint(decoded[..., :3] * alpha + checker * (1.0 - alpha)).astype(np.uint8)
        preview_path = output_dir / "path_boolean_operations_preview.png"
        cv2.imwrite(str(preview_path), preview)
        print(f"Visual preview saved to: {preview_path}")


if __name__ == "__main__":
    main()
