"""Phase 3 Visual Demonstration Script for DrawCV — Advanced Geometry Showcase.

Demonstrates:
- Ellipse: analytical extrema under affine rotation & non-uniform scaling, parametric anchors
- Polygon: non-degenerate Shoelace validation, centroid anchor, vertex markers
- Polyline: cumulative length, midpoint interpolation anchor
- RoundedRectangle: smooth corner arcs and analytical boundary hit-testing
- Arc: explicit start/sweep angle semantics with OPEN, CHORD, and PIE closures
- Arrow: screen-space arrowhead geometry (TRIANGLE, OPEN, DIAMOND, CIRCLE) invariant under shaft scaling
- BezierCurve: Quadratic and Cubic curves with adaptive de Casteljau world-space subdivision and analytical bounds
- Path: retained semantic vector commands (MoveTo, LineTo, QuadraticTo, CubicTo, Close)
- FillRule: topological interior evaluation (EVEN_ODD vs NON_ZERO) with 2x supersampling anti-aliasing
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
    BezierCurve,
    Circle,
    Color,
    Ellipse,
    FillRule,
    FillStyle,
    Line,
    OpenCVRenderer,
    Path as VectorPath,
    Point,
    Polygon,
    Polyline,
    Rectangle,
    RoundedRectangle,
    Scene,
    StrokeStyle,
)


def main():
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "phase3_demo.png"

    print("Creating DrawCV Phase 3 scene (1600x1000)...")
    # Modern dark canvas background (Slate 950: #0B0F19)
    scene = Scene(width=1600, height=1000, background=Color.from_hex("#0B0F19"))

    # Helper: section title text card
    def add_card(title: str, x: float, y: float, w: float, h: float):
        # Translucent glassmorphism container
        card = RoundedRectangle(
            x=x, y=y, width=w, height=h, corner_radius=12.0,
            fill=FillStyle(color=Color.from_hex("#161E2E"), opacity=0.7),
            stroke=StrokeStyle(color=Color.from_hex("#334155"), width=1.5)
        )
        scene.add(card)

    # =========================================================================
    # Section 1: Ellipses & Transformed Parametric Anchors
    # =========================================================================
    add_card("ELLIPSES & PARAMETRIC ANCHORS", 50, 40, 460, 280)

    # Rotated & non-uniformly scaled ellipse
    ellipse = Ellipse(
        center=Point(200, 180),
        radius_x=90.0,
        radius_y=45.0,
        fill=FillStyle(color=Color.from_hex("#38BDF8"), opacity=0.35),
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=2.5)
    )
    ellipse.rotate(35.0)
    scene.add(ellipse)

    # Show analytical world bounding box in dashed amber
    wb = ellipse.get_bounds()
    bbox_vis = Rectangle(
        position=wb.top_left,
        width=wb.width,
        height=wb.height,
        stroke=StrokeStyle(color=Color.from_hex("#F59E0B"), width=1.0)
    )
    scene.add(bbox_vis)

    # Parametric angle anchors at 0, 45, 90, 135, 180, 225, 270, 315 deg
    for ang in range(0, 360, 45):
        pt = ellipse.anchor_at_angle(ang)
        scene.add(Circle(
            center=pt, radius=4.0,
            fill=FillStyle(color=Color.from_hex("#F43F5E")),
            stroke=StrokeStyle(color=Color.WHITE, width=1.5)
        ))

    # Center anchor
    scene.add(Circle(
        center=ellipse.anchor("center"), radius=5.0,
        fill=FillStyle(color=Color.from_hex("#10B981")),
        stroke=StrokeStyle(color=Color.WHITE, width=1.5)
    ))

    # Second concentric rotated ellipse
    inner_ellipse = Ellipse(
        center=Point(380, 180),
        radius_x=50.0,
        radius_y=25.0,
        fill=FillStyle(color=Color.from_hex("#A855F7"), opacity=0.5),
        stroke=StrokeStyle(color=Color.from_hex("#C084FC"), width=2.0)
    )
    inner_ellipse.rotate(-45.0)
    scene.add(inner_ellipse)

    # =========================================================================
    # Section 2: Polygons & Polyline Metrics
    # =========================================================================
    add_card("POLYGONS, CENTROIDS & POLYLINES", 560, 40, 480, 280)

    # Non-trivial convex polygon (Pentagon)
    poly_pts = [
        Point(700, 100), Point(780, 130), Point(760, 220),
        Point(640, 220), Point(620, 130)
    ]
    poly = Polygon(
        vertices=poly_pts,
        fill=FillStyle(color=Color.from_hex("#10B981"), opacity=0.4),
        stroke=StrokeStyle(color=Color.from_hex("#34D399"), width=2.5)
    )
    poly.rotate(15.0)
    scene.add(poly)

    # Polygon Centroid anchor
    c_pt = poly.anchor("centroid")
    scene.add(Circle(
        center=c_pt, radius=6.0,
        fill=FillStyle(color=Color.from_hex("#F59E0B")),
        stroke=StrokeStyle(color=Color.WHITE, width=2.0)
    ))

    # Polyline with midpoint anchor
    pline_pts = [
        Point(840, 100), Point(890, 140), Point(950, 110),
        Point(920, 210), Point(990, 240)
    ]
    pline = Polyline(
        points=pline_pts, closed=False,
        stroke=StrokeStyle(color=Color.from_hex("#F97316"), width=3.5)
    )
    scene.add(pline)

    # Polyline midpoint anchor (at 50% cumulative length)
    mid_pt = pline.anchor("midpoint")
    scene.add(Circle(
        center=mid_pt, radius=6.0,
        fill=FillStyle(color=Color.from_hex("#38BDF8")),
        stroke=StrokeStyle(color=Color.WHITE, width=2.0)
    ))

    # =========================================================================
    # Section 3: RoundedRectangles with Corner Geometry
    # =========================================================================
    add_card("ROUNDED RECTANGLES", 1090, 40, 460, 280)

    # Nested rounded cards
    for idx, (rad, color_hex) in enumerate([(10, "#3B82F6"), (25, "#8B5CF6"), (40, "#EC4899")]):
        rrect = RoundedRectangle(
            x=1130 + idx * 40,
            y=80 + idx * 30,
            width=260 - idx * 20,
            height=160 - idx * 20,
            corner_radius=rad,
            fill=FillStyle(color=Color.from_hex(color_hex), opacity=0.3),
            stroke=StrokeStyle(color=Color.from_hex(color_hex), width=2.5)
        )
        scene.add(rrect)

    # =========================================================================
    # Section 4: Arcs & Sweep Closures (OPEN, CHORD, PIE)
    # =========================================================================
    add_card("ARCS (OPEN, CHORD, PIE & SIGNED SWEEPS)", 50, 360, 460, 280)

    # Arc 1: OPEN curve stroke (positive sweep)
    arc_open = Arc(
        center=Point(130, 500),
        radius_x=60.0, radius_y=60.0,
        start_angle=30.0, sweep_angle=150.0,
        closure=ArcClosure.OPEN,
        stroke=StrokeStyle(color=Color.from_hex("#38BDF8"), width=4.0)
    )
    scene.add(arc_open)

    # Arc 2: CHORD closure
    arc_chord = Arc(
        center=Point(270, 500),
        radius_x=60.0, radius_y=45.0,
        start_angle=200.0, sweep_angle=120.0,
        closure=ArcClosure.CHORD,
        fill=FillStyle(color=Color.from_hex("#EC4899"), opacity=0.4),
        stroke=StrokeStyle(color=Color.from_hex("#F472B6"), width=2.5)
    )
    scene.add(arc_chord)

    # Arc 3: PIE sector (negative counter-clockwise sweep)
    arc_pie = Arc(
        center=Point(410, 500),
        radius_x=55.0, radius_y=55.0,
        start_angle=90.0, sweep_angle=-110.0,
        closure=ArcClosure.PIE,
        fill=FillStyle(color=Color.from_hex("#F59E0B"), opacity=0.5),
        stroke=StrokeStyle(color=Color.from_hex("#FCD34D"), width=2.5)
    )
    scene.add(arc_pie)

    # =========================================================================
    # Section 5: Arrows & Screen-Space Head Geometry Invariance
    # =========================================================================
    add_card("ARROWS (SCREEN-SPACE HEADS UNDER SCALE)", 560, 360, 480, 280)

    head_styles = [
        (ArrowHeadStyle.TRIANGLE, "#38BDF8", 430),
        (ArrowHeadStyle.DIAMOND,  "#34D399", 480),
        (ArrowHeadStyle.CIRCLE,   "#F43F5E", 530),
        (ArrowHeadStyle.OPEN,     "#FBBF24", 580),
    ]

    for style, hex_col, y_pos in head_styles:
        arrow = Arrow(
            start=Point(600, y_pos),
            end=Point(780, y_pos),
            head_length=18.0,
            head_width=12.0,
            head_style=style,
            stroke=StrokeStyle(color=Color.from_hex(hex_col), width=3.0),
            fill=FillStyle(color=Color.from_hex(hex_col))
        )
        scene.add(arrow)

    # Dramatically scaled arrow: 2.5x along shaft, proving head dimensions remain screen-space!
    scaled_arrow = Arrow(
        start=Point(840, 500),
        end=Point(900, 440),
        head_length=22.0,
        head_width=16.0,
        head_style=ArrowHeadStyle.TRIANGLE,
        stroke=StrokeStyle(color=Color.from_hex("#A855F7"), width=3.5),
        fill=FillStyle(color=Color.from_hex("#C084FC"))
    )
    scaled_arrow.scale(2.2, 0.6, pivot=Point(840, 500))
    scene.add(scaled_arrow)

    # =========================================================================
    # Section 6: Bézier Curves & Adaptive Subdivision
    # =========================================================================
    add_card("BÉZIER GEOMETRY (QUADRATIC & CUBIC)", 1090, 360, 460, 280)

    # Quadratic Bézier
    quad = BezierCurve.quadratic(
        p0=Point(1130, 540),
        p1=Point(1200, 420),
        p2=Point(1270, 540),
        stroke=StrokeStyle(color=Color.from_hex("#06B6D4"), width=3.5)
    )
    scene.add(quad)

    # Tangent / control lines for quadratic
    scene.add(Line(
        start=Point(1130, 540), end=Point(1200, 420),
        stroke=StrokeStyle(color=Color.from_hex("#475569"), width=1.5)
    ))
    scene.add(Line(
        start=Point(1200, 420), end=Point(1270, 540),
        stroke=StrokeStyle(color=Color.from_hex("#475569"), width=1.5)
    ))
    scene.add(Circle(
        center=Point(1200, 420), radius=5.0,
        fill=FillStyle(color=Color.from_hex("#06B6D4")),
        stroke=StrokeStyle(color=Color.WHITE, width=1.5)
    ))

    # Cubic Bézier (S-curve)
    cub = BezierCurve.cubic(
        p0=Point(1310, 550),
        p1=Point(1350, 410),
        p2=Point(1450, 610),
        p3=Point(1500, 440),
        stroke=StrokeStyle(color=Color.from_hex("#F43F5E"), width=4.0)
    )
    scene.add(cub)

    # Control handles for cubic
    scene.add(Line(start=Point(1310, 550), end=Point(1350, 410), stroke=StrokeStyle(color=Color.from_hex("#475569"), width=1.2)))
    scene.add(Line(start=Point(1500, 440), end=Point(1450, 610), stroke=StrokeStyle(color=Color.from_hex("#475569"), width=1.2)))
    scene.add(Circle(center=Point(1350, 410), radius=4.5, fill=FillStyle(color=Color.from_hex("#F43F5E")), stroke=StrokeStyle(color=Color.WHITE, width=1.5)))
    scene.add(Circle(center=Point(1450, 610), radius=4.5, fill=FillStyle(color=Color.from_hex("#F43F5E")), stroke=StrokeStyle(color=Color.WHITE, width=1.5)))

    # =========================================================================
    # Section 7: Retained Paths & Topological Fill Rules (EVEN_ODD vs NON_ZERO)
    # =========================================================================
    add_card("RETAINED PATHS & TOPOLOGICAL FILL RULES (EVEN_ODD vs NON_ZERO)", 50, 680, 1500, 280)

    # Path A: EVEN_ODD Donut Star
    # Outer 8-point star with inner circular hole
    path_eo = VectorPath(
        fill_rule=FillRule.EVEN_ODD,
        fill=FillStyle(color=Color.from_hex("#6366F1"), opacity=0.85),
        stroke=StrokeStyle(color=Color.from_hex("#A5B4FC"), width=2.0)
    )
    # Outer star
    center_a = Point(300, 820)
    path_eo.move_to(center_a.x + 80, center_a.y)
    for i in range(1, 16):
        ang = i * (math.pi / 8.0)
        r = 80 if i % 2 == 0 else 45
        path_eo.line_to(center_a.x + r * math.cos(ang), center_a.y + r * math.sin(ang))
    path_eo.close()

    # Inner circular hole subpath
    path_eo.move_to(center_a.x + 25, center_a.y)
    for i in range(1, 24):
        ang = i * (2.0 * math.pi / 24.0)
        path_eo.line_to(center_a.x + 25 * math.cos(ang), center_a.y + 25 * math.sin(ang))
    path_eo.close()
    scene.add(path_eo)

    # Path B: Multi-contour Bézier clover leaf (Retained commands)
    clover = VectorPath(
        fill_rule=FillRule.NON_ZERO,
        fill=FillStyle(color=Color.from_hex("#10B981"), opacity=0.75),
        stroke=StrokeStyle(color=Color.from_hex("#6EE7B7"), width=2.5)
    )
    # Start at center (800, 820)
    cx, cy = 800.0, 820.0
    clover.move_to(cx, cy)
    # 4 cubic petals
    clover.cubic_to(Point(cx - 70, cy - 30), Point(cx - 70, cy - 90), Point(cx, cy - 70))
    clover.cubic_to(Point(cx + 70, cy - 90), Point(cx + 70, cy - 30), Point(cx, cy))
    clover.cubic_to(Point(cx + 90, cy + 30), Point(cx + 30, cy + 90), Point(cx, cy + 70))
    clover.cubic_to(Point(cx - 30, cy + 90), Point(cx - 90, cy + 30), Point(cx, cy))
    clover.close()
    scene.add(clover)

    # Path C: NON_ZERO Concentric Winding Rings
    # Outer ring CW (+1), Middle ring CCW (-1) => Hole, Inner ring CW (+1) => Solid center
    rings = VectorPath(
        fill_rule=FillRule.NON_ZERO,
        fill=FillStyle(color=Color.from_hex("#EC4899"), opacity=0.8),
        stroke=StrokeStyle(color=Color.from_hex("#F472B6"), width=2.0)
    )
    rx, ry = 1300.0, 820.0
    # Outer CW (radius 80)
    rings.move_to(rx + 80, ry)
    for i in range(1, 32):
        ang = i * (2.0 * math.pi / 32.0)
        rings.line_to(rx + 80 * math.cos(ang), ry + 80 * math.sin(ang))
    rings.close()

    # Middle CCW (radius 55, negative angle step: -1 winding)
    rings.move_to(rx + 55, ry)
    for i in range(1, 32):
        ang = -i * (2.0 * math.pi / 32.0)
        rings.line_to(rx + 55 * math.cos(ang), ry + 55 * math.sin(ang))
    rings.close()

    # Inner CW (radius 30: +1 winding -> center filled)
    rings.move_to(rx + 30, ry)
    for i in range(1, 32):
        ang = i * (2.0 * math.pi / 32.0)
        rings.line_to(rx + 30 * math.cos(ang), ry + 30 * math.sin(ang))
    rings.close()
    scene.add(rings)

    # Render scene
    print("Rendering scene with OpenCVRenderer...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    print(f"Saving visual demo to {output_file}...")
    canvas.save(str(output_file))
    print("Phase 3 visual demonstration successfully rendered and saved!")


if __name__ == "__main__":
    main()
