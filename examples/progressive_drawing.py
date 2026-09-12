"""Demo: Progressive Drawing & Arc-Length Reveal (Phase 8).

Demonstrates progressive path drawing across Line, Polyline, Arrow, BezierCurve,
Path, FreehandStroke, and Arc primitives. Exports visual milestone frames.
"""

from pathlib import Path
from drawcv import (
    Arc,
    ArcClosure,
    Arrow,
    ArrowHeadStyle,
    BezierCurve,
    Canvas,
    Color,
    FillStyle,
    FreehandStroke,
    Line,
    OpenCVRenderer,
    Path as DrawPath,
    Point,
    Polyline,
    Scene,
    StrokePoint,
    StrokeStyle,
    Timing,
)


def run():
    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = Scene(900, 600, background=Color(245, 247, 250))

    # 1. Line
    line = Line(
        start=Point(50, 80),
        end=Point(250, 80),
        stroke=StrokeStyle(color=Color(40, 60, 100), width=4),
    )
    line.timing = Timing(start_time=0.0, duration=2.0, easing="ease_in_out")
    scene.add(line)

    # 2. Polyline
    poly = Polyline(
        points=[Point(350, 100), Point(420, 40), Point(480, 120), Point(550, 60)],
        stroke=StrokeStyle(color=Color(20, 140, 180), width=4),
    )
    poly.timing = Timing(start_time=0.0, duration=2.0, easing="ease_out")
    scene.add(poly)

    # 3. Arrow
    arrow = Arrow(
        start=Point(650, 80),
        end=Point(850, 80),
        head_length=20,
        head_width=16,
        head_style=ArrowHeadStyle.TRIANGLE,
        stroke=StrokeStyle(color=Color(220, 60, 60), width=4),
        fill=FillStyle(color=Color(220, 60, 60)),
    )
    arrow.timing = Timing(start_time=0.0, duration=2.0, easing="ease_in_out")
    scene.add(arrow)

    # 4. BezierCurve
    bezier = BezierCurve(
        p0=Point(50, 280),
        p1=Point(100, 180),
        p2=Point(200, 360),
        p3=Point(260, 240),
        stroke=StrokeStyle(color=Color(140, 60, 180), width=4),
    )
    bezier.timing = Timing(start_time=0.0, duration=2.0, easing="ease_in_out")
    scene.add(bezier)

    # 5. Compound Vector Path (with fill revealed at completion)
    path = DrawPath(
        stroke=StrokeStyle(color=Color(40, 160, 80), width=4),
        fill=FillStyle(color=Color(180, 230, 200, 0.6)),
    )
    path.move_to(350, 280).line_to(450, 180).line_to(550, 280).close()
    path.timing = Timing(start_time=0.0, duration=2.0, easing="ease_in_out")
    scene.add(path)

    # 6. Arc (Elliptical sweep)
    arc = Arc(
        center=Point(750, 260),
        radius_x=60,
        radius_y=50,
        start_angle=-90,
        sweep_angle=270,
        closure=ArcClosure.PIE,
        stroke=StrokeStyle(color=Color(240, 140, 20), width=4),
        fill=FillStyle(color=Color(255, 210, 130, 0.7)),
    )
    arc.timing = Timing(start_time=0.0, duration=2.0, easing="ease_out")
    scene.add(arc)

    # 7. Freehand Stroke
    pts = [
        StrokePoint(100, 480, pressure=0.2),
        StrokePoint(250, 420, pressure=0.5),
        StrokePoint(400, 520, pressure=0.8),
        StrokePoint(600, 440, pressure=0.6),
        StrokePoint(800, 500, pressure=0.3),
    ]
    freehand = FreehandStroke(
        points=pts,
        smoothing="chaikin",
        stroke=StrokeStyle(color=Color(60, 80, 120), width=5),
    )
    freehand.timing = Timing(start_time=0.0, duration=2.0, easing="linear")
    scene.add(freehand)

    # Export milestone progression frames: 25%, 50%, 75%, 100%
    milestones = [
        ("progressive_025.png", 0.5),
        ("progressive_050.png", 1.0),
        ("progressive_075.png", 1.5),
        ("progressive_100.png", 2.0),
    ]

    for filename, t in milestones:
        frame = scene.render_at_time(t)
        p = out_dir / filename
        frame.save(p)
        print(f"Saved milestone frame: {p} (t={t}s)")


if __name__ == "__main__":
    run()
