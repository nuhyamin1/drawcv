"""Phase 8 Demonstration: Temporal Drawing & Animation Engine.

Demonstrates:
1. Progressive path drawing with stroke-first reveal.
2. Synchronized Timeline with property animation tracks (transform, radius, color).
3. Non-destructive temporal evaluation and frame generation.
4. MP4 video encoding via OpenCV VideoWriter.
5. Canonical Schema 1.1 JSON persistence with Timeline tracks.
"""

from pathlib import Path
from drawcv import (
    Arc,
    ArcClosure,
    Circle,
    Color,
    FillStyle,
    Line,
    OpenCVRenderer,
    Point,
    Polyline,
    Scene,
    StrokeStyle,
    Timing,
    Transform,
    VideoRenderer,
)


def run():
    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DrawCV Phase 8 Animation & Temporal Engine Demo ===")

    # 1. Create artistic animated scene
    scene = Scene(800, 600, background=Color(20, 24, 33))

    # Core progressive polygon (star/diamond geometry)
    star_points = [
        Point(400, 150),
        Point(450, 250),
        Point(560, 270),
        Point(480, 350),
        Point(500, 460),
        Point(400, 410),
        Point(300, 460),
        Point(320, 350),
        Point(240, 270),
        Point(350, 250),
    ]
    poly = Polyline(
        points=star_points,
        closed=True,
        stroke=StrokeStyle(color=Color(255, 180, 50), width=4),
    )
    poly.timing = Timing(start_time=0.0, duration=2.0, easing="ease_in_out")
    scene.add(poly)

    # Pulsing center circle
    center_circle = Circle(
        center=Point(400, 310),
        radius=15.0,
        fill=FillStyle(color=Color(100, 200, 255)),
        id="center_orb",
    )
    scene.add(center_circle)

    # Orbiting decorative circle
    satellite = Circle(
        center=Point(400, 100),
        radius=12.0,
        fill=FillStyle(color=Color(255, 80, 120)),
        id="satellite_orb",
    )
    scene.add(satellite)

    # Connecting beam line
    beam = Line(
        start=Point(400, 310),
        end=Point(400, 100),
        stroke=StrokeStyle(color=Color(150, 180, 255, 0.7), width=2),
        id="beam_line",
    )
    scene.add(beam)

    # Decorative sweep arcs
    arc1 = Arc(
        center=Point(400, 310),
        radius_x=120,
        radius_y=120,
        start_angle=0,
        sweep_angle=360,
        closure=ArcClosure.OPEN,
        stroke=StrokeStyle(color=Color(80, 160, 240, 0.5), width=2),
    )
    arc1.timing = Timing(start_time=0.5, duration=1.5, easing="ease_out")
    scene.add(arc1)

    # 2. Timeline tracks
    # Pulse center orb radius from 15 to 40
    scene.animate(center_circle, "radius", 15.0, 40.0, duration=2.0, easing="ease_in_out")
    # Shift center orb color from light blue to cyan
    scene.animate(
        center_circle,
        "fill.color",
        Color(100, 200, 255),
        Color(0, 255, 200),
        duration=2.0,
    )
    # Rotate star geometry 360 degrees
    scene.animate(
        poly,
        "transform",
        Transform(rotation=0.0, pivot=Point(400, 310)),
        Transform(rotation=180.0, pivot=Point(400, 310)),
        duration=2.0,
        easing="ease_in_out",
    )

    print(f"Scene temporal duration: {scene.temporal_duration:.2f} seconds")

    # 3. Non-destructive rendering verification
    orig_radius = center_circle.radius
    print(f"Authored circle radius before rendering: {orig_radius}")

    f_mid = scene.render_at_time(1.0)
    print(f"Authored circle radius after render_at_time(1.0): {center_circle.radius} (preserved!)")
    assert center_circle.radius == orig_radius

    # 4. Save canonical Schema 1.1 JSON
    json_path = out_dir / "phase8_scene.json"
    scene.save_json(json_path)
    print(f"Saved Schema 1.1 JSON to: {json_path}")

    # Verify JSON reload
    reloaded_scene = Scene.load_json(json_path)
    assert len(reloaded_scene.timeline.tracks) == 3
    print("Schema 1.1 JSON round-trip verified successfully!")

    # 5. Render MP4 video
    video_path = out_dir / "phase8_animation.mp4"
    print("Rendering MP4 video via OpenCV VideoWriter...")
    VideoRenderer.render_video(
        scene,
        output_path=video_path,
        duration=2.0,
        fps=30,
        fourcc="mp4v",
    )
    print(f"Generated video: {video_path} ({video_path.stat().st_size:,} bytes, 60 frames)")

    # 6. Render image sequence
    seq_dir = out_dir / "phase8_frames"
    paths = VideoRenderer.render_image_sequence(
        scene,
        output_dir=seq_dir,
        pattern="frame_%03d.png",
        duration=2.0,
        fps=10,
    )
    print(f"Rendered image sequence ({len(paths)} frames) into: {seq_dir}")

    print("=== Phase 8 Demo Completed Successfully! ===")


if __name__ == "__main__":
    run()
