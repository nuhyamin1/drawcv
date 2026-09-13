"""Comprehensive tests for Phase 8 Timeline, non-destructive sampling, schema 1.1, and video rendering."""

import math
from pathlib import Path
import numpy as np
import pytest

from drawcv.animation.timeline import Timeline
from drawcv.animation.timing import Timing
from drawcv.animation.track import AnimationTrack
from drawcv.animation.video_renderer import VideoRenderer
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.group import Group
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def test_non_destructive_render_at_time():
    scene = Scene(400, 400)
    circle = Circle(center=Point(100, 100), radius=50.0, fill=FillStyle(color=Color.red()))
    scene.add(circle)

    # Authored initial state
    orig_radius = circle.radius
    orig_color = circle.fill.color
    orig_opacity = circle.opacity
    orig_transform = circle.transform.copy()
    orig_prog = circle.render_progress

    # Animate radius, fill color, opacity, and transform
    scene.animate(circle, "radius", 50.0, 150.0, duration=10.0)
    scene.animate(circle, "fill.color", Color.red(), Color.blue(), duration=10.0)
    scene.animate(circle, "opacity", 1.0, 0.2, duration=10.0)
    scene.animate(circle, "transform", Transform(rotation=0.0), Transform(rotation=180.0), duration=10.0)

    # Add line with timing
    line = Line(start=Point(0, 0), end=Point(100, 100))
    line.timing = Timing(duration=10.0)
    scene.add(line)
    orig_line_prog = line.render_progress

    # Sample at t=8, t=2, t=8
    f8_1 = scene.render_at_time(8.0)
    f2 = scene.render_at_time(2.0)
    f8_2 = scene.render_at_time(8.0)

    # Invariant: identical outputs for identical timestamps
    assert np.array_equal(f8_1.buffer, f8_2.buffer)
    assert not np.array_equal(f8_1.buffer, f2.buffer)

    # Invariant: authored state preserved in-place with exact object identity
    assert circle.radius == orig_radius
    assert circle.fill.color == orig_color
    assert circle.opacity == orig_opacity
    assert circle.transform == orig_transform
    assert circle.render_progress == orig_prog
    assert line.render_progress == orig_line_prog


def test_history_neutrality_under_temporal_rendering():
    scene = Scene(200, 200)
    circle = Circle(center=Point(50, 50), radius=20.0)
    scene.add(circle)
    assert scene.history.can_undo is True
    undo_stack_len = len(scene.history._undo_stack)

    scene.animate(circle, "radius", 20.0, 80.0, duration=2.0)
    # Rendering frames or sampling must not add history commands
    scene.render_at_time(1.0)
    frames = list(VideoRenderer.render_frames(scene, duration=1.0, fps=10))
    assert len(frames) == 10
    assert len(scene.history._undo_stack) == undo_stack_len


def test_drawable_timing_automatically_drives_render_progress():
    scene = Scene(200, 200)
    line = Line(start=Point(0, 0), end=Point(100, 0))
    line.timing = Timing(start_time=2.0, duration=4.0)
    scene.add(line)

    # At t=1.0 (before start): progress is 0.0
    scene.render_at_time(1.0)
    # Authored progress remains 1.0
    assert line.render_progress == 1.0

    # Test explicit sample in-place
    scene.sample(4.0)  # midpoint: t=2 + 2 = 4 -> 50%
    assert pytest.approx(line.render_progress) == 0.5


def test_schema_1_0_to_1_1_migration():
    # Canonical Phase 7 JSON string (version 1.0)
    legacy_json = """
    {
      "format": "drawcv",
      "version": "1.0",
      "scene": {
        "width": 300,
        "height": 300,
        "background": {"r": 255, "g": 255, "b": 255, "a": 1.0},
        "layers": [
          {
            "name": "default",
            "visible": true,
            "locked": false,
            "opacity": 1.0,
            "z_order": 0,
            "clip": null,
            "mask": null,
            "effects": [],
            "objects": [
              {
                "type": "circle",
                "id": "c1",
                "name": "RootCircle",
                "visible": true,
                "locked": false,
                "opacity": 1.0,
                "z_index": 0,
                "tags": [],
                "metadata": {},
                "transform": {"translation_x": 0.0, "translation_y": 0.0, "rotation": 0.0, "scale_x": 1.0, "scale_y": 1.0, "pivot": null},
                "clip": null,
                "mask": null,
                "effects": [],
                "center": {"x": 50.0, "y": 50.0},
                "radius": 25.0,
                "stroke": null,
                "fill": null
              },
              {
                "type": "group",
                "id": "g1",
                "name": "NestedGroup",
                "visible": true,
                "locked": false,
                "opacity": 1.0,
                "z_index": 0,
                "tags": [],
                "metadata": {},
                "transform": {"translation_x": 0.0, "translation_y": 0.0, "rotation": 0.0, "scale_x": 1.0, "scale_y": 1.0, "pivot": null},
                "clip": null,
                "mask": null,
                "effects": [],
                "children": [
                  {
                    "type": "line",
                    "id": "l1",
                    "name": "ChildLine",
                    "visible": true,
                    "locked": false,
                    "opacity": 1.0,
                    "z_index": 0,
                    "tags": [],
                    "metadata": {},
                    "transform": {"translation_x": 0.0, "translation_y": 0.0, "rotation": 0.0, "scale_x": 1.0, "scale_y": 1.0, "pivot": null},
                    "clip": null,
                    "mask": null,
                    "effects": [],
                    "start": {"x": 0.0, "y": 0.0},
                    "end": {"x": 20.0, "y": 20.0},
                    "stroke": null
                  }
                ]
              }
            ]
          }
        ]
      }
    }
    """
    # Load and migrate
    scene = Scene.from_json(legacy_json)
    root_circle = scene.get("c1")
    child_line = scene.get("l1")

    assert root_circle is not None
    assert root_circle.timing is None
    assert root_circle.render_progress == 1.0

    assert child_line is not None
    assert child_line.timing is None
    assert child_line.render_progress == 1.0

    assert isinstance(scene.timeline, Timeline)

    # Re-serialization writes the current stroke-aware schema.
    doc = scene.to_dict()
    assert doc["version"] == "1.2"


def test_typed_animation_tracks_round_trip():
    scene = Scene(300, 300)
    circle = Circle(center=Point(100, 100), radius=30.0, fill=FillStyle(color=Color.red()), id="c1")
    scene.add(circle)

    # Add typed tracks
    scene.animate(circle, "radius", 30.0, 90.0, duration=2.0)  # number
    scene.animate(circle, "center", Point(100, 100), Point(200, 200), duration=2.0)  # point
    scene.animate(circle, "fill.color", Color.red(), Color.blue(), duration=2.0)  # color
    scene.animate(
        circle,
        "transform",
        Transform(rotation=0.0),
        Transform(rotation=180.0),
        duration=2.0,
    )  # transform

    # JSON round trip
    json_text = scene.to_json()
    reloaded = Scene.from_json(json_text)

    assert len(reloaded.timeline.tracks) == 4
    t_radius, t_center, t_color, t_tf = reloaded.timeline.tracks

    assert t_radius.value_type == "number"
    assert t_radius.start_value == 30.0
    assert t_radius.end_value == 90.0

    assert t_center.value_type == "point"
    assert isinstance(t_center.start_value, Point)
    assert t_center.start_value == Point(100, 100)
    assert t_center.end_value == Point(200, 200)

    assert t_color.value_type == "color"
    assert isinstance(t_color.start_value, Color)
    assert t_color.start_value == Color.red()
    assert t_color.end_value == Color.blue()

    assert t_tf.value_type == "transform"
    assert isinstance(t_tf.start_value, Transform)
    assert t_tf.start_value.rotation == 0.0
    assert t_tf.end_value.rotation == 180.0

    # Evaluate reloaded scene at t=1.0 (50%)
    reloaded.sample(1.0)
    c_reloaded = reloaded.get("c1")
    assert pytest.approx(c_reloaded.radius) == 60.0
    assert pytest.approx(c_reloaded.center.x) == 150.0
    assert c_reloaded.fill.color == Color(128, 0, 128)  # blend red & blue
    assert pytest.approx(c_reloaded.transform.rotation) == 90.0


def test_frame_sampling_policy_ceil():
    scene = Scene(100, 100)

    # 1.0s at 30 fps -> exactly 30 frames
    frames_1s = list(VideoRenderer.render_frames(scene, duration=1.0, fps=30))
    assert len(frames_1s) == 30

    # 1/12s at 30 fps -> ceil(2.5) = 3 frames
    frames_fraction = list(VideoRenderer.render_frames(scene, duration=1.0 / 12.0, fps=30))
    assert len(frames_fraction) == 3

    # 0.01s at 30 fps -> ceil(0.3) = 1 frame
    frames_tiny = list(VideoRenderer.render_frames(scene, duration=0.01, fps=30))
    assert len(frames_tiny) == 1

    # 0 duration -> 0 frames
    frames_zero = list(VideoRenderer.render_frames(scene, duration=0.0, fps=30))
    assert len(frames_zero) == 0


def test_temporal_duration_derivation():
    scene = Scene(200, 200)
    assert scene.temporal_duration == 0.0

    # Top-level drawable with timing
    line = Line(start=Point(0, 0), end=Point(50, 50))
    line.timing = Timing(start_time=1.0, duration=3.0)  # ends at 4.0
    scene.add(line)
    assert scene.temporal_duration == 4.0

    # Group child with timing
    c = Circle(center=Point(10, 10), radius=5)
    c.timing = Timing(start_time=2.0, duration=5.0)  # ends at 7.0
    grp = Group(children=[c])
    scene.add(grp)
    assert scene.temporal_duration == 7.0

    # Timeline track extending further
    scene.animate(line, "start.x", 0.0, 100.0, duration=9.0)  # ends at 9.0
    assert scene.temporal_duration == 9.0

    # Looping animation makes temporal_duration infinite
    looping_track = scene.animate(c, "radius", 5.0, 20.0, duration=2.0, loop=True)
    assert math.isinf(scene.temporal_duration)

    # render_frames must reject infinite duration unless explicit duration given
    with pytest.raises(ValidationError, match="infinite"):
        list(VideoRenderer.render_frames(scene, duration=None))

    finite_frames = list(VideoRenderer.render_frames(scene, duration=2.0, fps=10))
    assert len(finite_frames) == 20


def test_video_export_mp4(tmp_path: Path):
    scene = Scene(120, 120, background=Color.white())
    circle = Circle(center=Point(60, 60), radius=10.0, fill=FillStyle(color=Color.blue()))
    scene.add(circle)
    scene.animate(circle, "radius", 10.0, 40.0, duration=0.5)

    output_video = tmp_path / "test_animation.mp4"
    res_path = VideoRenderer.render_video(scene, output_video, duration=0.5, fps=20)

    assert res_path.exists()
    assert res_path.stat().st_size > 0


def test_image_sequence_export(tmp_path: Path):
    scene = Scene(100, 100)
    line = Line(start=Point(0, 0), end=Point(80, 80))
    line.timing = Timing(duration=0.2)
    scene.add(line)

    seq_dir = tmp_path / "frames"
    paths = VideoRenderer.render_image_sequence(scene, seq_dir, pattern="frame_%03d.png", duration=0.2, fps=10)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
