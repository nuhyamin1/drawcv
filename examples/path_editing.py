"""Curve-preserving trim/split gallery: python -m examples.path_editing."""
from pathlib import Path as FilePath

from drawcv import Scene, Path, Point, Color, StrokeStyle, Text, OpenCVRenderer


def build_scene():
    scene = Scene(900, 560, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    scene.add(Text('Trim and split curves by distance', position=Point(30, 20), color=ink, font_scale=.8))
    curve = Path(stroke=StrokeStyle(color=Color(190, 200, 210), width=7))
    curve.move_to(60, 180).cubic_to(Point(280, -10), Point(520, 320), Point(830, 110))
    scene.add(curve)
    middle = curve.trim(.2, .8, tolerance=.01)
    middle.stroke.color = Color(5, 130, 145)
    scene.add(middle)
    scene.add(Text('trim(0.2, 0.8) / original in gray', position=Point(30, 250), color=ink, font_scale=.6))
    lower = curve.to_path()
    lower.move(0, 220)
    prefix, suffix = lower.split_at(.4, tolerance=.01)
    prefix.stroke.color = Color(100, 80, 180)
    suffix.stroke.color = Color(225, 130, 30)
    scene.add(prefix)
    scene.add(suffix)
    scene.add(Text('split_at(0.4) / prefix and suffix remain editable curves',
                   position=Point(30, 490), color=ink, font_scale=.6))
    return scene


if __name__ == '__main__':
    output = FilePath(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'path_editing.png')
    scene.save_json(output / 'path_editing.json')
