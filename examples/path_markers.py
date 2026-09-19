"""Reusable marker gallery: python -m examples.path_markers."""
from pathlib import Path as FilePath

from drawcv import (Path, Point, Circle, Marker, Color, FillStyle, StrokeStyle,
                    Scene, Text, OpenCVRenderer)


def build_scene():
    scene = Scene(900, 480, background=Color(248,250,252))
    ink = Color(24,45,70)
    scene.add(Text('Reusable vector markers follow paths', position=Point(30,20), color=ink, font_scale=.8))
    arrow = Path(fill=FillStyle(color=Color(5,130,145)))
    arrow.move_to(0,0).line_to(-5,-2.5).line_to(-3.5,0).line_to(-5,2.5).close()
    ends = Marker(arrow, orient='auto-start-reverse')
    dot = Circle(center=Point(0,0),radius=1.4,fill=FillStyle(color=Color(100,80,180))).to_path()
    for row in range(2):
        y = 130+row*200
        path = Path(stroke=StrokeStyle(color=ink, width=5))
        path.move_to(90,y)
        if row == 0:
            path.line_to(390,y).line_to(500,y+60).line_to(800,y+60)
        else:
            path.cubic_to(Point(250,y-90),Point(610,y+110),Point(800,y))
        path.marker_start = path.marker_end = ends
        path.marker_mid = Marker(dot)
        scene.add(path)
    scene.add(Text('Shared arrowheads / vertex dots / curve tangents', position=Point(30,430), color=ink, font_scale=.6))
    return scene


if __name__ == '__main__':
    output = FilePath(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'path_markers.png')
    scene.save_json(output / 'path_markers.json')
    (output / 'path_markers.svg').write_text(scene.to_svg(), encoding='utf-8')
