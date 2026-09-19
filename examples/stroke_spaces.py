"""Compare screen/object stroke spaces: python -m examples.stroke_spaces."""
from pathlib import Path
import numpy as np
from drawcv import (Scene, Text, Point, Color, Polyline, StrokeStyle, CapStyle,
                    JoinStyle, Transform, OpenCVRenderer)


def build_scene():
    scene = Scene(900, 540, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    scene.add(Text('Stroke space / same geometry and transform', position=Point(30, 20),
                   color=ink, font_scale=.8))
    for column, space in enumerate(('screen', 'object')):
        x = 40 + column * 440
        scene.add(Text(space.upper(), position=Point(x, 85), color=ink, font_scale=.8))
        subtitle = 'Width and dashes stay in pixels' if space == 'screen' else 'Outline scales and shears with shape'
        scene.add(Text(subtitle, position=Point(x, 120), color=ink, font_scale=.5))
        for row, dashes in enumerate(((), (18, 24))):
            y = 190 + row * 200
            matrix = np.array([[2., .7, x+35], [.2, 1.5, y], [0., 0., 1.]])
            scene.add(Polyline(points=[Point(0, 0), Point(110, 0), Point(110, 55)],
                               stroke=StrokeStyle(color=Color(5, 130, 145), width=14, space=space,
                                                  cap_style=CapStyle.ROUND, join_style=JoinStyle.ROUND,
                                                  dash_array=dashes), transform=Transform.from_matrix(matrix)))
            scene.add(Text('width=14' if not dashes else 'dash_array=(18, 24)',
                           position=Point(x, y+120), color=ink, font_scale=.5))
    return scene


if __name__ == '__main__':
    output = Path(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'stroke_spaces.png')
    scene.save_json(output / 'stroke_spaces.json')
