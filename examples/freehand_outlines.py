"""Editable pressure outlines: python -m examples.freehand_outlines."""
from pathlib import Path

from drawcv import (Scene, Text, Point, StrokePoint, Color, FreehandStroke,
                    StrokeStyle, OpenCVRenderer)


def build_scene():
    scene = Scene(900, 600, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    scene.add(Text('Freehand strokes become editable filled paths',
                   position=Point(30, 20), color=ink, font_scale=.8))
    for column, label in enumerate(('ORIGINAL STROKE', 'CONVERTED OUTLINE')):
        scene.add(Text(label, position=Point(40 + column * 450, 80), color=ink, font_scale=.65))
    for row, (label, mode, dashes) in enumerate([
            ('Pressure / smooth taper', 'pressure', ()),
            ('Pressure / dashed', 'pressure', (35, 20)),
            ('Velocity / faster means thinner', 'velocity', ())]):
        y = 160 + row * 150
        source = FreehandStroke(
            points=[StrokePoint(50, y, pressure=0, velocity=100),
                    StrokePoint(130, y+35, pressure=.6, velocity=40),
                    StrokePoint(230, y-15, pressure=1, velocity=0),
                    StrokePoint(350, y+30, pressure=.15, velocity=85)],
            stroke=StrokeStyle(color=Color(5, 130, 145), width=20, dash_array=dashes),
            variable_width=True, width_mode=mode, min_width=0, max_width=32,
            velocity_max=100, smoothing='chaikin', smoothing_iterations=2)
        scene.add(source)
        outline = source.stroke_to_path(tolerance=.15)
        outline.transform.translation_x = 450
        outline.fill.color = Color(100, 80, 180)
        scene.add(outline)
        scene.add(Text(label, position=Point(40, y+65), color=ink, font_scale=.5))
    return scene


if __name__ == '__main__':
    output = Path(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'freehand_outlines.png')
    scene.save_json(output / 'freehand_outlines.json')
