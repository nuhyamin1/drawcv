"""Filled-region offsets: python -m examples.path_offsets."""
from pathlib import Path as FilePath

from drawcv import Scene, Path, Point, Color, FillStyle, StrokeStyle, Text, OpenCVRenderer


def build_scene():
    scene = Scene(960, 400, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    scene.add(Text('Offsets expand regions and contract their holes', position=Point(25, 20), color=ink, font_scale=.8))
    source = Path(fill=FillStyle(color=Color(5, 130, 145)))
    source.move_to(70, 120).line_to(230, 120).line_to(230, 280).line_to(70, 280).close()
    source.move_to(120, 170).line_to(180, 170).line_to(180, 230).line_to(120, 230).close()
    for column, distance in enumerate((-15, 0, 15)):
        result = source.offset(distance, tolerance=.1)
        result.move(column*320, 0)
        scene.add(result)
        reference = source.to_path()
        reference.fill = None
        reference.stroke = StrokeStyle(color=Color(120, 135, 150), width=1, dash_array=(5, 5))
        reference.move(column*320, 0)
        scene.add(reference)
        scene.add(Text(f'offset({distance:+d})', position=Point(80+column*320, 330), color=ink, font_scale=.7))
    return scene


if __name__ == '__main__':
    output = FilePath(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'path_offsets.png')
    scene.save_json(output / 'path_offsets.json')
