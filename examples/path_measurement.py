"""Place markers by arc length: python -m examples.path_measurement."""
from pathlib import Path as FilePath

from drawcv import Scene, Path, Point, Circle, Line, Color, FillStyle, StrokeStyle, Text, OpenCVRenderer


def build_scene():
    scene = Scene(900, 400, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    scene.add(Text('Equal-distance markers and tangent directions',
                   position=Point(30, 20), color=ink, font_scale=.8))
    curve = Path(stroke=StrokeStyle(color=Color(5, 130, 145), width=4))
    curve.move_to(60, 270).cubic_to(Point(280, -70), Point(520, 460), Point(830, 160))
    scene.add(curve)
    for index in range(13):
        progress = index / 12
        point = curve.point_at(progress, tolerance=.01)
        tangent = curve.tangent_at(progress, tolerance=.01)
        tip = Point(point.x+24*tangent.x, point.y+24*tangent.y)
        scene.add(Line(start=point, end=tip, stroke=StrokeStyle(color=Color(100, 80, 180), width=3)))
        scene.add(Circle(center=point, radius=5, fill=FillStyle(color=ink)))
    scene.add(Text(f'Length: {curve.length(tolerance=.01):.2f} units | 12 equal intervals',
                   position=Point(30, 345), color=ink, font_scale=.6))
    return scene


if __name__ == '__main__':
    output = FilePath(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / 'path_measurement.png')
    scene.save_json(output / 'path_measurement.json')
