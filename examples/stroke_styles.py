"""Run: python -m examples.stroke_styles (writes a PNG and editable JSON)."""
from pathlib import Path as FilePath

from drawcv import (Scene, OpenCVRenderer, Point, Color, StrokeStyle,
                    CapStyle, JoinStyle, Line, Polyline, Text,
                    FreehandStroke, StrokePoint, Path)


def build_scene():
    scene = Scene(960, 680, background=Color(248, 250, 252))
    ink = Color(24, 45, 70)
    accent = Color(5, 130, 145)

    def label(text, x, y, scale=.55):
        scene.add(Text(text, position=Point(x, y), color=ink, font_scale=scale))

    label("pydrawcv / editable stroke styles", 35, 18, .9)
    label("CAPS", 35, 82)
    for i, cap in enumerate(CapStyle):
        x = 70 + i * 300
        label(cap.value, x, 115)
        scene.add(Line(start=Point(x, 170), end=Point(x+170, 170),
                       stroke=StrokeStyle(color=accent, width=28, cap_style=cap)))
        # Thin guides mark the authored endpoints.
        for gx in (x, x+170):
            scene.add(Line(start=Point(gx, 146), end=Point(gx, 194),
                           stroke=StrokeStyle(color=Color(235, 100, 80), width=1)))

    label("JOINS", 35, 220)
    for i, join in enumerate(JoinStyle):
        x = 70 + i * 300
        label(join.value, x, 252)
        scene.add(Polyline(points=[Point(x, 360), Point(x+85, 285), Point(x+170, 360)],
                           stroke=StrokeStyle(color=ink, width=26, join_style=join)))

    label("DASHES / continuous through corners", 35, 400)
    scene.add(Polyline(points=[Point(45, 465), Point(310, 465), Point(350, 430), Point(600, 430)],
                       stroke=StrokeStyle(color=accent, width=12, dash_array=(24, 14),
                                          cap_style=CapStyle.BUTT, join_style=JoinStyle.MITER)))
    contour = Path(stroke=StrokeStyle(color=ink, width=10, dash_array=(28, 12),
                                      cap_style=CapStyle.ROUND, dash_offset=8))
    contour.move_to(705, 415).line_to(900, 415).line_to(900, 495).line_to(705, 495).close()
    scene.add(contour)
    label("PRESSURE / widths interpolate at dash cuts", 35, 525)
    scene.add(FreehandStroke(
        points=[StrokePoint(45, 605, pressure=0), StrokePoint(880, 605, pressure=1)],
        stroke=StrokeStyle(color=accent, width=20, dash_array=(40, 40), cap_style=CapStyle.ROUND),
        variable_width=True, width_mode="pressure", min_width=3, max_width=36))
    return scene


def main():
    output = FilePath(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / "stroke_styles.png")
    scene.save_json(output / "stroke_styles.json")


if __name__ == "__main__":
    main()
