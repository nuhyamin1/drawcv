"""Text labels attached to shapes: python -m examples.shape_labels."""
from pathlib import Path as FilePath

from drawcv import (Circle, Color, FillStyle, Label, OpenCVRenderer, Path, Point, Rectangle,
                    RoundedRectangle, Scene, StrokeStyle, Text)

INK = Color(24, 45, 70)
CELL = FillStyle(color=Color(226, 236, 248))


def build_scene():
    scene = Scene(900, 560, background=Color(248, 250, 252))
    scene.add(Text("Labels align inside or around their shape, and follow its edits",
                   position=Point(30, 20), color=INK, font_scale=0.75, thickness=2))

    # Nine inside alignments on the same box size.
    for row, vertical_align in enumerate(("top", "middle", "bottom")):
        for col, align in enumerate(("left", "center", "right")):
            cell = Rectangle(position=Point(30 + col * 150, 70 + row * 100), width=140, height=90,
                             fill=CELL, stroke=StrokeStyle(color=Color(150, 170, 200), width=1))
            cell.label = Label(f"{vertical_align}\n{align}", align=align, vertical_align=vertical_align,
                               padding=8, color=INK, font_scale=0.5)
            scene.add(cell)

    # Outside placements act as captions.
    node = Circle(center=Point(690, 170), radius=46, fill=FillStyle(color=Color(252, 222, 200)),
                  stroke=StrokeStyle(color=Color(200, 120, 70), width=2))
    node.label = Label("node", color=INK, font_scale=0.7, thickness=2)
    scene.add(node)
    for placement in ("above", "below", "left", "right"):
        caption = Circle(center=Point(690, 170), radius=46)
        caption.label = Label(placement, placement=placement, padding=8, color=Color(120, 80, 50), font_scale=0.5)
        scene.add(caption)

    # A label rotates, scales and moves with its host.
    card = RoundedRectangle(x=560, y=300, width=240, height=90, corner_radius=14,
                            fill=FillStyle(color=Color(210, 238, 222)),
                            stroke=StrokeStyle(color=Color(60, 140, 90), width=2))
    card.transform.rotation = -12
    card.label = Label("Rotated with its card", color=Color(30, 90, 50), font_scale=0.65, thickness=2)
    scene.add(card)

    # Paths can be written as SVG path data.
    # Transform scale would scale the label too, so the geometry is authored at size.
    heart = Path.from_svg_path("M0 -16 C-20 -44 -68 -24 -52 16 C-40 44 0 64 0 76 C0 64 40 44 52 16 "
                               "C68 -24 20 -44 0 -16 Z", fill=FillStyle(color=Color(230, 80, 100)))
    heart.move(170, 400)
    heart.label = Label("Path.from_svg_path", placement="below", padding=12, color=INK, font_scale=0.55)
    scene.add(heart)
    return scene


if __name__ == "__main__":
    output = FilePath(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    scene = build_scene()
    OpenCVRenderer().render(scene).save(output / "shape_labels.png")
    scene.save_json(output / "shape_labels.json")
    scene.save_svg(output / "shape_labels.svg")
