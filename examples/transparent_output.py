"""Transparent PNG and an external compositing contact sheet.

Run from the repository root: python -m examples.transparent_output
"""
from pathlib import Path
import cv2
import numpy as np

from drawcv import (Scene, OpenCVRenderer, Color, Point, Circle, Rectangle,
                    FillStyle, StrokeStyle, Group, Transform, ImageObject,
                    BlurEffect, ShadowEffect, Mask, MaskMapping)


def build_scene():
    scene = Scene(420, 280, background=Color(0, 0, 0, 0))
    red = Rectangle(position=Point(45, 55), width=135, height=125,
                    fill=FillStyle(color=Color(255, 0, 0, .6)),
                    transform=Transform(rotation=-14),
                    effects=[BlurEffect(kernel_size=19, sigma=3)])
    scene.add(Group(children=[red], opacity=.7))

    # Hidden green RGB is deliberately present under alpha zero. Resampling
    # must discard it before filtering to keep the blue silhouette clean.
    image = np.full((36, 36, 4), (0, 255, 0, 0), np.uint8)
    image[7:29, 7:29] = (225, 130, 20, 190)
    scene.add(ImageObject(image, position=Point(220, 30), width=155, height=130,
                          transform=Transform(rotation=18), opacity=.8,
                          effects=[ShadowEffect(offset_x=12, offset_y=12, blur_radius=4,
                                                color=Color(20, 30, 65, .7))]))

    fade = np.tile(np.linspace(0, 255, 420).astype(np.uint8), (280, 1))
    group = Group(opacity=.7, mask=Mask(fade, mapping=MaskMapping.ABSOLUTE))
    for x, color in [(160, Color(0, 180, 150)), (235, Color(245, 165, 25))]:
        group.add(Circle(center=Point(x, 200), radius=42,
                         fill=FillStyle(color=color), stroke=StrokeStyle(color=color, width=5)))
    scene.add(group)
    return scene


def main():
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    scene = build_scene()
    canvas = OpenCVRenderer().render(scene, alpha=True)
    png_path = output / "transparent_output.png"
    canvas.save(png_path)
    scene.save_json(output / "transparent_output.json")

    # Independent external composition of the actual decoded PNG, rather than
    # a second call through the library's flatten/compositing implementation.
    decoded = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    alpha = decoded[..., 3:4].astype(float) / 255
    yy, xx = np.indices((280, 420))
    checker = np.where(((xx // 20 + yy // 20) % 2)[..., None], 230, 190)
    backgrounds = [np.full((280, 420, 3), 255), np.full((280, 420, 3), (25, 25, 30)),
                   np.full((280, 420, 3), (80, 125, 185)), np.repeat(checker, 3, axis=2)]
    sheet = np.full((680, 900, 3), (245, 247, 250), np.uint8)
    for i, (title, background) in enumerate(zip(
        ["WHITE", "DARK", "COLOR", "CHECKERBOARD"], backgrounds
    )):
        x, y = 20 + (i % 2)*440, 45 + (i // 2)*330
        cv2.putText(sheet, title, (x, y-14), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 55, 35), 1, cv2.LINE_AA)
        sheet[y:y+280, x:x+420] = np.rint(decoded[..., :3]*alpha + background*(1-alpha)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(output / "transparent_output_preview.png"), sheet)


if __name__ == "__main__":
    main()
