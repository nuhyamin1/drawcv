"""Retained gradient gallery. Run: python -m examples.gradient_fills."""
from pathlib import Path
import cv2
import numpy as np
from drawcv import (Scene, OpenCVRenderer, Color, Point, FillStyle, GradientStop,
                    LinearGradient, RadialGradient, RoundedRectangle, Circle,
                    Transform, Group, BlurEffect)


def build_scene():
    scene = Scene(780, 480, background=Color(0, 0, 0, 0))
    stops = (GradientStop(0, Color(255, 100, 60)),
             GradientStop(.5, Color(190, 65, 200)),
             GradientStop(1, Color(30, 150, 245)))
    scene.add(RoundedRectangle(x=25, y=50, width=210, height=145, corner_radius=22,
        fill=FillStyle(paint=LinearGradient(Point(25, 50), Point(235, 195), stops))))
    scene.add(Circle(center=Point(390, 125), radius=76,
        fill=FillStyle(paint=RadialGradient(Point(368, 100), 110, (
            GradientStop(0, Color(255, 245, 185)), GradientStop(.45, Color(255, 135, 60)),
            GradientStop(1, Color(130, 35, 95)))))))
    hard = (GradientStop(.15, Color(35, 200, 175)), GradientStop(.5, Color(35, 200, 175)),
            GradientStop(.5, Color(85, 70, 205)), GradientStop(.85, Color(85, 70, 205)))
    scene.add(RoundedRectangle(x=545, y=50, width=210, height=145, corner_radius=22,
        fill=FillStyle(paint=LinearGradient(Point(565, 50), Point(735, 50), hard))))
    for x, space in [(35, "object"), (295, "world")]:
        scene.add(RoundedRectangle(x=x, y=300, width=190, height=120, corner_radius=18,
            transform=Transform(rotation=18, pivot=Point(x+95, 360)),
            fill=FillStyle(paint=LinearGradient(Point(x, 360), Point(x+190, 360), stops, space))))
    glow = Circle(center=Point(650, 360), radius=82,
        fill=FillStyle(paint=RadialGradient(Point(650, 360), 76, (
            GradientStop(0, Color(245, 60, 75)), GradientStop(.35, Color(245, 60, 75, .8)),
            GradientStop(1, Color(245, 60, 75, 0))))),
        effects=[BlurEffect(kernel_size=15, sigma=3)])
    scene.add(Group(children=[glow], opacity=.75))
    return scene


def main():
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    scene = build_scene()
    canvas = OpenCVRenderer().render(scene, alpha=True)
    path = output / "gradient_fills.png"
    canvas.save(path)
    scene.save_json(output / "gradient_fills.json")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    a = decoded[..., 3:4].astype(float)/255
    yy, xx = np.indices((480, 780))
    checker = np.repeat(np.where(((xx//16 + yy//16) % 2)[..., None], 48, 40), 3, axis=2)
    preview = np.rint(decoded[..., :3]*a + checker*(1-a)).astype(np.uint8)
    for i, label in enumerate(["LINEAR / THREE STOPS", "RADIAL / OFFSET CENTER", "DUPLICATES / HARD EDGE",
                               "OBJECT / ROTATES WITH SHAPE", "WORLD / FIXED DIRECTION", "ALPHA / BLUR / GROUP OPACITY"]):
        cv2.putText(preview, label, (15+(i % 3)*260, 27+(i//3)*240),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "gradient_fills_preview.png"), preview)


if __name__ == "__main__":
    main()
