"""Retained, portable multilingual Text. Run: python -m examples.retained_typography."""
from pathlib import Path
import cv2
import numpy as np
from drawcv import (Scene, OpenCVRenderer, Text, FontAsset, Point, Color, FillStyle,
    GradientStop, LinearGradient, Group, Transform, ShadowEffect, Rectangle, StrokeStyle)


def build_scene():
    assets = Path(__file__).resolve().parents[1] / 'tests/assets/fonts'
    fonts = tuple(FontAsset.from_file(assets/name) for name in
                  ['sourcesans3.otf', 'notosansthai.ttf', 'notosansarabic.ttf'])
    scene = Scene(1080, 680, background=Color(0, 0, 0, 0))
    scene.add(Text('RETAINED TYPOGRAPHY / DrawCV', position=Point(42,22), font_scale=.8,
                   color=Color(25,45,75), padding=10, background_fill=Color(255,255,255,.95)))
    content = ('Portable fonts. Editable text. Shared baselines.\n'
        'ภาษาไทยต้องตัดคำอย่างถูกต้อง เพื่อให้อ่านข้อความได้ง่ายขึ้น\n'
        'Latin + ไทย: น้ำ กุ้ง กิ๊ง ปู่ ผู้ 123\n'
        'Arabic + Latin: العربية DrawCV 123\n'
        'ABC (العربية) 123\n'
        'Advance width, paragraph box, and visible ink are distinct.')
    paint = LinearGradient(Point(60,100), Point(960,590), (
        GradientStop(0,Color(252,254,255,.98)), GradientStop(1,Color(208,232,249,.95))))
    text = Text(content, fonts=fonts, font_size=28, wrap_width=900, line_spacing=1.1,
                position=Point(60,110), padding=18, color=Color(20,40,65),
                background_fill=FillStyle(paint=paint), background_radius=16,
                effects=[ShadowEffect(offset_x=8,offset_y=12,blur_radius=5,color=Color(10,30,60,.25))])
    metrics = text.measure()
    guides = []
    for box,color in [(metrics.layout_bounds, Color(60,125,200,.65)),
                      (metrics.ink_bounds, Color(10,150,110,.65))]:
        guides.append(Rectangle(position=Point(box.x,box.y),width=box.width,height=box.height,
                                stroke=StrokeStyle(color=color,width=1)))
    scene.add(Group(children=[text,*guides],opacity=.95,
                    transform=Transform(rotation=1.0,pivot=Point(540,340))))
    scene.add(Text('Blue: layout box    Green: ink bounds    Font bytes travel with JSON',
                   position=Point(55,620),font_scale=.54,color=Color(25,45,75),padding=8,
                   background_fill=Color(255,255,255,.95)))
    return scene


def main():
    output = Path(__file__).resolve().parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene, alpha=True)
    canvas.save(output/'retained_typography.png')
    scene.save_json(output/'retained_typography.json')
    loaded = Scene.load_json(output/'retained_typography.json')
    assert np.array_equal(canvas.buffer,renderer.render(loaded,alpha=True).buffer)
    y,x = np.indices((scene.height,scene.width))
    checker = np.where(((x//24+y//24)%2)[...,None],225,205)
    a = canvas.buffer[...,3:4]/255
    preview = np.rint(canvas.buffer[...,:3]*a+checker*(1-a)).astype(np.uint8)
    cv2.imwrite(str(output/'retained_typography_preview.png'),preview)


if __name__ == '__main__':
    main()
