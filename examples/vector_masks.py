"""Editable mask fades: python -m examples.vector_masks."""
from pathlib import Path as FilePath

from drawcv import (Scene, Path, Point, Rectangle, Color, FillStyle, LinearGradient,
                    GradientStop, VectorMask, BoundingBox, Transform, Text, OpenCVRenderer)


def build_scene():
    scene=Scene(960,450,background=Color(248,250,252))
    ink=Color(24,45,70)
    scene.add(Text('Vector masks / editable gradients and cutouts',position=Point(25,20),color=ink,font_scale=.8))
    for column,(mode,label) in enumerate([('alpha','Alpha / transparent to opaque'),('luminance','Luminance / black to white')]):
        colors=(Color(255,255,255,0),Color(255,255,255)) if mode=='alpha' else (Color(0,0,0),Color(255,255,255))
        art=Rectangle(width=400,height=220,fill=FillStyle(paint=LinearGradient(
            Point(0,0),Point(400,0),(GradientStop(0,colors[0]),GradientStop(1,colors[1]))))).to_path()
        region=Path().move_to(0,20).line_to(400,20).line_to(360,210).line_to(30,190).close()
        art.clip=region
        mask=VectorMask(art,BoundingBox(0,0,400,220),mode=mode)
        shape=Rectangle(width=400,height=220,fill=FillStyle(color=Color(5,130,145)),mask=mask,
                        transform=Transform(translation_x=40+column*480,translation_y=100))
        scene.add(shape)
        scene.add(Text(label,position=Point(40+column*480,350),color=ink,font_scale=.6))
    return scene


if __name__=='__main__':
    output=FilePath(__file__).resolve().parent/'output';output.mkdir(exist_ok=True)
    scene=build_scene()
    OpenCVRenderer().render(scene).save(output/'vector_masks.png')
    scene.save_json(output/'vector_masks.json')
    (output/'vector_masks.svg').write_text(scene.to_svg(),encoding='utf-8')
