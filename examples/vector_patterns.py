"""Editable repeating motifs: python -m examples.vector_patterns."""
from pathlib import Path as FilePath

from drawcv import (Path, Point, Circle, Rectangle, Group, Color, FillStyle,
                    StrokeStyle, VectorPattern, Transform, Scene, Text, OpenCVRenderer)


def build_scene():
    scene=Scene(960,430,background=Color(248,250,252))
    ink=Color(24,45,70)
    scene.add(Text('Editable vector patterns / paths and groups',position=Point(25,20),color=ink,font_scale=.8))
    dot=Circle(center=Point(7,7),radius=3,fill=FillStyle(color=Color(5,130,145))).to_path()
    stripe=Rectangle(width=5,height=18,fill=FillStyle(color=Color(100,80,180))).to_path()
    motif=Group(children=[
        Path(fill=FillStyle(color=Color(225,130,30))).move_to(12,2).line_to(22,12).line_to(12,22).line_to(2,12).close(),
        Circle(center=Point(12,12),radius=3,fill=FillStyle(color=ink)).to_path()])
    patterns=[VectorPattern(dot,14,14,spacing=(3,3)),
              VectorPattern(stripe,14,18,transform=Transform(rotation=30,pivot=Point(0,0))),
              VectorPattern(motif,24,24,spacing=(6,6))]
    for column,(pattern,label) in enumerate(zip(patterns,['Dots / adjustable spacing','Stripes / rotated grid','Grouped vector motif'])):
        x=30+column*315
        scene.add(Rectangle(position=Point(x,100),width=270,height=235,
                            fill=FillStyle(paint=pattern),stroke=StrokeStyle(color=ink,width=2)))
        scene.add(Text(label,position=Point(x,365),color=ink,font_scale=.55))
    return scene


if __name__=='__main__':
    output=FilePath(__file__).resolve().parent/'output';output.mkdir(exist_ok=True)
    scene=build_scene()
    OpenCVRenderer().render(scene).save(output/'vector_patterns.png')
    scene.save_json(output/'vector_patterns.json')
    (output/'vector_patterns.svg').write_text(scene.to_svg(),encoding='utf-8')
