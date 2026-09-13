"""Fixed workloads without random/system assets or optional typography engines."""
import math
import numpy as np
from drawcv import (Scene, Point, StrokePoint, Color, Rectangle, Circle, Polyline,
    FreehandStroke, Group, FillStyle, StrokeStyle, LinearGradient, RadialGradient,
    GradientStop, Mask, BlurEffect, ShadowEffect, ClipRect, Transform)

CASES = ('dense_bgr', 'small_strokes', 'small_gradients', 'long_freehand',
         'dense_dashes', 'deep_transparency', 'masks_effects', 'animation')
TIMES = (0.0, 0.5, 1.0)
STRESS_CASES = ('deep_transparency_stress',)


def build_scene(case):
    scene = Scene(640, 360, background=Color(20, 28, 40, 0))
    stops = (GradientStop(0, Color(250, 50, 30, .2)),
             GradientStop(.5, Color(60, 190, 170, .8)), GradientStop(1, Color(40, 80, 245)))
    with scene.history.suspended():
        if case in ('dense_bgr', 'small_strokes', 'small_gradients', 'animation'):
            count = {'dense_bgr': 1200, 'small_strokes': 480, 'small_gradients': 72, 'animation': 24}[case]
            for i in range(count):
                x, y = (i*37) % 615, (i*53) % 335
                fill = FillStyle(color=Color((i*13)%256, (i*31)%256, (i*59)%256))
                stroke = None
                if case == 'small_strokes':
                    fill = None
                    stroke = StrokeStyle(width=2+(i%4), color=Color(220, 170, 100, .7))
                if case in ('small_gradients', 'animation'):
                    paint = (LinearGradient(Point(x, y), Point(x+35, y+25), stops) if i%2 else
                             RadialGradient(Point(x+14, y+12), 24, stops))
                    fill = FillStyle(paint=paint)
                obj = Rectangle(id=f'box-{i}', position=Point(x, y), width=25, height=20, fill=fill, stroke=stroke)
                scene.add(obj)
                if case == 'animation':
                    scene.animate(obj, 'transform.translation_x', 0, 18, duration=1)
                    scene.animate(obj, 'opacity', 1, .4, duration=1)
        elif case == 'long_freehand':
            points = [StrokePoint(20+(i%600)/600*600,
                25+(i//600)*31+10*math.sin(i*.09), pressure=.5+.4*math.sin(i*.023)) for i in range(6000)]
            scene.add(FreehandStroke(id='ink', points=points, stroke=StrokeStyle(width=3),
                variable_width=True, width_mode='pressure', min_width=1, max_width=5))
        elif case == 'dense_dashes':
            for i in range(24):
                points = [Point(12+j*8.8, 10+i*14+5*math.sin(j*.9)) for j in range(70)]
                scene.add(Polyline(id=f'dashes-{i}', points=points, stroke=StrokeStyle(width=2,
                    color=Color(70, 190, 240, .8), dash_array=(2, 1, 3, 1), dash_offset=i*.3)))
        elif case in ('deep_transparency', 'deep_transparency_stress'):
            count = 24 if case.endswith('_stress') else 3
            children = [Circle(id=f'disc-{i}', center=Point(120+(i%8)*53, 95+(i//8)*60), radius=30,
                fill=FillStyle(color=Color(240, 40+i*5, 150, .6))) for i in range(count)]
            group = Group(id='depth-0', children=children, opacity=.95)
            for i in range(1, 10 if case.endswith('_stress') else 6):
                group = Group(id=f'depth-{i}', children=[group], opacity=.95,
                    transform=Transform(rotation=1, pivot=Point(320, 180)))
            scene.add(group)
        elif case == 'masks_effects':
            mask = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (48, 1))
            for i in range(16):
                x, y = 25+(i%4)*155, 25+(i//4)*82
                obj = Rectangle(id=f'effect-{i}', position=Point(x, y), width=90, height=42,
                    fill=FillStyle(paint=LinearGradient(Point(x, y), Point(x+90, y+42), stops)),
                    mask=Mask(mask.copy()), clip=ClipRect(x-5, y-5, 112, 66), opacity=.8,
                    effects=[BlurEffect(kernel_size=7), ShadowEffect(offset_x=5, offset_y=6, blur_radius=3)])
                scene.add(obj)
        else:
            raise ValueError(f'Unknown benchmark: {case}')
    return scene


def describe(scene):
    def walk(obj, depth):
        yield obj, depth
        for child in getattr(obj, 'children', ()):
            yield from walk(child, depth+1)
    nodes = [item for obj in scene.objects for item in walk(obj, 1)]
    return {'width': scene.width, 'height': scene.height, 'objects_including_groups': len(nodes),
            'max_depth': max((depth for _, depth in nodes), default=0),
            'raw_points': sum(len(getattr(obj, 'points', ())) for obj, _ in nodes),
            'timeline_tracks': len(scene.timeline.tracks)}
