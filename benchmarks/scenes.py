"""Fixed workloads without random/system assets or optional typography engines."""
import math
import numpy as np
from drawcv import (Scene, Point, StrokePoint, Color, Rectangle, Circle, Polyline,
    FreehandStroke, Group, FillStyle, StrokeStyle, LinearGradient, RadialGradient,
    GradientStop, Mask, BlurEffect, ShadowEffect, GlowEffect, BrightnessContrastEffect,
    SaturationEffect, HueShiftEffect, GrayscaleEffect, SepiaEffect, ColorMatrixEffect,
    ClipRect, Transform)

CASES = ('dense_bgr', 'small_strokes', 'small_gradients', 'long_freehand',
         'dense_dashes', 'deep_transparency', 'masks_effects', 'animation',
         'single_blur', 'single_shadow', 'blur_shadow_blur', 'glow_shadow',
         'color_stack_5', 'nested_hierarchy_effects', 'large_translucent_spatial',
         'dense_small_effects')
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
        elif case == 'single_blur':
            for i in range(16):
                x, y = 30 + (i % 4) * 145, 30 + (i // 4) * 75
                obj = Rectangle(id=f'blur-{i}', position=Point(x, y), width=80, height=45,
                                fill=FillStyle(color=Color(200, 60, 40, 0.9)),
                                effects=[BlurEffect(kernel_size=15, sigma=4.0)])
                scene.add(obj)
        elif case == 'single_shadow':
            for i in range(16):
                x, y = 30 + (i % 4) * 145, 30 + (i // 4) * 75
                obj = Rectangle(id=f'shadow-{i}', position=Point(x, y), width=80, height=45,
                                fill=FillStyle(color=Color(40, 120, 220, 0.9)),
                                effects=[ShadowEffect(offset_x=8.0, offset_y=8.0, blur_radius=4.0, color=Color(0, 0, 0, 0.6))])
                scene.add(obj)
        elif case == 'blur_shadow_blur':
            for i in range(12):
                x, y = 40 + (i % 4) * 140, 40 + (i // 3) * 90
                obj = Circle(id=f'bsb-{i}', center=Point(x + 30, y + 30), radius=25,
                             fill=FillStyle(color=Color(50, 200, 100, 0.85)),
                             effects=[
                                 BlurEffect(kernel_size=7, sigma=2.0),
                                 ShadowEffect(offset_x=6.0, offset_y=6.0, blur_radius=3.0),
                                 BlurEffect(kernel_size=7, sigma=2.0),
                             ])
                scene.add(obj)
        elif case == 'glow_shadow':
            for i in range(12):
                x, y = 40 + (i % 4) * 140, 40 + (i // 3) * 90
                obj = Circle(id=f'gs-{i}', center=Point(x + 30, y + 30), radius=25,
                             fill=FillStyle(color=Color(220, 50, 180, 0.9)),
                             effects=[
                                 GlowEffect(blur_radius=6.0, color=Color(255, 220, 0, 1.0)),
                                 ShadowEffect(offset_x=8.0, offset_y=8.0, blur_radius=3.0),
                             ])
                scene.add(obj)
        elif case == 'color_stack_5':
            for i in range(16):
                x, y = 30 + (i % 4) * 145, 30 + (i // 4) * 75
                obj = Rectangle(id=f'color-{i}', position=Point(x, y), width=80, height=45,
                                fill=FillStyle(color=Color((i * 37) % 256, (i * 73) % 256, (i * 109) % 256, 0.9)),
                                effects=[
                                    BrightnessContrastEffect(brightness=0.05, contrast=1.1),
                                    SaturationEffect(factor=1.2),
                                    HueShiftEffect(angle=30.0),
                                    GrayscaleEffect(intensity=0.15),
                                    SepiaEffect(intensity=0.1),
                                ])
                scene.add(obj)
        elif case == 'nested_hierarchy_effects':
            for i in range(4):
                layer = scene.create_layer(f'layer-{i}', z_order=i * 10)
                layer.effects = [BlurEffect(kernel_size=7)]
                group = Group(id=f'group-{i}')
                group.effects = [ShadowEffect(offset_x=10.0, offset_y=10.0, blur_radius=3.0)]
                child = Rectangle(id=f'nested-{i}', position=Point(50 + i * 120, 100), width=70, height=70,
                                  fill=FillStyle(color=Color(240, 120, 40, 0.9)),
                                  effects=[GlowEffect(blur_radius=5.0, color=Color(255, 255, 0))])
                group.add(child)
                layer.add(group)
        elif case == 'large_translucent_spatial':
            for i in range(4):
                obj = Circle(id=f'large-{i}', center=Point(160 + i * 100, 180), radius=70,
                             fill=FillStyle(color=Color(100 + i * 30, 200 - i * 30, 250, 0.4)),
                             effects=[
                                 GlowEffect(blur_radius=12.0, color=Color(255, 255, 100)),
                                 BlurEffect(kernel_size=21, sigma=5.0),
                             ])
                scene.add(obj)
        elif case == 'dense_small_effects':
            for i in range(48):
                x, y = 20 + (i % 8) * 75, 20 + (i // 8) * 55
                obj = Rectangle(id=f'dense-{i}', position=Point(x, y), width=35, height=25,
                                fill=FillStyle(color=Color((i * 19) % 256, (i * 47) % 256, (i * 83) % 256, 0.85)),
                                effects=[
                                    ShadowEffect(offset_x=3.0, offset_y=3.0, blur_radius=2.0),
                                    BrightnessContrastEffect(brightness=0.1, contrast=1.05),
                                ])
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
