"""Fixed workloads without random/system assets or optional typography engines."""
import math
import numpy as np
from drawcv import (Scene, Point, StrokePoint, Color, Rectangle, Circle, Polyline,
    FreehandStroke, Group, FillStyle, StrokeStyle, LinearGradient, RadialGradient,
    GradientStop, Mask, BlurEffect, ShadowEffect, GlowEffect, BrightnessContrastEffect,
    SaturationEffect, HueShiftEffect, GrayscaleEffect, SepiaEffect, ColorMatrixEffect,
    ClipRect, Transform, RoundedRectangle, Polygon, Arrow, Ellipse, Line, Text, FontFamily)

CASES = ('dense_bgr', 'small_strokes', 'small_gradients', 'long_freehand',
         'dense_dashes', 'deep_transparency', 'masks_effects', 'animation',
         'single_blur', 'single_shadow', 'blur_shadow_blur', 'glow_shadow',
         'color_stack_5', 'nested_hierarchy_effects', 'large_translucent_spatial',
         'dense_small_effects', 'lesson', 'lesson_25', 'lesson_no_effects')
TIMES = (0.0, 0.5, 1.0)
STRESS_CASES = ('deep_transparency_stress',)
# Cases rendered as legacy BGR (alpha=False); every other case renders BGRA.
BGR_CASES = ('dense_bgr', 'lesson', 'lesson_25', 'lesson_no_effects')
LESSON_CASES = ('lesson', 'lesson_25', 'lesson_no_effects')


def _two_stop(start, end):
    return (GradientStop(0, Color(*start)), GradientStop(1, Color(*end)))


def _label(layer, key, x, y, width, text):
    """TutorDraw-style label: white plate, hairline border and Hershey text."""
    layer.add(Rectangle(id=f'{key}-plate', position=Point(x, y), width=width, height=44,
        fill=FillStyle(color=Color(255, 255, 255)), stroke=StrokeStyle(width=1, color=Color(194, 207, 223))))
    layer.add(Text(text, Point(x+10, y+10), color=Color(28, 43, 65), font_scale=.7, id=f'{key}-text'))


def build_lesson(case):
    """960x540 schematic lesson mirroring Illustrate's lunar-eclipse frame.

    Built directly with DrawCV using the same object types and parameters that
    TutorDraw emits: a gradient panel with a drop shadow, a 0.85-opacity gradient
    polygon, an arrow, radial-gradient circles (one with a glow), a dashed ellipse,
    a caption and an overlay layer of three labels with a leader line.
    """
    effects = case != 'lesson_no_effects'
    scene = Scene(960, 540, background=Color(246, 248, 253))
    with scene.history.suspended():
        scene.add(RoundedRectangle(id='sky', x=36, y=36, width=888, height=468, corner_radius=22,
            fill=FillStyle(paint=LinearGradient(Point(36, 36), Point(924, 504),
                _two_stop((236, 241, 252), (252, 244, 236)))),
            stroke=StrokeStyle(width=2, color=Color(214, 222, 236)),
            effects=[ShadowEffect(offset_x=0, offset_y=4, blur_radius=10, color=Color(22, 40, 72),
                                  opacity=.22)] if effects else []))
        scene.add(Polygon(id='umbra', vertices=[Point(430, 218), Point(430, 352), Point(904, 316), Point(904, 254)],
            fill=FillStyle(paint=LinearGradient(Point(430, 285), Point(904, 285),
                _two_stop((120, 128, 150), (206, 212, 228)))), opacity=.85))
        scene.add(Arrow(id='sunlight', start=Point(196, 285), end=Point(352, 285), head_length=22, head_width=17,
            stroke=StrokeStyle(width=5, color=Color(222, 158, 40)), fill=FillStyle(color=Color(222, 158, 40))))
        scene.add(Circle(id='sun', center=Point(124, 285), radius=58,
            fill=FillStyle(paint=RadialGradient(Point(104, 262), 86, _two_stop((255, 248, 214), (234, 154, 24)))),
            effects=[GlowEffect(blur_radius=15, color=Color(255, 198, 74), opacity=.5)] if effects else []))
        scene.add(Circle(id='earth', center=Point(430, 285), radius=52,
            fill=FillStyle(paint=RadialGradient(Point(412, 266), 74, _two_stop((104, 168, 236), (34, 84, 158))))))
        scene.add(Text('schematic - sizes and distances not to scale', Point(70, 478), color=Color(128, 140, 164),
            font_family=FontFamily.DUPLEX, font_scale=.55, id='caption'))
        scene.add(Ellipse(id='orbit', center=Point(430, 285), radius_x=320, radius_y=150,
            stroke=StrokeStyle(width=2, color=Color(166, 180, 206), dash_array=(12, 9))))
        scene.add(Circle(id='moon', center=Point(742, 140), radius=23,
            fill=FillStyle(paint=RadialGradient(Point(734, 132), 32, _two_stop((226, 230, 240), (142, 152, 174))))))
        if case == 'lesson_25':
            # 17 more objects with the same mix of paints, effects and strokes.
            for i in range(4):
                x = 560+i*90
                scene.add(Circle(id=f'star-{i}', center=Point(x, 430), radius=26,
                    fill=FillStyle(paint=RadialGradient(Point(x-8, 422), 38,
                        _two_stop((255, 244, 200), (230, 170, 60)))),
                    effects=[GlowEffect(blur_radius=8, color=Color(255, 210, 90), opacity=.5)] if i % 2 else []))
            for i in range(3):
                x = 120+i*150
                scene.add(Polygon(id=f'wedge-{i}', vertices=[Point(x, 90), Point(x+110, 80), Point(x+90, 170)],
                    fill=FillStyle(paint=LinearGradient(Point(x, 90), Point(x+110, 170),
                        _two_stop((150, 170, 210), (220, 226, 240)))), opacity=.85))
            for i in range(3):
                y = 360+i*30
                scene.add(Arrow(id=f'ray-{i}', start=Point(80, y), end=Point(300, y-40), head_length=16, head_width=12,
                    stroke=StrokeStyle(width=3, color=Color(222, 158, 40)), fill=FillStyle(color=Color(222, 158, 40))))
            for i in range(2):
                scene.add(Ellipse(id=f'path-{i}', center=Point(742, 285), radius_x=90+i*40, radius_y=60+i*30,
                    stroke=StrokeStyle(width=2, color=Color(166, 180, 206), dash_array=(8, 6))))
            for i in range(3):
                x = 600+i*100
                scene.add(RoundedRectangle(id=f'card-{i}', x=x, y=60, width=80, height=50, corner_radius=10,
                    fill=FillStyle(paint=LinearGradient(Point(x, 60), Point(x+80, 110),
                        _two_stop((255, 255, 255), (232, 238, 248)))),
                    stroke=StrokeStyle(width=1, color=Color(194, 207, 223)),
                    effects=[ShadowEffect(offset_x=0, offset_y=3, blur_radius=6, color=Color(22, 40, 72),
                                          opacity=.2)]))
            for i in range(2):
                scene.add(Text(f'note {i+1}: light travels in straight lines', Point(560, 200+i*24),
                    color=Color(80, 92, 120), font_scale=.5, id=f'note-{i}'))
        labels = scene.create_layer('labels', z_order=10)
        _label(labels, 'sun', 0, 263, 57, 'Sun')
        _label(labels, 'earth', 284, 413, 70, 'Earth')
        labels.add(Line(id='moon-leader', start=Point(742, 262), end=Point(889, 197.15890410958903),
            stroke=StrokeStyle(width=2, color=Color(75, 104, 140))))
        _label(labels, 'moon', 889, 159.5, 71, 'Moon')
    return scene


def build_scene(case):
    if case in LESSON_CASES:
        return build_lesson(case)
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
