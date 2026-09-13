"""Export a mixed vector/raster gallery; optionally render it with resvg.

Run python -m examples.svg_export. Install examples/svg-requirements.txt for
the independent PNG and comparison contact sheet.
"""
from dataclasses import asdict
from pathlib import Path as FilePath
import json
import cv2
import numpy as np
from drawcv import (Scene, Color, Point, FillStyle, StrokeStyle, LinearGradient,
    RadialGradient, GradientStop, RoundedRectangle, Circle, Path, Polyline, Text,
    Transform, Group, ClipRect, BlurEffect, ShadowEffect, OpenCVRenderer,
    CapStyle, JoinStyle)


def build_scene():
    scene = Scene(900, 570, background=Color(0, 0, 0, 0))
    stops = (GradientStop(0, Color(250, 130, 65)), GradientStop(.5, Color(195, 65, 160)),
             GradientStop(1, Color(65, 150, 240)))
    scene.add(RoundedRectangle(x=35, y=55, width=230, height=170, corner_radius=24,
        fill=FillStyle(paint=LinearGradient(Point(35, 55), Point(265, 225), stops)),
        stroke=StrokeStyle(color=Color(225, 230, 240), width=3), name='Linear gradient'))
    hole = Path(fill=FillStyle(paint=RadialGradient(Point(420, 120), 120, stops)),
                stroke=StrokeStyle(color=Color(245, 235, 210), width=4), name='Editable curves and hole')
    hole.move_to(335, 65).cubic_to(Point(550, 20), Point(565, 250), Point(340, 220)).close()
    hole.move_to(390, 105).line_to(390, 170).line_to(450, 170).line_to(450, 105).close()
    scene.add(hole)
    stripe = RoundedRectangle(x=640, y=65, width=205, height=140, corner_radius=18,
        fill=FillStyle(paint=LinearGradient(Point(640, 65), Point(845, 205), stops, space='world')),
        stroke=StrokeStyle(color=Color.white(), width=5, dash_array=(12, 8)),
        transform=Transform(rotation=-12), clip=ClipRect(650, 45, 180, 145), name='World gradient and clip')
    scene.add(stripe)
    for index, cap in enumerate(CapStyle):
        scene.add(Polyline(points=[Point(50, 330+index*45), Point(115, 300+index*45), Point(250, 330+index*45)],
            stroke=StrokeStyle(color=Color(80+index*60, 180, 230-index*50), width=12,
                cap_style=cap, join_style=JoinStyle.ROUND, dash_array=(22, 12)), name=cap.value))
    left = Circle(center=Point(410, 360), radius=65, fill=FillStyle(color=Color(235, 75, 85)))
    right = Circle(center=Point(470, 390), radius=65, fill=FillStyle(color=Color(65, 130, 245, .7)))
    scene.add(Group(children=[left, right], opacity=.6, name='Isolated group opacity'))
    scene.add(Circle(center=Point(750, 375), radius=65,
        fill=FillStyle(paint=RadialGradient(Point(725, 350), 115, stops)),
        effects=[BlurEffect(kernel_size=9), ShadowEffect(offset_x=12, offset_y=14, blur_radius=8)],
        opacity=.8, name='Raster fallback: effects'))
    labels = ['VECTOR / LINEAR PAINT', 'VECTOR / CURVES + HOLE', 'VECTOR / CLIP + WORLD PAINT',
              'VECTOR / DASHES + CAPS', 'VECTOR / GROUP OPACITY', 'PNG FALLBACK / BLUR + SHADOW']
    for index, label in enumerate(labels):
        scene.add(Text(label, position=Point(25+(index%3)*300, 16+(index//3)*260),
            font_scale=.42, color=Color(235, 238, 245), name=label))
    scene.add(Text('Editable paths and paints. Embedded PNGs preserve unsupported rendering.',
        position=Point(25, 520), font_scale=.5, color=Color(220, 225, 235)))
    return scene


def flatten(pixels, background):
    alpha = pixels[..., 3:4].astype(float)/255
    return np.rint(pixels[..., :3]*alpha + background*(1-alpha)).clip(0, 255).astype(np.uint8)


def main():
    output = FilePath(__file__).parent / 'output'
    output.mkdir(exist_ok=True)
    scene = build_scene()
    result = scene.save_svg(output / 'svg_export.svg')
    scene.save_json(output / 'svg_export.json')
    assert Scene.from_json(scene.to_json()).to_svg() == result.svg
    reference = OpenCVRenderer().render(scene, alpha=True).buffer
    cv2.imwrite(str(output / 'svg_export_drawcv.png'), reference)
    report = {'fallbacks': [asdict(item) for item in result.fallbacks]}
    try:
        import resvg_py
    except ImportError:
        print('Install examples/svg-requirements.txt for independent previews.')
    else:
        rendered = resvg_py.svg_to_bytes(svg_string=result.svg, skip_system_fonts=True)
        (output / 'svg_export_resvg.png').write_bytes(rendered)
        decoded = cv2.imdecode(np.frombuffer(rendered, np.uint8), cv2.IMREAD_UNCHANGED)
        yy, xx = np.indices((scene.height, scene.width))
        checker = np.repeat(np.where(((xx//20+yy//20)%2)[..., None], 49, 40), 3, axis=2)
        panels = [flatten(reference, checker), flatten(decoded, checker)]
        for panel, title in zip(panels, ['DrawCV / reference', 'SVG / resvg independent renderer']):
            cv2.putText(panel, title, (25, 561), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 225, 235), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / 'svg_export_comparison.png'), np.hstack(panels))
        cv2.imwrite(str(output / 'svg_export_preview.png'), panels[1])
        # Antialiasing boundaries differ. Store aggregate error without presenting
        # it as a pixel-identical vector rendering guarantee.
        diff = np.abs(flatten(reference, np.array([40, 40, 40])).astype(float)
                      - flatten(decoded, np.array([40, 40, 40])).astype(float))
        report.update(reference_renderer='resvg-py 0.5.0', mean_absolute_channel_error=float(diff.mean()),
                      percentile_99_channel_error=float(np.percentile(diff, 99)))
    (output / 'svg_export_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
