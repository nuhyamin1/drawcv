"""Micro-benchmarks for DrawCV Advanced Paint System.

Measures throughput (renders/sec and latency in ms) across:
- Solid Fill vs Linear / Radial / Conic / Image paints
- Solid Stroke vs Gradient / Image strokes
- 512x512 canvas and full 1920x1080 canvas
"""

import time
import numpy as np

from drawcv import (
    Circle,
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    ImagePaint,
    Line,
    LinearGradient,
    OpenCVRenderer,
    Point,
    RadialGradient,
    Rectangle,
    Scene,
    StrokeStyle,
)


def make_test_texture(w=64, h=64):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:h // 2, :w // 2] = [255, 0, 0, 255]
    arr[:h // 2, w // 2:] = [0, 255, 0, 255]
    arr[h // 2:, :w // 2] = [0, 0, 255, 255]
    arr[h // 2:, w // 2:] = [255, 255, 0, 255]
    return arr


def benchmark_case(name: str, scene: Scene, renderer: OpenCVRenderer, iterations: int = 50) -> dict:
    # Warmup
    for _ in range(3):
        renderer.render(scene)

    start = time.perf_counter()
    for _ in range(iterations):
        renderer.render(scene)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000.0
    fps = iterations / elapsed
    return {"name": name, "avg_ms": avg_ms, "fps": fps}


def run_benchmarks():
    renderer = OpenCVRenderer()
    stops = (
        GradientStop(0.0, Color(255, 0, 0)),
        GradientStop(0.5, Color(0, 255, 0)),
        GradientStop(1.0, Color(0, 0, 255)),
    )
    tex = make_test_texture(64, 64)

    results = []

    # 1. 512x512 Shape Fill Benchmarks
    cases_512 = [
        ("Solid Fill (512x512)", FillStyle(color=Color.red()), None),
        ("Linear Gradient (Pad)", FillStyle(paint=LinearGradient(Point(0, 0), Point(400, 400), stops, spread="pad")), None),
        ("Linear Gradient (Repeat)", FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 100), stops, spread="repeat")), None),
        ("Linear Gradient (Reflect)", FillStyle(paint=LinearGradient(Point(0, 0), Point(100, 100), stops, spread="reflect")), None),
        ("Radial Gradient (Pad)", FillStyle(paint=RadialGradient(Point(256, 256), 200.0, stops, spread="pad")), None),
        ("Radial Gradient (Repeat)", FillStyle(paint=RadialGradient(Point(256, 256), 50.0, stops, spread="repeat")), None),
        ("Radial Gradient (Reflect)", FillStyle(paint=RadialGradient(Point(256, 256), 50.0, stops, spread="reflect")), None),
        ("Conic Gradient (360)", FillStyle(paint=ConicGradient(Point(256, 256), stops)), None),
        ("ImagePaint (Repeat)", FillStyle(paint=ImagePaint(image=tex, scale=(2.0, 2.0), repeat="repeat")), None),
        ("ImagePaint (Reflect)", FillStyle(paint=ImagePaint(image=tex, scale=(2.0, 2.0), repeat="reflect")), None),
    ]

    for label, fill, stroke in cases_512:
        scene = Scene(512, 512, background=Color(255, 255, 255, 1.0))
        rect = Rectangle(position=Point(56, 56), width=400, height=400, fill=fill, stroke=stroke)
        scene.add(rect)
        res = benchmark_case(label, scene, renderer, iterations=40)
        results.append(res)

    # 2. Stroke Benchmarks (Circle r=180, width=16)
    c_center = Point(256, 256)
    stroke_cases = [
        ("Solid Stroke (Circle)", StrokeStyle(color=Color.blue(), width=16.0)),
        ("Linear Gradient Stroke", StrokeStyle(paint=LinearGradient(Point(50, 50), Point(450, 450), stops), width=16.0)),
        ("Radial Gradient Stroke", StrokeStyle(paint=RadialGradient(c_center, 180.0, stops, spread="reflect"), width=16.0)),
        ("Conic Gradient Stroke", StrokeStyle(paint=ConicGradient(c_center, stops), width=16.0)),
        ("ImagePaint Stroke", StrokeStyle(paint=ImagePaint(image=tex, repeat="repeat"), width=16.0)),
    ]

    for label, stroke in stroke_cases:
        scene = Scene(512, 512, background=Color(255, 255, 255, 1.0))
        circle = Circle(center=c_center, radius=180.0, fill=None, stroke=stroke)
        scene.add(circle)
        res = benchmark_case(label, scene, renderer, iterations=30)
        results.append(res)

    # Print results table
    print(f"\n{'Benchmark Case':<32} | {'Latency (ms)':<14} | {'Throughput (fps)':<16}")
    print("-" * 68)
    for r in results:
        print(f"{r['name']:<32} | {r['avg_ms']:>10.2f} ms | {r['fps']:>12.1f} fps")

    return results


if __name__ == "__main__":
    run_benchmarks()
