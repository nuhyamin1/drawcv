"""Micro-benchmarks for DrawCV Advanced Raster Effects.

Measures throughput (renders/sec and latency in ms) across:
- Convolution (3x3, 5x5, 9x9)
- Sharpen (unsharp mask)
- Emboss (directional luminance relief)
- Edge Detection (Sobel vs Laplacian)
- Deterministic Noise / Film Grain (monochrome vs multi-channel)
- Displacement Mapping (Nearest vs Linear vs Cubic)
- 512x512 canvas and 1920x1080 full HD canvas
"""

import time
import numpy as np

from drawcv import (
    Circle,
    Color,
    ConvolutionEffect,
    DisplacementChannel,
    DisplacementMapEffect,
    EdgeDetectionEffect,
    EdgeDetectionMethod,
    EmbossEffect,
    FillStyle,
    ImageInterpolation,
    ImagePaint,
    NoiseEffect,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    SharpenEffect,
)


def make_displacement_texture(w=128, h=128):
    """Generate smooth sine-wave displacement texture."""
    ys, xs = np.indices((h, w), dtype=np.float32)
    r = np.clip((np.sin(xs * 0.1) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((np.cos(ys * 0.1) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    a = np.full((h, w), 255, dtype=np.uint8)
    return np.stack([b, g, r, a], axis=-1)


def benchmark_case(name: str, scene: Scene, renderer: OpenCVRenderer, iterations: int = 40) -> dict:
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
    disp_tex = make_displacement_texture(128, 128)
    disp_paint = ImagePaint(image=disp_tex, repeat="repeat")

    results = []

    # 1. 512x512 Canvas Effects Benchmarks
    k3 = [
        [0.0, -1.0, 0.0],
        [-1.0, 5.0, -1.0],
        [0.0, -1.0, 0.0],
    ]
    k5 = [[1.0 / 25.0] * 5 for _ in range(5)]
    k9 = [[1.0 / 81.0] * 9 for _ in range(9)]

    cases_512 = [
        ("Baseline (No Effects, 512x512)", []),
        ("Convolution 3x3", [ConvolutionEffect(kernel=k3)]),
        ("Convolution 5x5", [ConvolutionEffect(kernel=k5)]),
        ("Convolution 9x9", [ConvolutionEffect(kernel=k9)]),
        ("Sharpen (amount=1.5, r=1.5)", [SharpenEffect(amount=1.5, radius=1.5)]),
        ("Emboss (strength=1.5, angle=135)", [EmbossEffect(strength=1.5, angle=135.0)]),
        ("Edge Detection (Sobel)", [EdgeDetectionEffect(method=EdgeDetectionMethod.SOBEL)]),
        ("Edge Detection (Laplacian)", [EdgeDetectionEffect(method=EdgeDetectionMethod.LAPLACIAN)]),
        ("Noise (monochrome)", [NoiseEffect(amount=0.15, seed=42, monochrome=True)]),
        ("Noise (color channels)", [NoiseEffect(amount=0.15, seed=42, monochrome=False)]),
        ("Displacement (Nearest)", [DisplacementMapEffect(map=disp_paint, scale_x=15.0, scale_y=15.0, interpolation=ImageInterpolation.NEAREST)]),
        ("Displacement (Linear)", [DisplacementMapEffect(map=disp_paint, scale_x=15.0, scale_y=15.0, interpolation=ImageInterpolation.LINEAR)]),
        ("Displacement (Cubic)", [DisplacementMapEffect(map=disp_paint, scale_x=15.0, scale_y=15.0, interpolation=ImageInterpolation.CUBIC)]),
        ("Stacked Pipeline (Noise -> Sharpen -> Emboss)", [
            NoiseEffect(amount=0.1, seed=1),
            SharpenEffect(amount=1.0, radius=1.0),
            EmbossEffect(strength=1.0, angle=135.0),
        ]),
    ]

    print("=" * 72)
    print(f"{'DrawCV Advanced Raster Effects Benchmark (512x512)':^72}")
    print("=" * 72)
    print(f"{'Benchmark Case':<48} | {'Avg (ms)':<10} | {'FPS':<8}")
    print("-" * 72)

    for label, effects in cases_512:
        scene = Scene(512, 512, background=Color(240, 240, 245, 1.0))
        rect = Rectangle(
            position=Point(56, 56),
            width=400,
            height=400,
            fill=FillStyle(color=Color(100, 150, 220, 1.0)),
        )
        rect.effects = effects
        scene.add(rect)
        res = benchmark_case(label, scene, renderer, iterations=30)
        results.append(res)
        print(f"{res['name']:<48} | {res['avg_ms']:>8.2f}ms | {res['fps']:>8.1f}")

    # 2. 1920x1080 HD Benchmark Cases
    print("\n" + "=" * 72)
    print(f"{'DrawCV HD Raster Effects Benchmark (1920x1080)':^72}")
    print("=" * 72)
    print(f"{'Benchmark Case':<48} | {'Avg (ms)':<10} | {'FPS':<8}")
    print("-" * 72)

    cases_hd = [
        ("HD Baseline (No Effects)", []),
        ("HD Sharpen (amount=1.5)", [SharpenEffect(amount=1.5, radius=2.0)]),
        ("HD Emboss", [EmbossEffect(strength=1.5, angle=135.0)]),
        ("HD Deterministic Noise", [NoiseEffect(amount=0.1, seed=42)]),
        ("HD Displacement (Linear)", [DisplacementMapEffect(map=disp_paint, scale_x=20.0, scale_y=20.0)]),
    ]

    for label, effects in cases_hd:
        scene = Scene(1920, 1080, background=Color(30, 30, 35, 1.0))
        rect = Rectangle(
            position=Point(160, 90),
            width=1600,
            height=900,
            fill=FillStyle(color=Color(80, 130, 200, 1.0)),
        )
        rect.effects = effects
        scene.add(rect)
        res = benchmark_case(label, scene, renderer, iterations=15)
        results.append(res)
        print(f"{res['name']:<48} | {res['avg_ms']:>8.2f}ms | {res['fps']:>8.1f}")

    print("=" * 72)
    return results


if __name__ == "__main__":
    run_benchmarks()
