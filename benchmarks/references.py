"""Save and compare exact reference renders.

    python -m benchmarks.references save DIR            # benchmark scenes -> DIR/*.png
    python -m benchmarks.references compare DIR         # re-render and diff against DIR
    python -m benchmarks.references compare-dirs A B    # diff two capture directories

Test-suite renders are captured with the companion pytest plugin:

    DRAWCV_CAPTURE_DIR=DIR python -m pytest -p benchmarks.capture_renders

Comparisons report, per image, the maximum channel difference, how many pixels
differ and the bounding box of the differing pixels. Any difference fails.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from drawcv import OpenCVRenderer
from benchmarks.scenes import BGR_CASES, CASES, TIMES, build_scene


def render_cases(cases=CASES):
    renderer = OpenCVRenderer()
    for case in cases:
        scene = build_scene(case)
        alpha = case not in BGR_CASES
        if case == 'animation':
            for i, t in enumerate(TIMES):
                yield f'{case}_{i}', scene.render_at_time(t, renderer, alpha=alpha).buffer
        else:
            yield case, renderer.render(scene, alpha=alpha).buffer


def diff(expected, actual):
    """Return None when identical, otherwise a description of the difference."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return f'shape/dtype {expected.shape}/{expected.dtype} != {actual.shape}/{actual.dtype}'
    if np.array_equal(expected, actual):
        return None
    delta = np.abs(expected.astype(np.int32) - actual.astype(np.int32))
    per_pixel = delta.max(axis=-1) if delta.ndim == 3 else delta
    ys, xs = np.nonzero(per_pixel)
    worst = np.unravel_index(np.argmax(per_pixel), per_pixel.shape)
    return (f'max channel diff {int(delta.max())} at (x={worst[-1]}, y={worst[0]}), '
            f'{len(xs)} pixels differ in x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}')


def load_capture(path):
    if path.suffix == '.png':
        return {'image': cv2.imread(str(path), cv2.IMREAD_UNCHANGED)}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compare_dirs(reference, actual):
    failures = checked = 0
    for ref_path in sorted(reference.iterdir()):
        if ref_path.suffix not in ('.png', '.npz'):
            continue
        other = actual / ref_path.name
        if not other.exists():
            print(f'MISSING {ref_path.name}')
            failures += 1
            continue
        expected, got = load_capture(ref_path), load_capture(other)
        if expected.keys() != got.keys():
            print(f'DIFF {ref_path.name}: render count {len(expected)} != {len(got)}')
            failures += 1
            continue
        for key in expected:
            checked += 1
            problem = diff(expected[key], got[key])
            if problem:
                print(f'DIFF {ref_path.name}[{key}]: {problem}')
                failures += 1
    extra = {p.name for p in actual.iterdir()} - {p.name for p in reference.iterdir()}
    for name in sorted(extra):
        print(f'EXTRA {name}')
    print(f'{checked} images compared, {failures} differences, {len(extra)} unexpected files')
    return failures == 0 and not extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('save', 'compare', 'compare-dirs'))
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.command == 'save':
        args.paths[0].mkdir(parents=True, exist_ok=True)
        for name, pixels in render_cases(args.cases):
            cv2.imwrite(str(args.paths[0] / f'{name}.png'), pixels)
        print(f'saved to {args.paths[0]}')
        return
    if args.command == 'compare':
        failures = 0
        for name, pixels in render_cases(args.cases):
            problem = diff(cv2.imread(str(args.paths[0] / f'{name}.png'), cv2.IMREAD_UNCHANGED), pixels)
            print(f'{name}: {problem or "identical"}')
            failures += problem is not None
        sys.exit(1 if failures else 0)
    sys.exit(0 if compare_dirs(*args.paths[:2]) else 1)


if __name__ == '__main__':
    main()
