"""Render the measured fixtures as a contact sheet, outside benchmark timing."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from drawcv import OpenCVRenderer
from benchmarks.scenes import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=Path('examples/output/performance_gallery.png'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    cv2.setNumThreads(report['environment']['opencv_threads'])
    tiles = []
    for item in report['results']:
        scene = build_scene(item['case'])
        renderer = OpenCVRenderer()
        pixels = (scene.render_at_time(0, renderer, alpha=item['alpha']).buffer if item['case'] == 'animation'
                  else renderer.render(scene, alpha=item['alpha']).buffer)
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == item['frame_hashes'][0]
        if item['alpha']:
            yy, xx = np.indices(pixels.shape[:2])
            bg = np.repeat(np.where(((xx//16+yy//16)%2)[..., None], 48, 38), 3, axis=2)
            a = pixels[..., 3:4].astype(float)/255
            pixels = np.rint(pixels[..., :3]*a + bg*(1-a)).astype(np.uint8)
        tile = np.full((344, 480, 3), (26, 22, 20), np.uint8)
        tile[64:334] = cv2.resize(pixels, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, item['case'].replace('_', ' ').upper(), (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (235, 230, 225), 1, cv2.LINE_AA)
        line = f"Median {item['median_ms']:.1f} ms"
        if 'median_speedup' in item:
            line += f"  |  {item['median_speedup']:.2f}x before/after"
        cv2.putText(tile, line, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .42, (170, 210, 150), 1, cv2.LINE_AA)
        tiles.append(tile)
    if len(tiles)%2:
        tiles.append(np.zeros_like(tiles[0]))
    canvas = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f'Could not save {args.output}')


if __name__ == '__main__':
    main()
