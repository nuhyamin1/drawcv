"""Stroke tessellation shared by all geometric outlines.

Contours carry (x, y, width) samples. Dashing interpolates width at cuts;
the tessellator unions segment ribbons, joins and caps into one coverage mask.
An optional affine map transforms object-space outlines before rasterization.
"""
import cv2
import numpy as np

from drawcv.core.enums import CapStyle, JoinStyle
from drawcv.core.exceptions import RenderError


def dash_contour(samples, closed, style):
    """Yield connected on-runs; restart phase for each contour."""
    pts = []
    for sample in samples:
        p = np.asarray(sample, dtype=float)
        if not pts or np.linalg.norm(p[:2] - pts[-1][:2]) > 1e-10:
            pts.append(p)
    if not pts:
        return
    if closed and len(pts) > 1 and np.linalg.norm(pts[-1][:2] - pts[0][:2]) < 1e-10:
        pts.pop()
    if not style.dash_array:
        yield pts, closed
        return
    pattern = tuple(style.dash_array)
    if len(pattern) % 2:
        pattern *= 2
    phase = style.dash_offset % sum(pattern)
    index = 0
    while phase >= pattern[index]:
        phase -= pattern[index]
        index = (index + 1) % len(pattern)
    remaining = pattern[index] - phase
    if len(pts) == 1:
        if index % 2 == 0:
            yield pts, False
        return
    walk = pts + [pts[0]] if closed else pts
    runs, run = [], []
    for a, b in zip(walk, walk[1:]):
        length = float(np.linalg.norm(b[:2] - a[:2]))
        distance = 0.0
        while distance < length:
            step = min(remaining, length - distance)
            if distance + step == distance:
                raise RenderError("Dash lengths are below coordinate precision")
            end = distance + step
            if index % 2 == 0:
                if not run:
                    run.append(a + (b - a) * (distance / length))
                run.append(a + (b - a) * (end / length))
            distance = end
            remaining -= step
            if remaining <= 1e-12:
                if run:
                    runs.append(run)
                    run = []
                index = (index + 1) % len(pattern)
                remaining = pattern[index]
    if run:
        runs.append(run)
    # A closed seam inside an on-run is a join, not two overlapping caps.
    if closed and runs:
        at_start = np.linalg.norm(runs[0][0][:2] - pts[0][:2]) < 1e-9
        at_end = np.linalg.norm(runs[-1][-1][:2] - pts[0][:2]) < 1e-9
        if at_start and at_end:
            if len(runs) == 1:
                yield runs[0][:-1], True
                return
            runs = [runs[-1] + runs[0][1:]] + runs[1:-1]
    for run in runs:
        yield run, False


def stroke_mask(width, height, contours, style, line_type, transform=None):
    """Rasterize all contours into a single union mask at subpixel precision."""
    mask = np.zeros((height, width), dtype=np.uint8)
    outline_scale = float(np.linalg.norm(transform[:2, :2], ord=2)) if transform is not None else 1.0

    def polygon(points):
        points = np.asarray(points)
        if transform is not None:
            points = points @ transform[:2, :2].T + transform[:2, 2]
        pts = np.rint(points * 256).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=line_type, shift=8)

    def disk(p, radius):
        if radius <= 0:
            return
        if transform is not None:
            # Tessellate in stroke space; transform the outline, never the mask.
            # Round caps and joins become ellipses under anisotropic transforms.
            world_radius = radius * outline_scale
            angle = 2 * np.arccos(np.clip(1 - 0.125 / max(world_radius, 0.125), -1, 1))
            count = max(12, int(np.ceil(2 * np.pi / max(angle, 1e-6))))
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            polygon(p + radius * np.column_stack((np.cos(angles), np.sin(angles))))
            return
        cv2.circle(mask, tuple(np.rint(p * 256).astype(int)),
                   int(round(radius * 256)), 255, -1, line_type, shift=8)

    _stroke_geometry(contours, style, polygon, disk)
    return mask


def _stroke_geometry(contours, style, polygon, disk):
    """Emit shared ribbons, caps and joins through geometry callbacks."""
    for samples, closed in contours:
        for run, loop in dash_contour(samples, closed, style):
            points = np.asarray(run)
            xy, radii = points[:, :2], points[:, 2] / 2
            if not np.any(radii > 0):
                continue
            count = len(points)
            if count == 1:
                if style.cap_style == CapStyle.ROUND:
                    disk(xy[0], radii[0])
                elif style.cap_style == CapStyle.SQUARE:
                    polygon(xy[0] + np.array([[-1,-1],[1,-1],[1,1],[-1,1]]) * radii[0])
                continue
            edges = count if loop else count - 1
            directions, normals = [], []
            for i in range(edges):
                j = (i + 1) % count
                delta = xy[j] - xy[i]
                direction = delta / np.linalg.norm(delta)
                normal = np.array([-direction[1], direction[0]])
                directions.append(direction)
                normals.append(normal)
                polygon([xy[i] + normal*radii[i], xy[j] + normal*radii[j],
                         xy[j] - normal*radii[j], xy[i] - normal*radii[i]])
            for i in (range(count) if loop else range(1, count - 1)):
                prev, nxt = (i - 1) % edges, i % edges
                p, radius = xy[i], radii[i]
                if style.join_style == JoinStyle.ROUND:
                    disk(p, radius)
                    continue
                u, v = directions[prev], directions[nxt]
                cross = u[0]*v[1] - u[1]*v[0]
                if abs(cross) < 1e-10:
                    continue
                side = -1 if cross > 0 else 1
                a = p + side*normals[prev]*radius
                b = p + side*normals[nxt]*radius
                delta = b - a
                tip = a + u*((delta[0]*v[1] - delta[1]*v[0])/cross)
                if style.join_style == JoinStyle.MITER and np.linalg.norm(tip-p) <= style.miter_limit*radius:
                    polygon([p, a, tip, b])
                else:
                    polygon([p, a, b])
            if not loop:
                for i, direction in ((0, -directions[0]), (-1, directions[-1])):
                    p, radius = xy[i], radii[i]
                    if style.cap_style == CapStyle.ROUND:
                        disk(p, radius)
                    elif style.cap_style == CapStyle.SQUARE:
                        n = np.array([-direction[1], direction[0]])*radius
                        extension = direction*radius
                        polygon([p+n, p+extension+n, p+extension-n, p-n])
