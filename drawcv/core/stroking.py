"""Stroke tessellation shared by all geometric outlines.

Contours carry (x, y, width) samples. Dashing interpolates width at cuts;
the tessellator unions segment ribbons, joins and caps into one coverage mask.
An optional affine map transforms object-space outlines before rasterization.
"""
import functools
import math

import cv2
import numpy as np

from drawcv.core.coverage import coverage_mask, edge_coverage
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
    """Rasterize all contours into a single union mask at subpixel precision.

    Antialiased strokes collect every ribbon, join and cap piece with one
    orientation and rasterize their nonzero union with area-exact coverage, so
    overlaps neither accumulate nor leave seams. Aliased line types keep
    OpenCV's binary polygon fill.
    """
    if line_type == cv2.LINE_AA:
        return _exact_stroke_mask(width, height, contours, style, transform)
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


@functools.lru_cache(maxsize=256)
def _unit_circle(count):
    """Counter-clockwise unit circle vertices; read-only and shared."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    circle.flags.writeable = False
    return circle


# Up to this many samples, per-piece construction beats array setup overhead.
_SMALL_STROKE = 32


def _small_stroke_mask(width, height, runs, style, transform):
    """The same pieces as _exact_stroke_mask, emitted one at a time."""
    pieces = []
    outline_scale = float(np.linalg.norm(transform[:2, :2], ord=2)) if transform is not None else 1.0

    def polygon(points):
        points = np.asarray(points, dtype=np.float64)
        if transform is not None:
            points = points @ transform[:2, :2].T + transform[:2, 2]
        x, y = points[:, 0], points[:, 1]
        area = x[:-1] @ y[1:] - x[1:] @ y[:-1] + x[-1] * y[0] - x[0] * y[-1]
        if area != 0:
            pieces.append(points if area > 0 else points[::-1])

    def disk(p, radius):
        if radius > 0:
            world = max(radius * outline_scale, 0.125)
            angle = 2 * math.acos(min(1.0, max(-1.0, 1 - 0.125 / world)))
            polygon(p + radius * _unit_circle(max(12, math.ceil(2 * math.pi / max(angle, 1e-6)))))

    _run_geometry(runs, style, polygon, disk)
    return coverage_mask(pieces, width, height)


def _exact_stroke_mask(width, height, contours, style, transform):
    """Area-exact union of the pieces _stroke_geometry emits, built for all runs at once."""
    runs = [(np.asarray(run, dtype=np.float64).reshape(-1, 3), loop)
            for samples, closed in contours for run, loop in dash_contour(samples, closed, style)]
    runs = [(points, loop) for points, loop in runs if np.any(points[:, 2] > 0)]
    if not runs:
        return np.zeros((height, width), dtype=np.uint8)
    if sum(len(points) for points, _ in runs) <= _SMALL_STROKE:
        return _small_stroke_mask(width, height, runs, style, transform)
    points = np.concatenate([points for points, _ in runs])
    xy, radii = points[:, :2], points[:, 2] / 2
    length = np.fromiter((len(points) for points, _ in runs), dtype=np.int64, count=len(runs))
    loop = np.fromiter((loop for _, loop in runs), dtype=bool, count=len(runs))
    start = np.cumsum(length) - length
    single = length == 1
    groups = []  # (M, K, 2) arrays of K-vertex pieces
    disks = [np.zeros((0, 2)), np.zeros(0)]

    def spread(counts):
        """Run index and position within the run for counts[r] items per run."""
        owner = np.repeat(np.arange(len(counts)), counts)
        return owner, np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)

    def add_disks(centers, radius):
        disks[0] = np.concatenate((disks[0], centers))
        disks[1] = np.concatenate((disks[1], radius))

    # Dots: runs with a single sample take only a cap.
    dots = start[single]
    if style.cap_style == CapStyle.ROUND:
        add_disks(xy[dots], radii[dots])
    elif style.cap_style == CapStyle.SQUARE:
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
        groups.append(xy[dots, None] + square * radii[dots, None, None])

    # Ribbons: one quad per segment; closed runs also join their last sample to the first.
    segments = np.where(single, 0, np.where(loop, length, length - 1))
    segment_start = np.cumsum(segments) - segments
    owner, local = spread(segments)
    first = start[owner] + local
    second = start[owner] + (local + 1) % length[owner]
    delta = xy[second] - xy[first]
    direction = delta / np.hypot(delta[:, 0], delta[:, 1])[:, None]
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    offset_a, offset_b = normal * radii[first, None], normal * radii[second, None]
    groups.append(np.stack((xy[first] + offset_a, xy[second] + offset_b,
                            xy[second] - offset_b, xy[first] - offset_a), axis=1))

    # Joins: every vertex of a closed run, interior vertices of an open one.
    joins = np.where(single, 0, np.where(loop, length, np.maximum(length - 2, 0)))
    owner, local = spread(joins)
    vertex = np.where(loop[owner], local, local + 1)
    previous = segment_start[owner] + (vertex - 1) % segments[owner]
    following = segment_start[owner] + vertex % segments[owner]
    p, radius = xy[start[owner] + vertex], radii[start[owner] + vertex]
    if style.join_style == JoinStyle.ROUND:
        add_disks(p, radius)
    else:
        u, v = direction[previous], direction[following]
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        turning = np.abs(cross) >= 1e-10
        p, radius, u, v, cross = p[turning], radius[turning], u[turning], v[turning], cross[turning]
        side = np.where(cross > 0, -1.0, 1.0)[:, None]
        a = p + side * normal[previous][turning] * radius[:, None]
        b = p + side * normal[following][turning] * radius[:, None]
        gap = b - a
        tip = a + u * ((gap[:, 0] * v[:, 1] - gap[:, 1] * v[:, 0]) / cross)[:, None]
        miter = np.zeros(len(p), dtype=bool)
        if style.join_style == JoinStyle.MITER:
            miter = np.hypot(*(tip - p).T) <= style.miter_limit * radius
        # Bevels repeat b; the zero-length edge contributes nothing.
        groups.append(np.stack((p, a, np.where(miter[:, None], tip, b), b), axis=1))

    # Caps at both ends of open runs.
    capped = ~loop & ~single
    if style.cap_style != CapStyle.BUTT and capped.any():
        ends = np.concatenate((start[capped], start[capped] + length[capped] - 1))
        outward = np.concatenate((-direction[segment_start[capped]],
                                  direction[segment_start[capped] + segments[capped] - 1]))
        p, radius = xy[ends], radii[ends]
        if style.cap_style == CapStyle.ROUND:
            add_disks(p, radius)
        else:
            n = np.column_stack((-outward[:, 1], outward[:, 0])) * radius[:, None]
            extension = outward * radius[:, None]
            groups.append(np.stack((p + n, p + extension + n, p + extension - n, p - n), axis=1))

    centers, disk_radii = disks
    keep = disk_radii > 0
    if keep.any():
        centers, disk_radii = centers[keep], disk_radii[keep]
        # 1/8 px chord error in world space; ellipses under anisotropic transforms.
        outline_scale = float(np.linalg.norm(transform[:2, :2], ord=2)) if transform is not None else 1.0
        world = np.maximum(disk_radii * outline_scale, 0.125)
        angle = 2 * np.arccos(np.clip(1 - 0.125 / world, -1, 1))
        counts = np.maximum(12, np.ceil(2 * np.pi / np.maximum(angle, 1e-6))).astype(int)
        for n in np.unique(counts):
            chosen = counts == n
            groups.append(centers[chosen, None] + disk_radii[chosen, None, None] * _unit_circle(int(n)))

    starts, ends = [], []
    for pieces in groups:
        if not len(pieces):
            continue
        if transform is not None:
            pieces = pieces @ transform[:2, :2].T + transform[:2, 2]
        pieces = pieces[np.isfinite(pieces).all(axis=(1, 2))]
        x, y = pieces[..., 0], pieces[..., 1]
        area = np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        # One orientation for every piece: nonzero winding is then their union.
        pieces = np.where((area < 0)[:, None, None], pieces[:, ::-1], pieces)[area != 0]
        starts.append(pieces.reshape(-1, 2))
        ends.append(np.roll(pieces, -1, axis=1).reshape(-1, 2))
    if not starts:
        return np.zeros((height, width), dtype=np.uint8)
    return edge_coverage(np.concatenate(starts), np.concatenate(ends), width, height)


def _stroke_geometry(contours, style, polygon, disk):
    """Emit shared ribbons, caps and joins through geometry callbacks."""
    _run_geometry(((np.asarray(run), loop) for samples, closed in contours
                   for run, loop in dash_contour(samples, closed, style)), style, polygon, disk)


def _run_geometry(runs, style, polygon, disk):
    """Emit ribbons, caps and joins for already dashed (points, loop) runs."""
    for points, loop in runs:
        points = np.asarray(points)
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
