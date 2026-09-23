"""Area-exact antialiased polygon coverage with nonzero and even-odd fill rules.

Pixel (i, j) is the square [j, j + 1] x [i, i + 1], as in SVG, so shapes on
whole-pixel coordinates have crisp edges. Coverage is sampled on horizontal
sub-rows through each pixel. On every sub-row the winding number along sorted
edge crossings gives the exact inside spans, so overlapping or self-intersecting
contours follow the fill rule instead of accumulating. Each span contributes its
exact horizontal extent, including fractional end pixels. Edges are straight, so
evaluating crossings at sub-row centers integrates the area exactly within a
sub-row; only structure finer than 1/subrows px vertically is approximated.

OpenCV's antialiased fillPoly instead draws a one-pixel antialiased line around
a binary fill, adding about 0.7 px of coverage outside every edge.
"""
from __future__ import annotations

import math

import numpy as np

SUBROWS = 4


def coverage_mask(contours, width: int, height: int, *, even_odd: bool = False,
                  subrows: int = SUBROWS, out: np.ndarray | None = None) -> np.ndarray:
    """Rasterize closed contours into a uint8 (height, width) coverage mask.

    contours: sequences of (x, y) vertices in world pixels; each is closed
    implicitly. Nonzero winding is the default, so contours sharing an
    orientation form their exact union.
    """
    mask = out if out is not None else np.zeros((height, width), dtype=np.uint8)
    # A contour with a non-finite vertex is skipped whole: partial contours would
    # leave the winding unbalanced.
    arrays = [a for a in (np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in contours)
              if len(a) >= 3 and np.isfinite(a).all()]
    if not arrays or width <= 0 or height <= 0:
        return mask
    starts = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    if len(arrays) == 1 and len(starts) == 4 and _axis_aligned_rectangle(starts):
        return _rectangle_coverage(starts, width, height, mask, out is not None)
    # Each vertex connects to the next one of its own contour; the last closes it.
    following = np.arange(1, len(starts) + 1)
    lengths = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    closing = np.cumsum(lengths) - 1
    following[closing] = closing - lengths + 1
    return edge_coverage(starts, starts[following], width, height, even_odd=even_odd,
                         subrows=subrows, out=mask, merge=out is not None)


def edge_coverage(starts, ends, width: int, height: int, *, even_odd: bool = False,
                  subrows: int = SUBROWS, out: np.ndarray | None = None, merge: bool = True) -> np.ndarray:
    """Coverage of finite directed edges (N, 2) -> (N, 2) forming closed contours.

    Without ``out`` a new mask is returned; with it, ``merge`` keeps the brighter
    of the existing and new coverage.
    """
    mask = out if out is not None else np.zeros((height, width), dtype=np.uint8)
    if not len(starts) or width <= 0 or height <= 0:
        return mask
    x0, y0, x1, y1 = starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]

    row0 = max(0, math.floor(min(y0.min(), y1.min())))
    row1 = min(height, math.ceil(max(y0.max(), y1.max())))
    col0 = max(0, math.floor(min(x0.min(), x1.min())))
    col1 = min(width, math.ceil(max(x0.max(), x1.max())))
    if row1 <= row0 or col1 <= col0:
        return mask

    # Sub-row centers c = (k + 0.5) / S; an edge crosses those with lo <= c < hi.
    s = subrows
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    first = np.maximum(np.ceil(lo * s - 0.5), row0 * s).astype(np.int64)
    last = np.minimum(np.ceil(hi * s - 0.5), row1 * s).astype(np.int64)
    counts = np.maximum(last - first, 0)
    total = int(counts.sum())
    if total == 0:
        return mask
    edge = np.repeat(np.arange(len(counts)), counts)
    k = first[edge] + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
    center = (k + 0.5) / s
    ey0, ey1 = y0[edge], y1[edge]
    x = x0[edge] + (center - ey0) * (x1[edge] - x0[edge]) / (ey1 - ey0)
    direction = np.where(ey1 > ey0, 1, -1)

    order = np.lexsort((x, k))
    k, x, direction = k[order], x[order], direction[order]
    # Every closed contour crosses a sub-row equally often in both directions,
    # so the running sum restarts at zero on each sub-row.
    after = np.cumsum(direction)
    before = after - direction
    if even_odd:
        inside_after, inside_before = (after & 1) != 0, (before & 1) != 0
    else:
        inside_after, inside_before = after != 0, before != 0
    sign = inside_after.astype(np.int8) - inside_before.astype(np.int8)
    keep = sign != 0
    k, x, sign = k[keep], x[keep], sign[keep]

    # A span edge at x covers (1 - f) of its pixel and all pixels to the right.
    cols = col1 - col0
    stride = cols + 2
    u = x - col0
    cell = np.floor(u)
    frac = u - cell
    left = cell < 0
    cell[left], frac[left] = -1, 1.0
    right = cell >= cols
    cell[right], frac[right] = cols, 0.0
    base = (k // s - row0) * stride + cell.astype(np.int64) + 1
    weight = sign * (255.0 / s)  # accumulate directly in 0..255 units
    rows = row1 - row0
    size = rows * stride
    # Coverage is piecewise constant between span-edge events and returns to zero
    # at each row end, so accumulate only at events and expand the runs.
    events, slot = np.unique(np.concatenate((base, np.minimum(base + 1, size - 1))), return_inverse=True)
    level = np.cumsum(np.bincount(slot, np.concatenate((weight * (1 - frac), weight * frac))))
    np.clip(level, 0.0, 255.0, out=level)
    level = (level + 0.5).astype(np.int16)  # round half up
    dense = np.zeros(size, dtype=np.int16)
    dense[events] = np.diff(level, prepend=0)
    np.cumsum(dense, out=dense)
    region = dense.reshape(rows, stride)[:, 1:cols + 1].astype(np.uint8)
    if merge:
        np.maximum(mask[row0:row1, col0:col1], region, out=mask[row0:row1, col0:col1])
    else:
        mask[row0:row1, col0:col1] = region
    return mask


def _axis_aligned_rectangle(points: np.ndarray) -> bool:
    """Four vertices whose edges alternate between horizontal and vertical."""
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = points.tolist()
    return ((ay == by and bx == cx and cy == dy and dx == ax)
            or (ax == bx and by == cy and cx == dx and dy == ay))


def _rectangle_coverage(points, width, height, mask, merge):
    """Exact coverage of an axis-aligned rectangle: per-column times per-row overlap."""
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    col0, col1 = max(0, math.floor(left)), min(width, math.ceil(right))
    row0, row1 = max(0, math.floor(top)), min(height, math.ceil(bottom))
    if col1 <= col0 or row1 <= row0:
        return mask
    cols = np.arange(col0, col1, dtype=np.float64)
    rows = np.arange(row0, row1, dtype=np.float64)
    across = np.clip(np.minimum(cols + 1, right) - np.maximum(cols, left), 0.0, 1.0)
    down = np.clip(np.minimum(rows + 1, bottom) - np.maximum(rows, top), 0.0, 1.0)
    region = (np.outer(down, across) * 255.0 + 0.5).astype(np.uint8)
    if merge:
        np.maximum(mask[row0:row1, col0:col1], region, out=mask[row0:row1, col0:col1])
    else:
        mask[row0:row1, col0:col1] = region
    return mask
