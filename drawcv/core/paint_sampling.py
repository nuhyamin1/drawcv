"""Renderer-side sampling of retained gradient paint at integer pixel centers."""
import numpy as np
from drawcv.styles.paint import LinearGradient


def sample_gradient(paint, matrix, width, height, *, origin=(0, 0)):
    """Sample a rectangular region at global integer pixel centers.

    An offset grid preserves object/world paint coordinates when the renderer
    limits work to the actual coverage region. Arithmetic matches the full grid.
    """
    y, x = np.indices((height, width), dtype=float)
    x += origin[0]
    y += origin[1]
    if paint.space == "object":
        inverse = np.linalg.inv(matrix)
        x, y = (inverse[0, 0]*x + inverse[0, 1]*y + inverse[0, 2],
                inverse[1, 0]*x + inverse[1, 1]*y + inverse[1, 2])
    if isinstance(paint, LinearGradient):
        dx, dy = paint.end.x-paint.start.x, paint.end.y-paint.start.y
        t = ((x-paint.start.x)*dx + (y-paint.start.y)*dy) / (dx*dx + dy*dy)
    else:
        t = np.hypot(x-paint.center.x, y-paint.center.y) / paint.radius
    positions = np.array([s.position for s in paint.stops])
    colors = np.array([[s.color.b, s.color.g, s.color.r, s.color.a] for s in paint.stops])
    right = np.searchsorted(positions, t, side="right")
    lo = np.clip(right-1, 0, len(positions)-1)
    hi = np.clip(right, 0, len(positions)-1)
    span = positions[hi]-positions[lo]
    mix = np.zeros_like(t)
    np.divide(t-positions[lo], span, out=mix, where=span > 0)
    mix = mix.clip(0, 1)[..., None]
    samples = colors[lo]*(1-mix) + colors[hi]*mix
    samples[..., :3] *= samples[..., 3:4]
    return samples.astype(np.float32)
