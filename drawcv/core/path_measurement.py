"""Arc-length queries on semantic curves, independent of rendering resolution."""
import math

import numpy as np

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import svg_arc_to_center_parameterization
from drawcv.shapes.path import MoveTo, LineTo, QuadraticTo, CubicTo, EllipticalArcTo, Close


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def _integral(speed, a, b, tolerance, depth=0):
    def gauss(lo, hi):
        half = (hi - lo) / 2
        return half * sum(w * speed((hi + lo) / 2 + half * n) for n, w in zip(_NODES, _WEIGHTS))
    middle = (a + b) / 2
    whole = gauss(a, b)
    split = gauss(a, middle) + gauss(middle, b)
    if abs(split - whole) <= tolerance or depth >= 18:
        return float(split)
    return (_integral(speed, a, middle, tolerance / 2, depth + 1)
            + _integral(speed, middle, b, tolerance / 2, depth + 1))


def _parameter_at_length(speed, length, distance, tolerance):
    """Invert the same arc-length model used by queries and path editing."""
    if distance <= 0:
        return 0.
    if distance >= length:
        return 1.
    lo, hi = 0., 1.
    for _ in range(40):
        mid = (lo + hi) / 2
        if _integral(speed, 0, mid, tolerance / 4) < distance:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _curve(start, cmd, matrix):
    def vector(p):
        return np.array([p.x, p.y], dtype=float)
    p0 = vector(start)
    end = cmd.point if isinstance(cmd, LineTo) else cmd.end
    p1 = vector(end)
    if isinstance(cmd, LineTo) or (isinstance(cmd, EllipticalArcTo) and start == end):
        value = lambda t: p0 * (1-t) + p1 * t
        derivative = lambda t: p1-p0
    elif isinstance(cmd, QuadraticTo):
        c = vector(cmd.control)
        value = lambda t: (1-t)**2*p0 + 2*(1-t)*t*c + t*t*p1
        derivative = lambda t: 2*((1-t)*(c-p0) + t*(p1-c))
    elif isinstance(cmd, CubicTo):
        c, d = vector(cmd.control1), vector(cmd.control2)
        value = lambda t: (1-t)**3*p0 + 3*(1-t)**2*t*c + 3*(1-t)*t*t*d + t**3*p1
        derivative = lambda t: 3*((1-t)**2*(c-p0) + 2*(1-t)*t*(d-c) + t*t*(p1-d))
    else:
        center, rx, ry, phi, theta, sweep = svg_arc_to_center_parameterization(
            start, end, cmd.radius_x, cmd.radius_y, cmd.x_axis_rotation, cmd.large_arc, cmd.sweep)
        rotation = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        center = vector(center)
        value = lambda t: center + rotation @ np.array([rx*math.cos(theta+t*sweep), ry*math.sin(theta+t*sweep)])
        derivative = lambda t: rotation @ np.array([-rx*sweep*math.sin(theta+t*sweep), ry*sweep*math.cos(theta+t*sweep)])
    linear, translation = matrix[:2, :2], matrix[:2, 2]
    def position(t):
        # Preserve authored endpoints exactly, including degenerate SVG arcs.
        return linear @ (p0 if t == 0 else p1 if t == 1 else value(t)) + translation
    return position, lambda t: linear @ derivative(t)


class Measurement:
    def __init__(self, path, *, tolerance, space, subpath):
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValidationError('tolerance must be a finite positive number')
        if space not in ('local', 'world'):
            raise ValidationError("space must be 'local' or 'world'")
        if subpath is not None and (isinstance(subpath, bool) or not isinstance(subpath, int) or not 0 <= subpath < len(path.subpaths)):
            raise ValidationError('subpath must be a valid non-negative subpath index')
        matrix = np.eye(3) if space == 'local' else path.world_matrix
        selected = path.subpaths if subpath is None else [path.subpaths[subpath]]
        self.segments = []
        self.first = None
        self.last = None
        self.tolerance = tolerance / max(1, sum(len(s.commands) + 1 for s in selected))
        for contour in selected:
            current = start = Point(0, 0)
            active = False
            def remember(p):
                mapped = matrix @ np.array([p.x, p.y, 1.])
                if self.first is None:
                    self.first = mapped[:2]
                self.last = mapped[:2]
            def segment(cmd):
                position, derivative = _curve(current, cmd, matrix)
                speed = lambda t: float(np.linalg.norm(derivative(t)))
                length = speed(0) if isinstance(cmd, LineTo) else _integral(speed, 0, 1, self.tolerance)
                if length > 0:
                    self.segments.append((position, derivative, speed, length))
            for cmd in contour.commands:
                if isinstance(cmd, MoveTo):
                    current = start = cmd.point
                    active = True
                    remember(current)
                elif isinstance(cmd, Close):
                    if active:
                        segment(LineTo(start))
                        current = start
                        remember(current)
                elif isinstance(cmd, (LineTo, QuadraticTo, CubicTo, EllipticalArcTo)):
                    if not active:
                        remember(current)
                        active = True
                    segment(cmd)
                    current = cmd.point if isinstance(cmd, LineTo) else cmd.end
                    remember(current)
                else:
                    raise ValidationError('Unsupported path command for measurement')
            if contour.closed and active and current != start:
                segment(LineTo(start))
                remember(start)
        self.length = math.fsum(s[3] for s in self.segments)

    def query(self, progress, *, tangent=False):
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValidationError('progress must be a finite number in [0, 1]')
        if self.first is None:
            raise ValidationError('Cannot query an empty path')
        if not self.segments:
            if tangent:
                raise ValidationError('A zero-length path has no tangent')
            return Point(*self.first)
        target = progress * self.length
        selected = self.segments[-1]
        distance = target
        for selected in self.segments:
            if distance <= selected[3] or math.isclose(distance, selected[3], rel_tol=1e-14, abs_tol=0):
                distance = min(distance, selected[3])
                break
            distance -= selected[3]
        position, derivative, speed, length = selected
        if progress == 0:
            t = 0.
        elif progress == 1 or distance >= length:
            t = 1.
        else:
            t = _parameter_at_length(speed, length, distance, self.tolerance)
        if not tangent:
            if progress == 0:
                return Point(*self.first)
            if progress == 1:
                return Point(*self.last)
            return Point(*position(t))
        direction = derivative(t)
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            # Use an incoming one-sided direction at stationary points, outgoing
            # at the start. No invented vector for a completely collapsed path.
            for step in (1e-7, 1e-5, 1e-3):
                direction = derivative(max(0., t-step) if t > 0 else min(1., t+step))
                norm = float(np.linalg.norm(direction))
                if norm > 0:
                    break
        if norm == 0:
            raise ValidationError('No tangent exists at this position')
        return Point(*(direction / norm))
