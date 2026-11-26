"""Arc-length queries on semantic curves, independent of rendering resolution."""
from bisect import bisect_left
import math

import numpy as np

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import svg_arc_to_center_parameterization
from drawcv.shapes.path import MoveTo, LineTo, QuadraticTo, CubicTo, EllipticalArcTo, Close


def _parameter_at_length(table, length, distance):
    if distance <= 0:
        return 0.
    if distance >= length:
        return 1.
    return table.parameter(distance) if table is not None else distance / length


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


class _ArcTable:
    """Integrate a piecewise quadratic speed model once, then invert it locally."""
    def __init__(self, speed, tolerance):
        self.ends = []
        self.intervals = []
        self.length = 0.

        def build(a, b, left, mid, right, depth):
            width = b-a
            q1, q3 = speed(a+width/4), speed(b-width/4)
            # Compare the quadratic interpolant against fresh quarter samples.
            error = max(abs(q1-(3*left+6*mid-right)/8),
                        abs(q3-(3*right+6*mid-left)/8))
            if depth < 18 and error > tolerance:
                center = (a+b)/2
                build(a, center, left, q1, mid, depth+1)
                build(center, b, mid, q3, right, depth+1)
                return
            # Integral from a to a+width*u is u*(c0+u*(c1+u*c2)).
            c0 = width*left
            c1 = width*(4*mid-3*left-right)/2
            c2 = width*(2*left+2*right-4*mid)/3
            self.intervals.append((a, width, c0, c1, c2))
            self.length += width*(left+4*mid+right)/6
            self.ends.append(self.length)

        build(0., 1., speed(0.), speed(.5), speed(1.), 0)

    def parameter(self, distance):
        index = bisect_left(self.ends, distance)
        a, width, c0, c1, c2 = self.intervals[index]
        target = distance - (self.ends[index-1] if index else 0.)
        lo, hi = 0., 1.
        for _ in range(36):
            u = (lo+hi)/2
            if u*(c0+u*(c1+u*c2)) < target:
                lo = u
            else:
                hi = u
        return a + width*(lo+hi)/2


def _validate_options(path, tolerance, space, subpath):
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValidationError('tolerance must be a finite positive number')
    if space not in ('local', 'world'):
        raise ValidationError("space must be 'local' or 'world'")
    if subpath is not None and (isinstance(subpath, bool) or not isinstance(subpath, int) or not 0 <= subpath < len(path.subpaths)):
        raise ValidationError('subpath must be a valid non-negative subpath index')


def measurement_for(path, tolerance, space, subpath):
    _validate_options(path, tolerance, space, subpath)
    selected = path.subpaths if subpath is None else (path.subpaths[subpath],)
    geometry = tuple((tuple(s.commands), s.closed) for s in selected)
    matrix = path.world_matrix if space == "world" else None
    key = (tolerance, space, subpath, geometry, matrix.tobytes() if matrix is not None else None)
    cached = getattr(path, "_measurement_cache", None)
    if cached is None or cached[0] != key:
        cached = key, Measurement(selected, tolerance, matrix if matrix is not None else np.eye(3))
        path._measurement_cache = cached
    return cached[1]


class Measurement:
    def __init__(self, selected, tolerance, matrix):
        self.segments = []
        self.command_lengths = []
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
                table = None if isinstance(cmd, LineTo) else _ArcTable(speed, self.tolerance)
                length = table.length if table is not None else speed(0)
                self.command_lengths.append((table, length))
                if length > 0:
                    self.segments.append((position, derivative, table, length))
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
        self.ends = []
        self.length = 0.
        for _, _, _, length in self.segments:
            self.length += length
            self.ends.append(self.length)

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
        index = bisect_left(self.ends, target)
        if index and math.isclose(target, self.ends[index-1], rel_tol=1e-14):
            index -= 1
        position, derivative, table, length = self.segments[index]
        distance = target - (self.ends[index-1] if index else 0.)
        if progress == 0:
            t = 0.
        elif progress == 1 or distance >= length:
            t = 1.
        else:
            t = _parameter_at_length(table, length, distance)
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
