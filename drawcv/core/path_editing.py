"""Non-destructive semantic path trimming by normalized arc length."""
import copy
import math

import numpy as np

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import svg_arc_to_center_parameterization
from drawcv.core.path_measurement import Measurement, _curve, _integral, _parameter_at_length
from drawcv.shapes.path import Path, Subpath, MoveTo, LineTo, QuadraticTo, CubicTo, EllipticalArcTo, Close


def _progress(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError('Trim/split progress must be a finite number in [0, 1]')


def _portion(start, cmd, a, b):
    """Return start and command for a parameter interval, retaining curve type."""
    if a == 0 and b == 1:
        return start.copy(), copy.deepcopy(cmd)
    position, _ = _curve(start, cmd, np.eye(3))
    first, last = Point(*position(a)), Point(*position(b))
    if isinstance(cmd, LineTo):
        return first, LineTo(last)
    if isinstance(cmd, EllipticalArcTo):
        _, rx, ry, _, _, sweep = svg_arc_to_center_parameterization(
            start, cmd.end, cmd.radius_x, cmd.radius_y, cmd.x_axis_rotation, cmd.large_arc, cmd.sweep)
        return first, EllipticalArcTo(rx, ry, cmd.x_axis_rotation, abs(sweep*(b-a)) > math.pi, cmd.sweep, last)
    points = [start, cmd.control, cmd.end] if isinstance(cmd, QuadraticTo) else [start, cmd.control1, cmd.control2, cmd.end]
    def split(points, t):
        levels = [points]
        while len(levels[-1]) > 1:
            levels.append([Point((1-t)*p.x+t*q.x, (1-t)*p.y+t*q.y)
                           for p, q in zip(levels[-1], levels[-1][1:])])
        return [level[0] for level in levels], [level[-1] for level in reversed(levels)]
    left, _ = split(points, b)
    _, part = split(left, a/b)
    if isinstance(cmd, QuadraticTo):
        return part[0], QuadraticTo(part[1], part[2])
    return part[0], CubicTo(part[1], part[2], part[3])


def trim(path, start, end, *, tolerance=.1, space='local', subpath=None, preserve_world_transform=False):
    _progress(start)
    _progress(end)
    if start > end:
        raise ValidationError('Trim start must not exceed end; wrapped intervals are not supported')
    measurement = Measurement(path, tolerance=tolerance, space=space, subpath=subpath)
    result = path.to_path(preserve_world_transform=preserve_world_transform)
    selected = path.subpaths if subpath is None else [path.subpaths[subpath]]
    result.subpaths = []
    if start == end:
        return result
    if start == 0 and end == 1:
        result.subpaths = copy.deepcopy(selected)
        return result
    if measurement.length == 0:
        return result
    lower, upper = start * measurement.length, end * measurement.length
    matrix = np.eye(3) if space == 'local' else path.world_matrix
    offset = 0.
    for contour in selected:
        records = []
        current = origin = Point(0, 0)
        active, run = False, 0
        def append(cmd):
            _, derivative = _curve(current, cmd, matrix)
            speed = lambda t: float(np.linalg.norm(derivative(t)))
            length = speed(0) if isinstance(cmd, LineTo) else _integral(speed, 0, 1, measurement.tolerance)
            records.append((current, cmd, speed, length, run))
        for cmd in contour.commands:
            if isinstance(cmd, MoveTo):
                current = origin = cmd.point
                active = True
                run += 1
            elif isinstance(cmd, Close):
                if active:
                    append(LineTo(origin))
                    current = origin
                    run += 1
            else:
                active = True
                append(cmd)
                current = cmd.point if isinstance(cmd, LineTo) else cmd.end
        if contour.closed and active and current != origin:
            append(LineTo(origin))
        contour_length = math.fsum(record[3] for record in records)
        if lower <= offset and offset + contour_length <= upper:
            result.subpaths.append(copy.deepcopy(contour))
            offset += contour_length
            continue
        output, previous_run = None, None
        for first, cmd, speed, length, run in records:
            left, right = max(lower, offset), min(upper, offset + length)
            if length > 0 and right > left:
                a = _parameter_at_length(speed, length, left-offset, measurement.tolerance)
                b = _parameter_at_length(speed, length, right-offset, measurement.tolerance)
                point, portion = _portion(first, cmd, a, b)
                if output is None or previous_run != run:
                    output = Subpath([MoveTo(point)])
                    result.subpaths.append(output)
                output.commands.append(portion)
                previous_run = run
            offset += length
    return result


def split_at(path, progress, **options):
    _progress(progress)
    return trim(path, 0., progress, **options), trim(path, progress, 1., **options)
