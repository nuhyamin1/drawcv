"""Exact local path geometry shared by conversion and SVG export."""
from __future__ import annotations

import copy
import math

from drawcv.core.enums import ArcClosure
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.shapes.path import (Path, Subpath, MoveTo, LineTo, QuadraticTo,
                                CubicTo, EllipticalArcTo, Close)


def path_subpaths(obj):
    """Return independent semantic contours, without flattening curves."""
    from drawcv.shapes.arc import Arc
    from drawcv.shapes.bezier import BezierCurve
    from drawcv.shapes.circle import Circle
    from drawcv.shapes.ellipse import Ellipse
    from drawcv.shapes.line import Line
    from drawcv.shapes.polygon import Polygon
    from drawcv.shapes.polyline import Polyline
    from drawcv.shapes.rectangle import Rectangle
    from drawcv.shapes.rounded_rectangle import RoundedRectangle

    if isinstance(obj, Path):
        return copy.deepcopy(obj.subpaths)
    commands = []
    closed = False

    def arc(center, rx, ry, start, sweep):
        def point(angle):
            a = math.radians(angle)
            return Point(center.x + rx * math.cos(a), center.y + ry * math.sin(a))
        first = point(start)
        commands.append(MoveTo(first) if not commands else LineTo(first))
        # Identical endpoints describe an empty SVG arc, so split full turns.
        count = 2 if abs(sweep) == 360 else 1
        for i in range(count):
            if sweep == 0:
                commands.append(LineTo(first.copy()))
                break
            end = first.copy() if abs(sweep) == 360 and i == count - 1 else point(start + sweep * (i + 1) / count)
            commands.append(EllipticalArcTo(rx, ry, 0, abs(sweep / count) > 180, sweep > 0, end))

    if isinstance(obj, (Circle, Ellipse)):
        rx, ry = (obj.radius, obj.radius) if isinstance(obj, Circle) else (obj.radius_x, obj.radius_y)
        if rx <= 0 or ry <= 0:
            return []
        arc(obj.center, rx, ry, 0, 360)
        closed = True
    elif isinstance(obj, Arc):
        arc(obj.center, obj.radius_x, obj.radius_y, obj.start_angle, obj.sweep_angle)
        if obj.closure == ArcClosure.PIE:
            commands.append(LineTo(obj.center.copy()))
        closed = obj.closure != ArcClosure.OPEN
    elif isinstance(obj, RoundedRectangle) and obj.corner_radius > 0:
        x, y, w, h, r = obj.x, obj.y, obj.width, obj.height, obj.corner_radius
        for center, start in [(Point(x+r, y+r), 180), (Point(x+w-r, y+r), 270),
                              (Point(x+w-r, y+h-r), 0), (Point(x+r, y+h-r), 90)]:
            arc(center, r, r, start, 90)
        closed = True
    elif isinstance(obj, BezierCurve):
        commands = [MoveTo(obj.p0.copy()), CubicTo(obj.p1.copy(), obj.p2.copy(), obj.p3.copy())
                    if obj.p3 is not None else QuadraticTo(obj.p1.copy(), obj.p2.copy())]
    else:
        if isinstance(obj, Line):
            points = [obj.start, obj.end]
        elif isinstance(obj, (Rectangle, RoundedRectangle)):
            if obj.width <= 0 or obj.height <= 0:
                return []
            points = obj.corners if isinstance(obj, Rectangle) else [Point(obj.x, obj.y), Point(obj.x+obj.width, obj.y),
                      Point(obj.x+obj.width, obj.y+obj.height), Point(obj.x, obj.y+obj.height)]
            closed = True
        elif isinstance(obj, Polygon):
            points, closed = obj.vertices, True
        elif isinstance(obj, Polyline):
            points, closed = obj.points, obj.closed
        else:
            raise ValidationError(f"Exact to_path() conversion is not supported for {type(obj).__name__}")
        commands = [(MoveTo if i == 0 else LineTo)(p.copy()) for i, p in enumerate(points)]
    if closed and commands:
        commands.append(Close())
    return [Subpath(commands, closed)] if commands else []


def to_path(obj, *, preserve_world_transform=False):
    from drawcv.core.transform import Transform
    from drawcv.shapes.arc import Arc

    if not isinstance(preserve_world_transform, bool):
        raise ValidationError("preserve_world_transform must be a boolean")
    result = Path(subpaths=path_subpaths(obj))
    for name in ("name", "visible", "locked", "opacity", "blend_mode", "z_index",
                 "tags", "metadata", "clip", "mask", "effects", "timing", "render_progress",
                 "stroke", "fill", "fill_rule"):
        if hasattr(obj, name):
            setattr(result, name, copy.deepcopy(getattr(obj, name)))
    if isinstance(obj, Arc) and obj.closure == ArcClosure.OPEN:
        result.fill = None
    # Freeze the resolved pivot: changing representation must not move geometry.
    matrix = obj.world_matrix if preserve_world_transform else obj.transform.get_matrix(
        default_pivot=obj.get_geometry_bounds().center)
    result.transform = Transform.from_matrix(matrix)
    return result
