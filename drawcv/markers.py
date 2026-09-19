"""Reusable compound-path artwork placed at semantic path vertices."""
from dataclasses import dataclass, field
import math

import numpy as np

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.core.validation import validated_setattr


@dataclass
class Marker:
    """Reusable Path artwork; ref is the attachment point in marker coordinates.

    units='stroke' multiplies size by host stroke width (one without a stroke).
    units='user' ignores stroke width. Both follow the host geometry transform.
    orient is 'auto', 'auto-start-reverse', or a clockwise angle in degrees.
    Nested markers are rejected. Artwork is referenced live, not inserted in a scene.
    """
    path: object
    ref: Point = field(default_factory=lambda: Point(0, 0))
    orient: str | float = 'auto'
    units: str = 'stroke'
    size: float = 1.

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_initialized', True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)

    def _validate(self):
        from drawcv.shapes.path import Path
        if not isinstance(self.path, Path):
            raise ValidationError('Marker artwork must be a Path; convert shapes with to_path()')
        if any(getattr(self.path, name, None) is not None for name in ('marker_start', 'marker_mid', 'marker_end')):
            raise ValidationError('Nested marker artwork is not supported')
        if not isinstance(self.ref, Point):
            raise ValidationError('Marker ref must be a Point')
        if self.units not in ('stroke', 'user'):
            raise ValidationError("Marker units must be 'stroke' or 'user'")
        if isinstance(self.orient, str):
            if self.orient not in ('auto', 'auto-start-reverse'):
                raise ValidationError('Invalid marker orientation')
        elif isinstance(self.orient, bool) or not isinstance(self.orient, (int, float)) or not math.isfinite(self.orient):
            raise ValidationError('Marker orientation must be a finite angle or auto mode')
        if isinstance(self.size, bool) or not isinstance(self.size, (int, float)) or not math.isfinite(self.size) or self.size <= 0:
            raise ValidationError('Marker size must be finite and positive')

    def to_dict(self):
        self._validate()
        return dict(path=self.path.to_dict(), ref=self.ref.to_dict(), orient=self.orient, units=self.units, size=self.size)

    @classmethod
    def from_dict(cls, data):
        from drawcv.shapes.path import Path
        return cls(Path.from_dict(data['path']), Point.from_dict(data.get('ref', {'x': 0, 'y': 0})),
                   data.get('orient', 'auto'), data.get('units', 'stroke'), data.get('size', 1.))


def _vertices(path):
    from drawcv.shapes.path import MoveTo, LineTo, Close
    from drawcv.core.path_measurement import _curve

    def direction(start, cmd, end):
        _, derivative = _curve(start, cmd, np.eye(3))
        for t in ((1., 1.-1e-7, 1.-1e-4) if end else (0., 1e-7, 1e-4)):
            v = derivative(t)
            norm = np.linalg.norm(v)
            if norm > 0:
                return v/norm
        return None

    def placements(edges):
        if not edges:
            return
        incoming = [direction(a, cmd, True) for a, cmd, b in edges]
        outgoing = [direction(a, cmd, False) for a, cmd, b in edges]
        def angle(v):
            return math.atan2(v[1], v[0]) if v is not None else 0.
        first = next((v for v in outgoing if v is not None), None)
        last = next((v for v in reversed(incoming) if v is not None), None)
        yield 'start', edges[0][0], angle(first)
        for i in range(1, len(edges)):
            a = next((v for v in reversed(incoming[:i]) if v is not None), None)
            b = next((v for v in outgoing[i:] if v is not None), None)
            if a is None:
                theta = angle(b)
            elif b is None:
                theta = angle(a)
            else:
                theta = angle(a) + ((angle(b)-angle(a)+math.pi) % (2*math.pi)-math.pi)/2
            yield 'mid', edges[i][0], theta
        yield 'end', edges[-1][2], angle(last)

    for subpath in path.subpaths:
        current = origin = Point(0, 0)
        edges = []
        for cmd in subpath.commands:
            if isinstance(cmd, MoveTo):
                yield from placements(edges)
                edges = []
                current = origin = cmd.point
            elif isinstance(cmd, Close):
                edges.append((current, LineTo(origin), origin))
                current = origin
            else:
                end = cmd.point if isinstance(cmd, LineTo) else cmd.end
                edges.append((current, cmd, end))
                current = end
        if subpath.closed and edges and not isinstance(subpath.commands[-1], Close):
            edges.append((current, LineTo(origin), origin))
        yield from placements(edges)


def marker_instances(path, matrix=None):
    """Independent world-positioned copies; never reparent shared artwork."""
    if not any(getattr(path, 'marker_'+role) is not None for role in ('start', 'mid', 'end')):
        return
    for role in ('start', 'mid', 'end'):
        marker = getattr(path, 'marker_'+role)
        if marker is not None:
            if not isinstance(marker, Marker):
                raise ValidationError('Path markers must be Marker instances or None')
            marker._validate()
    matrix = path.world_matrix if matrix is None else matrix
    for role, position, theta in _vertices(path):
        marker = getattr(path, 'marker_'+role)
        if marker is None:
            continue
        if isinstance(marker.orient, str):
            if marker.orient == 'auto-start-reverse' and role == 'start':
                theta += math.pi
        else:
            theta = math.radians(marker.orient)
        scale = marker.size * (path.stroke.width if marker.units == 'stroke' and path.stroke else 1.)
        c, s = math.cos(theta)*scale, math.sin(theta)*scale
        placement = np.array([[c, -s, position.x-c*marker.ref.x+s*marker.ref.y],
                              [s, c, position.y-s*marker.ref.x-c*marker.ref.y], [0, 0, 1.]])
        instance = marker.path.to_path()
        instance.transform = Transform.from_matrix(matrix @ placement @ instance.world_matrix)
        yield instance
