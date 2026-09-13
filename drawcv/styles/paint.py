"""Retained gradient descriptions; sampling belongs to the renderer."""
from dataclasses import dataclass
import math
from drawcv.core.color import Color
from drawcv.core.geometry import Point
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validated_setattr


@dataclass(frozen=True)
class GradientStop:
    position: float
    color: Color

    def __post_init__(self):
        if isinstance(self.position, bool) or not isinstance(self.position, (int, float)) or not 0 <= self.position <= 1:
            raise ValidationError("Stop position must be numeric in [0, 1]")
        if not isinstance(self.color, Color):
            raise ValidationError("Stop color must be a Color")

    def to_dict(self):
        return {"position": self.position, "color": self.color.to_dict()}


class _Gradient:
    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "stops", tuple(self.stops))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)
        if name == "stops" and isinstance(value, (tuple, list)):
            object.__setattr__(self, name, tuple(value))

    def _validate(self):
        if self.space not in ("object", "world"):
            raise ValidationError("Gradient space must be 'object' or 'world'")
        if not isinstance(self.stops, (tuple, list)) or len(self.stops) < 2:
            raise ValidationError("A gradient requires at least two ordered GradientStops")
        previous = -1
        for stop in self.stops:
            if not isinstance(stop, GradientStop) or stop.position < previous:
                raise ValidationError("GradientStops must be in nondecreasing position order")
            previous = stop.position

    def copy(self):
        return type(self).from_dict(self.to_dict())


@dataclass
class LinearGradient(_Gradient):
    start: Point
    end: Point
    stops: tuple[GradientStop, ...]
    space: str = "object"

    def _validate(self):
        super()._validate()
        if not isinstance(self.start, Point) or not isinstance(self.end, Point) or self.start == self.end:
            raise ValidationError("Linear gradient endpoints must be distinct Points")

    def to_dict(self):
        return {"type": "linear", "start": self.start.to_dict(), "end": self.end.to_dict(),
                "stops": [s.to_dict() for s in self.stops], "space": self.space}

    @classmethod
    def from_dict(cls, data):
        return cls(Point.from_dict(data["start"]), Point.from_dict(data["end"]),
                   _read_stops(data), data.get("space", "object"))


@dataclass
class RadialGradient(_Gradient):
    center: Point
    radius: float
    stops: tuple[GradientStop, ...]
    space: str = "object"

    def _validate(self):
        super()._validate()
        if not isinstance(self.center, Point):
            raise ValidationError("Radial gradient center must be a Point")
        if isinstance(self.radius, bool) or not isinstance(self.radius, (int, float)) or not math.isfinite(self.radius) or self.radius <= 0:
            raise ValidationError("Radial gradient radius must be positive and finite")

    def to_dict(self):
        return {"type": "radial", "center": self.center.to_dict(), "radius": self.radius,
                "stops": [s.to_dict() for s in self.stops], "space": self.space}

    @classmethod
    def from_dict(cls, data):
        return cls(Point.from_dict(data["center"]), data["radius"], _read_stops(data), data.get("space", "object"))


def _read_stops(data):
    return tuple(GradientStop(s["position"], Color.from_dict(s["color"])) for s in data["stops"])


def paint_from_dict(data):
    if not isinstance(data, dict) or data.get("type") not in ("linear", "radial"):
        raise ValidationError("Unknown gradient paint type")
    cls = LinearGradient if data["type"] == "linear" else RadialGradient
    return cls.from_dict(data)
