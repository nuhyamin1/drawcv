"""Retained gradient and image paint descriptions; sampling belongs to the renderer."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Union
import numpy as np

from drawcv.core.color import Color
from drawcv.core.enums import ImageInterpolation
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.raster import (
    decode_raster_png,
    encode_raster_png,
    validate_raster_image,
)
from drawcv.core.transform import Transform
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


class Paint(ABC):
    """Abstract base class for spatial retained paints (gradients, textures)."""

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)

    def _validate(self):
        if getattr(self, "space", None) not in ("object", "world"):
            raise ValidationError("Paint space must be 'object' or 'world'")
        if not isinstance(getattr(self, "transform", None), Transform):
            raise ValidationError("Paint transform must be a Transform instance")

    def copy(self):
        return type(self).from_dict(self.to_dict())

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Losslessly serialize paint configuration to a dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Paint:
        """Construct paint from dictionary representation."""
        pass


class _Gradient(Paint):
    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "stops", tuple(self.stops))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)
        if name == "stops" and isinstance(value, (tuple, list)):
            object.__setattr__(self, name, tuple(value))

    def _validate(self):
        super()._validate()
        if getattr(self, "spread", None) not in ("pad", "repeat", "reflect"):
            raise ValidationError("Gradient spread must be 'pad', 'repeat', or 'reflect'")
        if not isinstance(self.stops, (tuple, list)) or len(self.stops) < 2:
            raise ValidationError("A gradient requires at least two ordered GradientStops")
        previous = -1
        for stop in self.stops:
            if not isinstance(stop, GradientStop) or stop.position < previous:
                raise ValidationError("GradientStops must be in nondecreasing position order")
            previous = stop.position


@dataclass
class LinearGradient(_Gradient):
    start: Point
    end: Point
    stops: tuple[GradientStop, ...]
    space: str = "object"
    spread: str = "pad"
    transform: Transform = field(default_factory=Transform)

    def _validate(self):
        super()._validate()
        if not isinstance(self.start, Point) or not isinstance(self.end, Point) or self.start == self.end:
            raise ValidationError("Linear gradient endpoints must be distinct Points")

    def to_dict(self):
        return {
            "type": "linear",
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "stops": [s.to_dict() for s in self.stops],
            "space": self.space,
            "spread": self.spread,
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        tf = Transform.from_dict(data["transform"]) if "transform" in data and data["transform"] is not None else Transform()
        return cls(
            Point.from_dict(data["start"]),
            Point.from_dict(data["end"]),
            _read_stops(data),
            data.get("space", "object"),
            data.get("spread", "pad"),
            tf,
        )


@dataclass
class RadialGradient(_Gradient):
    center: Point
    radius: float
    stops: tuple[GradientStop, ...]
    space: str = "object"
    spread: str = "pad"
    transform: Transform = field(default_factory=Transform)

    def _validate(self):
        super()._validate()
        if not isinstance(self.center, Point):
            raise ValidationError("Radial gradient center must be a Point")
        if isinstance(self.radius, bool) or not isinstance(self.radius, (int, float)) or not math.isfinite(self.radius) or self.radius <= 0:
            raise ValidationError("Radial gradient radius must be positive and finite")

    def to_dict(self):
        return {
            "type": "radial",
            "center": self.center.to_dict(),
            "radius": self.radius,
            "stops": [s.to_dict() for s in self.stops],
            "space": self.space,
            "spread": self.spread,
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        tf = Transform.from_dict(data["transform"]) if "transform" in data and data["transform"] is not None else Transform()
        return cls(
            Point.from_dict(data["center"]),
            data["radius"],
            _read_stops(data),
            data.get("space", "object"),
            data.get("spread", "pad"),
            tf,
        )


@dataclass
class ConicGradient(Paint):
    center: Point
    stops: tuple[GradientStop, ...]
    start_angle: float = 0.0
    space: str = "object"
    transform: Transform = field(default_factory=Transform)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "stops", tuple(self.stops))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)
        if name == "stops" and isinstance(value, (tuple, list)):
            object.__setattr__(self, name, tuple(value))

    def _validate(self):
        super()._validate()
        if not isinstance(self.center, Point):
            raise ValidationError("Conic gradient center must be a Point")
        if isinstance(self.start_angle, bool) or not isinstance(self.start_angle, (int, float)) or not math.isfinite(self.start_angle):
            raise ValidationError("Conic gradient start_angle must be a finite number")
        if not isinstance(self.stops, (tuple, list)) or len(self.stops) < 2:
            raise ValidationError("A gradient requires at least two ordered GradientStops")
        previous = -1
        for stop in self.stops:
            if not isinstance(stop, GradientStop) or stop.position < previous:
                raise ValidationError("GradientStops must be in nondecreasing position order")
            previous = stop.position

    def to_dict(self):
        return {
            "type": "conic",
            "center": self.center.to_dict(),
            "start_angle": float(self.start_angle),
            "stops": [s.to_dict() for s in self.stops],
            "space": self.space,
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        tf = Transform.from_dict(data["transform"]) if "transform" in data and data["transform"] is not None else Transform()
        return cls(
            Point.from_dict(data["center"]),
            _read_stops(data),
            float(data.get("start_angle", 0.0)),
            data.get("space", "object"),
            tf,
        )


def _coerce_and_validate_scale(val: Any) -> tuple[float, float]:
    if isinstance(val, bool):
        raise ValidationError("ImagePaint scale cannot be a boolean")
    if isinstance(val, (int, float)):
        val = (val, val)
    elif not isinstance(val, (tuple, list)):
        raise ValidationError(f"ImagePaint scale must be a 2-tuple or list of positive numbers, got {type(val).__name__}")
    if len(val) != 2:
        raise ValidationError(f"ImagePaint scale must have exactly 2 elements, got {len(val)}")
    try:
        sx, sy = val[0], val[1]
        if isinstance(sx, bool) or isinstance(sy, bool):
            raise ValidationError("ImagePaint scale factors cannot be booleans")
        sx_f, sy_f = float(sx), float(sy)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"ImagePaint scale factors must be numeric: {e}") from e
    if not math.isfinite(sx_f) or not math.isfinite(sy_f) or sx_f <= 0.0 or sy_f <= 0.0:
        raise ValidationError("ImagePaint scale factors must be positive finite numbers (> 0)")
    return (sx_f, sy_f)


def _coerce_and_validate_origin(val: Any) -> Point:
    if isinstance(val, Point):
        if not math.isfinite(val.x) or not math.isfinite(val.y):
            raise ValidationError("ImagePaint origin coordinates must be finite")
        return val
    if isinstance(val, bool):
        raise ValidationError("ImagePaint origin cannot be a boolean")
    if not isinstance(val, (tuple, list)):
        raise ValidationError(f"ImagePaint origin must be a Point or 2-element sequence, got {type(val).__name__}")
    if len(val) != 2:
        raise ValidationError(f"ImagePaint origin sequence must have exactly 2 elements, got {len(val)}")
    try:
        ox, oy = val[0], val[1]
        if isinstance(ox, bool) or isinstance(oy, bool):
            raise ValidationError("ImagePaint origin coordinates cannot be booleans")
        ox_f, oy_f = float(ox), float(oy)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"ImagePaint origin coordinates must be numeric: {e}") from e
    if not math.isfinite(ox_f) or not math.isfinite(oy_f):
        raise ValidationError("ImagePaint origin coordinates must be finite numbers")
    return Point(ox_f, oy_f)


@dataclass(eq=False)
class ImagePaint(Paint):
    image: np.ndarray
    origin: Point = field(default_factory=lambda: Point(0.0, 0.0))
    scale: tuple[float, float] = (1.0, 1.0)
    repeat: str = "repeat"
    space: str = "object"
    interpolation: ImageInterpolation = ImageInterpolation.LINEAR
    opacity: float = 1.0
    transform: Transform = field(default_factory=Transform)

    def __post_init__(self):
        object.__setattr__(self, "origin", _coerce_and_validate_origin(self.origin))
        object.__setattr__(self, "scale", _coerce_and_validate_scale(self.scale))
        if isinstance(self.interpolation, str):
            try:
                object.__setattr__(self, "interpolation", ImageInterpolation(self.interpolation.strip().lower()))
            except ValueError:
                raise ValidationError(f"Invalid ImageInterpolation: '{self.interpolation}'")
        self._validate()
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        if name == "image":
            validate_raster_image(value)
            value = value.copy()
        elif name == "origin":
            value = _coerce_and_validate_origin(value)
        elif name == "scale":
            value = _coerce_and_validate_scale(value)
        elif name == "interpolation" and isinstance(value, str):
            try:
                value = ImageInterpolation(value.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid ImageInterpolation: '{value}'")
        validated_setattr(self, name, value)

    def _validate(self):
        super()._validate()
        validate_raster_image(self.image)
        if not isinstance(self.origin, Point) or not math.isfinite(self.origin.x) or not math.isfinite(self.origin.y):
            raise ValidationError("ImagePaint origin must be a Point with finite coordinates")
        if not isinstance(self.scale, tuple) or len(self.scale) != 2:
            raise ValidationError("ImagePaint scale must be a 2-tuple of positive finite numbers")
        for s in self.scale:
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
                raise ValidationError("ImagePaint scale factors must be positive finite numbers (> 0)")
        if self.repeat not in ("repeat", "reflect", "pad", "none"):
            raise ValidationError("ImagePaint repeat must be 'repeat', 'reflect', 'pad', or 'none'")
        if not isinstance(self.interpolation, ImageInterpolation):
            raise ValidationError("ImagePaint interpolation must be an ImageInterpolation")
        if self.interpolation == ImageInterpolation.AREA:
            raise ValidationError(
                "ImageInterpolation.AREA is unsupported for spatial ImagePaint/remap sampling; "
                "use LINEAR, CUBIC, LANCZOS, or NEAREST instead"
            )
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float)) or not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValidationError("ImagePaint opacity must be a finite number in [0, 1]")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImagePaint):
            return False
        return (
            self.origin == other.origin
            and self.scale == other.scale
            and self.repeat == other.repeat
            and self.space == other.space
            and self.interpolation == other.interpolation
            and self.opacity == other.opacity
            and self.transform == other.transform
            and np.array_equal(self.image, other.image)
        )

    def copy(self) -> ImagePaint:
        return ImagePaint(
            image=self.image,
            origin=Point(self.origin.x, self.origin.y),
            scale=self.scale,
            repeat=self.repeat,
            space=self.space,
            interpolation=self.interpolation,
            opacity=self.opacity,
            transform=self.transform.copy(),
        )

    def to_dict(self):
        return {
            "type": "image",
            "image": encode_raster_png(self.image, error_context="ImagePaint"),
            "origin": self.origin.to_dict(),
            "scale": list(self.scale),
            "repeat": self.repeat,
            "space": self.space,
            "interpolation": self.interpolation.value,
            "opacity": float(self.opacity),
            "transform": self.transform.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        img = decode_raster_png(data["image"], error_context="ImagePaint")
        tf = Transform.from_dict(data["transform"]) if "transform" in data and data["transform"] is not None else Transform()
        origin_val = Point.from_dict(data["origin"]) if "origin" in data else Point(0.0, 0.0)
        scale_val = tuple(data.get("scale", (1.0, 1.0)))
        interp = ImageInterpolation(data["interpolation"]) if "interpolation" in data else ImageInterpolation.LINEAR
        return cls(
            image=img,
            origin=origin_val,
            scale=scale_val,
            repeat=data.get("repeat", "repeat"),
            space=data.get("space", "object"),
            interpolation=interp,
            opacity=float(data.get("opacity", 1.0)),
            transform=tf,
        )


def _read_stops(data):
    return tuple(GradientStop(s["position"], Color.from_dict(s["color"])) for s in data["stops"])


def paint_from_dict(data: dict[str, Any]) -> Paint:
    if not isinstance(data, dict) or "type" not in data:
        raise ValidationError("Paint data must be a dict with a 'type' field")
    ptype = data["type"]
    if ptype == "linear":
        return LinearGradient.from_dict(data)
    elif ptype == "radial":
        return RadialGradient.from_dict(data)
    elif ptype == "conic":
        return ConicGradient.from_dict(data)
    elif ptype == "image":
        return ImagePaint.from_dict(data)
    elif ptype == "vector_pattern":
        from drawcv.styles.pattern import VectorPattern
        return VectorPattern.from_dict(data)
    else:
        raise ValidationError(f"Unknown paint type: '{ptype}'")


PaintLike = Union[Color, Paint]
