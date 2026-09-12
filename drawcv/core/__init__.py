"""Core classes, models, and exceptions for DrawCV."""

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import ArcClosure, ArrowHeadStyle, CapStyle, FillRule, JoinStyle, LineType
from drawcv.core.exceptions import (
    DrawCVError,
    ObjectNotFoundError,
    RenderError,
    ValidationError,
)
from drawcv.core.geometry import Point, distance_point_to_segment
from drawcv.core.transform import Transform

__all__ = [
    "ArcClosure",
    "ArrowHeadStyle",
    "BoundingBox",
    "CapStyle",
    "Color",
    "distance_point_to_segment",
    "Drawable",
    "DrawCVError",
    "FillRule",
    "JoinStyle",
    "LineType",
    "ObjectNotFoundError",
    "Point",
    "RenderError",
    "Transform",
    "ValidationError",
]

