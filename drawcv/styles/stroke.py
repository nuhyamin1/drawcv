"""Stroke style specification for outline rendering."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import Any

from drawcv.core.color import Color
from drawcv.core.enums import CapStyle, JoinStyle, LineType
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validated_setattr
from drawcv.styles.paint import Paint, PaintLike, paint_from_dict

_UNSET = object()


@dataclass(init=False)
class StrokeStyle:
    """Stroke appearance style for outline drawing.

    Attributes:
        paint: Stroke paint (Color, LinearGradient, RadialGradient, ConicGradient, ImagePaint).
        color: Solid color shorthand (raises ValidationError if paint is non-solid).
        width: Stroke thickness in the selected space (width > 0).
        space: "screen" (default) or "object"; controls width and dash geometry.
        opacity: Stroke opacity multiplier in range [0.0, 1.0].
        line_type: Anti-aliasing / pixel connectivity type (default: LineType.AA).
        cap_style: End cap style for open contours and each dash.
        join_style: Segment join style.
        dash_array: Alternating positive on/off lengths in the selected space; empty is solid.
        dash_offset: Signed distance into the repeating dash pattern.
        miter_limit: Maximum miter length divided by half-width; bevel beyond this.
    """
    width: float
    opacity: float
    line_type: LineType
    cap_style: CapStyle
    join_style: JoinStyle
    dash_array: tuple[float, ...]
    dash_offset: float
    miter_limit: float
    space: str
    _paint: PaintLike

    def __init__(
        self,
        color: Any = _UNSET,
        width: float = 1.0,
        opacity: float = 1.0,
        line_type: LineType = LineType.AA,
        cap_style: CapStyle = CapStyle.ROUND,
        join_style: JoinStyle = JoinStyle.ROUND,
        dash_array: tuple[float, ...] = (),
        dash_offset: float = 0.0,
        miter_limit: float = 4.0,
        *,
        paint: Any = _UNSET,
        space: str = "screen",
    ):
        if color is not _UNSET and paint is not _UNSET:
            raise ValidationError("Specify color or paint, not both")
        if color is not _UNSET and not isinstance(color, Color):
            raise ValidationError("Stroke color must be a Color; use paint for gradients and image paints")
        if paint is not _UNSET and not isinstance(paint, (Color, Paint)):
            raise ValidationError("Invalid stroke paint")

        self.width = width
        self.opacity = opacity
        self.line_type = line_type
        self.cap_style = cap_style
        self.join_style = join_style
        self.dash_array = tuple(dash_array)
        self.dash_offset = dash_offset
        self.miter_limit = miter_limit
        self.space = space

        chosen_paint = paint if paint is not _UNSET else (color if color is not _UNSET else Color.black())
        self.paint = chosen_paint

        self._validate()
        object.__setattr__(self, "_initialized", True)

    @property
    def paint(self) -> PaintLike:
        """The active stroke paint (Color or Paint subclass)."""
        return self._paint

    @paint.setter
    def paint(self, value: PaintLike):
        if not isinstance(value, (Color, Paint)):
            raise ValidationError("Invalid stroke paint")
        object.__setattr__(self, "_paint", value)

    @property
    def color(self) -> Color:
        """Stroke color shorthand; gradients and image paints have no single color."""
        if not isinstance(self._paint, Color):
            raise ValidationError("Gradient and image strokes have no single color; use stroke.paint")
        return self._paint

    @color.setter
    def color(self, value: Color):
        if not isinstance(value, Color):
            raise ValidationError("Stroke color must be a Color")
        object.__setattr__(self, "_paint", value)

    def _validate(self):
        if not isinstance(self.space, str) or self.space not in ("screen", "object"):
            raise ValidationError("Stroke space must be 'screen' or 'object'")
        if not isinstance(self.paint, (Color, Paint)):
            raise ValidationError(f"StrokeStyle 'paint' must be a Color or Paint, got {type(self.paint).__name__}")
        if not isinstance(self.width, (int, float)) or isinstance(self.width, bool):
            raise ValidationError(f"StrokeStyle 'width' must be numeric, got {type(self.width).__name__}")
        if math.isnan(self.width) or math.isinf(self.width) or self.width <= 0:
            raise ValidationError(f"StrokeStyle 'width' must be strictly positive, got {self.width}")
        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise ValidationError(f"StrokeStyle 'opacity' must be numeric, got {type(self.opacity).__name__}")
        if not (0.0 <= float(self.opacity) <= 1.0):
            raise ValidationError(f"StrokeStyle 'opacity' must be in range [0.0, 1.0], got {self.opacity}")
        if not isinstance(self.line_type, LineType):
            raise ValidationError(f"StrokeStyle 'line_type' must be a LineType, got {type(self.line_type).__name__}")
        if not isinstance(self.cap_style, CapStyle):
            raise ValidationError(f"StrokeStyle 'cap_style' must be a CapStyle, got {type(self.cap_style).__name__}")
        if not isinstance(self.join_style, JoinStyle):
            raise ValidationError(f"StrokeStyle 'join_style' must be a JoinStyle, got {type(self.join_style).__name__}")

        if not isinstance(self.dash_array, (tuple, list)):
            raise ValidationError("dash_array must be a list or tuple of positive lengths")
        for length in self.dash_array:
            if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
                raise ValidationError("dash_array lengths must be positive finite numbers")
        if self.dash_array and not math.isfinite(sum(self.dash_array) * 2):
            raise ValidationError("dash_array total must be finite")
        for name in ("dash_offset", "miter_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValidationError(f"{name} must be a finite number")
        if self.miter_limit < 1:
            raise ValidationError("miter_limit must be at least 1")

    @property
    def bounds_padding(self) -> float:
        """Conservative stroke-space extent, including square caps and miter tips."""
        factor = self.miter_limit if self.join_style == JoinStyle.MITER else 1.0
        if self.cap_style == CapStyle.SQUARE:
            factor = max(factor, math.sqrt(2))
        return self.width * factor / 2

    def __setattr__(self, name, value):
        if name in ("color", "paint"):
            allowed = (Color,) if name == "color" else (Color, Paint)
            if not isinstance(value, allowed):
                raise ValidationError(f"Invalid stroke {name}")
            object.__setattr__(self, "_paint", value)
        else:
            validated_setattr(self, name, value)
            if name == "dash_array" and isinstance(value, (tuple, list)):
                object.__setattr__(self, name, tuple(value))

    def world_padding(self, matrix) -> float:
        """Conservative stroke extent after an affine transform."""
        scale = float(np.linalg.norm(matrix[:2, :2], ord=2)) if self.space == "object" else 1.0
        return self.bounds_padding * scale

    def world_width(self, matrix) -> float:
        """Maximum world-space width, for conservative geometric hit testing."""
        scale = float(np.linalg.norm(matrix[:2, :2], ord=2)) if self.space == "object" else 1.0
        return self.width * scale

    def copy(self) -> StrokeStyle:
        """Return an independent copy of this StrokeStyle."""
        return StrokeStyle(
            paint=self.paint.copy(),
            width=self.width,
            opacity=self.opacity,
            line_type=self.line_type,
            cap_style=self.cap_style,
            join_style=self.join_style,
            dash_array=self.dash_array,
            dash_offset=self.dash_offset,
            miter_limit=self.miter_limit,
            space=self.space,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        result = {
            "width": float(self.width),
            "opacity": float(self.opacity),
            "line_type": self.line_type.value,
            "cap_style": self.cap_style.value,
            "join_style": self.join_style.value,
            "dash_array": list(self.dash_array),
            "dash_offset": self.dash_offset,
            "miter_limit": self.miter_limit,
            "space": self.space,
        }
        result["color" if isinstance(self.paint, Color) else "paint"] = self.paint.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StrokeStyle:
        """Construct a StrokeStyle from a dictionary."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValidationError(f"StrokeStyle data must be a dict, got {type(data).__name__}")

        if data.get("paint") is not None and data.get("color") is not None:
            raise ValidationError("Stroke data cannot contain both color and paint")

        line_type_val = LineType(data["line_type"]) if "line_type" in data else LineType.AA
        cap_style_val = CapStyle(data["cap_style"]) if "cap_style" in data else CapStyle.ROUND
        join_style_val = JoinStyle(data["join_style"]) if "join_style" in data else JoinStyle.ROUND

        if data.get("paint") is not None:
            paint_val = paint_from_dict(data["paint"])
            return cls(
                paint=paint_val,
                width=float(data.get("width", 1.0)),
                opacity=float(data.get("opacity", 1.0)),
                line_type=line_type_val,
                cap_style=cap_style_val,
                join_style=join_style_val,
                dash_array=data.get("dash_array", ()),
                dash_offset=data.get("dash_offset", 0.0),
                miter_limit=data.get("miter_limit", 4.0),
                space=data.get("space", "screen"),
            )
        else:
            color_val = Color.from_dict(data["color"]) if "color" in data else Color.black()
            return cls(
                color=color_val,
                width=float(data.get("width", 1.0)),
                opacity=float(data.get("opacity", 1.0)),
                line_type=line_type_val,
                cap_style=cap_style_val,
                join_style=join_style_val,
                dash_array=data.get("dash_array", ()),
                dash_offset=data.get("dash_offset", 0.0),
                miter_limit=data.get("miter_limit", 4.0),
                space=data.get("space", "screen"),
            )
