"""Fill appearance and retained paint, independent of rasterization."""
from dataclasses import dataclass
from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validated_setattr
from drawcv.styles.paint import LinearGradient, RadialGradient, paint_from_dict

_UNSET = object()


@dataclass(init=False)
class FillStyle:
    """Fill description with a backward-compatible solid color shorthand."""
    enabled: bool
    opacity: float
    _paint: Color | LinearGradient | RadialGradient

    def __init__(self, enabled=True, color=_UNSET, opacity=1.0, *, paint=_UNSET):
        if color is not _UNSET and paint is not _UNSET:
            raise ValidationError("Specify color or paint, not both")
        if color is not _UNSET and not isinstance(color, Color):
            raise ValidationError("Fill color must be a Color; use paint for gradients")
        self.enabled = enabled
        self.opacity = opacity
        self.paint = paint if paint is not _UNSET else (color if color is not _UNSET else Color.white())
        self._validate()
        object.__setattr__(self, "_initialized", True)

    @property
    def paint(self):
        """The active Color, LinearGradient, or RadialGradient."""
        return self._paint

    @property
    def color(self):
        """Solid color shorthand; gradients have no single color."""
        if not isinstance(self.paint, Color):
            raise ValidationError("Gradient fills have no single color; use fill.paint")
        return self.paint

    def __setattr__(self, name, value):
        if name in ("color", "paint"):
            allowed = (Color,) if name == "color" else (Color, LinearGradient, RadialGradient)
            if not isinstance(value, allowed):
                raise ValidationError(f"Invalid fill {name}")
            object.__setattr__(self, "_paint", value)
        else:
            validated_setattr(self, name, value)

    def _validate(self):
        if not isinstance(self.enabled, bool):
            raise ValidationError("Fill enabled must be boolean")
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float)) or not 0 <= self.opacity <= 1:
            raise ValidationError("Fill opacity must be numeric in [0, 1]")
        if not isinstance(self.paint, (Color, LinearGradient, RadialGradient)):
            raise ValidationError("Invalid fill paint")

    def copy(self):
        return FillStyle(enabled=self.enabled, opacity=self.opacity, paint=self.paint.copy())

    def to_dict(self):
        result = {"enabled": self.enabled, "opacity": float(self.opacity)}
        result["color" if isinstance(self.paint, Color) else "paint"] = self.paint.to_dict()
        return result

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValidationError("FillStyle data must be a dict")
        if "paint" in data:
            if "color" in data:
                raise ValidationError("Fill data cannot contain both color and paint")
            return cls(enabled=data.get("enabled", True), opacity=data.get("opacity", 1.0),
                       paint=paint_from_dict(data["paint"]))
        return cls(enabled=bool(data.get("enabled", True)), opacity=float(data.get("opacity", 1.0)),
                   color=Color.from_dict(data["color"]) if "color" in data else Color.white())
