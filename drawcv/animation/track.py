"""Animation track representation and typed value interpolation."""

from __future__ import annotations
import copy
from typing import Any, Callable

from drawcv.animation.interpolation import (
    lerp,
    lerp_bounds,
    lerp_color,
    lerp_point,
    lerp_transform,
    resolve_interpolator,
)
from drawcv.animation.timing import Timing
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.exceptions import SerializationError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform


def _infer_value_type(val: Any) -> str:
    if isinstance(val, Color):
        return "color"
    if isinstance(val, Point):
        return "point"
    if isinstance(val, Transform):
        return "transform"
    if isinstance(val, BoundingBox):
        return "bounds"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return "number"
    raise ValidationError(f"Unsupported animation value type: {type(val).__name__}")


def _serialize_track_value(val: Any, val_type: str) -> Any:
    if val_type == "number":
        return float(val)
    if hasattr(val, "to_dict"):
        return val.to_dict()
    raise SerializationError(f"Cannot serialize track value of type '{val_type}'")


def _deserialize_track_value(data_val: Any, val_type: str) -> Any:
    if val_type == "number":
        return float(data_val)
    if val_type == "color":
        return Color.from_dict(data_val)
    if val_type == "point":
        return Point.from_dict(data_val)
    if val_type == "transform":
        return Transform.from_dict(data_val)
    if val_type == "bounds":
        return BoundingBox.from_dict(data_val)
    raise SerializationError(f"Unrecognized track value_type '{val_type}'")


class AnimationTrack:
    """Individual property animation track targeting a scene entity.

    Attributes:
        target_id: UUID of the target Drawable in the scene graph.
        property_path: Dot-separated path to the animated property (e.g. 'radius', 'fill.color').
        start_value: Initial value at t=0.
        end_value: Terminal value at t=duration.
        timing: Timing configuration.
        value_type: Serialized type discriminator ('number', 'point', 'color', 'transform', 'bounds').
        interpolator: Optional custom interpolation function.
    """

    def __init__(
        self,
        target_id: str,
        property_path: str,
        start_value: Any,
        end_value: Any,
        timing: Timing | None = None,
        value_type: str | None = None,
        interpolator: Callable[[Any, Any, float], Any] | None = None,
    ):
        if not isinstance(target_id, str) or not target_id:
            raise ValidationError("AnimationTrack 'target_id' must be a non-empty string")
        if not isinstance(property_path, str) or not property_path:
            raise ValidationError("AnimationTrack 'property_path' must be a non-empty string")
        if property_path == "blend_mode" or property_path.endswith(".blend_mode"):
            raise ValidationError("Animation of discrete property 'blend_mode' is not supported")

        self.target_id = target_id
        self.property_path = property_path
        self.start_value = start_value
        self.end_value = end_value
        self.timing = timing if timing is not None else Timing()

        inferred_type = _infer_value_type(start_value)
        self.value_type = value_type if value_type is not None else inferred_type

        if interpolator is not None:
            self.interpolator = interpolator
        else:
            self.interpolator = resolve_interpolator(self.start_value, self.end_value)

    def _resolve_target_property(self, target: Any) -> tuple[Any, str]:
        """Navigate dot-separated property path to (parent_object, attribute_name)."""
        parts = self.property_path.split(".")
        curr = target
        for part in parts[:-1]:
            if not hasattr(curr, part):
                raise ValidationError(f"Object {curr} has no property '{part}' along '{self.property_path}'")
            curr = getattr(curr, part)
            if curr is None:
                raise ValidationError(f"Property '{part}' along '{self.property_path}' is None")
        attr_name = parts[-1]
        if not hasattr(curr, attr_name):
            raise ValidationError(f"Target object has no attribute '{attr_name}'")
        return curr, attr_name

    def evaluate(self, time: float, target: Any | None = None) -> Any:
        """Evaluate the track at the given time and apply to target if provided."""
        eased_t = self.timing.evaluate(time)
        val = self.interpolator(self.start_value, self.end_value, eased_t)

        if target is not None:
            parts = self.property_path.split(".")
            parent_obj, attr = self._resolve_target_property(target)
            try:
                setattr(parent_obj, attr, val)
            except Exception as exc:
                # Handle immutable / frozen dataclass parents (Point, Color, BoundingBox)
                if len(parts) >= 2:
                    grandparent = target
                    for p in parts[:-2]:
                        grandparent = getattr(grandparent, p)
                    field_name = parts[-2]
                    curr_val = getattr(grandparent, field_name)
                    if isinstance(curr_val, Point):
                        new_pt = Point(val if attr == "x" else curr_val.x, val if attr == "y" else curr_val.y)
                        setattr(grandparent, field_name, new_pt)
                    elif isinstance(curr_val, Color):
                        new_col = Color(
                            int(round(val)) if attr == "r" else curr_val.r,
                            int(round(val)) if attr == "g" else curr_val.g,
                            int(round(val)) if attr == "b" else curr_val.b,
                            float(val) if attr == "a" else curr_val.a,
                        )
                        setattr(grandparent, field_name, new_col)
                    elif isinstance(curr_val, BoundingBox):
                        new_box = BoundingBox(
                            val if attr == "x" else curr_val.x,
                            val if attr == "y" else curr_val.y,
                            val if attr == "width" else curr_val.width,
                            val if attr == "height" else curr_val.height,
                        )
                        setattr(grandparent, field_name, new_box)
                    else:
                        raise exc
                else:
                    raise exc

        return val

    def to_dict(self) -> dict[str, Any]:
        """Serialize track to plain JSON-compatible dictionary."""
        return {
            "target_id": self.target_id,
            "property_path": self.property_path,
            "value_type": self.value_type,
            "start_value": _serialize_track_value(self.start_value, self.value_type),
            "end_value": _serialize_track_value(self.end_value, self.value_type),
            "timing": self.timing.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], scene: Any | None = None) -> AnimationTrack:
        """Construct an AnimationTrack from serialized dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"AnimationTrack data must be a dict, got {type(data).__name__}")

        val_type = data.get("value_type", "number")
        start_val = _deserialize_track_value(data["start_value"], val_type)
        end_val = _deserialize_track_value(data["end_value"], val_type)
        timing_val = Timing.from_dict(data["timing"])

        return cls(
            target_id=data["target_id"],
            property_path=data["property_path"],
            start_value=start_val,
            end_value=end_val,
            timing=timing_val,
            value_type=val_type,
        )
