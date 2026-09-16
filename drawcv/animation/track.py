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


def _trigger_validation(obj: Any) -> None:
    """Run explicit post-mutation validation hook on object if present."""
    if hasattr(obj, "_validate") and callable(obj._validate):
        obj._validate()


def _traverse_property_path(root: Any, path: str) -> list[tuple[Any, str | int]]:
    """Traverse a dot-separated property path supporting attributes and sequence indices.

    Returns a list of (container, token_or_index) pairs along the path.
    """
    tokens = path.split(".")
    chain: list[tuple[Any, str | int]] = []
    curr = root
    for token in tokens:
        if isinstance(curr, (list, tuple)):
            try:
                idx = int(token)
                chain.append((curr, idx))
                curr = curr[idx]
            except (ValueError, IndexError):
                raise ValidationError(f"Invalid sequence index '{token}' for container along '{path}'")
        else:
            if not hasattr(curr, token):
                raise ValidationError(f"Object {curr} has no attribute or element '{token}' along '{path}'")
            chain.append((curr, token))
            curr = getattr(curr, token)
            if curr is None and token != tokens[-1]:
                raise ValidationError(f"Property '{token}' along '{path}' is None")
    return chain


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

    def _resolve_target_property(self, target: Any) -> tuple[Any, str | int]:
        """Navigate dot-separated property path to (parent_object, attribute_or_index)."""
        chain = _traverse_property_path(target, self.property_path)
        return chain[-1]

    def evaluate(self, time: float, target: Any | None = None) -> Any:
        """Evaluate the track at the given time and apply transactionally to target if provided."""
        eased_t = self.timing.evaluate(time)
        val = self.interpolator(self.start_value, self.end_value, eased_t)

        if target is not None:
            chain = _traverse_property_path(target, self.property_path)
            parent, attr = chain[-1]

            # Check if parent is an immutable / frozen dataclass (Point, Color, BoundingBox)
            if isinstance(parent, (Color, Point, BoundingBox)):
                if len(chain) < 2:
                    raise ValidationError(f"Cannot mutate root immutable object along '{self.property_path}'")
                grandparent, parent_token = chain[-2]
                old_parent = parent

                if isinstance(parent, Point):
                    new_parent = Point(
                        val if attr == "x" else parent.x,
                        val if attr == "y" else parent.y,
                    )
                elif isinstance(parent, Color):
                    new_parent = Color(
                        int(round(val)) if attr == "r" else parent.r,
                        int(round(val)) if attr == "g" else parent.g,
                        int(round(val)) if attr == "b" else parent.b,
                        float(val) if attr == "a" else parent.a,
                    )
                elif isinstance(parent, BoundingBox):
                    new_parent = BoundingBox(
                        val if attr == "x" else parent.x,
                        val if attr == "y" else parent.y,
                        val if attr == "width" else parent.width,
                        val if attr == "height" else parent.height,
                    )

                # Transactional mutation on grandparent
                if isinstance(grandparent, list):
                    grandparent[parent_token] = new_parent
                    try:
                        _trigger_validation(grandparent)
                    except Exception:
                        grandparent[parent_token] = old_parent
                        raise
                else:
                    setattr(grandparent, str(parent_token), new_parent)
                    try:
                        _trigger_validation(grandparent)
                    except Exception:
                        setattr(grandparent, str(parent_token), old_parent)
                        raise
            else:
                # Direct mutable assignment with transactional rollback
                if isinstance(parent, list):
                    old_val = parent[attr]
                    parent[attr] = val
                    try:
                        _trigger_validation(parent)
                    except Exception:
                        parent[attr] = old_val
                        raise
                else:
                    old_val = getattr(parent, str(attr))
                    setattr(parent, str(attr), val)
                    try:
                        _trigger_validation(parent)
                    except Exception:
                        setattr(parent, str(attr), old_val)
                        raise

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
