"""Abstract base class for all retained-mode drawable objects in DrawCV."""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
import math
import uuid
from typing import Any
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.enums import BlendMode, coerce_blend_mode
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import atomic_transforms
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform


@dataclass(eq=False)
class Drawable(ABC):
    """Abstract base class for retained-mode drawing entities.
    
    All drawables maintain identity, appearance flags, metadata, tags,
    and an affine transform hierarchy separating local geometry from world space.
    """
    supports_progressive_rendering: bool = False

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend_mode: BlendMode = field(default=BlendMode.NORMAL, kw_only=True)
    z_index: int = 0
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    transform: Transform = field(default_factory=Transform)
    clip: Any | None = field(default=None)
    mask: Any | None = field(default=None)
    effects: list[Any] = field(default_factory=list)
    timing: Any | None = field(default=None)
    render_progress: float = 1.0
    _parent: Any | None = field(default=None, repr=False, compare=False)
    _layer: Any | None = field(default=None, repr=False, compare=False)
    _scene: Any | None = field(default=None, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "blend_mode":
            value = coerce_blend_mode(value)
        super().__setattr__(name, value)

    def __post_init__(self):
        self._validate_common()

    def _validate_common(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("Drawable 'id' must be a non-empty string")
        if self.name is not None and not isinstance(self.name, str):
            raise ValidationError("Drawable 'name' must be a string or None")
        if not isinstance(self.visible, bool):
            raise ValidationError("Drawable 'visible' must be a boolean")
        if not isinstance(self.locked, bool):
            raise ValidationError("Drawable 'locked' must be a boolean")
        if not isinstance(self.opacity, (int, float)) or isinstance(self.opacity, bool):
            raise ValidationError("Drawable 'opacity' must be a numeric float")
        if not (0.0 <= float(self.opacity) <= 1.0):
            raise ValidationError(f"Drawable 'opacity' must be in range [0.0, 1.0], got {self.opacity}")
        self.blend_mode = coerce_blend_mode(self.blend_mode)
        if not isinstance(self.z_index, int) or isinstance(self.z_index, bool):
            raise ValidationError("Drawable 'z_index' must be an integer")
        if not isinstance(self.tags, set):
            raise ValidationError("Drawable 'tags' must be a set")
        if not isinstance(self.metadata, dict):
            raise ValidationError("Drawable 'metadata' must be a dict")
        if not isinstance(self.transform, Transform):
            raise ValidationError("Drawable 'transform' must be a Transform instance")
        if not isinstance(self.effects, list):
            raise ValidationError("Drawable 'effects' must be a list")
        if self.timing is not None:
            from drawcv.animation.timing import Timing
            if not isinstance(self.timing, Timing):
                raise ValidationError(f"Drawable 'timing' must be a Timing instance or None, got {type(self.timing).__name__}")
        if not isinstance(self.render_progress, (int, float)) or isinstance(self.render_progress, bool):
            raise ValidationError("Drawable 'render_progress' must be numeric")
        if math.isnan(self.render_progress) or math.isinf(self.render_progress):
            raise ValidationError("Drawable 'render_progress' must be finite")

    @property
    def progress(self) -> float:
        """Normalized render progress clamped to [0.0, 1.0]."""
        return self.render_progress

    @progress.setter
    def progress(self, val: float | int) -> None:
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValidationError(f"Drawable 'progress' must be numeric, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValidationError(f"Drawable 'progress' must be finite, got {val}")
        self.render_progress = float(max(0.0, min(1.0, val)))

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_effect_bounds(self) -> BoundingBox:
        """Calculate authoritative world-space visual bounds inflated by attached post-processing effects."""
        base_bounds = self.get_bounds()
        if not self.effects:
            return base_bounds

        left_pad = 0.0
        right_pad = 0.0
        top_pad = 0.0
        bottom_pad = 0.0
        for eff in self.effects:
            if hasattr(eff, "get_padding"):
                lp, rp, tp, bp = eff.get_padding()
                left_pad = max(left_pad, lp)
                right_pad = max(right_pad, rp)
                top_pad = max(top_pad, tp)
                bottom_pad = max(bottom_pad, bp)

        return BoundingBox(
            base_bounds.left - left_pad,
            base_bounds.top - top_pad,
            base_bounds.width + left_pad + right_pad,
            base_bounds.height + top_pad + bottom_pad,
        )

    @abstractmethod
    def get_geometry_bounds(self) -> BoundingBox:
        """Calculate intrinsic geometric bounds in local space (excluding stroke)."""
        pass

    @abstractmethod
    def get_local_bounds(self) -> BoundingBox:
        """Calculate the object's visual bounds at identity transform, including nominal stroke width.
        
        Note: Under the non-scaling screen-stroke rule, stroke width is in screen pixels,
        so transform(get_local_bounds()) != get_bounds(). Use get_bounds() for actual rendered bounds.
        """
        pass

    @abstractmethod
    def get_bounds(self) -> BoundingBox:
        """Calculate the authoritative actual rendered world-space visual AABB after transformation
        under the non-scaling screen-stroke rule.
        """
        pass

    def get_bounds_in_parent(self) -> BoundingBox:
        """Calculate this object's visual bounds in its immediate parent's coordinate space.
        
        Evaluated bottom-up: maps intrinsic local geometry through this object's local transform
        without traversing ancestor group transforms.
        """
        local_pivot = self.transform.pivot
        if local_pivot is None:
            local_pivot = self.get_geometry_bounds().center
        corners = self.get_local_bounds().corners
        pts = [self.transform.transform_point(c, default_pivot=local_pivot) for c in corners]
        min_x = min(p.x for p in pts)
        max_x = max(p.x for p in pts)
        min_y = min(p.y for p in pts)
        max_y = max(p.y for p in pts)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    @property
    def bounds(self) -> BoundingBox:
        """Convenience property returning get_bounds()."""
        return self.get_bounds()

    def show(self) -> Drawable:
        """Make this object visible."""
        self.visible = True
        return self

    def hide(self) -> Drawable:
        """Hide this object."""
        self.visible = False
        return self

    def lock(self) -> Drawable:
        """Lock this object against mutation."""
        self.locked = True
        return self

    def unlock(self) -> Drawable:
        """Unlock this object."""
        self.locked = False
        return self

    # -------------------------------------------------------------------------
    # Coordinate Spaces & Transformation
    # -------------------------------------------------------------------------

    @property
    def world_matrix(self) -> np.ndarray:
        """Derive the 3x3 affine transformation matrix M mapping local space to world space."""
        default_pivot = self.transform.pivot
        if default_pivot is None:
            default_pivot = self.get_geometry_bounds().center
        local_matrix = self.transform.get_matrix(default_pivot=default_pivot)
        if self._parent is not None:
            return self._parent.world_matrix @ local_matrix
        return local_matrix

    def to_world(self, local_point: Point) -> Point:
        """Map a Point from object local space to world space."""
        default_pivot = self.transform.pivot
        if default_pivot is None:
            default_pivot = self.get_geometry_bounds().center
        pt = self.transform.transform_point(local_point, default_pivot=default_pivot)
        if self._parent is not None:
            return self._parent.to_world(pt)
        return pt

    def to_local(self, world_point: Point) -> Point:
        """Map a Point from world space back to object local space via inverse matrix M^-1."""
        if self._parent is not None:
            world_point = self._parent.to_local(world_point)
        default_pivot = self.get_geometry_bounds().center
        return self.transform.inverse_transform_point(world_point, default_pivot=default_pivot)

    @atomic_transforms
    def move(self, dx: float | int, dy: float | int) -> None:
        """Translate the object by accumulating into transform translation."""
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (dx, dy)):
            raise ValidationError("Move deltas must be numeric")
        self.transform.translation_x += float(dx)
        self.transform.translation_y += float(dy)

    @atomic_transforms
    def rotate(self, degrees: float | int, pivot: Point | None = None) -> None:
        """Rotate the object clockwise by degrees (optional pivot in local space)."""
        if isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or not math.isfinite(degrees):
            raise ValidationError("Rotation angle must be numeric")
        self.transform.rotation += float(degrees)
        if pivot is not None:
            self.transform.pivot = pivot
        elif self.transform.pivot is None:
            self.transform.pivot = self.get_geometry_bounds().center

    @atomic_transforms
    def scale(self, sx: float | int, sy: float | int | None = None, pivot: Point | None = None) -> None:
        """Scale the object horizontally by sx and vertically by sy (sx, sy > 0)."""
        if isinstance(sx, bool) or not isinstance(sx, (int, float)) or not math.isfinite(sx) or sx <= 0:
            raise ValidationError(f"Scale 'sx' must be strictly positive, got {sx}")
        if sy is not None and (isinstance(sy, bool) or not isinstance(sy, (int, float)) or not math.isfinite(sy)):
            raise ValidationError("Scale sy must be finite and numeric")
        sy_val = float(sy) if sy is not None else float(sx)
        if sy_val <= 0:
            raise ValidationError(f"Scale 'sy' must be strictly positive, got {sy_val}")

        self.transform.scale_x *= float(sx)
        self.transform.scale_y *= sy_val
        if pivot is not None:
            self.transform.pivot = pivot
        elif self.transform.pivot is None:
            self.transform.pivot = self.get_geometry_bounds().center

    # -------------------------------------------------------------------------
    # Cascading States (Visibility, Opacity, Locking)
    # -------------------------------------------------------------------------

    @property
    def effective_visible(self) -> bool:
        """Effective visibility considering object, ancestor groups, and layer."""
        if not self.visible:
            return False
        if self._parent is not None and not self._parent.effective_visible:
            return False
        if self._layer is not None and not self._layer.visible:
            return False
        return True

    @property
    def effective_opacity(self) -> float:
        """Effective opacity combining object, ancestor group hierarchy, and layer."""
        op = float(self.opacity)
        if self._parent is not None:
            op *= self._parent.effective_opacity
        if self._layer is not None:
            op *= float(self._layer.opacity)
        return max(0.0, min(1.0, op))

    @property
    def effective_locked(self) -> bool:
        """Effective locked status considering object, ancestor groups, and layer."""
        if self.locked:
            return True
        if self._parent is not None and self._parent.effective_locked:
            return True
        if self._layer is not None and self._layer.locked:
            return True
        return False

    # -------------------------------------------------------------------------
    # Hierarchy & Scene Synchronization
    # -------------------------------------------------------------------------

    @property
    def parent(self) -> Any | None:
        """Parent Group containing this drawable, or None if top-level."""
        return self._parent

    @property
    def layer(self) -> Any | None:
        """Layer containing this drawable (or enclosing group's layer), or None."""
        if self._layer is not None:
            return self._layer
        if self._parent is not None:
            return self._parent.layer
        return None

    @property
    def scene(self) -> Any | None:
        """Scene containing this drawable, or None if unattached."""
        return self._scene

    def _set_scene(self, scene: Any | None) -> None:
        """Propagate scene reference and synchronize with scene ID registry."""
        old_scene = self._scene
        if old_scene is scene:
            return

        if old_scene is not None:
            old_scene._unregister_id(self)

        self._scene = scene

        if scene is not None:
            scene._register_id(self)

    def _reparent(self, new_parent: Any | None, new_layer: Any | None, preserve_world_transform: bool = True) -> None:
        """Re-parent this drawable while preserving its authoritative world-space transform."""
        if preserve_world_transform:
            old_world = self.world_matrix
            if new_parent is not None:
                new_parent_world = new_parent.world_matrix
                new_local = np.linalg.inv(new_parent_world) @ old_world
            else:
                new_local = old_world
            self.transform = Transform.from_matrix(new_local)

        self._parent = new_parent
        self._layer = new_layer

    # -------------------------------------------------------------------------
    # Tags & Metadata Helpers
    # -------------------------------------------------------------------------

    def add_tag(self, tag: str) -> Drawable:
        """Add a tag to this object (returns self for chaining)."""
        if not isinstance(tag, str) or not tag.strip():
            raise ValidationError("Tag must be a non-empty string")
        self.tags.add(tag.strip())
        return self

    def remove_tag(self, tag: str) -> Drawable:
        """Remove a tag from this object if present (returns self for chaining)."""
        self.tags.discard(tag.strip())
        return self

    def has_tag(self, tag: str) -> bool:
        """Check if this object has the specified tag."""
        return tag.strip() in self.tags

    def set_metadata(self, key: str, value: Any) -> Drawable:
        """Set a metadata entry (returns self for chaining)."""
        if not isinstance(key, str) or not key.strip():
            raise ValidationError("Metadata key must be a non-empty string")
        self.metadata[key] = value
        return self

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve a metadata value by key, or default if missing."""
        return self.metadata.get(key, default)

    # -------------------------------------------------------------------------
    # Anchors, Hit-Testing & Cloning
    # -------------------------------------------------------------------------

    @abstractmethod
    def anchor(self, name: str) -> Point:
        """Retrieve a named geometric anchor point mapped to world coordinates."""
        pass

    @abstractmethod
    def contains_point(self, world_point: Point) -> bool:
        """Test whether a world-space Point intersects this object."""
        pass

    def clone(self, new_id: bool = True) -> Drawable:
        """Create a deep copy of this drawable entity."""
        cloned = copy.deepcopy(self)
        if new_id:
            cloned.id = str(uuid.uuid4())
        cloned._parent = None
        cloned._layer = None
        cloned._scene = None
        return cloned

    # -------------------------------------------------------------------------
    # In-Place Semantic State Capture & Restoration (Live Identity Preservation)
    # -------------------------------------------------------------------------

    def _get_shape_state(self) -> dict[str, Any]:
        """Override in subclasses to return dictionary of shape-specific fields."""
        return {}

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        """Override in subclasses to apply shape-specific fields in-place."""
        pass

    def _get_semantic_state(self) -> dict[str, Any]:
        """Capture authoritative semantic state snapshot for in-place undo/redo."""
        state = {
            "id": self.id,
            "name": self.name,
            "visible": self.visible,
            "locked": self.locked,
            "opacity": float(self.opacity),
            "blend_mode": self.blend_mode,
            "z_index": int(self.z_index),
            "tags": sorted(list(self.tags)),
            "metadata": copy.deepcopy(self.metadata),
            "transform": self.transform.copy(),
            "clip": copy.deepcopy(self.clip),
            "mask": copy.deepcopy(self.mask),
            "effects": [copy.deepcopy(e) for e in self.effects],
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "render_progress": float(self.render_progress),
        }
        state.update(self._get_shape_state())
        return state

    def _apply_semantic_state(self, state: dict[str, Any]) -> None:
        """Apply state snapshot in-place to preserve live object identity."""
        self.name = state.get("name")
        self.visible = bool(state.get("visible", True))
        self.locked = bool(state.get("locked", False))
        self.opacity = float(state.get("opacity", 1.0))
        if "blend_mode" in state:
            self.blend_mode = coerce_blend_mode(state["blend_mode"])
        self.z_index = int(state.get("z_index", 0))
        self.tags = set(state.get("tags", []))
        self.metadata = copy.deepcopy(state.get("metadata", {}))

        tf_val = state.get("transform")
        if isinstance(tf_val, Transform):
            self.transform = tf_val.copy()
        elif isinstance(tf_val, dict):
            self.transform = Transform.from_dict(tf_val)

        clip_val = state.get("clip")
        self.clip = copy.deepcopy(clip_val)

        mask_val = state.get("mask")
        self.mask = copy.deepcopy(mask_val)

        effs = state.get("effects", [])
        self.effects = [copy.deepcopy(e) for e in effs]

        timing_val = state.get("timing")
        if isinstance(timing_val, dict):
            from drawcv.animation.timing import Timing
            self.timing = Timing.from_dict(timing_val)
        else:
            self.timing = timing_val
        self.render_progress = float(state.get("render_progress", 1.0))

        self._apply_shape_state(state)

    # -------------------------------------------------------------------------
    # Serialization Helpers
    # -------------------------------------------------------------------------

    def _base_to_dict(self) -> dict[str, Any]:
        """Serialize common retained-mode attributes to plain JSON-compatible primitives."""
        return {
            "id": self.id,
            "name": self.name,
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "opacity": float(self.opacity),
            "blend_mode": self.blend_mode.value,
            "z_index": int(self.z_index),
            "tags": sorted(list(self.tags)),
            "metadata": copy.deepcopy(self.metadata),
            "transform": self.transform.to_dict(),
            "clip": self.clip.to_dict() if self.clip is not None else None,
            "mask": self.mask.to_dict() if self.mask is not None else None,
            "effects": [e.to_dict() for e in self.effects],
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "render_progress": float(self.render_progress),
        }

    @classmethod
    def _base_from_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Parse common retained-mode attributes from serialized data dictionary."""
        from drawcv.effects import clip_from_dict, effect_from_dict, Mask

        clip_val = clip_from_dict(data.get("clip"))
        mask_val = Mask.from_dict(data.get("mask"))
        effs = [effect_from_dict(e) for e in data.get("effects", [])]
        tf_val = Transform.from_dict(data.get("transform"))

        timing_data = data.get("timing")
        from drawcv.animation.timing import Timing
        timing_val = Timing.from_dict(timing_data) if isinstance(timing_data, dict) else None

        return {
            "id": data.get("id", str(uuid.uuid4())),
            "name": data.get("name"),
            "visible": bool(data.get("visible", True)),
            "locked": bool(data.get("locked", False)),
            "opacity": float(data.get("opacity", 1.0)),
            "blend_mode": coerce_blend_mode(data.get("blend_mode", BlendMode.NORMAL)),
            "z_index": int(data.get("z_index", 0)),
            "tags": set(data.get("tags", [])),
            "metadata": copy.deepcopy(data.get("metadata", {})),
            "transform": tf_val,
            "clip": clip_val,
            "mask": mask_val,
            "effects": effs,
            "timing": timing_val,
            "render_progress": float(data.get("render_progress", 1.0)),
        }

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Drawable:
        """Construct a Drawable from dictionary representation by dispatching to registered type."""
        from drawcv.serialization.registry import get_drawable_deserializer
        if not isinstance(data, dict):
            raise ValidationError(f"Drawable data must be a dict, got {type(data).__name__}")
        drawable_type = data.get("type")
        if not drawable_type:
            raise ValidationError("Drawable data dictionary missing required 'type' field")
        deserializer = get_drawable_deserializer(drawable_type)
        return deserializer(data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Drawable):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        return hash(self.id)

