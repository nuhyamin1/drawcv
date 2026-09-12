"""Abstract base class for all retained-mode drawable objects in DrawCV."""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
import uuid
from typing import Any
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform


@dataclass(eq=False)
class Drawable(ABC):
    """Abstract base class for retained-mode drawing entities.
    
    All drawables maintain identity, appearance flags, metadata, tags,
    and an affine transform hierarchy separating local geometry from world space.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    z_index: int = 0
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    transform: Transform = field(default_factory=Transform)
    _parent: Any | None = field(default=None, repr=False, compare=False)
    _layer: Any | None = field(default=None, repr=False, compare=False)
    _scene: Any | None = field(default=None, repr=False, compare=False)

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
        if not isinstance(self.z_index, int) or isinstance(self.z_index, bool):
            raise ValidationError("Drawable 'z_index' must be an integer")
        if not isinstance(self.tags, set):
            raise ValidationError("Drawable 'tags' must be a set")
        if not isinstance(self.metadata, dict):
            raise ValidationError("Drawable 'metadata' must be a dict")
        if not isinstance(self.transform, Transform):
            raise ValidationError("Drawable 'transform' must be a Transform instance")

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

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
        geom_bounds = self.get_geometry_bounds()
        local_pivot = self.transform.pivot if self.transform.pivot is not None else geom_bounds.center
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
        default_pivot = self.get_geometry_bounds().center
        local_matrix = self.transform.get_matrix(default_pivot=default_pivot)
        if self._parent is not None:
            return self._parent.world_matrix @ local_matrix
        return local_matrix

    def to_world(self, local_point: Point) -> Point:
        """Map a Point from object local space to world space."""
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

    def move(self, dx: float | int, dy: float | int) -> None:
        """Translate the object by accumulating into transform translation."""
        if not isinstance(dx, (int, float)) or not isinstance(dy, (int, float)):
            raise ValidationError("Move deltas must be numeric")
        self.transform.translation_x += float(dx)
        self.transform.translation_y += float(dy)

    def rotate(self, degrees: float | int, pivot: Point | None = None) -> None:
        """Rotate the object clockwise by degrees (optional pivot in local space)."""
        if not isinstance(degrees, (int, float)):
            raise ValidationError("Rotation angle must be numeric")
        self.transform.rotation += float(degrees)
        if pivot is not None:
            self.transform.pivot = pivot
        elif self.transform.pivot is None:
            self.transform.pivot = self.get_geometry_bounds().center

    def scale(self, sx: float | int, sy: float | int | None = None, pivot: Point | None = None) -> None:
        """Scale the object horizontally by sx and vertically by sy (sx, sy > 0)."""
        if not isinstance(sx, (int, float)) or sx <= 0:
            raise ValidationError(f"Scale 'sx' must be strictly positive, got {sx}")
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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Drawable):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        return hash(self.id)
