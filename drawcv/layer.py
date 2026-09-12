"""Rendering Layer organization for DrawCV scenes."""

from __future__ import annotations
from typing import Any, Iterator, TypeVar

from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError

T = TypeVar("T", bound=Drawable)


class Layer:
    """A rendering organization layer managing top-level scene drawables and global stacking order.
    
    Layers represent independent rendering passes or semantic document levels
    (e.g., background, main, annotations, foreground).
    """

    def __init__(
        self,
        name: str,
        *,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
        z_order: int = 0,
        scene: Any | None = None,
    ):
        self._validate_params(name, visible, locked, opacity, z_order)
        self.name = name
        self.visible = visible
        self.locked = locked
        self.opacity = float(opacity)
        self.z_order = z_order
        self._scene: Any | None = scene
        self._objects: list[Drawable] = []

    def _validate_params(self, name: str, visible: bool, locked: bool, opacity: float, z_order: int):
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Layer name must be a non-empty string")
        if not isinstance(visible, bool):
            raise ValidationError("Layer visible must be a boolean")
        if not isinstance(locked, bool):
            raise ValidationError("Layer locked must be a boolean")
        if not isinstance(opacity, (int, float)) or isinstance(opacity, bool):
            raise ValidationError("Layer opacity must be numeric")
        if not (0.0 <= float(opacity) <= 1.0):
            raise ValidationError(f"Layer opacity must be in range [0.0, 1.0], got {opacity}")
        if not isinstance(z_order, int) or isinstance(z_order, bool):
            raise ValidationError("Layer z_order must be an integer")

    # -------------------------------------------------------------------------
    # Object Management & Invariants
    # -------------------------------------------------------------------------

    def add(self, drawable: Drawable) -> None:
        """Add a top-level drawable to this layer.
        
        Enforces strict tree invariants:
        - Detaches from any prior parent Group or Layer.
        - Synchronizes ID registration with owning Scene.
        """
        if not isinstance(drawable, Drawable):
            raise ValidationError(f"Expected Drawable, got {type(drawable).__name__}")
        if drawable in self._objects:
            raise ValidationError(f"Drawable with id '{drawable.id}' already exists in layer '{self.name}'")

        # Detach from prior parent group if any
        if drawable._parent is not None:
            drawable._parent.remove(drawable, preserve_world_transform=False)

        # Detach from prior layer if any
        if drawable._layer is not None and drawable._layer is not self:
            drawable._layer.remove(drawable)

        # If bound to a scene, verify ID uniqueness
        if self._scene is not None:
            self._scene._check_id_unique(drawable)

        drawable._layer = self
        drawable._parent = None
        self._objects.append(drawable)

        # Propagate scene registry down the drawable subtree
        if self._scene is not None:
            drawable._set_scene(self._scene)

    def remove(self, drawable_or_id: Drawable | str) -> Drawable:
        """Remove a top-level drawable from this layer by reference or ID."""
        target: Drawable | None = None
        if isinstance(drawable_or_id, Drawable):
            for obj in self._objects:
                if obj is drawable_or_id:
                    target = obj
                    break
        elif isinstance(drawable_or_id, str):
            for obj in self._objects:
                if obj.id == drawable_or_id:
                    target = obj
                    break
        else:
            raise ValidationError("Expected Drawable or string ID")

        if target is None:
            id_str = drawable_or_id.id if isinstance(drawable_or_id, Drawable) else drawable_or_id
            raise ObjectNotFoundError(f"Drawable with id '{id_str}' not found in layer '{self.name}'")

        self._objects.remove(target)

        # Unregister from scene if attached
        if self._scene is not None:
            target._set_scene(None)

        target._layer = None
        return target

    def get(self, id: str) -> Drawable | None:
        """Retrieve a direct top-level drawable by its ID."""
        for obj in self._objects:
            if obj.id == id:
                return obj
        return None

    def clear(self) -> None:
        """Remove and unbind all drawables from this layer."""
        while self._objects:
            self.remove(self._objects[-1])

    @property
    def objects(self) -> list[Drawable]:
        """Return a shallow copy list of top-level drawables in this layer."""
        return list(self._objects)

    @property
    def scene(self) -> Any | None:
        """Owning Scene reference, or None if unattached."""
        return self._scene

    def __iter__(self) -> Iterator[Drawable]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int) -> Drawable:
        return self._objects[index]

    # -------------------------------------------------------------------------
    # Scene Hierarchy Synchronization
    # -------------------------------------------------------------------------

    def _set_scene(self, scene: Any | None) -> None:
        """Bind or unbind this layer to a Scene, propagating to all contained objects."""
        self._scene = scene
        for obj in self._objects:
            obj._set_scene(scene)

    # -------------------------------------------------------------------------
    # Layer State Modifiers
    # -------------------------------------------------------------------------

    def show(self) -> None:
        """Make this layer visible."""
        self.visible = True

    def hide(self) -> None:
        """Hide this layer and all its contents."""
        self.visible = False

    def lock(self) -> None:
        """Lock this layer to prevent mutation."""
        self.locked = True

    def unlock(self) -> None:
        """Unlock this layer."""
        self.locked = False

    def __repr__(self) -> str:
        return f"Layer(name='{self.name}', z_order={self.z_order}, objects={len(self._objects)}, visible={self.visible}, locked={self.locked})"
