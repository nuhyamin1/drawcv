"""Scene graph and retained-mode object manager for DrawCV."""

from __future__ import annotations
from typing import Any, Callable, Iterator, TypeVar

from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError
from drawcv.core.geometry import Point
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.selection import Selection

T = TypeVar("T", bound=Drawable)


class Scene:
    """Retained-mode drawing document model.
    
    The Scene is the authoritative source of document dimensions, background color,
    rendering layers, global ID registry, and all retained drawable entities.
    """

    def __init__(self, width: int, height: int, background: Color = Color.white()):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValidationError(f"Scene width must be a positive integer, got {width}")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValidationError(f"Scene height must be a positive integer, got {height}")
        if not isinstance(background, Color):
            raise ValidationError(f"Scene background must be a Color, got {type(background).__name__}")

        self.width = width
        self.height = height
        self.background = background

        self._layers: dict[str, Layer] = {}
        self._layer_order: list[str] = []
        self._id_map: dict[str, Drawable] = {}

        # Initialize authoritative default layer
        self.create_layer("default", z_order=0)

    # -------------------------------------------------------------------------
    # Layer Management
    # -------------------------------------------------------------------------

    def create_layer(
        self,
        name: str,
        z_order: int | None = None,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
    ) -> Layer:
        """Create and register a new rendering Layer."""
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Layer name must be a non-empty string")
        clean_name = name.strip()
        if clean_name in self._layers:
            raise ValidationError(f"Layer '{clean_name}' already exists in scene")

        if z_order is None:
            max_z = max((l.z_order for l in self._layers.values()), default=-10)
            resolved_z = max_z + 10
        else:
            resolved_z = z_order

        layer = Layer(
            name=clean_name,
            visible=visible,
            locked=locked,
            opacity=opacity,
            z_order=resolved_z,
            scene=self,
        )
        self._layers[clean_name] = layer
        self._layer_order.append(clean_name)
        return layer

    def get_layer(self, name: str) -> Layer | None:
        """Retrieve a Layer by name, or None if not present."""
        return self._layers.get(name)

    def remove_layer(self, name: str, remove_objects: bool = True) -> Layer:
        """Remove a Layer from the scene.
        
        The 'default' layer is protected and cannot be removed.
        If remove_objects=True, all layer drawables are recursively removed and unregistered.
        If remove_objects=False, top-level objects are migrated to the 'default' layer.
        """
        if name == "default":
            raise ValidationError("Cannot remove the default layer")
        if name not in self._layers:
            raise ObjectNotFoundError(f"Layer '{name}' not found in scene")

        layer = self._layers.pop(name)
        self._layer_order.remove(name)

        if remove_objects:
            layer.clear()
        else:
            default_layer = self._layers["default"]
            for obj in list(layer.objects):
                layer.remove(obj)
                default_layer.add(obj)

        layer._set_scene(None)
        return layer

    @property
    def layers(self) -> list[Layer]:
        """Return all layers sorted by ascending z_order (creation order as tie-breaker)."""
        layer_items = [
            (idx, self._layers[name]) for idx, name in enumerate(self._layer_order)
            if name in self._layers
        ]
        sorted_layers = sorted(layer_items, key=lambda item: (item[1].z_order, item[0]))
        return [layer for _, layer in sorted_layers]

    def move_layer_to_front(self, name: str) -> None:
        """Move layer to highest z_order."""
        layer = self._get_layer_or_raise(name)
        max_z = max((l.z_order for l in self._layers.values()), default=0)
        layer.z_order = max_z + 10

    def move_layer_to_back(self, name: str) -> None:
        """Move layer to lowest z_order."""
        layer = self._get_layer_or_raise(name)
        min_z = min((l.z_order for l in self._layers.values()), default=0)
        layer.z_order = min_z - 10

    def move_layer_forward(self, name: str) -> None:
        """Increment layer z_order."""
        layer = self._get_layer_or_raise(name)
        layer.z_order += 1

    def move_layer_backward(self, name: str) -> None:
        """Decrement layer z_order."""
        layer = self._get_layer_or_raise(name)
        layer.z_order -= 1

    def show_layer(self, name: str) -> None:
        """Show the specified layer."""
        self._get_layer_or_raise(name).show()

    def hide_layer(self, name: str) -> None:
        """Hide the specified layer."""
        self._get_layer_or_raise(name).hide()

    def lock_layer(self, name: str) -> None:
        """Lock the specified layer."""
        self._get_layer_or_raise(name).lock()

    def unlock_layer(self, name: str) -> None:
        """Unlock the specified layer."""
        self._get_layer_or_raise(name).unlock()

    def set_layer_opacity(self, name: str, opacity: float) -> None:
        """Set opacity for the specified layer."""
        layer = self._get_layer_or_raise(name)
        if not (0.0 <= float(opacity) <= 1.0):
            raise ValidationError("Layer opacity must be between 0.0 and 1.0")
        layer.opacity = float(opacity)

    def _get_layer_or_raise(self, name: str) -> Layer:
        if name not in self._layers:
            raise ObjectNotFoundError(f"Layer '{name}' not found in scene")
        return self._layers[name]

    # -------------------------------------------------------------------------
    # ID Registry Management
    # -------------------------------------------------------------------------

    def _check_id_unique(self, drawable: Drawable) -> None:
        """Recursively verify that drawable and all its descendants have unique IDs."""
        if drawable.id in self._id_map:
            raise ValidationError(f"Drawable with id '{drawable.id}' already exists in scene")
        if isinstance(drawable, Group):
            for child in drawable.children:
                self._check_id_unique(child)

    def _register_id(self, drawable: Drawable) -> None:
        """Register drawable and all its descendants into the scene ID map."""
        if drawable.id in self._id_map and self._id_map[drawable.id] is not drawable:
            raise ValidationError(f"Duplicate ID collision: '{drawable.id}' already in scene")
        self._id_map[drawable.id] = drawable
        if isinstance(drawable, Group):
            for child in drawable.children:
                self._register_id(child)

    def _unregister_id(self, drawable: Drawable) -> None:
        """Unregister drawable and all its descendants from the scene ID map."""
        self._id_map.pop(drawable.id, None)
        if isinstance(drawable, Group):
            for child in drawable.children:
                self._unregister_id(child)

    # -------------------------------------------------------------------------
    # Object Management
    # -------------------------------------------------------------------------

    def add(self, drawable: Drawable, layer: str = "default") -> None:
        """Add a top-level drawable to the specified layer (default: 'default')."""
        target_layer = self.get_layer(layer)
        if target_layer is None:
            raise ObjectNotFoundError(f"Layer '{layer}' does not exist in scene")
        target_layer.add(drawable)

    def remove(self, id: str) -> Drawable:
        """Remove a drawable by its ID from whichever parent group or layer holds it."""
        obj = self.get(id)
        if obj is None:
            raise ObjectNotFoundError(f"Drawable with id '{id}' not found in scene")
        if obj.effective_locked:
            raise ValidationError(f"Cannot remove locked drawable '{id}'")

        if obj._parent is not None:
            return obj._parent.remove(obj)
        elif obj._layer is not None:
            return obj._layer.remove(obj)
        else:
            self._unregister_id(obj)
            return obj

    def get(self, id: str) -> Drawable | None:
        """Fast O(1) retrieval of any drawable in the scene (including inside groups)."""
        return self._id_map.get(id)

    def clear(self) -> None:
        """Remove all objects and layers (re-initializing default layer)."""
        for layer in list(self._layers.values()):
            layer.clear()
        self._layers.clear()
        self._layer_order.clear()
        self._id_map.clear()
        self.create_layer("default", z_order=0)

    @property
    def objects(self) -> list[Drawable]:
        """Return all top-level drawables across all layers in rendering order."""
        all_objs: list[Drawable] = []
        for layer in self.layers:
            all_objs.extend(layer.objects)
        return all_objs

    # -------------------------------------------------------------------------
    # Object State Modifiers
    # -------------------------------------------------------------------------

    def show(self, id: str) -> None:
        """Set visible=True for the specified object."""
        self._get_or_raise(id).visible = True

    def hide(self, id: str) -> None:
        """Set visible=False for the specified object."""
        self._get_or_raise(id).visible = False

    def lock(self, id: str) -> None:
        """Set locked=True for the specified object."""
        self._get_or_raise(id).locked = True

    def unlock(self, id: str) -> None:
        """Set locked=False for the specified object."""
        self._get_or_raise(id).locked = False

    def move_to_front(self, id: str) -> None:
        """Move object to front within its container (group or layer)."""
        obj = self._get_or_raise(id)
        if obj._parent is not None:
            max_z = max((o.z_index for o in obj._parent.children), default=0)
            obj.z_index = max_z + 1
            if obj in obj._parent._children:
                obj._parent._children.remove(obj)
                obj._parent._children.append(obj)
        elif obj._layer is not None:
            max_z = max((o.z_index for o in obj._layer.objects), default=0)
            obj.z_index = max_z + 1
            if obj in obj._layer._objects:
                obj._layer._objects.remove(obj)
                obj._layer._objects.append(obj)

    def move_to_back(self, id: str) -> None:
        """Move object to back within its container (group or layer)."""
        obj = self._get_or_raise(id)
        if obj._parent is not None:
            min_z = min((o.z_index for o in obj._parent.children), default=0)
            obj.z_index = min_z - 1
            if obj in obj._parent._children:
                obj._parent._children.remove(obj)
                obj._parent._children.insert(0, obj)
        elif obj._layer is not None:
            min_z = min((o.z_index for o in obj._layer.objects), default=0)
            obj.z_index = min_z - 1
            if obj in obj._layer._objects:
                obj._layer._objects.remove(obj)
                obj._layer._objects.insert(0, obj)

    def move_forward(self, id: str) -> None:
        """Increment object z-index."""
        self._get_or_raise(id).z_index += 1

    def move_backward(self, id: str) -> None:
        """Decrement object z-index."""
        self._get_or_raise(id).z_index -= 1

    # -------------------------------------------------------------------------
    # Lookup & Queries
    # -------------------------------------------------------------------------

    def find(
        self,
        predicate: Callable[[Drawable], bool] | None = None,
        recursive: bool = True,
        **kwargs: Any
    ) -> list[Drawable]:
        """Search all drawables matching an optional predicate and attribute kwargs."""
        results: list[Drawable] = []
        target_pool = self._id_map.values() if recursive else self.objects

        for obj in target_pool:
            match = True
            if predicate is not None and not predicate(obj):
                match = False
            if match and kwargs:
                for key, val in kwargs.items():
                    if hasattr(obj, key):
                        if getattr(obj, key) != val:
                            match = False
                            break
                    elif key in obj.metadata:
                        if obj.metadata[key] != val:
                            match = False
                            break
                    else:
                        match = False
                        break
            if match:
                results.append(obj)
        return results

    def find_first(
        self,
        predicate: Callable[[Drawable], bool] | None = None,
        recursive: bool = True,
        **kwargs: Any
    ) -> Drawable | None:
        """Find the first matching drawable, or None."""
        matches = self.find(predicate=predicate, recursive=recursive, **kwargs)
        return matches[0] if matches else None

    def find_by_name(self, name: str, recursive: bool = True) -> list[Drawable]:
        """Find all drawables matching name."""
        return self.find(name=name, recursive=recursive)

    def find_by_tag(self, tag: str, recursive: bool = True) -> list[Drawable]:
        """Find all drawables having the specified tag."""
        return [
            obj for obj in (self._id_map.values() if recursive else self.objects)
            if tag in obj.tags
        ]

    def find_by_type(self, cls: type[T], recursive: bool = True) -> list[T]:
        """Find all drawables that are instances of cls."""
        return [
            obj for obj in (self._id_map.values() if recursive else self.objects)
            if isinstance(obj, cls)
        ]

    def find_by_metadata(self, key: str, value: Any = None, recursive: bool = True) -> list[Drawable]:
        """Find all drawables containing metadata key (and matching value if given)."""
        pool = self._id_map.values() if recursive else self.objects
        if value is None:
            return [obj for obj in pool if key in obj.metadata]
        return [obj for obj in pool if obj.metadata.get(key) == value]

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select(self, items: list[Drawable | str] | None = None) -> Selection:
        """Create a logical Selection across the specified drawables or IDs."""
        return Selection(self, items)

    def select_all(self) -> Selection:
        """Select all drawables in the scene."""
        return Selection(self, list(self._id_map.values()))

    def select_by_tag(self, tag: str) -> Selection:
        """Select all drawables with the given tag."""
        return Selection(self, self.find_by_tag(tag))

    def select_by_type(self, cls: type[T]) -> Selection:
        """Select all drawables of the specified type."""
        return Selection(self, self.find_by_type(cls))

    # -------------------------------------------------------------------------
    # Hit Testing (Deterministic Reverse Painter's Order)
    # -------------------------------------------------------------------------

    def hit_test(self, x: float | int, y: float | int) -> list[Drawable]:
        """Query all visible objects intersecting (x, y) in topmost-first order.
        
        Traverses:
        1. Layers: descending layer.z_order (skips hidden layers).
        2. Within Layer: descending (drawable.z_index, insertion_order).
        3. Within Groups: recursively descending (child.z_index, child_order).
        """
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValidationError("Hit-test coordinates must be numeric")

        point = Point(x, y)
        hits: list[Drawable] = []

        # Traverse layers in descending order (topmost layer first)
        for layer in reversed(self.layers):
            if not layer.visible or layer.opacity <= 0.0:
                continue

            visible_items = [
                (idx, obj) for idx, obj in enumerate(layer.objects)
                if obj.visible and obj.opacity > 0.0
            ]
            sorted_top_first = sorted(
                visible_items,
                key=lambda item: (item[1].z_index, item[0]),
                reverse=True
            )

            for _, obj in sorted_top_first:
                self._hit_test_recursive(obj, point, hits)

        return hits

    def _hit_test_recursive(self, drawable: Drawable, point: Point, hits: list[Drawable]) -> None:
        if not drawable.effective_visible or drawable.effective_opacity <= 0.0:
            return

        if isinstance(drawable, Group):
            # Check group children topmost-first
            indexed_children = [
                (idx, child) for idx, child in enumerate(drawable.children)
                if child.visible and child.opacity > 0.0
            ]
            sorted_children = sorted(
                indexed_children,
                key=lambda item: (item[1].z_index, item[0]),
                reverse=True
            )
            for _, child in sorted_children:
                self._hit_test_recursive(child, point, hits)
            # Also register the group itself if its children were hit or if it intersects
            if drawable.contains_point(point) and drawable not in hits:
                hits.append(drawable)
        else:
            if drawable.contains_point(point):
                hits.append(drawable)

    def hit_test_top(self, x: float | int, y: float | int) -> Drawable | None:
        """Query the single topmost visible object intersecting (x, y), or None."""
        hits = self.hit_test(x, y)
        return hits[0] if hits else None

    def _get_or_raise(self, id: str) -> Drawable:
        obj = self.get(id)
        if obj is None:
            raise ObjectNotFoundError(f"Drawable with id '{id}' not found in scene")
        return obj

    def __len__(self) -> int:
        return len(self._id_map)

    def __iter__(self) -> Iterator[Drawable]:
        return iter(self.objects)

    def __repr__(self) -> str:
        return f"Scene(width={self.width}, height={self.height}, layers={len(self._layers)}, objects={len(self._id_map)})"
