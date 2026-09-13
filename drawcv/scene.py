"""Scene graph and retained-mode object manager for DrawCV."""

from __future__ import annotations
import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, TypeVar

from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.group import Group
from drawcv.history.command import CompoundCommand
from drawcv.history.commands import (
    AddObjectCommand,
    GroupCommand,
    RemoveObjectCommand,
    ReorderCommand,
    StateEditCommand,
    TransformCommand,
    UngroupCommand,
)
from drawcv.history.manager import HistoryManager
from drawcv.layer import Layer
from drawcv.selection import Selection
from drawcv.serialization import (
    CURRENT_FORMAT_IDENTIFIER,
    CURRENT_SCHEMA_VERSION,
    from_json as parse_json,
    to_json as serialize_json,
)
from drawcv.serialization.registry import SchemaMigrator

T = TypeVar("T", bound=Drawable)


class Scene:
    """Retained-mode drawing document model.
    
    The Scene is the authoritative source of document dimensions, background color,
    rendering layers, global ID registry, reversible history, and all retained drawable entities.
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
        self.history = HistoryManager()
        from drawcv.animation.timeline import Timeline
        self.timeline = Timeline()

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
        clip: Any | None = None,
        mask: Any | None = None,
        effects: list[Any] | None = None,
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
            clip=clip,
            mask=mask,
            effects=effects,
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
    # Untracked Internal Primitives (Execution Path for Commands & Loading)
    # -------------------------------------------------------------------------

    def _add_untracked(self, drawable: Drawable, layer: str = "default", index: int | None = None) -> None:
        """Internal untracked insertion preserving exact live instance."""
        target_layer = self.get_layer(layer)
        if target_layer is None:
            raise ObjectNotFoundError(f"Layer '{layer}' does not exist in scene")
        self._check_id_unique(drawable)

        drawable._layer = target_layer
        drawable._parent = None
        if index is not None and 0 <= index <= len(target_layer._objects):
            target_layer._objects.insert(index, drawable)
        else:
            target_layer._objects.append(drawable)

        drawable._set_scene(self)

    def _remove_untracked(self, id: str) -> Drawable:
        """Internal untracked removal of drawable from its container."""
        obj = self.get(id)
        if obj is None:
            raise ObjectNotFoundError(f"Drawable with id '{id}' not found in scene")

        if obj._parent is not None:
            return obj._parent.remove(obj)
        elif obj._layer is not None:
            return obj._layer.remove(obj)
        else:
            self._unregister_id(obj)
            return obj

    # -------------------------------------------------------------------------
    # Object Management (Tracked Public Operations)
    # -------------------------------------------------------------------------

    def add(self, drawable: Drawable, layer: str = "default") -> None:
        """Add a top-level drawable to the specified layer (default: 'default') with history tracking."""
        target_layer = self.get_layer(layer)
        if target_layer is None:
            raise ObjectNotFoundError(f"Layer '{layer}' does not exist in scene")
        idx = len(target_layer._objects)
        self._add_untracked(drawable, layer=layer)
        self.history.record(
            AddObjectCommand(self, drawable, layer_name=layer, index=idx)
        )

    def remove(self, id: str) -> Drawable:
        """Remove a drawable by its ID from whichever parent group or layer holds it with history tracking."""
        obj = self.get(id)
        if obj is None:
            raise ObjectNotFoundError(f"Drawable with id '{id}' not found in scene")
        if obj.effective_locked:
            raise ValidationError(f"Cannot remove locked drawable '{id}'")

        parent_grp = obj._parent
        layer_nm = obj._layer.name if obj._layer is not None else None
        if parent_grp is not None:
            idx = parent_grp._children.index(obj) if obj in parent_grp._children else -1
        elif obj._layer is not None:
            idx = obj._layer._objects.index(obj) if obj in obj._layer._objects else -1
        else:
            idx = -1

        removed = self._remove_untracked(id)
        self.history.record(
            RemoveObjectCommand(
                scene=self,
                drawable=removed,
                layer_name=layer_nm,
                parent_group=parent_grp,
                index=idx,
            )
        )
        return removed

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
        self.history.clear()
        self.create_layer("default", z_order=0)

    @property
    def objects(self) -> list[Drawable]:
        """Return all top-level drawables across all layers in rendering order."""
        all_objs: list[Drawable] = []
        for layer in self.layers:
            all_objs.extend(layer.objects)
        return all_objs

    # -------------------------------------------------------------------------
    # Object State Modifiers & History-Tracked Transformations
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
        """Move object to front within its container (group or layer) with history tracking."""
        obj = self._get_or_raise(id)
        if obj._parent is not None:
            container = obj._parent
            pool = container._children
        elif obj._layer is not None:
            container = obj._layer
            pool = container._objects
        else:
            return

        old_idx = pool.index(obj) if obj in pool else -1
        old_z = obj.z_index
        max_z = max((o.z_index for o in pool), default=0)
        new_z = max_z + 1
        obj.z_index = new_z
        if obj in pool:
            pool.remove(obj)
            pool.append(obj)
        new_idx = len(pool) - 1
        self.history.record(ReorderCommand(container, obj, old_idx, new_idx, old_z, new_z, description="Move to Front"))

    def move_to_back(self, id: str) -> None:
        """Move object to back within its container (group or layer) with history tracking."""
        obj = self._get_or_raise(id)
        if obj._parent is not None:
            container = obj._parent
            pool = container._children
        elif obj._layer is not None:
            container = obj._layer
            pool = container._objects
        else:
            return

        old_idx = pool.index(obj) if obj in pool else -1
        old_z = obj.z_index
        min_z = min((o.z_index for o in pool), default=0)
        new_z = min_z - 1
        obj.z_index = new_z
        if obj in pool:
            pool.remove(obj)
            pool.insert(0, obj)
        new_idx = 0
        self.history.record(ReorderCommand(container, obj, old_idx, new_idx, old_z, new_z, description="Move to Back"))

    def move_forward(self, id: str) -> None:
        """Increment object z-index with history tracking."""
        obj = self._get_or_raise(id)
        container = obj._parent or obj._layer
        if container is None:
            return
        pool = container._children if hasattr(container, "_children") else container._objects
        old_idx = pool.index(obj) if obj in pool else -1
        old_z = obj.z_index
        new_z = old_z + 1
        obj.z_index = new_z
        self.history.record(ReorderCommand(container, obj, old_idx, old_idx, old_z, new_z, description="Move Forward"))

    def move_backward(self, id: str) -> None:
        """Decrement object z-index with history tracking."""
        obj = self._get_or_raise(id)
        container = obj._parent or obj._layer
        if container is None:
            return
        pool = container._children if hasattr(container, "_children") else container._objects
        old_idx = pool.index(obj) if obj in pool else -1
        old_z = obj.z_index
        new_z = old_z - 1
        obj.z_index = new_z
        self.history.record(ReorderCommand(container, obj, old_idx, old_idx, old_z, new_z, description="Move Backward"))

    def move_object(self, id: str, dx: float | int, dy: float | int) -> None:
        """Translate an object with history tracking."""
        obj = self._get_or_raise(id)
        before = obj.transform.copy()
        obj.move(dx, dy)
        after = obj.transform.copy()
        self.history.record(TransformCommand(obj, before, after, description="Move Object"))

    def rotate_object(self, id: str, degrees: float | int, pivot: Point | None = None) -> None:
        """Rotate an object with history tracking."""
        obj = self._get_or_raise(id)
        before = obj.transform.copy()
        obj.rotate(degrees, pivot=pivot)
        after = obj.transform.copy()
        self.history.record(TransformCommand(obj, before, after, description="Rotate Object"))

    def scale_object(self, id: str, sx: float | int, sy: float | int | None = None, pivot: Point | None = None) -> None:
        """Scale an object with history tracking."""
        obj = self._get_or_raise(id)
        before = obj.transform.copy()
        obj.scale(sx, sy=sy, pivot=pivot)
        after = obj.transform.copy()
        self.history.record(TransformCommand(obj, before, after, description="Scale Object"))

    def restyle_object(self, id: str, **styles: Any) -> None:
        """Modify object styling with in-place history tracking."""
        obj = self._get_or_raise(id)
        with self.edit(obj, name="Restyle Object"):
            for k, v in styles.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
                else:
                    raise ValidationError(f"Unknown style attribute '{k}' for {type(obj).__name__}")

    def group(self, drawables: list[Drawable | str], name: str | None = None) -> Group:
        """Group drawables into a new Group container with history tracking."""
        if not drawables:
            raise ValidationError("Cannot create an empty group")
        resolved: list[Drawable] = []
        for item in drawables:
            obj = self._get_or_raise(item if isinstance(item, str) else item.id)
            if obj.effective_locked:
                raise ValidationError(f"Cannot group locked drawable '{obj.id}'")
            if obj not in resolved:
                resolved.append(obj)

        first = resolved[0]
        target_layer = first.layer
        target_layer_name = target_layer.name if target_layer is not None else "default"

        children_info: list[dict[str, Any]] = []
        group_index = 0
        for d in resolved:
            orig_parent = d._parent
            orig_layer = d._layer.name if d._layer is not None else target_layer_name
            if orig_parent is not None:
                orig_idx = orig_parent._children.index(d) if d in orig_parent._children else 0
            elif d._layer is not None:
                orig_idx = d._layer._objects.index(d) if d in d._layer._objects else 0
            else:
                orig_idx = 0
            if d is first:
                group_index = orig_idx

            children_info.append({
                "child": d,
                "orig_parent": orig_parent,
                "orig_layer": orig_layer,
                "orig_index": orig_idx,
                "orig_transform": d.transform.copy(),
                "grouped_transform": Transform.from_matrix(d.world_matrix),
            })

        new_group = Group(name=name)
        cmd = GroupCommand(
            scene=self,
            group=new_group,
            children_info=children_info,
            layer_name=target_layer_name,
            group_index=group_index,
        )
        cmd.execute()
        self.history.record(cmd)
        return new_group

    def ungroup(self, group_or_id: Group | str) -> list[Drawable]:
        """Unpack a Group into its enclosing layer with history tracking."""
        group_obj = self._get_or_raise(group_or_id if isinstance(group_or_id, str) else group_or_id.id)
        if not isinstance(group_obj, Group):
            raise ValidationError(f"Object '{group_obj.id}' is not a Group")
        if group_obj.effective_locked:
            raise ValidationError(f"Cannot ungroup locked group '{group_obj.id}'")

        layer = group_obj.layer
        layer_name = layer.name if layer is not None else "default"
        group_index = layer._objects.index(group_obj) if (layer and group_obj in layer._objects) else 0

        children_info: list[dict[str, Any]] = []
        for child in list(group_obj.children):
            children_info.append({
                "child": child,
                "orig_transform": child.transform.copy(),
                "unpacked_transform": Transform.from_matrix(child.world_matrix),
            })

        cmd = UngroupCommand(
            scene=self,
            group=group_obj,
            children_info=children_info,
            layer_name=layer_name,
            group_index=group_index,
        )
        cmd.execute()
        self.history.record(cmd)
        return [info["child"] for info in children_info]

    # -------------------------------------------------------------------------
    # History Operations & Context Managers
    # -------------------------------------------------------------------------

    @contextmanager
    def edit(self, *drawables: Drawable | str, name: str = "Edit") -> Generator[list[Drawable], None, None]:
        """Context manager for tracked in-place mutations.
        
        Normalizes targets:
        - Resolves IDs to live Drawable instances.
        - Discards duplicates.
        - Ancestor deduplication: if target A is a descendant of target B, target A is pruned
          because target B's recursive semantic state captures all descendants.
        - Captures before snapshots.
        - Rolls back on unhandled exceptions.
        - Records StateEditCommand preserving live object identities.
        """
        resolved: list[Drawable] = []
        for item in drawables:
            obj = self._get_or_raise(item if isinstance(item, str) else item.id)
            if obj not in resolved:
                resolved.append(obj)

        resolved_set = set(resolved)
        targets: list[Drawable] = []
        for obj in resolved:
            curr = obj._parent
            has_ancestor_in_targets = False
            while curr is not None:
                if curr in resolved_set:
                    has_ancestor_in_targets = True
                    break
                curr = curr._parent
            if not has_ancestor_in_targets:
                targets.append(obj)

        befores = [d._get_semantic_state() for d in targets]

        try:
            yield targets
            # Dataclass geometry can be edited directly: validate it before
            # recording a successful transaction, including group descendants.
            def validate_raw(obj):
                # Run raw-field checks before serializers can coerce invalid
                # types (e.g. bool radius to float). Normalize only a candidate.
                copy.copy(obj).__post_init__()
                if isinstance(obj, Group):
                    for child in obj.children:
                        validate_raw(child)

            for d in targets:
                validate_raw(d)
                type(d).from_dict(d.to_dict())
        except Exception:
            with self.history.suspended():
                for d, b in zip(targets, befores):
                    d._apply_semantic_state(b)
            raise

        afters = [d._get_semantic_state() for d in targets]

        cmd = StateEditCommand(
            targets=[(d, b, a) for d, b, a in zip(targets, befores, afters)],
            description=name,
        )
        self.history.record(cmd)

    def batch(self, name: str = "Batch") -> Generator[CompoundCommand, None, None]:
        """Context manager to group multiple operations into an atomic compound command."""
        return self.history.batch(name=name)

    def undo(self) -> bool:
        """Undo the most recent command."""
        return self.history.undo()

    def redo(self) -> bool:
        """Re-apply the most recently undone command."""
        return self.history.redo()

    @property
    def can_undo(self) -> bool:
        """Whether there are actions to undo."""
        return self.history.can_undo

    @property
    def can_redo(self) -> bool:
        """Whether there are actions to redo."""
        return self.history.can_redo

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
        """Query all visible objects intersecting (x, y) in topmost-first order."""
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValidationError("Hit-test coordinates must be numeric")

        point = Point(x, y)
        hits: list[Drawable] = []

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
            if drawable.contains_point(point) and drawable not in hits:
                hits.append(drawable)
        else:
            if drawable.contains_point(point):
                hits.append(drawable)

    def hit_test_top(self, x: float | int, y: float | int) -> Drawable | None:
        """Query the single topmost visible object intersecting (x, y), or None."""
        hits = self.hit_test(x, y)
        return hits[0] if hits else None

    # -------------------------------------------------------------------------
    # Temporal Evaluation & Non-Destructive Rendering (Phase 8)
    # -------------------------------------------------------------------------

    def animate(
        self,
        target: Any,
        property_path: str,
        start_value: Any,
        end_value: Any,
        duration: float | int = 1.0,
        easing: str | Callable[[float], float] = "linear",
        delay: float | int = 0.0,
        speed: float | int = 1.0,
        loop: bool = False,
    ) -> Any:
        """Convenience method to create and register an AnimationTrack on the scene timeline."""
        return self.timeline.animate(
            target=target,
            property_path=property_path,
            start_value=start_value,
            end_value=end_value,
            duration=duration,
            easing=easing,
            delay=delay,
            speed=speed,
            loop=loop,
        )

    @property
    def temporal_duration(self) -> float:
        """Maximum active endpoint across all Drawables with Timing and all Timeline tracks."""
        max_duration = 0.0

        for track in self.timeline.tracks:
            if track.timing.loop:
                return float("inf")
            max_duration = max(max_duration, track.timing.end_time)

        for d in self._id_map.values():
            if d.timing is not None:
                if d.timing.loop:
                    return float("inf")
                max_duration = max(max_duration, d.timing.end_time)

        return max_duration

    def sample(self, time: float) -> None:
        """Advance the scene model in-place to the given timestamp.

        Updates render_progress for all Drawables with Timing, and evaluates
        all Timeline animation tracks in insertion order.
        """
        for d in self._id_map.values():
            if d.timing is not None:
                d.render_progress = d.timing.get_progress(time)

        self.timeline.evaluate(time, self)

    def render_at_time(self, time: float, renderer: Any | None = None, *, alpha: bool = False) -> Any:
        """Observational, strictly non-destructive evaluation and rendering at a timestamp.

        Set alpha=True for straight BGRA output from an alpha-capable renderer.
        Captures the exact semantic state of all animated objects and timing progress,
        temporarily evaluates the scene at time t with history suspended, renders the frame,
        and unconditionally restores authored state in a finally block.
        """
        from drawcv.renderer import OpenCVRenderer
        if not isinstance(alpha, bool):
            raise ValidationError("alpha must be a boolean")
        active_renderer = renderer if renderer is not None else OpenCVRenderer()

        with self.history.suspended():
            # 1. Capture snapshot of all Timeline targets (semantic state)
            target_snapshots: dict[str, tuple[Drawable, dict[str, Any]]] = {}
            for track in self.timeline.tracks:
                if track.target_id not in target_snapshots:
                    obj = self.get(track.target_id)
                    if obj is not None:
                        target_snapshots[track.target_id] = (obj, obj._get_semantic_state())

            # 2. Capture original render_progress for all drawables with Timing
            timing_snapshots: list[tuple[Drawable, float]] = []
            for d in self._id_map.values():
                if d.timing is not None:
                    timing_snapshots.append((d, d.render_progress))

            try:
                self.sample(time)
                # Preserve compatibility with existing custom renderers that
                # implement render(scene) without output options.
                return active_renderer.render(self, alpha=True) if alpha else active_renderer.render(self)
            finally:
                # 3. Restore all render_progress values
                for d, orig_progress in timing_snapshots:
                    d.render_progress = orig_progress

                # 4. Restore all mutated semantic states in-place using _apply_semantic_state
                for obj, state in target_snapshots.values():
                    obj._apply_semantic_state(state)

    # -------------------------------------------------------------------------
    # Document Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return canonical DrawCV document envelope dictionary."""
        return {
            "format": CURRENT_FORMAT_IDENTIFIER,
            "version": CURRENT_SCHEMA_VERSION,
            "scene": {
                "width": self.width,
                "height": self.height,
                "background": self.background.to_dict(),
                "layers": [layer.to_dict() for layer in self.layers],
                "timeline": self.timeline.to_dict(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize Scene to strict, canonical JSON string."""
        return serialize_json(self.to_dict(), indent=indent)

    def save_json(self, filepath: str | Path, indent: int = 2) -> None:
        """Save Scene document to JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """Construct Scene from dictionary (migrating schema version if needed)."""
        if not isinstance(data, dict):
            raise ValidationError(f"Scene data must be a dict, got {type(data).__name__}")

        if "format" in data:
            migrated = SchemaMigrator.migrate(data)
            scene_data = migrated.get("scene", migrated)
        else:
            scene_data = data.get("scene", data)

        width = scene_data["width"]
        height = scene_data["height"]
        bg = Color.from_dict(scene_data["background"])
        scene = cls(width=width, height=height, background=bg)

        # Clear auto-created default layer
        scene._layers.clear()
        scene._layer_order.clear()
        scene._id_map.clear()

        for layer_data in scene_data.get("layers", []):
            layer = Layer.from_dict(layer_data, scene=scene)
            scene._layers[layer.name] = layer
            scene._layer_order.append(layer.name)

        if "default" not in scene._layers:
            scene.create_layer("default", z_order=0)

        # 2. Reconstruct Timeline and resolve target_ids against the populated scene graph
        from drawcv.animation.timeline import Timeline
        if "timeline" in scene_data:
            scene.timeline = Timeline.from_dict(scene_data["timeline"], scene=scene)
        else:
            scene.timeline = Timeline()

        # Loaded documents have clean history
        scene.history.clear()
        return scene

    @classmethod
    def from_json(cls, text: str) -> Scene:
        """Construct Scene from JSON string with envelope validation and migration."""
        parsed = parse_json(text, migrate=True)
        return cls.from_dict(parsed)

    @classmethod
    def load_json(cls, filepath: str | Path) -> Scene:
        """Load Scene document from a JSON file."""
        text = Path(filepath).read_text(encoding="utf-8")
        return cls.from_json(text)

    # -------------------------------------------------------------------------
    # Internal Helpers & Dunder Methods
    # -------------------------------------------------------------------------

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
