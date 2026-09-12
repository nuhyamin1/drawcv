"""Concrete command implementations for DrawCV history engine."""

from __future__ import annotations
import copy
from typing import Any, TYPE_CHECKING

from drawcv.history.command import Command
from drawcv.core.transform import Transform

if TYPE_CHECKING:
    from drawcv.core.drawable import Drawable
    from drawcv.group import Group
    from drawcv.layer import Layer
    from drawcv.scene import Scene


class StateEditCommand(Command):
    """Encapsulates in-place semantic state mutations preserving live object identities."""

    def __init__(
        self,
        targets: list[tuple[Drawable, dict[str, Any], dict[str, Any]]],
        description: str = "Edit",
    ):
        super().__init__(description=description)
        self.targets = targets  # list of (drawable_instance, before_state, after_state)

    def execute(self) -> None:
        """State was already applied by user/code; re-applying applies after_state."""
        self.redo()

    def undo(self) -> None:
        """Restore previous state in-place onto the live object instances."""
        for target, before_state, _ in self.targets:
            target._apply_semantic_state(before_state)

    def redo(self) -> None:
        """Re-apply new state in-place onto the live object instances."""
        for target, _, after_state in self.targets:
            target._apply_semantic_state(after_state)

    def is_noop(self) -> bool:
        """True if none of the targets had any semantic changes."""
        if not self.targets:
            return True
        for _, before_state, after_state in self.targets:
            if before_state != after_state:
                return False
        return True


class AddObjectCommand(Command):
    """Encapsulates adding a drawable to a scene layer."""

    def __init__(
        self,
        scene: Scene,
        drawable: Drawable,
        layer_name: str = "default",
        index: int | None = None,
        description: str = "Add Object",
    ):
        super().__init__(description=description)
        self.scene = scene
        self.drawable = drawable
        self.layer_name = layer_name
        self.index = index

    def execute(self) -> None:
        self.scene._add_untracked(self.drawable, layer=self.layer_name, index=self.index)

    def undo(self) -> None:
        self.scene._remove_untracked(self.drawable.id)

    def redo(self) -> None:
        self.execute()


class RemoveObjectCommand(Command):
    """Encapsulates removing a drawable from a layer or group, preserving container positioning."""

    def __init__(
        self,
        scene: Scene,
        drawable: Drawable,
        layer_name: str | None = None,
        parent_group: Group | None = None,
        index: int = -1,
        description: str = "Remove Object",
    ):
        super().__init__(description=description)
        self.scene = scene
        self.drawable = drawable
        self.layer_name = layer_name
        self.parent_group = parent_group
        self.index = index

    def execute(self) -> None:
        self.scene._remove_untracked(self.drawable.id)

    def undo(self) -> None:
        """Re-insert the exact live instance back into its container at original index."""
        if self.parent_group is not None:
            # Re-insert into parent group
            if self.index >= 0 and self.index < len(self.parent_group._children):
                self.parent_group._children.insert(self.index, self.drawable)
            else:
                self.parent_group._children.append(self.drawable)
            self.drawable._parent = self.parent_group
            self.drawable._layer = self.parent_group._layer
            if self.scene is not None:
                self.drawable._set_scene(self.scene)
        else:
            # Re-insert into layer
            layer = self.scene.get_layer(self.layer_name or "default")
            if layer is not None:
                if self.index >= 0 and self.index < len(layer._objects):
                    layer._objects.insert(self.index, self.drawable)
                else:
                    layer._objects.append(self.drawable)
                self.drawable._layer = layer
                self.drawable._parent = None
                self.drawable._set_scene(self.scene)

    def redo(self) -> None:
        self.execute()


class TransformCommand(Command):
    """Encapsulates affine transform modifications (move, rotate, scale)."""

    def __init__(
        self,
        drawable: Drawable,
        before_transform: Transform,
        after_transform: Transform,
        description: str = "Transform",
    ):
        super().__init__(description=description)
        self.drawable = drawable
        self.before_transform = before_transform.copy()
        self.after_transform = after_transform.copy()

    def execute(self) -> None:
        self.drawable.transform = self.after_transform.copy()

    def undo(self) -> None:
        self.drawable.transform = self.before_transform.copy()

    def redo(self) -> None:
        self.drawable.transform = self.after_transform.copy()

    def is_noop(self) -> bool:
        return self.before_transform == self.after_transform


class ReorderCommand(Command):
    """Encapsulates stacking order (z_index and container index) modifications."""

    def __init__(
        self,
        container: Layer | Group,
        drawable: Drawable,
        old_index: int,
        new_index: int,
        old_z_index: int,
        new_z_index: int,
        description: str = "Reorder",
    ):
        super().__init__(description=description)
        self.container = container
        self.drawable = drawable
        self.old_index = old_index
        self.new_index = new_index
        self.old_z_index = old_z_index
        self.new_z_index = new_z_index

    def execute(self) -> None:
        self._apply(self.new_index, self.new_z_index)

    def undo(self) -> None:
        self._apply(self.old_index, self.old_z_index)

    def redo(self) -> None:
        self._apply(self.new_index, self.new_z_index)

    def _apply(self, target_index: int, target_z: int) -> None:
        self.drawable.z_index = target_z
        pool = self._get_pool()
        if self.drawable in pool:
            pool.remove(self.drawable)
        if 0 <= target_index <= len(pool):
            pool.insert(target_index, self.drawable)
        else:
            pool.append(self.drawable)

    def _get_pool(self) -> list[Drawable]:
        if hasattr(self.container, "_objects"):
            return self.container._objects
        elif hasattr(self.container, "_children"):
            return self.container._children
        return []

    def is_noop(self) -> bool:
        return self.old_index == self.new_index and self.old_z_index == self.new_z_index


class GroupCommand(Command):
    """Encapsulates grouping drawables into a new Group container."""

    def __init__(
        self,
        scene: Scene,
        group: Group,
        children_info: list[dict[str, Any]],
        layer_name: str,
        group_index: int,
        description: str = "Group",
    ):
        super().__init__(description=description)
        self.scene = scene
        self.group = group
        self.children_info = children_info
        self.layer_name = layer_name
        self.group_index = group_index

    def execute(self) -> None:
        # Reparent children under group and add group to scene
        for info in self.children_info:
            child = info["child"]
            self.scene._remove_untracked(child.id)
            child.transform = info["grouped_transform"].copy()
            child._parent = self.group
            self.group._children.append(child)
        self.scene._add_untracked(self.group, layer=self.layer_name, index=self.group_index)

    def undo(self) -> None:
        # Remove group and restore children to original parents/layers
        self.scene._remove_untracked(self.group.id)
        self.group._children.clear()
        for info in self.children_info:
            child = info["child"]
            child.transform = info["orig_transform"].copy()
            child._parent = info["orig_parent"]
            if info["orig_parent"] is not None:
                info["orig_parent"]._children.insert(info["orig_index"], child)
                child._layer = info["orig_parent"]._layer
            else:
                layer = self.scene.get_layer(info["orig_layer"])
                if layer is not None:
                    layer._objects.insert(info["orig_index"], child)
                    child._layer = layer
            child._set_scene(self.scene)

    def redo(self) -> None:
        self.execute()


class UngroupCommand(Command):
    """Encapsulates unpacking a Group into its enclosing container."""

    def __init__(
        self,
        scene: Scene,
        group: Group,
        children_info: list[dict[str, Any]],
        layer_name: str,
        group_index: int,
        description: str = "Ungroup",
    ):
        super().__init__(description=description)
        self.scene = scene
        self.group = group
        self.children_info = children_info
        self.layer_name = layer_name
        self.group_index = group_index

    def execute(self) -> None:
        # Remove group and place children into target layer at group_index
        self.scene._remove_untracked(self.group.id)
        layer = self.scene.get_layer(self.layer_name)
        if layer is not None:
            curr_idx = self.group_index
            for info in self.children_info:
                child = info["child"]
                child.transform = info["unpacked_transform"].copy()
                child._parent = None
                child._layer = layer
                layer._objects.insert(curr_idx, child)
                child._set_scene(self.scene)
                curr_idx += 1

    def undo(self) -> None:
        # Remove children and restore group
        for info in self.children_info:
            child = info["child"]
            self.scene._remove_untracked(child.id)
            child.transform = info["orig_transform"].copy()
            child._parent = self.group
            self.group._children.append(child)
        self.scene._add_untracked(self.group, layer=self.layer_name, index=self.group_index)

    def redo(self) -> None:
        self.execute()
