"""Logical selection management for DrawCV scenes."""

from __future__ import annotations
import math
from typing import Any, Iterator

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError
from drawcv.core.geometry import Point
from drawcv.group import Group


class Selection:
    """A logical multi-object selection controller.
    
    Provides coordinated multi-object spatial manipulation (move, rotate around collective center,
    scaling), batch property adjustments, lock protection, and grouping.
    """

    def __init__(self, scene: Any, items: list[Drawable | str] | None = None):
        self._scene = scene
        self._objects: list[Drawable] = []
        if items:
            for item in items:
                self.add(item)

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    def add(self, item: Drawable | str) -> None:
        """Add a drawable or ID to the selection."""
        if isinstance(item, str):
            obj = self._scene.get(item)
            if obj is None:
                raise ObjectNotFoundError(f"Drawable with id '{item}' not found in scene")
        elif isinstance(item, Drawable):
            obj = item
        else:
            raise ValidationError(f"Expected Drawable or string ID, got {type(item).__name__}")

        if obj not in self._objects:
            self._objects.append(obj)

    def remove(self, item: Drawable | str) -> None:
        """Remove a drawable or ID from the selection."""
        target: Drawable | None = None
        if isinstance(item, str):
            for obj in self._objects:
                if obj.id == item:
                    target = obj
                    break
        elif isinstance(item, Drawable):
            if item in self._objects:
                target = item
        else:
            raise ValidationError("Expected Drawable or string ID")

        if target is not None:
            self._objects.remove(target)

    def clear(self) -> None:
        """Clear all selected objects."""
        self._objects.clear()

    @property
    def objects(self) -> list[Drawable]:
        """Return a shallow copy list of selected drawables."""
        return list(self._objects)

    @property
    def ids(self) -> list[str]:
        """Return the IDs of all selected drawables."""
        return [obj.id for obj in self._objects]

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[Drawable]:
        return iter(self._objects)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(obj.id == item for obj in self._objects)
        return item in self._objects

    # -------------------------------------------------------------------------
    # Bounds & Ancestor Normalization
    # -------------------------------------------------------------------------

    @property
    def bounds(self) -> BoundingBox:
        """Collective world-space visual AABB enclosing all selected objects."""
        if not self._objects:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)

        union_box: BoundingBox | None = None
        for obj in self._objects:
            b = obj.get_bounds()
            if union_box is None:
                union_box = b
            else:
                union_box = union_box.union(b)

        return union_box if union_box is not None else BoundingBox(0.0, 0.0, 0.0, 0.0)

    def _get_transform_roots(self) -> list[Drawable]:
        """Return selected drawables filtering out descendants of any selected ancestor Group.
        
        This prevents double-transforming children when both an ancestor Group and its child
        are present in the selection.
        """
        selected_set = set(self._objects)
        roots: list[Drawable] = []
        for obj in self._objects:
            curr = obj._parent
            has_ancestor_in_selection = False
            while curr is not None:
                if curr in selected_set:
                    has_ancestor_in_selection = True
                    break
                curr = curr._parent
            if not has_ancestor_in_selection:
                roots.append(obj)
        return roots

    def _check_not_locked(self, roots: list[Drawable]) -> None:
        """Raise ValidationError if any transform root is effective-locked."""
        for root in roots:
            if root.effective_locked:
                raise ValidationError(f"Cannot transform or modify locked drawable '{root.id}'")

    # -------------------------------------------------------------------------
    # Spatial Transformation Operations
    # -------------------------------------------------------------------------

    def move(self, dx: float | int, dy: float | int) -> None:
        """Translate all selected transform roots by (dx, dy)."""
        roots = self._get_transform_roots()
        self._check_not_locked(roots)
        for root in roots:
            root.move(dx, dy)

    def rotate(self, degrees: float | int, pivot: Point | None = None) -> None:
        """Rotate all selected transform roots rigidly around pivot (default: selection center)."""
        roots = self._get_transform_roots()
        if not roots:
            return
        self._check_not_locked(roots)

        p = pivot if pivot is not None else self.bounds.center
        rad = math.radians(float(degrees))
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        for root in roots:
            C = root.get_bounds().center
            dx0 = C.x - p.x
            dy0 = C.y - p.y
            dx1 = dx0 * cos_a - dy0 * sin_a
            dy1 = dx0 * sin_a + dy0 * cos_a
            new_C = Point(p.x + dx1, p.y + dy1)
            shift_x = new_C.x - C.x
            shift_y = new_C.y - C.y

            root.rotate(degrees)
            root.move(shift_x, shift_y)

    def scale(self, sx: float | int, sy: float | int | None = None, pivot: Point | None = None) -> None:
        """Scale all selected transform roots relative to pivot (default: selection center)."""
        roots = self._get_transform_roots()
        if not roots:
            return
        self._check_not_locked(roots)

        sy_val = float(sy) if sy is not None else float(sx)
        p = pivot if pivot is not None else self.bounds.center

        for root in roots:
            C = root.get_bounds().center
            dx0 = C.x - p.x
            dy0 = C.y - p.y
            new_C = Point(p.x + dx0 * float(sx), p.y + dy0 * sy_val)
            shift_x = new_C.x - C.x
            shift_y = new_C.y - C.y

            root.scale(sx, sy_val)
            root.move(shift_x, shift_y)

    # -------------------------------------------------------------------------
    # Batch State Modifiers
    # -------------------------------------------------------------------------

    def delete(self) -> list[Drawable]:
        """Delete all selected transform roots from their respective parent group or layer."""
        roots = self._get_transform_roots()
        self._check_not_locked(roots)
        removed: list[Drawable] = []
        for root in roots:
            if root._parent is not None:
                root._parent.remove(root)
                removed.append(root)
            elif root._layer is not None:
                root._layer.remove(root)
                removed.append(root)
            elif self._scene is not None and self._scene.get(root.id) is not None:
                self._scene.remove(root.id)
                removed.append(root)

        self._objects.clear()
        return removed

    def show(self) -> None:
        """Make all selected objects visible."""
        for obj in self._objects:
            obj.visible = True

    def hide(self) -> None:
        """Hide all selected objects."""
        for obj in self._objects:
            obj.visible = False

    def lock(self) -> None:
        """Lock all selected objects."""
        for obj in self._objects:
            obj.locked = True

    def unlock(self) -> None:
        """Unlock all selected objects."""
        for obj in self._objects:
            obj.locked = False

    def set_opacity(self, opacity: float) -> None:
        """Set opacity for all selected objects."""
        for obj in self._objects:
            obj.opacity = float(opacity)

    def set_z_index(self, z_index: int) -> None:
        """Set z_index for all selected objects."""
        for obj in self._objects:
            obj.z_index = int(z_index)

    # -------------------------------------------------------------------------
    # Grouping
    # -------------------------------------------------------------------------

    def group(self, name: str | None = None) -> Group:
        """Bundle selected transform roots into a new Group within their shared container.
        
        Enforces that all transform roots must share the same immediate container
        (the same Layer or the same parent Group) to prevent hierarchy and painter order ambiguity.
        """
        roots = self._get_transform_roots()
        if not roots:
            raise ValidationError("Cannot create an empty group from selection")

        self._check_not_locked(roots)

        first = roots[0]
        shared_parent = first._parent
        shared_layer = first._layer

        for root in roots[1:]:
            if root._parent != shared_parent or root._layer != shared_layer:
                raise ValidationError("Cannot group objects across different layers or different parent groups")

        new_group = Group(name=name)

        # Reparent items into the new group
        for root in roots:
            new_group.add(root, preserve_world_transform=True)

        # Place new group into the shared container
        if shared_parent is not None:
            shared_parent.add(new_group, preserve_world_transform=True)
        elif shared_layer is not None:
            shared_layer.add(new_group)
        elif self._scene is not None:
            self._scene.add(new_group)

        # Update selection to the new group
        self._objects = [new_group]
        return new_group
