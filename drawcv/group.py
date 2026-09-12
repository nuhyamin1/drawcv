"""Hierarchical Group container for DrawCV drawable objects."""

from __future__ import annotations
import copy
from typing import Any, Iterator, TypeVar
import uuid

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform

T = TypeVar("T", bound=Drawable)


class Group(Drawable):
    """A retained-mode container grouping multiple drawables into a unified hierarchy.
    
    Groups are themselves Drawables and maintain their own affine transform,
    effective opacity, visibility, and locking status that cascade down to all descendants.
    """

    def __init__(
        self,
        children: list[Drawable] | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
        z_index: int = 0,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform: Transform | None = None,
    ):
        super_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "locked": locked,
            "opacity": opacity,
            "z_index": z_index,
        }
        if id is not None:
            super_kwargs["id"] = id
        if tags is not None:
            super_kwargs["tags"] = tags
        if metadata is not None:
            super_kwargs["metadata"] = metadata
        if transform is not None:
            super_kwargs["transform"] = transform

        super().__init__(**super_kwargs)
        self._children: list[Drawable] = []

        if children:
            for child in children:
                self.add(child, preserve_world_transform=False)

    # -------------------------------------------------------------------------
    # Child Management & Invariants
    # -------------------------------------------------------------------------

    def add(self, child: Drawable, preserve_world_transform: bool = True) -> None:
        """Add a child drawable to this group.
        
        Enforces strict tree invariants:
        - Prevents self-addition and cycles.
        - Detaches child from prior parent group or layer.
        - Preserves world-space visual position and orientation if requested.
        - Synchronizes child and descendants with the owning Scene ID registry.
        """
        if not isinstance(child, Drawable):
            raise ValidationError(f"Group child must be a Drawable, got {type(child).__name__}")
        if child is self:
            raise ValidationError("Cannot add a group to itself")

        # Check for cyclic ancestor containment
        curr = self._parent
        while curr is not None:
            if curr is child:
                raise ValidationError("Cycle detected: cannot add an ancestor as a child")
            curr = curr._parent

        # Detach from prior parent group if already parented
        if child._parent is not None:
            child._parent.remove(child, preserve_world_transform=False)

        # Detach from prior direct layer membership if top-level
        if child._layer is not None:
            child._layer.remove(child)

        # If this group belongs to a Scene, check ID uniqueness for child and subtree
        if self._scene is not None:
            self._scene._check_id_unique(child)

        # Reparent with world-transform preservation
        child._reparent(new_parent=self, new_layer=None, preserve_world_transform=preserve_world_transform)
        self._children.append(child)

        # Synchronize scene registry down the new child subtree
        if self._scene is not None:
            child._set_scene(self._scene)

    def remove(self, drawable_or_id: Drawable | str, preserve_world_transform: bool = True) -> Drawable:
        """Remove a child drawable from this group by reference or ID."""
        child_to_remove: Drawable | None = None
        if isinstance(drawable_or_id, Drawable):
            for child in self._children:
                if child is drawable_or_id:
                    child_to_remove = child
                    break
        elif isinstance(drawable_or_id, str):
            for child in self._children:
                if child.id == drawable_or_id:
                    child_to_remove = child
                    break
        else:
            raise ValidationError("Expected Drawable or string ID")

        if child_to_remove is None:
            id_str = drawable_or_id.id if isinstance(drawable_or_id, Drawable) else drawable_or_id
            raise ObjectNotFoundError(f"Drawable with id '{id_str}' not found in group")

        self._children.remove(child_to_remove)

        # Unregister from scene if attached
        if self._scene is not None:
            child_to_remove._set_scene(None)

        # Detach parent link with world-transform preservation
        child_to_remove._reparent(new_parent=None, new_layer=None, preserve_world_transform=preserve_world_transform)
        return child_to_remove

    def get(self, id: str, recursive: bool = True) -> Drawable | None:
        """Retrieve a child drawable by its ID."""
        for child in self._children:
            if child.id == id:
                return child
            if recursive and isinstance(child, Group):
                found = child.get(id, recursive=True)
                if found is not None:
                    return found
        return None

    def clear(self) -> None:
        """Remove and unbind all children from this group."""
        while self._children:
            self.remove(self._children[-1], preserve_world_transform=False)

    @property
    def children(self) -> list[Drawable]:
        """Return a shallow copy of the children list."""
        return list(self._children)

    @property
    def objects(self) -> list[Drawable]:
        """Alias for children."""
        return list(self._children)

    def __iter__(self) -> Iterator[Drawable]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index: int) -> Drawable:
        return self._children[index]

    # -------------------------------------------------------------------------
    # Scene Hierarchy Synchronization
    # -------------------------------------------------------------------------

    def _set_scene(self, scene: Any | None) -> None:
        """Propagate scene reference and synchronize registry down through all children."""
        super()._set_scene(scene)
        for child in self._children:
            child._set_scene(scene)

    # -------------------------------------------------------------------------
    # Bounds Hierarchy
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in group local space.
        
        Evaluated bottom-up as the union of all children's bounds in group space.
        Its center serves as the default pivot for group rotation/scaling.
        """
        if not self._children:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)

        union_box: BoundingBox | None = None
        for child in self._children:
            cb = child.get_bounds_in_parent()
            if union_box is None:
                union_box = cb
            else:
                union_box = union_box.union(cb)
        return union_box if union_box is not None else BoundingBox(0.0, 0.0, 0.0, 0.0)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds of the group at identity transform."""
        return self.get_geometry_bounds()

    def get_bounds(self) -> BoundingBox:
        """Authoritative world-space visual AABB enclosing all children's world bounds."""
        if not self._children:
            return BoundingBox(self.transform.translation_x, self.transform.translation_y, 0.0, 0.0)

        union_box: BoundingBox | None = None
        for child in self._children:
            cb = child.get_bounds()
            if union_box is None:
                union_box = cb
            else:
                union_box = union_box.union(cb)

        return union_box if union_box is not None else BoundingBox(0.0, 0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Anchors, Hit-Testing & Cloning
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve a named geometric anchor point mapped to world coordinates.
        
        Supported anchor names for Group:
            'center', 'top', 'bottom', 'left', 'right',
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
        """
        norm = name.strip().lower().replace("-", "_").replace(" ", "_")
        b = self.get_bounds()

        if norm == "center":
            return b.center
        elif norm == "top":
            return Point(b.left + b.width / 2.0, b.top)
        elif norm == "bottom":
            return Point(b.left + b.width / 2.0, b.bottom)
        elif norm == "left":
            return Point(b.left, b.top + b.height / 2.0)
        elif norm == "right":
            return Point(b.right, b.top + b.height / 2.0)
        elif norm in ("top_left", "topleft"):
            return b.top_left
        elif norm in ("top_right", "topright"):
            return b.top_right
        elif norm in ("bottom_left", "bottomleft"):
            return b.bottom_left
        elif norm in ("bottom_right", "bottomright"):
            return b.bottom_right
        else:
            valid = "center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right"
            raise ValidationError(f"Unknown anchor '{name}' for Group. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test whether a world-space point intersects any visible child in this group."""
        if not self.effective_visible:
            return False
        # Test in reverse order (topmost child first)
        for child in reversed(self._children):
            if child.effective_visible and child.contains_point(world_point):
                return True
        return False

    def clone(self, new_id: bool = True) -> Group:
        """Create a deep copy of this group and its child hierarchy with fresh IDs."""
        cloned_group = Group(
            id=str(uuid.uuid4()) if new_id else self.id,
            name=self.name,
            visible=self.visible,
            locked=self.locked,
            opacity=self.opacity,
            z_index=self.z_index,
            tags=set(self.tags),
            metadata=copy.deepcopy(self.metadata),
            transform=copy.deepcopy(self.transform),
        )

        for child in self._children:
            cloned_child = child.clone(new_id=new_id)
            cloned_child._parent = cloned_group
            cloned_group._children.append(cloned_child)

        return cloned_group

    # -------------------------------------------------------------------------
    # Search Helpers
    # -------------------------------------------------------------------------

    def find_by_name(self, name: str, recursive: bool = True) -> list[Drawable]:
        """Find all children matching name."""
        results: list[Drawable] = []
        for child in self._children:
            if child.name == name:
                results.append(child)
            if recursive and isinstance(child, Group):
                results.extend(child.find_by_name(name, recursive=True))
        return results

    def find_by_tag(self, tag: str, recursive: bool = True) -> list[Drawable]:
        """Find all children having the specified tag."""
        results: list[Drawable] = []
        for child in self._children:
            if tag in child.tags:
                results.append(child)
            if recursive and isinstance(child, Group):
                results.extend(child.find_by_tag(tag, recursive=True))
        return results

    def find_by_type(self, cls: type[T], recursive: bool = True) -> list[T]:
        """Find all children that are instances of the specified class."""
        results: list[T] = []
        for child in self._children:
            if isinstance(child, cls):
                results.append(child)
            if recursive and isinstance(child, Group):
                results.extend(child.find_by_type(cls, recursive=True))
        return results
