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
        clip: Any | None = None,
        mask: Any | None = None,
        effects: list[Any] | None = None,
        timing: Any | None = None,
        render_progress: float = 1.0,
        **kwargs: Any,
    ):
        super_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "locked": locked,
            "opacity": opacity,
            "z_index": z_index,
            "clip": clip,
            "mask": mask,
            "timing": timing,
            "render_progress": float(render_progress),
        }
        if id is not None:
            super_kwargs["id"] = id
        if tags is not None:
            super_kwargs["tags"] = tags
        if metadata is not None:
            super_kwargs["metadata"] = metadata
        if transform is not None:
            super_kwargs["transform"] = transform
        if effects is not None:
            super_kwargs["effects"] = effects

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

    def get_effect_bounds(self) -> BoundingBox:
        """Calculate authoritative world-space visual bounds enclosing all children (with child effects) and group effects."""
        if not self._children:
            base_bounds = self.get_bounds()
        else:
            union_box: BoundingBox | None = None
            for child in self._children:
                cb = child.get_effect_bounds()
                union_box = cb if union_box is None else union_box.union(cb)
            base_bounds = union_box if union_box is not None else self.get_bounds()

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
            clip=copy.deepcopy(self.clip),
            mask=copy.deepcopy(self.mask),
            effects=copy.deepcopy(self.effects),
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

    # -------------------------------------------------------------------------
    # In-Place Semantic State Capture & Restoration (Recursive Descendant State)
    # -------------------------------------------------------------------------

    def _get_shape_state(self) -> dict[str, Any]:
        """Capture recursive semantic state and ordering of all descendants."""
        return {
            "children_states": [child._get_semantic_state() for child in self._children],
            "children_instances": list(self._children),
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        """Restore descendant states in-place while preserving existing child Python object identities."""
        children_states = state.get("children_states", [])
        instances_by_id = {c.id: c for c in state.get("children_instances", [])}
        instances_by_id.update({c.id: c for c in self._children})
        if self._scene is not None:
            for s in children_states:
                cid = s.get("id")
                if cid and cid not in instances_by_id:
                    sc_obj = self._scene.get(cid)
                    if sc_obj is not None:
                        instances_by_id[cid] = sc_obj

        restored_children: list[Drawable] = []
        for child_state in children_states:
            cid = child_state.get("id")
            child = instances_by_id.get(cid)
            if child is not None:
                child._apply_semantic_state(child_state)
                child._parent = self
                child._layer = self._layer
                if self._scene is not None and child._scene is not self._scene:
                    child._set_scene(self._scene)
                restored_children.append(child)
            else:
                child = Drawable.from_dict(child_state)
                child._parent = self
                child._layer = self._layer
                if self._scene is not None:
                    child._set_scene(self._scene)
                restored_children.append(child)

        for old_child in self._children:
            if old_child not in restored_children and old_child._parent is self:
                old_child._parent = None

        self._children = restored_children

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "group",
            "children": [child.to_dict() for child in self._children],
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Group:
        """Construct a Group from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        children_data = data.get("children", [])
        children: list[Drawable] = []
        for cd in children_data:
            children.append(Drawable.from_dict(cd))

        group = cls(children=None, **base_kwargs)
        for child in children:
            child._parent = group
            group._children.append(child)
        return group
