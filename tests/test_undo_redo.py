"""Comprehensive test suite for DrawCV Phase 7 — Reversible Undo/Redo Engine."""

import numpy as np
import pytest

from drawcv.core.color import Color
from drawcv.core.geometry import Point
from drawcv.group import Group
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


class TestLiveObjectIdentity:
    """Ensure in-place state restoration preserves exact live Python object identities."""

    def test_edit_preserves_live_identity(self):
        scene = Scene(500, 500)
        c = Circle(center=Point(100, 100), radius=50.0, id="c1")
        scene.add(c)

        original_radius = c.radius
        with scene.edit(c):
            c.radius = 200.0

        assert c.radius == 200.0
        assert scene.get("c1") is c

        scene.undo()

        assert c.radius == original_radius
        assert scene.get("c1") is c  # Exact live object identity preserved!

        scene.redo()
        assert c.radius == 200.0
        assert scene.get("c1") is c

    def test_group_recursive_semantic_snapshot_and_in_place_restoration(self):
        """Reviewer Requirement 1: Group._get_semantic_state() recursively captures
        descendant state, and Group._apply_semantic_state() restores states in-place
        while preserving child Python object identities.
        """
        scene = Scene(600, 600)
        child = Circle(center=Point(50, 50), radius=100.0, id="child_c")
        group = Group(children=[child], id="grp_1")
        scene.add(group)

        assert group.children[0] is child
        original_radius = child.radius

        with scene.edit(group):
            child.radius = 200.0

        assert child.radius == 200.0

        scene.undo()

        # The child state is restored in-place onto the same object instance
        assert child.radius == original_radius
        assert group.children[0] is child
        assert scene.get("child_c") is child

        scene.redo()

        assert child.radius == 200.0
        assert group.children[0] is child
        assert scene.get("child_c") is child

    def test_ancestor_deduplication_in_scene_edit(self):
        """Ensure scene.edit(group, child) normalizes to [group] to avoid double snapshots."""
        scene = Scene(600, 600)
        child = Circle(center=Point(50, 50), radius=100.0, id="child_c")
        group = Group(children=[child], id="grp_1")
        scene.add(group)

        with scene.edit(group, child):
            child.radius = 300.0

        assert child.radius == 300.0

        scene.undo()

        assert child.radius == 100.0
        assert group.children[0] is child
        assert scene.get("child_c") is child

    def test_deeply_nested_group_hierarchy_restoration(self):
        scene = Scene(600, 600)
        leaf = Circle(center=Point(10, 10), radius=25.0, id="leaf")
        inner_group = Group(children=[leaf], id="inner")
        outer_group = Group(children=[inner_group], id="outer")
        scene.add(outer_group)

        with scene.edit(outer_group):
            leaf.radius = 75.0

        scene.undo()

        assert leaf.radius == 25.0
        assert outer_group.children[0] is inner_group
        assert inner_group.children[0] is leaf
        assert scene.get("leaf") is leaf


class TestContainerAndStackingOrder:
    """Test reversible container insertion, deletion, and z-ordering."""

    def test_remove_and_undo_preserves_layer_and_index(self):
        scene = Scene(600, 600)
        o1 = Circle(center=Point(10, 10), radius=10, id="o1")
        o2 = Circle(center=Point(20, 20), radius=20, id="o2")
        o3 = Circle(center=Point(30, 30), radius=30, id="o3")
        scene.add(o1)
        scene.add(o2)
        scene.add(o3)

        default_layer = scene.get_layer("default")
        assert default_layer.objects == [o1, o2, o3]

        # Remove middle object
        removed = scene.remove("o2")
        assert removed is o2
        assert default_layer.objects == [o1, o3]
        assert scene.get("o2") is None

        # Undo removal
        scene.undo()
        assert scene.get("o2") is o2
        assert default_layer.objects == [o1, o2, o3]  # Restored at exact index!

        # Redo removal
        scene.redo()
        assert scene.get("o2") is None
        assert default_layer.objects == [o1, o3]

    def test_reorder_operations_undo_redo(self):
        scene = Scene(600, 600)
        o1 = Rectangle(position=Point(10, 10), width=50, height=50, id="o1")
        o2 = Rectangle(position=Point(20, 20), width=50, height=50, id="o2")
        scene.add(o1)
        scene.add(o2)

        layer = scene.get_layer("default")
        assert layer.objects == [o1, o2]

        scene.move_to_back("o2")
        assert layer.objects == [o2, o1]

        scene.undo()
        assert layer.objects == [o1, o2]

        scene.redo()
        assert layer.objects == [o2, o1]


class TestHistoryEngineInvariants:
    """Test redo invalidation, no-op filtering, exception rollback, and drift-free cycles."""

    def test_redo_branch_invalidation(self):
        scene = Scene(500, 500)
        c = Circle(center=Point(50, 50), radius=20, id="c")
        scene.add(c)

        with scene.edit(c):
            c.radius = 40.0

        assert scene.can_undo is True
        scene.undo()
        assert c.radius == 20.0
        assert scene.can_redo is True

        # Perform a new modifying action: branch is invalidated!
        with scene.edit(c):
            c.radius = 80.0

        assert scene.can_redo is False
        assert scene.can_undo is True
        assert c.radius == 80.0

    def test_noop_edit_filtering(self):
        scene = Scene(500, 500)
        c = Circle(center=Point(50, 50), radius=20, id="c")
        scene.add(c)
        initial_undo_count = scene.history.undo_count

        with scene.edit(c):
            # No changes performed
            pass

        assert scene.history.undo_count == initial_undo_count

    def test_exception_rollback(self):
        scene = Scene(500, 500)
        c = Circle(center=Point(50, 50), radius=20, id="c")
        scene.add(c)

        initial_count = scene.history.undo_count

        with pytest.raises(ValueError, match="Simulated Error"):
            with scene.edit(c):
                c.radius = 999.0
                raise ValueError("Simulated Error")

        # Rolled back immediately in-place!
        assert c.radius == 20.0
        assert scene.history.undo_count == initial_count

    def test_drift_free_50_cycles(self):
        scene = Scene(600, 600)
        c = Circle(center=Point(123.456, 234.567), radius=45.678, id="c")
        scene.add(c)

        initial_x = c.center.x
        initial_y = c.center.y
        initial_r = c.radius

        with scene.edit(c):
            c.center = Point(345.678, 456.789)
            c.radius = 89.012

        for _ in range(50):
            scene.undo()
            assert c.center.x == initial_x
            assert c.center.y == initial_y
            assert c.radius == initial_r
            scene.redo()
            assert c.center.x == 345.678
            assert c.center.y == 456.789
            assert c.radius == 89.012

        scene.undo()
        assert c.center.x == initial_x
        assert c.center.y == initial_y
        assert c.radius == initial_r


class TestGroupAndUngroupCommands:
    """Test reversible grouping and ungrouping workflows."""

    def test_group_and_ungroup_undo_redo(self):
        scene = Scene(800, 800)
        o1 = Circle(center=Point(100, 100), radius=30, id="o1")
        o2 = Rectangle(position=Point(200, 200), width=50, height=50, id="o2")
        scene.add(o1)
        scene.add(o2)

        # Group them
        grp = scene.group([o1, o2], name="MyGroup")
        assert grp.id in scene._id_map
        assert grp.children[0] is o1
        assert grp.children[1] is o2

        # Undo grouping
        scene.undo()
        assert scene.get("o1") is o1
        assert scene.get("o2") is o2
        assert scene.get(grp.id) is None
        assert o1._parent is None

        # Redo grouping
        scene.redo()
        assert scene.get(grp.id) is grp
        assert grp.children[0] is o1

        # Ungroup
        unpacked = scene.ungroup(grp)
        assert len(unpacked) == 2
        assert scene.get(grp.id) is None
        assert scene.get("o1") is o1
        assert scene.get("o2") is o2

        # Undo ungrouping
        scene.undo()
        assert scene.get(grp.id) is grp
        assert grp.children[0] is o1


class TestSelectionBatchUndo:
    """Test multi-object batch modifications through Selection."""

    def test_selection_batch_movement(self):
        scene = Scene(600, 600)
        c1 = Circle(center=Point(100, 100), radius=20, id="c1")
        c2 = Circle(center=Point(200, 200), radius=20, id="c2")
        scene.add(c1)
        scene.add(c2)

        with scene.batch("Move Selection"):
            scene.move_object("c1", 15, 25)
            scene.move_object("c2", 15, 25)

        assert c1.transform.translation_x == 15
        assert c2.transform.translation_x == 15

        # Single undo reverses both!
        scene.undo()

        assert c1.transform.translation_x == 0
        assert c2.transform.translation_x == 0

        # Single redo reapplies both!
        scene.redo()

        assert c1.transform.translation_x == 15
        assert c2.transform.translation_x == 15


class TestVisualStateEquivalence:
    """Test that rendering before mutation matches rendering after undo."""

    def test_render_matches_after_undo(self):
        scene = Scene(300, 300, background=Color(250, 250, 250))
        rect = Rectangle(position=Point(50, 50), width=100, height=80, fill=FillStyle(color=Color.red()), id="r")
        scene.add(rect)

        renderer = OpenCVRenderer()
        img_initial = renderer.render(scene).to_numpy()

        with scene.edit(rect):
            rect.position = Point(120, 120)
            rect.width = 160
            rect.fill = FillStyle(color=Color.blue())

        img_mutated = renderer.render(scene).to_numpy()
        assert not np.array_equal(img_initial, img_mutated)

        scene.undo()
        img_after_undo = renderer.render(scene).to_numpy()

        assert np.array_equal(img_initial, img_after_undo)
