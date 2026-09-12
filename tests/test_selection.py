"""Unit tests for Selection in DrawCV."""

import math
import pytest

from drawcv import (
    Circle,
    Group,
    ObjectNotFoundError,
    Point,
    Rectangle,
    Scene,
    Selection,
    ValidationError,
)


def test_selection_creation_and_bounds():
    scene = Scene(800, 600)
    r1 = Rectangle(position=Point(100, 100), width=50, height=50, id="r1")
    r2 = Rectangle(position=Point(200, 100), width=50, height=50, id="r2")
    scene.add(r1)
    scene.add(r2)

    sel = scene.select(["r1", r2])
    assert len(sel) == 2
    assert "r1" in sel
    assert "r2" in sel

    # Collective bounds: (100, 100, 150, 50)
    b = sel.bounds
    assert math.isclose(b.left, 100.0)
    assert math.isclose(b.right, 250.0)
    assert math.isclose(b.top, 100.0)
    assert math.isclose(b.bottom, 150.0)
    assert math.isclose(b.center.x, 175.0)
    assert math.isclose(b.center.y, 125.0)


def test_selection_move_rotate_scale():
    scene = Scene(800, 600)
    r1 = Rectangle(position=Point(100, 100), width=50, height=50, id="r1")
    r2 = Rectangle(position=Point(200, 100), width=50, height=50, id="r2")
    scene.add(r1)
    scene.add(r2)

    sel = scene.select(["r1", "r2"])

    # Move selection
    sel.move(30, 40)
    assert math.isclose(r1.get_bounds().left, 130.0)
    assert math.isclose(r2.get_bounds().left, 230.0)

    # Scale selection relative to collective center (205, 165)
    sel.scale(2.0, 2.0)
    scaled_b = sel.bounds
    assert math.isclose(scaled_b.width, 300.0)

    # Rotate selection 90 degrees around collective center
    sel.rotate(90)
    assert sel.bounds.width > 0


def test_selection_ancestor_descendant_normalization():
    scene = Scene(800, 600)
    c = Circle(center=Point(100, 100), radius=20, id="c")
    group = Group([c], id="g")
    scene.add(group)

    # Both group and its child are selected
    sel = scene.select([group, c])
    assert len(sel) == 2

    # Move selection
    sel.move(50, 0)

    # Child should move ONLY ONCE (via group), not twice!
    # Initial center was 100. Moved by 50 -> center should be 150 (not 200)
    assert math.isclose(c.to_world(c.center).x, 150.0)


def test_selection_lock_protection():
    scene = Scene(800, 600)
    r1 = Rectangle(position=Point(100, 100), width=50, height=50, locked=True)
    r2 = Rectangle(position=Point(200, 100), width=50, height=50)
    scene.add(r1)
    scene.add(r2)

    sel = scene.select([r1, r2])

    with pytest.raises(ValidationError, match="locked"):
        sel.move(10, 10)

    with pytest.raises(ValidationError, match="locked"):
        sel.rotate(45)

    with pytest.raises(ValidationError, match="locked"):
        sel.scale(2)

    with pytest.raises(ValidationError, match="locked"):
        sel.delete()


def test_selection_grouping_and_container_boundaries():
    scene = Scene(800, 600)
    r1 = Rectangle(position=Point(100, 100), width=50, height=50, id="r1")
    r2 = Rectangle(position=Point(200, 100), width=50, height=50, id="r2")
    scene.add(r1)
    scene.add(r2)

    sel = scene.select([r1, r2])
    new_group = sel.group(name="CombinedGroup")
    assert isinstance(new_group, Group)
    assert len(new_group) == 2
    assert r1.parent is new_group
    assert r2.parent is new_group
    assert new_group.layer.name == "default"

    # Cross-layer grouping MUST raise ValidationError
    scene.create_layer("L2", z_order=10)
    r3 = Rectangle(position=Point(300, 300), width=40, height=40, id="r3")
    scene.add(r3, layer="L2")

    sel_cross = scene.select([new_group, r3])
    with pytest.raises(ValidationError, match="different layers"):
        sel_cross.group()


def test_selection_batch_modifiers():
    scene = Scene(800, 600)
    r1 = Rectangle(position=Point(100, 100), width=50, height=50)
    r2 = Rectangle(position=Point(200, 100), width=50, height=50)
    scene.add(r1)
    scene.add(r2)

    sel = scene.select([r1, r2])

    sel.hide()
    assert r1.visible is False and r2.visible is False

    sel.show()
    assert r1.visible is True and r2.visible is True

    sel.set_opacity(0.4)
    assert r1.opacity == 0.4 and r2.opacity == 0.4

    sel.set_z_index(5)
    assert r1.z_index == 5 and r2.z_index == 5

    # Delete
    deleted = sel.delete()
    assert len(deleted) == 2
    assert scene.get(r1.id) is None
    assert scene.get(r2.id) is None
