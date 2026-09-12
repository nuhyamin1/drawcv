"""Unit tests for Group container in DrawCV."""

import math
import pytest

from drawcv import (
    BoundingBox,
    Circle,
    Color,
    Drawable,
    FillStyle,
    Group,
    Line,
    ObjectNotFoundError,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    StrokeStyle,
    ValidationError,
)


def test_group_creation_and_child_management():
    c1 = Circle(center=Point(100, 100), radius=50, id="c1")
    r1 = Rectangle(position=Point(200, 200), width=80, height=60, id="r1")

    group = Group([c1, r1], name="test_group")
    assert len(group) == 2
    assert group[0] is c1
    assert group[1] is r1
    assert c1.parent is group
    assert r1.parent is group

    # Iteration
    items = list(group)
    assert items == [c1, r1]

    # Get by ID
    assert group.get("c1") is c1
    assert group.get("r1") is r1
    assert group.get("nonexistent") is None

    # Remove child
    removed = group.remove("c1")
    assert removed is c1
    assert c1.parent is None
    assert len(group) == 1
    assert group.get("c1") is None

    with pytest.raises(ObjectNotFoundError):
        group.remove("nonexistent")

    # Clear
    group.clear()
    assert len(group) == 0
    assert r1.parent is None


def test_group_cycle_prevention():
    g1 = Group(name="g1")
    g2 = Group(name="g2")

    # Cannot add group to itself
    with pytest.raises(ValidationError, match="itself"):
        g1.add(g1)

    # Cannot create cycle: g1 -> g2 -> g1
    g1.add(g2)
    with pytest.raises(ValidationError, match="Cycle detected"):
        g2.add(g1)


def test_group_strict_ownership_invariant():
    scene = Scene(800, 600)
    c1 = Circle(center=Point(100, 100), radius=30, id="c1")
    scene.add(c1)
    assert c1.layer.name == "default"
    assert c1 in scene.get_layer("default").objects

    # Adding to group removes from direct layer membership
    group = Group(id="g1")
    scene.add(group)
    group.add(c1)

    assert c1.parent is group
    assert c1 not in scene.get_layer("default").objects
    assert c1.layer.name == "default"  # inherited through group
    assert scene.get("c1") is c1


def test_group_world_transform_preserving_reparenting():
    # Existing circle positioned at (300, 300)
    circle = Circle(center=Point(300, 300), radius=50)
    orig_center_world = circle.to_world(circle.center)
    orig_bounds = circle.get_bounds()

    # Group rotated by 45 degrees and moved by (50, 20)
    group = Group()
    group.move(50, 20)
    group.rotate(45)

    # Add circle to group with world transform preservation
    group.add(circle, preserve_world_transform=True)

    # World position and bounds must NOT have jumped
    new_center_world = circle.to_world(circle.center)
    assert math.isclose(orig_center_world.x, new_center_world.x, abs_tol=1e-3)
    assert math.isclose(orig_center_world.y, new_center_world.y, abs_tol=1e-3)

    new_bounds = circle.get_bounds()
    assert math.isclose(orig_bounds.left, new_bounds.left, abs_tol=1e-3)
    assert math.isclose(orig_bounds.top, new_bounds.top, abs_tol=1e-3)


def test_group_transforms_propagation():
    r1 = Rectangle(position=Point(100, 100), width=100, height=100)
    group = Group([r1])

    init_bounds = group.get_bounds()
    assert math.isclose(init_bounds.left, 100.0)
    assert math.isclose(init_bounds.top, 100.0)

    # Move group
    group.move(50, 25)
    moved_bounds = group.get_bounds()
    assert math.isclose(moved_bounds.left, 150.0)
    assert math.isclose(moved_bounds.top, 125.0)
    assert math.isclose(r1.get_bounds().left, 150.0)
    assert math.isclose(r1.get_bounds().top, 125.0)

    # Rotate group
    group.rotate(90, pivot=Point(200, 175))
    rotated_bounds = r1.get_bounds()
    assert rotated_bounds.width > 0
    assert rotated_bounds.height > 0


def test_group_bounds_and_anchors():
    r1 = Rectangle(position=Point(100, 100), width=50, height=50)
    r2 = Rectangle(position=Point(200, 100), width=50, height=50)
    group = Group([r1, r2])

    gb = group.get_bounds()
    assert math.isclose(gb.left, 100.0)
    assert math.isclose(gb.right, 250.0)
    assert math.isclose(gb.top, 100.0)
    assert math.isclose(gb.bottom, 150.0)
    assert math.isclose(gb.width, 150.0)
    assert math.isclose(gb.height, 50.0)

    # Anchors
    assert math.isclose(group.anchor("center").x, 175.0)
    assert math.isclose(group.anchor("center").y, 125.0)
    assert math.isclose(group.anchor("top_left").x, 100.0)
    assert math.isclose(group.anchor("top_left").y, 100.0)
    assert math.isclose(group.anchor("bottom_right").x, 250.0)
    assert math.isclose(group.anchor("bottom_right").y, 150.0)


def test_group_contains_point_hit_test():
    r1 = Rectangle(position=Point(100, 100), width=50, height=50)
    r2 = Rectangle(position=Point(200, 100), width=50, height=50)
    group = Group([r1, r2])

    assert group.contains_point(Point(120, 120)) is True
    assert group.contains_point(Point(220, 120)) is True
    # Gap between them
    assert group.contains_point(Point(170, 120)) is False

    # When hidden
    group.visible = False
    assert group.contains_point(Point(120, 120)) is False


def test_group_effective_states_cascading():
    c = Circle(center=Point(50, 50), radius=20, opacity=0.8)
    group = Group([c], opacity=0.5, visible=True, locked=False)

    # Cascading opacity: 0.8 * 0.5 = 0.4
    assert math.isclose(c.effective_opacity, 0.4)
    assert c.effective_visible is True
    assert c.effective_locked is False

    # Hide group
    group.hide()
    assert c.effective_visible is False

    # Lock group
    group.lock()
    assert c.effective_locked is True


def test_group_deep_cloning():
    c = Circle(center=Point(50, 50), radius=20, id="c_orig")
    g = Group([c], id="g_orig", name="OriginalGroup")

    cloned = g.clone(new_id=True)
    assert cloned.id != g.id
    assert cloned.name == g.name
    assert len(cloned) == 1
    assert cloned[0].id != c.id
    assert cloned[0].parent is cloned
    assert cloned[0].center == c.center


def test_group_rendering():
    scene = Scene(400, 300, background=Color.white())
    r = Rectangle(position=Point(50, 50), width=100, height=60, fill=FillStyle(color=Color.red()))
    group = Group([r], opacity=0.7)
    scene.add(group)

    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)
    assert canvas.width == 400
    assert canvas.height == 300
