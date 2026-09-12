"""Unit tests for Scene object lookup, registry synchronization, tags, and metadata."""

import pytest

from drawcv import (
    Circle,
    Drawable,
    Group,
    Line,
    Point,
    Rectangle,
    Scene,
    ValidationError,
)


def test_scene_id_registry_and_collision_prevention():
    scene = Scene(800, 600)
    c1 = Circle(center=Point(100, 100), radius=20, id="c1")
    scene.add(c1)

    # Collision at scene top-level
    c2 = Circle(center=Point(200, 200), radius=20, id="c1")
    with pytest.raises(ValidationError, match="already exists"):
        scene.add(c2)

    # Collision via layer.add
    l2 = scene.create_layer("L2")
    with pytest.raises(ValidationError, match="already exists"):
        l2.add(c2)

    # Collision via group.add
    group = Group(id="g1")
    scene.add(group)
    with pytest.raises(ValidationError, match="already exists"):
        group.add(c2)

    # Adding new child with unique ID synchronizes immediately
    c3 = Circle(center=Point(300, 300), radius=20, id="c3")
    group.add(c3)
    assert scene.get("c3") is c3

    # Removing child unregisters it
    group.remove("c3")
    assert scene.get("c3") is None


def test_tags_and_metadata_helpers():
    c = Circle(center=Point(100, 100), radius=20)
    c.add_tag("ui").add_tag("primary")
    assert c.has_tag("ui") is True
    assert c.has_tag("primary") is True
    assert c.has_tag("secondary") is False

    c.remove_tag("primary")
    assert c.has_tag("primary") is False

    # Metadata
    c.set_metadata("author", "alice").set_metadata("version", 2)
    assert c.get_metadata("author") == "alice"
    assert c.get_metadata("version") == 2
    assert c.get_metadata("nonexistent", default=42) == 42


def test_recursive_lookups_in_scene():
    scene = Scene(800, 600)
    c1 = Circle(center=Point(50, 50), radius=10, name="target_item", id="c1")
    c1.add_tag("active").set_metadata("role", "admin")

    c2 = Circle(center=Point(70, 70), radius=10, name="other_item", id="c2")
    c2.add_tag("active").set_metadata("role", "guest")

    nested_group = Group([c1], name="inner_group", id="inner")
    outer_group = Group([nested_group, c2], name="outer_group", id="outer")
    scene.add(outer_group)

    # O(1) get even deep in nested groups
    assert scene.get("c1") is c1
    assert scene.get("inner") is nested_group
    assert scene.get("outer") is outer_group

    # find_by_name
    assert scene.find_by_name("target_item") == [c1]
    assert scene.find_by_name("inner_group") == [nested_group]

    # find_by_tag
    active_items = scene.find_by_tag("active")
    assert len(active_items) == 2
    assert c1 in active_items and c2 in active_items

    # find_by_type
    circles = scene.find_by_type(Circle)
    assert len(circles) == 2
    groups = scene.find_by_type(Group)
    assert len(groups) == 2

    # find_by_metadata
    admin_items = scene.find_by_metadata("role", "admin")
    assert admin_items == [c1]

    # General find with predicate and kwargs
    res = scene.find(predicate=lambda o: isinstance(o, Circle), role="guest")
    assert res == [c2]

    first = scene.find_first(name="target_item")
    assert first is c1


def test_scene_selection_shortcuts():
    scene = Scene(800, 600)
    c1 = Circle(center=Point(10, 10), radius=5, id="c1").add_tag("widget")
    c2 = Circle(center=Point(20, 20), radius=5, id="c2").add_tag("widget")
    r1 = Rectangle(position=Point(30, 30), width=10, height=10, id="r1")
    scene.add(c1)
    scene.add(c2)
    scene.add(r1)

    sel_all = scene.select_all()
    assert len(sel_all) == 3

    sel_tag = scene.select_by_tag("widget")
    assert len(sel_tag) == 2
    assert "c1" in sel_tag and "c2" in sel_tag

    sel_type = scene.select_by_type(Rectangle)
    assert len(sel_type) == 1
    assert "r1" in sel_type
