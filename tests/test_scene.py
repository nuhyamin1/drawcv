"""Unit tests for Scene document management and spatial querying."""

import pytest

from drawcv.core.color import Color
from drawcv.core.exceptions import ObjectNotFoundError, ValidationError
from drawcv.core.geometry import Point
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.line import Line
from drawcv.shapes.rectangle import Rectangle


def test_scene_construction():
    s = Scene(width=800, height=600, background=Color.white())
    assert s.width == 800
    assert s.height == 600
    assert s.background == Color.white()
    assert len(s) == 0


def test_scene_validation():
    with pytest.raises(ValidationError):
        Scene(width=0, height=600)
    with pytest.raises(ValidationError):
        Scene(width=800, height=-10)
    with pytest.raises(ValidationError):
        Scene(width=800, height=600, background="white")  # type: ignore


def test_scene_add_and_duplicate_id():
    s = Scene(800, 600)
    c1 = Circle(center=Point(100, 100), radius=20, id="circle-1")
    s.add(c1)
    assert len(s) == 1
    assert s.get("circle-1") is c1

    c2 = Circle(center=Point(200, 200), radius=30, id="circle-1")
    with pytest.raises(ValidationError):
        s.add(c2)


def test_scene_remove():
    s = Scene(800, 600)
    c1 = Circle(center=Point(100, 100), radius=20, id="c1")
    s.add(c1)
    removed = s.remove("c1")
    assert removed is c1
    assert len(s) == 0
    assert s.get("c1") is None

    with pytest.raises(ObjectNotFoundError):
        s.remove("c1")


def test_scene_queries():
    s = Scene(800, 600)
    c1 = Circle(center=Point(50, 50), radius=10, name="target")
    c1.tags.add("blue")
    c1.tags.add("shape")

    r1 = Rectangle(position=Point(100, 100), width=50, height=50, name="box")
    r1.tags.add("blue")

    l1 = Line(start=Point(0, 0), end=Point(10, 10))

    s.add(c1)
    s.add(r1)
    s.add(l1)

    assert s.find_by_name("target") == [c1]
    assert s.find_by_name("nonexistent") == []

    assert set(s.find_by_tag("blue")) == {c1, r1}
    assert s.find_by_tag("shape") == [c1]

    assert s.find_by_type(Circle) == [c1]
    assert s.find_by_type(Rectangle) == [r1]
    assert s.find_by_type(Line) == [l1]


def test_scene_visibility_and_locking():
    s = Scene(800, 600)
    c = Circle(center=Point(50, 50), radius=10, id="c1")
    s.add(c)

    assert c.visible is True
    s.hide("c1")
    assert c.visible is False
    s.show("c1")
    assert c.visible is True

    assert c.locked is False
    s.lock("c1")
    assert c.locked is True
    s.unlock("c1")
    assert c.locked is False


def test_scene_z_order_reordering():
    s = Scene(800, 600)
    c1 = Circle(center=Point(10, 10), radius=5, id="c1")
    c2 = Circle(center=Point(20, 20), radius=5, id="c2")
    c3 = Circle(center=Point(30, 30), radius=5, id="c3")
    s.add(c1)
    s.add(c2)
    s.add(c3)

    # Initially all have z_index = 0
    s.move_to_front("c1")
    assert c1.z_index > c3.z_index
    assert s.objects[-1] is c1

    s.move_to_back("c1")
    assert c1.z_index < c2.z_index
    assert s.objects[0] is c1

    s.move_forward("c1")
    assert c1.z_index == -1 + 1 == 0
