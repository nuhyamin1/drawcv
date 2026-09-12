"""Unit tests for Layers in DrawCV."""

import math
import pytest

from drawcv import (
    Circle,
    Color,
    Layer,
    ObjectNotFoundError,
    OpenCVRenderer,
    Point,
    Rectangle,
    Scene,
    ValidationError,
)


def test_layer_creation_and_ordering():
    scene = Scene(800, 600)
    # Default layer exists at z_order 0
    assert len(scene.layers) == 1
    assert scene.layers[0].name == "default"
    assert scene.layers[0].z_order == 0

    l_bg = scene.create_layer("background", z_order=-10)
    l_fg = scene.create_layer("foreground", z_order=10)

    # Layers are sorted by ascending z_order
    sorted_layers = scene.layers
    assert [l.name for l in sorted_layers] == ["background", "default", "foreground"]

    # Duplicate name raises
    with pytest.raises(ValidationError, match="already exists"):
        scene.create_layer("background")


def test_layer_reordering():
    scene = Scene(800, 600)
    scene.create_layer("L1", z_order=10)
    scene.create_layer("L2", z_order=20)

    scene.move_layer_to_back("L2")
    assert scene.get_layer("L2").z_order < scene.get_layer("default").z_order

    scene.move_layer_to_front("L2")
    assert scene.get_layer("L2").z_order > scene.get_layer("L1").z_order

    scene.move_layer_backward("L2")
    scene.move_layer_forward("L1")


def test_default_layer_protection_and_layer_removal():
    scene = Scene(800, 600)
    c1 = Circle(center=Point(100, 100), radius=20, id="c1")
    scene.add(c1)

    # Cannot delete default layer
    with pytest.raises(ValidationError, match="Cannot remove the default layer"):
        scene.remove_layer("default")

    # Add custom layer with objects
    scene.create_layer("annotations", z_order=50)
    c2 = Circle(center=Point(200, 200), radius=30, id="c2")
    c3 = Circle(center=Point(300, 300), radius=40, id="c3")
    scene.add(c2, layer="annotations")
    scene.add(c3, layer="annotations")
    assert scene.get("c2") is c2
    assert scene.get("c3") is c3

    # Remove with remove_objects=False: migrates to default layer
    scene.remove_layer("annotations", remove_objects=False)
    assert scene.get("c2") is c2
    assert c2.layer.name == "default"
    assert c2 in scene.get_layer("default").objects

    # Now create another layer and remove with remove_objects=True: deletes & unregisters
    scene.create_layer("temp", z_order=60)
    c4 = Circle(center=Point(400, 400), radius=20, id="c4")
    scene.add(c4, layer="temp")
    assert scene.get("c4") is c4

    scene.remove_layer("temp", remove_objects=True)
    assert scene.get("c4") is None


def test_layer_cascading_states():
    scene = Scene(800, 600)
    layer = scene.create_layer("test_layer", opacity=0.5, visible=True, locked=False)
    c = Circle(center=Point(100, 100), radius=25, opacity=0.8)
    scene.add(c, layer="test_layer")

    # Opacity cascades: 0.8 * 0.5 = 0.4
    assert math.isclose(c.effective_opacity, 0.4)
    assert c.effective_visible is True
    assert c.effective_locked is False

    # Hide layer
    layer.hide()
    assert c.effective_visible is False

    # Show and lock layer
    layer.show()
    layer.lock()
    assert c.effective_locked is True


def test_layer_hit_testing_order():
    scene = Scene(800, 600)
    # Background layer with large rectangle
    scene.create_layer("bg", z_order=10)
    # Foreground layer with smaller circle on top of rectangle
    scene.create_layer("fg", z_order=20)

    rect = Rectangle(position=Point(50, 50), width=200, height=200, id="rect_bg")
    circ = Circle(center=Point(100, 100), radius=30, id="circ_fg")

    scene.add(rect, layer="bg")
    scene.add(circ, layer="fg")

    # Point (100, 100) hits both circle and rectangle
    hits = scene.hit_test(100, 100)
    assert len(hits) == 2
    # Foreground layer item MUST be first!
    assert hits[0].id == "circ_fg"
    assert hits[1].id == "rect_bg"

    # If foreground layer is hidden, only background is hit
    scene.hide_layer("fg")
    hits_hidden = scene.hit_test(100, 100)
    assert len(hits_hidden) == 1
    assert hits_hidden[0].id == "rect_bg"
