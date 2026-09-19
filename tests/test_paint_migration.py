"""Tests for Schema 1.6 migration and paint serialization round-tripping."""

import numpy as np
import pytest

from drawcv import (
    Color,
    ConicGradient,
    FillStyle,
    GradientStop,
    Group,
    ImagePaint,
    Layer,
    LinearGradient,
    Line,
    Point,
    RadialGradient,
    Rectangle,
    Scene,
    StrokeStyle,
    Transform,
    paint_from_dict,
)
from drawcv.serialization.registry import CURRENT_SCHEMA_VERSION, SchemaMigrator


def test_schema_version_is_1_13():
    assert CURRENT_SCHEMA_VERSION == "1.13"


def test_migrate_1_5_to_1_6_legacy_document():
    doc_1_5 = {
        "format": "drawcv",
        "version": "1.5",
        "scene": {
            "width": 200,
            "height": 200,
            "background": {"r": 0, "g": 0, "b": 0, "a": 1.0},
            "timeline": {"tracks": []},
            "layers": [
                {
                    "name": "layer1",
                    "visible": True,
                    "locked": False,
                    "opacity": 1.0,
                    "blend_mode": "normal",
                    "objects": [
                        {
                            "type": "rectangle",
                            "position": {"x": 10.0, "y": 10.0},
                            "width": 80.0,
                            "height": 60.0,
                            "fill": {
                                "enabled": True,
                                "opacity": 1.0,
                                "paint": {
                                    "type": "linear",
                                    "start": {"x": 0.0, "y": 0.0},
                                    "end": {"x": 100.0, "y": 0.0},
                                    "stops": [
                                        {"position": 0.0, "color": {"r": 255, "g": 0, "b": 0, "a": 1.0}},
                                        {"position": 1.0, "color": {"r": 0, "g": 0, "b": 255, "a": 1.0}},
                                    ],
                                    "space": "object",
                                    },
                            },
                            "stroke": {
                                "enabled": True,
                                "color": {"r": 0, "g": 255, "b": 0, "a": 1.0},
                                "width": 2.0,
                                "opacity": 1.0,
                            },
                        }
                    ],
                }
            ],
        },
    }

    migrated_1_6 = SchemaMigrator.migrate(doc_1_5, target_version="1.6")
    assert migrated_1_6["version"] == "1.6"

    migrated_1_7 = SchemaMigrator.migrate(doc_1_5, target_version="1.7")
    assert migrated_1_7["version"] == "1.7"

    migrated_1_8 = SchemaMigrator.migrate(doc_1_5, target_version="1.8")
    assert migrated_1_8["version"] == "1.8"

    migrated = SchemaMigrator.migrate(doc_1_5)
    assert migrated["version"] == CURRENT_SCHEMA_VERSION

    # Verify paint defaults injected
    rect_data = migrated["scene"]["layers"][0]["objects"][0]
    linear_data = rect_data["fill"]["paint"]
    assert linear_data["spread"] == "pad"
    assert "transform" in linear_data
    assert rect_data["stroke"].get("paint") is None
    assert "color" in rect_data["stroke"]

    scene = Scene.from_dict(doc_1_5)
    assert scene.to_dict()["version"] == CURRENT_SCHEMA_VERSION
    loaded_rect = scene.layers[0].objects[0]
    assert isinstance(loaded_rect.fill.paint, LinearGradient)
    assert loaded_rect.fill.paint.spread == "pad"
    assert loaded_rect.stroke.color == Color.green()


def test_full_chain_migration_1_0_to_1_6():
    doc_1_0 = {
        "format": "drawcv",
        "version": "1.0",
        "scene": {
            "width": 100,
            "height": 100,
            "background": {"r": 255, "g": 255, "b": 255, "a": 1.0},
            "layers": [
                {
                    "name": "base",
                    "visible": True,
                    "locked": False,
                    "opacity": 1.0,
                    "objects": [
                        {
                            "type": "line",
                            "start": {"x": 0.0, "y": 0.0},
                            "end": {"x": 50.0, "y": 50.0},
                            "stroke": {
                                "enabled": True,
                                "color": {"r": 0, "g": 0, "b": 0, "a": 1.0},
                                "width": 1.0,
                                "opacity": 1.0,
                            },
                        }
                    ],
                }
            ],
        },
    }

    scene = Scene.from_dict(doc_1_0)
    assert scene.to_dict()["version"] == CURRENT_SCHEMA_VERSION
    assert scene.timeline is not None
    assert scene.layers[0].objects[0].stroke.color == Color(0, 0, 0, 1.0)


def test_transform_matrix_preservation_in_paint_serialization():
    # Construct a non-trivial affine transform with shear via from_matrix
    shear_matrix = np.array([
        [1.0, 0.5, 10.0],
        [0.2, 1.0, 20.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    tf = Transform.from_matrix(shear_matrix)

    paint = ConicGradient(
        center=Point(50, 50),
        stops=(GradientStop(0.0, Color.red()), GradientStop(1.0, Color.blue())),
        start_angle=45.0,
        transform=tf,
    )

    data = paint.to_dict()
    restored = paint_from_dict(data)

    assert isinstance(restored, ConicGradient)
    np.testing.assert_allclose(
        restored.transform.get_matrix(),
        shear_matrix,
        rtol=1e-5,
        atol=1e-5,
    )
