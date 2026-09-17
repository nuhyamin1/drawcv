"""Registry for serializable drawable types and schema version migration management."""

from __future__ import annotations
from typing import Any, Callable, Type
import copy

from drawcv.core.exceptions import (
    InvalidFormatError,
    SerializationError,
    UnknownDrawableTypeError,
    UnsupportedVersionError,
)

CURRENT_SCHEMA_VERSION = "1.8"
CURRENT_FORMAT_IDENTIFIER = "drawcv"

_DRAWABLE_REGISTRY: dict[str, type] = {}
_DESERIALIZER_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_drawable_type(
    type_name: str,
    cls: type,
    deserializer: Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Register a drawable type identifier with its class and deserializer.
    
    Args:
        type_name: Canonical string type name (e.g., 'circle', 'line').
        cls: The Drawable subclass.
        deserializer: Optional custom deserializer callable. If omitted,
            cls.from_dict is used.
    """
    if not isinstance(type_name, str) or not type_name.strip():
        raise SerializationError("Drawable type name must be a non-empty string")
    
    clean_name = type_name.strip().lower()
    _DRAWABLE_REGISTRY[clean_name] = cls
    if deserializer is not None:
        _DESERIALIZER_REGISTRY[clean_name] = deserializer
    elif hasattr(cls, "from_dict"):
        _DESERIALIZER_REGISTRY[clean_name] = getattr(cls, "from_dict")


_BUILTIN_REGISTERED = False


def register_builtin_types() -> None:
    """Register all standard built-in DrawCV drawable types."""
    global _BUILTIN_REGISTERED
    if _BUILTIN_REGISTERED:
        return
    _BUILTIN_REGISTERED = True

    from drawcv.group import Group
    from drawcv.shapes.arc import Arc
    from drawcv.shapes.arrow import Arrow
    from drawcv.shapes.bezier import BezierCurve
    from drawcv.shapes.circle import Circle
    from drawcv.shapes.ellipse import Ellipse
    from drawcv.shapes.freehand import FreehandStroke
    from drawcv.shapes.image import ImageObject
    from drawcv.shapes.line import Line
    from drawcv.shapes.path import Path
    from drawcv.shapes.polygon import Polygon
    from drawcv.shapes.polyline import Polyline
    from drawcv.shapes.rectangle import Rectangle
    from drawcv.shapes.rounded_rectangle import RoundedRectangle
    from drawcv.shapes.text import Text

    register_drawable_type("arc", Arc)
    register_drawable_type("arrow", Arrow)
    register_drawable_type("bezier", BezierCurve)
    register_drawable_type("bezier_curve", BezierCurve)
    register_drawable_type("circle", Circle)
    register_drawable_type("ellipse", Ellipse)
    register_drawable_type("freehand", FreehandStroke)
    register_drawable_type("image", ImageObject)
    register_drawable_type("image_object", ImageObject)
    register_drawable_type("line", Line)
    register_drawable_type("path", Path)
    register_drawable_type("polygon", Polygon)
    register_drawable_type("polyline", Polyline)
    register_drawable_type("rectangle", Rectangle)
    register_drawable_type("rounded_rectangle", RoundedRectangle)
    register_drawable_type("rounded_rect", RoundedRectangle)
    register_drawable_type("text", Text)
    register_drawable_type("group", Group)


def get_drawable_class(type_name: str) -> type:
    """Retrieve the registered class for the given type name."""
    register_builtin_types()
    clean_name = str(type_name).strip().lower()
    if clean_name not in _DRAWABLE_REGISTRY:
        raise UnknownDrawableTypeError(f"Unrecognized drawable type '{type_name}'")
    return _DRAWABLE_REGISTRY[clean_name]


def get_drawable_deserializer(type_name: str) -> Callable[[dict[str, Any]], Any]:
    """Retrieve the deserializer function for the given type name."""
    register_builtin_types()
    clean_name = str(type_name).strip().lower()
    if clean_name not in _DESERIALIZER_REGISTRY:
        if clean_name in _DRAWABLE_REGISTRY:
            cls = _DRAWABLE_REGISTRY[clean_name]
            if hasattr(cls, "from_dict"):
                return getattr(cls, "from_dict")
        raise UnknownDrawableTypeError(f"Unrecognized drawable type '{type_name}'")
    return _DESERIALIZER_REGISTRY[clean_name]


def is_drawable_type_registered(type_name: str) -> bool:
    """Check if a drawable type identifier is registered."""
    register_builtin_types()
    return str(type_name).strip().lower() in _DRAWABLE_REGISTRY


class SchemaMigrator:
    """Manages document format version validation and forward migration pipelines."""

    _migrations: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    @classmethod
    def register_migration(
        cls,
        from_version: str,
        migration_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Register a schema migration function converting from from_version to next version."""
        cls._migrations[from_version] = migration_fn

    @classmethod
    def unregister_migration(cls, from_version: str) -> None:
        """Unregister a previously registered migration (useful for test fixtures)."""
        cls._migrations.pop(from_version, None)

    @classmethod
    def validate_envelope(cls, data: dict[str, Any]) -> None:
        """Validate the canonical root document envelope."""
        if not isinstance(data, dict):
            raise InvalidFormatError(f"Root document must be a dictionary, got {type(data).__name__}")
        
        if "format" not in data:
            raise InvalidFormatError("Missing required document header 'format'")
        if data["format"] != CURRENT_FORMAT_IDENTIFIER:
            raise InvalidFormatError(
                f"Invalid format identifier '{data['format']}'. Expected '{CURRENT_FORMAT_IDENTIFIER}'"
            )
        
        if "version" not in data:
            if "schema_version" in data:
                data["version"] = data["schema_version"]
            else:
                raise InvalidFormatError("Missing required document header 'version'")
        
        version_str = str(data["version"]).strip()
        if not version_str or not any(c.isdigit() for c in version_str):
            raise InvalidFormatError(f"Malformed version string: '{data['version']}'")

    @classmethod
    def migrate(cls, data: dict[str, Any], target_version: str = CURRENT_SCHEMA_VERSION) -> dict[str, Any]:
        """Validate and migrate document data up to target_version.
        
        Returns a migrated, plain dictionary compatible with target_version schema.
        """
        cls.validate_envelope(data)
        current = copy.deepcopy(data)
        doc_version = str(current["version"]).strip()

        if doc_version == target_version:
            return current

        # Sequential migration chain
        visited = set()
        while doc_version != target_version:
            if doc_version in visited:
                raise SerializationError(f"Cyclic migration detected at version '{doc_version}'")
            visited.add(doc_version)

            if doc_version not in cls._migrations:
                raise UnsupportedVersionError(
                    f"Unsupported schema version '{doc_version}'. Current supported version is '{target_version}'."
                )

            migration_fn = cls._migrations[doc_version]
            current = migration_fn(current)
            doc_version = str(current.get("version", "")).strip()

        return current


def _migrate_1_0_to_1_1(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate document schema 1.0 to 1.1 supplying Phase 8 temporal defaults."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.1"

    scene_data = migrated.get("scene", {})
    if "timeline" not in scene_data:
        scene_data["timeline"] = {"tracks": []}

    def _patch_drawable(d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        if "timing" not in d:
            d["timing"] = None
        if "render_progress" not in d:
            d["render_progress"] = 1.0
        if d.get("type") == "group" and "children" in d:
            for child in d.get("children", []):
                _patch_drawable(child)

    for layer in scene_data.get("layers", []):
        for obj_data in layer.get("objects", []):
            _patch_drawable(obj_data)

    return migrated


SchemaMigrator.register_migration("1.0", _migrate_1_0_to_1_1)


def _migrate_1_1_to_1_2(data: dict[str, Any]) -> dict[str, Any]:
    """Stroke readers supply solid/zero-offset/4x-miter defaults recursively."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.2"
    return migrated


SchemaMigrator.register_migration("1.1", _migrate_1_1_to_1_2)


def _migrate_1_2_to_1_3(data):
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.3"
    return migrated


SchemaMigrator.register_migration("1.2", _migrate_1_2_to_1_3)


def _migrate_1_3_to_1_4(data):
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.4"
    return migrated


SchemaMigrator.register_migration("1.3", _migrate_1_3_to_1_4)


def _migrate_1_4_to_1_5(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate document schema 1.4 to 1.5 supplying blend_mode='normal' default."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.5"

    def _patch_drawable(d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        if "blend_mode" not in d:
            d["blend_mode"] = "normal"
        if d.get("type") == "group" and "children" in d:
            for child in d.get("children", []):
                _patch_drawable(child)

    scene_data = migrated.get("scene", {})
    for layer in scene_data.get("layers", []):
        if isinstance(layer, dict):
            if "blend_mode" not in layer:
                layer["blend_mode"] = "normal"
            for obj_data in layer.get("objects", []):
                _patch_drawable(obj_data)

    return migrated


SchemaMigrator.register_migration("1.4", _migrate_1_4_to_1_5)


def _migrate_1_5_to_1_6(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate document schema 1.5 to 1.6 supplying paint, spread, and stroke paint defaults."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.6"

    def _patch_paint(paint_dict: dict[str, Any]) -> None:
        if not isinstance(paint_dict, dict):
            return
        ptype = paint_dict.get("type")
        if ptype in ("linear", "radial"):
            if "spread" not in paint_dict:
                paint_dict["spread"] = "pad"
            if "transform" not in paint_dict:
                paint_dict["transform"] = {
                    "translation_x": 0.0,
                    "translation_y": 0.0,
                    "rotation": 0.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "pivot": None,
                }
        elif ptype == "conic":
            if "start_angle" not in paint_dict:
                paint_dict["start_angle"] = 0.0
            if "transform" not in paint_dict:
                paint_dict["transform"] = {
                    "translation_x": 0.0,
                    "translation_y": 0.0,
                    "rotation": 0.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "pivot": None,
                }

    def _patch_drawable(d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        fill = d.get("fill")
        if isinstance(fill, dict):
            paint = fill.get("paint")
            if isinstance(paint, dict):
                _patch_paint(paint)

        stroke = d.get("stroke")
        if isinstance(stroke, dict):
            if isinstance(stroke.get("paint"), dict):
                _patch_paint(stroke["paint"])

        if d.get("type") == "group" and "children" in d:
            for child in d.get("children", []):
                _patch_drawable(child)

    scene_data = migrated.get("scene", {})
    for layer in scene_data.get("layers", []):
        if isinstance(layer, dict):
            for obj_data in layer.get("objects", []):
                _patch_drawable(obj_data)

    return migrated


SchemaMigrator.register_migration("1.5", _migrate_1_5_to_1_6)


def _migrate_1_6_to_1_7(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate document schema 1.6 to 1.7 migrating legacy polygon clips."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.7"
    if "schema_version" in migrated:
        migrated["schema_version"] = "1.7"

    def _patch_clip_dict(clip_dict: Any) -> None:
        if isinstance(clip_dict, dict) and clip_dict.get("type") == "path" and "points" in clip_dict:
            clip_dict["type"] = "polygon"

    def _patch_drawable(d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        if "clip" in d and d["clip"] is not None:
            _patch_clip_dict(d["clip"])
        if d.get("type") == "group" and "children" in d:
            for child in d.get("children", []):
                _patch_drawable(child)

    scene_data = migrated.get("scene", {})
    for layer in scene_data.get("layers", []):
        if isinstance(layer, dict):
            if "clip" in layer and layer["clip"] is not None:
                _patch_clip_dict(layer["clip"])
            for obj_data in layer.get("objects", []):
                _patch_drawable(obj_data)

    return migrated


SchemaMigrator.register_migration("1.6", _migrate_1_6_to_1_7)


def _migrate_1_7_to_1_8(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-migrate document schema 1.7 to 1.8 for M2 exact paths and reusable content."""
    migrated = copy.deepcopy(data)
    migrated["version"] = "1.8"
    if "schema_version" in migrated:
        migrated["schema_version"] = "1.8"
    return migrated


SchemaMigrator.register_migration("1.7", _migrate_1_7_to_1_8)
