"""Comprehensive test suite for DrawCV Phase 7 — Serialization and Schema Migration."""

import copy
import json
import math
import numpy as np
import pytest

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, ArrowHeadStyle, BlurType, FillRule, FontFamily, JoinStyle, LineType, MaskMapping, TextAlignment
from drawcv.core.exceptions import InvalidFormatError, SerializationError, UnknownDrawableTypeError, UnsupportedVersionError
from drawcv.core.geometry import Point, StrokePoint
from drawcv.core.transform import Transform
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect
from drawcv.effects.mask import Mask
from drawcv.effects.shadow import ShadowEffect
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.serialization import (
    CURRENT_FORMAT_IDENTIFIER,
    CURRENT_SCHEMA_VERSION,
    SchemaMigrator,
    from_json,
    get_drawable_class,
    is_drawable_type_registered,
    register_drawable_type,
    to_json,
)
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.image import ImageObject
from drawcv.shapes.line import Line
from drawcv.shapes.path import Close, CubicTo, LineTo, MoveTo, Path, QuadraticTo, Subpath
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.shapes.text import Text
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


class TestPrimitivesSerialization:
    """Test semantic serialization of core mathematical and styling primitives."""

    def test_point_round_trip(self):
        p = Point(123.45, 678.90)
        d = p.to_dict()
        assert d == {"x": 123.45, "y": 678.90}
        restored = Point.from_dict(d)
        assert restored.x == 123.45
        assert restored.y == 678.90

    def test_stroke_point_round_trip(self):
        sp = StrokePoint(10.5, 20.5, pressure=0.8, timestamp=100.0, velocity=15.0)
        d = sp.to_dict()
        assert d["x"] == 10.5
        assert d["pressure"] == 0.8
        assert d["timestamp"] == 100.0
        assert d["velocity"] == 15.0
        restored = StrokePoint.from_dict(d)
        assert restored.x == 10.5
        assert restored.pressure == 0.8
        assert restored.timestamp == 100.0
        assert restored.velocity == 15.0

    def test_color_lossless_float_alpha(self):
        # 0.5 cannot be accurately represented in 8-bit without 1/255 drift if quantized
        c = Color(100, 150, 200, 0.5)
        d = c.to_dict()
        assert d == {"r": 100, "g": 150, "b": 200, "a": 0.5}
        restored = Color.from_dict(d)
        assert restored.r == 100
        assert restored.g == 150
        assert restored.b == 200
        assert restored.a == 0.5  # Zero drift!

    def test_color_from_dict_flexible_inputs(self):
        assert Color.from_dict("#FF0000").r == 255
        assert Color.from_dict([10, 20, 30]).g == 20
        assert Color.from_dict([10, 20, 30, 0.25]).a == 0.25

    def test_transform_round_trip(self):
        tf = Transform(
            translation_x=50.0,
            translation_y=100.0,
            rotation=45.0,
            scale_x=1.5,
            scale_y=2.0,
            pivot=Point(25.0, 35.0),
        )
        d = tf.to_dict()
        restored = Transform.from_dict(d)
        assert restored == tf

    def test_bounding_box_round_trip(self):
        bb = BoundingBox(10.0, 20.0, 100.0, 200.0)
        d = bb.to_dict()
        assert d == {"x": 10.0, "y": 20.0, "width": 100.0, "height": 200.0}
        restored = BoundingBox.from_dict(d)
        assert restored == bb

    def test_stroke_style_round_trip(self):
        stroke = StrokeStyle(
            color=Color(255, 0, 128, 0.8),
            width=3.5,
            opacity=0.9,
            line_type=LineType.AA,
            join_style=JoinStyle.ROUND,
        )
        d = stroke.to_dict()
        restored = StrokeStyle.from_dict(d)
        assert restored.color.a == 0.8
        assert restored.width == 3.5
        assert restored.opacity == 0.9
        assert restored.line_type == LineType.AA
        assert restored.join_style == JoinStyle.ROUND

    def test_fill_style_round_trip(self):
        fill = FillStyle(
            enabled=True,
            color=Color(0, 255, 0, 0.4),
            opacity=0.85,
        )
        d = fill.to_dict()
        restored = FillStyle.from_dict(d)
        assert restored.enabled is True
        assert restored.color.g == 255
        assert restored.color.a == 0.4
        assert restored.opacity == 0.85


class TestEffectsSerialization:
    """Test serialization of clipping, raster masks, and raster effects."""

    def test_clip_rect_round_trip(self):
        cr = ClipRect(10, 20, 200, 300)
        d = cr.to_dict()
        assert d["type"] == "rect"
        restored = ClipRect.from_dict(d)
        assert restored.x == 10
        assert restored.width == 200

    def test_clip_path_round_trip(self):
        pts = [Point(0, 0), Point(100, 0), Point(50, 100)]
        cp = ClipPath(pts)
        d = cp.to_dict()
        assert d["type"] == "path"
        assert len(d["points"]) == 3
        restored = ClipPath.from_dict(d)
        assert len(restored.points) == 3
        assert restored.points[1] == Point(100, 0)

    def test_mask_raster_envelope_and_validation(self):
        arr = np.zeros((100, 150), dtype=np.uint8)
        arr[20:80, 30:120] = 255
        mask = Mask(arr, inverted=True)
        d = mask.to_dict()

        assert d["inverted"] is True
        raster = d["buffer"]
        assert raster["encoding"] == "png_base64"
        assert raster["dtype"] == "uint8"
        assert raster["shape"] == [100, 150]
        assert isinstance(raster["data"], str)

        restored = Mask.from_dict(d)
        assert restored.inverted is True
        assert restored.buffer.shape == (100, 150)
        assert restored.buffer.dtype == np.uint8
        assert np.array_equal(restored.buffer, arr)

    def test_blur_effect_round_trip(self):
        blur = BlurEffect(kernel_size=7, sigma=2.5, blur_type=BlurType.GAUSSIAN)
        d = blur.to_dict()
        assert d["type"] == "blur"
        assert d["blur_type"] == "gaussian"
        assert d["kernel_size"] == 7
        assert d["sigma"] == 2.5
        restored = BlurEffect.from_dict(d)
        assert restored.blur_type == BlurType.GAUSSIAN
        assert restored.kernel_size == 7
        assert restored.sigma == 2.5

    def test_shadow_effect_round_trip(self):
        shadow = ShadowEffect(
            offset_x=10.0,
            offset_y=15.0,
            blur_radius=5.0,
            color=Color(0, 0, 0, 0.6),
            opacity=0.9,
        )
        d = shadow.to_dict()
        assert d["type"] == "shadow"
        assert d["color"]["a"] == 0.6
        restored = ShadowEffect.from_dict(d)
        assert restored.offset_x == 10.0
        assert restored.offset_y == 15.0
        assert restored.blur_radius == 5.0
        assert restored.color.a == 0.6
        assert restored.opacity == 0.9


class TestAllShapesSerialization:
    """Exhaustive test for all 14 shapes and Group."""

    def test_line(self):
        shape = Line(
            start=Point(10, 20),
            end=Point(100, 200),
            stroke=StrokeStyle(color=Color.blue(), width=2),
            tags={"tag1", "tag2"},
            metadata={"source": "cad"},
        )
        d = shape.to_dict()
        assert d["type"] == "line"
        assert d["tags"] == ["tag1", "tag2"]
        assert d["metadata"] == {"source": "cad"}
        restored = Line.from_dict(d)
        assert restored.start == shape.start
        assert restored.end == shape.end
        assert restored.tags == {"tag1", "tag2"}
        assert restored.metadata == {"source": "cad"}

    def test_rectangle(self):
        shape = Rectangle(
            position=Point(50, 60),
            width=200,
            height=150,
            fill=FillStyle(color=Color.green()),
            stroke=StrokeStyle(color=Color.black(), width=1),
        )
        d = shape.to_dict()
        assert d["type"] == "rectangle"
        restored = Rectangle.from_dict(d)
        assert restored.position == shape.position
        assert restored.width == shape.width
        assert restored.height == shape.height

    def test_rounded_rectangle(self):
        shape = RoundedRectangle(
            x=30,
            y=40,
            width=120,
            height=80,
            corner_radius=15.0,
            fill=FillStyle(color=Color.red()),
        )
        d = shape.to_dict()
        assert d["type"] == "rounded_rectangle"
        restored = RoundedRectangle.from_dict(d)
        assert restored.corner_radius == shape.corner_radius
        assert restored.x == 30
        assert restored.y == 40

    def test_circle(self):
        shape = Circle(
            center=Point(150, 150),
            radius=45.5,
            fill=FillStyle(color=Color(255, 200, 0, 0.7)),
        )
        d = shape.to_dict()
        assert d["type"] == "circle"
        assert d["radius"] == 45.5
        restored = Circle.from_dict(d)
        assert restored.center == shape.center
        assert restored.radius == 45.5

    def test_ellipse(self):
        shape = Ellipse(
            center=Point(200, 200),
            radius_x=60,
            radius_y=30,
        )
        d = shape.to_dict()
        assert d["type"] == "ellipse"
        restored = Ellipse.from_dict(d)
        assert restored.radius_x == 60
        assert restored.radius_y == 30

    def test_arc(self):
        shape = Arc(
            center=Point(100, 100),
            radius_x=50,
            radius_y=50,
            start_angle=0,
            sweep_angle=180,
            closure=ArcClosure.PIE,
        )
        d = shape.to_dict()
        assert d["type"] == "arc"
        assert d["closure"] == "pie"
        restored = Arc.from_dict(d)
        assert restored.closure == ArcClosure.PIE
        assert restored.start_angle == 0
        assert restored.sweep_angle == 180

    def test_arrow(self):
        shape = Arrow(
            start=Point(20, 20),
            end=Point(120, 120),
            head_style=ArrowHeadStyle.TRIANGLE,
            head_length=18.0,
            head_width=12.0,
        )
        d = shape.to_dict()
        assert d["type"] == "arrow"
        assert d["head_style"] == "triangle"
        assert d["head_length"] == 18.0
        assert d["head_width"] == 12.0
        restored = Arrow.from_dict(d)
        assert restored.head_style == ArrowHeadStyle.TRIANGLE
        assert restored.head_length == 18.0
        assert restored.head_width == 12.0

    def test_polygon_and_polyline(self):
        pts = [Point(10, 10), Point(50, 20), Point(80, 90), Point(20, 60)]
        poly = Polygon(vertices=pts, fill=FillStyle(color=Color.cyan()))
        pline = Polyline(points=pts, closed=False)

        d_poly = poly.to_dict()
        assert d_poly["type"] == "polygon"
        restored_poly = Polygon.from_dict(d_poly)
        assert len(restored_poly.vertices) == 4

        d_pline = pline.to_dict()
        assert d_pline["type"] == "polyline"
        restored_pline = Polyline.from_dict(d_pline)
        assert len(restored_pline.points) == 4
        assert restored_pline.closed is False

    def test_bezier_curve(self):
        bez = BezierCurve(
            p0=Point(10, 10),
            p1=Point(50, 150),
            p2=Point(150, 150),
            p3=Point(200, 10),
        )
        d = bez.to_dict()
        assert d["type"] == "bezier"
        assert "p0" in d
        assert "p3" in d
        restored = BezierCurve.from_dict(d)
        assert restored.p1 == bez.p1
        assert restored.p2 == bez.p2
        assert restored.p3 == bez.p3

    def test_path_with_commands(self):
        path = Path()
        path.move_to(10, 10)
        path.line_to(50, 10)
        path.quadratic_to(Point(70, 30), Point(50, 50))
        path.cubic_to(Point(30, 70), Point(10, 50), Point(10, 10))
        path.close()

        d = path.to_dict()
        assert d["type"] == "path"
        assert len(d["subpaths"]) == 1
        assert len(d["subpaths"][0]["commands"]) == 5

        restored = Path.from_dict(d)
        assert len(restored.subpaths) == 1
        cmds = restored.subpaths[0].commands
        assert isinstance(cmds[0], MoveTo)
        assert isinstance(cmds[1], LineTo)
        assert isinstance(cmds[2], QuadraticTo)
        assert isinstance(cmds[3], CubicTo)
        assert isinstance(cmds[4], Close)

    def test_freehand_stroke(self):
        pts = [
            StrokePoint(10, 10, pressure=0.2),
            StrokePoint(20, 15, pressure=0.5),
            StrokePoint(30, 25, pressure=0.8),
        ]
        fh = FreehandStroke(
            points=pts,
            smoothing="chaikin",
            smoothing_iterations=2,
            simplification="rdp",
            simplification_tolerance=1.5,
            variable_width=True,
            min_width=2.0,
            max_width=8.0,
        )
        d = fh.to_dict()
        assert d["type"] == "freehand"
        assert d["smoothing"] == "chaikin"
        assert d["variable_width"] is True
        assert len(d["points"]) == 3

        restored = FreehandStroke.from_dict(d)
        assert restored.smoothing == "chaikin"
        assert restored.smoothing_iterations == 2
        assert restored.simplification == "rdp"
        assert restored.simplification_tolerance == 1.5
        assert len(restored.points) == 3
        assert restored.points[1].pressure == 0.5

    def test_text(self):
        txt = Text(
            text="DrawCV 2.0",
            position=Point(100, 150),
            color=Color(50, 100, 150),
            font_family=FontFamily.SIMPLEX,
            font_scale=1.5,
            thickness=2,
            alignment=TextAlignment.CENTER,
        )
        d = txt.to_dict()
        assert d["type"] == "text"
        assert d["text"] == "DrawCV 2.0"
        assert d["alignment"] == "center"
        restored = Text.from_dict(d)
        assert restored.text == "DrawCV 2.0"
        assert restored.font_scale == 1.5
        assert restored.alignment == TextAlignment.CENTER
        assert restored.color.r == 50

    def test_image_object_raster_payload(self):
        img_arr = np.zeros((40, 60, 3), dtype=np.uint8)
        img_arr[10:30, 10:50] = [120, 180, 240]
        iobj = ImageObject(image=img_arr, position=Point(15, 25))
        d = iobj.to_dict()

        assert d["type"] == "image"
        raster = d["image"]
        assert raster["encoding"] == "png_base64"
        assert raster["dtype"] == "uint8"
        assert raster["shape"] == [40, 60, 3]

        restored = ImageObject.from_dict(d)
        assert restored.position == Point(15, 25)
        assert restored.image.shape == (40, 60, 3)
        assert np.array_equal(restored.image, img_arr)

    def test_group_hierarchy_round_trip(self):
        c1 = Circle(center=Point(10, 10), radius=5, id="c1")
        c2 = Rectangle(position=Point(20, 20), width=30, height=40, id="r1")
        sub_group = Group(children=[c2], id="sub_grp", name="Sub")
        main_group = Group(children=[c1, sub_group], id="main_grp", name="Main")

        d = main_group.to_dict()
        assert d["type"] == "group"
        assert len(d["children"]) == 2
        assert d["children"][1]["type"] == "group"

        restored = Group.from_dict(d)
        assert restored.id == "main_grp"
        assert len(restored.children) == 2
        assert restored.children[0].id == "c1"
        assert restored.children[1].id == "sub_grp"
        assert len(restored.children[1].children) == 1
        assert restored.children[1].children[0].id == "r1"


class TestSceneDocumentSerialization:
    """Test Scene document envelope, JSON encoding, canonical sorting, and versioning."""

    def test_scene_root_envelope(self):
        scene = Scene(800, 600, background=Color(30, 30, 30))
        c = Circle(center=Point(100, 100), radius=50, fill=FillStyle(color=Color.red()))
        scene.add(c)

        doc = scene.to_dict()
        assert doc["format"] == CURRENT_FORMAT_IDENTIFIER
        assert doc["version"] == CURRENT_SCHEMA_VERSION
        assert "scene" in doc
        s_data = doc["scene"]
        assert s_data["width"] == 800
        assert s_data["height"] == 600
        assert s_data["background"]["r"] == 30
        assert len(s_data["layers"]) == 1
        assert len(s_data["layers"][0]["objects"]) == 1

    def test_canonical_json_sorting(self):
        scene = Scene(400, 400)
        c1 = Circle(center=Point(50, 50), radius=20, id="c1")
        # Set arbitrary metadata insertion order
        c1.metadata = {"z": 100, "a": 200, "m": 300}
        scene.add(c1)

        json_str_1 = scene.to_json()

        # Change insertion order
        c1.metadata = {"a": 200, "m": 300, "z": 100}
        json_str_2 = scene.to_json()

        # sort_keys=True guarantees bit-for-bit equivalence!
        assert json_str_1 == json_str_2

    def test_strict_json_rejection_of_non_finite_floats(self):
        data = {"format": "drawcv", "version": "1.0", "val": float("nan")}
        with pytest.raises(SerializationError, match="Non-finite float"):
            to_json(data)

        data_inf = {"format": "drawcv", "version": "1.0", "val": float("inf")}
        with pytest.raises(SerializationError, match="Non-finite float"):
            to_json(data_inf)

    def test_invalid_format_and_version(self):
        bad_format = json.dumps({"format": "unknown", "version": "1.0", "scene": {}})
        with pytest.raises(InvalidFormatError):
            Scene.from_json(bad_format)

        bad_version = json.dumps({"format": "drawcv", "version": "99.0", "scene": {}})
        with pytest.raises(UnsupportedVersionError):
            Scene.from_json(bad_version)

    def test_schema_migration_pipeline(self):
        # Register a mock 0.9 -> 1.0 migration
        def migrate_0_9_to_1_0(doc: dict) -> dict:
            migrated = copy.deepcopy(doc)
            migrated["version"] = "1.0"
            if "legacy_title" in migrated["scene"]:
                migrated["scene"]["title"] = migrated["scene"].pop("legacy_title")
            return migrated

        SchemaMigrator.register_migration("0.9", migrate_0_9_to_1_0)
        try:
            legacy_doc = {
                "format": "drawcv",
                "version": "0.9",
                "scene": {
                    "width": 640,
                    "height": 480,
                    "background": {"r": 0, "g": 0, "b": 0, "a": 1.0},
                    "legacy_title": "Old Document",
                    "layers": [],
                },
            }
            json_text = json.dumps(legacy_doc)
            scene = Scene.from_json(json_text)
            assert scene.width == 640
            assert scene.height == 480
        finally:
            SchemaMigrator.unregister_migration("0.9")

    def test_loaded_scene_has_empty_history(self):
        scene = Scene(500, 500)
        scene.add(Circle(center=Point(100, 100), radius=20))
        assert scene.can_undo is True

        json_text = scene.to_json()
        loaded = Scene.from_json(json_text)
        assert loaded.can_undo is False
        assert loaded.can_redo is False
        assert loaded.history.undo_count == 0

    def test_file_save_and_load(self, tmp_path):
        filepath = tmp_path / "test_doc.drawcv"
        scene = Scene(600, 400, background=Color(240, 240, 240))
        rect = Rectangle(position=Point(50, 50), width=100, height=80, fill=FillStyle(color=Color.blue()))
        scene.add(rect)

        scene.save_json(filepath)
        assert filepath.exists()

        loaded = Scene.load_json(filepath)
        assert loaded.width == 600
        assert loaded.height == 400
        assert len(loaded.objects) == 1
        assert isinstance(loaded.objects[0], Rectangle)
        assert loaded.objects[0].width == 100

    def test_pixel_perfect_visual_equivalence(self):
        scene = Scene(300, 300, background=Color(255, 255, 255))
        rect = Rectangle(position=Point(30, 30), width=100, height=100, fill=FillStyle(color=Color.red()))
        circle = Circle(center=Point(150, 150), radius=60, fill=FillStyle(color=Color(0, 0, 255, 0.7)))
        scene.add(rect)
        scene.add(circle)

        renderer = OpenCVRenderer()
        original_render = renderer.render(scene)

        # Round-trip via JSON
        json_data = scene.to_json()
        restored_scene = Scene.from_json(json_data)
        restored_render = renderer.render(restored_scene)

        assert np.array_equal(original_render.to_numpy(), restored_render.to_numpy())
