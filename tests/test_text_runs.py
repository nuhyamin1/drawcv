import json
from pathlib import Path
import pytest
from drawcv import (
    Color,
    FontAsset,
    FontFamily,
    Point,
    Scene,
    Text,
    TextAnchor,
    TextRun,
    ValidationError,
    CURRENT_SCHEMA_VERSION,
)
from drawcv.serialization.registry import SchemaMigrator

FONT_PATH = Path("tests/assets/fonts/notosans.ttf")
FONT_ASSET = FontAsset.from_file(FONT_PATH)


def test_text_anchor_values():
    assert TextAnchor.START.value == "start"
    assert TextAnchor.MIDDLE.value == "middle"
    assert TextAnchor.END.value == "end"


def test_text_run_creation_and_validation():
    run = TextRun(
        text="Hello",
        fonts=(FONT_ASSET,),
        font_size=24.0,
        font_family_name="Noto Sans",
        fill=Color.red(),
        fill_opacity=0.8,
        x=10.0,
        y=20.0,
        dx=2.0,
        dy=-1.0,
        text_anchor=TextAnchor.MIDDLE,
        direction="ltr",
    )
    assert run.text == "Hello"
    assert run.fonts == (FONT_ASSET,)
    assert run.font_size == 24.0
    assert run.font_family_name == "Noto Sans"
    assert run.fill == Color.red()
    assert run.fill_opacity == 0.8
    assert run.x == 10.0
    assert run.y == 20.0
    assert run.dx == 2.0
    assert run.dy == -1.0
    assert run.text_anchor == TextAnchor.MIDDLE
    assert run.direction == "ltr"

    # Immutability
    with pytest.raises(AttributeError):
        run.text = "Changed"

    # Validation errors
    with pytest.raises(ValidationError, match="must be a string"):
        TextRun(123)  # type: ignore
    with pytest.raises(ValidationError, match="font_size must be positive"):
        TextRun("test", font_size=-5)
    with pytest.raises(ValidationError, match="fill_opacity must be in range"):
        TextRun("test", fill_opacity=1.5)
    with pytest.raises(ValidationError, match="Invalid TextAnchor"):
        TextRun("test", text_anchor="invalid")
    with pytest.raises(ValidationError, match="direction must be"):
        TextRun("test", direction="top-to-bottom")


def test_text_run_serialization_roundtrip():
    run = TextRun(
        text="Sample",
        fonts=(FONT_ASSET,),
        font_size=18.0,
        font_family_name="Noto Sans",
        font_weight="bold",
        font_style="italic",
        fill=Color.blue(),
        fill_opacity=0.5,
        x=50.0,
        y=100.0,
        dx=5.0,
        dy=2.0,
        text_anchor=TextAnchor.END,
        direction="rtl",
        is_font_substituted=True,
    )
    d = run.to_dict()
    restored = TextRun.from_dict(d)
    assert restored == run
    assert restored.text == "Sample"
    assert restored.fonts[0].sha256 == FONT_ASSET.sha256
    assert restored.font_size == 18.0
    assert restored.font_family_name == "Noto Sans"
    assert restored.font_weight == "bold"
    assert restored.font_style == "italic"
    assert restored.fill == Color.blue()
    assert restored.fill_opacity == 0.5
    assert restored.x == 50.0
    assert restored.y == 100.0
    assert restored.dx == 5.0
    assert restored.dy == 2.0
    assert restored.text_anchor == TextAnchor.END
    assert restored.direction == "rtl"
    assert restored.is_font_substituted is True


def test_text_with_runs_concatenation_and_properties():
    r1 = TextRun("Hello ", fill=Color.red())
    r2 = TextRun("World", fill=Color.blue())
    t = Text(
        runs=[r1, r2],
        position=Point(100, 200),
        text_origin="baseline",
        text_anchor=TextAnchor.START,
        font_family_name="Noto Sans",
        fonts=(FONT_ASSET,),
        font_size=28.0,
    )
    # Automatic concatenation
    assert t.text == "Hello World"
    assert t.runs == (r1, r2)
    assert t.text_origin == "baseline"
    assert t.text_anchor == TextAnchor.START
    assert t.font_family_name == "Noto Sans"
    assert t.font_family == FontFamily.SIMPLEX  # Backward-compatible Hershey enum untouched!


def test_text_with_runs_serialization_roundtrip():
    r1 = TextRun("Part 1 ", fill=Color.green())
    r2 = TextRun("Part 2", fill=Color.yellow(), dx=3.0)
    t = Text(
        runs=[r1, r2],
        position=Point(40, 80),
        text_origin="baseline",
        text_anchor="middle",
        font_family_name="Noto Sans",
        fonts=(FONT_ASSET,),
        font_size=20.0,
        fill_opacity=0.9,
    )
    d = t.to_dict()
    assert "runs" in d
    assert len(d["runs"]) == 2
    assert d["text_origin"] == "baseline"
    assert d["text_anchor"] == "middle"
    assert d["font_family_name"] == "Noto Sans"
    assert d["fill_opacity"] == 0.9

    restored = Text.from_dict(d)
    assert restored.text == "Part 1 Part 2"
    assert len(restored.runs) == 2
    assert restored.runs[0].text == "Part 1 "
    assert restored.runs[1].text == "Part 2"
    assert restored.runs[1].dx == 3.0
    assert restored.text_origin == "baseline"
    assert restored.text_anchor == TextAnchor.MIDDLE
    assert restored.font_family_name == "Noto Sans"
    assert restored.fill_opacity == 0.9


def test_legacy_text_deserialization_defaults():
    assert CURRENT_SCHEMA_VERSION == "1.10"
    legacy_doc = {
        "format": "drawcv",
        "version": "1.8",
        "scene": {
            "width": 800,
            "height": 600,
            "background": {"r": 255, "g": 255, "b": 255, "a": 1.0},
            "layers": [
                {
                    "name": "Layer 1",
                    "visible": True,
                    "locked": False,
                    "opacity": 1.0,
                    "objects": [
                        {
                            "type": "text",
                            "text": "Legacy simple text",
                            "position": {"x": 10.0, "y": 20.0},
                            "color": {"r": 0, "g": 0, "b": 0, "a": 1.0},
                            "font_family": "simplex",
                            "font_scale": 1.0,
                            "thickness": 1,
                            "alignment": "left",
                            "background_fill": None,
                            "background_radius": 0.0,
                            "padding": 0.0,
                        }
                    ],
                }
            ],
        },
    }
    # 1. 1.8 -> 1.9 schema forward migration
    migrated = SchemaMigrator.migrate(legacy_doc)
    assert migrated["version"] == CURRENT_SCHEMA_VERSION
    patch_obj = migrated["scene"]["layers"][0]["objects"][0]
    assert patch_obj["text_origin"] == "top_left"
    assert patch_obj["fill_none"] is False

    # 2. Verify 1.8 legacy text deserializes in current reader
    scene_loaded = Scene.from_dict(legacy_doc)
    assert scene_loaded.to_dict()["version"] == CURRENT_SCHEMA_VERSION
    t = scene_loaded.layers[0].objects[0]
    assert t.text == "Legacy simple text"
    assert t.runs is None
    assert t.text_origin == "top_left"
    assert t.text_anchor is None
    assert t.font_family == FontFamily.SIMPLEX
    assert t.fill_none is False

    # 3. 1.9 rich Text -> JSON -> current reader
    rich_t = Text(
        runs=(TextRun("Hello "), TextRun("World", fill=Color(255, 0, 0))),
        position=Point(10, 20),
        text_origin="baseline",
    )
    rich_scene = Scene(300, 100)
    rich_scene.add(rich_t)
    doc_1_9 = rich_scene.to_dict()
    assert doc_1_9["version"] == CURRENT_SCHEMA_VERSION
    json_str = json.dumps(doc_1_9)
    restored = Scene.from_dict(json.loads(json_str))
    assert restored.to_dict()["version"] == CURRENT_SCHEMA_VERSION
    r_text = restored.layers[0].objects[0]
    assert r_text.runs is not None
    assert len(r_text.runs) == 2
    assert r_text.runs[1].text == "World"

    # 4. Downgrade rejection: 1.9 document cannot be downgraded to 1.8
    from drawcv.core.exceptions import UnsupportedVersionError
    with pytest.raises(UnsupportedVersionError):
        SchemaMigrator.migrate(doc_1_9, target_version="1.8")
