"""Visible stroke coverage and retained-state regressions for milestone 1."""
import json
import numpy as np
import pytest

from drawcv import (OpenCVRenderer, Scene, Line, Polyline, Path, Point, Color, StrokeStyle,
                    CapStyle, JoinStyle, LineType, Group, Transform,
                    FreehandStroke, StrokePoint, Circle, FillStyle)
from drawcv.core.exceptions import ValidationError
from drawcv.shapes.path import Subpath


def render(obj, width=180, height=140):
    scene = Scene(width, height, background=Color.white())
    scene.add(obj)
    return OpenCVRenderer().render(scene).buffer


def line(style):
    return Line(start=Point(30, 50), end=Point(130, 50), stroke=style)


@pytest.mark.parametrize("line_type", list(LineType))
def test_caps_have_distinct_visible_coverage(line_type):
    frames = {cap: render(line(StrokeStyle(width=20, cap_style=cap, line_type=line_type)))
              for cap in CapStyle}
    assert frames[CapStyle.BUTT][50, 23, 0] == 255
    assert frames[CapStyle.ROUND][50, 23, 0] < 20
    assert frames[CapStyle.SQUARE][50, 23, 0] < 20
    assert frames[CapStyle.ROUND][42, 22, 0] > 200
    assert frames[CapStyle.SQUARE][42, 22, 0] < 20


def test_joins_and_miter_limit_change_corner_pixels():
    points = [Point(30, 90), Point(80, 90), Point(80, 30)]
    def frame(join, limit=4):
        return render(Polyline(points=points, stroke=StrokeStyle(
            width=30, join_style=join, miter_limit=limit)))
    miter, bevel, rounded = (frame(j) for j in (JoinStyle.MITER, JoinStyle.BEVEL, JoinStyle.ROUND))
    assert miter[103, 93, 0] < 20
    assert bevel[103, 93, 0] > 200
    assert rounded[100, 90, 0] < 100
    assert bevel[100, 90, 0] > 200
    assert np.array_equal(frame(JoinStyle.MITER, 1), bevel)


def test_dashes_continue_across_vertices_and_offset_wraps():
    points = [Point(20, 40), Point(25, 40), Point(120, 40)]
    style = StrokeStyle(width=6, cap_style=CapStyle.BUTT, dash_array=(12, 8))
    image = render(Polyline(points=points, stroke=style))
    assert image[40, 30, 0] == 0
    assert image[40, 36, 0] == 255
    assert image[40, 44, 0] == 0
    style.dash_offset = -5
    negative = render(Polyline(points=points, stroke=style))
    style.dash_offset = 15
    assert np.array_equal(negative, render(Polyline(points=points, stroke=style)))
    assert negative[40, 22, 0] == 255
    assert negative[40, 29, 0] == 0


def test_odd_pattern_repeats_and_input_list_is_copied():
    values = [10, 5, 3]
    style = StrokeStyle(width=4, dash_array=values, cap_style=CapStyle.BUTT)
    values[0] = -1
    assert style.dash_array == (10, 5, 3)
    a = render(line(style))
    style.dash_array = (10, 5, 3, 10, 5, 3)
    assert np.array_equal(a, render(line(style)))


def test_closed_caps_are_ignored_and_dash_seam_is_joined():
    points = [Point(40, 40), Point(100, 40), Point(100, 100), Point(40, 100)]
    def frame(cap, dash=()):
        return render(Polyline(points=points, closed=True, stroke=StrokeStyle(
            width=16, cap_style=cap, join_style=JoinStyle.MITER,
            dash_array=dash, dash_offset=10)))
    assert np.array_equal(frame(CapStyle.BUTT), frame(CapStyle.SQUARE))
    # Perimeter=240, pattern=40: start and end fall inside one on-run.
    dashed = frame(CapStyle.BUTT, (30, 10))
    assert dashed[34, 34, 0] == 0
    assert dashed[40, 65, 0] == 255


def test_compound_paths_restart_phase_and_skip_empty_subpaths():
    style = StrokeStyle(width=8, dash_array=(10, 10), cap_style=CapStyle.BUTT)
    path = Path(stroke=style)
    path.move_to(20, 30).line_to(120, 30)
    path.move_to(20, 70).line_to(120, 70)
    path.subpaths.insert(0, Subpath(closed=True))
    image = render(path)
    assert np.array_equal(image[30, 15:125], image[70, 15:125])
    assert image[30, 35, 0] == 255
    assert image[30, 65, 0] == 0


def test_compound_strokes_composite_once_at_crossings():
    path = Path(stroke=StrokeStyle(width=14, opacity=.5))
    path.move_to(20, 50).line_to(130, 50)
    path.move_to(70, 20).line_to(70, 100)
    image = render(path)
    assert image[50, 70, 0] == image[50, 40, 0] == 128


def test_world_pixel_dash_lengths_and_width_under_parent_transform():
    obj = Line(start=Point(0, 0), end=Point(40, 0), stroke=StrokeStyle(
        width=6, dash_array=(10, 10), cap_style=CapStyle.BUTT))
    group = Group(children=[obj], transform=Transform(
        scale_x=3, scale_y=2, translation_x=20, translation_y=50, pivot=Point(0, 0)))
    image = render(group)
    assert image[50, 25, 0] == 0
    assert image[50, 35, 0] == 255
    assert image[50, 45, 0] == 0
    assert image[56, 25, 0] == 255


def test_progressive_dashes_preserve_pivot_parent_and_source():
    obj = line(StrokeStyle(width=6, dash_array=(12, 8), cap_style=CapStyle.BUTT))
    obj.transform.rotation = 90
    group = Group(children=[obj], opacity=.5, transform=Transform(translation_x=20))
    scene = Scene(180, 140, background=Color.white())
    scene.add(group)
    full = OpenCVRenderer().render(scene).buffer.copy()
    original = obj.to_dict()
    obj.progress = .5
    partial = OpenCVRenderer().render(scene).buffer
    # Default pivot stays at the full line's center: first half runs down x=100.
    assert np.array_equal(full[5:45, 95:106], partial[5:45, 95:106])
    assert partial[5, 100, 0] == 128
    assert partial[65, 100, 0] == 255
    obj.progress = 1
    assert obj.to_dict() == original
    assert obj.parent is group


def test_variable_width_dashes_interpolate_width_and_caps():
    obj = FreehandStroke(points=[StrokePoint(20, 60, pressure=0), StrokePoint(140, 60, pressure=1)],
        stroke=StrokeStyle(width=10, dash_array=(20, 20), cap_style=CapStyle.BUTT),
        variable_width=True, width_mode="pressure", min_width=4, max_width=24)
    image = render(obj)
    assert image[60, 50, 0] == 255
    assert image[67, 110, 0] == 0
    assert image[67, 30, 0] == 255


def test_miter_not_clipped_by_isolated_group_and_bounds():
    obj = Polyline(points=[Point(40, 110), Point(80, 30), Point(120, 110)],
        stroke=StrokeStyle(width=20, join_style=JoinStyle.MITER))
    plain = render(obj)
    obj2 = obj.clone()
    isolated = render(Group(children=[obj2], opacity=.5))
    ys, xs = np.where(plain[:, :, 0] < 20)
    assert ys.min() < 15
    assert np.all(isolated[ys, xs, 0] < 150)
    bounds = obj.get_bounds()
    assert bounds.top <= ys.min() and bounds.left <= xs.min()


def test_degenerate_line_caps_and_duplicate_vertices():
    for cap in CapStyle:
        style = StrokeStyle(width=16, cap_style=cap)
        image = render(Line(start=Point(50, 50), end=Point(50, 50), stroke=style))
        assert (image[50, 50, 0] == 255) == (cap == CapStyle.BUTT)
    style = StrokeStyle(width=10, dash_array=(10, 10))
    a = render(Polyline(points=[Point(20, 40), Point(80, 40)], stroke=style))
    b = render(Polyline(points=[Point(20, 40), Point(20, 40), Point(80, 40)], stroke=style))
    assert np.array_equal(a, b)


def test_style_roundtrip_clone_history_and_numeric_animation():
    scene = Scene(180, 140, background=Color.white())
    obj = line(StrokeStyle(width=8, dash_array=(12, 8), cap_style=CapStyle.SQUARE,
                           join_style=JoinStyle.MITER, miter_limit=3))
    scene.add(obj)
    before = OpenCVRenderer().render(scene).buffer.copy()
    with scene.edit(obj):
        obj.stroke.dash_offset = 7
        obj.stroke.dash_array = (7, 13)
        obj.stroke.miter_limit = 6
    after = OpenCVRenderer().render(scene).buffer.copy()
    assert not np.array_equal(before, after)
    assert scene.undo() and scene.get(obj.id) is obj
    assert np.array_equal(before, OpenCVRenderer().render(scene).buffer)
    assert scene.redo() and np.array_equal(after, OpenCVRenderer().render(scene).buffer)
    clone = obj.clone()
    assert clone.stroke == obj.stroke and clone.stroke is not obj.stroke
    scene.animate(obj, "stroke.dash_offset", 0, 20, duration=2)
    scene.animate(obj, "stroke.miter_limit", 2, 6, duration=2)
    saved = scene.to_dict()
    loaded = Scene.from_dict(json.loads(json.dumps(saved)))
    assert saved["version"] == "1.3"
    assert np.array_equal(scene.render_at_time(.5).buffer, loaded.render_at_time(.5).buffer)
    assert np.array_equal(scene.render_at_time(.5).buffer, scene.render_at_time(.5).buffer)
    assert scene.to_dict() == saved
    assert scene.get(obj.id) is obj


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_legacy_style_defaults(version):
    scene = Scene(100, 100)
    obj = line(StrokeStyle())
    scene.add(obj)
    data = scene.to_dict()
    data["version"] = version
    for layer in data["scene"]["layers"]:
        for shape in layer["objects"]:
            for key in ("dash_array", "dash_offset", "miter_limit"):
                shape["stroke"].pop(key)
    loaded = Scene.from_dict(data)
    assert loaded.get(obj.id).stroke == StrokeStyle()


@pytest.mark.parametrize("field,value", [
    ("width", -1), ("width", float("nan")), ("opacity", 2), ("color", "red"),
    ("cap_style", "butt"), ("join_style", None), ("line_type", "AA"),
    ("dash_array", (4, -1)), ("dash_array", [0]), ("dash_array", [True]),
    ("dash_array", [float("inf")]), ("dash_array", "2,3"),
    ("dash_offset", float("nan")), ("dash_offset", True), ("miter_limit", .5),
])
def test_rejected_style_assignment_preserves_valid_state(field, value):
    style = StrokeStyle()
    before = style.to_dict()
    with pytest.raises(ValidationError):
        setattr(style, field, value)
    assert style.to_dict() == before


@pytest.mark.parametrize("field,value", [("enabled", 1), ("color", None), ("opacity", -1)])
def test_rejected_fill_assignment_is_atomic(field, value):
    fill = FillStyle()
    before = fill.to_dict()
    with pytest.raises(ValidationError):
        setattr(fill, field, value)
    assert fill.to_dict() == before
    assert isinstance(fill.enabled, bool)


def test_invalid_transform_preserves_authoritative_matrix():
    matrix = np.array([[1, .6, 20], [0, 1, 30], [0, 0, 1.]])
    transform = Transform.from_matrix(matrix)
    for field, value in [("scale_x", -1), ("pivot", "bad"), ("translation_y", float("nan"))]:
        before = transform.to_dict()
        with pytest.raises(ValidationError):
            setattr(transform, field, value)
        assert transform.to_dict() == before
        assert np.array_equal(transform.get_matrix(), matrix)


@pytest.mark.parametrize("operation", [
    lambda obj: obj.move(10, float("nan")),
    lambda obj: obj.rotate(30, pivot="bad"),
    lambda obj: obj.scale(2, float("nan")),
    lambda obj: obj.scale(2, 3, pivot="bad"),
])
def test_failed_compound_transform_restores_object(operation):
    obj = line(StrokeStyle())
    transform = obj.transform
    before = obj.to_dict()
    with pytest.raises(ValidationError):
        operation(obj)
    assert obj.transform is transform
    assert obj.to_dict() == before


@pytest.mark.parametrize("invalid_radius", [-1, True, "bad"])
def test_scene_edit_rejects_invalid_geometry_and_restores_all_targets(invalid_radius):
    scene = Scene(100, 100)
    first, second = Circle(radius=10), Circle(radius=20)
    group = Group(children=[first, second])
    scene.add(group)
    before = scene.to_dict()
    count = scene.history.undo_count
    with pytest.raises(ValidationError):
        with scene.edit(group):
            first.radius = 30
            second.radius = invalid_radius
    assert scene.to_dict() == before
    assert scene.get(first.id) is first and scene.get(second.id) is second
    assert scene.history.undo_count == count


def test_nested_failed_batch_restores_state_and_keeps_redo():
    scene = Scene(100, 100)
    obj = Circle(radius=10)
    scene.add(obj)
    with scene.edit(obj):
        obj.radius = 20
    scene.undo()
    before = scene.to_dict()
    counts = scene.history.undo_count, scene.history.redo_count
    with pytest.raises(ValidationError):
        with scene.batch():
            with scene.edit(obj):
                obj.radius = 30
            with scene.batch():
                scene.add(Circle(radius=5))
                raise ValidationError("abort")
    assert scene.to_dict() == before
    assert (scene.history.undo_count, scene.history.redo_count) == counts
    assert scene.redo() and obj.radius == 20


def test_failed_relative_placement_restores_first_move():
    from drawcv.positioning import place_above
    obj = Circle(center=Point(20, 20), radius=5)
    before = obj.to_dict()
    with pytest.raises(ValidationError):
        place_above(obj, Circle(center=Point(80, 80), radius=5), align="invalid")
    assert obj.to_dict() == before


def test_selection_overflow_rolls_back_earlier_members():
    from drawcv import Selection
    first, second = Circle(radius=2), Circle(radius=2)
    second.transform.translation_x = 1e308
    selection = Selection(Scene(100, 100), [first, second])
    before = [obj.to_dict() for obj in (first, second)]
    with pytest.raises(ValidationError):
        selection.move(1e308, 0)
    assert [obj.to_dict() for obj in (first, second)] == before


def test_failed_compound_redo_preserves_stack_and_completed_commands():
    from drawcv.history.command import Command, CompoundCommand
    from drawcv.history.manager import HistoryManager
    values = []
    class Append(Command):
        fail = False
        def execute(self):
            if self.fail:
                raise ValidationError("failure before mutation")
            values.append(self)
        def undo(self):
            assert values.pop() is self
    first, second = Append(), Append()
    command = CompoundCommand([first, second])
    history = HistoryManager()
    command.execute()
    history.record(command)
    history.undo()
    second.fail = True
    with pytest.raises(ValidationError):
        history.redo()
    assert values == []
    assert history.redo_count == 1 and history.undo_count == 0
    second.fail = False
    assert history.redo() and values == [first, second]


@pytest.mark.parametrize("shape", ["rectangle", "rounded", "circle", "ellipse", "polygon", "arc", "bezier", "arrow"])
def test_every_outline_renders_visible_dash_gaps(shape):
    from drawcv import Rectangle, RoundedRectangle, Ellipse, Polygon, Arc, BezierCurve, Arrow
    factories = {
        "rectangle": lambda s: Rectangle(position=Point(30, 30), width=90, height=60, stroke=s),
        "rounded": lambda s: RoundedRectangle(x=30, y=30, width=90, height=60, corner_radius=10, stroke=s),
        "circle": lambda s: Circle(center=Point(75, 65), radius=40, stroke=s),
        "ellipse": lambda s: Ellipse(center=Point(75, 65), radius_x=50, radius_y=30, stroke=s),
        "polygon": lambda s: Polygon(vertices=[Point(30, 90), Point(80, 30), Point(130, 90)], stroke=s),
        "arc": lambda s: Arc(center=Point(75, 65), radius_x=50, radius_y=30, sweep_angle=240, stroke=s),
        "bezier": lambda s: BezierCurve(p0=Point(20, 90), p1=Point(60, 10), p2=Point(130, 80), stroke=s),
        "arrow": lambda s: Arrow(start=Point(20, 60), end=Point(140, 60), stroke=s),
    }
    solid = render(factories[shape](StrokeStyle(width=5, cap_style=CapStyle.BUTT)))
    dashed = render(factories[shape](StrokeStyle(width=5, cap_style=CapStyle.BUTT, dash_array=(8, 12))))
    assert np.count_nonzero(dashed[:, :, 0] < 100) < .8 * np.count_nonzero(solid[:, :, 0] < 100)


def test_partial_closed_path_stays_open_until_completion():
    points = [Point(30, 30), Point(110, 30), Point(110, 110), Point(30, 110)]
    obj = Polyline(points=points, closed=True, stroke=StrokeStyle(
        width=8, cap_style=CapStyle.BUTT, dash_array=(12, 8)))
    obj.progress = .5
    partial = render(obj)
    assert partial[30, 35, 0] == 0
    assert partial[90, 30, 0] == 255
    obj.progress = 1
    complete = render(obj)
    assert complete[85, 30, 0] == 0


def test_sheared_circle_fill_and_stroke_follow_same_world_transform():
    obj = Circle(center=Point(0, 0), radius=20, fill=FillStyle(color=Color.red()),
                 stroke=StrokeStyle(width=4, dash_array=(8, 8)))
    group = Group(children=[obj], transform=Transform.from_matrix(
        np.array([[2, .5, 80], [0, 1, 60], [0, 0, 1.]])))
    image = render(group)
    assert tuple(image[60, 110]) == (0, 0, 255)
    assert tuple(image[90, 80]) == (255, 255, 255)


def test_zero_pressure_width_does_not_paint():
    obj = FreehandStroke(points=[StrokePoint(20, 60, pressure=0), StrokePoint(140, 60, pressure=0)],
        stroke=StrokeStyle(width=10, dash_array=(20, 20)), variable_width=True,
        width_mode="pressure", min_width=0, max_width=20)
    assert np.all(render(obj) == 255)


def test_atomic_positioning_keeps_keyword_calls_and_validation_errors():
    from drawcv import place_above, distribute_horizontally
    obj, reference = Circle(radius=5), Circle(center=Point(50, 50), radius=10)
    place_above(a=obj, b=reference, align="center")
    assert obj.get_bounds().bottom == reference.get_bounds().top - 20
    distribute_horizontally(objects=(obj, reference), spacing=10)
    assert reference.get_bounds().left == obj.get_bounds().right + 10
    with pytest.raises(ValidationError):
        place_above(a="bad", b=reference)

