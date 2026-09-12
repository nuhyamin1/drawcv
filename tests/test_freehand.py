"""Comprehensive unit tests for Freehand Engine (Phase 5)."""

from dataclasses import FrozenInstanceError
import math
import pytest

from drawcv.canvas import Canvas
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point, StrokePoint
from drawcv.core.path_processing import (
    catmull_rom_spline,
    chaikin_smooth,
    compute_path_length,
    interpolate_metadata,
    interpolate_stroke_point,
    rdp_simplify,
)
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.shapes.freehand import FreehandStroke
from drawcv.styles.stroke import StrokeStyle


# =============================================================================
# 1. StrokePoint Semantics & Immutability
# =============================================================================

def test_strokepoint_creation_and_immutability():
    sp = StrokePoint(10.0, 20.0, pressure=0.75, timestamp=100.0, velocity=50.0)
    assert sp.x == 10.0
    assert sp.y == 20.0
    assert sp.pressure == 0.75
    assert sp.timestamp == 100.0
    assert sp.velocity == 50.0

    # Test frozen immutability
    with pytest.raises(FrozenInstanceError):
        sp.x = 30.0  # type: ignore

    with pytest.raises(FrozenInstanceError):
        sp.pressure = 0.5  # type: ignore

    # translate returns a new instance, leaving original intact
    sp2 = sp.translate(5.0, -10.0)
    assert sp2.x == 15.0
    assert sp2.y == 10.0
    assert sp2.pressure == 0.75
    assert sp2.timestamp == 100.0
    assert sp2.velocity == 50.0
    assert sp.x == 10.0
    assert sp.y == 20.0


def test_strokepoint_validation():
    # Valid pressures [0.0, 1.0]
    StrokePoint(0, 0, pressure=0.0)
    StrokePoint(0, 0, pressure=1.0)
    StrokePoint(0, 0, pressure=0.5)

    # Invalid pressures
    with pytest.raises(ValidationError):
        StrokePoint(0, 0, pressure=-0.1)

    with pytest.raises(ValidationError):
        StrokePoint(0, 0, pressure=1.01)

    with pytest.raises(ValidationError):
        StrokePoint(0, 0, pressure="high")  # type: ignore

    # Invalid velocity
    with pytest.raises(ValidationError):
        StrokePoint(0, 0, velocity=-5.0)

    # Non-numeric or non-finite coords
    with pytest.raises(ValidationError):
        StrokePoint(float("nan"), 0)

    with pytest.raises(ValidationError):
        StrokePoint(0, float("inf"))


def test_strokepoint_methods_and_conversion():
    sp = StrokePoint(3.0, 4.0, pressure=0.8)
    assert sp.to_tuple() == (3.0, 4.0)
    assert sp.to_point() == Point(3.0, 4.0)
    assert sp.distance_to(Point(0.0, 0.0)) == pytest.approx(5.0)

    # from_point constructor
    p = Point(12.0, 34.0)
    sp_from_p = StrokePoint.from_point(p, pressure=0.9, timestamp=123.0)
    assert sp_from_p.x == 12.0
    assert sp_from_p.y == 34.0
    assert sp_from_p.pressure == 0.9
    assert sp_from_p.timestamp == 123.0


# =============================================================================
# 2. Metadata Interpolation & Math Primitives
# =============================================================================

def test_interpolate_metadata_rules():
    # Both present: linear interpolation
    assert interpolate_metadata(0.2, 0.8, 0.5) == pytest.approx(0.5)
    assert interpolate_metadata(0.0, 1.0, 0.25) == pytest.approx(0.25)

    # Clamping
    assert interpolate_metadata(-0.5, 1.5, 0.5, clamp_range=(0.0, 1.0)) == pytest.approx(0.5)

    # Deterministic None propagation
    assert interpolate_metadata(0.4, None, 0.5) == pytest.approx(0.4)
    assert interpolate_metadata(None, 0.6, 0.5) == pytest.approx(0.6)
    assert interpolate_metadata(None, None, 0.5) is None


def test_interpolate_stroke_point():
    p1 = StrokePoint(0.0, 0.0, pressure=0.2, timestamp=10.0, velocity=100.0)
    p2 = StrokePoint(100.0, 200.0, pressure=0.8, timestamp=20.0, velocity=200.0)
    mid = interpolate_stroke_point(p1, p2, 0.5)

    assert mid.x == pytest.approx(50.0)
    assert mid.y == pytest.approx(100.0)
    assert mid.pressure == pytest.approx(0.5)
    assert mid.timestamp == pytest.approx(15.0)
    assert mid.velocity == pytest.approx(150.0)


# =============================================================================
# 3. Modular Path Processing Algorithms
# =============================================================================

def test_rdp_simplification():
    # Points along a straight line: should simplify to just 2 endpoints
    line_pts = [StrokePoint(x, x * 2.0, pressure=x / 10.0) for x in range(11)]
    simplified = rdp_simplify(line_pts, epsilon=0.5)
    assert len(simplified) == 2
    assert simplified[0].x == 0.0
    assert simplified[-1].x == 10.0
    assert simplified[0].pressure == 0.0
    assert simplified[-1].pressure == 1.0

    # Point with a spike should be preserved
    spiky_pts = [
        StrokePoint(0, 0),
        StrokePoint(50, 0),
        StrokePoint(50, 100),  # acute spike
        StrokePoint(50, 0),
        StrokePoint(100, 0),
    ]
    simp_spiky = rdp_simplify(spiky_pts, epsilon=1.0)
    assert any(p.y == 100 for p in simp_spiky)

    # Invalid epsilon
    with pytest.raises(ValidationError):
        rdp_simplify(line_pts, epsilon=-1.0)


def test_chaikin_smooth():
    # Triangle wave
    pts = [
        StrokePoint(0, 0, pressure=0.1),
        StrokePoint(50, 100, pressure=0.5),
        StrokePoint(100, 0, pressure=0.9),
    ]
    smoothed = chaikin_smooth(pts, iterations=1, ratio=0.25)
    # 3 points -> 1 + 2*(len-1) + 1 = 6 points
    assert len(smoothed) == 6
    # Open endpoints are strictly preserved
    assert smoothed[0].x == 0.0
    assert smoothed[0].pressure == 0.1
    assert smoothed[-1].x == 100.0
    assert smoothed[-1].pressure == 0.9

    # Validation
    with pytest.raises(ValidationError):
        chaikin_smooth(pts, iterations=-1)
    with pytest.raises(ValidationError):
        chaikin_smooth(pts, ratio=0.0)
    with pytest.raises(ValidationError):
        chaikin_smooth(pts, ratio=0.5)


def test_catmull_rom_spline():
    ctrl_pts = [
        StrokePoint(0, 0, pressure=0.2, timestamp=1.0),
        StrokePoint(50, 50, pressure=0.4, timestamp=2.0),
        StrokePoint(100, 0, pressure=0.8, timestamp=3.0),
        StrokePoint(150, 50, pressure=1.0, timestamp=4.0),
    ]
    spline = catmull_rom_spline(ctrl_pts, samples_per_segment=4)

    # Spline passes exactly through all interior control points
    # Control points should be at index 0, 4, 8, 12
    assert spline[0].x == pytest.approx(0.0)
    assert spline[0].y == pytest.approx(0.0)
    assert spline[0].pressure == pytest.approx(0.2)

    assert spline[4].x == pytest.approx(50.0)
    assert spline[4].y == pytest.approx(50.0)
    assert spline[4].pressure == pytest.approx(0.4)

    assert spline[8].x == pytest.approx(100.0)
    assert spline[8].y == pytest.approx(0.0)
    assert spline[8].pressure == pytest.approx(0.8)

    assert spline[12].x == pytest.approx(150.0)
    assert spline[12].y == pytest.approx(50.0)
    assert spline[12].pressure == pytest.approx(1.0)

    # Timestamps are strictly monotonic
    for i in range(len(spline) - 1):
        assert spline[i + 1].timestamp >= spline[i].timestamp
        # Pressures never overshoot [0, 1]
        assert 0.0 <= spline[i].pressure <= 1.0


# =============================================================================
# 4. FreehandStroke Non-Destructive Source Retention & add_point
# =============================================================================

def test_freehand_non_destructive_retention():
    raw_points = [
        StrokePoint(0, 0, pressure=0.1),
        StrokePoint(50, 20, pressure=0.5),
        StrokePoint(100, 0, pressure=0.9),
    ]
    stroke = FreehandStroke(
        points=raw_points,
        stroke=StrokeStyle(color=Color.blue(), width=6.0),
        smoothing="chaikin",
        smoothing_iterations=2,
        interpolation="catmull_rom",
        interpolation_samples=6,
    )

    # Original points retain exact length and values
    assert len(stroke.points) == 3
    assert stroke.points[0] == raw_points[0]
    assert stroke.points[1] == raw_points[1]
    assert stroke.points[2] == raw_points[2]

    # Processed points produce refined geometry without modifying source points
    proc = stroke.get_processed_points()
    assert len(proc) > len(stroke.points)
    assert len(stroke.points) == 3

    # Rendering does not mutate source points
    renderer = OpenCVRenderer()
    canvas = Canvas(200, 200)
    renderer._render_freehand(stroke, canvas)
    assert len(stroke.points) == 3


def test_add_point_and_velocity_auto_calculation():
    stroke = FreehandStroke(stroke=StrokeStyle(width=4.0))

    # Add first point at t=0
    p1 = stroke.add_point((0, 0), pressure=0.2, timestamp=0.0)
    assert p1.velocity is None

    # Add second point at t=1.0, distance = 100 -> velocity = 100 px/s
    p2 = stroke.add_point((100, 0), pressure=0.5, timestamp=1.0)
    assert p2.velocity == pytest.approx(100.0)

    # Add third point with dt=0 -> protected from zero division, yields v=0.0
    p3 = stroke.add_point((100, 10), pressure=0.6, timestamp=1.0)
    assert p3.velocity == 0.0

    # Non-monotonic timestamp raises ValidationError
    with pytest.raises(ValidationError):
        stroke.add_point((110, 10), timestamp=0.5)

    # Explicit caller-provided velocity is preserved
    p4 = stroke.add_point((150, 0), timestamp=2.0, velocity=999.0)
    assert p4.velocity == 999.0


# =============================================================================
# 5. Parameter Validation Suite
# =============================================================================

def test_freehand_parameter_validation():
    # Invalid stroke width / min_width / max_width
    with pytest.raises(ValidationError):
        FreehandStroke(min_width=-1.0)

    with pytest.raises(ValidationError):
        FreehandStroke(min_width=10.0, max_width=5.0)

    # Invalid velocity bounds
    with pytest.raises(ValidationError):
        FreehandStroke(velocity_min=-1.0)

    with pytest.raises(ValidationError):
        FreehandStroke(velocity_min=100.0, velocity_max=50.0)

    # Invalid algorithm names
    with pytest.raises(ValidationError):
        FreehandStroke(smoothing="spline")  # supported: "chaikin"

    with pytest.raises(ValidationError):
        FreehandStroke(simplification="pca")  # supported: "rdp"

    with pytest.raises(ValidationError):
        FreehandStroke(interpolation="bezier")  # supported: "catmull_rom"

    with pytest.raises(ValidationError):
        FreehandStroke(width_mode="random")  # supported: "constant", "pressure", "velocity"


# =============================================================================
# 6. Authoritative Processing Pipeline Order
# =============================================================================

def test_authoritative_pipeline_order():
    # Points along a triangle wave with small collinear noise (noise < 1.0 px)
    pts = [
        StrokePoint(0, 0),
        StrokePoint(25, 50.1),  # redundant noise on segment (0,0)-(50,100)
        StrokePoint(50, 100),   # salient corner
        StrokePoint(75, 50.1),  # redundant noise on segment (50,100)-(100,0)
        StrokePoint(100, 0),
    ]

    # Without RDP: 5 points -> Chaikin (10 pts) -> Catmull-Rom (9*4+1 = 37 pts)
    stroke_no_rdp = FreehandStroke(
        points=pts,
        simplification=None,
        smoothing="chaikin",
        smoothing_iterations=1,
        interpolation="catmull_rom",
        interpolation_samples=4,
    )
    assert len(stroke_no_rdp.get_processed_points()) == 37

    # With RDP:
    # RDP simplifies 5 points to 3 points (0,0), (50,100), (100,0)
    # Then Chaikin smooths 3 points into 6 points
    # Then Catmull-Rom interpolates between 6 points (5 segments * 4 = 20 + 1 = 21 points)
    stroke = FreehandStroke(
        points=pts,
        simplification="rdp",
        simplification_tolerance=1.0,
        smoothing="chaikin",
        smoothing_iterations=1,
        interpolation="catmull_rom",
        interpolation_samples=4,
    )
    processed = stroke.get_processed_points()
    assert len(processed) == 21
    # Endpoints remain at x=0 and x=100
    assert processed[0].x == pytest.approx(0.0)
    assert processed[-1].x == pytest.approx(100.0)


# =============================================================================
# 7. Variable Width & Visual Bounds Accounting
# =============================================================================

def test_variable_width_and_bounds():
    pts = [
        StrokePoint(100, 100, pressure=0.0),
        StrokePoint(200, 100, pressure=1.0),
    ]
    # base stroke width 10.0, min_width=4.0, max_width=20.0
    stroke = FreehandStroke(
        points=pts,
        stroke=StrokeStyle(color=Color.black(), width=10.0),
        variable_width=True,
        width_mode="pressure",
        min_width=4.0,
        max_width=20.0,
    )

    widths = stroke.get_point_widths()
    assert widths[0] == pytest.approx(4.0)
    assert widths[1] == pytest.approx(20.0)

    # Intrinsic centerline bounds: [100, 100, 100, 0]
    geom_box = stroke.get_geometry_bounds()
    assert geom_box.left == 100.0
    assert geom_box.right == 200.0
    assert geom_box.top == 100.0
    assert geom_box.bottom == 100.0

    # Local bounds must expand by max(w_i)/2 = 20.0/2 = 10.0!
    local_box = stroke.get_local_bounds()
    assert local_box.left == pytest.approx(90.0)
    assert local_box.right == pytest.approx(210.0)
    assert local_box.top == pytest.approx(90.0)
    assert local_box.bottom == pytest.approx(110.0)


def test_velocity_width_mapping():
    pts = [
        StrokePoint(0, 0, velocity=0.0),
        StrokePoint(100, 0, velocity=500.0),
        StrokePoint(200, 0, velocity=1000.0),
    ]
    stroke = FreehandStroke(
        points=pts,
        variable_width=True,
        width_mode="velocity",
        min_width=2.0,
        max_width=12.0,
        velocity_min=0.0,
        velocity_max=1000.0,
    )
    widths = stroke.get_point_widths()
    # Faster velocity produces thinner lines:
    # at v=0: w=max=12.0
    # at v=500: w=mid=7.0
    # at v=1000: w=min=2.0
    assert widths[0] == pytest.approx(12.0)
    assert widths[1] == pytest.approx(7.0)
    assert widths[2] == pytest.approx(2.0)


# =============================================================================
# 8. Hit Testing & Anchors
# =============================================================================

def test_hit_testing_and_anchors():
    pts = [StrokePoint(0, 100), StrokePoint(200, 100)]
    stroke = FreehandStroke(
        points=pts,
        stroke=StrokeStyle(width=10.0),
    )

    # Point directly on stroke line
    assert stroke.contains_point(100, 100) is True
    # Point within stroke thickness (half-width = 5 + tolerance 4 = 9)
    assert stroke.contains_point(100, 108) is True
    # Point far away
    assert stroke.contains_point(100, 150) is False

    # Anchors
    assert stroke.anchor("start") == Point(0, 100)
    assert stroke.anchor("end") == Point(200, 100)
    assert stroke.anchor("midpoint") == Point(100, 100)


# =============================================================================
# 9. Scene, Layer, Group & Transform Integration
# =============================================================================

def test_freehand_in_scene_group_hierarchy():
    scene = Scene(width=800, height=600)
    layer = scene.create_layer("ink_layer")

    stroke = FreehandStroke(
        id="signature",
        points=[StrokePoint(50, 50), StrokePoint(150, 50)],
        stroke=StrokeStyle(color=Color.red(), width=8.0),
    )
    layer.add(stroke)

    assert scene.get("signature") is stroke
    assert stroke.scene is scene
    assert stroke.layer is layer

    # Group integration & hierarchical transforms
    group = Group(id="ink_group")
    layer.add(group)
    group.add(stroke)

    assert stroke.parent is group
    # Rotate group 90 degrees around (100, 50)
    group.rotate(90.0, pivot=Point(100, 50))

    # World bounds should reflect 90-degree rotation (horizontal line becomes vertical)
    w_bounds = stroke.get_bounds()
    assert w_bounds.center.x == pytest.approx(100.0, abs=1.0)
    assert w_bounds.center.y == pytest.approx(50.0, abs=1.0)
    assert w_bounds.height > w_bounds.width


# =============================================================================
# 10. Rasterization & Renderer Smoke Tests
# =============================================================================

def test_renderer_constant_and_variable_width():
    scene = Scene(width=400, height=400, background=Color.white())
    renderer = OpenCVRenderer()

    # 1. Constant width stroke
    s1 = FreehandStroke(
        points=[StrokePoint(50, 50), StrokePoint(150, 150)],
        stroke=StrokeStyle(color=Color.black(), width=4.0),
    )
    scene.add(s1)

    # 2. Variable width stroke with Chaikin & Catmull-Rom
    s2 = FreehandStroke(
        points=[
            StrokePoint(50, 200, pressure=0.1),
            StrokePoint(100, 250, pressure=1.0),
            StrokePoint(150, 200, pressure=0.2),
        ],
        stroke=StrokeStyle(color=Color.blue(), width=6.0),
        smoothing="chaikin",
        interpolation="catmull_rom",
        variable_width=True,
        width_mode="pressure",
        min_width=2.0,
        max_width=16.0,
    )
    scene.add(s2)

    # 3. Translucent stroke
    s3 = FreehandStroke(
        points=[StrokePoint(200, 50), StrokePoint(200, 300)],
        stroke=StrokeStyle(color=Color(255, 0, 0, a=0.5), width=10.0),
        variable_width=True,
    )
    scene.add(s3)

    canvas = renderer.render(scene)
    assert canvas.width == 400
    assert canvas.height == 400
    # Check that canvas has non-white pixels (shapes were drawn)
    assert not (canvas.buffer == 255).all()
