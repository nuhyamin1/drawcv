# DrawCV

A clean, extensible, object-oriented 2D drawing engine and graphics library built on **OpenCV** and **NumPy** as the raster rendering backend.

Unlike immediate-mode OpenCV functions (`cv2.line`, `cv2.rectangle`, `cv2.circle`) that permanently alter pixels in-place, **DrawCV** operates on a **strict retained-mode graphics architecture**. Every drawn object remains addressable, editable, movable, transformable, hideable, reorderable, and re-renderable after creation.

```
Drawable Entities  ───►  Scene Graph  ───►  OpenCVRenderer  ───►  Canvas Buffer
   (Object Model)      (Authoritative)      (Centralized AA      (NumPy / OpenCV)
                                             & Compositing)
```

---

## Key Architectural Principles

1. **Retained-Mode Graphics**: The `Scene` is authoritative. The raster image is purely disposable and reproducible upon re-rendering. Mutating an object's properties in place and invoking `.render()` regenerates the scene without manual pixel clearing or damage rect calculations.
2. **Decoupled Primitives (Zero OpenCV in Shapes)**: `Line`, `Rectangle`, and `Circle` do not import or call `cv2`. All rasterization, OpenCV calls, coordinate quantization, and alpha compositing are centralized strictly inside `OpenCVRenderer`.
3. **Immutable Value Objects**: `Point` is an immutable `@dataclass(frozen=True)` with vector arithmetic. `Color` and `BoundingBox` enforce strict validation (no silent clamping of invalid channel or dimension values).
4. **Clean Geometry Ownership**: `Scene` owns logical dimensions (`width`, `height`) and `background`. `Canvas` owns the resulting BGR NumPy array.
5. **Exact Alpha Compositing**: Translucent strokes and fills are blended mathematically:
   $$\text{effective\_alpha} = \text{color.a} \times \text{style.opacity} \times \text{drawable.opacity}$$
   $$\text{dst} = \text{src} \cdot \alpha + \text{dst} \cdot (1 - \alpha)$$
6. **No Silent Unsupported Behavior**: Unimplemented transforms (e.g. arbitrary rotation of rectangles or non-uniform scaling) raise `RenderError` rather than silently failing to render.

---

## Installation

Requires Python 3.12+, NumPy, and OpenCV:

```bash
git clone https://github.com/your-username/DrawCV.git
cd DrawCV
pip install -e .
```

---

## Coordinate System

DrawCV uses standard computer graphics screen coordinates:

- **Origin `(0, 0)`**: Top-left corner of the canvas.
- **+X axis**: Extends horizontally to the right.
- **+Y axis**: Extends vertically downward.
- **Subpixel Precision**: Model geometry is stored as floating-point numbers (`Point(120.35, 440.81)`). The renderer is responsible for coordinate quantization and anti-aliasing during rasterization.

---

## Quickstart Example

```python
from drawcv import (
    Scene,
    OpenCVRenderer,
    Point,
    Color,
    Circle,
    Line,
    Rectangle,
    StrokeStyle,
    FillStyle,
)

# 1. Create a retained Scene
scene = Scene(width=1200, height=800, background=Color.white())

# 2. Add shapes
circle = Circle(
    center=Point(400, 300),
    radius=100,
    stroke=StrokeStyle(color=Color.black(), width=4.0),
    fill=FillStyle(color=Color.from_hex("#73B9FF"), opacity=0.5)
)

line = Line(
    start=Point(100, 100),
    end=Point(800, 500),
    stroke=StrokeStyle(color=Color.red(), width=5.0)
)

rectangle = Rectangle(
    position=Point(700, 150),
    width=220,
    height=150,
    stroke=StrokeStyle(color=Color.black(), width=3.0),
    fill=FillStyle(color=Color.green(), opacity=0.4)
)

scene.add(circle)
scene.add(line)
scene.add(rectangle)

# 3. Retained-mode in-place manipulation
circle.center = Point(500, 350)
circle.radius = 130
line.stroke.color = Color.blue()
rectangle.width = 280

# 4. Render to Canvas
renderer = OpenCVRenderer()
canvas = renderer.render(scene)

# 5. Export to disk
canvas.save("output.png")
```

---

## Core Components (Phase 1)

### Geometry & Color
- `Point(x, y)`: Immutable 2D vector supporting `+`, `-`, `*`, `/`, `.translate(dx, dy)`, `.distance_to(other)`.
- `Color(r, g, b, a=1.0)`: RGBA color model with `.from_hex()`, `.from_rgb()`, `.from_bgr()`, `.to_bgr()`, `.to_hex()`, `.black()`, `.white()`, `.red()`, etc.
- `BoundingBox(x, y, width, height)`: Spatial bounding box with `left`, `right`, `top`, `bottom`, `center`, `corners`, `.contains()`, `.intersects()`, `.union()`, `.expand()`.

### Styling
- `StrokeStyle(color, width, opacity, line_type)`: Anti-aliased stroke styling (`LineType.AA`, `LINE_8`, `LINE_4`).
- `FillStyle(enabled, color, opacity)`: Interior shape fill styling.

### Shapes
- `Line(start, end, stroke)`: Computes `length`, `angle`, `center`, and stroke-aware bounds.
- `Rectangle(position, width, height, stroke, fill)`: Computes `area`, `center`, `corners`, and stroke-aware bounds.
- `Circle(center, radius, stroke, fill)`: Computes `diameter`, `area`, `circumference`, and stroke-aware bounds.

### Advanced Geometry & Manipulation (Phases 2 & 3)
- **Transforms & Manipulation**: `.move(dx, dy)`, `.rotate(deg, pivot)`, `.scale(sx, sy, pivot)`, `.clone()`, exact geometric hit-testing.
- **Advanced Shapes**: `Ellipse`, `Arc` (OPEN, CHORD, PIE), `Polygon`, `Polyline`, `Arrow` (TRIANGLE, OPEN, DIAMOND, CIRCLE heads), `BezierCurve` (quadratic and cubic), `Path` (retained vector path with `MoveTo`, `LineTo`, `QuadraticTo`, `CubicTo`, `Close`), and `RoundedRectangle`.
- **Topological Fill Rules**: `FillRule.EVEN_ODD` and `FillRule.NON_ZERO` evaluation.

### Scene System (Phase 4)
- **Hierarchical Groups (`Group`)**: Container drawables supporting nested groups, cascaded transforms, non-destructive coordinate propagation, zero-shift world-transform reparenting, and collective world AABB computation.
- **Rendering Layers (`Layer`)**: Global rendering passes and stack organization (`z_order`), visibility, opacity, and cascading locked state.
- **Logical Selection (`Selection`)**: Multi-object selection controller with ancestor/descendant normalization (preventing double-transforms), rigid rotation around collective bounds center, scaling, batch property adjustments, and container-safe `.group()`.
- **Declarative Relative Positioning**: `align_left`, `align_right`, `align_top`, `align_bottom`, `align_center_x`, `align_center_y`, `align_centers`, `place_above`, `place_below`, `place_left_of`, `place_right_of` (with default `gap=20.0`), `distribute_horizontally`, `distribute_vertically`.
- **Authoritative Scene Registry & Query**: Fast $O(1)$ lookup via `scene.get(id)` across layers and groups, collision prevention, `find_by_name`, `find_by_tag`, `find_by_type`, `find_by_metadata`, and predicate searches.

### Freehand Engine (Phase 5)
- **`StrokePoint`**: Immutable frozen point holding coordinates with physical capture metadata (`pressure`, `timestamp`, `velocity`).
- **`FreehandStroke`**: Retained-mode freehand stroke retaining authoritative raw sampled input points without mutation.
- **Modular Path Processing Pipeline**:
  - **RDP Simplification (`rdp_simplify`)**: Noise and redundant point reduction while preserving salient corners and all metadata.
  - **Chaikin Smoothing (`chaikin_smooth`)**: Iterative corner-cutting curve refinement with open endpoint preservation.
  - **Catmull-Rom Spline (`catmull_rom_spline`)**: $C^1$-continuous cubic spline passing exactly through control points with centripetal parameterization ($\alpha=0.5$).
  - **Linear Metadata Interpolation**: Clamped linear interpolation for pressure $[0.0, 1.0]$ and timestamps preventing overshoot and preserving physical monotonicity.
- **Variable-Width Contour Rasterization**: Segment-normal tapered quadrilateral ribbons with circular joint/cap discs for stylus calligraphy (pressure-sensitive) and fountain pen flick dynamics (velocity-sensitive).

### Compositing & Effects (Phase 6)
- **Isolated Offscreen Surfaces**: Eliminates overlapping-geometry opacity accumulation by rasterizing translucent groups and layers to isolated premultiplied float32 BGRA buffers.
- **Single-Application Opacity Boundary**: Ancestor opacity is withheld during isolated subtree rendering and applied strictly once at the compositing boundary.
- **Premultiplied Alpha Pipeline**: Mathematically rigorous blending ($C_{\text{out}} = C_s + C_d(1 - \alpha_s)$, $\alpha_{\text{out}} = \alpha_s + \alpha_d(1 - \alpha_s)$) with 4-channel synchronous scaling across B, G, R, and A to prevent color fringing.
- **Effects**:
  - `BlurEffect`: Content blur (`BlurType.GAUSSIAN`, `BlurType.BOX`) executed on premultiplied intermediates without boundary halos.
  - `ShadowEffect`: Cast shadows with signed asymmetric offsets (`offset_x`, `offset_y`), blur radius, and shadow color, rendered behind base geometry.
  - **Effect-Inflated Buffers**: Asymmetric buffer expansion accommodating signed shadow offsets and blur penumbras.
- **Clipping & Masking**:
  - `ClipRect` & `ClipPath`: Vector stencils applied as the final post-effects clipping boundary.
  - `Mask`: Grayscale luminance/alpha masks with `MaskMapping.FIT_BOUNDS` mapped against pre-effect visual bounds.
- **New Retained Primitives**:
  - `ImageObject`: Raster images supporting BGR and BGRA, source cropping, target sizing, affine warps, and alpha compositing.
  - `Text`: Typography with font families (`FontFamily.SANS_SERIF`, `SERIF`, `MONOSPACE`, `SCRIPT`), alignments (`LEFT`, `CENTER`, `RIGHT`), and background plates with optional `background_radius`.

- **Phase 7 Capabilities (Persistence & History Engine)**:
  - **Canonical JSON Document Envelope**: Root header (`{"format": "drawcv", "version": "1.0", "scene": ...}`) with bit-for-bit deterministic serialization (`sort_keys=True`, `allow_nan=False`).
  - **Strict Version Migration Pipeline**: `SchemaMigrator` enables sequential forward migrations for backward compatibility across schema versions.
  - **Structured Raster Payload**: Lossless PNG base64 encoding with decoded shape and `uint8` dtype validation for `Mask` and `ImageObject`.
  - **Lossless Float Alpha**: Colors serialize alpha as exact floating-point numbers (`0.5 == 0.5`), preventing 8-bit quantization drift.
  - **In-Place Live Object Identity Preservation**: History undo/redo mutates existing live Python instances via `_apply_semantic_state()`, preserving references (`scene.get(id) is original_obj`).
  - **Recursive Group Semantic Snapshots**: `Group._get_semantic_state()` captures descendant states and ordering; `Group._apply_semantic_state()` restores states in-place.
  - **Reversible Operations**: `add`, `remove`, `move_to_front`, `move_to_back`, `move_forward`, `move_backward`, `group`, `ungroup`, `move_object`, `rotate_object`, `scale_object`, `restyle_object`, `with scene.edit(...)`, and `with scene.batch(...)`.
  - **History Invariants**: Automatic redo branch invalidation, no-op edit filtering, exception rollback, and drift-free 50-cycle undo/redo.

- **Phase 8 Capabilities (Temporal Drawing & Animation Engine)**:
  - **Timing & Pacing Model**: Dedicated `Timing` value object controlling `start_time`, `duration`, `delay`, `speed`, and `loop`, with clamped `get_progress()` for progressive reveal and raw `evaluate()` for property tracks.
  - **Easing Suite**: 22 standard easing curves across Linear, Quadratic, Cubic, Sine, Exponential, Circular, Elastic, and Bounce families.
  - **Arc-Length Progressive Path Slicing**: True Euclidean path-length parameterization for `Line`, `Polyline`, `Arrow`, `BezierCurve`, `Path`, `FreehandStroke`, and `Arc` (sweep angle reveal).
  - **Stroke-First Reveal Semantics**: While `render_progress < 1.0`, shapes progressively expose outline stroke only; interior fills are suppressed until reaching full completion ($p = 1.0$).
  - **Explicit Progressive Capability**: `supports_progressive_rendering` prevents slicing recursion and ensures non-path shapes (e.g. `Rectangle`, `Circle`) render safely.
  - **Observational Non-Destructive Sampling**: `scene.render_at_time(t)` evaluates timestamps with suspended history and unconditionally restores authored state in `finally`, preserving live object identity and preventing history pollution.
  - **Automatic Timing Binding**: `drawable.timing` automatically drives `drawable.render_progress` during temporal evaluation without requiring explicit Timeline tracks.
  - **Synchronized Timeline**: Multi-track property animations with deterministic insertion-order conflict resolution and comprehensive `scene.temporal_duration` derivation.
  - **Typed Value Serialization**: `AnimationTrack` persists typed value codecs (`"number"`, `"point"`, `"color"`, `"transform"`, `"bounds"`) and string-only easing contracts.
  - **Document Schema 1.1 & Migration**: Automatic forward migration from Phase 7 (`"1.0"`) to `"1.1"`, injecting temporal defaults across layers and nested groups.
  - **Multi-Frame & Video Export**: `VideoRenderer` encoding playable MP4 video via OpenCV `cv2.VideoWriter`, image sequences, and optional Pillow animated GIF.

---

## Running Tests & Demos

Run automated unit tests (240 tests, 100% green):

```bash
pytest tests/ -v
```

Run visual demonstrations:

```bash
python examples/phase1_demo.py
python examples/phase2_demo.py
python examples/phase3_demo.py
python examples/phase4_demo.py
python examples/phase5_demo.py
python examples/phase6_demo.py
python examples/phase7_demo.py
python examples/progressive_drawing.py
python examples/phase8_demo.py
```

Generated outputs will be saved to `examples/output/`.

---

## Development Roadmap

- [x] **Phase 1 — Foundation**: Retained scene graph, Point, Color, BoundingBox, Line, Rectangle, Circle, StrokeStyle, FillStyle, OpenCVRenderer, Canvas, true alpha compositing, unit tests, visual demos.
- [x] **Phase 2 — Manipulation**: Full affine transform rendering (rotation, non-uniform scaling), object cloning, hit testing.
- [x] **Phase 3 — Advanced Geometry**: Ellipse, Arc, Polygon, Polyline, Arrow, Bézier curves, General Path, RoundedRectangle.
- [x] **Phase 4 — Scene System**: Semantic Groups, Rendering Layers, Selection models, Relative positioning utilities, Centralized scene registry.
- [x] **Phase 5 — Freehand Engine**: FreehandStroke, StrokePoint, Chaikin smoothing, Ramer-Douglas-Peucker simplification, Catmull-Rom interpolation, pressure and velocity sensitive variable width.
- [x] **Phase 6 — Compositing & Effects**: Grayscale masks, clipping rectangles, clipping paths, ImageObject, Text, blur, drop shadows, isolated group/layer offscreen compositing.
- [x] **Phase 7 — Persistence & History**: Strict canonical JSON serialization, forward schema migration, command-based Undo/Redo engine, live identity preservation, recursive group state snapshots.
- [x] **Phase 8 — Temporal Drawing & Animation**: Timing metadata, arc-length progressive rendering, stroke-first reveal, 22 easing curves, non-destructive temporal sampling, Timeline multi-track synchronization, schema 1.1 migration, OpenCV MP4 video encoding.


