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

---

## Running Tests & Demos

Run automated unit tests:

```bash
pytest tests/ -v
```

Run visual demonstrations:

```bash
python examples/phase1_demo.py
python examples/phase2_demo.py
python examples/phase3_demo.py
python examples/phase4_demo.py
```

Generated outputs will be saved to `examples/output/`.

---

## Development Roadmap

- [x] **Phase 1 — Foundation**: Retained scene graph, Point, Color, BoundingBox, Line, Rectangle, Circle, StrokeStyle, FillStyle, OpenCVRenderer, Canvas, true alpha compositing, unit tests, visual demos.
- [x] **Phase 2 — Manipulation**: Full affine transform rendering (rotation, non-uniform scaling), object cloning, hit testing.
- [x] **Phase 3 — Advanced Geometry**: Ellipse, Arc, Polygon, Polyline, Arrow, Bézier curves, General Path, RoundedRectangle.
- [x] **Phase 4 — Scene System**: Semantic Groups, Rendering Layers, Selection models, Relative positioning utilities, Centralized scene registry.
- [ ] **Phase 5 — Freehand Engine**: FreehandStroke, StrokePoint, Chaikin smoothing, Ramer-Douglas-Peucker simplification, Catmull-Rom interpolation, pressure-sensitive width.
- [ ] **Phase 6 — Compositing & Effects**: Grayscale masks, clipping rectangles, clipping paths, ImageObject, blur, drop shadows, isolated group/layer offscreen compositing.
- [ ] **Phase 7 — Persistence & History**: JSON serialization/deserialization with format versioning, command-based Undo/Redo.
- [ ] **Phase 8 — Temporal Drawing**: Timing metadata, path-length-based progressive rendering (`render_progress`), easing functions, video/frame rendering.
