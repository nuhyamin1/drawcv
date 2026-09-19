# pydrawcv

An editable, retained-mode 2D drawing library built on NumPy and OpenCV.
Create illustrations, procedural graphics, diagrams, freehand strokes, and animation
as scene objects; render pixels whenever you need them.

**Install `pydrawcv`; import `drawcv`.** Requires Python 3.12 or newer.

```bash
pip install pydrawcv
```

For local development:

```bash
pip install -e ".[dev,typography]"
python -m pytest -q
```

## What's new in 0.10.0

Version 0.10.0 is a major feature release adding extensive vector-geometry tools, stroke spaces, and vector styling:

- **Screen-space and object-space strokes**: `StrokeStyle(space="screen")` keeps line widths and dashes constant in screen pixels, while `StrokeStyle(space="object")` scales and shears with affine transforms.
- **Curve-preserving shape-to-path conversion**: `shape.to_path()` converts circles, ellipses, rounded rectangles, and arcs into editable `Path` objects preserving exact elliptical curves.
- **Stroke-to-path outlining**: `shape.stroke_to_path(tolerance=0.25)` turns solid and dashed strokes—including pressure/velocity freehand ribbons—into filled vector outlines ready for editing, boolean operations, and SVG export.
- **Path measurement & curve queries**: `path.length()`, `path.point_at(progress)`, and `path.tangent_at(progress)` query arc-length positions and tangent unit vectors.
- **Curve-preserving trimming & splitting**: `path.trim(start, end)` and `path.split_at(progress)` extract segments along semantic curves without polygon flattening.
- **Filled-region offsets**: `path.offset(distance, join_style=...)` expands or contracts closed regions with round, bevel, or miter joins, correctly handling holes.
- **Reusable path markers**: `Marker(artwork, ...)` attaches reusable vector artwork to `path.marker_start`, `marker_mid`, and `marker_end` with auto-tangent orientation.
- **Editable vector patterns**: `VectorPattern(artwork, width, height, ...)` provides live tileable vector pattern fills and strokes with native SVG `<pattern>` export.
- **Editable alpha & luminance vector masks**: `VectorMask(artwork, bounds, mode=...)` enables soft fades and vector masks with native SVG `<mask />` export.
- **Rendering & SVG fidelity fixes**: Nonzero fills, clipping, root SVG attributes, transformed text clipping, and strict paint-order diagnostics.

See the complete [0.10.0 Release Notes](https://github.com/nuhyamin1/drawcv/blob/master/RELEASE_NOTES.md) for full API examples and compatibility contracts.

## Draw and export an image

```python
from drawcv import (
    Scene, OpenCVRenderer, Line, Circle, Point, Color,
    StrokeStyle, FillStyle, CapStyle,
)

scene = Scene(640, 360, background=Color.white())
node = Circle(center=Point(100, 180), radius=35,
              fill=FillStyle(color=Color(50, 140, 200)))
connection = Line(
    start=Point(150, 180), end=Point(560, 180),
    stroke=StrokeStyle(color=Color(30, 60, 90), width=8,
                       cap_style=CapStyle.BUTT, dash_array=(20, 12)),
)
scene.add(node)
scene.add(connection)
renderer = OpenCVRenderer()
renderer.render(scene).save("diagram.png")
```

The scene remains editable after export. `Canvas.buffer` is a uint8 BGR array;
`Canvas.to_numpy()` returns a copy. For transparent PNG output, set a transparent
scene background and explicitly request BGRA:

```python
scene.background = Color(0, 0, 0, 0)
transparent = renderer.render(scene, alpha=True)
transparent.save("diagram-transparent.png")
transparent.flatten(Color.white()).save("diagram-white.jpg")
```

See [transparent output and alpha semantics](https://github.com/nuhyamin1/drawcv/blob/master/docs/transparency.md).

## Edit, undo, and save the scene

```python
with scene.edit(connection, name="Restyle connection"):
    connection.stroke.width = 12
    connection.stroke.dash_offset = 8

scene.undo()
scene.redo()
assert scene.get(connection.id) is connection

scene.save_json("diagram.drawcv.json")
loaded = Scene.load_json("diagram.drawcv.json")
renderer.render(loaded).save("diagram-loaded.png")
```

Style and transform assignments reject invalid values without corrupting existing
state. Use `scene.edit` for compound changes and geometry edits, and `scene.batch`
for multiple recorded commands. See [reliable edits](https://github.com/nuhyamin1/drawcv/blob/master/docs/reliability.md) for
rollback guarantees and the limits of raw attribute/list mutation.

New scenes use schema **1.13**, including vector masks, vector patterns, reusable path markers, retained text runs, path clips, paints, and effects.
Schemas 1.0 through 1.12 load with compatible defaults. Older readers reject 1.13;
update readers before sharing new documents with them.

## Organize and position objects

Use `Group(children=[...])` for transform hierarchies and `scene.create_layer(name)`
plus `scene.add(obj, layer=name)` for rendering layers. Use `scene.select([...])`
for collective transforms and `scene.find_by_name/tag/type` for lookup. Objects
have stable IDs, visibility, opacity, z-order, tags, and metadata.

```python
from drawcv import place_right_of

with scene.edit(connection):
    place_right_of(connection, node, gap=20, align="center")
```

Coordinates start at the top left; x increases rightward, y downward, and positive
rotation is clockwise. Local geometry maps through object and ancestor transforms.
Stroke widths and dashes stay in **screen pixels** by default. Set
`StrokeStyle(space="object")` for widths, dashes, caps, and joins that
transform with the object. See [stroke spaces](https://github.com/nuhyamin1/drawcv/blob/master/docs/strokes.md).
Colors are authored as **RGB** integers 0–255 with alpha 0–1; OpenCV buffers use BGR.
Alpha compositing uses source-over with premultiplied internal surfaces for isolation.

## Animate a drawing

```python
from drawcv import Timing

connection.timing = Timing(duration=2)
scene.animate(connection, "stroke.dash_offset", 0, 32, duration=2)
scene.render_at_time(0.75).save("frame.png")
```

`render_at_time` restores authored state and does not add history entries.
`scene.sample(t)` deliberately updates the live model. `VideoRenderer` provides
frame iteration, image-sequence export, and video export through OpenCV codecs.
Line, Polyline, Arrow, BezierCurve, Path, FreehandStroke, and Arc support progressive
reveal. See [stroke/progress semantics](https://github.com/nuhyamin1/drawcv/blob/master/docs/strokes.md) for closure, transforms,
phase anchoring, and the different progress parameterizations.

## Examples and reference

![Caps, joins, dashes, and pressure strokes](https://raw.githubusercontent.com/nuhyamin1/drawcv/master/examples/output/stroke_styles.png)

Run examples as modules from the repository root:

```bash
python -m examples.stroke_styles
python -m examples.progressive_drawing
python -m examples.transparent_output
```

- [Detailed 0.10.0 Release Notes](https://github.com/nuhyamin1/drawcv/blob/master/RELEASE_NOTES.md).
- [Stroke comparison source](https://github.com/nuhyamin1/drawcv/blob/master/examples/stroke_styles.py) and [editable scene](https://github.com/nuhyamin1/drawcv/blob/master/examples/output/stroke_styles.json).
- [Vector path boolean operations and coordinate contracts](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-boolean.md); [runnable gallery](https://github.com/nuhyamin1/drawcv/blob/master/examples/path_boolean_operations.py).
- [Exact shape-to-path conversion and curve-preserving SVG export](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-conversion.md).
- [Stroke-to-path conversion for editable outlines and boolean operations](https://github.com/nuhyamin1/drawcv/blob/master/docs/stroke-conversion.md).
- [Path length, positions and tangent directions](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-measurement.md).
- [Curve-preserving trimming and splitting](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-editing.md).
- [Filled-region offsets: expansion, contraction, holes and joins](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-offsets.md).
- [Reusable path markers and custom arrowheads](https://github.com/nuhyamin1/drawcv/blob/master/docs/markers.md).
- [Editable vector-pattern paints](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-patterns.md).
- [Editable alpha and luminance vector masks](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-masks.md).
- [Transparent PNG example](https://github.com/nuhyamin1/drawcv/blob/master/examples/transparent_output.py) and [external-composition preview](https://github.com/nuhyamin1/drawcv/blob/master/examples/output/transparent_output_preview.png).
- [Gradient paints and coordinate contracts](https://github.com/nuhyamin1/drawcv/blob/master/docs/gradients.md); [runnable gallery](https://github.com/nuhyamin1/drawcv/blob/master/examples/gradient_fills.py).
- [Alpha API and compositing conventions](https://github.com/nuhyamin1/drawcv/blob/master/docs/transparency.md).
- [StrokeStyle API and conventions](https://github.com/nuhyamin1/drawcv/blob/master/docs/strokes.md).
- [Public API guide](https://github.com/nuhyamin1/drawcv/blob/master/docs/api.md).
- [Retained font typography: installation, metrics and layout](https://github.com/nuhyamin1/drawcv/blob/master/docs/typography.md).
- [SVG export: editable vectors, raster fallbacks and strict mode](https://github.com/nuhyamin1/drawcv/blob/master/docs/svg.md); [gallery](https://github.com/nuhyamin1/drawcv/blob/master/examples/svg_export.py).
- [Measured rendering performance](https://github.com/nuhyamin1/drawcv/blob/master/docs/performance.md); [reproducible benchmark commands](https://github.com/nuhyamin1/drawcv/blob/master/benchmarks/README.md).
- [Typography backend evaluation](https://github.com/nuhyamin1/drawcv/blob/master/docs/typography-evaluation.md).
- [Reliability and history](https://github.com/nuhyamin1/drawcv/blob/master/docs/reliability.md).
- [Prioritized roadmap and verified findings](https://github.com/nuhyamin1/drawcv/blob/master/docs/roadmap.md).
- Existing runnable examples: [retained editing](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase1_retained_mode.py),
  [groups/selection](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase4_demo.py), [freehand](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase5_demo.py),
  [compositing](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase6_demo.py), [persistence/history](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase7_demo.py),
  and [animation](https://github.com/nuhyamin1/drawcv/blob/master/examples/phase8_demo.py).

## Capabilities and current limits

| Area | Available | Limits / next work |
| --- | --- | --- |
| Geometry | Lines, polygons, circles/ellipses, arcs, rectangles, arrows, quadratic/cubic curves, compound paths, vector booleans (union, intersection, difference, XOR) | Curves rasterize as approximations during OpenCV rendering |
| Strokes | Butt/round/square caps, round/bevel/miter joins, miter limits, dashed and variable-width outlines | Screen/object stroke spaces; arrowhead outlines stay solid |
| Paint | Solid, linear/radial/conic gradients, gradient strokes, RGBA stops, repeat/reflect spread, image tiles, editable vector patterns, fill rules, blend modes | Nested vector patterns inside pattern artwork remain unsupported |
| Freehand | Raw editable samples; pressure/velocity widths; simplification, smoothing, interpolation | No textured brush system |
| Scene | Groups, layers, lookup, selection, relative positioning, affine transforms | Bounds may be conservative; hit testing is geometric and can select dash gaps |
| Compositing | Opt-in straight BGRA/PNG output; premultiplied images, raster masks, editable alpha/luminance vector masks, blur, shadows, isolated opacity, blend modes, raster effects | Alpha export currently PNG only |
| Typography | Hershey compatibility; optional TTF/OTF, Latin/Thai/Arabic shaping, ICU bidi, explicit fallback, wrapping and metrics | Other scripts, emoji, advanced format controls and glyph strokes remain unsupported |
| Persistence | JSON 1.13; forward migration; cloning; undo/redo | Direct edits require transaction discipline; older readers need updating |
| Animation | Timing, easing, numeric/value tracks, progressive drawing, video | No built-in enum/dash-array interpolation; codec availability varies |
| Interchange | Native retained SVG import & export with text interchange and reported PNG fallbacks; raster images, editable JSON, video | PDF export remains future work |
| Performance | Profiled benchmarks; coverage-region blending/gradients; cheaper transform resolution | Full-canvas masks/effect surfaces remain; no dirty-region redraw |

