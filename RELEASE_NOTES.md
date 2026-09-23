# pydrawcv 0.11.0 Release Notes

`pydrawcv` 0.11.0 adds shape labels, SVG path strings, paint order, SVG filter export
and vector export for arrows and pressure strokes, and replaces the rasterizer with
area-exact antialiasing on the SVG pixel grid. The Python package is `pydrawcv` and
the import name remains `drawcv`.

---

## Before You Upgrade

- **Rendered images change.** Antialiased edges are now area-exact: a 2 px line
  covers exactly 2 px (it covered 3.35 px) and fills are no longer about 4% too
  large. Pixels span `[k, k + 1]` as in SVG, so shapes also move half a pixel.
  Tests that compare against stored reference images need new references.
- **1 px lines on whole coordinates look lighter.** As in browsers, a 1 px line
  centered on `x = 70` covers two half pixels. Center hairlines on `.5` or use an
  even width for crisp lines. See [rasterization](https://github.com/nuhyamin1/drawcv/blob/master/docs/rasterization.md).
- **Scene files use schema 1.14.** 0.11.0 reads every older schema, but 0.10.x
  cannot open files saved by 0.11.0. Upgrade every reader before sharing files.
- **Effects export as SVG filters.** Blur, shadow, glow, color and convolution
  effects no longer produce PNG fallbacks, so `SVGExport.fallbacks` and strict
  export behave differently for scenes that use them.

---

## What's New in 0.11.0

### 1. Shape Labels

Attach text to any shape, laid out inside or around it. Labels follow moves,
resizes, rotation and animation, and export as native SVG text when they use a
retained font.

```python
from drawcv import Rectangle, Line, Point, Label, StrokeStyle

card = Rectangle(position=Point(40, 30), width=200, height=120)
card.label = Label("Centered on both axes", font_scale=0.8)
card.label.vertical_align = "bottom"   # top / middle / bottom; align: left / center / right

link = Line(start=Point(40, 200), end=Point(240, 200), stroke=StrokeStyle(width=4))
link.label = Label("HTTPS", placement="above", padding=6)  # captions outside the shape
```

### 2. SVG Path Strings

```python
from drawcv import Path, FillStyle, Color

badge = Path.from_svg_path("M10 10 h60 a10 10 0 0 1 10 10 v20 h-70 z",
                           fill=FillStyle(color=Color(50, 140, 200)))
badge.to_svg_path()  # "M 10 10 L 70 10 A 10 10 0 0 1 80 20 ..."
```

All SVG path commands are accepted, absolute and relative, including arcs.

### 3. Paint Order

`shape.paint_order = "stroke"` draws the stroke before the fill, like SVG
`paint-order`. It is saved, undoable, exported to SVG and read from imported SVG,
which previously rejected it.

### 4. Vector Export for Arrows, Pressure Strokes and Effects

- Arrows export as native shaft and head shapes; pressure and velocity freehand
  strokes export as their filled outline.
- Blur, shadow, glow, brightness/contrast, saturation, grayscale, sepia, hue shift,
  color matrix and convolution effects export as SVG filters computing the same
  operation. Sharpen, emboss, edge detection, noise, displacement and biased
  convolution keep reported PNG fallbacks.

### 5. Area-Exact Rasterization

Strokes, fills, paths and clips share a new rasterizer whose coverage equals the
geometric area, with nonzero and even-odd fill rules and seam-free stroke unions.
Rendering matches resvg on DrawCV's own SVG export (area ratio 1.000 to 1.005,
previously 0.80 to 0.89). Most benchmark scenes also render faster.

### 6. Performance

Large paths and paths inside nested groups no longer recompute their bounds for
every point; an imported 315-command outline went from 29.6 s to 0.14 s. Median
frame times on the reference benchmarks (`python -m benchmarks.render`, Windows,
one OpenCV thread) against 0.10.0.post1:

| Scene | 0.10.0.post1 | 0.11.0 |
| --- | ---: | ---: |
| Dense dash patterns | 339 ms | 187 ms |
| 6,000 pressure samples | 232 ms | 149 ms |
| Six nested opacity/transform groups | 110 ms | 68 ms |
| 72 small gradients | 66 ms | 44 ms |
| 24 objects / 48 animation tracks | 55 ms | 37 ms |
| 16 masked and effected objects | 93 ms | 72 ms |
| 480 small outlines | 266 ms | 281 ms |
| 1,200 opaque rectangles | 205 ms | 224 ms |

---

## Compatibility and Migration

- **JSON schema 1.14** adds labels and paint order. Schemas 1.0 through 1.13
  migrate forward on load; drawables without labels or paint order serialize as before.
- **Aliased strokes** (`LineType.LINE_4`, `LINE_8`), Hershey text and retained font
  text rasterize as before.
- **Gradients** sample pixel centers, like image paints and SVG viewers.
- **SVG import** keeps `paint-order` and reports malformed values as
  `SVG_INVALID_PAINT_ORDER`; `SVG_UNSUPPORTED_PAINT_ORDER` is no longer raised.

---

## Documentation Index

- [Shape Labels](https://github.com/nuhyamin1/drawcv/blob/master/docs/labels.md)
- [Paint Order](https://github.com/nuhyamin1/drawcv/blob/master/docs/paint-order.md)
- [Rasterization and the Pixel Grid](https://github.com/nuhyamin1/drawcv/blob/master/docs/rasterization.md)
- [SVG Export, Filters & Import](https://github.com/nuhyamin1/drawcv/blob/master/docs/svg.md)
- [Measured Performance](https://github.com/nuhyamin1/drawcv/blob/master/docs/performance.md)

---

# Earlier Releases

## pydrawcv 0.10.0

`pydrawcv` 0.10.0 is a major feature release adding extensive vector path manipulation, stroke geometry transformations, reusable markers, pattern paints, and vector masks. The Python package is `pydrawcv` and the import name remains `drawcv`.

---

### What's New in 0.10.0

### 1. Rendering and SVG Correctness Fixes
- **Nonzero Fills and Clipping**: Fixed rendering, clipping, and hit testing for self-intersecting paths with non-zero fill rules.
- **Root SVG Attributes**: Correctly preserved root SVG opacity, transforms, and viewbox coordinate mapping during native retained SVG import.
- **Transformed Text Clipping**: Fixed transformed text clipping during SVG export.
- **Gradient Paint Preservation**: Preserved gradient spread (`pad`, `repeat`, `reflect`) and paint transformation matrices during supported boolean conversions.
- **Explicit Paint Order Diagnostics**: Unsupported SVG `paint-order` attributes now fail explicitly with `SVGImportError` (`SVG_UNSUPPORTED_PAINT_ORDER`) rather than silently producing incorrect stacking.

### 2. Screen-Space and Object-Space Strokes
- `StrokeStyle` now supports `space="screen"` (the default) and `space="object"`.
- In screen space, stroke widths and dash lengths remain constant in viewport/screen pixels regardless of parent transforms.
- In object space, stroke widths, dashes, caps, joins, and miter limits transform with the object's affine matrix (scaling and shearing).
- SVG export outputs `vector-effect="non-scaling-stroke"` for screen-space strokes and native transformed paths for object-space strokes.
- See [docs/strokes.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/strokes.md) for full coordinate and styling contracts.

```python
from drawcv import StrokeStyle, CapStyle, JoinStyle

# Fixed screen-pixel outline
screen_stroke = StrokeStyle(width=4, space="screen")

# Scalable outline that transforms with the shape
object_stroke = StrokeStyle(width=4, space="object", cap_style=CapStyle.ROUND, join_style=JoinStyle.ROUND)
```

### 3. Curve-Preserving Shape-to-Path Conversion and SVG Export
- Added `shape.to_path(preserve_world_transform=False)` on circles, ellipses, rounded rectangles, arcs, lines, polygons, and Bézier curves.
- Circles, ellipses, rounded rectangles, and arcs retain exact elliptical arc commands (`ArcTo`) rather than approximating with polygons.
- Quadratic and cubic Bézier curves retain exact control points.
- Detached paths receive unique IDs and can optionally preserve full world transforms.
- SVG export uses exact curved commands for these shapes.
- See [docs/path-conversion.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-conversion.md).

```python
# Convert a circle or arc to an editable Path preserving exact curves
path = circle.to_path()
detached_path = child.to_path(preserve_world_transform=True)
```

### 4. Stroke-to-Path Conversion
- Added `shape.stroke_to_path(tolerance=0.25)` to convert standard stroked geometry and freehand strokes into editable filled `Path` outlines.
- Supports solid and dashed outlines, butt/round/square caps, and round/bevel/miter joins.
- Fully supports both `screen` and `object` stroke spaces under affine transforms.
- `FreehandStroke.stroke_to_path()` generates variable-width filled ribbon outlines from raw pressure and velocity sample streams, complete with RDP simplification, Chaikin smoothing, and Catmull-Rom interpolation.
- Resulting outlines are ready for boolean operations, custom fills, and SVG export.
- See [docs/stroke-conversion.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/stroke-conversion.md).

```python
# Create an editable filled outline of a stroked path
outline = path.stroke_to_path(tolerance=0.1)
combined = outline.union(other_outline)
```

### 5. Path Measurement: Length, Points, and Tangents
- Added `path.length(tolerance=0.1, space="local", subpath=None)`.
- Added `path.point_at(progress, tolerance=0.1, space="local", subpath=None)`.
- Added `path.tangent_at(progress, tolerance=0.1, space="local", subpath=None)`.
- Progress is normalized arc length in `[0, 1]`.
- Béziers and arcs use adaptive numerical integration of curve speed. Positions invert arc length against semantic curves, and tangents compute unit direction vectors.
- Supports both local and world coordinate spaces and individual subpath queries.
- See [docs/path-measurement.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-measurement.md).

```python
distance = path.length()
midpoint = path.point_at(0.5)      # Point(x, y) halfway along curve
direction = path.tangent_at(0.5)   # Point(dx, dy) unit tangent vector
```

### 6. Curve-Preserving Path Trimming and Splitting
- Added `path.trim(start, end, tolerance=0.1, space="local", subpath=None, preserve_world_transform=False)`.
- Added `path.split_at(progress, tolerance=0.1, space="local", subpath=None, preserve_world_transform=False)`.
- Uses normalized arc length parameters (`0 <= start <= end <= 1`).
- Preserves semantic curve commands: Béziers use de Casteljau subdivision; elliptical arcs preserve radii, rotation, and sweep.
- Preserves closure when whole subpaths are selected; cuts cleanly open subpaths.
- See [docs/path-editing.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-editing.md).

```python
middle_segment = path.trim(0.2, 0.8)
prefix, suffix = path.split_at(0.4)
```

### 7. Filled-Region Offsets with Configurable Joins
- Added `path.offset(distance, join_style=JoinStyle.ROUND, miter_limit=4.0, space="world", tolerance=0.25)`.
- Expands (`distance > 0`) or contracts (`distance < 0`) filled regions while properly shrinking or expanding interior holes.
- Supports `JoinStyle.ROUND`, `JoinStyle.BEVEL`, and `JoinStyle.MITER`.
- Resolves self-intersections and overlapping contours using Skia PathOps.
- See [docs/path-offsets.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-offsets.md).

```python
from drawcv import JoinStyle

expanded = path.offset(10, join_style=JoinStyle.ROUND)
contracted = path.offset(-5, join_style=JoinStyle.MITER)
```

### 8. Reusable Path Markers
- Added `Marker(artwork, ref=Point(0, 0), orient="auto", units="stroke", size=1.0)`.
- Paths support `path.marker_start`, `path.marker_mid`, and `path.marker_end`.
- Orientations include `"auto"` (follows curve tangent), `"auto-start-reverse"`, or fixed angle degrees.
- Units include `"stroke"` (scales with host stroke width) and `"user"` (fixed coordinate size).
- Markers reference artwork live without mutating host or source objects.
- JSON schema 1.11+ retains marker attachments; SVG export outputs markers as editable vector groups.
- See [docs/markers.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/markers.md).

```python
from drawcv import Marker, Path, Point, StrokeStyle

arrowhead = (Path().move_to(0, 0).line_to(-4, -2).line_to(-4, 2).close())
marker = Marker(arrowhead, ref=Point(0, 0), orient="auto-start-reverse")

path = Path(stroke=StrokeStyle(width=4))
path.marker_start = marker
path.marker_end = marker
```

### 9. Editable Vector-Pattern Paints
- Added `VectorPattern(artwork, width, height, spacing=(0, 0), origin=Point(0, 0), space="object", transform=Transform(), opacity=1.0)`.
- Use as a retained paint on `FillStyle(paint=pattern)` and `StrokeStyle(paint=pattern)`.
- Live editable artwork: changes to the source artwork update all usages on next render.
- Supports spacing gaps, tile clipping, and pattern affine transforms.
- Native SVG `<pattern>` export with vector content.
- See [docs/vector-patterns.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-patterns.md).

```python
from drawcv import VectorPattern, Circle, Point, Color, FillStyle

dot = Circle(center=Point(6, 6), radius=3, fill=FillStyle(color=Color.blue())).to_path()
pattern = VectorPattern(dot, width=12, height=12, spacing=(4, 4))
shape.fill = FillStyle(paint=pattern)
```

### 10. Editable Alpha and Luminance Vector Masks
- Added `VectorMask(artwork, bounds, mode="alpha", space="object", transform=Transform(), opacity=1.0)`.
- Modes: `"alpha"` (artwork alpha controls opacity) and `"luminance"` (Rec.709 luminance weighted `0.2126 R + 0.7152 G + 0.0722 B` controls opacity).
- Explicit `bounds` region; artwork outside bounds is clipped.
- Operates cleanly on isolated surfaces before final compositing.
- Native SVG `<mask>` export.
- See [docs/vector-masks.md](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-masks.md).

```python
from drawcv import VectorMask, BoundingBox, Circle, Point, FillStyle, Color

mask_art = Circle(center=Point(50, 50), radius=40, fill=FillStyle(color=Color.white())).to_path()
shape.mask = VectorMask(mask_art, bounds=BoundingBox(0, 0, 100, 100), mode="alpha")
```

---

### Compatibility and Migration

- **JSON Document Schema Version**: New documents are authored in **schema 1.13**. Package version (0.10.0) and document schema version (1.13) are distinct.
- **Forward Migration**: Older documents (schemas 1.0 through 1.12) migrate forward automatically on load.
- **Backward Readers**: Older `pydrawcv` versions (prior to schema 1.13 support) cannot load schema 1.13 documents. There is no supported export or downgrade to older schema versions.
- **SVG Interchange**:
  - Vector patterns and vector masks export to native SVG `<pattern>` and `<mask />` elements.
  - Native SVG pattern and mask *import* is not supported in this release; JSON is the primary interchange format for these objects.
  - Reusable path markers are expanded into editable SVG vector groups during export to guarantee visual fidelity across SVG renderers. Re-import preserves their vector appearance as separate paths, not native `<marker>` attachments.
- **Tolerances & Approximations**:
  - Curved offsets (`path.offset()`) and stroke outlines (`shape.stroke_to_path()`) evaluate curves using configurable approximation tolerances (`tolerance` parameter).
- **Nesting Restrictions**:
  - Nested vector patterns, image paints, text, masks, and effects inside pattern tile artwork are disallowed and reject with `ValidationError`.
  - Nested masks inside vector mask artwork are disallowed and reject with `ValidationError`.
- **Typography Scope**:
  - Further typography enhancements beyond existing HarfBuzz/FreeType/ICU shaping remain deferred for future releases.

---

### Documentation Index

- [Stroke Spaces & Styles](https://github.com/nuhyamin1/drawcv/blob/master/docs/strokes.md)
- [Shape-to-Path Conversion](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-conversion.md)
- [Stroke-to-Path Outlining](https://github.com/nuhyamin1/drawcv/blob/master/docs/stroke-conversion.md)
- [Path Measurement](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-measurement.md)
- [Path Trimming & Splitting](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-editing.md)
- [Filled-Region Offsets](https://github.com/nuhyamin1/drawcv/blob/master/docs/path-offsets.md)
- [Reusable Path Markers](https://github.com/nuhyamin1/drawcv/blob/master/docs/markers.md)
- [Vector Patterns](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-patterns.md)
- [Vector Masks](https://github.com/nuhyamin1/drawcv/blob/master/docs/vector-masks.md)
- [SVG Export & Import](https://github.com/nuhyamin1/drawcv/blob/master/docs/svg.md)
