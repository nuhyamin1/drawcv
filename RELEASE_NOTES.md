# pydrawcv 0.10.0 Release Notes

`pydrawcv` 0.10.0 is a major feature release adding extensive vector path manipulation, stroke geometry transformations, reusable markers, pattern paints, and vector masks. The Python package is `pydrawcv` and the import name remains `drawcv`.

---

## What's New in 0.10.0

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
- See [docs/strokes.md](docs/strokes.md) for full coordinate and styling contracts.

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
- See [docs/path-conversion.md](docs/path-conversion.md).

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
- See [docs/stroke-conversion.md](docs/stroke-conversion.md).

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
- See [docs/path-measurement.md](docs/path-measurement.md).

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
- See [docs/path-editing.md](docs/path-editing.md).

```python
middle_segment = path.trim(0.2, 0.8)
prefix, suffix = path.split_at(0.4)
```

### 7. Filled-Region Offsets with Configurable Joins
- Added `path.offset(distance, join_style=JoinStyle.ROUND, miter_limit=4.0, space="world", tolerance=0.25)`.
- Expands (`distance > 0`) or contracts (`distance < 0`) filled regions while properly shrinking or expanding interior holes.
- Supports `JoinStyle.ROUND`, `JoinStyle.BEVEL`, and `JoinStyle.MITER`.
- Resolves self-intersections and overlapping contours using Skia PathOps.
- See [docs/path-offsets.md](docs/path-offsets.md).

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
- See [docs/markers.md](docs/markers.md).

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
- See [docs/vector-patterns.md](docs/vector-patterns.md).

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
- See [docs/vector-masks.md](docs/vector-masks.md).

```python
from drawcv import VectorMask, BoundingBox, Circle, Point, FillStyle, Color

mask_art = Circle(center=Point(50, 50), radius=40, fill=FillStyle(color=Color.white())).to_path()
shape.mask = VectorMask(mask_art, bounds=BoundingBox(0, 0, 100, 100), mode="alpha")
```

---

## Compatibility and Migration

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

## Documentation Index

- [Stroke Spaces & Styles](docs/strokes.md)
- [Shape-to-Path Conversion](docs/path-conversion.md)
- [Stroke-to-Path Outlining](docs/stroke-conversion.md)
- [Path Measurement](docs/path-measurement.md)
- [Path Trimming & Splitting](docs/path-editing.md)
- [Filled-Region Offsets](docs/path-offsets.md)
- [Reusable Path Markers](docs/markers.md)
- [Vector Patterns](docs/vector-patterns.md)
- [Vector Masks](docs/vector-masks.md)
- [SVG Export & Import](docs/svg.md)
