# StrokeStyle reference and rendering contract

```python
from drawcv import StrokeStyle, Color, CapStyle, JoinStyle, LineType

stroke = StrokeStyle(
    color=Color(20, 100, 160), width=8, opacity=0.8,
    line_type=LineType.AA,
    cap_style=CapStyle.BUTT, join_style=JoinStyle.MITER,
    dash_array=(18, 10), dash_offset=0, miter_limit=4,
)
```

| Property | Default | Contract |
| --- | --- | --- |
| `color` | black | `Color` with RGB channels 0–255 and alpha 0–1 |
| `width` | `1.0` | Positive finite width in the selected stroke space; fractional values accepted |
| `space` | `"screen"` | `"screen"` for fixed pixel widths/dashes; `"object"` for transformed outlines |
| `opacity` | `1.0` | Numeric multiplier in [0, 1] |
| `line_type` | `LineType.AA` | AA, LINE_8, or LINE_4 rasterization |
| `cap_style` | `CapStyle.ROUND` | BUTT, ROUND, or SQUARE |
| `join_style` | `JoinStyle.ROUND` | ROUND, BEVEL, or MITER |
| `dash_array` | `()` | List/tuple of strictly positive finite on/off lengths; copied to an immutable tuple |
| `dash_offset` | `0.0` | Finite signed distance into the repeating pattern |
| `miter_limit` | `4.0` | Finite ratio >= 1: maximum tip distance from vertex divided by half-width |

Booleans are rejected as numeric style values. Invalid assignments raise
`ValidationError` and preserve the previous valid state. Use enum values, not
strings, when constructing styles; JSON uses enum strings. `copy()`, `to_dict()`,
and `from_dict()` include all style properties.

## Caps, joins, and dashes

Butt caps stop at the centerline endpoint. Round caps add a semicircle of radius
half-width. Square caps extend by half-width along the endpoint tangent. A
zero-length open contour paints a round dot or axis-aligned square; butt paints
nothing. Consecutive duplicate positions are collapsed for rasterization.

Round joins cover the outer turn with a circular arc. Bevel joins connect the two
outer offsets directly. Miter joins intersect their offset directions and fall back
to bevel when the tip exceeds `miter_limit`. Closed solid contours join the last
segment to the first and have no end caps.

Dash entries alternate **on, off**, beginning with on. An odd-length array repeats
twice to form an even-length cycle. Empty means solid. A positive offset consumes
that much of the pattern before the start point; negative offsets wrap modulo the
cycle length. For `(12, 8)`, offset `15` begins with five off pixels. Each connected
on-run uses the chosen caps; vertices inside it use the chosen joins. Caps can
visually close a short gap when width is large.

Dash distance continues through vertices and curves, including the closing edge.
On-runs that cross a closed contour's start/end seam are joined there. Each
compound subpath starts its own pattern at `dash_offset`; empty subpaths do not
shift contour closure flags. All subpaths of one stroke share a coverage mask and
are composited once. Fills retain their existing even-odd/non-zero rules.

## Coordinates and shape coverage

With `space="screen"` (the default), geometry is transformed before stroke expansion
and dash measurement. Widths and dash lengths/offset stay in screen pixels.
With `space="object"`, dash measurement and stroke expansion happen in local
geometry coordinates, then the outline is transformed through the object and its
ancestors. Nonuniform scale and shear affect caps, joins, and widths; a round cap
can become an ellipse. Paint coordinates remain controlled separately by `paint.space`.

```python
stroke = StrokeStyle(width=8, dash_array=(16, 10), space="object")
```

SVG ordinary strokes import as object space with their authored widths and dashes.
`vector-effect="non-scaling-stroke"` imports as screen space. Both export natively.
Resizing an SVG keeps exported screen strokes fixed, unlike earlier exports which
scaled with the document. Run `python -m examples.stroke_spaces` for a
[side-by-side comparison](../examples/output/stroke_spaces.png).

The shared renderer covers Line, Polyline, Polygon, Rectangle, RoundedRectangle,
Circle, Ellipse, Arc, BezierCurve, Path, and FreehandStroke. Arrow shafts use the
dash pattern; arrowhead outlines use the same cap/join style but stay solid so
markers remain legible. Arrowhead dimensions retain their existing screen-space
contract; their outlines use the chosen stroke space. Text glyph thickness is a separate Hershey setting and
does not use StrokeStyle.

Variable-width freehand retains all raw samples. Widths are evaluated on processed
points, then interpolated at dash cuts. Ribbons taper between samples; joins and
caps use the local sample width. Pressure/velocity processing remains independent
of the geometric dash pattern. Stroke extents include a conservative allowance
for miter tips and rotated square caps, preventing isolation/effect clipping.

## Progressive drawing and animation

`progress=0` hides the drawable. For supported path-like objects, fractional progress
uses the existing slicing API and strokes the resulting contour; an incomplete
closed outline is open and has a cap at the reveal endpoint. At completion, closed
joins and eligible fills return. Parent/layer context and the original default
pivot are preserved by the renderer without editing source geometry.

Dash phase stays anchored to each subpath's start; it does not restart at the moving
reveal endpoint. Existing progress parameterization is retained: local arc length
for line/polyline/freehand/Bezier/path slicing and sweep angle for arcs. Under
non-uniform transforms this is not constant world-space reveal speed. Processing
and adaptive flattening can slightly change curved/freehand geometry near a reveal
frontier. The style contract does not promise pixel-identical partial curve prefixes.

```python
# Animate moving dashes without changing the authored scene during rendering.
scene.animate(line, "stroke.dash_offset", 0, 28, duration=2)
frame = scene.render_at_time(0.5)
```

`width`, `opacity`, `dash_offset`, and `miter_limit` support existing numeric tracks.
Enum and tuple interpolation is not supported by the built-in animation system;
attempts to animate cap/join enums or dash arrays are rejected. Use explicit edits
for those discrete changes. Numeric easing must stay inside property constraints.

## Compatibility and limitations

- Scene files now write schema **1.11**. Schemas 1.0 through 1.10 migrate forward;
  omitted stroke space defaults to `"screen"`. Older
  readers reject 1.11 rather than silently discarding new styling. Keep old readers
  updated when sharing scenes. Package release/version publication is separate.
- Existing round styles may have different boundary pixels because outlines now
  use tessellated coverage instead of OpenCV thick-line defaults. Stroke opacity
  no longer accumulates at crossings within one compound Path.
- Bounds for miter/square styles are conservative rather than tight. Bounds-based
  anchors, mask fitting, and selection boxes may therefore be larger.
- Object stroke bounds use the maximum affine stretch as a conservative allowance.
  Singular transforms collapse the outline and produce no stroke coverage.
- Boolean results have world-space geometry. Object strokes under similarity
  transforms convert to equivalent screen strokes. Nonuniform/sheared object
  strokes raise `PathBooleanError`; remove the operand stroke and style the result
  explicitly. Stroke-to-path conversion remains future work.
- Hit testing retains its existing geometric/tolerance behavior: it may select a
  dashed gap and is not exact cap/join pixel coverage. It is intended for selecting
  editable paths, not querying rendered alpha.
- Curves are flattened approximations, rasterization uses OpenCV subpixel coverage,
  and extremely small dashes/large coordinates remain limited by numeric precision.
  There is no claim of resolution-independent analytical antialiasing.
- Masks remain canvas-sized. Dense dashes add geometry work; performance changes
  need the representative benchmarks in the roadmap.

Run `python -m examples.stroke_styles` from the repository to regenerate
[the comparison image](../examples/output/stroke_styles.png) and its editable JSON.

Object-space milestone verification: **1,224 tests pass** on Windows/Python 3.12,
including 35 new cases covering affine SVG parity, round/square/butt caps, joins,
dashes, reflections, paint coordinates, pressure strokes, effects, animation,
undo/redo, schema migration, and compressed-curve dash measurement. The stroke-space
comparison example was rendered and visually inspected. Other platforms and
lower-bound dependency versions were not rerun for this milestone.
