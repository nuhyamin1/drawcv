# Stroke to path

`shape.stroke_to_path(tolerance=0.25)` converts a standard stroke into an
editable filled `Path`. It supports freehand strokes and the same vector shapes as `to_path()`:
paths, lines, rectangles, rounded rectangles, circles, ellipses, arcs,
polygons, polylines and Bézier curves.

```python
outline = shape.stroke_to_path(tolerance=0.1)
scene.add(outline)
combined = outline.union(other_shape.stroke_to_path())
```

Solid and dashed strokes retain cap style, join style, miter limit and dash
offset. Odd-length dash patterns repeat, and each subpath restarts the dash
phase. Overlapping stroke pieces are merged; closed strokes retain their
interior holes. The original fill is excluded.

The result has a new ID, no parent and an identity transform. Its coordinates
are in world space, including ancestor transforms. Object-space strokes are
outlined before applying the transform; screen-space strokes are outlined
afterward. This allows boolean operations on outlines of nonuniformly scaled
or sheared strokes. Add the result directly to the scene, or use the default
world-preserving `group.add(outline)` when inserting into a group.

Stroke paint and stroke opacity become fill paint and fill opacity. Object
paint transforms are converted to world coordinates. Own drawable opacity,
visibility, blend mode, name, tags and metadata are retained independently.
Clips, masks, effects, timing and ancestor appearance are excluded: this is
the full stroke's geometry, regardless of animation progress. The source is
unchanged. Missing and zero-width strokes produce empty paths; a singular
object-space transform has no filled area.

Curves and round joins/caps are approximated with line segments. `tolerance`
is a positive finite world-unit subdivision tolerance; smaller values produce
more vertices. It is not a strict bound on total offset or dash-position error,
especially near sharp cusps. Boolean cleanup uses the existing PathOps backend
and its floating-point precision. SVG and JSON preserve the resulting filled
contours without requiring a new schema.

## Freehand strokes

Run `python -m examples.freehand_outlines` for a side-by-side gallery of original
pressure/velocity strokes and editable outlines.

`FreehandStroke.stroke_to_path()` supports constant, pressure and velocity width
modes, including dashes and both stroke spaces. It uses the same processed points
and evaluated widths as the renderer: RDP simplification, Chaikin smoothing and
Catmull-Rom interpolation run in their usual order. Missing pressure or velocity
uses the existing width-mode defaults. Widths interpolate along segments and at
dash cuts. Raw points and their pressure, velocity and timestamp data are unchanged.

```python
outline = freehand.stroke_to_path(tolerance=0.1)
scene.add(outline)  # editable filled geometry; also supports strict vector SVG
```

Freehand centerline detail is controlled by its processing settings, including
`interpolation_samples`; conversion tolerance refines round caps and joins, not
the sampled centerline. Empty strokes and strokes with all widths zero produce
empty outlines. Consecutive coincident samples follow the renderer's existing
rule of retaining the first sample's width. Conversion outlines the full stroke,
even if `render_progress` is partial; explicitly convert
`freehand.slice_at_progress(progress)` to outline a partial stroke.

Text and arrow outlining remain unsupported and raise `ValidationError`.
