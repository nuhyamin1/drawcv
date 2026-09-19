# Expanding and contracting filled regions

```python
from drawcv import JoinStyle

expanded = path.offset(10)
contracted = path.offset(-5)
sharp = path.offset(10, join_style=JoinStyle.MITER, miter_limit=4)
local = path.offset(10, space="local", tolerance=0.1)
```

`Path.offset()` expands the filled region for positive distances and contracts it
for negative distances. Holes shrink during expansion and grow during contraction,
regardless of contour direction. The source's fill rule defines the region even
when it has no visible fill. Intersections and overlapping contours are resolved
before offsetting. Regions may merge, split or disappear.

Only closed contours are accepted. Convert shapes with `to_path()` first; use
`stroke_to_path()` to outline an open centerline. This API does not generate a
one-sided parallel open curve.

`join_style` accepts `JoinStyle.ROUND` (default), `BEVEL`, or `MITER`.
The miter limit is the maximum corner extension divided by the offset distance;
exceeding it uses a bevel. Round offsets correspond to a circular neighborhood
of the normalized boundary. Bevel and miter joins alter exposed corners of this
boundary band. The band is added for expansion and removed for contraction.

## Coordinates and appearance

`space="world"` is the default: distance and tolerance are in world units,
including parent transforms. The detached result has world geometry and an
identity transform. `space="local"` offsets local geometry and retains the full
source world transform on the result. Under nonuniform scaling this produces
a different result from a uniform world-space offset.

The result has a new ID and no parent or scene. Fill, own opacity, visibility,
blend mode, name, tags and metadata are copied independently. Object-space fill
paints are mapped into world coordinates for world-space output. If the source
has no fill, the result gets the default `FillStyle()` so it can be displayed.
Source strokes, clips, masks, effects, timing, render progress and ancestor
appearance are excluded. The original path is unchanged.

## Approximation and limits

Output is editable closed polygon geometry with the nonzero fill rule. Curves
are flattened and round corners sampled using `tolerance` (default `0.25`) in
the selected coordinate space. This is a subdivision target rather than a strict
global error bound. Smaller values increase vertex count and computation cost;
PathOps cleanup also has floating-point precision limits. Very small details
may disappear. Zero distance still normalizes and polygonizes the source region.

An erosion larger than a region's thickness returns empty geometry. Empty inputs
also return empty geometry. Nonfinite distances, nonpositive tolerances, invalid
join options and open contours raise `ValidationError`; backend failures raise
`PathBooleanError`. No new dependency or JSON schema is required.

Run `python -m examples.path_offsets` for a gallery. This completes the offsets
part of the path-authoring milestone; reusable markers are the next roadmap item.
