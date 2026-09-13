# Gradient contract

The milestone 3 contract was fixed before renderer integration: first-class linear
and radial fill paints, absolute object-local or world coordinates, padded spread,
ordered RGBA stops, straight-channel interpolation in stored RGB space, schema 1.3,
and geometric-property animation. Focal points, repeat/reflect, gradient strokes,
stop-list morphing, patterns, and blend modes are outside this milestone.

```python
from drawcv import FillStyle, LinearGradient, RadialGradient, GradientStop, Point, Color

stops = (GradientStop(0, Color.red()), GradientStop(1, Color.blue()))
fill = FillStyle(paint=LinearGradient(Point(20, 20), Point(180, 20), stops))
glow = FillStyle(paint=RadialGradient(Point(100, 100), 80, (
    GradientStop(0, Color(255, 180, 20)),
    GradientStop(1, Color(255, 180, 20, 0)),
)))
```

## FillStyle and paint ownership

`FillStyle(enabled=True, color=..., opacity=1)` remains the solid-fill API,
including its positional argument order. `FillStyle(paint=...)` accepts a `Color`,
`LinearGradient`, or `RadialGradient`. Supplying both `color` and `paint` is an error.
There is one active paint, not a hidden solid fallback. Assigning `fill.color`
replaces the active paint with a solid Color. Reading `fill.color` on a gradient
raises `ValidationError`; inspect `fill.paint` instead.

Paint instances are mutable retained values. Sharing one instance shares edits;
`fill.copy()` and drawable cloning make independent copies. Invalid edits preserve
the previous value. GradientStop and its Color are immutable, and the stop sequence
is copied to a tuple. Replace `paint.stops` in a scene edit to change stops.
FillStyle holds descriptions and validation; it does not calculate pixels.

## Coordinates and transforms

Both gradient types accept `space="object"` (default) or `space="world"`.

- Object coordinates are the drawable's actual local coordinates, not normalized
  bounding-box fractions. A rectangle at `(100, 50)` needs gradient coordinates
  in that same local system. The inverse complete object/ancestor affine matrix
  maps each raster sample into gradient space. Object rotation, scale, shear, and
  nested group transforms carry the paint along with the geometry.
- World coordinates remain fixed in the scene. Moving or scaling the object moves
  its coverage through the gradient; it does not move the gradient itself.
- Raster samples are at integer pixel coordinates, matching the existing renderer.
  Coverage antialiasing and gradient sampling are separate; this is not analytic
  integration of a gradient over every pixel footprint.
- Linear `start` and `end` must be distinct Points. Projection onto that vector
  determines position. Radial `center` is a Point and `radius` is positive and
  finite. Distance divided by radius determines position. An object-local radial
  gradient becomes elliptical under nonuniform scale. There is no focal point.

## Stops, interpolation, and outside behavior

A gradient needs at least two GradientStops, supplied in nondecreasing position
order. Positions must be finite numbers in `[0, 1]`; out-of-range positions and
unordered lists are rejected, not sorted or clamped. Duplicate positions are
allowed. The last duplicate wins at the exact position; the first duplicate is
the color approached from below. This creates deterministic hard transitions.

Missing endpoints are allowed. Below the first stop, its color extends indefinitely;
above the last stop, its color extends indefinitely. This padded spread also covers
pixels outside a radial circle or beyond a linear endpoint.

B, G, R and alpha interpolate separately and linearly in stored channel space.
There is no linear-light, perceptual, or ICC conversion. The interpolated color is
then premultiplied by interpolated alpha for source-over composition and filtering.
Thus opaque red to transparent blue has a half-transparent purple midpoint; use
the same RGB at both stops for a fade that preserves hue.

## Coverage, opacity, and effects

Paint uses existing fill coverage for rectangles, rounded rectangles, circles,
ellipses, polygons, closed arcs, compound paths, arrowhead interiors, and text
background plates. Text glyphs and outlines still use their existing solid colors.
Compound paths retain their fill rules. Progressive path fills remain suppressed
until completion under the existing progressive API.

Stop alpha, fill opacity, coverage, masks, and object/group/layer opacity combine
through the premultiplied pipeline. Blur and shadows filter the resulting coverage
and premultiplied color. Clip geometry and mask mapping retain the contracts in
[transparent rendering](transparency.md). Incomplete paths are sliced before fill
evaluation; completed paths retain authored transform/paint coordinates.

Scenes containing gradients use that corrected pipeline even for BGR export,
flattened over the scene background RGB (background alpha remains ignored in BGR).
Solid-only scenes retain the legacy BGR path. Therefore solids in a mixed gradient
scene may show the alpha corrections documented for milestone 2. For transparent
PNG use an alpha-zero background and `render(scene, alpha=True)`.

## Persistence and animation

New documents write schema **1.3**. Older 1.0–1.2 scenes migrate forward without
changing solid fill dictionaries. Gradient fills serialize a typed `paint` object
with geometry, space, and all stop positions/colors/alpha. Older readers reject
1.3 instead of silently losing paint. Copy, clone, history, and temporal snapshots
all use this same semantic representation.

Supported built-in animation paths include `fill.paint.start`, `fill.paint.end`,
`fill.paint.center`, their Point components such as `fill.paint.center.x`,
`fill.paint.radius`, and `fill.opacity`. Values must remain valid at every sample:
radius must stay positive and endpoints distinct. Arbitrary stop arrays, stops,
space strings, and whole gradient objects are not built-in animation value types;
those tracks are rejected. Use explicit `scene.edit` changes for stop lists.

Full-canvas paint buffers are currently allocated when sampling. This milestone
does not introduce caching or claim a performance improvement. Extremely large
coordinates remain subject to floating-point and raster limits.

## Runnable example and verification

Run `python -m examples.gradient_fills` from the repository root. The
[example source](../examples/gradient_fills.py) exports editable JSON, transparent
PNG, and a checkerboard preview into `examples/output/gradient_fills*`.
The gallery compares rotated object-local and world-fixed fills and includes a
constant-hue radial fade with blur and group opacity.

The 41 focused cases in `tests/test_gradients.py` verify analytic samples,
transform/nesting semantics, all supported fill categories, compound holes,
mask/clip/opacity, shadow silhouette, transparent PNG over multiple backgrounds,
BGR/BGRA rendering, persistence, history, and successful/failed animation sampling.
Arrowheads retain the existing requirement for a visible stroke.
