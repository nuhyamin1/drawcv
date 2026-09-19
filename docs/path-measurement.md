# Measuring paths and following curves

Run `python -m examples.path_measurement` for a gallery of evenly spaced markers
with tangent directions along a cubic curve.

```python
distance = path.length(tolerance=0.1)
midpoint = path.point_at(0.5)
direction = path.tangent_at(0.5)  # Point containing a unit direction vector

world_midpoint = path.point_at(0.5, space="world")
second_contour_length = path.length(subpath=1)
```

Progress is normalized **arc length**, from 0 to 1, rather than a Bézier
parameter. A marker at 0.5 is halfway along the measured distance. Use
`math.degrees(math.atan2(direction.y, direction.x))` for a rotation angle in
DrawCV's clockwise, y-down coordinate system. Convert other shapes with
`shape.to_path()` before querying them.

All three methods accept `tolerance=0.1`, `space="local"`, and `subpath=None`.
Local measurements ignore transforms. World measurements include the path and
its ancestors; under nonuniform scaling, halfway in world distance can differ
from the transformed local halfway point. Stroke width, fills, clipping,
visibility and animation progress do not affect these full-centerline queries.

Lines have exact Euclidean lengths. Quadratic/cubic Béziers and elliptical arcs
use adaptive numerical integration of their speed. Positions invert arc length
and evaluate the semantic curve; tangents use its derivative, not a rendered
polygon edge. `tolerance` is a positive finite absolute integration-error target
in the selected coordinate units, distributed across commands. It is an
estimate, not a guaranteed bound; subdivision has a depth limit and floating-point
precision still applies. Tangent accuracy is not an angular tolerance.

## Boundaries and degenerate geometry

- Subpaths are traversed in order without adding distance across `MoveTo` gaps.
  `subpath` selects one original subpath by its non-negative integer index.
- Closing edges contribute distance, including an implicit `Subpath.closed`.
- At interior segment/subpath boundaries, the preceding nonzero segment wins.
  A disconnected contour therefore introduces a jump just after the boundary.
- `point_at(0)` and `point_at(1)` retain the first and last authored positions,
  including isolated moves. A closed path ends at its starting position.
- Zero-length segments add no distance. Endpoint tangents use the first/last
  nonzero segment. At a stationary curve point, a one-sided derivative is used:
  incoming except at the start. Cusps have no unique two-sided tangent.
- Empty paths have length zero; position and tangent queries raise
  `ValidationError`. An entirely zero-length path returns its first position
  but raises for tangent queries. A collapsed world transform follows these rules.
- Progress outside `[0, 1]`, invalid subpath indices, and nonfinite/invalid
  tolerances raise `ValidationError` rather than silently clamping.

Queries do not modify the path. Each call measures current geometry and transforms,
so edits are reflected immediately; repeated queries currently recompute the
measurement. No serialized fields or schema changes are required.
