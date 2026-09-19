# Converting shapes to paths

`shape.to_path()` returns an independent, editable `Path` with a new ID.
Circles, ellipses, rounded rectangles and arcs retain exact elliptical arc
commands. Quadratic and cubic Bézier curves retain their control points.
Lines, rectangles, polygons, polylines and existing paths are also supported.
Other types raise `ValidationError`; this is not a text outlining or stroke
outlining API.

```python
path = circle.to_path()
scene.add(path)

# Detach from a transformed group while retaining the world pose:
path = child.to_path(preserve_world_transform=True)
```

The default retains local geometry and the resolved local transform, like a
detached clone. Add it to the same parent with
`parent.add(path, preserve_world_transform=False)` to retain its original pose.
The optional world transform includes ancestor transforms without baking them
into geometry, preserving object-space strokes and paints. Ancestor opacity,
clips and effects are not copied. Own styles, clips, effects, metadata and
timing are deep-copied. The original shape is not changed or removed.

Conversion freezes the current resolved transform, including its implicit pivot.
Later geometry edits therefore do not automatically recenter that pivot.
Progress uses the resulting Path's arc-length slicing semantics; an Arc's
angle-based progress is not the same on an ellipse.

Full turns use two half-ellipse commands. Open arcs retain no fill, chord arcs
close directly, and pie arcs connect to their center. Zero-sweep arcs retain
a zero-length segment so round stroke caps can still display a point.

SVG export uses the same exact curved geometry, including affine transforms,
instead of polygons sampled at the export resolution. Screen-space strokes
remain non-scaling; object-space strokes retain the SVG transform. Existing
Path JSON commands encode the result without a schema change. Singular or
near-singular transforms of world-baked arcs follow the existing SVG Path
export limitation and raise `RenderError`.
