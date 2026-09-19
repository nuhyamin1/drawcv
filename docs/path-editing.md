# Trimming and splitting paths

```python
middle = path.trim(0.2, 0.8)
prefix, suffix = path.split_at(0.4)

# Measure the cut using transformed distances, and detach in the same world pose:
middle = path.trim(0.2, 0.8, space="world", preserve_world_transform=True)

# Work with one original subpath:
piece = path.trim(0.1, 0.9, subpath=1, tolerance=0.01)
```

Progress uses the [measurement API's](path-measurement.md) normalized arc length.
Lines, quadratic/cubic Béziers and elliptical arcs keep their semantic command
types. Béziers use de Casteljau subdivision; arcs keep their ellipse, corrected
radii, rotation and sweep direction. Numerical integration locates cut parameters;
geometry is not flattened to polygons. `tolerance` has the same meaning and
numerical limits as the measurement API.

Both operations return independent paths with new IDs and no parent, layer or
scene. The source is unchanged. Local coordinates and the resolved local transform
are retained by default, matching `to_path()`. Add a result to the same parent
with `parent.add(result, preserve_world_transform=False)` to retain its pose.
`space="world"` selects the distance metric; it does **not** bake the coordinates.
`preserve_world_transform=True` retains the full ancestor transform on a detached
result, preserving object-space strokes and paints. Ancestor appearance is not
copied.

Own styles, clips, masks, effects, metadata, timing and render progress are copied.
The operation measures full geometry regardless of render progress. Subsequent
animation acts on the extracted geometry. The source fill and dash offset are
retained without adjustment: the result is an editable path, not a pixel-identical
crop of the original. A cut contour becomes open, receives open stroke caps, and
uses the normal open-path fill behavior. Dashes restart on each resulting contour
with the copied dash offset.

## Range and closure rules

- `trim(start, end)` requires `0 <= start <= end <= 1`. Reversed/wrapped ranges
  are rejected; to cross a closed contour's seam, extract the two ranges separately.
- Equal endpoints return an empty path. `[0, 1]` copies all selected geometry,
  including isolated moves and zero-length commands.
- Complete selected subpaths preserve their closure. Partially selected subpaths
  are open, even when their endpoints happen to coincide. A retained portion of
  a closing edge becomes an ordinary line.
- Disconnected subpaths contribute no bridging distance and remain disconnected.
  At a split exactly between them, the prefix ends on the earlier contour and
  the suffix starts on the later one; those two endpoints need not coincide.
- Zero-length segments add no distance. They are retained in fully copied
  subpaths, and omitted from cut subpaths. Non-full ranges of an entirely
  zero-length path are empty.
- `split_at(progress)` returns `(trim(0, progress), trim(progress, 1))`.
  Splitting at either endpoint yields an empty path on that side.
- `subpath` selects an original non-negative subpath index; other subpaths are
  excluded. Invalid progress, indices, spaces or tolerances raise `ValidationError`.

Run `python -m examples.path_editing` for a gallery. Results use existing Path
commands and round-trip through JSON and SVG without a schema change.
