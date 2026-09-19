# Reusable path markers

```python
from drawcv import Path, Point, Marker, FillStyle, Color, StrokeStyle

arrow = (Path(fill=FillStyle(color=Color(220, 60, 60)))
         .move_to(0, 0).line_to(-4, -2).line_to(-4, 2).close())
marker = Marker(arrow, ref=Point(0, 0), orient="auto-start-reverse")
path = Path(stroke=StrokeStyle(width=4)).move_to(30, 60).line_to(150, 60)
path.marker_start = marker
path.marker_end = marker
```

Attach `Marker` values to `Path.marker_start`, `marker_mid`, or `marker_end`.
Other vector shapes can be converted with `to_path()`. Artwork is a compound
`Path`, including curves, holes, fills, strokes, paints and effects. Convert a
circle or polygon with `to_path()` to use it as artwork. Nested markers and group
artwork are not supported. The marker references its artwork live, so editing
one shared marker updates all paths using it; drawing never reparents or mutates
that artwork. The artwork's own local transform is applied, but its scene/parent
context is ignored.

## Placement and size

- `ref=Point(0, 0)` is the attachment point in marker coordinates (after the
  artwork's own local transform).
- `orient="auto"` follows the local curve tangent. `"auto-start-reverse"` adds
  180 degrees at starts only. A finite number specifies a clockwise angle.
- `units="stroke"` multiplies artwork size by the host's numeric stroke width,
  or one if it has no stroke. `units="user"` is independent of stroke width.
- `size=1.0` is an additional positive scale multiplier.

All placements follow the host's full geometry transform. User units are fixed
relative to stroke-width changes, not fixed screen pixels. This also applies
when the host uses a screen-space stroke: markers still follow the geometry
transform. Artwork strokes keep their own screen/object stroke-space setting.
Choose object-space artwork strokes when their thickness should scale with the
marker. Colors are explicit; SVG `context-stroke` and `context-fill` inheritance
are not implemented.

Start/end markers appear on each subpath; mid markers appear at **authored
segment junctions**, not flattened curve samples. Mid orientation bisects the
incoming/outgoing directions. Zero-length segments borrow the nearest usable
direction, or use zero degrees if all directions vanish. Closed contours have
both start and end markers at the closure point; move-only subpaths have none.
At an exact reversal, the clockwise-coordinate angle convention chooses the
negative half-turn bisector.

## Rendering and interchange

Markers paint after the host fill/stroke, in start/mid/end order per subpath.
They participate in host opacity, clips, masks and effects. Artwork can have its
own appearance and effects. Bounds include marker extents; hit-testing includes
marker artwork with the existing Path selection tolerance. As with other Path
hit tests, this is geometry selection, not per-pixel clip/effect coverage.
Progressive rendering places markers on the currently sliced path.

Cloning, `to_path()`, trimming and splitting copy marker definitions independently.
Trimmed paths receive markers at their new endpoints. Measurement, booleans,
offsets and stroke-to-path operate on the host geometry and exclude markers.

JSON schema **1.11** stores artwork and options on each attachment; older documents
migrate with no markers. JSON retains editable definitions but does not preserve
shared Python object identity between attachments. SVG exports markers as editable
vector paths inside the host's compositing group, including strict vector export
when the artwork permits it. Reimport retains their appearance as separate paths,
not marker attachments. Native SVG `<marker>` import and reusable `<marker>`
definitions on export are not part of this milestone. Artwork requiring a raster
fallback is reported, and strict export rejects it.

Run `python -m examples.path_markers` for a gallery.
