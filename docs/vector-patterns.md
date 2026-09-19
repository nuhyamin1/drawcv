# Editable vector-pattern paints

```python
from drawcv import Circle, Point, Color, FillStyle, VectorPattern, Transform

dot = Circle(center=Point(6, 6), radius=3,
             fill=FillStyle(color=Color(25, 120, 160))).to_path()
pattern = VectorPattern(dot, width=12, height=12, spacing=(4, 4))
shape.fill = FillStyle(paint=pattern)
pattern.transform = Transform(rotation=30, pivot=Point(0, 0))
```

`VectorPattern` is a retained paint for both `FillStyle` and `StrokeStyle`.
Artwork is a `Path` or a `Group` containing paths/groups. Convert other geometric
shapes with `to_path()` before inserting them. Artwork remains editable and is
referenced live: changing a shared tile updates all users on the next render.
Painting does not reparent or mutate the source. Its parent/scene context is
ignored; its own local transform and its children are retained.

## Tile coordinates

- `width`, `height`: positive tile bounds starting at `(0, 0)`. Artwork outside
  these bounds is clipped and does not bleed into adjacent tiles.
- `spacing=(0, 0)`: non-negative horizontal/vertical gaps. The repeat period is
  `(width + spacing[0], height + spacing[1])`.
- `origin=Point(0, 0)`: tile-grid origin in paint coordinates.
- `space="object"` (default): the grid follows the host and its ancestors.
  `"world"` keeps the grid independent of host transforms.
- `transform=Transform()`: paint-space affine transform for rotation, scale,
  shear or translation. Its implicit pivot is `(0, 0)`.
- `opacity=1`: opacity applied once to the composited tile.

Tile artwork supports ordinary solid/gradient paints, local transforms,
group opacity, clipping and path fill rules. Stroke widths inside tiles are
interpreted in tile units and scale with the pattern, including strokes authored
with the default screen space. Object-space paints follow each artwork object;
world-space paints inside the tile refer to the detached tile coordinates.
The host can independently have clips, masks, effects and opacity.

This milestone excludes nested vector patterns, image paints, text, masks,
effects and marker attachments **inside tile artwork**. These are rejected with
`ValidationError`, including live edits that introduce recursion. Raster/image
tiles remain available through `ImagePaint`. Use multiple paths in a group for
layered vector motifs. The tile repeats in both directions; reflect/pad modes
are not part of this API.

## Rendering and persistence

Raster output renders a transparent tile at a resolution selected from the full
paint-to-world transform, then samples it periodically with premultiplied-alpha
interpolation. This avoids a fixed-resolution bitmap asset and interpolation
halos. Canvas and SVG rasterizers can differ at antialiased tile, shape and clip
edges. Live edits are visible immediately; tiles currently rebuild for each
paint sampling call rather than using a persistent cache.

The raster working limit is 8192 pixels per tile dimension and 16,777,216 total
tile pixels. Excessive dimensions or transforms raise `RenderError` rather than
silently degrading detail. Singular pattern transforms produce transparent paint.

JSON schema **1.12** retains the complete editable artwork and pattern settings;
older documents migrate unchanged. Copies and JSON round trips are independent,
and shared Python object identity is not preserved through JSON.

SVG export uses native `<pattern>` with vector content, a tile viewport and a
pattern transform. Strict export succeeds for SVG-compatible artwork; unsupported
paint features such as conic gradients use the existing reported fallback policy.
Native SVG pattern **import** is not implemented in this milestone—JSON is the
editable round-trip format for patterns.

Run `python -m examples.vector_patterns` for a gallery and SVG export.
