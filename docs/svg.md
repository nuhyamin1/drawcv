# SVG export

SVG export preserves editable geometry and paint while embedding transparent PNGs
for rendering without a reliable native mapping. It requires only the base DrawCV
dependencies, describes the current frame, honors background alpha, and does not
modify scene objects or history.

```python
result = scene.save_svg("drawing.svg")
for fallback in result.fallbacks:
    print(fallback.entity, fallback.reason)

svg_text = scene.to_svg(strict=True)  # RenderError if any PNG fallback is needed
result = scene.export_svg()           # SVGExport(svg, fallbacks), without writing
result.save("another-copy.svg")
```

`SVGExporter(strict=False).render(scene)` is the equivalent renderer API.
`SVGExport` and `SVGFallback` are immutable descriptors exported from `drawcv`.
Strict export finishes before a destination file is opened, so a fallback error
leaves an existing file unchanged. Normal filesystem write errors remain filesystem
errors. Hidden and zero-opacity subtrees are omitted. Fallbacks are reported once
at each rasterized subtree root. A drawable that OpenCV cannot render raises
`RenderError` in either mode.

## Export matrix

| Retained feature | SVG representation |
| --- | --- |
| Line, rectangle, polygon, polyline | Editable world-space path |
| Path, quadratic/cubic BezierCurve | Editable M/L/Q/C/Z commands; compound fill rule retained |
| Circle, ellipse, rounded rectangle, arc | Editable polygonal path using the raster renderer's sampling |
| Constant-width freehand | Editable processed polyline; raw processing parameters remain in JSON |
| Solid, linear, radial fill | Native fill and gradient definitions, including stop alpha |
| Caps, joins, miter limit, dashes | Native stroke properties in scene-pixel units |
| Groups/layers, stacking, opacity | Nested isolated SVG groups in retained draw order |
| ClipRect, ClipPath | Native world-space clip paths on the owning group |
| Masks, blur, shadows | Cropped transparent PNG of the affected subtree |
| Hershey and retained font Text | Cropped transparent PNG, preserving existing shaping/layout |
| ImageObject | Cropped transparent PNG after DrawCV transforms/interpolation |
| Arrow and variable-width freehand | Cropped transparent PNG |

Exported text is not selectable or editable as text. No fonts are embedded in SVG.
Retained font rendering needs the typography extra on the exporting machine;
viewing the SVG does not. Glyph outlines, native image placement, SVG filter mapping,
and vector pressure ribbons remain possible extensions.

## Coordinates and compositing

Transforms are baked into path coordinates, including parent transforms and pivots.
Group structure remains, but authored transform decomposition does not. Stroke widths,
dashes, offsets and miter limits keep their screen-space meaning without depending
on `vector-effect`. Resizing the outer SVG scales the entire document; these units
apply at the original scene dimensions. Bezier commands remain curves, so SVG dash
lengths can differ slightly from DrawCV's flattened curve lengths.
Arc sampling targets 0.25 scene-pixel chord error, with the existing renderer's
two-degree minimum angular step. The target is not guaranteed for very large curves.

Object-space gradients use `userSpaceOnUse` and the object's world matrix as their
`gradientTransform`; world-space gradients omit that transform. Stop order, duplicate
offsets, padded end colors, sRGB channel interpolation, and separate stop alpha are
retained. See the [W3C paint server specification](https://www.w3.org/TR/SVG2/pservers.html).

Opacity is applied once by the enclosing object/group after its contents compose.
A fallback bakes its own root opacity, clip, mask and effects; enclosing vector
groups apply their opacity afterward. This prevents double multiplication and
preserves subtree isolation. Fallbacks use the premultiplied alpha renderer, encoded
as straight-alpha PNG data URLs at the scene's resolution and canvas clipping.
Upscaling cannot recover vector detail from them. No external image paths are emitted.

SVG antialiasing, curve flattening, integer-boundary coverage and pixel-center sampling
differ from OpenCV. `LINE_4`/`LINE_8` connectivity is not reproduced. Strict mode means
no raster fallback, not pixel identity. The tested SVG engine samples gradients at
half-pixel centers; DrawCV samples integer centers. Clip edges receive the viewer's
antialiasing. Use PNG export for exact raster reproduction.

## Persistence and animation

JSON remains the editable DrawCV format at schema 1.4. SVG is a one-way interchange
format: there is no SVG importer or SVG-to-DrawCV round trip. Paths can be edited in
SVG tools, but SVG does not retain every DrawCV object type or history/timeline state.
Groups carry `data-drawcv-id`; generated SVG IDs are unique and deterministic for the
same scene frame.

```python
from drawcv import SVGExporter
frame = scene.render_at_time(0.5, renderer=SVGExporter())
frame.save("frame.svg")
```

Temporal export uses the existing observational snapshot/restore path. Do not pass
`alpha=True` to this renderer: SVG always supports transparency. Progressive geometry
uses the existing slicing semantics and preserves its original transform context.
No animation elements are written to SVG.

## Verification and example

Run `python -m examples.svg_export` for the SVG, source JSON, fallback report and
DrawCV PNG. Installing `examples/svg-requirements.txt` also produces a resvg PNG and
side-by-side comparison. The gallery intentionally includes raster text labels and
one effects fallback.

The full Windows/Python 3.12.14 suite passes **473 tests**, including 36 SVG cases.
They cover editable commands, transformed gradient spaces/alpha, duplicate stops,
clips, dash units, isolated opacity, exact fallback PNGs, text, images, masks,
effects, strict failures, JSON/clone/history, and temporal restoration. Independent
tests use `resvg-py==0.5.0` and skip if that optional reference engine is absent.
The gallery was rendered and visually inspected. OpenCV/NumPy remain 5.0.0/2.5.3.
Native raster rendering code was not changed in this milestone.

Qt SVG was initially checked but cannot verify clipping because it
[does not support clipPath](https://doc.qt.io/qt-6/svgextensions.html).
The existing three-platform CI now installs resvg and renders this gallery alongside
the typography example. Remote Windows/macOS/Linux CI execution is still pending.
