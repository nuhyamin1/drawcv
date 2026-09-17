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

## Persistence, import and animation

JSON remains the primary editable DrawCV format at schema 1.7. DrawCV also provides a native,
high-fidelity retained SVG importer (`SVGImporter`) for supported SVG vector documents:

```python
from drawcv import Scene, SVGImporter

# Direct convenience constructors
scene = Scene.from_svg("<svg width='200' height='200'>...</svg>", strict=True)
scene = Scene.load_svg("vector_art.svg", strict=True)

# Advanced importer API with resource bounds and explicit viewport
importer = SVGImporter(strict=True, viewport=(400, 300))
result = importer.parse_file("input.svg")
scene = result.scene
```

### Strict import invariant

Anything accepted by `SVGImporter` in strict mode becomes genuine retained DrawCV objects, and subsequent export with `SVGExporter(strict=True)` succeeds with zero PNG raster fallbacks. Zero hidden rasterization is performed in the importer.

### Supported import subset (Milestone 1)

* **Shapes and paths**: `<path>` (M/m, L/l, Q/q, C/c, Z/z, compound paths, `fill-rule` nonzero and evenodd), `<rect>` (square and circular rounded rectangles with $rx=ry$), `<circle>`, `<ellipse>`, `<line>`, `<polyline>` (unfilled maps to `Polyline`, filled maps to open `Path`), `<polygon>` (maps to closed `Path`), `<g>`.
* **Transforms**: `matrix`, `translate`, `scale`, `rotate` (with pivot), `skewX`, `skewY`, and transform lists multiplied in SVG left-to-right composition order.
* **Style and cascade**: Presentation attributes and inline `style=""` declarations with full inheritance cascade; colors (`#hex`, `rgb()`, `rgba()`, 147 named CSS colors, `currentColor`, `none`, `transparent`); `display: none` subtree pruning; `visibility: hidden` with child `visibility: visible` override.
* **Stroke compatibility**: Ordinary strokes under uniform scale or reflection are normalized; strokes under anisotropic scaling or shear are strictly rejected (`SVG_STROKE_AFFINE_MISMATCH`) unless `vector-effect="non-scaling-stroke"` is declared.
* **Paint servers**: `<linearGradient>` and centered `<radialGradient>` in `objectBoundingBox` and `userSpaceOnUse`; context-aware percentage resolution in `userSpaceOnUse`; stop normalization (monotonicity, clamping, duplicate handling); template `href` inheritance; reference cycle detection; zero-size geometry bounding box handling.
* **Clipping**: Retained path clipping via `<clipPath>` (`userSpaceOnUse` and `objectBoundingBox`), clip transforms, and `clip-rule`. Supports a single exact geometry child from the `<path>`, non-rounded `<rect>`, `<polygon>`, and `<polyline>` subset with transform composition ($M_{\text{clipPath}} \cdot M_{\text{child}}$). Circular/elliptical/line clips, rounded rectangles, multiple clip children, and empty clip paths are strictly rejected (`SVG_UNSUPPORTED_CLIP_GEOMETRY`).
* **Compositing**: 12 supported `mix-blend-mode` values on shapes and groups. CSS `isolation: isolate` is strictly rejected.
* **Security & limits**: XML `DOCTYPE` and `ENTITY` declarations are forbidden (`SVG_SECURITY_VIOLATION`), external URLs in `href` are forbidden, and configurable limits protect against CPU exhaustion.
* **Deferred constructs**: `<mask>`, `<filter>`, `<text>`, `<image>`, `<pattern>`, `<use>`, `<symbol>`, percentage dimensions outside gradients, elliptical rounded corners ($rx \ne ry$), and path arc `A` commands are deferred to subsequent milestones.

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
