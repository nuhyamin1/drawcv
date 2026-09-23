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
| Solid, linear, radial fill/stroke | Native paint definitions, stop alpha, paint transforms, and pad/repeat/reflect spread |
| ImagePaint tiles | Native embedded image pattern for repeat + linear interpolation; other modes use reported PNG fallback |
| Caps, joins, miter limit, dashes | Native stroke properties in the selected screen/object space |
| Paint order | `paint-order` attribute; paths whose markers are not painted last split into fill, marker and stroke elements |
| Groups/layers, stacking, opacity | Nested isolated SVG groups in retained draw order |
| ClipRect, ClipPath | Native world-space clip paths on the owning group |
| Vector masks | Native `<mask>` |
| Raster masks | Cropped transparent PNG of the affected subtree |
| Blur, shadow, glow, color effects, convolution | Native `<filter>` in effect order; see [effects as filters](#effects-as-svg-filters) |
| Sharpen, emboss, edge detection, noise, displacement, biased convolution | Cropped transparent PNG of the affected subtree |
| Text | Native text/tspan for the supported font-resolved subset; reported PNG fallback for Hershey, wrapping, and other unsupported layouts |
| ImageObject | Cropped transparent PNG after DrawCV transforms/interpolation |
| Shape labels | Native text inside the host group for single-line font labels; reported `<host id>-label` PNG fallback otherwise |
| Arrow | Native shaft and head shapes in the arrow's coordinates; circle heads under a non-uniform transform use a PNG |
| Variable-width (pressure/velocity) freehand | Native filled outline from `stroke_to_path()` |

Native exported text remains editable. No fonts are embedded in SVG, so the viewer
needs matching fonts for matching appearance. Raster fallback text preserves its
rendered appearance without viewer fonts. Retained font rendering needs the typography
extra on the exporting machine. Glyph outlines and native image placement remain
possible extensions.

## Effects as SVG filters

Each supported effect becomes filter primitives that compute the same operation as
its raster processor, chained in list order inside one `<filter>` with
`color-interpolation-filters="sRGB"`. SVG applies a filter before clipping, masking
and opacity, the same order DrawCV uses.

| Effect | Primitives |
| --- | --- |
| Brightness/contrast, saturation, grayscale, sepia, hue shift, color matrix | `feColorMatrix` with DrawCV's Rec.709 coefficients, not SVG's `saturate` shorthand |
| Gaussian blur | `feGaussianBlur` when the kernel covers ±3σ; otherwise two 1-D `feConvolveMatrix` passes with OpenCV's truncated weights |
| Box blur | Two 1-D `feConvolveMatrix` passes |
| Shadow | Offset and blur of the alpha, flood color, merged under the source |
| Glow | Blurred alpha minus source alpha (`feComposite` arithmetic), flood color, merged under |
| Convolution | `feConvolveMatrix` with `preserveAlpha`, kernel rotated 180° (SVG convolves, OpenCV correlates) |

The filter region is the effect bounds clipped to the canvas, because DrawCV runs
effects on a canvas-sized surface. Applied by resvg to DrawCV's own unfiltered
pixels, the filters match DrawCV within about 3/255 per channel. Fractional shadow
offsets differ by up to about 11/255 because renderers interpolate `feOffset`
differently, and sharpening kernels differ slightly at faint antialiased edges.
Convolution with a non-zero bias falls back to PNG: DrawCV adds the bias before
clamping, while SVG clamps first.

## Coordinates and compositing

Screen-space strokes export with world-space path coordinates and
`vector-effect="non-scaling-stroke"`. Object-space strokes export local path
coordinates with the complete world transform on the path element. Widths and
dashes stay in their authored space, and fill/stroke paint mappings are adjusted
independently. Outer clipping remains in world coordinates. Group structure remains,
but authored transform decomposition does not. Bezier commands remain curves, so SVG dash
lengths can differ slightly from DrawCV's flattened curve lengths.
Circles, ellipses, rounded rectangles and arcs export as exact arc commands.

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

DrawCV and SVG viewers share a pixel grid (pixel k spans [k, k + 1], paints sample
centers) and both compute area coverage, so exported shapes typically match resvg
within a few levels at edges; see [rasterization](rasterization.md). Viewers flatten
curves and sub-sample rows differently, and `LINE_4`/`LINE_8` connectivity is not
reproduced. Strict mode means no raster fallback, not pixel identity. Use PNG export
for exact raster reproduction.

## Persistence, import and animation

JSON remains the primary editable DrawCV format at schema 1.14 (with migration from older schemas). DrawCV also provides a native,
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

### Supported import subset (Milestones 1 & 2)

* **Shapes and paths**: `<path>` (M/m, L/l, H/h, V/v, C/c, S/s, Q/q, T/t, A/a, Z/z, compound paths, `fill-rule` nonzero and evenodd), `<rect>` (square and non-uniform rounded rectangles with exact `EllipticalArcTo`), `<circle>`, `<ellipse>`, `<line>`, `<polyline>` (unfilled maps to `Polyline`, filled maps to open `Path`), `<polygon>` (maps to closed `Path`), `<g>`. Path numeric lexer strictly conforms to SVG grammar (supporting trailing decimals such as `1.`, `-2.`, `1.e2` and rejecting invalid separators).
* **Elliptical arc command (`EllipticalArcTo`)**: Exact native retained representation for elliptical arcs with parameters `radius_x`, `radius_y`, `x_axis_rotation`, `large_arc`, `sweep`, and `end`. Canonical JSON serialization discriminator is `"type": "elliptical_arc_to"`. Evaluated via world-space adaptive subdivision (chord midpoint test $\le \text{tol}$) under arbitrary affine transforms, exact progressive slicing via numerical integration over $[\theta_1, \theta_1 + \Delta\theta]$, and exact SVD-based affine transformation. Singular or near-singular transforms are strictly rejected at import and export.
* **`<use>` and `<symbol>` instantiation**: Full 4-tier retained hierarchy:
  1. `UseHostGroup`: hosts `T_use @ T_xy`, use-level opacity, mix-blend-mode, and use-level `clip-path`.
  2. `TargetHostGroup`: hosts referenced `<svg>`/`<symbol>` authored transforms, opacity, blend mode, and element-level `clip-path`.
  3. `ViewportGroup`: hosts viewport overflow clipping (`ClipRect(0, 0, W, H)` for `overflow: hidden` or UA defaults) in local viewport coordinates.
  4. `ViewBoxGroup`: hosts `M_viewBox` coordinate mapping and instantiated children.
  Non-viewport referenced targets (e.g. `<path>`, `<rect>`, `<g>`) instantiate directly into a single `UseHostGroup` without superfluous viewport layers.
* **Viewport overflow and coordinate spaces**: Viewport clipping for nested `<svg>`, `<symbol>`, and root `<svg>` operates in viewport coordinates outside the viewBox-transformed group (`RootViewportGroup` outside `RootViewBoxGroup`). UA default `overflow: hidden` is applied for nested `<svg>` and `<symbol>` unless explicitly set to `visible`. Non-inherited `overflow` on `<use>` does not leak into referenced viewports.
* **Clipping**: Retained path clipping via `<clipPath>` (`userSpaceOnUse` and `objectBoundingBox`), clip transforms, and `clip-rule`. Supports exact geometry children from `<path>`, `<rect>`, rounded `<rect>`, `<circle>`, `<ellipse>`, `<polygon>`, and `<polyline>` with transform composition ($M_{\text{clipPath}} \cdot M_{\text{child}}$). For `objectBoundingBox`, percentage geometry resolves in normalized `[0, 1]` viewport space before bounding-box mapping. Singular transforms on drawables with elliptical clips are strictly rejected.
* **Transforms**: `matrix`, `translate`, `scale`, `rotate` (with pivot), `skewX`, `skewY`, and transform lists multiplied in SVG left-to-right composition order.
* **Style and cascade**: Presentation attributes and inline `style=""` declarations with full inheritance cascade; colors (`#hex`, `rgb()`, `rgba()`, 147 named CSS colors, `currentColor`, `none`, `transparent`); `display: none` subtree pruning; `visibility: hidden` with child `visibility: visible` override.
* **Stroke compatibility**: Ordinary strokes retain `space="object"`, their authored width/dashes, and affine behavior including anisotropic scaling, shear, and reflection. `vector-effect="non-scaling-stroke"` retains `space="screen"`.
* **Paint servers**: `<linearGradient>` and centered `<radialGradient>` in `objectBoundingBox` and `userSpaceOnUse`; context-aware percentage resolution in `userSpaceOnUse`; stop normalization (monotonicity, clamping, duplicate handling); template `href` inheritance; reference cycle detection; zero-size geometry bounding box handling.
* **Compositing**: 12 supported `mix-blend-mode` values on shapes and groups. CSS `isolation: isolate` is strictly rejected.
* **Security & limits**: XML `DOCTYPE` and `ENTITY` declarations are forbidden (`SVG_SECURITY_VIOLATION`), external URLs in `href` are forbidden, reference cycles in `<use>` are detected and rejected (`SVG_REFERENCE_CYCLE`), and configurable limits protect against CPU/memory exhaustion (`max_use_instances`, exact `max_expanded_elements`, `max_reference_depth`, `max_file_bytes`, `max_elements`, etc.).
* **Text**: Supported `<text>`/`<tspan>` content imports as retained text with explicit font resolution. Advanced text placement and text-on-path remain restricted.
* **Root properties**: Root opacity is applied once to the composed subtree. Root transforms wrap the viewBox mapping, and root display and clip-path are retained in rendering behavior.
* **Paint order**: `paint-order` is retained on imported shapes, including inline style declarations and inheritance; `initial` resets it to normal. Malformed values are rejected with `SVG_INVALID_PAINT_ORDER`.
* **Deferred constructs**: `<mask>`, `<filter>`, `<image>`, and `<pattern>` import remain deferred. Export and import capabilities are not identical.

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
