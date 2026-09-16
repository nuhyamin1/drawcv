# Public API guide

Use keyword arguments for shape-specific fields: dataclass shapes inherit common
Drawable fields, so positional argument order is not a shape-only constructor.
All names below are imported from `drawcv`. The linked source docstrings contain
full signatures; this guide groups the public entry points by task.

| Task | Entry points | Source |
| --- | --- | --- |
| Build/render a scene | `Scene(width, height, background)`, `OpenCVRenderer().render(scene, alpha=False)` | [Scene](../drawcv/scene.py), [renderer](../drawcv/renderer.py) |
| Read/export pixels | `Canvas.buffer`, `has_alpha`, `to_numpy()`, `flatten(background)`, `save(path)` | [Canvas](../drawcv/canvas.py) |
| Points/colors/bounds | `Point(x, y)`, `Color(r, g, b, a)`, `BoundingBox(x, y, width, height)` | [core](../drawcv/core) |
| Gradient fills | `FillStyle(paint=LinearGradient(...))`, `RadialGradient(...)`, `GradientStop(...)` | [gradient reference](gradients.md) |
| Style geometry | `StrokeStyle(...)`, `FillStyle(enabled=True, color=..., opacity=1)` | [stroke reference](strokes.md), [fill](../drawcv/styles/fill.py) |
| Straight geometry | `Line(start=..., end=...)`, `Polyline(points=..., closed=False)`, `Polygon(vertices=...)` | [shapes](../drawcv/shapes) |
| Closed shapes | `Rectangle(position=..., width=..., height=...)`, `RoundedRectangle(x=..., y=..., width=..., height=..., corner_radius=...)`, `Circle(center=..., radius=...)`, `Ellipse(center=..., radius_x=..., radius_y=...)` | [shapes](../drawcv/shapes) |
| Curves/markers | `BezierCurve(p0=..., p1=..., p2=..., p3=None)`, `Arc(center=..., radius_x=..., radius_y=..., start_angle=..., sweep_angle=..., closure=...)`, `Arrow(start=..., end=...)` | [shapes](../drawcv/shapes) |
| Compound paths | `Path(stroke=..., fill=..., fill_rule=...)`, `move_to`, `line_to`, `quadratic_to`, `cubic_to`, `close` | [Path](../drawcv/shapes/path.py) |
| Freehand capture | `StrokePoint(x, y, pressure=..., timestamp=..., velocity=...)`, `FreehandStroke(points=..., stroke=...)`, `add_point`, `get_processed_points`, `get_point_widths` | [freehand](../drawcv/shapes/freehand.py) |
| Transform objects | `move`, `rotate`, `scale`, `to_world`, `to_local`, `clone`, `anchor`, `get_bounds`, `Transform.from_matrix` | [Drawable](../drawcv/core/drawable.py), [Transform](../drawcv/core/transform.py) |
| Organize objects | `Group(children=...)`, `group.add/remove`, `scene.create_layer`, `scene.add(obj, layer=...)`, `scene.group/ungroup` | [Group](../drawcv/group.py), [Layer](../drawcv/layer.py) |
| Find/select | `scene.get(id)`, `find`, `find_by_name/tag/type/metadata`, `select`, `select_all`, `hit_test`, `hit_test_top` | [Scene](../drawcv/scene.py), [Selection](../drawcv/selection.py) |
| Relative layout | `align_left/right/top/bottom/center_x/center_y`, `align_centers`, `place_above/below/left_of/right_of`, `distribute_horizontally/vertically` | [positioning](../drawcv/positioning.py) |
| Edit with history | `scene.edit`, `batch`, `move_object`, `rotate_object`, `scale_object`, `restyle_object`, `undo`, `redo` | [reliability](reliability.md) |
| Persist scene | `scene.to_dict/to_json/save_json`, `Scene.from_dict/from_json/load_json` | [Scene](../drawcv/scene.py) |
| Images/text | `ImageObject(image=..., position=...)`, `Text(text=..., position=..., color=...)` | [image](../drawcv/shapes/image.py), [text](../drawcv/shapes/text.py) |
| Clip/mask/effects | Drawable `clip`, `mask`, `effects`; `ClipRect`, `ClipPath`, `Mask`, `BlurEffect`, `ShadowEffect`, `GlowEffect`, `BrightnessContrastEffect`, `SaturationEffect`, `HueShiftEffect`, `GrayscaleEffect`, `SepiaEffect`, `ColorMatrixEffect` | [effects](../drawcv/effects) |
| Animate/export | `Timing`, `scene.animate`, `sample`, `render_at_time`, `VideoRenderer` | [animation](../drawcv/animation) |

See [transparent output](transparency.md) for BGRA Canvas construction, temporal frames,
PNG export, channel order, and straight/premultiplied alpha semantics.

Useful enums: `CapStyle`, `JoinStyle`, `LineType`, `FillRule`, `ArcClosure`,
`ArrowHeadStyle`, `ImageInterpolation`, `BlurType`, `MaskMapping`, `TextAlignment`.
`FontFamily` contains Hershey faces such as `SIMPLEX`, `PLAIN`, `DUPLEX`, `COMPLEX`,
`TRIPLEX`, `COMPLEX_SMALL`, `SCRIPT_SIMPLEX`, and `SCRIPT_COMPLEX`; it does not select
system sans/serif fonts.

## Generalized Ordered Effects Stack

Entities (`Drawable`, `Group`, `Layer`) support an authoritative, ordered effect sequence via the `effects` attribute:

```python
from drawcv import BlurEffect, ShadowEffect, GlowEffect, Color, Scene, Rectangle, Point

rect = Rectangle(position=Point(40, 40), width=100, height=100)
# Effects execute strictly in list order: Glow -> Shadow -> Blur
rect.effects = [
    GlowEffect(blur_radius=10.0, color=Color(255, 230, 0)),
    ShadowEffect(offset_x=8.0, offset_y=8.0, blur_radius=4.0),
    BlurEffect(kernel_size=15, sigma=4.0),
]
```

- **Execution Order**: Effects execute left-to-right on an isolated surface before mask, clip, opacity, and blend mode compositing. Duplicate effects compound sequentially.
- **Bounds Propagation**: Visual bounds propagate stage-by-stage (`get_effect_bounds()`). Pre-own-effect geometric bounds (`get_bounds()`) are decoupled from effect-input bounds (`get_effect_input_bounds()`).
- **Color Manipulation**: Built-in `BrightnessContrastEffect`, `SaturationEffect` (Rec.709), `HueShiftEffect` (W3C hueRotate), `GrayscaleEffect`, `SepiaEffect`, and `ColorMatrixEffect` (4x5 RGBA matrix with alpha-preserving row 3).
- **Transactional Animation**: Property paths such as `effects.0.blur_radius` and `effects.0.color.a` support animated sampling with automatic rollback if validation fails.
- **Spatial Boundary Semantics**: Spatial effects (blur, shadow, glow) consistently treat space beyond the isolated surface or canvas boundary as transparent zero (`cv2.BORDER_CONSTANT`) in both BGR and BGRA modes. *Compatibility Note*: The legacy pre-milestone renderer used reflected borders (`cv2.BORDER_DEFAULT`) for BGR output and transparent borders (`cv2.BORDER_CONSTANT`) for BGRA. Under the standardized semantics, effects behave identically regardless of output format, meaning objects touching canvas edges fade into the canvas background rather than reflecting edge pixels.

## Draw a compound path

```python
from drawcv import Path, Point, StrokeStyle, FillStyle, Color, FillRule

outline = Path(stroke=StrokeStyle(width=4, dash_array=(12, 6)),
               fill=FillStyle(color=Color(220, 235, 250)),
               fill_rule=FillRule.EVEN_ODD)
outline.move_to(30, 30).line_to(200, 30)
outline.cubic_to(Point(260, 30), Point(260, 160), Point(200, 160))
outline.line_to(30, 160).close()
outline.move_to(70, 70).line_to(140, 70).line_to(140, 120).line_to(70, 120).close()
scene.add(outline)
```

## Capture pressure samples

```python
from drawcv import FreehandStroke, StrokeStyle, StrokePoint

ink = FreehandStroke(points=[StrokePoint(20, 40, pressure=.2)],
                     stroke=StrokeStyle(width=8), variable_width=True,
                     width_mode="pressure", min_width=2, max_width=14)
ink.add_point((80, 60), pressure=.8)
ink.add_point((150, 40), pressure=.3)
scene.add(ink)
```

The object preserves raw samples. Processing options include `simplification="rdp"`,
`smoothing="chaikin"`, and `interpolation="catmull_rom"` with their associated
parameters; see the freehand source and runnable example for combinations.

## Optional retained font text

`Text(..., fonts=[FontAsset.from_file(path)], font_size=32, wrap_width=400)`
opts into font shaping. `Text.measure()` distinguishes advance, layout, paragraph
and ink bounds. See [typography contracts and installation](typography.md).

## SVG export

`scene.save_svg(path)` returns an `SVGExport` containing the SVG text and a tuple
of `SVGFallback(entity, reason)` records. `scene.export_svg()` returns the same
result without writing; `scene.to_svg()` returns only its text. Set `strict=True`
to reject required raster fallbacks before writing a file.

For an observational animation frame, use
`scene.render_at_time(t, renderer=SVGExporter()).save(path)`.
SVG retains editable geometry, gradients and clips; text, effects and other
unsupported rendering use embedded transparent PNGs. See the
[export matrix, coordinate contract and limitations](svg.md).
