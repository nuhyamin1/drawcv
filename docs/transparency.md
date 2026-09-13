# Transparent rendering and PNG export

Transparent output is opt-in. The existing renderer call still returns a
three-channel uint8 BGR Canvas with the same legacy behavior.

```python
from drawcv import Scene, OpenCVRenderer, Color, Circle, Point, FillStyle

scene = Scene(320, 240, background=Color(0, 0, 0, 0))
scene.add(Circle(center=Point(160, 120), radius=60,
                 fill=FillStyle(color=Color(255, 0, 0, .5))))
renderer = OpenCVRenderer()
canvas = renderer.render(scene, alpha=True)
assert canvas.has_alpha
assert canvas.buffer.shape == (240, 320, 4)
canvas.save("transparent.png")
canvas.flatten(Color.white()).save("on-white.jpg")
```

`alpha=True` preserves the scene background's alpha; it does not make an opaque
background transparent. The default `Scene` background is opaque white. Choose
an alpha-zero background explicitly when you want a transparent surrounding area.

## Public API

| Entry point | Behavior |
| --- | --- |
| `Canvas(w, h, buffer=None, *, alpha=False)` | BGR by default; `alpha=True` allocates transparent BGRA or copies a matching BGRA buffer |
| `canvas.has_alpha` | Read-only format flag |
| `canvas.buffer` | Writable uint8 BGR or **straight-alpha BGRA** array |
| `canvas.to_numpy()` | Independent copy, same format/channel order |
| `canvas.clear(color)` | BGRA stores color and alpha; default BGR retains its existing alpha-ignoring behavior |
| `canvas.flatten(background)` | New BGR Canvas composited over an opaque `Color`; source unchanged |
| `renderer.render(scene, *, alpha=False)` | Full scene rendered to the requested format |
| `renderer.render_drawable(drawable, canvas)` | Uses the destination Canvas format; preserves the destination buffer reference |
| `scene.render_at_time(t, renderer=None, *, alpha=False)` | Non-destructive temporal render in the requested format |
| `VideoRenderer.render_frames(..., alpha=False)` | BGR frames by default; optionally BGRA |
| `VideoRenderer.render_image_sequence(..., alpha=False)` | BGRA sequences require a `.png` filename pattern |

The flag must be a boolean. A four-channel buffer supplied to the default Canvas
is rejected; channel count is not inferred. BGRA `Canvas.save` currently supports
PNG only and rejects other extensions before writing. Explicitly flatten for JPEG
or other opaque formats. Video/GIF export remains the existing BGR path; no
alpha-capable video codec integration is added in this milestone.

## Alpha model and compositing

Public pixels and PNG files use **straight** (unassociated) alpha. A half-transparent
red pixel is BGRA `(0, 0, 255, 128)`, not `(0, 0, 128, 128)`. Newly rendered pixels
whose exported alpha rounds to zero have RGB zeroed as well.

All BGRA-mode drawing, filtering, and isolated composition uses **premultiplied**
float32 intermediates: BGR values in [0, 255], multiplied by alpha in [0, 1].
Source-over is:

```text
out_premultiplied_color = source_color + destination_color * (1 - source_alpha)
out_alpha              = source_alpha + destination_alpha * (1 - source_alpha)
```

Color alpha, style opacity, and raster coverage scale a primitive's contribution.
Object opacity is applied once to its combined contents, including overlapping fill
and stroke. Group/layer opacity is applied once after its children are composited;
overlapping opaque children do not become darker just because they overlap.
Nested opacity boundaries multiply without replacing mask or source alpha.

Images are premultiplied **before both display resizing and affine warping**. Hidden
RGB at alpha zero therefore cannot contaminate neighboring visible pixels.
Cubic/Lanczos interpolation overshoot is clamped to valid premultiplied color and
alpha ranges. Blur filters all four premultiplied channels together. Shadows derive
coverage from source alpha and use their own color/opacity before merging behind
the object. Text uses the same alpha pipeline and respects ancestor affine
transforms for both glyphs and background plates; typography remains Hershey-based.

Arithmetic operates in stored RGB channel space, consistent with the existing
renderer. There is no implicit linear-light conversion, ICC processing, or gamma
correction. External comparisons must use the same color-space convention.

## Mask, clip, and effect order

The existing effect ordering remains: render contents; derive shadow from their
alpha; blur contents; merge shadow behind contents; apply mask; apply clip; apply
entity opacity; composite into the destination. This is a defined pipeline, not an
arbitrary ordered stack of filters. As before, it uses the first blur and first
shadow effect attached to an entity.

- Masks multiply both premultiplied color and alpha, including effects.
- `FIT_BOUNDS` fits to the full pre-effect world bounding box, then crops to the
  canvas. Moving an object partly offscreen does not stretch the mask.
- `ABSOLUTE` coverage is aligned to the **canvas/world origin**, with crop/pad
  behavior and optional inversion. It is not transformed with local object axes.
- Clips are hard-edged stencils applied after effects. Drawable/group clip geometry
  is local and transformed to world space; layer clip coordinates are world space.
- Blur uses transparent-zero boundaries in BGRA mode, avoiding reflected coverage
  at the canvas edge. Rendering is limited to the canvas domain: geometry outside
  the canvas is discarded before filtering. Off-canvas objects casting effects
  back into view are not implemented by this milestone.

## External composition and precision

Read exported PNG with `cv2.IMREAD_UNCHANGED` to retain its alpha channel:

```python
import cv2
import numpy as np

png = cv2.imread("transparent.png", cv2.IMREAD_UNCHANGED)
a = png[..., 3:4].astype(float) / 255
white = np.rint(png[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
black = np.rint(png[..., :3] * a).astype(np.uint8)
```

PNG stores eight-bit color and alpha. Rounding occurs when the render finishes;
external composition can differ by about one channel value from composition before
quantization. Repeated `render_drawable` calls on a public Canvas also quantize
between calls. Prefer a full retained-scene render for the most consistent result.

The regression suite checks the semitransparent red shape → transform/blur/group
opacity → PNG → external white/black/color composition case. Visible edge pixels
retain red RGB rather than developing a dark or pale fringe. It separately compares
images with different hidden RGB under alpha zero through every interpolation mode.

## Compatibility and persistence

Milestone 2 required no Scene fields or JSON version change: schema **1.2** stored background
and color alpha. Output format is a render/export choice, not document content.
Serialization, cloning, undo/redo, and non-destructive temporal sampling continue
to use the existing retained model. Default custom renderers with a `render(scene)`
signature remain supported by `render_at_time`.

Solid-only scenes preserve legacy BGR raster behavior. Scenes with gradient paints or font text
use the corrected premultiplied pipeline for BGR too; see [gradients](gradients.md).
Milestone 3 introduced schema **1.3** for paints; 4B writes **1.4** for font assets. The opt-in BGRA path also
corrects legacy image/text opacity, image resizing, object opacity isolation, text
ancestor transforms, and partially offscreen mask fitting. Consequently,
`render(scene, alpha=True).flatten(background)` is not promised to match legacy BGR
pixels in those cases. Use it when you want corrected alpha composition with an
opaque final output. Seven representative default BGR frames were compared against
the pre-milestone checkout and were pixel-identical.

Run `python -m examples.transparent_output` to regenerate the
[transparent PNG](../examples/output/transparent_output.png),
[external-composition preview](../examples/output/transparent_output_preview.png),
and [editable JSON](../examples/output/transparent_output.json).
Gradients, patterns, and blend modes remain separate work.
