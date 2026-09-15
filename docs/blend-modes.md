# Retained Blend Modes and Advanced Compositing

DrawCV supports first-class retained blend modes across drawables, groups, and layers. All blend modes follow the [W3C Compositing and Blending Level 1](https://www.w3.org/TR/compositing-1/) specification and ISO 32000-1 (PDF) standard, providing consistent behavior between OpenCV raster rendering and SVG export.

```python
from drawcv import Scene, OpenCVRenderer, Color, Rectangle, Circle, Point, FillStyle, BlendMode

scene = Scene(400, 300, background=Color.white())

# Backdrop shape
backdrop = Rectangle(
    position=Point(50, 50),
    width=200,
    height=200,
    fill=FillStyle(color=Color(255, 60, 60))
)
scene.add(backdrop)

# Foreground shape with MULTIPLY blend mode
fg = Circle(
    center=Point(200, 150),
    radius=90,
    fill=FillStyle(color=Color(60, 120, 255)),
    blend_mode=BlendMode.MULTIPLY
)
scene.add(fg)

renderer = OpenCVRenderer()
canvas = renderer.render(scene)
canvas.save("blended.png")
```

---

## Public API

### BlendMode Enum

```python
from drawcv import BlendMode
```

| Enum Member | String Alias | CSS `mix-blend-mode` | Description |
| :--- | :--- | :--- | :--- |
| `BlendMode.NORMAL` | `"normal"` | `normal` | Standard source-over compositing (default). |
| `BlendMode.MULTIPLY` | `"multiply"` | `multiply` | Multiplies backdrop and source colors. Result is always darker. |
| `BlendMode.SCREEN` | `"screen"` | `screen` | Multiplies complements of colors. Result is always lighter. |
| `BlendMode.OVERLAY` | `"overlay"` | `overlay` | Multiplies or screens based on backdrop brightness. Preserves highlights and shadows. |
| `BlendMode.DARKEN` | `"darken"` | `darken` | Selects darker component: $\min(C_b, C_s)$. |
| `BlendMode.LIGHTEN` | `"lighten"` | `lighten` | Selects lighter component: $\max(C_b, C_s)$. |
| `BlendMode.COLOR_DODGE` | `"color_dodge"`, `"color-dodge"` | `color-dodge` | Brightens backdrop to reflect source color. |
| `BlendMode.COLOR_BURN` | `"color_burn"`, `"color-burn"` | `color-burn` | Darkens backdrop to reflect source color. |
| `BlendMode.HARD_LIGHT` | `"hard_light"`, `"hard-light"` | `hard-light` | Multiplies or screens based on source brightness. Equivalent to overlay with swapped inputs. |
| `BlendMode.SOFT_LIGHT` | `"soft_light"`, `"soft-light"` | `soft-light` | Soft ambient lighting based on source brightness. |
| `BlendMode.DIFFERENCE` | `"difference"` | `difference` | Subtracts darker color from lighter color: $\lvert C_b - C_s \rvert$. |
| `BlendMode.EXCLUSION` | `"exclusion"` | `exclusion` | Lower-contrast variant of difference: $C_b + C_s - 2 C_b C_s$. |

String values can be assigned directly to `.blend_mode` attributes on shapes, groups, and layers:
```python
shape.blend_mode = "multiply"
shape.blend_mode = "color-dodge"
shape.blend_mode = BlendMode.OVERLAY
```

---

## Compositing Architecture

### 1. Pure Blend Functions vs. Alpha Compositing

In accordance with W3C specifications, blend formulas operate on normalized, unpremultiplied color components:

$$C_s = \frac{c_s}{\alpha_s}, \quad C_b = \frac{c_b}{\alpha_b}$$

where $C_s, C_b \in [0, 1]$.

The general W3C compositing equation calculates the premultiplied output color $c_o$ and alpha $\alpha_o$:

$$c_o = (1 - \alpha_b) c_s + (1 - \alpha_s) c_b + \alpha_s \alpha_b B(C_b, C_s)$$

$$\alpha_o = \alpha_s + \alpha_b (1 - \alpha_s)$$

For opaque backdrops (such as default BGR `Canvas` where $\alpha_b = 1.0$), this simplifies directly to:

$$c_o = (1 - \alpha_s) C_b + \alpha_s B(C_b, C_s)$$

### 2. Local Isolated Compositing

DrawCV isolates rendering locally rather than forcing an entire scene through a global alpha pipeline:

* An entity with `blend_mode != BlendMode.NORMAL` sets `_needs_isolated_compositing(entity) -> True`.
* The entity's geometry is rendered into a **full-canvas float32 isolated surface**, because DrawCV primitives and transforms currently operate directly in world coordinates.
* Only the subsequent filter evaluation, clipping, blend, safe unpremultiplication, and destination-composite calculations are restricted to the visual ROI (the entity's bounding box).
* Pipeline stages are strictly evaluated in order:
  1. **Effects**: Content-aware blur, drop shadows.
  2. **Mask**: Alpha mask attenuation.
  3. **Clip**: Vector path and rectangle clipping.
  4. **Opacity**: Entity 4-channel opacity scaling.
  5. **Destination Composite**: Blended into destination buffer via `composite_blend(...)`.

### 3. Bit-for-Bit Preservation of NORMAL Mode

DrawCV guarantees bit-for-bit numerical equivalence for `BlendMode.NORMAL` against all existing reference frames and regression baselines. The fast-path for `NORMAL` reuses the exact dtypes, operations, clipping, and rounding semantics of the original source-over renderer.

---

## SVG Export

Blend modes export as CSS `mix-blend-mode` properties on the containing `<g>` elements:

```xml
<g id="drawcv-1" data-drawcv-id="c1" style="mix-blend-mode: multiply;">
  <path d="..." fill="rgb(60,120,255)" />
</g>
```

When an entity requires raster fallback (e.g. due to blur or complex masks), the fallback `<image>` is nested inside the `<g style="mix-blend-mode: ...">` wrapper, ensuring the rasterized element is blended into the SVG backdrop without double application of the blend mode.

---

## Documented Limitations

1. **Animation Limitation**:
   `blend_mode` is a discrete enumeration property and cannot participate in continuous numeric or color interpolation.
   Any attempt to animate `blend_mode` via `scene.animate(...)` or `AnimationTrack(..., property_path="blend_mode")` is rejected with a clear `ValidationError`. Discrete/step animation tracks may be introduced in a future milestone.
2. **Internal Normalization Helper**:
   `coerce_blend_mode()` is an internal helper in `drawcv.core.enums` and is not exported from the public top-level `drawcv` package. Users should assign enum members (`BlendMode.MULTIPLY`) or string aliases (`"multiply"`).
3. **Alpha Video Codecs**:
   Video rendering continues to output BGR frames; alpha-capable video codecs remain out of scope.
4. **Full-Canvas Isolated Buffers**:
   Non-normal blend modes allocate a full-canvas float32 isolated surface for the entity subtree because primitives and transforms evaluate in world coordinates. Although compositing arithmetic is tightly restricted to the visual ROI, allocating full-canvas isolated surfaces is a current performance characteristic; ROI-sized isolated buffers represent a possible future architectural optimization.
