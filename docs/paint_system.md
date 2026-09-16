# DrawCV Advanced Paint System

The **Advanced Paint System** extends DrawCV's retained-mode 2D graphics capabilities beyond basic solid fills and simple gradients to include gradient spread modes, deterministic full-turn conic (angular) gradients, paintable vector strokes, and affine-transformed bitmap image tiling (`ImagePaint`).

---

## 1. Architectural Model

### 1.1 Type Hierarchy & Separation of Concerns
In DrawCV, solid colors and spatial paints have fundamentally distinct coordinate semantics:
- `Color` represents an absolute, spatially invariant color value ($R, G, B, A$). It has no coordinate space, bounding box, or spatial transform. It remains the explicit high-performance fast path.
- `Paint` is the abstract base class for all spatially sampled paint types (`LinearGradient`, `RadialGradient`, `ConicGradient`, `ImagePaint`). Every `Paint` subclass specifies:
  - `space: "object" | "world"`: Whether sampling coordinates are evaluated relative to the drawable's local coordinate system via inverse entity world transform (`"object"`) or the global canvas coordinate system (`"world"`).
  - `transform: Transform`: An authoritative paint-local affine transformation matrix (including rotation, non-uniform scaling, translation, shear, and pivot).
- `PaintLike = Union[Color, Paint]` represents any fill or stroke paint.

### 1.2 Unified Coverage-to-Paint Compositing Pipeline
All spatial paints (fills and strokes) share a unified sampling and compositing pipeline in `OpenCVRenderer._composite_paint(...)`:
1. The primitive renders its geometric coverage into an antialiased 8-bit coverage mask (e.g. via `cv2.fillPoly` or `stroke_mask`).
2. The bounding box of the non-zero mask pixels is calculated to strictly limit sampling work.
3. The paint is sampled over the bounding box via `sample_paint(paint, world_matrix, width, height, origin=(x, y))`.
4. The premultiplied source buffer is modulated by the coverage mask and effective opacity.
5. Exact Porter-Duff source-over blending is executed onto the destination canvas buffer (isolated alpha surface or BGR canvas).

---

## 2. Gradient Spread Modes

`LinearGradient` and `RadialGradient` support the `spread` parameter with three standard modes:
- `"pad"` (default): Clamps parameter $t$ to $[0.0, 1.0]$. Outside the gradient span, the color of the nearest stop is replicated indefinitely.
- `"repeat"`: Wraps parameter $t$ periodically: $t_{wrapped} = t \pmod{1.0}$. The gradient repeats continuously across space.
- `"reflect"`: Alternates forward and backward in a triangle wave:
  $$m = t \pmod{2.0}, \quad t_{reflected} = \begin{cases} 2.0 - m & \text{if } m > 1.0 \\ m & \text{otherwise} \end{cases}$$

---

## 3. Conic (Angular) Gradients

`ConicGradient` defines a smooth full-turn ($360^\circ$) color sweep around a focal center point:

```python
conic = ConicGradient(
    center=Point(100, 100),
    stops=(
        GradientStop(0.0, Color.red()),
        GradientStop(0.25, Color.green()),
        GradientStop(0.5, Color.blue()),
        GradientStop(0.75, Color.yellow()),
        GradientStop(1.0, Color.red()),
    ),
    start_angle=0.0,       # 0 degrees aligns with +X axis (East)
    space="object",
    transform=Transform(rotation=45.0, pivot=Point(100, 100)),
)
```

### Coordinate and Angle Conventions
- Center Point: Defines the singularity point around which the angular sweep occurs. At the exact center pixel $(dx=0, dy=0)$, sampling defaults deterministically to stop 0.0.
- Angle Direction: Angles increase clockwise in standard 2D screen coordinates ($+X$ right, $+Y$ down):
  - $0^\circ$: East ($+X$ direction)
  - $90^\circ$: South ($+Y$ direction)
  - $180^\circ$: West ($-X$ direction)
  - $270^\circ$: North ($-Y$ direction)
- `start_angle`: Specifies an initial angular offset in degrees. The parameter is computed as $t = (( \text{degrees} - \text{start\_angle} ) \pmod{360.0}) / 360.0$.

---

## 4. Paintable Strokes

`StrokeStyle` accepts any `PaintLike` object via the `paint` property while preserving 100% backward compatibility with `color`:

```python
# Gradient stroke on a line
line = Line(
    start=Point(20, 20),
    end=Point(200, 20),
    stroke=StrokeStyle(
        paint=LinearGradient(Point(20, 20), Point(200, 20), STOPS),
        width=8.0,
    ),
)

# Solid color stroke (backward-compatible)
circle = Circle(
    center=Point(50, 50),
    radius=30,
    stroke=StrokeStyle(color=Color.blue(), width=2.0),
)
```

### Validation & Compatibility Rules:
- Specifying both `color` and non-Color `paint` simultaneously raises `ValidationError`.
- Accessing `stroke.color` on a solid stroke returns the `Color`. Accessing `stroke.color` on a gradient or image stroke raises `ValidationError("Gradient and image strokes have no single color; use stroke.paint")` (and accessing `fill.color` on non-solid fills raises `ValidationError("Gradient and image fills have no single color; use fill.paint")`).
- Setting `stroke.color = Color(...)` automatically updates `stroke.paint`.
- The shorthand `stroke.paint = ...` accepts either `Color` or `Paint`.

---

## 5. ImagePaint (Bitmap Textures)

`ImagePaint` provides retained bitmap texturing with repeat, pad, reflect, and finite clamping:

```python
paint = ImagePaint(
    image=texture_ndarray,           # uint8 NumPy array (BGR or BGRA)
    origin=Point(0.0, 0.0),          # Texture origin in paint coordinates
    scale=(1.0, 1.0),                # (scale_x, scale_y) mapping factors
    repeat="repeat",                 # "repeat" | "reflect" | "pad" | "none"
    space="object",                  # "object" | "world"
    interpolation=ImageInterpolation.LINEAR,
    opacity=1.0,
    transform=Transform(),
)
```

Supported interpolation modes for `ImagePaint` are `ImageInterpolation.LINEAR` (default), `ImageInterpolation.NEAREST`, `ImageInterpolation.CUBIC`, and `ImageInterpolation.LANCZOS`. `ImageInterpolation.AREA` is unsupported for spatial `ImagePaint` because OpenCV's `remap` does not support area resampling; configuring `AREA` on `ImagePaint` raises `ValidationError`. (`AREA` remains supported for `ImageObject` display scaling via `cv2.resize`.)

### Explicit Coordinate Composition Order
Sampling evaluates destination pixel centers through an authoritative transformation pipeline:
$$\text{Pixel Center } (X_w + 0.5, Y_w + 0.5) \xrightarrow{M_{entity}^{-1} \text{ if object-space}} (X_{obj}, Y_{obj}) \xrightarrow{M_{paint}^{-1}} (X_{paint}, Y_{paint}) \xrightarrow{\text{origin/scale}} (u, v) \xrightarrow{\text{repeat mapping}} (u_w, v_w) \xrightarrow{\text{texel centering}} \text{Texel } (u_w - 0.5, v_w - 0.5)$$

1. **Pixel Center**: Destination pixel grid integer coordinates $(X_w, Y_w)$ evaluate at pixel center $(X_w + 0.5, Y_w + 0.5)$ so that identity scale reproduces source texels exactly.
2. **Entity Space**: If `space == "object"`, coordinates are transformed by $M_{entity}^{-1}$ (the inverse of the entity's world matrix).
3. **Paint Transform**: Coordinates are transformed by $M_{paint}^{-1}$ (supporting translation, rotation, scale, and shear).
4. **Origin & Scale**: $u = (x - \text{origin.x}) / \text{scale.x}$ and $v = (y - \text{origin.y}) / \text{scale.y}$ with scale required to be positive finite numbers ($\text{scale.x} > 0, \text{scale.y} > 0$).
5. **Repeat Mode Mapping**:
   - `"repeat"`: $u_w = u \pmod W$, $v_w = v \pmod H$ with OpenCV `cv2.BORDER_WRAP`
   - `"reflect"`: Continuous triangle-wave reflection with `cv2.BORDER_REFLECT`
   - `"pad"`: Clamped to $[0, W]$ and $[0, H]$ with `cv2.BORDER_REPLICATE`
   - `"none"`: Finite image boundary with `cv2.BORDER_CONSTANT` transparent black $(0, 0, 0, 0)$, enabling continuous bilinear falloff without hard clipped edges.
6. **Source Texel Centering & Remap**: Continuous texel coordinates are converted to source texel centers ($map_x = u_w - 0.5, map_y = v_w - 0.5$) and sampled using OpenCV `cv2.remap` on premultiplied BGRA float32 buffers, preventing black halos at transparent boundaries.

---

## 6. Serialization & Schema 1.6 Migration

- Schema Version: Bumped from `"1.5"` to `"1.6"`.
- Migration Chain: `SchemaMigrator.register_migration("1.5", _migrate_1_5_to_1_6)` automatically handles schema forward migrations.
- Complete Affine Preservation: Paint transforms are serialized via `Transform.to_dict()` and `Transform.from_dict()`, ensuring non-decomposable matrices (such as shear produced via `Transform.from_matrix(...)`) are preserved across document serialization.
- Lossless PNG Raster Payloads: `ImagePaint` rasters are losslessly encoded to structured base64 PNG dictionaries.

---

## 7. SVG Export & Native vs. Fallback Semantics

`SVGExporter` strictly balances vector fidelity with visual correctness:

| Feature | SVG Representation | Rationale |
| :--- | :--- | :--- |
| `LinearGradient` (`spread="pad"`) | Native `<linearGradient spreadMethod="pad">` | Exact SVG standard support |
| `LinearGradient` (`spread="repeat"`) | Native `<linearGradient spreadMethod="repeat">` | Exact SVG standard support |
| `LinearGradient` (`spread="reflect"`) | Native `<linearGradient spreadMethod="reflect">` | Exact SVG standard support |
| `RadialGradient` (all spread modes) | Native `<radialGradient spreadMethod="...">` | Exact SVG standard support |
| `StrokeStyle.paint` (Gradients) | Native `stroke="url(#id)"` | SVG paths natively support gradient strokes |
| `ImagePaint` (`repeat="repeat"`, `interpolation=LINEAR`) | Native `<pattern><image>` | Exact userSpaceOnUse pattern tiling |
| `ImagePaint` (non-`LINEAR` interpolation) | `SVGFallback("image paint <interp> interpolation")` | SVG `<image>` has no portable non-linear filtering representation |
| `ImagePaint` (`repeat="reflect"`) | `SVGFallback("image paint reflect repeat")` | SVG `<pattern>` does not support mirrored tiling |
| `ImagePaint` (`repeat="pad"`) | `SVGFallback("image paint pad repeat")` | SVG `<pattern>` does not support clamped edge padding |
| `ImagePaint` (`repeat="none"`) | `SVGFallback("image paint none repeat")` | SVG `<pattern>` tiles infinitely in both axes |
| `ConicGradient` | `SVGFallback("conic gradient paint")` | SVG 1.1 / 2.0 has no native conic/mesh primitive |

In strict mode (`SVGExporter(strict=True)`), any fallback triggers a `RenderError`.

---

## 8. Documented Limitations

1. **Deferred Vector `PatternPaint`**: Tiling arbitrary retained vector scene subtrees (drawables/groups) is planned for a subsequent sub-milestone. Current tiling is provided via raster `ImagePaint`.
2. **Deferred `MeshGradient`**: Non-rendering mesh structures are excluded from the public API until complete tensor-patch evaluation is finalized.
3. **Discrete Stop Topologies in Animation**: `AnimationTrack` supports animating continuous scalar/vector properties of paints (`start`, `end`, `center`, `radius`, `start_angle`, `origin`, `transform`). Note that `ImagePaint.scale` is an affine 2-tuple property and is animated via `transform.scale_x` and `transform.scale_y`. Morphing between paints with differing stop counts or morphing across distinct paint classes (e.g. Linear to Conic) is not supported.
