# Vector Path Boolean Operations

DrawCV supports production-quality, constructive 2D vector path boolean operations powered by the `skia-pathops` geometry engine.

Boolean operations operate directly on retained semantic vector `Path` geometry, preserving quadratic and cubic Bezier curves without rasterization masks, canvas-resolution dependencies, or polygon flattening.

---

## Supported Operations

DrawCV provides both ergonomic named methods on `Path` and a generic programmatic method:

```python
from drawcv import Path, PathBooleanOp

# Named convenience methods
merged = left.union(right)          # Union: Area(A) union Area(B)
overlap = left.intersection(right)  # Intersection: Area(A) intersect Area(B)
cutout = body.difference(hole)      # Difference: Area(A) \ Area(B)
exclusive = left.xor(right)         # Symmetric Difference (XOR): (A union B) \ (A intersect B)

# Generic programmatic method
result = left.boolean(right, PathBooleanOp.UNION)
result = left.boolean(right, "difference")
```

---

## Core Semantic Contract

### 1. Geometry First, Not Raster Compositing
Boolean operations operate on the **topological 2D filled area** represented by a `Path`.
* They do not perform pixel-level raster compositing (`cv2.bitwise_and` / `bitwise_or`).
* They do not combine color layers or blend modes.
* The operation is fully defined even when `path.fill is None`. The `fill` property controls raster painting, whereas boolean operations determine 2D planar topology.

### 2. Stroke Width Exclusion
Strokes are purely rendering outlines. A stroked curve or path is **not** expanded into an inflated boolean region based on `stroke.width`. Stroke-to-path conversion is a distinct feature and is outside the boolean geometry engine.

### 3. Open Subpath Semantics
In 2D vector graphics, fill rules treat unclosed contours as having an implicit closing segment connecting their final vertex back to their initial vertex. In DrawCV:
* An open `Subpath` with geometric commands is implicitly closed for boolean fill evaluation.
* The source `Path` is never modified; its commands remain unclosed.

---

## Bezier Curve Preservation & Simplification

DrawCV's semantic vector commands:
* `MoveTo`
* `LineTo`
* `QuadraticTo`
* `CubicTo`
* `Close`

are passed directly to the `skia-pathops` backend. DrawCV **never** globally flattens Bezier curves into dense polylines before boolean evaluation.

### Curve Order Reduction
The Skia PathOps backend may mathematically reduce curve order where exact simplification is possible (e.g. cubic to quadratic or quadratic to line). This is mathematically exact reduction, not lossy polygonization.

### Conic Conversion Boundary
DrawCV retains lines, quadratic Beziers, and cubic Beziers natively. Skia's internal rational conics (if emitted) are converted to standard quadratic Beziers using an explicit canvas-space tolerance:
```python
_CONIC_TO_QUAD_TOLERANCE = 0.25  # world pixel units
```

---

## Coordinate Spaces & Transform Contract

Boolean operations represent the geometry visible in the scene, accounting for local transforms and ancestor `Group` / `Layer` hierarchies.

```text
operand local geometry
        |
        v
operand + ancestor transforms (to_world)
        |
        v
world-space Bezier geometry
        |
        v
Skia PathOps boolean operation
        |
        v
world-space DrawCV Path
        |
        v
identity transform & detached state
```

### Result Coordinate Contract
The returned `Path` has:
* Semantic command coordinates in **world space**.
* `transform = Transform()` (identity: translation 0, rotation 0, scale 1).
* `_parent = None`, `_layer = None`, `_scene = None` (detached from both scene graphs).
* `clip = None`, `mask = None`, `effects = []` (detached from rendering effects).
* Default visibility (`visible=True`), full opacity (`opacity=1.0`), and unlocked state (`locked=False`).

---

## Fill Rules & Winding Normalization

DrawCV supports both `FillRule.EVEN_ODD` and `FillRule.NON_ZERO`.

During boolean evaluation:
1. Input paths respect their respective fill rules (`EVEN_ODD` or `NON_ZERO`) when translated into backend topology.
2. The boolean operation executes with `fix_winding=True, keep_starting_points=False`.
3. Skia PathOps normalizes the output contours into standard winding topology (`WINDING`), directing outer contours and inner holes with opposing orientations.
4. The resulting DrawCV `Path` is explicitly assigned:
   ```python
   result.fill_rule = FillRule.NON_ZERO
   ```
5. Nested holes (e.g. donuts) retain their transparent interiors across both `FillRule.NON_ZERO` and `FillRule.EVEN_ODD`.

---

## Styling Inheritance Policy

The result inherits styling from the primary (left) operand `a`:
* **Stroke**: Deep copy of `a.stroke` (or `None`).
* **Solid Fill**: Deep copy of `a.fill` (or `None`).
* **World-Space Gradients**: Deep copy of `a.fill.paint` with `space="world"` unchanged.
* **Object-Space Linear Gradient**:
  Transformed into an exact world-space `LinearGradient` using inverse-transpose affine gradient vector projection ($g = A^{-T} v / |v|^2$, $w = g / |g|^2$).
* **Object-Space Radial Gradient**:
  If the effective transform is a Euclidean similarity transform (translation, rotation, uniform scale, or reflection), transformed into an exact world-space `RadialGradient` with scaled radius ($r_{world} = r \cdot s$, where $s$ is the uniform scale factor magnitude).
  If the transform includes non-uniform scale or shear (which would turn a circle into an ellipse), falls back conservatively to `fill = None` under the geometry-first policy rather than introducing an unrepresentable circular distortion.

---

## Empty and Degenerate Operands

All empty operand combinations have defined, non-crashing behavior:
* `empty.union(B)` $\rightarrow$ returns $B$'s geometry in world space with `empty`'s style.
* `A.union(empty)` $\rightarrow$ returns $A$'s geometry in world space with $A$'s style.
* `empty.intersection(B)` $\rightarrow$ returns an empty `Path`.
* `A.intersection(empty)` $\rightarrow$ returns an empty `Path`.
* `empty.difference(B)` $\rightarrow$ returns an empty `Path`.
* `A.difference(empty)` $\rightarrow$ returns $A$'s geometry in world space with $A$'s style.
* `empty.xor(B)` $\rightarrow$ returns $B$'s geometry in world space with `empty`'s style.
* `A.xor(empty)` $\rightarrow$ returns $A$'s geometry in world space with $A$'s style.
* `A.difference(A)` $\rightarrow$ returns an empty `Path`.
* `A.xor(A)` $\rightarrow$ returns an empty `Path`.

Empty operations always return fresh detached `Path` objects; source operands are never returned directly.

---

## Interchange & Persistence

* **JSON Serialization**: Boolean results are standard retained `Path` objects. They serialize and deserialize using existing schema **1.4** without schema bumps:
  ```python
  import json
  from drawcv import Path, to_json

  data = to_json(result.to_dict())
  restored = Path.from_dict(json.loads(data))
  ```
* **SVG Export**: Strict SVG export (`SVGExporter(strict=True)`) exports boolean results natively as `<path d="...">` elements with zero raster fallback.
* **OpenCV Rendering**: Evaluated by the existing renderer via `evaluate_fill_rule_mask`. No boolean-specific rendering state is introduced.
