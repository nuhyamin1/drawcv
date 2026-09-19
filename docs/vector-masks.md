# Editable vector masks

```python
from drawcv import Circle, Point, Color, FillStyle, VectorMask, BoundingBox

artwork = Circle(center=Point(50, 50), radius=40,
                 fill=FillStyle(color=Color(255, 255, 255))).to_path()
shape.mask = VectorMask(artwork, bounds=BoundingBox(0, 0, 100, 100),
                        mode="alpha", space="object")
```

`VectorMask` accepts a `Path` or a `Group` containing paths/groups. Convert other
shapes with `to_path()`. Its artwork remains live and reusable, without being
reparented or modified when drawing. Changes appear on the next render. The
artwork's original parent/scene context is ignored; its own local transform,
paints, clips, opacity, markers and effects are retained. Text and other drawable
types must first be represented as paths. Nested masks inside mask artwork are
rejected, including recursive live edits.

## Alpha and luminance

- `mode="alpha"` (default): mask artwork alpha controls visibility. Opaque black
  reveals just as much as opaque white; transparent artwork hides.
- `mode="luminance"`: brightness multiplied by alpha controls visibility. Opaque
  white reveals, opaque black hides, and gray reveals partially. Luminance is
  evaluated in sRGB with weights `0.2126 R + 0.7152 G + 0.0722 B`.
- `opacity=1.0`: an additional multiplier applied once to composited mask artwork.

Linear/radial gradients create soft fades. For alpha mode, vary gradient-stop
alpha; a black-to-white gradient with constant alpha is a luminance mask instead.
Groups can combine multiple shapes and paint types. Ordinary source-over painting
applies within mask artwork: a black shape in luminance mode can hide a white
shape behind it, while black in alpha mode does not erase anything.

## Bounds and coordinates

`bounds` is a required positive `BoundingBox` in mask coordinates. Coverage is zero
outside it; artwork/effects are clipped to that region. Bounds do not automatically
fit the host, so moving partly offscreen does not stretch the mask.

`space="object"` follows the host's full geometry transform. `space="world"`
keeps the mask stationary in canvas coordinates. `transform=Transform()` adds an
affine transform within that space, with an implicit origin pivot of `(0, 0)`.
Layers have no geometry transform, so their object/world coordinates coincide.
Artwork keeps its usual stroke-space and paint-space rules. Singular mask
transforms produce zero coverage.

Host effects run first, followed by masking, final clipping and host opacity.
Mask artwork effects run before its bounds clip. A mask can only reduce visible
coverage; geometry bounds and hit-testing remain geometry-based rather than
tracking mask pixels. The existing raster `Mask` API, including fit/absolute
mapping and inversion, remains supported independently.

## Interchange

JSON schema **1.13** stores editable artwork, bounds and mask options. Older raster
masks migrate unchanged. Clones, history snapshots and JSON round trips copy
definitions independently; shared Python identity is not preserved through JSON.

SVG export emits native `<mask>` elements with explicit user-space bounds,
alpha/luminance mode and transformed vector content. Strict export works for
SVG-compatible artwork; effects and other unsupported features use the existing
reported raster fallback policy. Native SVG mask import is not implemented in
this milestone; JSON is the editable round-trip format.

Raster coverage is generated from a transparent render of the artwork at canvas
resolution. Minor antialiasing and gradient-sampling differences from SVG viewers
can occur at boundaries. Coverage is regenerated on each render so live edits
are reflected immediately.

Run `python -m examples.vector_masks` for alpha/luminance fade examples.
