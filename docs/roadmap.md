# pydrawcv roadmap

This roadmap follows a source, test, and example audit on 2026-09-13. The package
is `pydrawcv`; Python imports use `drawcv`. Work remains centered on editable
scene objects, stable drawable identity, and observational animation rendering.

## Verified starting point

The original 240 tests passed. The repository contains compound paths, affine
transforms, groups/layers, selection, freehand processing, isolated compositing,
images, clipping/masks, effects, JSON/history, progressive drawing, and animation.

The audit confirmed these priorities:

1. Stroke rendering delegated outlines to OpenCV primitives, ignoring cap/join
   enums. StrokeStyle, FillStyle, and Transform assigned values before validating.
   Transform also discarded its authoritative matrix before validation.
2. Multi-step transforms and relative placement could retain earlier changes on
   failure. History batches recorded partial work on exceptions; scene edits did
   not validate directly edited geometry before recording it.
3. Canvas requires `(height, width, 3)` uint8 BGR buffers. A transparent scene
   background does not produce output alpha, despite internal BGRA surfaces.
4. Text uses Hershey faces; fills are solid colors. There are no native gradient,
   pattern, configurable drawing blend-mode, SVG/PDF export, or textured-brush APIs.
5. The renderer allocates canvas-sized masks/surfaces and redraws scenes. No
   representative benchmark supports an optimization priority yet.

## Milestone 1 — Reliability and strokes (implemented here)

- Validate style/transform candidates before committing; preserve valid values and
  affine matrices on rejection. Roll back compound transforms, selection transforms,
  placement/distribution, failed scene edits, and recorded history batches.
- Share screen-space stroke tessellation across geometric outlines: three caps,
  three joins, miter-limit fallback, dash arrays/offsets, and variable-width ribbons.
- Join closed seams, restart dashes at compound subpaths, and preserve original
  transform context during progressive rendering. Composite a compound stroke's
  coverage once so crossing subpaths do not accumulate stroke opacity.
- Expand conservative bounds for square caps and miter tips. Align transformed
  circle fills with the same world contour used for their outlines.
- Write schema 1.2 and read older scenes through forward migration. Exercise new
  style values through copies, clones, history, persistence, and numeric animation.
- Replace phase-oriented introductory documentation with task-oriented entry points,
  a stroke reference, limitations, and a runnable rendered example.

The focused tests include pixel probes and image comparisons. Fourteen selected
regressions fail against the original source (caps, compositing, mutation, and
rollback). See [stroke semantics](strokes.md) and [reliability scope](reliability.md).

Milestone 1 verification: its suite passed **299 tests** (240 existing plus 59 new
parameterized cases). The stroke example was executed, exported as PNG/JSON, and
visually inspected for caps, joins, dash gaps, and pressure taper. Tests ran on
Python 3.12.14, OpenCV 5.0.0, and NumPy 2.5.3; lower supported dependency versions
were not separately exercised. The standard command is `python -m pytest -q`.
Test duration is not a representative rendering benchmark.

## Milestone 2 — Transparent output (implemented)

Implemented `alpha=True` on Canvas construction, scene rendering, temporal rendering,
frame iteration, and PNG sequences. Public BGRA is straight uint8; internal filtering
and composition are premultiplied float32. `Canvas.flatten(background)` provides an
explicit opaque export. This milestone kept JSON at 1.2 because output format is not scene state.

The opt-in path corrects legacy opacity/filtering/transform cases while leaving
default BGR behavior intact. Seven representative BGR frames match the pre-milestone
checkout byte-for-byte. Alpha tests include external PNG composition over white,
black, and color, hidden RGB through all interpolation modes, nested opacity,
mask/clip coverage, blur/shadows, and observational temporal rendering. The PNG
contact sheet was independently composited and visually inspected. See
[alpha contracts and limitations](transparency.md).

Milestone 2 verification: **339 tests pass** (299 existing plus 40 alpha cases),
with Python 3.12.14, OpenCV 5.0.0, and NumPy 2.5.3. PNG decode matches the exported
BGRA buffer exactly; external red-edge composites match pre-quantization opaque
BGRA renders within one channel value. Source/clone/history/temporal state checks
and the seven-frame BGR compatibility comparison also pass. Lower supported
dependency versions were not separately exercised.

Acceptance scenes must cover transparent backgrounds; overlapping translucent
strokes/fills; imported BGRA pixels with hidden RGB; nested group/layer opacity;
clips; absolute and bounds-fitted masks; blur/shadow; and PNG save/reload. Check
alpha and colors numerically, including fully transparent and fractional pixels,
and composite exported images over both light and dark backgrounds for visual QA.
Retain BGR regression coverage and make video alpha limitations explicit.

## Milestone 3 — Gradient fills (implemented)

`LinearGradient`, `RadialGradient`, and immutable RGBA `GradientStop` values are
first-class fill paints. Object-local/world coordinates, affine behavior, ordered
duplicate stops, padded spread, and straight-channel interpolation were fixed
before integration. FillStyle retains the solid `color=` API; the renderer samples
paint independently of coverage. Gradient scenes use premultiplied composition
in both output formats; solid-only BGR rendering retains its previous path.

Schema 1.3 preserves gradient type, geometry, space, stops, and alpha. Geometry
animation, clone/history, and observational sampling are covered. Gradient strokes,
focal points, repeat/reflect, stop-list morphing, patterns, and blend modes remain
outside this milestone. See the [gradient contract](gradients.md) and
[runnable gallery](../examples/gradient_fills.py).

Verification: **380 tests pass**, including 41 gradient cases. Seven representative
solid-only BGR frames match the pre-milestone checkout byte-for-byte. PNG tests
check external white/black/color composites within one channel value and preserve
constant-hue transparent edges. The gallery was rendered and visually inspected.
Environment: Python 3.12.14, OpenCV 5.0.0, NumPy 2.5.3; lower supported versions
were not separately exercised. Milestone 5B later limits paint sampling to coverage regions.

## Milestone 4 — Typography (4A evaluated; 4B integrated)

4A compared shaping/rasterization and paragraph stacks with licensed fonts and
measured images. 4B adds opt-in retained font Text, explicit embedded font assets,
HarfBuzz/FreeType rasterization, ICU paragraph/line bidi, deterministic fallback,
Thai dictionary wrapping, and separate advance/layout/ink metrics. The initial
bidi candidate failed a bracket case and was replaced with ICU before release.

Font descriptors and options round-trip through schema 1.4, cloning, undo/redo,
and numeric animation. Font scenes use premultiplied output in both BGR and BGRA.
Hershey remains the dependency-free compatibility path. Bounded caches reuse
fonts, short shaped runs, glyphs and local paragraph masks.

Verification: **437 tests pass** on Windows/Python 3.12.14; eight pre-existing BGR
frames remain byte-identical. Measured ink matches raster coverage, mixed bidi
positions agree with Qt, and JSON reload of the rendered example is pixel-identical.
A three-platform CI matrix is included; macOS/Linux execution is still pending.
The supported script set is Latin/Thai/Arabic; emoji, additional scripts, format
controls, glyph strokes, justification and variable-axis controls remain future work.

See the [public contract and limits](typography.md), [4A evidence](typography-evaluation.md),
and [retained example](../examples/retained_typography.py).

## Milestone 5 — Interchange, performance, artistic range

**5A SVG export is implemented.** Native world-space paths, gradients, stroke
properties, groups and clips remain editable. Text, images, pressure strokes,
arrows, masks and effects use reported subtree PNG fallbacks; strict mode rejects
them. Source persistence remains JSON 1.4; SVG import is not implemented. Resvg
independently verifies output; 473 tests pass locally, including 36 SVG cases.
The gallery was rendered and inspected. Cross-platform CI execution is pending.
See [SVG contracts and mapping](svg.md).

**5B performance measurement and targeted optimization are implemented.** Eight
deterministic workloads record raw timing samples, median/p95, separate memory and
profile passes, environment/source fingerprints and exact frame hashes. Coverage
regions bound gradient work and unbounded mask blending; freehand resolves transforms
once per call; explicit pivots avoid unused geometry queries. All benchmark frames
match the baseline exactly across two after-runs. The suite passes 508 tests.
See [measured gains, memory evidence and limits](performance.md). Remote CI and lower
supported dependency versions remain unverified.

## Milestone 6 — Generalized Ordered Effects Architecture (implemented)

Replaced hardcoded blur/shadow branches with an extensible, stage-by-stage ordered effect stack:
- **Decoupled Effect Model**: Retained `Effect` dataclasses decoupled from OpenCV raster processing (`drawcv/effects/effect.py`).
- **Exact Support Bounds**: Finite convolution kernel support (`kernel_size // 2` for Blur, `gaussian_pad_for_radius(radius)` for Shadow and Glow). Pre-own-effect geometric bounds (`get_bounds()`) decoupled from effect-input visual bounds (`get_effect_input_bounds()`).
- **Stage Execution**: Left-to-right processing on isolated surfaces before mask, clip, opacity, and blend modes (`drawcv/effects/executor.py`).
- **True Outer Glow**: `GlowEffect` isolates outer alpha contour, strictly preserving interior translucent pixels without color wash.
- **Color Manipulation**: Full color suite (`BrightnessContrastEffect`, `SaturationEffect`, `HueShiftEffect`, `GrayscaleEffect`, `SepiaEffect`, `ColorMatrixEffect` with alpha preservation).
- **Transactional Animation**: Property path traversal supporting sequence indices (`effects.0.color.a`) with automatic state rollback on validation failure.
- **Boundary Compatibility**: Spatial effects consistently standardize on transparent-zero spatial boundaries (`cv2.BORDER_CONSTANT`) across both BGR and BGRA modes, eliminating legacy BGR edge-pixel reflection (`cv2.BORDER_DEFAULT`) when objects touch canvas boundaries.
- **8 Benchmark Scenarios**: Extended benchmark suite (`benchmarks/scenes.py`) with reproducible workloads and gallery example (`examples/generalized_effects.py`).

1. **SVG first:** build an export matrix mapping retained geometry, fills, stroke
   semantics, transforms, groups, text, clips, and masks. Specify raster embedding
   for effects or unsupported objects and an explicit strict-error mode. Compare
   exports in an independent renderer and test that JSON reload, cloning and history
   preserve exported geometry. Keep screen-space strokes/dashes compatible under transforms.
2. **Measure performance:** store reproducible scenes for thousands of objects,
   long freehand inputs, deep transparency, masks/effects, and animation frames.
   Record hardware, Python/NumPy/OpenCV versions, scene size, warmup, median/tail
   frame time, allocation/peak-memory measurements, and image-equivalence checks.
   Profile before selecting bounds-sized buffers, geometry/mask caching, or dirty
   region rendering. Include dense dash patterns introduced in milestone 1.
3. **PDF and brushes:** evaluate PDF against diagrams/print use cases and the SVG
   mapping; decide text embedding, vector preservation, and raster fallback.
   Evaluate textured brushes against ink, pencil, and paint examples. Specify
   deterministic seeds, stamp spacing/orientation, pressure/velocity response,
   retained parameters, temporal behavior, and asset persistence before expanding
   the public API.

Each milestone ships runnable source plus rendered examples, API/convention
updates, persistence notes, focused regressions, and a clear capability table.
Publishing or pushing changes remains a separate user-requested operation.
