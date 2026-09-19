# DrawCV 0.9.1: SVG drawing fidelity review

Reviewed 2026-09-19 at commit `4501ca6` (`chore: release version 0.9.1`).
The checkout was clean before this review. This is a source review of that checkout,
not a verification that the published PyPI wheel contains identical files.

## Follow-up implementation status

The five confirmed issue families below have now been addressed in the working
checkout. Nonzero topology is resolved with the existing PathOps dependency and
shared by rendering, retained clipping, and path/clip containment. Root SVG
opacity and transforms are retained, transformed text no longer transforms its
world-space clip again, and supported boolean gradient conversion preserves
paint-local transforms and spread. Unsupported paint orders now raise
`SVG_UNSUPPORTED_PAINT_ORDER`; configurable drawing order is still future work.

Verification: **1,196 checks pass** (1,189 regular tests plus the seven review
probes), including 42 new regular regression cases. The audit probes now assert
the explicit rejection contract for unsupported paint order. The command was
`python -m pytest -q tests docs/audits/svg_review_repros.py` on Windows/Python 3.12.
The original review and original test counts below are historical findings.

Root transform/viewBox composition follows
[SVG2 coordinate-system ordering](https://www.w3.org/TR/SVG/coords.html#ViewBoxAttribute)
and was independently checked in headless Chrome: a 2x viewBox and root translation
(20, 10) place local (5, 5) at (30, 20). resvg-py 0.5.0 instead places it at (50, 30),
so that test uses an equivalent explicit group hierarchy as its raster oracle.

Scope limits: no package version bump or publication; no claim of full SVG parity.
Object-space radial gradients still require a similarity transform for boolean
conversion. Conic/image paint and object-space stroke paint conversion are not
expanded by this patch. Equivalent floating-point gradient evaluation may differ
at exact repeat discontinuities; sampled field regression tests avoid those seams.
Cross-platform and lower-bound dependency runs remain outstanding.

## Assessment

DrawCV has a substantial retained drawing foundation. Keep it. The most valuable
next work is making combinations of existing features behave consistently, then
expanding vector authoring and interchange. Passing a strict SVG export currently
means no reported raster fallback; it does not guarantee correct visual semantics.

The existing suite passed **1,147 tests in 14.89 seconds** on Windows, Python 3.12,
with the existing project dependencies, typography extras, and resvg available.
Seven additional review probes fail, confirming five issue families below.
These probes are in [svg_review_repros.py](svg_review_repros.py), outside normal
test discovery. Run `python -m pytest -q docs/audits/svg_review_repros.py` with the
project's dev/typography dependencies and `resvg-py` installed. They assert the
desired behavior and intentionally fail on the reviewed release.

No production code was changed. This is a targeted review, not an exhaustive
audit of every effect, animation, history operation, or supported platform.

## Confirmed correctness findings

### 1. High: nonzero fill is incorrect for self-intersecting paths

Location: `drawcv/core/geometry_utils.py:443-456`; invoked by
`drawcv/renderer.py:529`.

`NON_ZERO` assigns one winding sign to an entire contour based on its signed area,
then uses `cv2.fillPoly` for that contour. This cannot represent winding numbers
that vary across the interior of a self-intersecting contour.

Reproduction: `M50 5 L76 86 L7 36 L93 36 L24 86 Z`, filled red with
`fill-rule="nonzero"`. At (50, 50), resvg produces opaque red; DrawCV produces
fully transparent black. Strict import and export both accept it. This affects
ordinary vector artwork, not just obscure XML features.

Fix: use actual winding evaluation, or resolve contours through a robust geometry
backend before rasterization. Keep evenodd separate. Include self-intersections,
repeated traversal, opposite-winding overlaps, clipping, and agreement between
rendering, booleans, and hit testing in regression coverage.

### 2. High: root SVG opacity and transform disappear

Location: `drawcv/svg_import.py:3821-3848`.

The root is resolved as a style source and viewBox wrapper, but its own opacity
and transform are not materialized. The root's non-inherited opacity is reset
when resolving children; its `transform` never enters their transform context.

Reproductions: a red rectangle under `<svg opacity="0.5">` renders with alpha
255 instead of 128. A rectangle under `<svg transform="translate(40 40)">`
remains at the origin. Both pass strict import/export.

Fix: represent the root's applicable compositing/transform properties in an actual
retained node, with deliberate ordering relative to viewBox and viewport clipping.
Also audit root display, clip-path, and blend handling; those were not separately
verified by these probes.

### 3. High: transformed text exports its clip in the wrong coordinate space

Locations: `drawcv/svg.py:209` and `drawcv/svg.py:269`.

The exporter builds the clip in world coordinates, then puts the text's world
transform on the same group that owns that clip. SVG applies the group transform
to the clip too, effectively transforming the already transformed clip again.

Reproduction: HELLO at font size 40, translated 80px right, with a local 45px-wide
clip. The original resvg ink spans x=84..124. The exported SVG's ink spans
x=160..201, showing the wrong portion of the text. Strict export reports success.

Fix: attach the world-space clip to an untransformed outer wrapper and transform
the text in an inner node, or express both in the same local coordinate system.
Cover translation, rotation, nonuniform scale, ancestor transforms, and path clips.

### 4. Medium: strict import silently ignores paint-order

Location: `drawcv/svg_import.py:1216` (style resolution) and the fixed fill-then-
stroke rendering in `drawcv/renderer.py:526-532`.

Reproduction: red rectangle with a thick blue stroke and
`paint-order="stroke fill"`. At an interior point covered by the stroke's inner
half, SVG is red and DrawCV is blue. The unsupported property is silently lost.

Fix now: reject unsupported appearance-changing properties with a diagnostic.
Fix for authoring: retain paint order, serialize it, render it consistently, and
export it. Audit ignored properties systematically; a generic ban on unknown
attributes would incorrectly reject harmless metadata and is not the solution.

### 5. High: boolean operations lose newer gradient semantics

Location: `drawcv/core/path_boolean.py:190-282`, especially reconstructed gradients
at lines 235 and 267.

The style conversion reconstructs gradients without `spread` or the paint-local
`transform`. A disjoint difference, which should leave the shape and appearance
unchanged, changes a repeating or translated gradient from BGRA
`[64, 0, 191, 255]` to `[255, 0, 0, 255]` at the probe point.

Fix: share an explicit paint coordinate conversion across boolean operations,
transform baking, and export. Preserve the effective mapping
`entity_world_matrix @ paint_matrix`, spread, alpha, and all paint-specific state.
Also audit object-space stroke paint, conic/image paint, and nonuniform radial
paint conversion: the current resolver does not comprehensively handle these newer
features. Those additional cases are source-review concerns, not counted among
the seven executed failing probes.

## What already exists

- Editable compound paths, quadratic/cubic curves and elliptical arc commands;
  groups, affine transforms, scene layers, and vector boolean operations.
- Caps, joins, dashes, variable-width freehand strokes, and progressive reveal.
- Linear/radial/conic paints, gradient strokes, repeat/reflect spread, image paint,
  blend modes, ordered effects, clipping, masks, and transparent output.
- Retained font assets, shaping, text runs, editable SVG text for a supported
  subset, and explicit font resolution.
- SVG import with paths, gradients, viewports, use/symbol expansion, clipping,
  and explicit rejections for many unsupported elements.
- JSON persistence, undo/redo, animation, and independent rendering comparisons.

This is useful work. The confirmed failures demonstrate missing integration cases;
they do not establish that the implementation is generally poor or that its
authoring model needs replacement.

## Prioritized development plan

### A. Fidelity patch before another feature release

Fix the five families above. Promote the probes into normal tests as fixes land.
Add feature-interaction coverage: paint x booleans x transforms; text x clips x
ancestors; opacity x root/group nesting; self-intersections x fill rule x clips.
Make strict import's guarantee explicit: preserve supported appearance semantics,
or reject with a useful reason.

Update the documentation in the same patch. README describes schema 1.4 while
`CURRENT_SCHEMA_VERSION` is 1.9; docs/svg.md still contains claims that all text
is rasterized and that text import is deferred. The paint table calls patterns
future work even though raster ImagePaint exists. Clearly distinguish image tiles
from retained vector patterns. These inconsistencies obscure usable features.

### B. Fundamental drawing control

1. **Both object-space and screen-space strokes.** Preserve the existing behavior
   for compatibility, but add an explicit stroke-space option. Ordinary SVG
   strokes transform with geometry; non-scaling strokes are a separate choice.
   Currently ordinary anisotropically scaled/sheared SVG strokes are rejected.
   Build the stroke outline in the chosen space and transform it consistently.
2. **Curve-preserving primitive conversion and export.** Native Circle/Ellipse/
   RoundedRectangle/Arc exports currently become polygonal paths, although retained
   EllipticalArcTo already exists. Export exact primitives or arc commands. The
   old flatten_arc helper also imposes a minimum 2-degree step, so its requested
   error tolerance is not a guarantee for large curves. Keep semantic vector
   geometry independent of raster sampling resolution.
3. **Path tools.** Unified shape-to-path conversion, path length/point/tangent
   queries, trimming/splitting, reversal, offsets, and stroke-to-path. These make
   procedural illustration, arrow placement, lettering, and editing much easier.
4. **Reusable markers and paint order.** Start/middle/end markers with orientation,
   units, and context paint are much more general than an Arrow special case.

### C. Rich vector artwork and interchange

Add retained vector patterns, off-center radial gradients, multi-child/grouped
clip paths, and retained vector masks with explicit alpha/luminance semantics.
ImagePaint is useful but does not replace a tile made of editable vector objects.
Import/export embedded images and patterns consistently: an exported native image
pattern currently meets an importer that rejects both pattern and image elements.

Add text-on-path, glyph outline conversion, stroke/gradient text, letter/word
spacing, and the SVG text positioning features intentionally excluded today.
Keep layout, fonts, and export fallback decisions explicit and deterministic.
Treat broader script support as a tested typography project, not a parser switch.

Add CSS class/style-sheet support if real designer SVG import is a product goal.
Choose a documented interoperable subset rather than promising every browser SVG
feature. Arbitrary HTML/foreignObject, scripting, and browser event semantics need
not be prerequisites for expressive Python illustration.

### D. Reliability and scaling

- Keep resvg comparisons and add a browser reference where engine support differs.
  Compare meaningful interior pixels, silhouette geometry, alpha, and masked color
  errors separately. Existing text tests allow 0.65..1.50 coverage ratio and 8px
  bounding-box/centroid deviations against DrawCV: useful smoke tests, but weak
  evidence of typography fidelity.
- Add installed-wheel tests from outside the checkout, base-only and typography
  environments, supported-platform runs, and lower-bound dependency jobs. Current
  CI already has three operating systems, but uses Python 3.12 and freshly resolved
  dependencies; a local pass is not proof of all advertised environments.
- Benchmark small paths on large canvases. Path fill evaluation still allocates
  full-canvas supersampled masks and a winding accumulator. Move this work into
  bounded regions or tiles after correctness is established. No new performance
  benchmark was run in this review.
- Keep the public retained scene model independent of the raster backend. Consider
  a second vector-capable renderer only against measured quality/performance
  criteria. Replacing the entire OpenCV implementation is not justified by this
  review alone.

## SVG reference basis

The [W3C painting chapter](https://www.w3.org/TR/SVG/painting.html) defines fill
rules, stroke behavior, paint order, and markers. The
[paint server chapter](https://www.w3.org/TR/SVG2/pservers.html) describes gradients
and vector/raster patterns. These inform the capability comparison; the concrete
bugs above were independently reproduced against code and resvg, or through
appearance-preserving boolean operations.
