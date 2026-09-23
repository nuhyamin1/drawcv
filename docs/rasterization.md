# Rasterization and the pixel grid

DrawCV rasterizes filled and stroked geometry with area-exact antialiasing on the
same pixel grid as SVG, HTML canvas and resvg: pixel `(row, column)` is the square
`[column, column + 1] × [row, row + 1]`, and its center is at `+0.5`.

- Shapes on whole-pixel coordinates have crisp edges. A rectangle from `(0, 0)` to
  `(100, 60)` fills exactly 100 × 60 pixels, and a canvas-sized background has no
  half-transparent border.
- A shape's coverage equals its geometric area. A 2 px line covers exactly 2 px, and
  a filled rectangle at fractional coordinates covers exactly `width × height`.
- As in SVG, a 1 px line centered on a whole coordinate covers two half-covered
  pixels. Center hairlines on `.5` (for example `x=70.5`) or use an even width to
  keep them crisp.

## How coverage is computed

`drawcv.core.coverage` evaluates each pixel row on four sub-rows, like the scan
converter resvg uses. On every sub-row the winding number along sorted edge
crossings gives the exact inside spans, and each span adds its exact horizontal
extent, including fractional end pixels. Nonzero and even-odd rules therefore
resolve self-intersections and overlaps directly, and axis-aligned rectangles use
an exact separable shortcut.

Strokes are tessellated into ribbons, joins and caps with one orientation and
rasterized as a single nonzero union. Overlapping pieces neither add opacity nor
leave seams where they meet. Paths, polygons, shape fills, path clips and polygon
clips (`ClipRect`, `ClipPath`) all use the same coverage.

Curves are flattened to within 0.1 px before rasterization. Paint is sampled at
pixel centers for gradients and image paints alike, and scaled `ImageObject`s map
pixel centers to pixel centers.

Unchanged: `LineType.LINE_4` and `LINE_8` strokes keep OpenCV's aliased output,
Hershey text uses `cv2.putText`, and retained font text uses FreeType's own
coverage.

## Accuracy

Measured against exact geometric area and against resvg rendering DrawCV's own SVG
export:

| Shape | Before | Now |
| --- | --- | --- |
| 2 px horizontal line | 3.35 px of coverage (+68%) | 2.00 px; identical to resvg |
| Rectangle at fractional position | +3.7% area | exact |
| Circle stroke, width 3 | +48% area | within 0.3% |
| Canvas-sized background | solid | solid; identical to resvg |
| Area ratio to resvg, typical strokes | 0.80–0.89 | 1.000–1.005 |

OpenCV's antialiased `fillPoly`, used previously, draws a binary fill plus a one
pixel antialiased outline, adding about 0.7 px of coverage outside every edge, and
it centered pixels on whole coordinates. Images rendered by earlier versions
therefore look slightly bolder and half a pixel offset compared with this version.

## Performance

Coverage work scales with edge length rather than pixel area. On the reference
benchmarks (`python -m benchmarks.render`), compared with the previous release
checkout: dense dashes 0.46×, pressure freehand 0.69×, effects and animation about
0.75×, the lesson scene 0.92× the time, 1,200 small opaque rectangles 1.05× and
480 small outlines 1.13×. See [performance](performance.md).
