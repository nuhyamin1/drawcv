# Measured rendering performance

Milestone 5B adds reproducible benchmarks and four targeted changes selected from
their profiles. It preserves the public API, JSON schema 1.4, default BGR behavior,
alpha arithmetic, scene identity and observational animation rendering.

## Changes selected from profiles

1. **Blend only the covered region.** Stroke/text masks that previously blended
   against the entire canvas now use the mask's nonzero bounding rectangle.
   Rasterizing the coverage mask is unchanged, including antialiasing and clipping.
2. **Sample gradients only where coverage exists.** The paint sampler receives a
   region origin and reconstructs the same global integer pixel coordinates.
   Object/world transforms and gradient interpolation keep their existing arithmetic.
3. **Resolve freehand transforms once per render call.** Geometry bounds and matrices
   are invariant while mapping one stroke's points. Matrix-vector operations and
   intermediate Point conversions retain their order to preserve raster rounding.
   Custom coordinate/transform methods retain the original dispatch path.
4. **Avoid unused geometry queries for explicit pivots.** Coordinate and parent-bound
   mapping no longer calculate a default geometry center when a pivot is supplied.
   Dynamic default pivots still derive their center from current geometry.

There is no cache between render calls and no new invalidation protocol. Edits,
cloning, history, masks/effects, progressive geometry and animation continue to
render current state. Full-canvas masks, isolation surfaces, image transforms and
effect buffers are still present; this is not a dirty-region renderer.

The before profiles explain the selection: small-stroke composition consumed about
6.23 of 6.72 profiled seconds; gradient sampling consumed 3.63 of 4.73 seconds;
freehand geometry bounds consumed 5.05 of 6.05 seconds. In the six-level hierarchy,
bounds queries consumed about 6.47 of 6.55 profiled seconds. These instrumented
figures locate costs; the timings below come from separate uninstrumented passes.

## Local timing evidence

Measured on 2026-09-13, Windows 11, Intel Core i7-1165G7, eight logical CPUs,
Python 3.12.14, NumPy 2.5.3 and OpenCV 5.0.0, with OpenCV fixed to one thread.
Each deterministic 640×360 scene runs in a fresh process. The main before/after
runs use two warmup frames and nine samples; the independent after-repeat uses
15 samples. Scene construction, hashing, profiling and memory tracing are excluded
from timing. Animation cycles through 0, 0.5 and 1 seconds.

| Case | Before median / p95 (ms) | After median / p95 (ms) | Median speedup | Repeat median (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1,200 opaque BGR rectangles | 214.3 / 222.4 | 217.8 / 230.9 | 0.98× | 212.0 |
| 480 small strokes | 5986.4 / 6440.0 | 251.3 / 291.0 | 23.82× | 267.6 |
| 72 small gradients | 4369.9 / 4738.1 | 58.9 / 65.7 | 74.19× | 63.7 |
| 6,000 freehand samples | 4321.5 / 7115.5 | 261.0 / 278.2 | 16.56× | 296.7 |
| Dense dash patterns | 904.1 / 976.0 | 374.0 / 513.5 | 2.42× | 464.1 |
| Six opacity/transform levels | 3800.3 / 4067.9 | 130.2 / 148.7 | 29.18× | 132.0 |
| 16 masks/effects objects | 1182.8 / 1198.1 | 274.4 / 320.5 | 4.31× | 256.7 |
| 24 objects / 48 animation tracks | 1738.0 / 2088.0 | 59.0 / 65.1 | 29.48× | 57.8 |

The opaque BGR case is effectively unchanged. The other targeted fixtures improve
in both after-runs, but the repeat also shows variability, especially freehand and
dashes. Nine samples do not establish a dependable tail distribution. These results
are local workload measurements, not universal speed guarantees or comparisons of
default OpenCV thread configurations across machines.

## Memory evidence

Memory tracing runs separately from timing. Traced peak allocations drop from
**39.0 to 17.6 MiB** for small gradients and animation, and **42.5 to 17.6 MiB**
for masks/effects. The gradient fixture's absolute process peak working set drops
from about **81.0 to 60.7 MiB**; masks/effects drops from **81.1 to 60.8 MiB**.

Other traced peaks barely change: BGRA output conversion and full-canvas surfaces
still set the peak even when composition becomes faster. OS working-set numbers
include imports, the retained scene, JSON and allocator caching. Tracemalloc includes
Python and participating NumPy allocations, not all native memory. Reports also
store positive retained allocation counts/bytes; these are not total allocation
traffic. See the [measurement definitions](../benchmarks/README.md).

## Reproduce and inspect

```bash
python -m benchmarks.render --output before.json
# Run against the changed renderer:
python -m benchmarks.render --output after.json --reference before.json
python -m benchmarks.render --output repeat.json --reference before.json --samples 15
python -m benchmarks.summary before.json after.json --repeat repeat.json
python -m benchmarks.gallery after.json
```

The checked-in [before](../benchmarks/results/windows_before.json),
[after](../benchmarks/results/windows_after.json), and
[repeat](../benchmarks/results/windows_repeat.json) reports include raw durations,
scene hashes, output hashes, source fingerprints, OS memory and profiles. The
baseline library was Git HEAD `a710723e8bcaebf2aa373ad50ccdc84eadaee485`;
after-runs used the uncommitted 5B source, identified by its full library hash.

The [fixture source](../benchmarks/scenes.py) is the authoritative reproducible
description. The before report combines completed per-case runs resumed after
calibrating the hierarchy fixture, with unchanged code/settings for all retained
results. Initial ten-level trials were interrupted and contribute no reported
before timings. Their larger form remains an optional stress case, outside the
routine eight-case comparison.

An [after-only stress run](../benchmarks/results/windows_stress_after.json) of ten
levels and 24 circles completed at a 6,513 ms median (three samples, one warmup).
There is no completed before result for a speedup comparison. This larger dynamic-
pivot workload remains expensive and makes the remaining hierarchy cost visible.

The [rendered contact sheet](../examples/output/performance_gallery.png) was inspected
and checks every displayed frame against its recorded hash. No rendering occurs in
the timing-report formatting step.

## Validation and remaining costs

**508 tests pass**, including 25 region/transform regressions and ten benchmark
contract cases. All ten reference frames (seven static plus three animation frames)
match byte-for-byte in both after-runs. Tests compare regional work with the original
full-canvas equations, cover off-canvas and transparent edges, duplicate gradient
stops, nested transforms, caps/joins/dashes, masks/effects, custom mapping, editing,
history and temporal restoration. A work-size regression guards against restoring
full-canvas sampling for a small gradient.
The built wheel was imported directly from its archive and reproduced all ten
reference frames. Development benchmarks and tests are excluded from that wheel.

The CI workflow includes a small benchmark-harness smoke run, with no timing
thresholds. This work was verified locally on Windows; remote platform CI and lower
supported dependency versions have not been run here.

Remaining measured targets include stroke/dash tessellation, repeated geometry work
in other primitive paths, dynamic-pivot hierarchy bounds, and full-canvas effects
and output conversion. Bounds-sized isolation buffers need separate filter-edge
and clipping analysis. Dirty regions and retained geometry caches would require an
explicit invalidation contract. They are not silently added in this milestone.
