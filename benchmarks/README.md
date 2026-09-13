# Rendering benchmarks

Run from the repository checkout with the base library dependencies installed:

```bash
python -m benchmarks.render --output before.json
# After changing the renderer:
python -m benchmarks.render --output after.json --reference before.json
```

The default runs eight deterministic 640×360 workloads in separate fresh Python
processes. Each has two warmup frames and nine timed frames. OpenCV uses one thread;
`--threads N`, `--warmup N`, `--samples N`, and `--cases NAME ...` are explicit controls.
Do not run concurrent benchmarks, tests, or other heavy jobs when comparing results.
Scene construction, JSON serialization, hashing, profiling and memory tracing are
outside the timed region. Garbage collection remains enabled during timing.
Use `--resume` to reuse completed cases after an interrupted run; it checks the
environment, full library source fingerprint, frame counts and fixture hashes.
Changing library source requires a fresh report. `--reference` also verifies cached
results when resuming a comparison.

| Case | Workload |
| --- | --- |
| dense_bgr | 1,200 small opaque rectangles, legacy BGR |
| small_strokes | 480 small translucent outlines, BGRA |
| small_gradients | 72 small linear/radial gradients with RGBA stops, BGRA |
| long_freehand | 6,000 retained pressure samples, BGRA |
| dense_dashes | 24 polylines × 70 points with a dense four-part dash pattern, BGRA |
| deep_transparency | Three circles nested inside six transformed opacity groups, BGRA |
| masks_effects | 16 gradients with masks, clips, blur and shadows, BGRA |
| animation | 24 gradient rectangles with 48 tracks; sample 0, 0.5 and 1 seconds cyclically |

`benchmarks.scenes.build_scene(name)` constructs the retained scene with stable IDs
and no random or system assets. Call `scene.save_json(path)` to store a full fixture.
`scene_sha256` hashes the complete source JSON. `frame_hashes` hashes the output bytes
(three timestamps for animation, one for static scenes). A reference comparison
requires matching source and frame hashes, otherwise the command fails. Renderer
source fingerprints and the Git HEAD identify the checkout without committing it.

`--cases deep_transparency_stress` expands the case to ten levels and 24 circles.
It is deliberately outside the default run: repeated recursive bounds calculations
made its pre-5B baseline impractical. An exploratory run of that larger case
was terminated before collecting a complete result; a ten-level/three-circle trial
was also stopped. Neither trial contributes reported timings. Six levels keeps the
default workload suitable for repeated measurement.
The after-only stress report records a completed 6.5-second median frame, so this
combination remains costly even after the targeted fixes.

## Reading results

Each result stores every frame duration, median, minimum, maximum and interpolated
95th percentile. Nine samples give a useful local comparison, not a reliable latency
distribution or a service-level guarantee. Increase `--samples` for tail analysis.
Animation timing mixes the three timestamps; inspect the raw sequence when comparing
particular frames. Warmup and measurement include observational snapshot/restore.

Timing, memory tracing, and cProfile are separate passes. The profile is one extra
frame at time zero, sorted by cumulative time. It is evidence for where to investigate,
not a replacement for the uninstrumented timings.

Memory fields distinguish:

- OS process RSS and its high-water mark, captured before warmup and after timing.
  These absolute process values include Python, NumPy, OpenCV, scene construction,
  retained source JSON and allocator caching. They are not per-frame allocation totals.
  Unix reports the high-water mark through `resource`; current RSS is null there.
- `traced_peak_bytes`: a separate frame measured with `tracemalloc`, after collection.
  This includes Python and registered NumPy allocations; native allocations that do
  not participate in tracemalloc are not counted.
- `traced_positive_retained_blocks/bytes`: positive snapshot differences while the
  returned image remains alive. These are retained allocations, not total allocation
  traffic; temporary allocations released within a frame are absent.

Compare on the same idle machine, power state, versions, thread count and settings.
OS scheduling, thermal throttling and other processes can affect the timings. There
are no speed assertions in CI. The reference option checks thread counts and exact
content; it does not make different hardware/software environments comparable.

Full-scene raster rendering remains the reference behavior. This milestone does not
introduce retained image caches, dirty regions or a performance promise for arbitrary
scenes. See [the measured results and chosen optimization](../docs/performance.md).

To inspect the measured output:

```bash
python -m benchmarks.gallery after.json
```

The contact sheet checks each image against the report's hash before drawing its
labels. It runs separately from timing. The benchmark package is excluded from the
published wheel; run it from the source checkout.
