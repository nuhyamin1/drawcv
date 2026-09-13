# Retained font text contract

`Text(..., fonts=(FontAsset, ...))`
opts into optional HarfBuzz/FreeType typography. `fonts=None` retains Hershey.
Font assets hold immutable standalone TTF/OTF bytes, loaded explicitly by the caller;
JSON embeds those bytes with SHA-256, never a system font name or native handle.

`font_size` is pixels (positive, quantized to 1/64px); `wrap_width` is optional
positive paragraph width; `line_spacing >= 1` multiplies the shared line height.
`direction` is `auto`, `ltr`, or `rtl`, resolved per explicit paragraph. Alignment
remains physical left/center/right. Position anchors the padded paragraph's top
edge and left/center/right edge, preserving Hershey's existing anchor convention.

`measure()` returns immutable local coordinates: maximum line **advance_width**,
unpadded **layout_bounds**, padded **paragraph_bounds**, exact untransformed raster
**ink_bounds** (or None), and individual line text, advance, baseline, line box and
ink bounds. Alignment uses advance, not ink. Line height uses the maximum ascent
and descent across supplied fonts and actual glyph extents, so fallback runs share
a baseline. Geometry/hit testing use the union of padded paragraph and ink; world
bounds transform that union with a one-local-pixel guard for bilinear resampling.
Effects do not enlarge these measurements. Ink bounds describe full glyph coverage,
before object opacity, masks, clips, effects or color alpha.

Fallback chooses the first supplied font that covers an entire script/direction
run. This preserves joining context and can select a fallback for a whole run even
when the primary covers some characters. Missing coverage raises ValidationError.
There is no OS fallback or implicit replacement glyph. 4B supports Latin, Thai,
Arabic, their combining marks, numbers and ordinary punctuation. Other scripts,
emoji, variation selectors, bidi formatting/isolate controls and join controls are
rejected explicitly. ICU resolves paragraph levels, paired brackets and per-line visual order.
Formatting/isolate controls remain outside the public 4B input contract.

Wrapping is greedy at ordinary spaces and Thai newmm dictionary word boundaries; each
candidate is reshaped. Break whitespace is trimmed. CRLF is treated as LF, explicit
blank lines survive, and overlong words raise instead of silently clipping or
breaking clusters. No justification, hyphenation or emergency character wrapping.
U+00A0 nonbreaking spaces glue adjacent tokens. Tabs and Unicode line/paragraph
separator characters are rejected; use LF/CRLF for explicit breaks. Unspaced
unsupported scripts do not silently fall back to character splitting.

Font faces, shaped runs, glyph rasters and paragraph masks use bounded caches.
Keys contain immutable font bytes and all layout parameters. Position, transforms,
color and opacity do not invalidate local layout. Native handles and caches are
never scene state. Font-size, wrap-width and line-spacing animation use numeric
tracks; strings/font lists use explicit edits, not built-in interpolation.

## Install and use

From a source checkout:

```shell
python -m pip install ".[typography]"
python -m examples.retained_typography
```

The extra adds uharfbuzz, freetype-py, ICU bindings, regex, and PyThaiNLP. On Windows
and Linux it selects `pyicu-wheels`; on macOS it selects `PyICU`, which requires ICU
build prerequisites (for example `brew install icu4c pkg-config` and an appropriate
`PKG_CONFIG_PATH`). The base installation still renders Hershey without any of them.
Missing typography dependencies raise a diagnostic when font layout is requested.
Qt/Pillow are evaluation references and are not production typography dependencies.

```python
from drawcv import FontAsset, Text, Point, Scene, Color, OpenCVRenderer

latin = FontAsset.from_file("my-fonts/Latin.otf")
thai = FontAsset.from_file("my-fonts/Thai.ttf")
label = Text("DrawCV ภาษาไทย 123", fonts=[latin, thai], font_size=32,
             wrap_width=420, line_spacing=1.2, position=Point(40, 40))
metrics = label.measure()
print(metrics.advance_width, metrics.layout_bounds, metrics.ink_bounds)
scene = Scene(520, 240, background=Color(0, 0, 0, 0))
scene.add(label)
OpenCVRenderer().render(scene, alpha=True).save("label.png")
```

Load fonts you have permission to embed. `FontAsset.from_file` snapshots bytes once;
later edits/deletion of that file do not affect existing scenes. Create and assign
a new asset to change the font. Sharing an immutable asset is safe; cloning does
not duplicate native handles. Font collections, bitmap/color fonts and variation
axis controls are not supported. Variable TTF fonts use their default instance.
`font_family`, `font_scale` and `thickness` remain Hershey options; font text uses
`fonts` and `font_size` and has no glyph-outline stroke option.

## Persistence, editing and animation

Documents now write **schema 1.4**; 1.0–1.3 migrate forward. Embedded fonts include
name, base64 bytes and SHA-256, checked on import. JSON size grows with font size;
assets are embedded per Text object, without document-level deduplication. An
inactive font configuration also survives persistence. Old default Hershey object
dictionaries keep their existing shape, inside the newer document envelope.

```python
with scene.edit(label):
    label.text = "ข้อความใหม่ DrawCV"
    label.font_size = 36
scene.undo()
scene.redo()
scene.animate(label, "font_size", 28, 40, duration=1)
frame = scene.render_at_time(.5, alpha=True)
```

Numeric options validate before assignment. Font coverage, supported scripts and
wrap overflow are checked when measuring or rendering, so loading a scene does
not require loading native engines. Failed temporal rendering restores authored
state. Font text does not implement progressive glyph reveal.

Scenes containing font text use the corrected premultiplied rendering path even
for BGR, matching the gradient-scene rule. Existing scenes with neither feature
retain their legacy BGR rasterization. Transforms apply to the laid-out coverage
mask; changing scale does not reflow paragraphs. Object/group opacity, masks,
clips, blur/shadows and PNG export use the same retained compositing pipeline.

## Caches, limits and verification

`drawcv.typography.layout.clear_text_cache()` releases the shared caches;
`text_cache_info()` reports their occupancy. Paragraph masks have a 32MiB/32-entry
budget. Other caches cap at 16 font/size pairs, 256 short shaped runs and 512 glyph
rasters; runs over 256 characters and glyph sizes above 128px bypass their caches.
Font text accepts at most 20,000 characters, font sizes up to 4096px, and 16M pixels
per layout/raster. These bounds constrain work but are not a general performance claim.
Native face access is serialized with a lock; measurement values are immutable.

Verification on Windows 11 x64 / Python 3.12.14: **437 tests pass**, including 36
retained font-text cases and 21 evaluation cases. Eight existing BGR frames are
byte-identical to the committed checkout. The retained example reloads its
embedded-font JSON with identical pixels. Mixed-script glyph positions and bracket
order are compared with Qt; transformed masks, measured ink, cache invalidation,
missing dependencies, history, temporal rollback and PNG composites are covered.

The tested ICU wheel contains ICU **77.1**. Packaging a different ICU/FreeType/
HarfBuzz version can affect metrics or pixels; explicit font bytes alone do not
guarantee identical output across engine versions. The added
[CI matrix](../.github/workflows/typography.yml) covers Windows, macOS and Linux,
but the macOS/Linux jobs have not been executed in this local session.

See the [retained example](../examples/retained_typography.py),
[rendered preview](../examples/output/retained_typography_preview.png), and
[4A evaluation](typography-evaluation.md). ICU supplies paragraph levels and
line reordering via its [bidi API](https://unicode-org.github.io/icu-docs/apidoc/released/icu4c/ubidi_8h.html).
The wheel distribution is listed on [PyPI](https://pypi.org/project/pyicu-wheels/2.15.2/).
