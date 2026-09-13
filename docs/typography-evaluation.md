# Milestone 4A: typography backend evaluation

Historical evaluation record. The subsequent [4B implementation](typography.md)
adds retained font Text and uses ICU for paragraph bidi.

The feasibility deliverable is complete. **Recommendation: use HarfBuzz + FreeType
as the optional glyph-rendering foundation, and design paragraph layout separately.**
The prototype is under `examples/`; it is not yet a supported `Text` backend.
Hershey rendering, runtime dependencies, and schema 1.3 remain unchanged.

## Results from this checkout

The runnable evaluation produces a [comparison image](../examples/output/typography_comparison.png),
[Thai wrapping image](../examples/output/typography_wrapping.png), and
[measured report](../examples/output/typography_report.json).

| Candidate | Observed result | Decision |
| --- | --- | --- |
| Pillow 12.3 BASIC / FreeType | Loads fonts, but no complex shaping; Arabic sample advance is 286px versus shaped 197.484px | Not sufficient for multilingual text |
| Pillow + RAQM | RAQM, HarfBuzz and FriBiDi unavailable in both the bundled build and the tested published Windows wheel | Do not silently fall back to BASIC |
| uharfbuzz + freetype-py | Latin ligatures, Thai marks, Arabic joining, TrueType and CFF OTF samples match Qt glyph selection | Preferred rendering foundation |
| PyThaiNLP newmm | Word boundaries produce three fitting Thai lines at 320px | Viable Thai segmentation component; not a universal line-break engine |
| Qt QTextLayout | Matching glyphs, bidi runs, fallback and alignment; native Thai WordWrap overflows in this build | Independent integration reference; not the default dependency |
| Pango | Paragraph layout and fallback are documented; not installed or exercised here | Alternative for a later native-stack evaluation |

At 40px, all four comparison samples select the same glyph IDs as Qt. Advance
differences are **0.063–0.156px**. The tests also compare positioned glyph origins
within 0.5px at 36px. FreeType visible ink edges agree with HarfBuzz outline extents
within 1.1px. These are deliberately different checks: text advance is not its ink
width, especially with spaces, bearings, accents and descenders.

The reference uses Qt's own layout and drawing, not the prototype's glyph-placement
code. Both integrations may use HarfBuzz internally; agreement is not independent
proof of the shaping algorithm. Screenshots were inspected for clipped marks,
baseline alignment and Arabic joining, but this is not a comprehensive language
or font-conformance suite.

The Thai sample is:

> ภาษาไทยต้องตัดคำอย่างถูกต้อง เพื่อให้อ่านข้อความได้ง่ายขึ้น

Qt's native WordWrap reports line advances **379.078px and 325.891px** for a 320px
box. Explicit newmm boundaries give **290.781px, 314.109px and 108.219px** through
the prototype. The test suite also feeds explicit Thai breaks to Qt and verifies
that its lines fit. This result applies to the tested build, not every Qt platform.

## What the fixture establishes

- Font files are explicit, unmodified assets with pinned source revisions and
  hashes. Noto Sans, Noto Sans Thai, Noto Sans Arabic and Adobe Source Sans 3 have
  their original OFL/copyright notices in [the font directory](../tests/assets/fonts/README.md).
- Horizontal shaping produces glyph IDs, source clusters, advances and offsets.
  Rasterization uses unhinted grayscale coverage and fractional glyph origins.
  Baseline, advance, outline extents and raster bounds remain distinct values.
- The limited LTR fallback prototype selects a font for a whole grapheme and
  merges adjacent spans using the same font. Missing coverage raises a diagnostic;
  it does not silently draw replacement boxes or consult installed system fonts.
- Thai wrapping uses dictionary word boundaries, measures reshaped candidate lines,
  trims break whitespace, preserves explicit blank lines, and rejects an overlong
  word. It does not split arbitrary codepoints to make a line fit.
- Qt checks supply a separate implementation of alignment, line spacing and mixed
  Arabic/Latin bidi. The direct prototype rejects mixed bidi rather than pretending
  that reversing a string implements the Unicode bidi algorithm.
- A rasterized text mask passes through the existing ImageObject transform, group
  opacity, clipping, blur and transparent PNG path. External white/black/color
  composites agree within one channel value. This verifies raster compatibility,
  not retained font-text serialization, history or editing.

## Reproduce

From the repository root in a Python 3.12 environment:

```shell
python -m pip install -e ".[dev]"
python -m pip install --only-binary=:all: -r examples/typography-requirements.txt
python -m examples.typography_evaluation
python -m pytest -q
```

The dependency versions are pinned for evaluation only. The fixture sets
`PYTHAINLP_READ_ONLY=1` and `PYTHAINLP_OFFLINE=1` unless already configured, so the
bundled dictionary is used without model/corpus downloads. Fonts load directly
from the repository; no system font installation is needed. Qt uses its offscreen
platform plugin. Environments still need the native prerequisites of their wheels.

Verified here: Windows 11 x64, Python 3.12.14, OpenCV 5.0.0, NumPy 2.5.3,
HarfBuzz 0.56.1 Python bindings, FreeType Python bindings 2.5.1, Qt/PySide6 6.11.2.
The JSON records the complete evaluation package versions. **401 tests pass:**
380 existing and 21 evaluation cases. Without the optional packages, the evaluation
module skips; its Qt comparisons skip separately if only Qt is absent.

Published [uharfbuzz wheels](https://pypi.org/project/uharfbuzz/0.56.1/#files) include
Windows, Linux and macOS builds. [freetype-py distributions](https://pypi.org/project/freetype-py/2.5.1/#files)
also provide platform wheels. This is distribution evidence, not a successful
cross-platform test run; macOS, Linux, other architectures and lower dependency
versions remain untested. The downloaded Windows Qt Essentials wheel was 76.9MB,
compared with approximately 1.5MB for uharfbuzz and 0.8MB for freetype-py; these
figures exclude transitive packages and installed footprint. Qt also carries its
own [licensing terms](https://doc.qt.io/qtforpython-6/licenses.html).

## Next: Milestone 4B integration contract

Before exposing a public font-text API, settle these remaining requirements:

1. An explicit font asset descriptor and ordered fallback list, with portable
   resolution, missing-asset diagnostics and stable identity for persistence.
2. A paragraph layer for bidi, script/language itemization and line-break decisions.
   The prototype's LTR fallback is insufficient for general contextual scripts,
   emoji sequences and mixed-direction paragraphs. Evaluate a proven bidi provider;
   do not implement it by reversing runs or guessing direction from the first glyph.
3. Baseline/line-box/ink bounds, wrapping and overflow semantics, line spacing and
   alignment, all consumed by both measurement and drawing. The current Hershey
   top-left positioning convention must remain compatible.
4. Retained Text integration with the existing coverage/compositing pipeline,
   including transforms, masks, effects, JSON, cloning, history and animation.
   Font files must not become serialized native handles or machine-specific caches.
5. Clean-environment Windows/macOS/Linux checks for the selected optional stack.

The public API, serialization migration and these integration tests belong to 4B.
The evaluation does not add variable-axis controls, color emoji, vertical text,
glyph strokes, font caches or performance claims.

## Design references

HarfBuzz intentionally leaves line breaking and higher-level layout to callers:
[HarfBuzz's scope](https://harfbuzz.github.io/what-harfbuzz-doesnt-do.html).
Pillow documents layout engines and separate advance/bounds measurement:
[ImageFont](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html),
[RAQM dependencies](https://pillow.readthedocs.io/en/stable/installation/building-from-source.html).
The Thai tokenizer contract is described in
[PyThaiNLP tokenization](https://pythainlp.org/docs/5.1/api/tokenize.html).
Higher-level alternatives expose paragraph layout directly:
[Qt QTextLayout](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QTextLayout.html),
[Pango Layout](https://docs.gtk.org/Pango/class.Layout.html).
