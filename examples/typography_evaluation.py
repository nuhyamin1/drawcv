"""Run with python -m examples.typography_evaluation after installing requirements.

Writes comparison PNGs and a machine-readable report; changes no public Text API.
"""
import os
os.environ.setdefault("PYTHAINLP_READ_ONLY", "1")
os.environ.setdefault("PYTHAINLP_OFFLINE", "1")
from pathlib import Path
import json
import platform
from importlib.metadata import version
import cv2
import numpy as np
from PIL import features, ImageFont
from fontTools.ttLib import TTFont
from examples._typography_backend import (FONT_DIR, shape_run, rasterize, wrap_thai,
    fallback_runs, line_advance, draw_line)
from examples._typography_qt import layout_reference

SAMPLES = [("LATIN / KERNING + LIGATURE", "AVATAR office", "notosans.ttf"),
           ("THAI / STACKED MARKS + SARA AM", "น้ำ กุ้ง กิ๊ง ปู่ ผู้", "notosansthai.ttf"),
           ("ARABIC / CONTEXTUAL JOINING", "السلام عليكم", "notosansarabic.ttf"),
           ("OTF / BEARINGS + DESCENDERS", "Ág j office", "sourcesans3.otf")]
PARAGRAPH = "ภาษาไทยต้องตัดคำอย่างถูกต้อง เพื่อให้อ่านข้อความได้ง่ายขึ้น"


def label(image, text, x, y, size=.48, color=(200, 200, 200)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def paint_mask(image, mask, x, y, color=(245, 245, 245)):
    h, w = mask.shape
    if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
        raise ValueError("Gallery would clip the sample")
    alpha = mask[..., None]/255
    target = image[y:y+h, x:x+w]
    target[:] = np.rint(np.array(color)*alpha+target*(1-alpha)).astype(np.uint8)


def main():
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    report = dict(platform=platform.platform(), python=platform.python_version(),
        packages={p: version(p) for p in ["Pillow", "uharfbuzz", "freetype-py", "fonttools", "regex", "pythainlp", "PySide6-Essentials"]},
        pillow_features={p: features.check(p) for p in ["raqm", "freetype2", "harfbuzz", "fribidi"]},
        samples=[], fonts=[])
    image = np.full((790, 1120, 3), (35, 29, 25), np.uint8)
    label(image, "TYPOGRAPHY / BACKEND EVALUATION", 28, 38, .75)
    label(image, "HarfBuzz + FreeType", 28, 83, .6)
    label(image, "Qt QTextLayout / independent integration", 580, 83, .6)
    label(image, "Blue: baseline   Green: outline bounds + 2px guide   White: visible ink", 28, 116, .46)
    for i, (title, text, name) in enumerate(SAMPLES):
        path = FONT_DIR/name
        run = shape_run(text, path, 40)
        raster = rasterize(run)
        q = layout_reference(text, [path], 40, 440)
        y = 160+i*155
        label(image, title, 28, y)
        baseline = y+63
        cv2.line(image, (28, baseline), (510, baseline), (180, 110, 70), 1)
        paint_mask(image, raster.mask, 36+raster.left, baseline+raster.top)
        left, top, right, bottom = run.outline_bounds
        cv2.rectangle(image, (36+round(left)-2, baseline+round(top)-2),
                      (36+round(right)+2, baseline+round(bottom)+2), (110, 160, 70), 1)
        qt_ys, qt_xs = np.where(q["mask"] > 0)
        qt_top, qt_left = qt_ys.min(), qt_xs.min()
        cropped = q["mask"][qt_top:qt_ys.max()+1, qt_left:qt_xs.max()+1]
        cv2.line(image, (580, baseline), (1080, baseline), (180, 110, 70), 1)
        paint_mask(image, cropped, 588+qt_left-40,
                   baseline+qt_top-40-round(q["lines"][0]["baseline"]))
        qt_glyphs = q["runs"][0]["glyphs"]
        if name == "notosansarabic.ttf":
            qt_glyphs = qt_glyphs[::-1]
        matched = qt_glyphs == [g["gid"] for g in run.glyphs]
        delta = abs(q["lines"][0]["width"]-run.advance)
        basic = ImageFont.truetype(str(path), 40, layout_engine=ImageFont.Layout.BASIC)
        label(image, f"Advance {run.advance:.3f}px / {len(run.glyphs)} glyphs", 28, y+108, .44)
        label(image, f"Glyphs {'MATCH' if matched else 'DIFFER'} / advance delta {delta:.3f}px", 580, y+108, .44)
        report["samples"].append(dict(text=text, font=name, advance=run.advance,
            outline_bounds=run.outline_bounds, raster_bounds=[raster.left, raster.top,
                raster.left+raster.mask.shape[1], raster.top+raster.mask.shape[0]],
            glyphs=run.glyphs, qt_glyph_match=matched, qt_advance_delta=delta,
            pillow_basic_advance=basic.getlength(text)))
        with TTFont(path) as font:
            report["fonts"].append(dict(file=name, family=font['name'].getDebugName(1),
                outlines="CFF" if "CFF " in font else "TrueType", variable="fvar" in font))
    cv2.imwrite(str(output / "typography_comparison.png"), image)

    paths = [FONT_DIR/"notosans.ttf", FONT_DIR/"notosansthai.ttf"]
    lines = wrap_thai(PARAGRAPH, paths, 30, 320)
    native = layout_reference(PARAGRAPH, paths, 30, 320, allow_fallback=True)
    report["thai_wrap"] = dict(source=PARAGRAPH, width=320, lines=lines,
        advances=[line_advance(line, paths, 30) for line in lines],
        qt_native_advances=[line["width"] for line in native["lines"]],
        qt_native_overflow=any(line["width"] > 320 for line in native["lines"]))
    report["fallback"] = [dict(text=r.text, font=r.font_path.name) for r in fallback_runs("DrawCV กิ๊ง 123", paths)]
    bidi = layout_reference("DrawCV العربية 123", [FONT_DIR/"notosans.ttf", FONT_DIR/"notosansarabic.ttf"], allow_fallback=True)
    report["qt_mixed_bidi"] = dict(text="DrawCV العربية 123", runs=bidi["runs"],
                                  missing_glyphs=sum(r["glyphs"].count(0) for r in bidi["runs"]))
    sheet = np.full((430, 1120, 3), (35, 29, 25), np.uint8)
    label(sheet, "THAI WORD BREAKS / SAME LINES, THREE ALIGNMENTS", 28, 36, .7)
    label(sheet, "newmm boundaries; each line reshaped. 320px width / 54px baseline step.", 28, 68, .48)
    for col, alignment in enumerate(["left", "center", "right"]):
        origin = 28+col*366
        label(sheet, alignment.upper(), origin, 109)
        cv2.rectangle(sheet, (origin, 130), (origin+320, 340), (75, 70, 65), 1)
        mask = np.zeros(sheet.shape[:2], np.uint8)
        for j, line in enumerate(lines):
            advance = line_advance(line, paths, 30)
            shift = {"left": 0, "center": (320-advance)/2, "right": 320-advance}[alignment]
            draw_line(mask, line, paths, 30, origin+shift, 174+j*54)
        paint_mask(sheet, mask, 0, 0)
    status = "overflows" if report["thai_wrap"]["qt_native_overflow"] else "fits"
    label(sheet, f"Qt native Thai word wrapping {status} in this build. Shown above: explicit dictionary breaks.", 28, 391, .46)
    cv2.imwrite(str(output / "typography_wrapping.png"), sheet)
    (output / "typography_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(glyph_matches=[s["qt_glyph_match"] for s in report["samples"]],
                         advance_deltas=[s["qt_advance_delta"] for s in report["samples"]],
                         thai_native_overflow=report["thai_wrap"]["qt_native_overflow"])))


if __name__ == "__main__":
    main()
