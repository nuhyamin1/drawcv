"""Independent Qt paragraph-layout reference for the typography evaluation."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (QGuiApplication, QFontDatabase, QFont, QTextLayout,
                          QTextOption, QImage, QPainter, QColor)

_app = None


def layout_reference(text, paths, size=36, width=900, alignment="left", spacing=1.25,
                     allow_fallback=False):
    global _app
    if _app is None:
        _app = QGuiApplication.instance() or QGuiApplication([])
    families = []
    for path in paths:
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            raise ValueError(f"Qt cannot load font: {path}")
        families.extend(QFontDatabase.applicationFontFamilies(font_id))
    font = QFont()
    font.setFamilies(families)
    font.setPixelSize(size)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias if allow_fallback else QFont.StyleStrategy.NoFontMerging)
    option = QTextOption()
    option.setUseDesignMetrics(True)
    option.setWrapMode(QTextOption.WrapMode.WordWrap)
    option.setAlignment({"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
                         "right": Qt.AlignmentFlag.AlignRight}[alignment])
    # QTextLayout handles a paragraph; Unicode line separator requests a hard line.
    layout = QTextLayout(text.replace("\n", "\u2028"), font)
    layout.setTextOption(option)
    layout.beginLayout()
    lines, y = [], 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        line.setPosition(QPointF(0, y))
        lines.append(dict(start=line.textStart(), length=line.textLength(),
                          width=line.horizontalAdvance(), x=line.naturalTextRect().x(),
                          baseline=y+line.ascent(), height=line.height()))
        y += line.height()*spacing
    layout.endLayout()
    # Generous padding lets tests inspect ink independently of declared extents.
    image = QImage(width+80, max(100, int(y)+80), QImage.Format.Format_RGBA8888)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setPen(QColor(255, 255, 255))
    layout.draw(painter, QPointF(40, 40))
    painter.end()
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width()*4].reshape(image.height(), image.width(), 4).copy()
    runs = []
    for run in layout.glyphRuns():
        runs.append(dict(family=run.rawFont().familyName(), glyphs=list(run.glyphIndexes()),
                         rtl=run.isRightToLeft(),
                         positions=[(p.x(), p.y()) for p in run.positions()]))
    return dict(mask=rgba[..., 3], lines=lines, runs=runs, families=families, padding=40)
