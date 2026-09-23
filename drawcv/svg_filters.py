"""SVG filter equivalents for DrawCV effect stacks.

Each supported effect maps to filter primitives computing the same operation as
its raster processor: color effects use feColorMatrix with DrawCV's own
coefficients (unpremultiplied color, clamped like apply_straight_color_op),
spatial effects use the same premultiplied or straight-color domain as the
processor. Effects without an equivalent return a fallback reason instead.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from drawcv.core.enums import BlurType
from drawcv.effects.blur import BlurEffect
from drawcv.effects.color import (BrightnessContrastEffect, ColorMatrixEffect, GrayscaleEffect,
                                  HueShiftEffect, SaturationEffect, SepiaEffect)
from drawcv.effects.convolution import ConvolutionEffect
from drawcv.effects.glow import GlowEffect
from drawcv.effects.shadow import ShadowEffect

# Rec.709 luma weights in RGB order, as used by the color processors.
_LUMA = np.array([0.2126, 0.7152, 0.0722])
_SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
_ALPHA_ONLY = "0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"


def _num(value: float) -> str:
    return format(float(value), ".10g")


def _values(rgb: np.ndarray, offset=(0.0, 0.0, 0.0)) -> str:
    """feColorMatrix values for an RGB 3x3 matrix and offset; alpha is kept."""
    rows = [[*rgb[i], 0.0, offset[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0, 0.0]]
    return "  ".join(" ".join(_num(v) for v in row) for row in rows)


def _color_matrix(effect) -> str | None:
    identity = np.eye(3)
    gray = np.tile(_LUMA, (3, 1))
    if isinstance(effect, BrightnessContrastEffect):
        c, b = float(effect.contrast), float(effect.brightness)
        k = 0.5 - 0.5 * c + b
        return _values(identity * c, (k, k, k))
    if isinstance(effect, SaturationEffect):
        f = float(effect.factor)
        return _values((1 - f) * gray + f * identity)
    if isinstance(effect, GrayscaleEffect):
        t = float(effect.intensity)
        return _values((1 - t) * identity + t * gray) if t > 0 else None
    if isinstance(effect, SepiaEffect):
        t = float(effect.intensity)
        return _values((1 - t) * identity + t * _SEPIA) if t > 0 else None
    if isinstance(effect, HueShiftEffect):
        rad = math.radians(float(effect.angle) % 360.0)
        c, s = math.cos(rad), math.sin(rad)
        m = np.array([
            [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
            [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
            [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
        ])
        return _values(m)
    if isinstance(effect, ColorMatrixEffect):
        rows = [list(map(float, effect.matrix[i])) for i in range(3)] + [[0.0, 0.0, 0.0, 1.0, 0.0]]
        return "  ".join(" ".join(_num(v) for v in row) for row in rows)
    raise TypeError(type(effect).__name__)


_COLOR_EFFECTS = (BrightnessContrastEffect, SaturationEffect, GrayscaleEffect, SepiaEffect,
                  HueShiftEffect, ColorMatrixEffect)


def unsupported_effect_reason(effects) -> str | None:
    """Why an effect stack cannot be exported as an SVG filter, or None."""
    for effect in effects:
        if isinstance(effect, _COLOR_EFFECTS + (BlurEffect, ShadowEffect, GlowEffect)):
            continue
        if isinstance(effect, ConvolutionEffect):
            # DrawCV adds the bias before clamping; SVG clamps first.
            if float(effect.bias) != 0.0:
                return "effect convolution with bias"
            continue
        return f"effect {effect.effect_type}"
    return None


class _Chain:
    def __init__(self, node: ET.Element):
        self.node = node
        self.current = "SourceGraphic"
        self.count = 0

    def add(self, tag: str, attrs: dict, *, advance: bool = True) -> str:
        self.count += 1
        result = f"r{self.count}"
        element = ET.SubElement(self.node, tag, {**attrs, "result": result})
        if advance:
            self.current = result
        self.element = element
        return result

    def separable(self, source: str, kernel: np.ndarray) -> str:
        """Premultiplied separable convolution with explicit weights."""
        k = len(kernel)
        values = " ".join(_num(v) for v in kernel[::-1])  # SVG convolves; OpenCV correlates
        common = {"kernelMatrix": values, "divisor": "1", "edgeMode": "none", "preserveAlpha": "false"}
        across = self.add("feConvolveMatrix", {"in": source, "order": f"{k} 1", **common}, advance=False)
        return self.add("feConvolveMatrix", {"in": across, "order": f"1 {k}", **common}, advance=False)

    def gaussian(self, source: str, sigma: float, ksize: int) -> str:
        """Gaussian blur of the given input, matching cv2.GaussianBlur((ksize, ksize), sigma)."""
        effective = sigma if sigma > 0 else 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        if ksize // 2 >= 3 * effective:
            return self.add("feGaussianBlur", {"in": source, "stdDeviation": _num(effective)}, advance=False)
        kernel = cv2.getGaussianKernel(ksize, sigma).ravel().astype(np.float64)
        return self.separable(source, kernel)


def _under(chain: _Chain, alpha: str, color, opacity: float, source: str) -> None:
    """Merge a colored alpha layer underneath source (base OVER layer)."""
    flood = chain.add("feFlood", {"flood-color": f"rgb({color.r},{color.g},{color.b})",
                                  "flood-opacity": _num(color.a * opacity)}, advance=False)
    layer = chain.add("feComposite", {"in": flood, "in2": alpha, "operator": "in"}, advance=False)
    chain.add("feMerge", {})
    ET.SubElement(chain.element, "feMergeNode", {"in": layer})
    ET.SubElement(chain.element, "feMergeNode", {"in": source})


def build_filter(defs: ET.Element, ident: str, effects, region) -> None:
    """Append a <filter> equivalent to effects, in userSpaceOnUse region (x, y, w, h)."""
    x, y, w, h = region
    node = ET.SubElement(defs, "filter", {
        "id": ident, "filterUnits": "userSpaceOnUse", "primitiveUnits": "userSpaceOnUse",
        "x": _num(x), "y": _num(y), "width": _num(w), "height": _num(h),
        "color-interpolation-filters": "sRGB"})
    chain = _Chain(node)
    for effect in effects:
        source = chain.current
        if isinstance(effect, _COLOR_EFFECTS):
            values = _color_matrix(effect)
            if values is not None:
                chain.add("feColorMatrix", {"in": source, "type": "matrix", "values": values})
        elif isinstance(effect, BlurEffect):
            k = int(effect.kernel_size)
            if k <= 1:
                continue
            if effect.blur_type == BlurType.GAUSSIAN:
                chain.current = chain.gaussian(source, float(effect.sigma), k)
            else:
                chain.current = chain.separable(source, np.full(k, 1.0 / k))
        elif isinstance(effect, ShadowEffect):
            alpha = chain.add("feColorMatrix", {"in": source, "type": "matrix", "values": _ALPHA_ONLY}, advance=False)
            moved = chain.add("feOffset", {"in": alpha, "dx": _num(effect.offset_x), "dy": _num(effect.offset_y)},
                              advance=False)
            if effect.blur_radius > 0:
                from drawcv.effects.gaussian import gaussian_kernel_size_for_radius
                moved = chain.gaussian(moved, float(effect.blur_radius),
                                       gaussian_kernel_size_for_radius(effect.blur_radius))
            _under(chain, moved, effect.color, float(effect.opacity), source)
        elif isinstance(effect, GlowEffect):
            if effect.blur_radius <= 0:
                continue  # the blurred alpha equals the source alpha: no outer glow
            from drawcv.effects.gaussian import gaussian_kernel_size_for_radius
            alpha = chain.add("feColorMatrix", {"in": source, "type": "matrix", "values": _ALPHA_ONLY}, advance=False)
            blurred = chain.gaussian(alpha, float(effect.blur_radius),
                                     gaussian_kernel_size_for_radius(effect.blur_radius))
            outer = chain.add("feComposite", {"in": blurred, "in2": alpha, "operator": "arithmetic",
                                              "k1": "0", "k2": "1", "k3": "-1", "k4": "0"}, advance=False)
            _under(chain, outer, effect.color, float(effect.opacity), source)
        elif isinstance(effect, ConvolutionEffect):
            kernel = np.array(effect.kernel, dtype=np.float64) * float(effect.factor)
            rows, cols = kernel.shape
            chain.add("feConvolveMatrix", {
                "in": source, "order": f"{cols} {rows}",
                "kernelMatrix": " ".join(_num(v) for v in kernel[::-1, ::-1].ravel()),
                "divisor": "1", "edgeMode": "none", "preserveAlpha": "true"})
        else:
            raise TypeError(f"Unsupported effect {type(effect).__name__}")
    if chain.count == 0:
        # A filter without primitives renders nothing; all effects were no-ops.
        chain.add("feOffset", {"in": "SourceGraphic", "dx": "0", "dy": "0"})
