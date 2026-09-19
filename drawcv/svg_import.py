"""Native retained SVG importer for DrawCV.

Constructs genuine retained DrawCV objects from supported SVG documents with
high semantic fidelity, strict validation, and zero hidden rasterization.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path as FilePath
from typing import Any, Callable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import BlendMode, CapStyle, FillRule, JoinStyle
from drawcv.core.exceptions import DrawCVError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.effects.clipping import ClipRect
from drawcv.group import Group
from drawcv.scene import Scene
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.line import Line
from drawcv.shapes.path import Close, CubicTo, EllipticalArcTo, LineTo, MoveTo, Path, QuadraticTo, Subpath
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.shapes.text import Text, TextAnchor, TextRun
from drawcv.typography.resolver import FontResolver, ResolvedFontDescriptor
from drawcv.styles.fill import FillStyle
from drawcv.styles.paint import GradientStop, LinearGradient, PaintLike, RadialGradient
from drawcv.styles.stroke import StrokeStyle

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ALLOWED_ALIGNMENTS = {
    "none",
    "xMinYMin", "xMidYMin", "xMaxYMin",
    "xMinYMid", "xMidYMid", "xMaxYMid",
    "xMinYMax", "xMidYMax", "xMaxYMax",
}
ALLOWED_MEET_OR_SLICE = {"meet", "slice"}


def extract_element_tag(raw_tag: str) -> str:
    """Extract local element tag and strictly validate XML namespace."""
    if raw_tag.startswith("{"):
        ns, local = raw_tag[1:].split("}", 1)
        if ns != SVG_NS:
            raise SVGImportError(
                f"Element '<{local}>' has unsupported XML namespace '{ns}'",
                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_NAMESPACE", severity="error", message=f"Foreign XML namespace '{ns}'"),
            )
        return local
    return raw_tag


# -----------------------------------------------------------------------------
# Diagnostics and Exceptions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SVGImportDiagnostic:
    """Structured diagnostic reported during SVG import."""
    code: str
    severity: str  # "error", "warning", "info"
    message: str
    element_tag: str | None = None
    source_id: str | None = None
    attribute: str | None = None


class SVGImportError(DrawCVError, ValueError):
    """Raised when SVG import fails or encounters an unsupported construct in strict mode."""
    def __init__(self, message: str, *, diagnostic: SVGImportDiagnostic | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class SVGImportLimits:
    """Configurable resource and security bounds for SVG import."""
    max_source_bytes: int = 10 * 1024 * 1024       # 10 MB raw SVG input
    max_nesting_depth: int = 256                   # Maximum XML element depth
    max_elements: int = 50_000                     # Maximum total XML elements
    max_attribute_length: int = 2 * 1024 * 1024   # 2 MB (for path data strings)
    max_path_tokens: int = 100_000                 # Maximum tokens in a single path
    max_points_per_poly: int = 20_000              # Maximum points in a polyline/polygon
    max_resources: int = 1_000                     # Maximum defs/id items
    max_reference_depth: int = 16                  # Maximum href recursion depth
    max_scene_dimension: int = 16_384              # Maximum canvas width or height
    max_coordinate_magnitude: float = 1e7          # Bound coordinate values
    max_use_instances: int = 1_000                 # Maximum <use> elements processed
    max_expanded_elements: int = 50_000            # Maximum materialized DrawCV nodes via expansion


@dataclass(frozen=True)
class SVGImportResult:
    """Immutable result of an SVG import operation containing the Scene and diagnostics."""
    scene: Scene
    diagnostics: tuple[SVGImportDiagnostic, ...] = ()


# -----------------------------------------------------------------------------
# Standard W3C CSS / SVG Named Color Map
# -----------------------------------------------------------------------------

NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "aliceblue": (240, 248, 255), "antiquewhite": (250, 235, 215), "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212), "azure": (240, 255, 255), "beige": (245, 245, 220),
    "bisque": (255, 228, 196), "black": (0, 0, 0), "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255), "blueviolet": (138, 43, 226), "brown": (165, 42, 42),
    "burlywood": (222, 184, 135), "cadetblue": (95, 158, 160), "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30), "coral": (255, 127, 80), "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220), "crimson": (220, 20, 60), "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139), "darkcyan": (0, 139, 139), "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169), "darkgreen": (0, 100, 0),
    "darkkhaki": (189, 183, 107), "darkmagenta": (139, 0, 139), "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0), "darkorchid": (153, 50, 204), "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122), "darkseagreen": (143, 188, 143), "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79), "darkslategrey": (47, 79, 79), "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211), "deeppink": (255, 20, 147), "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105), "dimgrey": (105, 105, 105), "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34), "floralwhite": (255, 250, 240), "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255), "gainsboro": (220, 220, 220), "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0), "goldenrod": (218, 165, 32), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "green": (0, 128, 0), "greenyellow": (173, 255, 47),
    "honeydew": (240, 255, 240), "hotpink": (255, 105, 180), "indianred": (205, 92, 92),
    "indigo": (75, 0, 130), "ivory": (255, 255, 240), "khaki": (240, 230, 140),
    "lavender": (230, 230, 250), "lavenderblush": (255, 240, 245), "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205), "lightblue": (173, 216, 230), "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255), "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211), "lightgreen": (144, 238, 144),
    "lightpink": (255, 182, 193), "lightsalmon": (255, 160, 122), "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250), "lightslategray": (119, 136, 153), "lightslategrey": (119, 136, 153),
    "lightsteelblue": (176, 196, 222), "lightyellow": (255, 255, 224), "lime": (0, 255, 0),
    "limegreen": (50, 205, 50), "linen": (250, 240, 230), "magenta": (255, 0, 255),
    "maroon": (128, 0, 0), "mediumaquamarine": (102, 205, 170), "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211), "mediumpurple": (147, 112, 219), "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238), "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204), "mediumvioletred": (199, 21, 133), "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250), "mistyrose": (255, 228, 225), "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173), "navy": (0, 0, 128), "oldlace": (253, 245, 230),
    "olive": (128, 128, 0), "olivedrab": (107, 142, 35), "orange": (255, 165, 0),
    "orangered": (255, 69, 0), "orchid": (218, 112, 214), "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152), "paleturquoise": (175, 238, 238), "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213), "peachpuff": (255, 218, 185), "peru": (205, 133, 63),
    "pink": (255, 192, 203), "plum": (221, 160, 221), "powderblue": (176, 224, 230),
    "purple": (128, 0, 128), "rebeccapurple": (102, 51, 153), "red": (255, 0, 0),
    "rosybrown": (188, 143, 143), "royalblue": (65, 105, 225), "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114), "sandybrown": (244, 164, 96), "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238), "sienna": (160, 82, 45), "silver": (192, 192, 192),
    "skyblue": (135, 206, 235), "slateblue": (106, 90, 205), "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144), "snow": (255, 250, 250), "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180), "tan": (210, 180, 140), "teal": (0, 128, 128),
    "thistle": (216, 191, 216), "tomato": (255, 99, 71), "turquoise": (64, 224, 208),
    "violet": (238, 130, 238), "wheat": (245, 222, 179), "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245), "yellow": (255, 255, 0), "yellowgreen": (154, 205, 50),
}


# -----------------------------------------------------------------------------
# Length & Dimension Parser
# -----------------------------------------------------------------------------

RE_LENGTH = re.compile(r"^([+-]?(?:[0-9]*\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?)(px|%|em|rem|ex|ch|cm|mm|in|pt|pc|vw|vh|vmin|vmax)?$")

@dataclass(frozen=True)
class SVGLength:
    """Parsed SVG length with explicit unit."""
    value: float
    unit: str  # "", "px", "%", or an unsupported unit

    @classmethod
    def parse(cls, text: str, limits: SVGImportLimits | None = None) -> SVGLength:
        cleaned = text.strip()
        m = RE_LENGTH.match(cleaned)
        if not m:
            raise SVGImportError(
                f"Malformed SVG length: '{text}'",
                diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_LENGTH", severity="error", message=f"Malformed SVG length: '{text}'"),
            )
        val = float(m.group(1))
        if not math.isfinite(val):
            raise SVGImportError(
                f"Non-finite SVG length: '{text}'",
                diagnostic=SVGImportDiagnostic(code="SVG_NON_FINITE_COORDINATE", severity="error", message=f"Non-finite length: '{text}'"),
            )
        max_mag = limits.max_coordinate_magnitude if limits is not None else 1e7
        if abs(val) > max_mag:
            raise SVGImportError(
                f"Coordinate magnitude {val} exceeds limit of {max_mag}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
            )
        unit = m.group(2) or ""
        return cls(val, unit)

    def to_fraction(self, context_name: str = "", limits: SVGImportLimits | None = None) -> float:
        """Resolve length as a normalized fraction [0.0, 1.0] for objectBoundingBox."""
        max_mag = limits.max_coordinate_magnitude if limits is not None else 1e7
        if self.unit == "%":
            val = self.value / 100.0
        elif self.unit in ("", "px"):
            val = self.value
        else:
            raise SVGImportError(
                f"Unsupported length unit '{self.unit}' for fraction '{context_name}'",
                diagnostic=SVGImportDiagnostic(
                    code="SVG_UNSUPPORTED_LENGTH_UNIT", severity="error",
                    message=f"Unsupported length unit '{self.unit}'", attribute=context_name,
                ),
            )
        if abs(val) > max_mag:
            raise SVGImportError(
                f"Resolved coordinate magnitude {val} exceeds limit of {max_mag} for {context_name}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
            )
        return val

    def to_absolute(self, reference_length: float | None = None, context_name: str = "", limits: SVGImportLimits | None = None) -> float:
        """Resolve to scalar pixels. Strictly rejects unsupported units and unreferenced percentages."""
        max_mag = limits.max_coordinate_magnitude if limits is not None else 1e7
        if self.unit in ("", "px"):
            val = self.value
        elif self.unit == "%":
            if reference_length is None:
                raise SVGImportError(
                    f"Percentage unit '%' is not supported for '{context_name}' in Milestone 1 without viewport context",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_LENGTH_UNIT", severity="error",
                        message=f"Percentage unit unsupported for '{context_name}'", attribute=context_name,
                    ),
                )
            val = (self.value / 100.0) * reference_length
        else:
            raise SVGImportError(
                f"Unsupported length unit '{self.unit}' in '{self.value}{self.unit}' for {context_name}",
                diagnostic=SVGImportDiagnostic(
                    code="SVG_UNSUPPORTED_LENGTH_UNIT", severity="error",
                    message=f"Unsupported length unit '{self.unit}'", attribute=context_name,
                ),
            )
        if abs(val) > max_mag:
            raise SVGImportError(
                f"Resolved coordinate magnitude {val} exceeds limit of {max_mag} for {context_name}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
            )
        return val


# -----------------------------------------------------------------------------
# SVG Viewport Context & ViewBox Mapping
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SVGViewportContext:
    """Represents an active SVG viewport for length and percentage resolution."""
    width: float
    height: float

    @property
    def normalized_diagonal(self) -> float:
        return math.sqrt((self.width ** 2 + self.height ** 2) / 2.0)

    def resolve_x(self, length: SVGLength | None, default: float = 0.0, context_name: str = "", limits: SVGImportLimits | None = None) -> float:
        if length is None:
            return default
        return length.to_absolute(reference_length=self.width, context_name=context_name, limits=limits)

    def resolve_y(self, length: SVGLength | None, default: float = 0.0, context_name: str = "", limits: SVGImportLimits | None = None) -> float:
        if length is None:
            return default
        return length.to_absolute(reference_length=self.height, context_name=context_name, limits=limits)

    def resolve_diagonal(self, length: SVGLength | None, default: float = 0.0, context_name: str = "", limits: SVGImportLimits | None = None) -> float:
        if length is None:
            return default
        return length.to_absolute(reference_length=self.normalized_diagonal, context_name=context_name, limits=limits)


def compute_viewbox_matrix(
    vb_attr: str | None,
    target_w: float,
    target_h: float,
    par_attr: str = "xMidYMid meet",
    limits: SVGImportLimits | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Calculates the 3x3 affine transformation matrix and effective viewport for an SVG viewBox."""
    if not vb_attr:
        return np.eye(3, dtype=np.float64), (float(target_w), float(target_h))

    vb_nums = [float(t) for t in re.split(r"[\s,]+", vb_attr.strip()) if t]
    if len(vb_nums) != 4:
        raise SVGImportError("viewBox requires exactly 4 numbers: min_x min_y width height")
    min_x, min_y, vb_w, vb_h = vb_nums
    if vb_w <= 0 or vb_h <= 0 or not math.isfinite(vb_w) or not math.isfinite(vb_h):
        raise SVGImportError(f"Invalid viewBox dimensions: ({vb_w}, {vb_h})")

    max_mag = limits.max_coordinate_magnitude if limits is not None else 1e7
    for coord in (min_x, min_y, vb_w, vb_h):
        if abs(coord) > max_mag:
            raise SVGImportError(
                f"viewBox coordinate magnitude {coord} exceeds limit of {max_mag}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
            )

    par = par_attr.strip() if par_attr else "xMidYMid meet"
    par_parts = par.split()
    if len(par_parts) > 2:
        raise SVGImportError(
            f"Malformed preserveAspectRatio: '{par}'",
            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_ASPECT_RATIO", severity="error", message=f"Invalid preserveAspectRatio: '{par}'"),
        )
    align = par_parts[0] if par_parts else "xMidYMid"
    meet_or_slice = par_parts[1] if len(par_parts) > 1 else "meet"

    if align not in ALLOWED_ALIGNMENTS:
        raise SVGImportError(
            f"Unsupported preserveAspectRatio alignment: '{align}'",
            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_ASPECT_RATIO", severity="error", message=f"Invalid alignment: '{align}'"),
        )
    if align != "none" and meet_or_slice not in ALLOWED_MEET_OR_SLICE:
        raise SVGImportError(
            f"Unsupported preserveAspectRatio meetOrSlice: '{meet_or_slice}'",
            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_ASPECT_RATIO", severity="error", message=f"Invalid meetOrSlice: '{meet_or_slice}'"),
        )

    if align == "none":
        sx = target_w / vb_w
        sy = target_h / vb_h
        tx = -min_x * sx
        ty = -min_y * sy
    else:
        scale_x = target_w / vb_w
        scale_y = target_h / vb_h
        s = min(scale_x, scale_y) if meet_or_slice == "meet" else max(scale_x, scale_y)
        sx, sy = s, s

        if "xMin" in align:
            tx = -min_x * s
        elif "xMax" in align:
            tx = (target_w - vb_w * s) - min_x * s
        else:  # xMid
            tx = (target_w - vb_w * s) / 2.0 - min_x * s

        if "yMin" in align:
            ty = -min_y * s
        elif "yMax" in align:
            ty = (target_h - vb_h * s) - min_y * s
        else:  # yMid
            ty = (target_h - vb_h * s) / 2.0 - min_y * s

    M_viewbox = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    return M_viewbox, (float(vb_w), float(vb_h))
# -----------------------------------------------------------------------------

RE_RGB_FUNC = re.compile(r"^rgba?\s*\(\s*([^)]+)\s*\)$", re.IGNORECASE)

class SVGColorParser:
    """Parses SVG/CSS color values into DrawCV Color instances."""

    @classmethod
    def parse(cls, text: str, current_color: Color | None = None) -> Color | None:
        """Parse color string. Returns None for 'none'. Resolves 'currentColor'."""
        cleaned = text.strip().lower()
        if cleaned == "none":
            return None
        if cleaned in ("transparent",):
            return Color(0, 0, 0, 0.0)
        if cleaned == "currentcolor":
            return current_color if current_color is not None else Color.black()

        if cleaned in NAMED_COLORS:
            r, g, b = NAMED_COLORS[cleaned]
            return Color(r, g, b, 1.0)

        if cleaned.startswith("#"):
            try:
                return Color.from_hex(cleaned)
            except ValidationError as err:
                raise SVGImportError(
                    f"Malformed hex color: '{text}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_COLOR", severity="error", message=str(err)),
                ) from err

        m_rgb = RE_RGB_FUNC.match(cleaned)
        if m_rgb:
            raw_args = m_rgb.group(1).strip()
            # Supports comma-separated or whitespace-slash separated
            if "/" in raw_args:
                parts, alpha_part = raw_args.split("/", 1)
                tokens = re.split(r"[\s,]+", parts.strip()) + [alpha_part.strip()]
            else:
                tokens = re.split(r"[\s,]+", raw_args)
            tokens = [t for t in tokens if t]
            if len(tokens) not in (3, 4):
                raise SVGImportError(
                    f"Invalid rgb/rgba function arguments: '{text}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_COLOR", severity="error", message=f"Invalid rgb syntax: '{text}'"),
                )

            rgb_vals = []
            for tok in tokens[:3]:
                if tok.endswith("%"):
                    pct = float(tok[:-1])
                    rgb_vals.append(int(round(max(0.0, min(100.0, pct)) * 2.55)))
                else:
                    rgb_vals.append(int(round(max(0.0, min(255.0, float(tok))))))

            alpha_val = 1.0
            if len(tokens) == 4:
                tok_a = tokens[3]
                if tok_a.endswith("%"):
                    alpha_val = max(0.0, min(1.0, float(tok_a[:-1]) / 100.0))
                else:
                    alpha_val = max(0.0, min(1.0, float(tok_a)))

            return Color(rgb_vals[0], rgb_vals[1], rgb_vals[2], alpha_val)

        raise SVGImportError(
            f"Unsupported or malformed color: '{text}'",
            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_COLOR", severity="error", message=f"Unrecognized color: '{text}'"),
        )


# -----------------------------------------------------------------------------
# Transform Parser
# -----------------------------------------------------------------------------

RE_TRANSFORM_FUNC = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")

class SVGTransformParser:
    """Parses SVG transform attribute lists into exact 3x3 affine numpy matrices."""

    @classmethod
    def parse_to_matrix(cls, text: str, limits: SVGImportLimits | None = None) -> np.ndarray:
        if not text or not text.strip():
            return np.eye(3, dtype=np.float64)

        M = np.eye(3, dtype=np.float64)
        matches = list(RE_TRANSFORM_FUNC.finditer(text.strip()))
        if not matches:
            raise SVGImportError(
                f"Malformed SVG transform: '{text}'",
                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TRANSFORM", severity="error", message=f"Malformed transform: '{text}'"),
            )

        # Check for unparsed non-whitespace characters between functions
        last_end = 0
        for m in matches:
            span_before = text[last_end:m.start()].strip(" \t\r\n,")
            if span_before:
                raise SVGImportError(
                    f"Unexpected characters in transform list: '{span_before}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TRANSFORM", severity="error", message=f"Syntax error in transform: '{text}'"),
                )
            last_end = m.end()

        tail = text[last_end:].strip(" \t\r\n,")
        if tail:
            raise SVGImportError(
                f"Unexpected trailing characters in transform list: '{tail}'",
                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TRANSFORM", severity="error", message=f"Syntax error in transform: '{text}'"),
            )

        for m in matches:
            fn_name = m.group(1).strip()
            args_str = m.group(2).strip()
            num_tokens = [t for t in re.split(r"[\s,]+", args_str) if t]
            try:
                nums = [float(t) for t in num_tokens]
            except ValueError as err:
                raise SVGImportError(
                    f"Malformed numeric arguments in transform '{fn_name}': {args_str}",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TRANSFORM", severity="error", message=str(err)),
                ) from err

            max_mag = limits.max_coordinate_magnitude if limits is not None else 1e7
            for n in nums:
                if not math.isfinite(n):
                    raise SVGImportError(
                        f"Non-finite coordinate in transform '{fn_name}': {n}",
                        diagnostic=SVGImportDiagnostic(code="SVG_NON_FINITE_COORDINATE", severity="error", message="Non-finite transform coordinate"),
                    )
                if abs(n) > max_mag:
                    raise SVGImportError(
                        f"Transform coordinate magnitude {n} exceeds limit of {max_mag}",
                        diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
                    )

            if fn_name == "matrix":
                if len(nums) != 6:
                    raise SVGImportError("SVG matrix() requires exactly 6 arguments")
                a, b, c, d, e, f = nums
                M_fn = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)

            elif fn_name == "translate":
                if len(nums) == 1:
                    tx, ty = nums[0], 0.0
                elif len(nums) == 2:
                    tx, ty = nums[0], nums[1]
                else:
                    raise SVGImportError("SVG translate() requires 1 or 2 arguments")
                M_fn = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

            elif fn_name == "scale":
                if len(nums) == 1:
                    sx, sy = nums[0], nums[0]
                elif len(nums) == 2:
                    sx, sy = nums[0], nums[1]
                else:
                    raise SVGImportError("SVG scale() requires 1 or 2 arguments")
                M_fn = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            elif fn_name == "rotate":
                if len(nums) == 1:
                    angle_deg, cx, cy = nums[0], 0.0, 0.0
                elif len(nums) == 3:
                    angle_deg, cx, cy = nums[0], nums[1], nums[2]
                else:
                    raise SVGImportError("SVG rotate() requires 1 or 3 arguments")
                rad = math.radians(angle_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                R = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                if cx != 0.0 or cy != 0.0:
                    T_fwd = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                    T_inv = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                    M_fn = T_fwd @ R @ T_inv
                else:
                    M_fn = R

            elif fn_name == "skewX":
                if len(nums) != 1:
                    raise SVGImportError("SVG skewX() requires exactly 1 argument")
                tan_a = math.tan(math.radians(nums[0]))
                M_fn = np.array([[1.0, tan_a, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            elif fn_name == "skewY":
                if len(nums) != 1:
                    raise SVGImportError("SVG skewY() requires exactly 1 argument")
                tan_a = math.tan(math.radians(nums[0]))
                M_fn = np.array([[1.0, 0.0, 0.0], [tan_a, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            else:
                raise SVGImportError(
                    f"Unsupported SVG transform function: '{fn_name}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TRANSFORM", severity="error", message=f"Unsupported transform '{fn_name}'"),
                )

            # Multiply in SVG left-to-right matrix composition order
            M = M @ M_fn

        return M


# -----------------------------------------------------------------------------
# SVG Path Lexer & State-Machine Parser
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# SVG Path Parser (Full SVG Suite: MmLlHhVvCcSsQqTtAaZz)
# -----------------------------------------------------------------------------

RE_NUMBER = re.compile(r"^[+-]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?")


class SVGPathParser:
    """Parses SVG path 'd' string into a retained DrawCV Path."""

    @classmethod
    def parse(cls, d_str: str, fill_rule: FillRule = FillRule.NON_ZERO, limits: SVGImportLimits | None = None) -> Path:
        max_tokens = limits.max_path_tokens if limits else 100_000
        max_coord_mag = limits.max_coordinate_magnitude if limits else 1e7
        token_count = 0

        s = d_str.strip()
        n = len(s)
        pos = 0

        if not s:
            return Path(subpaths=[], fill_rule=fill_rule)

        def skip_wsp() -> None:
            nonlocal pos
            while pos < n and s[pos] in " \t\r\n":
                pos += 1

        def consume_comma_wsp() -> None:
            nonlocal pos
            skip_wsp()
            if pos < n and s[pos] == ",":
                pos += 1
                skip_wsp()
                if pos < n and s[pos] == ",":
                    raise SVGImportError(
                        "Consecutive commas in path data",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_MALFORMED_PATH",
                            severity="error",
                            message="Consecutive commas in path data",
                        ),
                    )

        def check_token_budget() -> None:
            nonlocal token_count
            token_count += 1
            if token_count > max_tokens:
                raise SVGImportError(
                    f"Path token count exceeded limit of {max_tokens}",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_RESOURCE_LIMIT_EXCEEDED",
                        severity="error",
                        message="Path token count exceeded limit",
                    ),
                )

        def consume_number() -> float:
            nonlocal pos
            consume_comma_wsp()
            if pos >= n:
                raise SVGImportError(
                    "Premature end of path data expecting coordinate",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message="Missing coordinate in path data",
                    ),
                )
            if s[pos] in "MmLlHhVvCcSsQqTtAaZz":
                raise SVGImportError(
                    f"Expected coordinate but found command '{s[pos]}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message=f"Unexpected command '{s[pos]}' where coordinate expected",
                    ),
                )
            m = RE_NUMBER.match(s[pos:])
            if not m:
                raise SVGImportError(
                    f"Malformed number at position {pos} in path data: '{s[pos:pos+10]}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message="Malformed number in path data",
                    ),
                )
            num_str = m.group(0)
            pos += len(num_str)
            check_token_budget()
            val = float(num_str)
            if not math.isfinite(val):
                raise SVGImportError(
                    f"Non-finite coordinate in path: {val}",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_NON_FINITE_COORDINATE",
                        severity="error",
                        message=f"Non-finite path coordinate: {val}",
                    ),
                )
            if abs(val) > max_coord_mag:
                raise SVGImportError(
                    f"Path coordinate magnitude {val} exceeds limit of {max_coord_mag}",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_RESOURCE_LIMIT_EXCEEDED",
                        severity="error",
                        message="Coordinate magnitude exceeded limit",
                    ),
                )
            return val

        def consume_flag() -> bool:
            nonlocal pos
            consume_comma_wsp()
            if pos >= n:
                raise SVGImportError(
                    "Premature end of path data expecting arc flag ('0' or '1')",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message="Missing arc flag in path data",
                    ),
                )
            ch = s[pos]
            if ch not in ("0", "1"):
                raise SVGImportError(
                    f"Expected arc flag '0' or '1' but found '{ch}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message=f"Invalid arc flag '{ch}'",
                    ),
                )
            pos += 1
            check_token_budget()
            return ch == "1"

        def has_more_coordinates() -> bool:
            save_pos = pos
            while save_pos < n and s[save_pos] in " \t\r\n":
                save_pos += 1
            if save_pos < n and s[save_pos] == ",":
                save_pos += 1
                while save_pos < n and s[save_pos] in " \t\r\n":
                    save_pos += 1
            if save_pos >= n:
                return False
            ch = s[save_pos]
            if ch.isdigit():
                return True
            if ch in "+-":
                return save_pos + 1 < n and (s[save_pos + 1].isdigit() or s[save_pos + 1] == ".")
            if ch == ".":
                return save_pos + 1 < n and s[save_pos + 1].isdigit()
            return False

        subpaths: list[Subpath] = []
        active_subpath: Subpath | None = None
        current_pt = Point(0.0, 0.0)
        start_pt = Point(0.0, 0.0)
        last_cubic_ctrl: Point | None = None
        last_quad_ctrl: Point | None = None

        while True:
            skip_wsp()
            if pos >= n:
                break
            if s[pos] == ",":
                raise SVGImportError(
                    "Unexpected comma in path data",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message="Unexpected comma in path data",
                    ),
                )
            ch = s[pos]
            if ch in "BbRr":
                raise SVGImportError(
                    f"SVG path command '{ch}' is not supported",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_PATH_COMMAND",
                        severity="error",
                        message=f"Unsupported path command '{ch}'",
                    ),
                )

            if ch not in "MmLlHhVvCcSsQqTtAaZz":
                raise SVGImportError(
                    f"Path data must start with a command, got '{ch}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message=f"Illegal path character: '{ch}'",
                    ),
                )
            cmd = ch
            pos += 1
            check_token_budget()
            skip_wsp()
            if pos < n and s[pos] == ",":
                raise SVGImportError(
                    f"Unexpected comma immediately following path command '{cmd}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_PATH",
                        severity="error",
                        message=f"Comma immediately after command '{cmd}'",
                    ),
                )

            if cmd in ("Z", "z"):
                if active_subpath is not None and active_subpath.commands:
                    active_subpath.commands.append(Close())
                    active_subpath.closed = True
                    current_pt = start_pt
                active_subpath = None
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if cmd not in ("M", "m") and active_subpath is None:
                if not subpaths:
                    raise SVGImportError(
                        f"SVG path command '{cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_MALFORMED_PATH",
                            severity="error",
                            message=f"Path command '{cmd}' without prior MoveTo",
                        ),
                    )
                # When Z/z is followed by a command other than M/m, the next subpath
                # begins at the initial point of the just-closed subpath (start_pt).
                active_subpath = Subpath(commands=[MoveTo(start_pt)], closed=False)
                subpaths.append(active_subpath)
                current_pt = start_pt
                last_cubic_ctrl = None
                last_quad_ctrl = None

            if cmd == "M":
                x = consume_number()
                y = consume_number()
                current_pt = Point(x, y)
                start_pt = current_pt
                active_subpath = Subpath(commands=[MoveTo(current_pt)], closed=False)
                subpaths.append(active_subpath)
                last_cubic_ctrl = None
                last_quad_ctrl = None
                while has_more_coordinates():
                    x = consume_number()
                    y = consume_number()
                    current_pt = Point(x, y)
                    active_subpath.commands.append(LineTo(current_pt))
                continue

            elif cmd == "m":
                dx = consume_number()
                dy = consume_number()
                if not subpaths:
                    current_pt = Point(dx, dy)
                else:
                    current_pt = Point(current_pt.x + dx, current_pt.y + dy)
                start_pt = current_pt
                active_subpath = Subpath(commands=[MoveTo(current_pt)], closed=False)
                subpaths.append(active_subpath)
                last_cubic_ctrl = None
                last_quad_ctrl = None
                while has_more_coordinates():
                    dx = consume_number()
                    dy = consume_number()
                    current_pt = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(LineTo(current_pt))
                continue

            elif cmd == "L":
                while True:
                    x = consume_number()
                    y = consume_number()
                    current_pt = Point(x, y)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "l":
                while True:
                    dx = consume_number()
                    dy = consume_number()
                    current_pt = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "H":
                while True:
                    x = consume_number()
                    current_pt = Point(x, current_pt.y)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "h":
                while True:
                    dx = consume_number()
                    current_pt = Point(current_pt.x + dx, current_pt.y)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "V":
                while True:
                    y = consume_number()
                    current_pt = Point(current_pt.x, y)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "v":
                while True:
                    dy = consume_number()
                    current_pt = Point(current_pt.x, current_pt.y + dy)
                    active_subpath.commands.append(LineTo(current_pt))
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "C":
                while True:
                    c1x = consume_number()
                    c1y = consume_number()
                    c2x = consume_number()
                    c2y = consume_number()
                    x = consume_number()
                    y = consume_number()
                    ctrl1 = Point(c1x, c1y)
                    ctrl2 = Point(c2x, c2y)
                    end = Point(x, y)
                    active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))
                    current_pt = end
                    last_cubic_ctrl = ctrl2
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "c":
                while True:
                    dc1x = consume_number()
                    dc1y = consume_number()
                    dc2x = consume_number()
                    dc2y = consume_number()
                    dx = consume_number()
                    dy = consume_number()
                    ctrl1 = Point(current_pt.x + dc1x, current_pt.y + dc1y)
                    ctrl2 = Point(current_pt.x + dc2x, current_pt.y + dc2y)
                    end = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))
                    current_pt = end
                    last_cubic_ctrl = ctrl2
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "S":
                while True:
                    c2x = consume_number()
                    c2y = consume_number()
                    x = consume_number()
                    y = consume_number()
                    if last_cubic_ctrl is not None:
                        ctrl1 = Point(2.0 * current_pt.x - last_cubic_ctrl.x, 2.0 * current_pt.y - last_cubic_ctrl.y)
                    else:
                        ctrl1 = current_pt
                    ctrl2 = Point(c2x, c2y)
                    end = Point(x, y)
                    active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))
                    current_pt = end
                    last_cubic_ctrl = ctrl2
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "s":
                while True:
                    dc2x = consume_number()
                    dc2y = consume_number()
                    dx = consume_number()
                    dy = consume_number()
                    if last_cubic_ctrl is not None:
                        ctrl1 = Point(2.0 * current_pt.x - last_cubic_ctrl.x, 2.0 * current_pt.y - last_cubic_ctrl.y)
                    else:
                        ctrl1 = current_pt
                    ctrl2 = Point(current_pt.x + dc2x, current_pt.y + dc2y)
                    end = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))
                    current_pt = end
                    last_cubic_ctrl = ctrl2
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "Q":
                while True:
                    cx = consume_number()
                    cy = consume_number()
                    x = consume_number()
                    y = consume_number()
                    ctrl = Point(cx, cy)
                    end = Point(x, y)
                    active_subpath.commands.append(QuadraticTo(ctrl, end))
                    current_pt = end
                    last_quad_ctrl = ctrl
                    last_cubic_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "q":
                while True:
                    dcx = consume_number()
                    dcy = consume_number()
                    dx = consume_number()
                    dy = consume_number()
                    ctrl = Point(current_pt.x + dcx, current_pt.y + dcy)
                    end = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(QuadraticTo(ctrl, end))
                    current_pt = end
                    last_quad_ctrl = ctrl
                    last_cubic_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "T":
                while True:
                    x = consume_number()
                    y = consume_number()
                    if last_quad_ctrl is not None:
                        ctrl = Point(2.0 * current_pt.x - last_quad_ctrl.x, 2.0 * current_pt.y - last_quad_ctrl.y)
                    else:
                        ctrl = current_pt
                    end = Point(x, y)
                    active_subpath.commands.append(QuadraticTo(ctrl, end))
                    current_pt = end
                    last_quad_ctrl = ctrl
                    last_cubic_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "t":
                while True:
                    dx = consume_number()
                    dy = consume_number()
                    if last_quad_ctrl is not None:
                        ctrl = Point(2.0 * current_pt.x - last_quad_ctrl.x, 2.0 * current_pt.y - last_quad_ctrl.y)
                    else:
                        ctrl = current_pt
                    end = Point(current_pt.x + dx, current_pt.y + dy)
                    active_subpath.commands.append(QuadraticTo(ctrl, end))
                    current_pt = end
                    last_quad_ctrl = ctrl
                    last_cubic_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "A":
                while True:
                    rx = abs(consume_number())
                    ry = abs(consume_number())
                    phi = consume_number() % 360.0
                    large_arc = consume_flag()
                    sweep = consume_flag()
                    x = consume_number()
                    y = consume_number()
                    end = Point(x, y)
                    if current_pt != end:
                        if rx == 0.0 or ry == 0.0:
                            active_subpath.commands.append(LineTo(end))
                        else:
                            active_subpath.commands.append(
                                EllipticalArcTo(
                                    radius_x=rx,
                                    radius_y=ry,
                                    x_axis_rotation=phi,
                                    large_arc=large_arc,
                                    sweep=sweep,
                                    end=end,
                                )
                            )
                    current_pt = end
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

            elif cmd == "a":
                while True:
                    rx = abs(consume_number())
                    ry = abs(consume_number())
                    phi = consume_number() % 360.0
                    large_arc = consume_flag()
                    sweep = consume_flag()
                    dx = consume_number()
                    dy = consume_number()
                    end = Point(current_pt.x + dx, current_pt.y + dy)
                    if current_pt != end:
                        if rx == 0.0 or ry == 0.0:
                            active_subpath.commands.append(LineTo(end))
                        else:
                            active_subpath.commands.append(
                                EllipticalArcTo(
                                    radius_x=rx,
                                    radius_y=ry,
                                    x_axis_rotation=phi,
                                    large_arc=large_arc,
                                    sweep=sweep,
                                    end=end,
                                )
                            )
                    current_pt = end
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                    if not has_more_coordinates():
                        break

        return Path(subpaths=subpaths, fill_rule=fill_rule)


# -----------------------------------------------------------------------------
# Computed Style Layer & Inheritance Cascade
# -----------------------------------------------------------------------------

SUPPORTED_BLEND_MODES = {
    "normal": BlendMode.NORMAL,
    "multiply": BlendMode.MULTIPLY,
    "screen": BlendMode.SCREEN,
    "overlay": BlendMode.OVERLAY,
    "darken": BlendMode.DARKEN,
    "lighten": BlendMode.LIGHTEN,
    "color-dodge": BlendMode.COLOR_DODGE,
    "color-burn": BlendMode.COLOR_BURN,
    "hard-light": BlendMode.HARD_LIGHT,
    "soft-light": BlendMode.SOFT_LIGHT,
    "difference": BlendMode.DIFFERENCE,
    "exclusion": BlendMode.EXCLUSION,
}

@dataclass
class ComputedStyle:
    """Represents computed CSS/SVG style values at a specific element in the scene hierarchy."""
    # Inherited properties
    fill: str | None = "black"
    fill_opacity: float = 1.0
    fill_rule: FillRule = FillRule.NON_ZERO
    stroke: str | None = None
    stroke_opacity: float = 1.0
    stroke_width: float = 1.0
    raw_stroke_width: SVGLength | None = None
    stroke_linecap: CapStyle = CapStyle.BUTT
    stroke_linejoin: JoinStyle = JoinStyle.MITER
    stroke_miterlimit: float = 4.0
    stroke_dasharray: tuple[float, ...] = ()
    raw_stroke_dasharray: tuple[SVGLength, ...] | None = None
    stroke_dashoffset: float = 0.0
    raw_stroke_dashoffset: SVGLength | None = None
    color: Color = Color(0, 0, 0, 1.0)
    visibility: str = "visible"
    clip_rule: FillRule = FillRule.NON_ZERO

    # Typography properties (inherited)
    font_family: str | None = None
    font_size: float = 16.0
    font_weight: str | int | None = None
    font_style: str | None = None
    text_anchor: str = "start"
    direction: str = "ltr"
    xml_space: str = "default"

    # Non-inherited properties (reset per-element)
    opacity: float = 1.0
    display: str = "inline"
    mix_blend_mode: BlendMode = BlendMode.NORMAL
    clip_path: str | None = None
    vector_effect: str = "none"
    overflow: str | None = None

    def inherit_child(self) -> ComputedStyle:
        """Create a new child style inheriting all inherited properties while resetting non-inherited ones."""
        return ComputedStyle(
            fill=self.fill,
            fill_opacity=self.fill_opacity,
            fill_rule=self.fill_rule,
            stroke=self.stroke,
            stroke_opacity=self.stroke_opacity,
            stroke_width=self.stroke_width,
            raw_stroke_width=self.raw_stroke_width,
            stroke_linecap=self.stroke_linecap,
            stroke_linejoin=self.stroke_linejoin,
            stroke_miterlimit=self.stroke_miterlimit,
            stroke_dasharray=self.stroke_dasharray,
            raw_stroke_dasharray=self.raw_stroke_dasharray,
            stroke_dashoffset=self.stroke_dashoffset,
            raw_stroke_dashoffset=self.raw_stroke_dashoffset,
            color=self.color,
            visibility=self.visibility,
            clip_rule=self.clip_rule,
            font_family=self.font_family,
            font_size=self.font_size,
            font_weight=self.font_weight,
            font_style=self.font_style,
            text_anchor=self.text_anchor,
            direction=self.direction,
            xml_space=self.xml_space,
            opacity=1.0,
            display="inline",
            mix_blend_mode=BlendMode.NORMAL,
            clip_path=None,
            vector_effect="none",
            overflow=None,
        )


class SVGStyleResolver:
    """Resolves presentation attributes and inline style declarations."""

    @classmethod
    def parse_inline_style(cls, style_str: str) -> dict[str, str]:
        declarations: dict[str, str] = {}
        if not style_str:
            return declarations
        parts = style_str.split(";")
        for p in parts:
            if not p.strip():
                continue
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            declarations[k.strip().lower()] = v.strip()
        return declarations

    @classmethod
    def resolve(cls, element: ET.Element, parent_style: ComputedStyle | None = None, limits: SVGImportLimits | None = None) -> ComputedStyle:
        style = parent_style.inherit_child() if parent_style else ComputedStyle()

        # Gather presentation attributes
        attrs = {k.lower(): v.strip() for k, v in element.attrib.items()}
        # Inline style declarations override presentation attributes
        inline = cls.parse_inline_style(attrs.get("style", ""))

        # Check strict rejection of isolation: isolate
        isolation_val = inline.get("isolation", attrs.get("isolation", "")).strip().lower()
        if isolation_val == "isolate":
            raise SVGImportError(
                "CSS 'isolation: isolate' is not supported in Milestone 1",
                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_ISOLATION", severity="error", message="CSS 'isolation: isolate' is not supported"),
            )

        def get_prop(name: str) -> str | None:
            if name in inline:
                return inline[name]
            return attrs.get(name)

        # Drawing currently has a fixed fill/stroke order. Do not silently accept
        # a declaration that changes the picture. Prefixes below expand to SVG's
        # normal order; inherit/unset are safe because ancestors obey this gate.
        paint_order = get_prop("paint-order")
        if paint_order is not None:
            normalized_order = " ".join(paint_order.lower().split())
            if normalized_order not in ("normal", "fill", "fill stroke", "fill stroke markers",
                                         "inherit", "initial", "unset"):
                raise SVGImportError(
                    f"Unsupported paint-order '{paint_order}'; DrawCV currently paints fill before stroke",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_PAINT_ORDER", severity="error",
                        message=f"Unsupported paint-order '{paint_order}'", attribute="paint-order",
                    ),
                )

        # 1. Color (for currentColor)
        color_val = get_prop("color")
        if color_val:
            parsed_c = SVGColorParser.parse(color_val)
            if parsed_c:
                style.color = parsed_c

        # 2. Fill
        fill_val = get_prop("fill")
        if fill_val is not None:
            style.fill = fill_val

        # 3. Fill-opacity
        fill_op = get_prop("fill-opacity")
        if fill_op is not None:
            style.fill_opacity = max(0.0, min(1.0, float(fill_op)))

        # 4. Fill-rule
        fr_val = get_prop("fill-rule")
        if fr_val is not None:
            fr_clean = fr_val.lower().replace("-", "")
            style.fill_rule = FillRule.EVEN_ODD if "evenodd" in fr_clean else FillRule.NON_ZERO

        # 5. Stroke
        stroke_val = get_prop("stroke")
        if stroke_val is not None:
            style.stroke = stroke_val

        # 6. Stroke-opacity
        stroke_op = get_prop("stroke-opacity")
        if stroke_op is not None:
            style.stroke_opacity = max(0.0, min(1.0, float(stroke_op)))

        # 7. Stroke-width
        sw_val = get_prop("stroke-width")
        if sw_val is not None:
            length = SVGLength.parse(sw_val, limits=limits)
            style.raw_stroke_width = length
            if length.unit != "%":
                style.stroke_width = max(0.0, length.to_absolute(context_name="stroke-width", limits=limits))

        # 8. Stroke-linecap
        lc_val = get_prop("stroke-linecap")
        if lc_val is not None:
            lc_clean = lc_val.lower().strip()
            if lc_clean == "butt":
                style.stroke_linecap = CapStyle.BUTT
            elif lc_clean == "round":
                style.stroke_linecap = CapStyle.ROUND
            elif lc_clean == "square":
                style.stroke_linecap = CapStyle.SQUARE
            else:
                raise SVGImportError(f"Unsupported stroke-linecap: '{lc_val}'")

        # 9. Stroke-linejoin
        lj_val = get_prop("stroke-linejoin")
        if lj_val is not None:
            lj_clean = lj_val.lower().strip()
            if lj_clean == "miter":
                style.stroke_linejoin = JoinStyle.MITER
            elif lj_clean == "round":
                style.stroke_linejoin = JoinStyle.ROUND
            elif lj_clean == "bevel":
                style.stroke_linejoin = JoinStyle.BEVEL
            else:
                raise SVGImportError(f"Unsupported stroke-linejoin: '{lj_val}'")

        # 10. Stroke-miterlimit
        ml_val = get_prop("stroke-miterlimit")
        if ml_val is not None:
            style.stroke_miterlimit = max(1.0, float(ml_val))

        # 11. Stroke-dasharray
        da_val = get_prop("stroke-dasharray")
        if da_val is not None:
            da_clean = da_val.strip().lower()
            if da_clean in ("none", ""):
                style.stroke_dasharray = ()
                style.raw_stroke_dasharray = ()
            else:
                num_tokens = [t for t in re.split(r"[\s,]+", da_clean) if t]
                raw_lengths = [SVGLength.parse(tok, limits=limits) for tok in num_tokens]
                style.raw_stroke_dasharray = tuple(raw_lengths)
                if not any(l.unit == "%" for l in raw_lengths):
                    dashes = []
                    for parsed_l in raw_lengths:
                        val = parsed_l.to_absolute(context_name="stroke-dasharray", limits=limits)
                        if val < 0.0:
                            raise SVGImportError(
                                f"Negative value in stroke-dasharray: '{parsed_l.value}'",
                                diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_STYLE", severity="error", message="Negative stroke-dasharray value"),
                            )
                        dashes.append(val)
                    if len(dashes) % 2 == 1:
                        dashes = dashes * 2  # SVG rule: odd dasharray repeated
                    style.stroke_dasharray = tuple(dashes)

        # 12. Stroke-dashoffset
        do_val = get_prop("stroke-dashoffset")
        if do_val is not None:
            parsed_do = SVGLength.parse(do_val.strip(), limits=limits)
            style.raw_stroke_dashoffset = parsed_do
            if parsed_do.unit != "%":
                style.stroke_dashoffset = parsed_do.to_absolute(context_name="stroke-dashoffset", limits=limits)

        # 13. Visibility
        vis_val = get_prop("visibility")
        if vis_val is not None:
            vis_clean = vis_val.strip().lower()
            if vis_clean == "collapse":
                vis_clean = "hidden"
            style.visibility = vis_clean

        # 14. Clip-rule
        cr_val = get_prop("clip-rule")
        if cr_val is not None:
            cr_clean = cr_val.lower().replace("-", "")
            style.clip_rule = FillRule.EVEN_ODD if "evenodd" in cr_clean else FillRule.NON_ZERO

        # 15. Opacity (non-inherited)
        op_val = get_prop("opacity")
        if op_val is not None:
            style.opacity = max(0.0, min(1.0, float(op_val)))

        # 16. Display (non-inherited)
        disp_val = get_prop("display")
        if disp_val is not None:
            style.display = disp_val.strip().lower()

        # 17. Mix-blend-mode (non-inherited)
        bm_val = get_prop("mix-blend-mode")
        if bm_val is not None:
            bm_clean = bm_val.strip().lower()
            if bm_clean not in SUPPORTED_BLEND_MODES:
                raise SVGImportError(
                    f"Unsupported mix-blend-mode: '{bm_val}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_BLEND_MODE", severity="error", message=f"Unsupported blend mode: '{bm_val}'"),
                )
            style.mix_blend_mode = SUPPORTED_BLEND_MODES[bm_clean]

        # 18. Clip-path (non-inherited)
        cp_val = get_prop("clip-path")
        if cp_val is not None:
            cp_clean = cp_val.strip()
            if cp_clean.lower() != "none":
                if not RE_URL_REF.match(cp_clean):
                    raise SVGImportError(
                        f"Unsupported or malformed clip-path syntax: '{cp_val}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_PATH", severity="error", message=f"Unsupported clip-path '{cp_val}'", attribute="clip-path"),
                    )
                style.clip_path = cp_clean
            else:
                style.clip_path = None

        # 19. Vector-effect (non-inherited)
        ve_val = get_prop("vector-effect")
        if ve_val is not None:
            ve_clean = ve_val.strip().lower()
            if ve_clean not in ("none", "non-scaling-stroke"):
                raise SVGImportError(
                    f"Unsupported vector-effect: '{ve_val}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_VECTOR_EFFECT",
                        severity="error",
                        message=f"Unsupported vector-effect '{ve_val}'",
                        attribute="vector-effect",
                    ),
                )
            style.vector_effect = ve_clean

        # 20. Overflow (non-inherited, author-specified overrides UA default)
        ov_val = get_prop("overflow")
        if ov_val is not None:
            ov_clean = ov_val.strip().lower()
            if ov_clean not in ("visible", "hidden", "scroll", "auto"):
                raise SVGImportError(
                    f"Unsupported or invalid overflow value: '{ov_val}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_STYLE",
                        severity="error",
                        message=f"Invalid overflow value: '{ov_val}'",
                        attribute="overflow",
                    ),
                )
            style.overflow = ov_clean

        # 21. Typography properties
        fam_val = get_prop("font-family")
        if fam_val is not None:
            style.font_family = fam_val.strip()

        fs_val = get_prop("font-size")
        if fs_val is not None:
            parsed_fs = SVGLength.parse(fs_val.strip(), limits=limits)
            if parsed_fs.unit == "%":
                computed_fs = style.font_size * parsed_fs.value / 100.0
            else:
                computed_fs = parsed_fs.to_absolute(context_name="font-size", limits=limits)
            if computed_fs < 0.0:
                raise SVGImportError(
                    f"Negative font-size '{fs_val}' is invalid",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_MALFORMED_LENGTH",
                        severity="error",
                        message=f"Negative font-size '{fs_val}'",
                        attribute="font-size",
                    ),
                )
            if computed_fs == 0.0:
                raise SVGImportError(
                    "Font-size zero is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_FONT_SIZE",
                        severity="error",
                        message="Font-size zero is not supported",
                        attribute="font-size",
                    ),
                )
            if computed_fs > 4096.0:
                raise SVGImportError(
                    f"Font-size {computed_fs} exceeds maximum allowed size of 4096px",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_FONT_SIZE",
                        severity="error",
                        message="Font-size exceeds 4096px",
                        attribute="font-size",
                    ),
                )
            style.font_size = computed_fs

        fw_val = get_prop("font-weight")
        if fw_val is not None:
            style.font_weight = fw_val.strip().lower()

        fst_val = get_prop("font-style")
        if fst_val is not None:
            style.font_style = fst_val.strip().lower()

        ta_val = get_prop("text-anchor")
        if ta_val is not None:
            clean_ta = ta_val.strip().lower()
            if clean_ta in ("start", "middle", "end"):
                style.text_anchor = clean_ta

        dir_val = get_prop("direction")
        if dir_val is not None:
            clean_dir = dir_val.strip().lower()
            if clean_dir in ("ltr", "rtl"):
                style.direction = clean_dir

        ws_val = get_prop("white-space")
        if ws_val is not None:
            clean_ws = ws_val.strip().lower()
            if clean_ws in ("normal", "collapse"):
                style.xml_space = "default"
            else:
                raise SVGImportError(
                    f"CSS 'white-space: {ws_val}' is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_CSS_PROPERTY",
                        severity="error",
                        message=f"CSS 'white-space: {ws_val}' is not supported",
                        attribute="white-space",
                    ),
                )
        else:
            xml_space_val = attrs.get("xml:space") or attrs.get("{http://www.w3.org/xml/1998/namespace}space") or element.attrib.get("{http://www.w3.org/XML/1998/namespace}space")
            if xml_space_val is not None:
                clean_xs = xml_space_val.strip().lower()
                if clean_xs in ("default", "preserve"):
                    style.xml_space = clean_xs

        base_val = get_prop("dominant-baseline") or get_prop("alignment-baseline")
        if base_val is not None:
            clean_base = base_val.strip().lower()
            if clean_base not in ("auto", "alphabetic"):
                raise SVGImportError(
                    f"SVG baseline property '{base_val}' is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_BASELINE_PROPERTY",
                        severity="error",
                        message=f"Unsupported baseline property '{base_val}'",
                        attribute="dominant-baseline",
                    ),
                )

        return style


# -----------------------------------------------------------------------------
# Defs, Resource Templates, and Paint-Server Binding
# -----------------------------------------------------------------------------

RE_URL_REF = re.compile(r"^url\(\s*#([^)]+)\s*\)$")

@dataclass
class StopTemplate:
    offset: float
    color: Color

@dataclass
class LinearGradientTemplate:
    id: str
    x1: SVGLength = field(default_factory=lambda: SVGLength(0.0, "%"))
    y1: SVGLength = field(default_factory=lambda: SVGLength(0.0, "%"))
    x2: SVGLength = field(default_factory=lambda: SVGLength(100.0, "%"))
    y2: SVGLength = field(default_factory=lambda: SVGLength(0.0, "%"))
    gradientUnits: str = "objectBoundingBox"
    spreadMethod: str = "pad"
    gradientTransform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    stops: list[StopTemplate] = field(default_factory=list)
    href: str | None = None

@dataclass
class RadialGradientTemplate:
    id: str
    cx: SVGLength = field(default_factory=lambda: SVGLength(50.0, "%"))
    cy: SVGLength = field(default_factory=lambda: SVGLength(50.0, "%"))
    r: SVGLength = field(default_factory=lambda: SVGLength(50.0, "%"))
    fx: SVGLength | None = None
    fy: SVGLength | None = None
    fr: SVGLength | None = None
    gradientUnits: str = "objectBoundingBox"
    spreadMethod: str = "pad"
    gradientTransform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    stops: list[StopTemplate] = field(default_factory=list)
    href: str | None = None

@dataclass
class ClipPathTemplate:
    id: str
    clipPathUnits: str = "userSpaceOnUse"
    transform: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    element: ET.Element | None = None
    computed_style: ComputedStyle | None = None


class SVGDefsRegistry:
    """Manages Pass 1 resource indexing, tri-state cycle detection, and template resolution."""

    def __init__(self, limits: SVGImportLimits):
        self.limits = limits
        self.elements_by_id: dict[str, ET.Element] = {}
        self.gradient_templates: dict[str, LinearGradientTemplate | RadialGradientTemplate] = {}
        self.clip_templates: dict[str, ClipPathTemplate] = {}
        self.resolution_states: dict[str, str] = {}  # "UNRESOLVED", "RESOLVING", "RESOLVED"

    def register_element(self, element_id: str, element: ET.Element) -> None:
        if len(self.elements_by_id) >= self.limits.max_resources:
            raise SVGImportError(
                f"Resource count exceeded limit of {self.limits.max_resources}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Resource count exceeded limit"),
            )
        if element_id in self.elements_by_id:
            raise SVGImportError(
                f"Duplicate element id '{element_id}' found in SVG",
                diagnostic=SVGImportDiagnostic(code="SVG_DUPLICATE_ID", severity="error", message=f"Duplicate id '{element_id}'", source_id=element_id),
            )
        self.elements_by_id[element_id] = element

    def resolve_gradient_template(self, gradient_id: str, depth: int = 0) -> LinearGradientTemplate | RadialGradientTemplate:
        if depth > self.limits.max_reference_depth:
            raise SVGImportError(
                f"Gradient reference depth exceeded limit of {self.limits.max_reference_depth}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Reference recursion depth exceeded limit"),
            )
        if gradient_id not in self.gradient_templates:
            raise SVGImportError(
                f"Unresolved gradient reference '#{gradient_id}'",
                diagnostic=SVGImportDiagnostic(code="SVG_UNRESOLVED_REFERENCE", severity="error", message=f"Unresolved reference '#{gradient_id}'", source_id=gradient_id),
            )

        state = self.resolution_states.get(gradient_id, "UNRESOLVED")
        if state == "RESOLVING":
            raise SVGImportError(
                f"Cyclic reference detected in gradient '#{gradient_id}'",
                diagnostic=SVGImportDiagnostic(code="SVG_REFERENCE_CYCLE", severity="error", message=f"Reference cycle detected: '#{gradient_id}'", source_id=gradient_id),
            )
        if state == "RESOLVED":
            return self.gradient_templates[gradient_id]

        self.resolution_states[gradient_id] = "RESOLVING"
        tmpl = self.gradient_templates[gradient_id]

        if tmpl.href:
            href_id = tmpl.href.lstrip("#")
            base_tmpl = self.resolve_gradient_template(href_id, depth=depth + 1)
            # Inherit attributes if absent locally
            if not tmpl.stops and base_tmpl.stops:
                tmpl.stops = list(base_tmpl.stops)
            if "gradientUnits" not in tmpl.element_attribs and hasattr(base_tmpl, "gradientUnits"):
                tmpl.gradientUnits = base_tmpl.gradientUnits
            if "spreadMethod" not in tmpl.element_attribs and hasattr(base_tmpl, "spreadMethod"):
                tmpl.spreadMethod = base_tmpl.spreadMethod
            if "gradientTransform" not in tmpl.element_attribs and hasattr(base_tmpl, "gradientTransform"):
                tmpl.gradientTransform = base_tmpl.gradientTransform.copy()

            if isinstance(tmpl, LinearGradientTemplate) and isinstance(base_tmpl, LinearGradientTemplate):
                for coord in ("x1", "y1", "x2", "y2"):
                    if coord not in tmpl.element_attribs:
                        setattr(tmpl, coord, getattr(base_tmpl, coord))
            elif isinstance(tmpl, RadialGradientTemplate) and isinstance(base_tmpl, RadialGradientTemplate):
                for coord in ("cx", "cy", "r", "fx", "fy", "fr"):
                    if coord not in tmpl.element_attribs and getattr(base_tmpl, coord) is not None:
                        setattr(tmpl, coord, getattr(base_tmpl, coord))

        self.resolution_states[gradient_id] = "RESOLVED"
        return tmpl


def normalize_gradient_stops(stops_in: list[StopTemplate]) -> tuple[GradientStop, ...]:
    """Applies W3C SVG gradient stop normalization rules: clamping, monotonicity, and duplicates."""
    if not stops_in:
        return ()

    norm_stops: list[GradientStop] = []
    prev_offset = 0.0

    for s in stops_in:
        raw_off = max(0.0, min(1.0, float(s.offset)))
        eff_off = max(raw_off, prev_offset)
        prev_offset = eff_off
        norm_stops.append(GradientStop(eff_off, s.color))

    return tuple(norm_stops)


def bind_paint_server(
    ref_id: str,
    target_drawable: Drawable,
    defs_registry: SVGDefsRegistry,
    current_viewport: tuple[float, float] | SVGViewportContext,
    user_to_world_matrix: np.ndarray,
) -> PaintLike | None:
    """Binds an SVG gradient template to a PaintLike result (Color | LinearGradient | RadialGradient | None)."""
    tmpl = defs_registry.resolve_gradient_template(ref_id)
    raw_stops = normalize_gradient_stops(tmpl.stops)

    if len(raw_stops) == 0:
        return None  # SVG spec: 0 stops paints nothing
    if len(raw_stops) == 1:
        return raw_stops[0].color  # SVG spec: 1 stop paints solid color

    bbox = target_drawable.get_geometry_bounds()
    if isinstance(current_viewport, SVGViewportContext):
        V_w, V_h = current_viewport.width, current_viewport.height
        diag_norm = current_viewport.normalized_diagonal
    else:
        V_w, V_h = current_viewport
        diag_norm = math.sqrt(V_w * V_w + V_h * V_h) / math.sqrt(2.0)

    if tmpl.gradientUnits == "objectBoundingBox":
        if bbox.width <= 0.0 or bbox.height <= 0.0:
            return None  # W3C SVG: objectBoundingBox on zero-size geometry fails to paint

        M_bbox = np.array([
            [bbox.width, 0.0, bbox.x],
            [0.0, bbox.height, bbox.y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        M_paint = M_bbox @ tmpl.gradientTransform

        if isinstance(tmpl, LinearGradientTemplate):
            x1 = tmpl.x1.to_fraction(context_name="x1", limits=defs_registry.limits)
            y1 = tmpl.y1.to_fraction(context_name="y1", limits=defs_registry.limits)
            x2 = tmpl.x2.to_fraction(context_name="x2", limits=defs_registry.limits)
            y2 = tmpl.y2.to_fraction(context_name="y2", limits=defs_registry.limits)
            return LinearGradient(
                start=Point(x1, y1),
                end=Point(x2, y2),
                stops=raw_stops,
                space="object",
                spread=tmpl.spreadMethod,
                transform=Transform.from_matrix(M_paint),
            )
        else:  # RadialGradientTemplate
            cx = tmpl.cx.to_fraction(context_name="cx", limits=defs_registry.limits)
            cy = tmpl.cy.to_fraction(context_name="cy", limits=defs_registry.limits)
            r = tmpl.r.to_fraction(context_name="r", limits=defs_registry.limits)
            fx = tmpl.fx.to_fraction(context_name="fx", limits=defs_registry.limits) if tmpl.fx else cx
            fy = tmpl.fy.to_fraction(context_name="fy", limits=defs_registry.limits) if tmpl.fy else cy
            fr = tmpl.fr.to_fraction(context_name="fr", limits=defs_registry.limits) if tmpl.fr else 0.0

            if r < 0.0:
                raise SVGImportError(
                    f"Radial gradient radius 'r' cannot be negative: {r}",
                    diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient radius: {r}", attribute="r"),
                )
            if fr < 0.0:
                raise SVGImportError(
                    f"Radial gradient focal radius 'fr' cannot be negative: {fr}",
                    diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient focal radius: {fr}", attribute="fr"),
                )
            if r == 0.0:
                return raw_stops[-1].color  # W3C SVG spec: r == 0 paints solid color of the last stop

            if not (math.isclose(fx, cx, abs_tol=1e-6) and math.isclose(fy, cy, abs_tol=1e-6) and math.isclose(fr, 0.0, abs_tol=1e-6)):
                raise SVGImportError(
                    f"Eccentric radial gradient (fx={fx}, fy={fy}, fr={fr}) is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_RADIAL_FOCUS", severity="error", message="Eccentric radial focus is unsupported"),
                )
            return RadialGradient(
                center=Point(cx, cy),
                radius=r,
                stops=raw_stops,
                space="object",
                spread=tmpl.spreadMethod,
                transform=Transform.from_matrix(M_paint),
            )

    else:  # userSpaceOnUse
        M_paint = user_to_world_matrix @ tmpl.gradientTransform

        if isinstance(tmpl, LinearGradientTemplate):
            x1 = tmpl.x1.to_absolute(reference_length=V_w, context_name="x1", limits=defs_registry.limits)
            y1 = tmpl.y1.to_absolute(reference_length=V_h, context_name="y1", limits=defs_registry.limits)
            x2 = tmpl.x2.to_absolute(reference_length=V_w, context_name="x2", limits=defs_registry.limits)
            y2 = tmpl.y2.to_absolute(reference_length=V_h, context_name="y2", limits=defs_registry.limits)
            return LinearGradient(
                start=Point(x1, y1),
                end=Point(x2, y2),
                stops=raw_stops,
                space="world",
                spread=tmpl.spreadMethod,
                transform=Transform.from_matrix(M_paint),
            )
        else:  # RadialGradientTemplate
            cx = tmpl.cx.to_absolute(reference_length=V_w, context_name="cx", limits=defs_registry.limits)
            cy = tmpl.cy.to_absolute(reference_length=V_h, context_name="cy", limits=defs_registry.limits)
            r = tmpl.r.to_absolute(reference_length=diag_norm, context_name="r", limits=defs_registry.limits)
            fx = tmpl.fx.to_absolute(reference_length=V_w, context_name="fx", limits=defs_registry.limits) if tmpl.fx else cx
            fy = tmpl.fy.to_absolute(reference_length=V_h, context_name="fy", limits=defs_registry.limits) if tmpl.fy else cy
            fr = tmpl.fr.to_absolute(reference_length=diag_norm, context_name="fr", limits=defs_registry.limits) if tmpl.fr else 0.0

            if r < 0.0:
                raise SVGImportError(
                    f"Radial gradient radius 'r' cannot be negative: {r}",
                    diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient radius: {r}", attribute="r"),
                )
            if fr < 0.0:
                raise SVGImportError(
                    f"Radial gradient focal radius 'fr' cannot be negative: {fr}",
                    diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient focal radius: {fr}", attribute="fr"),
                )
            if r == 0.0:
                return raw_stops[-1].color  # W3C SVG spec: r == 0 paints solid color of the last stop

            if not (math.isclose(fx, cx, abs_tol=1e-6) and math.isclose(fy, cy, abs_tol=1e-6) and math.isclose(fr, 0.0, abs_tol=1e-6)):
                raise SVGImportError(
                    f"Eccentric radial gradient (fx={fx}, fy={fy}, fr={fr}) is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_RADIAL_FOCUS", severity="error", message="Eccentric radial focus is unsupported"),
                )
            return RadialGradient(
                center=Point(cx, cy),
                radius=r,
                stops=raw_stops,
                space="world",
                spread=tmpl.spreadMethod,
                transform=Transform.from_matrix(M_paint),
            )


# -----------------------------------------------------------------------------
# SVG Text Stream Whitespace Normalization
# -----------------------------------------------------------------------------

@dataclass
class SVGTextSegment:
    text: str
    element: ET.Element
    style: ComputedStyle
    is_element_start: bool = False
    x: float | None = None
    y: float | None = None
    dx: float | None = None
    dy: float | None = None


def normalize_svg_text_stream(
    segments: list[SVGTextSegment],
    parent_map: dict[ET.Element, ET.Element] | None = None,
) -> list[SVGTextSegment]:
    """Normalize whitespace across flattened XML text content stream according to SVG/XML rules.

    In xml:space="default":
      - Newlines (\\r, \\n) are removed (NOT converted to spaces).
      - Tabs (\\t) become spaces.
      - Leading and trailing spaces are removed.
      - Contiguous spaces across node boundaries are collapsed to a single space.
    In xml:space="preserve":
      - Newlines and tabs become spaces and all spaces are preserved.
    """
    # 1. Transform newlines and tabs
    transformed: list[SVGTextSegment] = []
    for seg in segments:
        text = seg.text
        if seg.style.xml_space == "preserve":
            t = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
        else:
            t = text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", " ")
        transformed.append(SVGTextSegment(t, seg.element, seg.style, seg.is_element_start, seg.x, seg.y, seg.dx, seg.dy))

    # 2. Collapse contiguous spaces and remove leading spaces across stream
    collapsed: list[SVGTextSegment] = []
    at_start = True
    last_was_space = False
    for seg in transformed:
        if seg.style.xml_space == "preserve":
            collapsed.append(seg)
            if seg.text:
                at_start = False
                last_was_space = seg.text.endswith(" ")
        else:
            chars = []
            for ch in seg.text:
                if ch == " ":
                    if at_start or last_was_space:
                        continue
                    chars.append(" ")
                    last_was_space = True
                else:
                    chars.append(ch)
                    at_start = False
                    last_was_space = False
            collapsed.append(SVGTextSegment("".join(chars), seg.element, seg.style, seg.is_element_start, seg.x, seg.y, seg.dx, seg.dy))

    if parent_map is None and segments:
        all_elements = {seg.element for seg in segments}
        parent_map = {child: parent for parent in all_elements for child in parent}

    # 3. Strip trailing spaces if default mode
    final: list[SVGTextSegment] = []
    trailing = True
    for seg in reversed(collapsed):
        if trailing and seg.style.xml_space == "default":
            t = seg.text.rstrip(" ")
            if t:
                trailing = False
            final.append(SVGTextSegment(t, seg.element, seg.style, seg.is_element_start, seg.x, seg.y, seg.dx, seg.dy))
        else:
            final.append(seg)
            if seg.text:
                trailing = False

    # 4. Propagate pending positioning forward to first addressable descendant text segment within owner
    def is_descendant(child: ET.Element, ancestor: ET.Element) -> bool:
        if parent_map is None:
            return True
        curr: ET.Element | None = child
        while curr is not None:
            if curr is ancestor:
                return True
            curr = parent_map.get(curr)
        return False

    pending_x_stack: list[tuple[ET.Element, float]] = []
    pending_y_stack: list[tuple[ET.Element, float]] = []
    pending_dx_stack: list[tuple[ET.Element, float]] = []
    pending_dy_stack: list[tuple[ET.Element, float]] = []

    ordered = list(reversed(final))
    result: list[SVGTextSegment] = []
    for seg in ordered:
        pending_x_stack = [entry for entry in pending_x_stack if is_descendant(seg.element, entry[0])]
        pending_y_stack = [entry for entry in pending_y_stack if is_descendant(seg.element, entry[0])]
        pending_dx_stack = [entry for entry in pending_dx_stack if is_descendant(seg.element, entry[0])]
        pending_dy_stack = [entry for entry in pending_dy_stack if is_descendant(seg.element, entry[0])]

        if seg.x is not None:
            pending_x_stack.append((seg.element, seg.x))
        if seg.y is not None:
            pending_y_stack.append((seg.element, seg.y))
        if seg.dx is not None:
            pending_dx_stack.append((seg.element, seg.dx))
        if seg.dy is not None:
            pending_dy_stack.append((seg.element, seg.dy))

        if not seg.text:
            continue

        run_x = pending_x_stack[-1][1] if pending_x_stack else None
        pending_x_stack.clear()

        run_y = pending_y_stack[-1][1] if pending_y_stack else None
        pending_y_stack.clear()

        run_dx = pending_dx_stack[-1][1] if pending_dx_stack else None
        pending_dx_stack.clear()

        run_dy = pending_dy_stack[-1][1] if pending_dy_stack else None
        pending_dy_stack.clear()

        result.append(SVGTextSegment(
            text=seg.text,
            element=seg.element,
            style=seg.style,
            is_element_start=seg.is_element_start,
            x=run_x,
            y=run_y,
            dx=run_dx,
            dy=run_dy,
        ))

    return result


# -----------------------------------------------------------------------------
# Main SVG Importer Architecture
# -----------------------------------------------------------------------------

class SVGImporter:
    """Retained-mode SVG document importer with strict retained DrawCV mapping."""

    def __init__(
        self,
        *,
        strict: bool = True,
        strict_fonts: bool = True,
        viewport: tuple[int, int] | None = None,
        limits: SVGImportLimits | None = None,
        font_resolver: FontResolver | None = None,
    ):
        if not strict:
            raise NotImplementedError("Permissive SVG import (strict=False) is not yet implemented in Milestone 1")
        self.strict = strict
        self.strict_fonts = strict_fonts
        self.viewport = viewport
        self.limits = limits if limits is not None else SVGImportLimits()
        self.font_resolver = font_resolver

    def parse_file(self, filepath: str | FilePath) -> SVGImportResult:
        p = FilePath(filepath)
        if p.stat().st_size > self.limits.max_source_bytes:
            raise SVGImportError(
                f"SVG source file size ({p.stat().st_size} bytes) exceeds limit of {self.limits.max_source_bytes}",
                diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Source size exceeded limit"),
            )
        data = p.read_bytes()
        return self.parse(data)

    def parse(self, source: str | bytes) -> SVGImportResult:
        if isinstance(source, bytes):
            if len(source) > self.limits.max_source_bytes:
                raise SVGImportError(
                    f"SVG source size ({len(source)} bytes) exceeds limit of {self.limits.max_source_bytes}",
                    diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Source size exceeded limit"),
                )
            text = source.decode("utf-8")
        else:
            if len(source.encode("utf-8")) > self.limits.max_source_bytes:
                raise SVGImportError(
                    f"SVG source size exceeds limit of {self.limits.max_source_bytes}",
                    diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Source size exceeded limit"),
                )
            text = source

        # Strict security check: disallow DTD / entity declarations before XML parsing
        if "<!DOCTYPE" in text or "<!ENTITY" in text:
            raise SVGImportError(
                "XML DOCTYPE and ENTITY declarations are forbidden for security reasons",
                diagnostic=SVGImportDiagnostic(code="SVG_SECURITY_VIOLATION", severity="error", message="DOCTYPE/ENTITY forbidden"),
            )

        try:
            root = ET.fromstring(text)
        except ET.ParseError as err:
            raise SVGImportError(
                f"Malformed SVG XML: {err}",
                diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_XML", severity="error", message=str(err)),
            ) from err

        # Verify root element
        tag_name = extract_element_tag(root.tag)
        if tag_name.lower() != "svg":
            raise SVGImportError(
                f"Root XML element must be <svg>, got <{tag_name}>",
                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_ELEMENT", severity="error", message=f"Root element must be <svg>, got <{tag_name}>"),
            )

        # ---------------------------------------------------------------------
        # Viewport and ViewBox Processing
        # ---------------------------------------------------------------------
        w_attr = root.attrib.get("width")
        h_attr = root.attrib.get("height")
        vb_attr = root.attrib.get("viewBox")

        if self.viewport is not None:
            vw, vh = self.viewport
            if not (isinstance(vw, (int, float)) and isinstance(vh, (int, float))):
                raise SVGImportError("Caller viewport dimensions must be numeric")
            if abs(vw - round(vw)) >= 1e-6 or abs(vh - round(vh)) >= 1e-6:
                raise SVGImportError(
                    f"Non-integer caller viewport dimensions ({vw}, {vh}) are not permitted in strict mode",
                    diagnostic=SVGImportDiagnostic(code="SVG_NONINTEGER_VIEWPORT", severity="error", message="Non-integer caller viewport dimensions"),
                )
            scene_w, scene_h = int(round(vw)), int(round(vh))
        elif w_attr is not None and h_attr is not None:
            len_w = SVGLength.parse(w_attr, limits=self.limits)
            len_h = SVGLength.parse(h_attr, limits=self.limits)
            if len_w.unit == "%" or len_h.unit == "%":
                raise SVGImportError(
                    "Root percentage width/height requires caller viewport=(W, H)",
                    diagnostic=SVGImportDiagnostic(code="SVG_INVALID_DIMENSIONS", severity="error", message="Root percentage requires caller viewport"),
                )
            val_w = len_w.to_absolute(context_name="root width", limits=self.limits)
            val_h = len_h.to_absolute(context_name="root height", limits=self.limits)

            if abs(val_w - round(val_w)) >= 1e-6 or abs(val_h - round(val_h)) >= 1e-6:
                raise SVGImportError(
                    f"Non-integer root viewport dimensions ({val_w}, {val_h}) are not permitted in strict mode",
                    diagnostic=SVGImportDiagnostic(code="SVG_NONINTEGER_VIEWPORT", severity="error", message="Non-integer root viewport dimensions"),
                )
            scene_w, scene_h = int(round(val_w)), int(round(val_h))
        else:
            raise SVGImportError(
                "Omitted root width/height requires caller viewport=(W, H)",
                diagnostic=SVGImportDiagnostic(code="SVG_INVALID_DIMENSIONS", severity="error", message="Root width/height omitted"),
            )

        if scene_w <= 0 or scene_h <= 0 or scene_w > self.limits.max_scene_dimension or scene_h > self.limits.max_scene_dimension:
            raise SVGImportError(
                f"Invalid root scene dimensions: ({scene_w}, {scene_h})",
                diagnostic=SVGImportDiagnostic(code="SVG_INVALID_DIMENSIONS", severity="error", message="Dimensions out of valid range"),
            )

        # Calculate viewBox transformation and root viewport context
        M_viewbox, (eff_w, eff_h) = compute_viewbox_matrix(
            vb_attr, float(scene_w), float(scene_h),
            root.attrib.get("preserveAspectRatio", "xMidYMid meet"),
            limits=self.limits,
        )
        effective_viewport = (eff_w, eff_h)
        root_viewport = SVGViewportContext(width=eff_w, height=eff_h)

        scene = Scene(width=scene_w, height=scene_h, background=Color(0, 0, 0, 0))
        defs_registry = SVGDefsRegistry(self.limits)

        # ---------------------------------------------------------------------
        # Pass 1: Index IDs, defs, and resource templates
        # ---------------------------------------------------------------------
        total_elements = 0

        def traverse_pass1(elem: ET.Element, current_depth: int, parent_style: ComputedStyle) -> None:
            nonlocal total_elements
            total_elements += 1
            if total_elements > self.limits.max_elements:
                raise SVGImportError(
                    f"Element count exceeded limit of {self.limits.max_elements}",
                    diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Element count exceeded limit"),
                )
            if current_depth > self.limits.max_nesting_depth:
                raise SVGImportError(
                    f"Nesting depth exceeded limit of {self.limits.max_nesting_depth}",
                    diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Nesting depth exceeded limit"),
                )

            # Check attribute length limit
            for k, v in elem.attrib.items():
                if len(v) > self.limits.max_attribute_length:
                    raise SVGImportError(
                        f"Attribute '{k}' length ({len(v)}) exceeds limit of {self.limits.max_attribute_length}",
                        diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Attribute length exceeded limit"),
                    )

            elem_id = elem.attrib.get("id")
            if elem_id:
                defs_registry.register_element(elem_id, elem)

            tag = extract_element_tag(elem.tag)
            tag_lower = tag.lower()

            # Reject explicitly deferred elements in strict mode
            if tag_lower in ("mask",):
                raise SVGImportError(
                    "SVG <mask> is not supported in Milestone 1 (requires retained vector mask)",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_MASK", severity="error", message="SVG <mask> unsupported", element_tag=tag),
                )
            if tag_lower in ("filter",):
                raise SVGImportError(
                    "SVG <filter> is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_FILTER", severity="error", message="SVG <filter> unsupported", element_tag=tag),
                )
            if tag_lower == "textpath":
                raise SVGImportError(
                    "SVG <textPath> is not supported",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TEXTPATH", severity="error", message="SVG <textPath> unsupported", element_tag=tag),
                )
            if tag_lower in ("image",):
                raise SVGImportError(
                    "SVG <image> is deferred in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_IMAGE", severity="error", message="SVG <image> deferred", element_tag=tag),
                )
            if tag_lower in ("pattern",):
                raise SVGImportError(
                    "SVG <pattern> is deferred in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_PATTERN", severity="error", message="SVG <pattern> deferred", element_tag=tag),
                )

            # Check external URLs in href attributes
            for attr_name in ("href", f"{{{XLINK_NS}}}href"):
                val = elem.attrib.get(attr_name)
                if val and not val.startswith("#"):
                    raise SVGImportError(
                        f"External reference '{val}' is forbidden in offline SVG import",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_EXTERNAL_REFERENCE", severity="error", message=f"External reference '{val}' forbidden"),
                    )

            # Compute style context for this resource tree
            style = SVGStyleResolver.resolve(elem, parent_style, limits=self.limits)

            # Index gradient templates
            if tag_lower == "lineargradient" and elem_id:
                gu = elem.attrib.get("gradientUnits", "objectBoundingBox")
                if gu not in ("objectBoundingBox", "userSpaceOnUse"):
                    raise SVGImportError(
                        f"Unsupported gradientUnits: '{gu}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_GRADIENT_UNITS", severity="error", message=f"Unsupported gradientUnits: '{gu}'"),
                    )
                sm = elem.attrib.get("spreadMethod", "pad")
                if sm not in ("pad", "reflect", "repeat"):
                    raise SVGImportError(
                        f"Unsupported spreadMethod: '{sm}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_SPREAD_METHOD", severity="error", message=f"Unsupported spreadMethod: '{sm}'"),
                    )
                tf_mat = SVGTransformParser.parse_to_matrix(elem.attrib.get("gradientTransform", ""), limits=self.limits)
                href_val = elem.attrib.get("href", elem.attrib.get(f"{{{XLINK_NS}}}href"))
                tmpl = LinearGradientTemplate(
                    id=elem_id,
                    x1=SVGLength.parse(elem.attrib.get("x1", "0%"), limits=self.limits),
                    y1=SVGLength.parse(elem.attrib.get("y1", "0%"), limits=self.limits),
                    x2=SVGLength.parse(elem.attrib.get("x2", "100%"), limits=self.limits),
                    y2=SVGLength.parse(elem.attrib.get("y2", "0%"), limits=self.limits),
                    gradientUnits=gu,
                    spreadMethod=sm,
                    gradientTransform=tf_mat,
                    stops=[],
                    href=href_val,
                )
                tmpl.element_attribs = {k.lower(): v for k, v in elem.attrib.items()}
                # Parse stop children
                for child in elem:
                    c_tag = extract_element_tag(child.tag).lower()
                    if c_tag == "stop":
                        c_style = SVGStyleResolver.resolve(child, style, limits=self.limits)
                        off_len = SVGLength.parse(child.attrib.get("offset", "0"), limits=self.limits)
                        off_val = off_len.value / 100.0 if off_len.unit == "%" else off_len.value
                        stop_col = SVGColorParser.parse(child.attrib.get("stop-color", "black"), current_color=c_style.color) or Color.black()
                        stop_op = max(0.0, min(1.0, float(child.attrib.get("stop-opacity", "1.0"))))
                        # Inline style overrides
                        in_decls = SVGStyleResolver.parse_inline_style(child.attrib.get("style", ""))
                        if "stop-color" in in_decls:
                            stop_col = SVGColorParser.parse(in_decls["stop-color"], current_color=c_style.color) or Color.black()
                        if "stop-opacity" in in_decls:
                            stop_op = max(0.0, min(1.0, float(in_decls["stop-opacity"])))
                        merged_col = stop_col.with_alpha(stop_col.a * stop_op)
                        tmpl.stops.append(StopTemplate(offset=off_val, color=merged_col))

                defs_registry.gradient_templates[elem_id] = tmpl

            elif tag_lower == "radialgradient" and elem_id:
                gu = elem.attrib.get("gradientUnits", "objectBoundingBox")
                if gu not in ("objectBoundingBox", "userSpaceOnUse"):
                    raise SVGImportError(
                        f"Unsupported gradientUnits: '{gu}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_GRADIENT_UNITS", severity="error", message=f"Unsupported gradientUnits: '{gu}'"),
                    )
                sm = elem.attrib.get("spreadMethod", "pad")
                if sm not in ("pad", "reflect", "repeat"):
                    raise SVGImportError(
                        f"Unsupported spreadMethod: '{sm}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_SPREAD_METHOD", severity="error", message=f"Unsupported spreadMethod: '{sm}'"),
                    )
                tf_mat = SVGTransformParser.parse_to_matrix(elem.attrib.get("gradientTransform", ""), limits=self.limits)
                href_val = elem.attrib.get("href", elem.attrib.get(f"{{{XLINK_NS}}}href"))
                r_str = elem.attrib.get("r", "50%")
                parsed_r = SVGLength.parse(r_str, limits=self.limits)
                if parsed_r.value < 0:
                    raise SVGImportError(
                        f"Radial gradient radius 'r' cannot be negative: '{r_str}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient radius: '{r_str}'", attribute="r"),
                    )

                parsed_fr = None
                if "fr" in elem.attrib:
                    fr_str = elem.attrib["fr"]
                    parsed_fr = SVGLength.parse(fr_str, limits=self.limits)
                    if parsed_fr.value < 0:
                        raise SVGImportError(
                            f"Radial gradient focal radius 'fr' cannot be negative: '{fr_str}'",
                            diagnostic=SVGImportDiagnostic(code="SVG_INVALID_GRADIENT_RADIUS", severity="error", message=f"Negative radial gradient focal radius: '{fr_str}'", attribute="fr"),
                        )

                tmpl = RadialGradientTemplate(
                    id=elem_id,
                    cx=SVGLength.parse(elem.attrib.get("cx", "50%"), limits=self.limits),
                    cy=SVGLength.parse(elem.attrib.get("cy", "50%"), limits=self.limits),
                    r=parsed_r,
                    fx=SVGLength.parse(elem.attrib["fx"], limits=self.limits) if "fx" in elem.attrib else None,
                    fy=SVGLength.parse(elem.attrib["fy"], limits=self.limits) if "fy" in elem.attrib else None,
                    fr=parsed_fr,
                    gradientUnits=gu,
                    spreadMethod=sm,
                    gradientTransform=tf_mat,
                    stops=[],
                    href=href_val,
                )
                tmpl.element_attribs = {k.lower(): v for k, v in elem.attrib.items()}
                for child in elem:
                    c_tag = extract_element_tag(child.tag).lower()
                    if c_tag == "stop":
                        c_style = SVGStyleResolver.resolve(child, style, limits=self.limits)
                        off_len = SVGLength.parse(child.attrib.get("offset", "0"), limits=self.limits)
                        off_val = off_len.value / 100.0 if off_len.unit == "%" else off_len.value
                        stop_col = SVGColorParser.parse(child.attrib.get("stop-color", "black"), current_color=c_style.color) or Color.black()
                        stop_op = max(0.0, min(1.0, float(child.attrib.get("stop-opacity", "1.0"))))
                        in_decls = SVGStyleResolver.parse_inline_style(child.attrib.get("style", ""))
                        if "stop-color" in in_decls:
                            stop_col = SVGColorParser.parse(in_decls["stop-color"], current_color=c_style.color) or Color.black()
                        if "stop-opacity" in in_decls:
                            stop_op = max(0.0, min(1.0, float(in_decls["stop-opacity"])))
                        merged_col = stop_col.with_alpha(stop_col.a * stop_op)
                        tmpl.stops.append(StopTemplate(offset=off_val, color=merged_col))

                defs_registry.gradient_templates[elem_id] = tmpl

            elif tag_lower == "clippath" and elem_id:
                clip_units = elem.attrib.get("clipPathUnits", "userSpaceOnUse")
                if clip_units not in ("userSpaceOnUse", "objectBoundingBox"):
                    raise SVGImportError(
                        f"Unsupported clipPathUnits: '{clip_units}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_PATH_UNITS", severity="error", message=f"Unsupported clipPathUnits: '{clip_units}'"),
                    )
                tf_mat = SVGTransformParser.parse_to_matrix(elem.attrib.get("transform", ""), limits=self.limits)

                # Filter visual children inside clipPath
                visual_children = []
                for child in elem:
                    ct = extract_element_tag(child.tag).lower()
                    if ct in ("desc", "title", "metadata"):
                        continue
                    visual_children.append((ct, child))

                if len(visual_children) == 0:
                    raise SVGImportError(
                        f"Empty <clipPath id='{elem_id}'> is unsupported in Milestone 1",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message="Empty clipPath"),
                    )
                if len(visual_children) > 1:
                    raise SVGImportError(
                        f"<clipPath id='{elem_id}'> with multiple children ({len(visual_children)}) is unsupported in Milestone 1",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message="Multiple clip children unsupported"),
                    )

                child_tag, child_elem = visual_children[0]
                if child_tag not in ("path", "rect", "polygon", "polyline", "circle", "ellipse"):
                    raise SVGImportError(
                        f"Unsupported clipPath child geometry <{child_tag}> (line and grouped clips are unsupported)",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message=f"Unsupported clip geometry <{child_tag}>"),
                    )

                defs_registry.clip_templates[elem_id] = ClipPathTemplate(
                    id=elem_id,
                    clipPathUnits=clip_units,
                    transform=tf_mat,
                    element=child_elem,
                    computed_style=style,
                )

            for child in elem:
                traverse_pass1(child, current_depth + 1, style)

        traverse_pass1(root, 0, ComputedStyle())

        # ---------------------------------------------------------------------
        # Pass 2: Visible Scene Graph Construction
        # ---------------------------------------------------------------------
        root_drawables: list[Drawable] = []
        use_instance_count = 0
        expanded_element_count = 0
        visiting_ids: list[str] = []

        def on_materialized(node: Drawable) -> Drawable:
            nonlocal expanded_element_count
            if visiting_ids:
                expanded_element_count += 1
                if expanded_element_count > self.limits.max_expanded_elements:
                    raise SVGImportError(
                        f"Expanded element count exceeded limit of {self.limits.max_expanded_elements}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_RESOURCE_LIMIT_EXCEEDED",
                            severity="error",
                            message="Expanded element count exceeded limit",
                            element_tag="use",
                        ),
                    )
            return node

        def check_affine_singular_for_arcs(matrix: np.ndarray, element_tag: str = "path") -> None:
            """Validate that an affine transformation matrix is non-singular and well-conditioned for elliptical arcs."""
            A = np.asarray(matrix[:2, :2], dtype=np.float64)
            det_A = float(np.linalg.det(A))
            if abs(det_A) < 1e-12 or float(np.linalg.cond(A)) > 1e10:
                raise SVGImportError(
                    f"Singular or near-singular transform (det={det_A:.2e}) on EllipticalArcTo cannot be represented",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_TRANSFORM",
                        severity="error",
                        message="Singular or near-singular transform on EllipticalArcTo",
                        element_tag=element_tag,
                    ),
                )

        def drawable_has_arcs(d: Drawable) -> bool:
            if isinstance(d, Path):
                return any(isinstance(cmd, EllipticalArcTo) for sub in d.subpaths for cmd in sub.commands)
            return False

        def apply_clip_path(
            target_node: Drawable,
            node_style: ComputedStyle,
            owner_ctm: np.ndarray,
            owner_viewport: SVGViewportContext,
        ) -> None:
            if not node_style.clip_path:
                return

            m_clip = RE_URL_REF.match(node_style.clip_path)
            if not m_clip:
                raise SVGImportError(
                    f"Unsupported or malformed clip-path syntax: '{node_style.clip_path}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_CLIP_PATH",
                        severity="error",
                        message=f"Unsupported clip-path '{node_style.clip_path}'",
                        attribute="clip-path",
                    ),
                )
            clip_id = m_clip.group(1)
            if clip_id not in defs_registry.clip_templates:
                raise SVGImportError(
                    f"Unresolved clipPath reference '#{clip_id}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNRESOLVED_REFERENCE",
                        severity="error",
                        message=f"Unresolved clipPath '#{clip_id}'",
                    ),
                )
            cp_tmpl = defs_registry.clip_templates[clip_id]
            if cp_tmpl.element is None:
                raise SVGImportError(
                    f"ClipPath '#{clip_id}' has no geometry",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_CLIP_GEOMETRY",
                        severity="error",
                        message="ClipPath has no geometry",
                    ),
                )

            # For objectBoundingBox clip content, percentage geometry resolves in normalized [0, 1] x [0, 1] space
            if cp_tmpl.clipPathUnits == "objectBoundingBox":
                clip_vp = SVGViewportContext(width=1.0, height=1.0)
            else:
                clip_vp = owner_viewport

            clip_style = SVGStyleResolver.resolve(cp_tmpl.element, cp_tmpl.computed_style, limits=self.limits)
            clip_d = build_element(cp_tmpl.element, cp_tmpl.computed_style, np.eye(3, dtype=np.float64), clip_vp)
            if clip_d is None:
                raise SVGImportError(
                    f"ClipPath '#{clip_id}' geometry produced no drawable",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_CLIP_GEOMETRY",
                        severity="error",
                        message="ClipPath geometry is empty",
                    ),
                )

            # Compose transforms: M_child is clip_d's own local transform
            M_child = clip_d.transform.get_matrix(Point(0, 0))
            M_clip_base = cp_tmpl.transform @ M_child

            if cp_tmpl.clipPathUnits == "objectBoundingBox":
                bbox = target_node.get_geometry_bounds()
                M_obb = np.array([
                    [bbox.width, 0.0, bbox.x],
                    [0.0, bbox.height, bbox.y],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                M_clip_total = M_obb @ M_clip_base
            else:
                M_clip_total = M_clip_base

            clip_has_arcs = False
            if isinstance(clip_d, Path):
                clip_d.transform = Transform.from_matrix(M_clip_total)
                clip_d.fill_rule = clip_style.clip_rule
                target_node.clip = clip_d
                clip_has_arcs = drawable_has_arcs(clip_d)
            elif isinstance(clip_d, Rectangle) and np.allclose(M_clip_total, np.eye(3)):
                target_node.clip = ClipRect(clip_d.position.x, clip_d.position.y, clip_d.width, clip_d.height)
                clip_has_arcs = False
            elif isinstance(clip_d, Rectangle):
                p_clip = Path(fill_rule=clip_style.clip_rule)
                p_clip.move_to(Point(clip_d.position.x, clip_d.position.y))
                p_clip.line_to(Point(clip_d.position.x + clip_d.width, clip_d.position.y))
                p_clip.line_to(Point(clip_d.position.x + clip_d.width, clip_d.position.y + clip_d.height))
                p_clip.line_to(Point(clip_d.position.x, clip_d.position.y + clip_d.height))
                p_clip.close()
                p_clip.transform = Transform.from_matrix(M_clip_total)
                target_node.clip = p_clip
                clip_has_arcs = False
            elif isinstance(clip_d, RoundedRectangle):
                rx = clip_d.corner_radius
                ry = clip_d.corner_radius
                x, y, w, h = clip_d.x, clip_d.y, clip_d.width, clip_d.height
                p_clip = Path(fill_rule=clip_style.clip_rule)
                if rx <= 0.0 or ry <= 0.0:
                    p_clip.move_to(Point(x, y))
                    p_clip.line_to(Point(x + w, y))
                    p_clip.line_to(Point(x + w, y + h))
                    p_clip.line_to(Point(x, y + h))
                    p_clip.close()
                    p_clip.transform = Transform.from_matrix(M_clip_total)
                    target_node.clip = p_clip
                    clip_has_arcs = False
                else:
                    p_clip.move_to(Point(x + rx, y))
                    p_clip.line_to(Point(x + w - rx, y))
                    p_clip.arc_to(rx, ry, 0.0, False, True, x + w, y + ry)
                    p_clip.line_to(Point(x + w, y + h - ry))
                    p_clip.arc_to(rx, ry, 0.0, False, True, x + w - rx, y + h)
                    p_clip.line_to(Point(x + rx, y + h))
                    p_clip.arc_to(rx, ry, 0.0, False, True, x, y + h - ry)
                    p_clip.line_to(Point(x, y + ry))
                    p_clip.arc_to(rx, ry, 0.0, False, True, x + rx, y)
                    p_clip.close()
                    p_clip.transform = Transform.from_matrix(M_clip_total)
                    target_node.clip = p_clip
                    clip_has_arcs = True
            elif isinstance(clip_d, Circle):
                p_clip = Path(fill_rule=clip_style.clip_rule)
                cx, cy, r = clip_d.center.x, clip_d.center.y, clip_d.radius
                p_clip.move_to(Point(cx + r, cy))
                p_clip.arc_to(r, r, 0.0, False, True, cx - r, cy)
                p_clip.arc_to(r, r, 0.0, False, True, cx + r, cy)
                p_clip.close()
                p_clip.transform = Transform.from_matrix(M_clip_total)
                target_node.clip = p_clip
                clip_has_arcs = True
            elif isinstance(clip_d, Ellipse):
                p_clip = Path(fill_rule=clip_style.clip_rule)
                cx, cy, rx, ry = clip_d.center.x, clip_d.center.y, clip_d.radius_x, clip_d.radius_y
                p_clip.move_to(Point(cx + rx, cy))
                p_clip.arc_to(rx, ry, 0.0, False, True, cx - rx, cy)
                p_clip.arc_to(rx, ry, 0.0, False, True, cx + rx, cy)
                p_clip.close()
                p_clip.transform = Transform.from_matrix(M_clip_total)
                target_node.clip = p_clip
                clip_has_arcs = True
            elif isinstance(clip_d, Polyline):
                p_clip = Path(fill_rule=clip_style.clip_rule)
                if clip_d.points:
                    p_clip.move_to(clip_d.points[0])
                    for pt in clip_d.points[1:]:
                        p_clip.line_to(pt)
                p_clip.close()
                p_clip.transform = Transform.from_matrix(M_clip_total)
                target_node.clip = p_clip
                clip_has_arcs = False
            else:
                raise SVGImportError(
                    f"Unsupported clip geometry '{type(clip_d).__name__}'",
                    diagnostic=SVGImportDiagnostic(
                        code="SVG_UNSUPPORTED_CLIP_GEOMETRY",
                        severity="error",
                        message=f"Unsupported clip geometry '{type(clip_d).__name__}'",
                    ),
                )

            # Strict singular check covering clip arcs transformed into world coordinates during export
            if clip_has_arcs:
                check_affine_singular_for_arcs(owner_ctm @ M_clip_total, element_tag="clipPath")

        def build_element(
            elem: ET.Element,
            parent_style: ComputedStyle,
            cumulative_ctm: np.ndarray,
            current_viewport: SVGViewportContext,
        ) -> Drawable | None:
            nonlocal use_instance_count, expanded_element_count
            tag = extract_element_tag(elem.tag)
            tag_lower = tag.lower()

            # Skip non-visual elements in visible tree
            if tag_lower in ("defs", "lineargradient", "radialgradient", "clippath", "title", "desc", "metadata", "symbol"):
                return None

            style = SVGStyleResolver.resolve(elem, parent_style, limits=self.limits)
            if style.display == "none":
                return None

            # Local transform on this element
            tf_mat = SVGTransformParser.parse_to_matrix(elem.attrib.get("transform", ""), limits=self.limits)
            element_ctm = cumulative_ctm @ tf_mat

            # Parse geometry
            drawable: Drawable | None = None

            if tag_lower == "g":
                children_drawables: list[Drawable] = []
                for child in elem:
                    ch_d = build_element(child, style, element_ctm, current_viewport)
                    if ch_d is not None:
                        children_drawables.append(ch_d)
                drawable = Group(
                    children=children_drawables,
                    transform=Transform.from_matrix(tf_mat),
                    opacity=style.opacity,
                    blend_mode=style.mix_blend_mode,
                    visible=True,  # Keeps group open so visibility:visible children can render
                )

            elif tag_lower == "svg":
                # Nested <svg> establishing a new viewport
                x = current_viewport.resolve_x(
                    SVGLength.parse(elem.attrib.get("x", "0"), limits=self.limits),
                    context_name="svg x",
                    limits=self.limits,
                )
                y = current_viewport.resolve_y(
                    SVGLength.parse(elem.attrib.get("y", "0"), limits=self.limits),
                    context_name="svg y",
                    limits=self.limits,
                )
                w_attr = elem.attrib.get("width")
                h_attr = elem.attrib.get("height")
                if w_attr is not None and w_attr.strip().lower() != "auto":
                    W = current_viewport.resolve_x(SVGLength.parse(w_attr, limits=self.limits), context_name="svg width", limits=self.limits)
                else:
                    W = current_viewport.width
                if h_attr is not None and h_attr.strip().lower() != "auto":
                    H = current_viewport.resolve_y(SVGLength.parse(h_attr, limits=self.limits), context_name="svg height", limits=self.limits)
                else:
                    H = current_viewport.height

                if W < 0 or H < 0:
                    raise SVGImportError(
                        f"Negative viewport dimension in <svg>: width={W}, height={H}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_INVALID_ATTRIBUTE_VALUE",
                            severity="error",
                            message=f"Negative viewport dimension: width={W}, height={H}",
                            element_tag="svg",
                            attribute="width" if W < 0 else "height",
                        ),
                    )
                if W == 0 or H == 0:
                    return None

                T_xy = np.array([
                    [1.0, 0.0, float(x)],
                    [0.0, 1.0, float(y)],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                T_host = tf_mat @ T_xy

                eff_ov = style.overflow or "hidden"  # UA default for nested svg is hidden
                if eff_ov in ("hidden", "scroll"):
                    viewport_clip = ClipRect(0.0, 0.0, W, H)
                else:
                    viewport_clip = None

                vb_attr = elem.attrib.get("viewBox")
                par_attr = elem.attrib.get("preserveAspectRatio", "xMidYMid meet")
                M_viewbox, inner_vp_dims = compute_viewbox_matrix(vb_attr, W, H, par_attr, limits=self.limits)
                inner_viewport = SVGViewportContext(width=inner_vp_dims[0], height=inner_vp_dims[1])

                inner_ctm = element_ctm @ T_xy @ M_viewbox
                children_drawables = []
                for child in elem:
                    ch_d = build_element(child, style, inner_ctm, inner_viewport)
                    if ch_d is not None:
                        children_drawables.append(ch_d)

                view_box_group = on_materialized(Group(
                    children=children_drawables,
                    transform=Transform.from_matrix(M_viewbox),
                    visible=True,
                ))
                viewport_group = on_materialized(Group(
                    children=[view_box_group],
                    transform=Transform(),
                    clip=viewport_clip,
                    visible=True,
                ))
                drawable = Group(
                    children=[viewport_group],
                    transform=Transform.from_matrix(T_host),
                    opacity=style.opacity,
                    blend_mode=style.mix_blend_mode,
                    visible=True,
                )

            elif tag_lower == "use":
                use_instance_count += 1
                if use_instance_count > self.limits.max_use_instances:
                    raise SVGImportError(
                        f"<use> instance count exceeded limit of {self.limits.max_use_instances}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_RESOURCE_LIMIT_EXCEEDED",
                            severity="error",
                            message="Use instance count exceeded limit",
                            element_tag="use",
                        ),
                    )

                href_val = elem.attrib.get("href") or elem.attrib.get(f"{{{XLINK_NS}}}href")
                if not href_val:
                    return None
                if not href_val.startswith("#"):
                    raise SVGImportError(
                        f"External reference '{href_val}' is forbidden in offline SVG import",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNSUPPORTED_EXTERNAL_REFERENCE",
                            severity="error",
                            message=f"External reference '{href_val}' forbidden",
                            element_tag="use",
                        ),
                    )

                target_id = href_val.lstrip("#")
                if target_id in visiting_ids:
                    raise SVGImportError(
                        f"Reference cycle detected in <use>: '#{target_id}'",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_REFERENCE_CYCLE",
                            severity="error",
                            message=f"Reference cycle detected: '#{target_id}'",
                            element_tag="use",
                            source_id=target_id,
                        ),
                    )
                if len(visiting_ids) >= self.limits.max_reference_depth:
                    raise SVGImportError(
                        f"Reference depth exceeded limit of {self.limits.max_reference_depth}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_RESOURCE_LIMIT_EXCEEDED",
                            severity="error",
                            message="Reference depth exceeded limit",
                            element_tag="use",
                            source_id=target_id,
                        ),
                    )

                if target_id not in defs_registry.elements_by_id:
                    raise SVGImportError(
                        f"Unresolved reference '#{target_id}' in <use>",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNRESOLVED_REFERENCE",
                            severity="error",
                            message=f"Unresolved reference '#{target_id}'",
                            element_tag="use",
                            source_id=target_id,
                        ),
                    )

                target_elem = defs_registry.elements_by_id[target_id]
                target_tag = extract_element_tag(target_elem.tag).lower()

                use_x = current_viewport.resolve_x(
                    SVGLength.parse(elem.attrib.get("x", "0"), limits=self.limits),
                    context_name="use x",
                    limits=self.limits,
                )
                use_y = current_viewport.resolve_y(
                    SVGLength.parse(elem.attrib.get("y", "0"), limits=self.limits),
                    context_name="use y",
                    limits=self.limits,
                )

                use_w_attr = elem.attrib.get("width")
                use_h_attr = elem.attrib.get("height")
                if use_w_attr is not None and use_w_attr.strip().lower() != "auto":
                    w_check = current_viewport.resolve_x(SVGLength.parse(use_w_attr, limits=self.limits), context_name="use width", limits=self.limits)
                    if w_check < 0:
                        raise SVGImportError(
                            f"Negative width on <use>: {w_check}",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_INVALID_ATTRIBUTE_VALUE",
                                severity="error",
                                message=f"Negative width on <use>: {w_check}",
                                element_tag="use",
                                attribute="width",
                            ),
                        )
                if use_h_attr is not None and use_h_attr.strip().lower() != "auto":
                    h_check = current_viewport.resolve_y(SVGLength.parse(use_h_attr, limits=self.limits), context_name="use height", limits=self.limits)
                    if h_check < 0:
                        raise SVGImportError(
                            f"Negative height on <use>: {h_check}",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_INVALID_ATTRIBUTE_VALUE",
                                severity="error",
                                message=f"Negative height on <use>: {h_check}",
                                element_tag="use",
                                attribute="height",
                            ),
                        )

                T_xy = np.array([
                    [1.0, 0.0, float(use_x)],
                    [0.0, 1.0, float(use_y)],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                T_host = tf_mat @ T_xy

                inherited_style = style.inherit_child()
                visiting_ids.append(target_id)
                try:
                    if target_tag in ("symbol", "svg"):
                        explicit_use_w = (use_w_attr is not None and use_w_attr.strip().lower() != "auto")
                        explicit_use_h = (use_h_attr is not None and use_h_attr.strip().lower() != "auto")

                        if explicit_use_w:
                            W = current_viewport.resolve_x(
                                SVGLength.parse(use_w_attr, limits=self.limits),
                                context_name="use width",
                                limits=self.limits,
                            )
                        else:
                            tgt_w_attr = target_elem.attrib.get("width")
                            if tgt_w_attr is not None and tgt_w_attr.strip().lower() != "auto":
                                W = current_viewport.resolve_x(
                                    SVGLength.parse(tgt_w_attr, limits=self.limits),
                                    context_name="target width",
                                    limits=self.limits,
                                )
                            else:
                                W = current_viewport.width

                        if explicit_use_h:
                            H = current_viewport.resolve_y(
                                SVGLength.parse(use_h_attr, limits=self.limits),
                                context_name="use height",
                                limits=self.limits,
                            )
                        else:
                            tgt_h_attr = target_elem.attrib.get("height")
                            if tgt_h_attr is not None and tgt_h_attr.strip().lower() != "auto":
                                H = current_viewport.resolve_y(
                                    SVGLength.parse(tgt_h_attr, limits=self.limits),
                                    context_name="target height",
                                    limits=self.limits,
                                )
                            else:
                                H = current_viewport.height

                        if W < 0 or H < 0:
                            raise SVGImportError(
                                f"Negative viewport dimension in <use>: width={W}, height={H}",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_INVALID_ATTRIBUTE_VALUE",
                                    severity="error",
                                    message=f"Negative viewport dimension in <use>: width={W}, height={H}",
                                    element_tag="use",
                                    attribute="width" if W < 0 else "height",
                                ),
                            )
                        if W == 0 or H == 0:
                            return None

                        tgt_tf_mat = SVGTransformParser.parse_to_matrix(target_elem.attrib.get("transform", ""), limits=self.limits)
                        sym_x = current_viewport.resolve_x(
                            SVGLength.parse(target_elem.attrib.get("x", "0"), limits=self.limits),
                            context_name="target x",
                            limits=self.limits,
                        )
                        sym_y = current_viewport.resolve_y(
                            SVGLength.parse(target_elem.attrib.get("y", "0"), limits=self.limits),
                            context_name="target y",
                            limits=self.limits,
                        )
                        if sym_x != 0.0 or sym_y != 0.0:
                            T_sym = np.array([
                                [1.0, 0.0, float(sym_x)],
                                [0.0, 1.0, float(sym_y)],
                                [0.0, 0.0, 1.0],
                            ], dtype=np.float64)
                        else:
                            T_sym = np.eye(3, dtype=np.float64)

                        T_tgt_total = tgt_tf_mat @ T_sym

                        tgt_style = SVGStyleResolver.resolve(target_elem, inherited_style, limits=self.limits)
                        if target_tag == "svg" and tgt_style.display == "none":
                            return None
                        if target_tag == "symbol":
                            tgt_style.display = "inline"
                        # Do not let non-inherited overflow from <use> inherit into referenced svg/symbol
                        eff_ov = tgt_style.overflow or "hidden"

                        if eff_ov in ("hidden", "scroll"):
                            viewport_clip = ClipRect(0.0, 0.0, W, H)
                        else:
                            viewport_clip = None

                        vb_attr = target_elem.attrib.get("viewBox")
                        par_attr = target_elem.attrib.get("preserveAspectRatio", "xMidYMid meet")
                        M_viewbox, inner_vp_dims = compute_viewbox_matrix(
                            vb_attr, W, H, par_attr, limits=self.limits
                        )
                        inner_viewport = SVGViewportContext(width=inner_vp_dims[0], height=inner_vp_dims[1])

                        inner_ctm = element_ctm @ T_xy @ T_tgt_total @ M_viewbox
                        instantiated_children = []
                        for child in target_elem:
                            ch_d = build_element(child, tgt_style, inner_ctm, inner_viewport)
                            if ch_d is not None:
                                instantiated_children.append(ch_d)

                        view_box_group = on_materialized(Group(
                            children=instantiated_children,
                            transform=Transform.from_matrix(M_viewbox),
                            visible=True,
                        ))
                        viewport_group = on_materialized(Group(
                            children=[view_box_group],
                            transform=Transform(),
                            clip=viewport_clip,
                            visible=True,
                        ))

                        target_has_state = (
                            not np.allclose(T_tgt_total, np.eye(3))
                            or tgt_style.opacity != 1.0
                            or tgt_style.mix_blend_mode != BlendMode.NORMAL
                            or tgt_style.clip_path is not None
                        )
                        if target_has_state:
                            target_host = on_materialized(Group(
                                children=[viewport_group],
                                transform=Transform.from_matrix(T_tgt_total),
                                opacity=tgt_style.opacity,
                                blend_mode=tgt_style.mix_blend_mode,
                                visible=True,
                            ))
                            if tgt_style.clip_path:
                                target_ctm = element_ctm @ T_xy @ T_tgt_total
                                apply_clip_path(target_host, tgt_style, target_ctm, current_viewport)
                            top_child = target_host
                        else:
                            top_child = viewport_group

                        drawable = Group(
                            children=[top_child],
                            transform=Transform.from_matrix(T_host),
                            opacity=style.opacity,
                            blend_mode=style.mix_blend_mode,
                            visible=True,
                        )

                    else:
                        inner_ctm = element_ctm @ T_xy
                        target_drawable = build_element(target_elem, inherited_style, inner_ctm, current_viewport)
                        if target_drawable is None:
                            return None

                        drawable = Group(
                            children=[target_drawable],
                            transform=Transform.from_matrix(T_host),
                            opacity=style.opacity,
                            blend_mode=style.mix_blend_mode,
                            visible=True,
                        )

                    source_id = elem.attrib.get("id")
                    if source_id:
                        drawable.name = source_id
                        drawable.metadata["svg:id"] = source_id
                    drawable.metadata["svg:tag"] = tag

                    if style.clip_path:
                        apply_clip_path(drawable, style, element_ctm, current_viewport)

                    return on_materialized(drawable)
                finally:
                    visiting_ids.pop()

            elif tag_lower == "path":
                d_str = elem.attrib.get("d", "")
                drawable = SVGPathParser.parse(d_str, fill_rule=style.fill_rule, limits=self.limits)
                if drawable_has_arcs(drawable):
                    check_affine_singular_for_arcs(element_ctm, element_tag="path")
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "rect":
                x = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("x", "0"), limits=self.limits), context_name="rect x", limits=self.limits)
                y = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("y", "0"), limits=self.limits), context_name="rect y", limits=self.limits)
                w = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("width", "0"), limits=self.limits), context_name="rect width", limits=self.limits)
                h = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("height", "0"), limits=self.limits), context_name="rect height", limits=self.limits)

                if w < 0 or h < 0:
                    raise SVGImportError(
                        f"Rectangle dimensions must be non-negative: width={w}, height={h}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_INVALID_ATTRIBUTE_VALUE",
                            severity="error",
                            message=f"Negative rectangle dimension: width={w}, height={h}",
                            element_tag="rect",
                            attribute="width" if w < 0 else "height",
                        ),
                    )
                if w == 0 or h == 0:
                    return None  # SVG spec: zero-dimension rect is omitted

                rx_attr = elem.attrib.get("rx")
                ry_attr = elem.attrib.get("ry")
                rx_is_auto = (rx_attr is None or rx_attr.strip().lower() == "auto")
                ry_is_auto = (ry_attr is None or ry_attr.strip().lower() == "auto")

                if rx_is_auto and ry_is_auto:
                    drawable = Rectangle(position=Point(x, y), width=w, height=h)
                else:
                    rx_val: float | None = None
                    if not rx_is_auto:
                        rx_len = SVGLength.parse(rx_attr, limits=self.limits)
                        rx_val = rx_len.to_absolute(reference_length=w, context_name="rx", limits=self.limits)
                        if rx_val < 0:
                            raise SVGImportError(
                                f"Negative rect rx: {rx_val}",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_INVALID_ATTRIBUTE_VALUE",
                                    severity="error",
                                    message=f"Negative rx: {rx_val}",
                                    element_tag="rect",
                                    attribute="rx",
                                ),
                            )
                    ry_val: float | None = None
                    if not ry_is_auto:
                        ry_len = SVGLength.parse(ry_attr, limits=self.limits)
                        ry_val = ry_len.to_absolute(reference_length=h, context_name="ry", limits=self.limits)
                        if ry_val < 0:
                            raise SVGImportError(
                                f"Negative rect ry: {ry_val}",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_INVALID_ATTRIBUTE_VALUE",
                                    severity="error",
                                    message=f"Negative ry: {ry_val}",
                                    element_tag="rect",
                                    attribute="ry",
                                ),
                            )

                    eff_rx = rx_val if rx_val is not None else ry_val
                    eff_ry = ry_val if ry_val is not None else rx_val

                    if eff_rx == 0.0 or eff_ry == 0.0:
                        drawable = Rectangle(position=Point(x, y), width=w, height=h)
                    else:
                        # Clamp per W3C SVG used-value rules (50% max)
                        eff_rx = min(eff_rx, w / 2.0)
                        eff_ry = min(eff_ry, h / 2.0)

                        if math.isclose(eff_rx, eff_ry, rel_tol=1e-7, abs_tol=1e-7):
                            drawable = RoundedRectangle(x=x, y=y, width=w, height=h, corner_radius=eff_rx)
                        else:
                            # Non-uniform rounded rect using exact EllipticalArcTo
                            p = Path(fill_rule=style.fill_rule)
                            p.move_to(Point(x + eff_rx, y))
                            p.line_to(Point(x + w - eff_rx, y))
                            p.arc_to(eff_rx, eff_ry, 0.0, False, True, x + w, y + eff_ry)
                            p.line_to(Point(x + w, y + h - eff_ry))
                            p.arc_to(eff_rx, eff_ry, 0.0, False, True, x + w - eff_rx, y + h)
                            p.line_to(Point(x + eff_rx, y + h))
                            p.arc_to(eff_rx, eff_ry, 0.0, False, True, x, y + h - eff_ry)
                            p.line_to(Point(x, y + eff_ry))
                            p.arc_to(eff_rx, eff_ry, 0.0, False, True, x + eff_rx, y)
                            p.close()
                            check_affine_singular_for_arcs(element_ctm, element_tag="rect")
                            drawable = p
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "circle":
                cx = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("cx", "0"), limits=self.limits), context_name="circle cx", limits=self.limits)
                cy = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("cy", "0"), limits=self.limits), context_name="circle cy", limits=self.limits)
                r = current_viewport.resolve_diagonal(SVGLength.parse(elem.attrib.get("r", "0"), limits=self.limits), context_name="circle r", limits=self.limits)
                if r < 0:
                    raise SVGImportError(
                        f"Circle radius must be non-negative: {r}",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_INVALID_ATTRIBUTE_VALUE",
                            severity="error",
                            message=f"Negative circle radius: {r}",
                            element_tag="circle",
                            attribute="r",
                        ),
                    )
                if r == 0:
                    return None
                drawable = Circle(center=Point(cx, cy), radius=r)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "ellipse":
                cx = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("cx", "0"), limits=self.limits), context_name="ellipse cx", limits=self.limits)
                cy = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("cy", "0"), limits=self.limits), context_name="ellipse cy", limits=self.limits)

                rx_attr = elem.attrib.get("rx")
                ry_attr = elem.attrib.get("ry")
                rx_is_auto = (rx_attr is None or rx_attr.strip().lower() == "auto")
                ry_is_auto = (ry_attr is None or ry_attr.strip().lower() == "auto")

                if rx_is_auto and ry_is_auto:
                    return None  # SVG2: both auto -> non-rendering ellipse

                rx_val: float | None = None
                if not rx_is_auto:
                    rx_len = SVGLength.parse(rx_attr, limits=self.limits)
                    rx_val = current_viewport.resolve_x(rx_len, context_name="ellipse rx", limits=self.limits)
                    if rx_val < 0:
                        raise SVGImportError(
                            f"Ellipse rx must be non-negative: rx={rx_val}",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_INVALID_ATTRIBUTE_VALUE",
                                severity="error",
                                message=f"Negative ellipse radius: rx={rx_val}",
                                element_tag="ellipse",
                                attribute="rx",
                            ),
                        )

                ry_val: float | None = None
                if not ry_is_auto:
                    ry_len = SVGLength.parse(ry_attr, limits=self.limits)
                    ry_val = current_viewport.resolve_y(ry_len, context_name="ellipse ry", limits=self.limits)
                    if ry_val < 0:
                        raise SVGImportError(
                            f"Ellipse ry must be non-negative: ry={ry_val}",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_INVALID_ATTRIBUTE_VALUE",
                                severity="error",
                                message=f"Negative ellipse radius: ry={ry_val}",
                                element_tag="ellipse",
                                attribute="ry",
                            ),
                        )

                eff_rx = rx_val if rx_val is not None else ry_val
                eff_ry = ry_val if ry_val is not None else rx_val

                if eff_rx <= 0 or eff_ry <= 0:
                    return None

                drawable = Ellipse(center=Point(cx, cy), radius_x=eff_rx, radius_y=eff_ry)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "line":
                x1 = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("x1", "0"), limits=self.limits), context_name="line x1", limits=self.limits)
                y1 = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("y1", "0"), limits=self.limits), context_name="line y1", limits=self.limits)
                x2 = current_viewport.resolve_x(SVGLength.parse(elem.attrib.get("x2", "0"), limits=self.limits), context_name="line x2", limits=self.limits)
                y2 = current_viewport.resolve_y(SVGLength.parse(elem.attrib.get("y2", "0"), limits=self.limits), context_name="line y2", limits=self.limits)
                drawable = Line(start=Point(x1, y1), end=Point(x2, y2))
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "polyline":
                pts_str = elem.attrib.get("points", "")
                num_tokens = [float(t) for t in re.split(r"[\s,]+", pts_str.strip()) if t]
                if len(num_tokens) % 2 != 0:
                    raise SVGImportError("Polyline points list must contain an even number of coordinates")
                for c in num_tokens:
                    if not math.isfinite(c):
                        raise SVGImportError(
                            f"Non-finite coordinate in polyline points: {c}",
                            diagnostic=SVGImportDiagnostic(code="SVG_NON_FINITE_COORDINATE", severity="error", message=f"Non-finite coordinate: {c}"),
                        )
                    if abs(c) > self.limits.max_coordinate_magnitude:
                        raise SVGImportError(
                            f"Coordinate magnitude {c} exceeds limit of {self.limits.max_coordinate_magnitude}",
                            diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
                        )
                pts = [Point(num_tokens[i], num_tokens[i+1]) for i in range(0, len(num_tokens), 2)]
                if len(pts) > self.limits.max_points_per_poly:
                    raise SVGImportError("Polyline point count exceeded limit")

                # If fill is enabled, map to open Path to preserve open stroke with closed fill
                if style.fill and style.fill.lower() != "none" and len(pts) >= 2:
                    p = Path(fill_rule=style.fill_rule)
                    p.move_to(pts[0])
                    for pt in pts[1:]:
                        p.line_to(pt)
                    drawable = p
                else:
                    if len(pts) < 2:
                        return None
                    drawable = Polyline(points=pts, closed=False)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "polygon":
                pts_str = elem.attrib.get("points", "")
                num_tokens = [float(t) for t in re.split(r"[\s,]+", pts_str.strip()) if t]
                if len(num_tokens) % 2 != 0:
                    raise SVGImportError("Polygon points list must contain an even number of coordinates")
                for c in num_tokens:
                    if not math.isfinite(c):
                        raise SVGImportError(
                            f"Non-finite coordinate in polygon points: {c}",
                            diagnostic=SVGImportDiagnostic(code="SVG_NON_FINITE_COORDINATE", severity="error", message=f"Non-finite coordinate: {c}"),
                        )
                    if abs(c) > self.limits.max_coordinate_magnitude:
                        raise SVGImportError(
                            f"Coordinate magnitude {c} exceeds limit of {self.limits.max_coordinate_magnitude}",
                            diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
                        )
                pts = [Point(num_tokens[i], num_tokens[i+1]) for i in range(0, len(num_tokens), 2)]
                if len(pts) > self.limits.max_points_per_poly:
                    raise SVGImportError("Polygon point count exceeded limit")
                if len(pts) < 3:
                    return None

                # Closed Path mapping for robust winding and stroke join handling
                p = Path(fill_rule=style.fill_rule)
                p.move_to(pts[0])
                for pt in pts[1:]:
                    p.line_to(pt)
                p.close()
                drawable = p
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "text":
                if style.stroke and style.stroke.lower() != "none":
                    raise SVGImportError(
                        "SVG text stroke is not supported in Milestone 1",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNSUPPORTED_TEXT_STROKE",
                            severity="error",
                            message="Text stroke is not supported",
                            element_tag="text",
                        ),
                    )
                if style.fill and style.fill.startswith("url("):
                    raise SVGImportError(
                        "SVG text paint servers are not supported in Milestone 1",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNSUPPORTED_TEXT_PAINT_SERVER",
                            severity="error",
                            message="Text paint server is not supported",
                            element_tag="text",
                        ),
                    )

                def _check_coord_list(attr_name: str, val: str | None, tag_name: str):
                    if val is not None and len(re.split(r"[\s,]+", val.strip())) > 1:
                        raise SVGImportError(
                            f"Multi-value coordinate lists on SVG {tag_name} are not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_TEXT_COORDINATE_LIST",
                                severity="error",
                                message=f"Multi-value coordinate list '{val}' on {attr_name} is not supported",
                                element_tag=tag_name,
                                attribute=attr_name,
                            ),
                        )

                def _validate_text_element(node: ET.Element, tag_name: str) -> None:
                    node_attrs = {k.lower(): v.strip() for k, v in node.attrib.items()}
                    node_inline = SVGStyleResolver.parse_inline_style(node_attrs.get("style", ""))

                    def get_val(key: str) -> str | None:
                        return node_inline.get(key, node_attrs.get(key))

                    if tag_name == "tspan":
                        op_val = get_val("opacity")
                        if op_val is not None:
                            try:
                                op_f = float(op_val)
                            except (ValueError, TypeError):
                                op_f = None
                            if op_f is not None and op_f < 1.0:
                                raise SVGImportError(
                                    "Per-tspan opacity is not supported in Milestone 1 (use fill-opacity)",
                                    diagnostic=SVGImportDiagnostic(
                                        code="SVG_UNSUPPORTED_TEXT_OPACITY",
                                        severity="error",
                                        message="Per-tspan opacity is not supported",
                                        element_tag="tspan",
                                        attribute="opacity",
                                    ),
                                )

                    rot_val = get_val("rotate")
                    if rot_val is not None and rot_val.strip() not in ("", "0", "0deg", "0rad"):
                        raise SVGImportError(
                            "SVG text rotate is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_TEXT_ROTATE",
                                severity="error",
                                message="Text rotate is not supported",
                                element_tag=tag_name,
                                attribute="rotate",
                            ),
                        )

                    tl_val = get_val("textlength")
                    if tl_val is not None:
                        raise SVGImportError(
                            "SVG textLength is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_TEXT_LENGTH",
                                severity="error",
                                message="SVG textLength is not supported",
                                element_tag=tag_name,
                                attribute="textLength",
                            ),
                        )

                    la_val = get_val("lengthadjust")
                    if la_val is not None:
                        raise SVGImportError(
                            "SVG lengthAdjust is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_TEXT_LENGTH_ADJUST",
                                severity="error",
                                message="SVG lengthAdjust is not supported",
                                element_tag=tag_name,
                                attribute="lengthAdjust",
                            ),
                        )

                    ls_val = get_val("letter-spacing")
                    if ls_val is not None and ls_val.strip().lower() not in ("normal", "0", "0px", ""):
                        raise SVGImportError(
                            "SVG letter-spacing is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_LETTER_SPACING",
                                severity="error",
                                message="letter-spacing is not supported",
                                element_tag=tag_name,
                                attribute="letter-spacing",
                            ),
                        )

                    ws_val = get_val("word-spacing")
                    if ws_val is not None and ws_val.strip().lower() not in ("normal", "0", "0px", ""):
                        raise SVGImportError(
                            "SVG word-spacing is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_WORD_SPACING",
                                severity="error",
                                message="word-spacing is not supported",
                                element_tag=tag_name,
                                attribute="word-spacing",
                            ),
                        )

                    td_val = get_val("text-decoration")
                    if td_val is not None and td_val.strip().lower() not in ("none", ""):
                        raise SVGImportError(
                            "SVG text-decoration is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_TEXT_DECORATION",
                                severity="error",
                                message="text-decoration is not supported",
                                element_tag=tag_name,
                                attribute="text-decoration",
                            ),
                        )

                    wm_val = get_val("writing-mode")
                    if wm_val is not None and wm_val.strip().lower() not in ("horizontal-tb", "lr", "lr-tb", ""):
                        raise SVGImportError(
                            f"SVG writing-mode '{wm_val}' is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNSUPPORTED_WRITING_MODE",
                                severity="error",
                                message=f"writing-mode '{wm_val}' is not supported",
                                element_tag=tag_name,
                                attribute="writing-mode",
                            ),
                        )

                    for base_prop in ("alignment-baseline", "dominant-baseline", "baseline-shift"):
                        bp_val = get_val(base_prop)
                        if bp_val is not None and bp_val.strip().lower() not in ("auto", "alphabetic", "baseline", "0", "0px", ""):
                            raise SVGImportError(
                                f"SVG {base_prop} is not supported in Milestone 1",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_UNSUPPORTED_BASELINE_PROPERTY",
                                    severity="error",
                                    message=f"{base_prop} is not supported",
                                    element_tag=tag_name,
                                    attribute=base_prop,
                                ),
                            )

                    _check_coord_list("x", node_attrs.get("x"), tag_name)
                    _check_coord_list("y", node_attrs.get("y"), tag_name)
                    _check_coord_list("dx", node_attrs.get("dx"), tag_name)
                    _check_coord_list("dy", node_attrs.get("dy"), tag_name)

                _check_coord_list("x", elem.attrib.get("x"), "text")
                _check_coord_list("y", elem.attrib.get("y"), "text")
                _check_coord_list("dx", elem.attrib.get("dx"), "text")
                _check_coord_list("dy", elem.attrib.get("dy"), "text")

                text_x = current_viewport.resolve_x(
                    SVGLength.parse(elem.attrib.get("x", "0"), limits=self.limits),
                    context_name="text x",
                    limits=self.limits,
                ) if elem.attrib.get("x") else 0.0

                text_y = current_viewport.resolve_y(
                    SVGLength.parse(elem.attrib.get("y", "0"), limits=self.limits),
                    context_name="text y",
                    limits=self.limits,
                ) if elem.attrib.get("y") else 0.0

                text_dx = current_viewport.resolve_x(
                    SVGLength.parse(elem.attrib.get("dx", "0"), limits=self.limits),
                    context_name="text dx",
                    limits=self.limits,
                ) if elem.attrib.get("dx") else 0.0

                text_dy = current_viewport.resolve_y(
                    SVGLength.parse(elem.attrib.get("dy", "0"), limits=self.limits),
                    context_name="text dy",
                    limits=self.limits,
                ) if elem.attrib.get("dy") else 0.0

                text_x += text_dx
                text_y += text_dy

                raw_segments: list[SVGTextSegment] = []

                def collect_text_tree(node: ET.Element, current_style: ComputedStyle, is_root: bool = False) -> None:
                    node_tag = extract_element_tag(node.tag).lower()
                    if not is_root:
                        if node_tag == "textpath":
                            raise SVGImportError(
                                "SVG <textPath> is not supported in Milestone 1",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_UNSUPPORTED_TEXTPATH",
                                    severity="error",
                                    message="SVG <textPath> is not supported",
                                    element_tag="textPath",
                                ),
                            )
                        if node_tag != "tspan":
                            raise SVGImportError(
                                f"Unsupported child element <{node_tag}> in <text>",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_UNSUPPORTED_ELEMENT",
                                    severity="error",
                                    message=f"Unsupported child element <{node_tag}> in <text>",
                                    element_tag=node_tag,
                                ),
                            )
                        _validate_text_element(node, "tspan")
                        node_attrs = {k.lower(): v.strip() for k, v in node.attrib.items()}
                        node_inline = SVGStyleResolver.parse_inline_style(node_attrs.get("style", ""))
                        dir_val = node_inline.get("direction", node_attrs.get("direction"))
                        if dir_val is not None:
                            if "x" not in node_attrs and "y" not in node_attrs:
                                raise SVGImportError(
                                    "Per-tspan direction without absolute positioning (x or y) is not supported in Milestone 1",
                                    diagnostic=SVGImportDiagnostic(
                                        code="SVG_UNSUPPORTED_TEXT_DIRECTION",
                                        severity="error",
                                        message="Per-tspan direction without absolute positioning is not supported",
                                        element_tag="tspan",
                                        attribute="direction",
                                    ),
                                )
                        node_style = SVGStyleResolver.resolve(node, current_style, limits=self.limits)
                        if node_style.stroke and node_style.stroke.lower() != "none":
                            raise SVGImportError(
                                "SVG text stroke is not supported in Milestone 1",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_UNSUPPORTED_TEXT_STROKE",
                                    severity="error",
                                    message="Text stroke is not supported",
                                    element_tag="tspan",
                                ),
                            )
                        if node_style.fill and node_style.fill.startswith("url("):
                            raise SVGImportError(
                                "SVG text paint servers are not supported in Milestone 1",
                                diagnostic=SVGImportDiagnostic(
                                    code="SVG_UNSUPPORTED_TEXT_PAINT_SERVER",
                                    severity="error",
                                    message="Text paint server is not supported",
                                    element_tag="tspan",
                                ),
                            )
                        x_str = node.attrib.get("x")
                        y_str = node.attrib.get("y")
                        dx_str = node.attrib.get("dx")
                        dy_str = node.attrib.get("dy")
                        elem_x = current_viewport.resolve_x(SVGLength.parse(x_str, limits=self.limits), context_name="tspan x", limits=self.limits) if x_str is not None else None
                        elem_y = current_viewport.resolve_y(SVGLength.parse(y_str, limits=self.limits), context_name="tspan y", limits=self.limits) if y_str is not None else None
                        elem_dx = current_viewport.resolve_x(SVGLength.parse(dx_str, limits=self.limits), context_name="tspan dx", limits=self.limits) if dx_str is not None else None
                        elem_dy = current_viewport.resolve_y(SVGLength.parse(dy_str, limits=self.limits), context_name="tspan dy", limits=self.limits) if dy_str is not None else None
                    else:
                        _validate_text_element(node, "text")
                        node_style = current_style
                        elem_x = None
                        elem_y = None
                        elem_dx = None
                        elem_dy = None

                    if node.text:
                        raw_segments.append(SVGTextSegment(node.text, node, node_style, is_element_start=True, x=elem_x, y=elem_y, dx=elem_dx, dy=elem_dy))
                    elif not is_root and (elem_x is not None or elem_y is not None or elem_dx is not None or elem_dy is not None):
                        raw_segments.append(SVGTextSegment("", node, node_style, is_element_start=True, x=elem_x, y=elem_y, dx=elem_dx, dy=elem_dy))

                    for child in node:
                        collect_text_tree(child, node_style, is_root=False)
                        if child.tail:
                            raw_segments.append(SVGTextSegment(child.tail, node, node_style, is_element_start=False))

                collect_text_tree(elem, style, is_root=True)

                if self.font_resolver is None:
                    raise SVGImportError(
                        "SVG text import requires an injected FontResolver",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_MISSING_FONT_RESOLVER",
                            severity="error",
                            message="SVG text import requires an injected FontResolver",
                            element_tag="text",
                        ),
                    )

                parent_map = {child: parent for parent in elem.iter() for child in parent}
                norm_segments = normalize_svg_text_stream(raw_segments, parent_map=parent_map)
                runs: list[TextRun] = []

                root_fill_none = (style.fill is not None and style.fill.strip().lower() == "none")
                root_color = SVGColorParser.parse(style.fill, current_color=style.color) if (style.fill and not root_fill_none) else Color.black()
                root_fill_op = style.fill_opacity
                root_font_size = style.font_size
                root_font_weight = style.font_weight
                root_font_style = style.font_style
                root_direction = style.direction
                root_xml_space = style.xml_space
                root_text_anchor = TextAnchor(style.text_anchor)

                root_desc = self.font_resolver.resolve(style.font_family, weight=style.font_weight, style=style.font_style)
                if root_desc is None:
                    raise SVGImportError(
                        f"Font family '{style.font_family}' could not be resolved",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNRESOLVED_FONT",
                            severity="error",
                            message=f"Font family '{style.font_family}' could not be resolved",
                            element_tag="text",
                        ),
                    )
                if (self.strict and self.strict_fonts) and root_desc.is_substituted:
                    raise SVGImportError(
                        f"Font family '{style.font_family}' was substituted and is disallowed in strict mode",
                        diagnostic=SVGImportDiagnostic(
                            code="SVG_UNRESOLVED_FONT",
                            severity="error",
                            message=f"Font family '{style.font_family}' was substituted",
                            element_tag="text",
                        ),
                    )
                root_fam_name = root_desc.semantic_family
                root_fonts = root_desc.fonts
                root_is_sub = root_desc.is_substituted

                for seg in norm_segments:
                    if not seg.text:
                        continue
                    seg_elem = seg.element
                    seg_style = seg.style
                    run_x: float | None = seg.x
                    run_y: float | None = seg.y
                    run_dx = seg.dx if seg.dx is not None else 0.0
                    run_dy = seg.dy if seg.dy is not None else 0.0

                    seg_desc = self.font_resolver.resolve(seg_style.font_family, weight=seg_style.font_weight, style=seg_style.font_style)
                    if seg_desc is None:
                        raise SVGImportError(
                            f"Font family '{seg_style.font_family}' could not be resolved",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNRESOLVED_FONT",
                                severity="error",
                                message=f"Font family '{seg_style.font_family}' could not be resolved",
                                element_tag=extract_element_tag(seg_elem.tag),
                            ),
                        )
                    if (self.strict and self.strict_fonts) and seg_desc.is_substituted:
                        raise SVGImportError(
                            f"Font family '{seg_style.font_family}' was substituted and is disallowed in strict mode",
                            diagnostic=SVGImportDiagnostic(
                                code="SVG_UNRESOLVED_FONT",
                                severity="error",
                                message=f"Font family '{seg_style.font_family}' was substituted",
                                element_tag=extract_element_tag(seg_elem.tag),
                            ),
                        )
                    run_fonts = seg_desc.fonts
                    run_fam = seg_desc.semantic_family
                    run_sub = seg_desc.is_substituted

                    seg_is_none = (seg_style.fill is not None and seg_style.fill.strip().lower() == "none")
                    if seg_is_none:
                        if not root_fill_none:
                            seg_fill = None
                            seg_fill_none = True
                        else:
                            seg_fill = None
                            seg_fill_none = False
                    else:
                        seg_c = SVGColorParser.parse(seg_style.fill, current_color=seg_style.color) if seg_style.fill else Color.black()
                        if root_fill_none or seg_c != root_color:
                            seg_fill = seg_c
                            seg_fill_none = False
                        else:
                            seg_fill = None
                            seg_fill_none = False

                    if not math.isclose(seg_style.fill_opacity, root_fill_op, abs_tol=1e-5):
                        seg_fill_op = seg_style.fill_opacity
                    else:
                        seg_fill_op = None

                    run_font_size = seg_style.font_size if not math.isclose(seg_style.font_size, root_font_size, abs_tol=1e-5) else None
                    run_fam_override = run_fam if run_fam != root_fam_name else None
                    run_weight_override = seg_style.font_weight if seg_style.font_weight != root_font_weight else None
                    run_style_override = seg_style.font_style if seg_style.font_style != root_font_style else None
                    run_direction_override = seg_style.direction if seg_style.direction != root_direction else None
                    run_xml_space_override = seg_style.xml_space if seg_style.xml_space != root_xml_space else None
                    seg_ta = TextAnchor(seg_style.text_anchor)
                    run_ta_override = seg_ta if seg_ta != root_text_anchor else None

                    has_font_override = (run_fam_override is not None or run_weight_override is not None or run_style_override is not None)
                    run_fonts_to_store = run_fonts if has_font_override else (run_fonts if run_fonts != root_fonts else None)

                    runs.append(TextRun(
                        text=seg.text,
                        fonts=run_fonts_to_store,
                        font_size=run_font_size,
                        font_family_name=run_fam_override,
                        font_weight=run_weight_override,
                        font_style=run_style_override,
                        fill=seg_fill,
                        fill_none=seg_fill_none,
                        fill_opacity=seg_fill_op,
                        x=run_x,
                        y=run_y,
                        dx=run_dx,
                        dy=run_dy,
                        text_anchor=run_ta_override,
                        direction=run_direction_override,
                        xml_space=run_xml_space_override,
                        is_font_substituted=run_sub,
                    ))

                if not runs:
                    return None

                has_complex_structure = (len(runs) > 1 or any(r.x is not None or r.y is not None or r.dx != 0.0 or r.dy != 0.0 for r in runs)
                                         or any(seg.element is not elem for seg in norm_segments if seg.text))
                if has_complex_structure:
                    drawable = Text(
                        runs=tuple(runs),
                        position=Point(text_x, text_y),
                        text_origin="baseline",
                        text_anchor=root_text_anchor,
                        font_size=style.font_size,
                        font_family_name=root_fam_name,
                        font_weight=style.font_weight,
                        font_style=style.font_style,
                        fonts=root_fonts,
                        color=root_color or Color.black(),
                        fill_opacity=style.fill_opacity,
                        fill_none=root_fill_none,
                        direction=style.direction,
                        xml_space=root_xml_space,
                        is_font_substituted=root_is_sub,
                    )
                else:
                    drawable = Text(
                        text=runs[0].text,
                        position=Point(text_x, text_y),
                        text_origin="baseline",
                        text_anchor=root_text_anchor,
                        font_size=style.font_size,
                        font_family_name=root_fam_name,
                        font_weight=style.font_weight,
                        font_style=style.font_style,
                        fonts=root_fonts,
                        color=root_color or Color.black(),
                        fill_opacity=style.fill_opacity,
                        fill_none=root_fill_none,
                        direction=style.direction,
                        xml_space=root_xml_space,
                        is_font_substituted=root_is_sub,
                    )
                drawable.transform = Transform.from_matrix(tf_mat)

            else:
                raise SVGImportError(
                    f"Unsupported SVG element: <{tag}>",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_ELEMENT", severity="error", message=f"Unsupported element <{tag}>", element_tag=tag),
                )

            if drawable is None:
                return None

            # Apply metadata
            source_id = elem.attrib.get("id")
            if source_id:
                drawable.name = source_id
                drawable.metadata["svg:id"] = source_id
            drawable.metadata["svg:tag"] = tag

            # Apply visibility
            if not isinstance(drawable, Group):
                drawable.visible = (style.visibility not in ("hidden", "collapse"))
            else:
                drawable.visible = True

            # Non-group opacity & blend mode
            if not isinstance(drawable, Group):
                drawable.opacity = style.opacity
                drawable.blend_mode = style.mix_blend_mode

            # Apply Fill
            if hasattr(drawable, "fill") and style.fill and style.fill.lower() != "none":
                if style.fill.startswith("url("):
                    m_url = RE_URL_REF.match(style.fill)
                    if not m_url:
                        raise SVGImportError(
                            f"Unsupported or malformed fill url reference: '{style.fill}'",
                            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_URL_REFERENCE", severity="error", message=f"Malformed fill url '{style.fill}'", attribute="fill"),
                        )
                    ref_id = m_url.group(1)
                    paint_res = bind_paint_server(ref_id, drawable, defs_registry, current_viewport, element_ctm)
                    if paint_res is not None:
                        if isinstance(paint_res, Color):
                            drawable.fill = FillStyle(color=paint_res, opacity=style.fill_opacity)
                        else:
                            drawable.fill = FillStyle(paint=paint_res, opacity=style.fill_opacity)
                    else:
                        drawable.fill = None
                else:
                    col = SVGColorParser.parse(style.fill, current_color=style.color)
                    if col:
                        drawable.fill = FillStyle(color=col, opacity=style.fill_opacity)
                    else:
                        drawable.fill = None

            # Apply Stroke
            eff_sw = style.stroke_width
            if style.raw_stroke_width is not None and style.raw_stroke_width.unit == "%":
                eff_sw = current_viewport.resolve_diagonal(style.raw_stroke_width, context_name="stroke-width", limits=self.limits)

            eff_dashes = style.stroke_dasharray
            if style.raw_stroke_dasharray is not None and any(l.unit == "%" for l in style.raw_stroke_dasharray):
                dashes = []
                for parsed_l in style.raw_stroke_dasharray:
                    val = current_viewport.resolve_diagonal(parsed_l, context_name="stroke-dasharray", limits=self.limits)
                    if val < 0.0:
                        raise SVGImportError("Negative stroke-dasharray value")
                    dashes.append(val)
                if len(dashes) % 2 == 1:
                    dashes = dashes * 2
                eff_dashes = tuple(dashes)

            eff_offset = style.stroke_dashoffset
            if style.raw_stroke_dashoffset is not None and style.raw_stroke_dashoffset.unit == "%":
                eff_offset = current_viewport.resolve_diagonal(style.raw_stroke_dashoffset, context_name="stroke-dashoffset", limits=self.limits)

            if hasattr(drawable, "stroke") and style.stroke and style.stroke.lower() != "none" and eff_sw > 0:
                # Ordinary SVG strokes are measured before the element CTM.
                scaled_w, scaled_dashes, scaled_offset = eff_sw, eff_dashes, eff_offset
                stroke_space = "screen" if style.vector_effect == "non-scaling-stroke" else "object"

                if style.stroke.startswith("url("):
                    m_url = RE_URL_REF.match(style.stroke)
                    if not m_url:
                        raise SVGImportError(
                            f"Unsupported or malformed stroke url reference: '{style.stroke}'",
                            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_URL_REFERENCE", severity="error", message=f"Malformed stroke url '{style.stroke}'", attribute="stroke"),
                        )
                    ref_id = m_url.group(1)
                    paint_res = bind_paint_server(ref_id, drawable, defs_registry, current_viewport, element_ctm)
                    if paint_res is not None:
                        st_kw = {
                            "width": scaled_w,
                            "space": stroke_space,
                            "opacity": style.stroke_opacity,
                            "cap_style": style.stroke_linecap,
                            "join_style": style.stroke_linejoin,
                            "dash_array": scaled_dashes,
                            "dash_offset": scaled_offset,
                            "miter_limit": style.stroke_miterlimit,
                        }
                        if isinstance(paint_res, Color):
                            drawable.stroke = StrokeStyle(color=paint_res, **st_kw)
                        else:
                            drawable.stroke = StrokeStyle(paint=paint_res, **st_kw)
                    else:
                        drawable.stroke = None
                else:
                    col = SVGColorParser.parse(style.stroke, current_color=style.color)
                    if col:
                        drawable.stroke = StrokeStyle(
                            color=col,
                            width=scaled_w,
                            space=stroke_space,
                            opacity=style.stroke_opacity,
                            cap_style=style.stroke_linecap,
                            join_style=style.stroke_linejoin,
                            dash_array=scaled_dashes,
                            dash_offset=scaled_offset,
                            miter_limit=style.stroke_miterlimit,
                        )
                    else:
                        drawable.stroke = None

            # Apply ClipPath
            if style.clip_path:
                apply_clip_path(drawable, style, element_ctm, current_viewport)

            return on_materialized(drawable)

        # Build all top-level children
        root_style = SVGStyleResolver.resolve(root, limits=self.limits)
        if root_style.display == "none":
            return SVGImportResult(scene=scene)
        root_transform = SVGTransformParser.parse_to_matrix(root.attrib.get("transform", ""), limits=self.limits)
        root_ctm = root_transform @ M_viewbox
        root_clip = ClipRect(0.0, 0.0, float(scene_w), float(scene_h)) if root_style.overflow in ("hidden", "scroll") else None

        for child in root:
            d = build_element(child, root_style, root_ctm, root_viewport)
            if d is not None:
                root_drawables.append(d)

        # Authored transform surrounds the viewport; viewBox maps its contents.
        # Keep root compositing outside both, so overlapping children receive
        # root opacity once. Explicit clip-path uses the contents' user space.
        if not np.allclose(M_viewbox, np.eye(3)) or root_style.clip_path:
            viewbox_group = Group(children=root_drawables, transform=Transform.from_matrix(M_viewbox))
            viewbox_group.metadata["svg:root_viewbox"] = True
            if root_style.clip_path:
                apply_clip_path(viewbox_group, root_style, root_ctm, root_viewport)
            root_drawables = [viewbox_group]
        if root_clip is not None:
            root_viewport_group = Group(children=root_drawables, clip=root_clip)
            root_viewport_group.metadata["svg:root_viewport"] = True
            root_drawables = [root_viewport_group]
        if (not np.allclose(root_transform, np.eye(3)) or root_style.opacity != 1.0
                or root_style.mix_blend_mode != BlendMode.NORMAL):
            root_host = Group(children=root_drawables, transform=Transform.from_matrix(root_transform),
                              opacity=root_style.opacity, blend_mode=root_style.mix_blend_mode)
            root_host.metadata["svg:root"] = True
            root_drawables = [root_host]
        for d in root_drawables:
            scene.add(d)

        return SVGImportResult(scene=scene)
