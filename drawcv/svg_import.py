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
from drawcv.shapes.path import Close, CubicTo, LineTo, MoveTo, Path, QuadraticTo, Subpath
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
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
# Color Parser
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

RE_PATH_TOKEN = re.compile(r"([MmLlQqCcZzAaHhVvSsTt])|([+-]?(?:[0-9]*\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?)")

class SVGPathParser:
    """Parses SVG path 'd' string into a retained DrawCV Path."""

    @classmethod
    def parse(cls, d_str: str, fill_rule: FillRule = FillRule.NON_ZERO, limits: SVGImportLimits | None = None) -> Path:
        max_tokens = limits.max_path_tokens if limits else 100_000
        tokens: list[str] = []
        pos = 0
        cleaned = d_str.strip()

        for m in RE_PATH_TOKEN.finditer(cleaned):
            skipped = cleaned[pos:m.start()].strip(" \t\r\n,")
            if skipped:
                raise SVGImportError(
                    f"Unexpected characters in path data: '{skipped}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Illegal path characters: '{skipped}'"),
                )
            tokens.append(m.group(0))
            pos = m.end()
            if len(tokens) > max_tokens:
                raise SVGImportError(
                    f"Path token count exceeded limit of {max_tokens}",
                    diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Path token count exceeded limit"),
                )

        tail = cleaned[pos:].strip(" \t\r\n,")
        if tail:
            raise SVGImportError(
                f"Unexpected trailing characters in path data: '{tail}'",
                diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Illegal path tail: '{tail}'"),
            )

        if not tokens:
            return Path(subpaths=[], fill_rule=fill_rule)

        subpaths: list[Subpath] = []
        active_subpath: Subpath | None = None
        current_pt = Point(0.0, 0.0)
        start_pt = Point(0.0, 0.0)

        idx = 0
        n_tokens = len(tokens)
        current_cmd = ""

        while idx < n_tokens:
            token = tokens[idx]
            if token in "MmLlQqCcZzAaHhVvSsTt":
                current_cmd = token
                idx += 1
            elif not current_cmd:
                raise SVGImportError(
                    f"Path data must start with a command, got '{token}'",
                    diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Missing initial command in path: '{token}'"),
                )

            if current_cmd in "AaHhVvSsTt":
                raise SVGImportError(
                    f"SVG path command '{current_cmd}' is not supported in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_PATH_COMMAND", severity="error", message=f"Unsupported path command '{current_cmd}'"),
                )

            if current_cmd in ("Z", "z"):
                if active_subpath is not None and active_subpath.commands:
                    active_subpath.commands.append(Close())
                    active_subpath.closed = True
                    current_pt = start_pt
                active_subpath = None
                current_cmd = ""
                continue

            # Coordinate consumer helper
            def next_float() -> float:
                nonlocal idx
                if idx >= n_tokens:
                    raise SVGImportError(
                        f"Premature end of path data for command '{current_cmd}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Missing coordinates for '{current_cmd}'"),
                    )
                tok = tokens[idx]
                if tok in "MmLlQqCcZzAaHhVvSsTt":
                    raise SVGImportError(
                        f"Expected coordinate but got command '{tok}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Unexpected command '{tok}' where coordinate expected"),
                    )
                idx += 1
                val = float(tok)
                if not math.isfinite(val):
                    raise SVGImportError(
                        f"Non-finite coordinate in path: {val}",
                        diagnostic=SVGImportDiagnostic(code="SVG_NON_FINITE_COORDINATE", severity="error", message=f"Non-finite path coordinate: {val}"),
                    )
                max_coord_mag = limits.max_coordinate_magnitude if limits else 1e7
                if abs(val) > max_coord_mag:
                    raise SVGImportError(
                        f"Path coordinate magnitude {val} exceeds limit of {max_coord_mag}",
                        diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
                    )
                return val

            if current_cmd == "M":
                x, y = next_float(), next_float()
                current_pt = Point(x, y)
                start_pt = current_pt
                active_subpath = Subpath(commands=[MoveTo(current_pt)], closed=False)
                subpaths.append(active_subpath)
                # Subsequent coordinates following M become implicit L commands
                current_cmd = "L"

            elif current_cmd == "m":
                dx, dy = next_float(), next_float()
                current_pt = Point(current_pt.x + dx, current_pt.y + dy)
                start_pt = current_pt
                active_subpath = Subpath(commands=[MoveTo(current_pt)], closed=False)
                subpaths.append(active_subpath)
                # Subsequent coordinates following m become implicit l commands
                current_cmd = "l"

            elif current_cmd == "L":
                x, y = next_float(), next_float()
                current_pt = Point(x, y)
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(LineTo(current_pt))

            elif current_cmd == "l":
                dx, dy = next_float(), next_float()
                current_pt = Point(current_pt.x + dx, current_pt.y + dy)
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(LineTo(current_pt))

            elif current_cmd == "Q":
                cx, cy = next_float(), next_float()
                x, y = next_float(), next_float()
                ctrl = Point(cx, cy)
                end = Point(x, y)
                current_pt = end
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(QuadraticTo(ctrl, end))

            elif current_cmd == "q":
                dcx, dcy = next_float(), next_float()
                dx, dy = next_float(), next_float()
                ctrl = Point(current_pt.x + dcx, current_pt.y + dcy)
                end = Point(current_pt.x + dx, current_pt.y + dy)
                current_pt = end
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(QuadraticTo(ctrl, end))

            elif current_cmd == "C":
                c1x, c1y = next_float(), next_float()
                c2x, c2y = next_float(), next_float()
                x, y = next_float(), next_float()
                ctrl1 = Point(c1x, c1y)
                ctrl2 = Point(c2x, c2y)
                end = Point(x, y)
                current_pt = end
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))

            elif current_cmd == "c":
                dc1x, dc1y = next_float(), next_float()
                dc2x, dc2y = next_float(), next_float()
                dx, dy = next_float(), next_float()
                ctrl1 = Point(current_pt.x + dc1x, current_pt.y + dc1y)
                ctrl2 = Point(current_pt.x + dc2x, current_pt.y + dc2y)
                end = Point(current_pt.x + dx, current_pt.y + dy)
                current_pt = end
                if active_subpath is None:
                    raise SVGImportError(
                        f"SVG path command '{current_cmd}' must follow a MoveTo command",
                        diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_PATH", severity="error", message=f"Path command '{current_cmd}' without prior MoveTo"),
                    )
                active_subpath.commands.append(CubicTo(ctrl1, ctrl2, end))

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
    stroke_linecap: CapStyle = CapStyle.BUTT
    stroke_linejoin: JoinStyle = JoinStyle.MITER
    stroke_miterlimit: float = 4.0
    stroke_dasharray: tuple[float, ...] = ()
    stroke_dashoffset: float = 0.0
    color: Color = Color(0, 0, 0, 1.0)
    visibility: str = "visible"
    clip_rule: FillRule = FillRule.NON_ZERO

    # Non-inherited properties (reset per-element)
    opacity: float = 1.0
    display: str = "inline"
    mix_blend_mode: BlendMode = BlendMode.NORMAL
    clip_path: str | None = None
    vector_effect: str = "none"

    def inherit_child(self) -> ComputedStyle:
        """Create a new child style inheriting all inherited properties while resetting non-inherited ones."""
        return ComputedStyle(
            fill=self.fill,
            fill_opacity=self.fill_opacity,
            fill_rule=self.fill_rule,
            stroke=self.stroke,
            stroke_opacity=self.stroke_opacity,
            stroke_width=self.stroke_width,
            stroke_linecap=self.stroke_linecap,
            stroke_linejoin=self.stroke_linejoin,
            stroke_miterlimit=self.stroke_miterlimit,
            stroke_dasharray=self.stroke_dasharray,
            stroke_dashoffset=self.stroke_dashoffset,
            color=self.color,
            visibility=self.visibility,
            clip_rule=self.clip_rule,
            opacity=1.0,
            display="inline",
            mix_blend_mode=BlendMode.NORMAL,
            clip_path=None,
            vector_effect="none",
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
            else:
                num_tokens = [t for t in re.split(r"[\s,]+", da_clean) if t]
                dashes = []
                for tok in num_tokens:
                    parsed_l = SVGLength.parse(tok, limits=limits)
                    val = parsed_l.to_absolute(context_name="stroke-dasharray", limits=limits)
                    if val < 0.0:
                        raise SVGImportError(
                            f"Negative value in stroke-dasharray: '{tok}'",
                            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_STYLE", severity="error", message=f"Negative stroke-dasharray value: '{tok}'"),
                        )
                    dashes.append(val)
                if len(dashes) % 2 == 1:
                    dashes = dashes * 2  # SVG rule: odd dasharray repeated
                style.stroke_dasharray = tuple(dashes)

        # 12. Stroke-dashoffset
        do_val = get_prop("stroke-dashoffset")
        if do_val is not None:
            parsed_do = SVGLength.parse(do_val.strip(), limits=limits)
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

        return style


# -----------------------------------------------------------------------------
# Stroke Similarity Normalization
# -----------------------------------------------------------------------------

def evaluate_stroke_compatibility(
    stroke_width: float,
    dash_array: tuple[float, ...],
    dash_offset: float,
    ctm: np.ndarray,
    vector_effect: str,
    element_tag: str = "",
) -> tuple[float, tuple[float, ...], float]:
    """Evaluates stroke transform compatibility.

    Returns: (scaled_width, scaled_dash_array, scaled_dash_offset).
    Raises: SVGImportError if ordinary stroke is under non-uniform scaling or shear.
    """
    if vector_effect == "non-scaling-stroke":
        return stroke_width, dash_array, dash_offset

    A = ctm[:2, :2]
    G = A.T @ A
    g00, g01, g11 = G[0, 0], G[0, 1], G[1, 1]

    s2 = (g00 + g11) / 2.0
    if s2 < 1e-12:
        return 0.0, (), 0.0

    err_scale = abs(g00 - g11) / s2
    err_shear = 2.0 * abs(g01) / s2

    if err_scale < 1e-5 and err_shear < 1e-5:
        # Uniform similarity transform
        s = math.sqrt(s2)
        scaled_w = stroke_width * s
        scaled_dashes = tuple(d * s for d in dash_array)
        scaled_offset = dash_offset * s
        return scaled_w, scaled_dashes, scaled_offset

    raise SVGImportError(
        f"Element '<{element_tag}>' has ordinary visible stroke under anisotropic scale or shear. "
        "Specify vector-effect='non-scaling-stroke' or apply uniform scaling.",
        diagnostic=SVGImportDiagnostic(
            code="SVG_STROKE_AFFINE_MISMATCH", severity="error",
            message="Ordinary stroke under anisotropic scaling or shear is unsupported",
            element_tag=element_tag,
        ),
    )


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
    current_viewport: tuple[float, float],
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
# Main SVG Importer Architecture
# -----------------------------------------------------------------------------

class SVGImporter:
    """Retained-mode SVG document importer with strict retained DrawCV mapping."""

    def __init__(
        self,
        *,
        strict: bool = True,
        viewport: tuple[int, int] | None = None,
        limits: SVGImportLimits | None = None,
    ):
        if not strict:
            raise NotImplementedError("Permissive SVG import (strict=False) is not yet implemented in Milestone 1")
        self.strict = strict
        self.viewport = viewport
        self.limits = limits if limits is not None else SVGImportLimits()

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

        # Calculate viewBox transformation
        M_viewbox = np.eye(3, dtype=np.float64)
        effective_viewport = (float(scene_w), float(scene_h))

        if vb_attr:
            vb_nums = [float(t) for t in re.split(r"[\s,]+", vb_attr.strip()) if t]
            if len(vb_nums) != 4:
                raise SVGImportError("viewBox requires exactly 4 numbers: min_x min_y width height")
            min_x, min_y, vb_w, vb_h = vb_nums
            if vb_w <= 0 or vb_h <= 0 or not math.isfinite(vb_w) or not math.isfinite(vb_h):
                raise SVGImportError(f"Invalid viewBox dimensions: ({vb_w}, {vb_h})")

            for coord in (min_x, min_y, vb_w, vb_h):
                if abs(coord) > self.limits.max_coordinate_magnitude:
                    raise SVGImportError(
                        f"viewBox coordinate magnitude {coord} exceeds limit of {self.limits.max_coordinate_magnitude}",
                        diagnostic=SVGImportDiagnostic(code="SVG_RESOURCE_LIMIT_EXCEEDED", severity="error", message="Coordinate magnitude exceeded limit"),
                    )

            par = root.attrib.get("preserveAspectRatio", "xMidYMid meet").strip()
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
                sx = scene_w / vb_w
                sy = scene_h / vb_h
                tx = -min_x * sx
                ty = -min_y * sy
            else:
                scale_x = scene_w / vb_w
                scale_y = scene_h / vb_h
                s = min(scale_x, scale_y) if meet_or_slice == "meet" else max(scale_x, scale_y)
                sx, sy = s, s

                if "xMin" in align:
                    tx = -min_x * s
                elif "xMax" in align:
                    tx = (scene_w - vb_w * s) - min_x * s
                else:  # xMid
                    tx = (scene_w - vb_w * s) / 2.0 - min_x * s

                if "yMin" in align:
                    ty = -min_y * s
                elif "yMax" in align:
                    ty = (scene_h - vb_h * s) - min_y * s
                else:  # yMid
                    ty = (scene_h - vb_h * s) / 2.0 - min_y * s

            M_viewbox = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
            effective_viewport = (vb_w, vb_h)

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
            if tag_lower in ("text", "tspan", "textpath"):
                raise SVGImportError(
                    "SVG typography <text> is deferred in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_TEXT", severity="error", message="SVG <text> deferred", element_tag=tag),
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
            if tag_lower in ("use", "symbol"):
                raise SVGImportError(
                    "SVG <use>/<symbol> is deferred in Milestone 1",
                    diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_USE", severity="error", message="SVG <use>/<symbol> deferred", element_tag=tag),
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
                if child_tag not in ("path", "rect", "polygon", "polyline"):
                    raise SVGImportError(
                        f"Unsupported clipPath child geometry <{child_tag}> in Milestone 1 (circle, ellipse, line, and grouped clips are unsupported)",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message=f"Unsupported clip geometry <{child_tag}>"),
                    )

                # If rect, reject rounded rect
                if child_tag == "rect":
                    rx_attr = child_elem.attrib.get("rx")
                    ry_attr = child_elem.attrib.get("ry")
                    if rx_attr or ry_attr:
                        rx_val = SVGLength.parse(rx_attr, limits=self.limits).to_absolute(context_name="rx", limits=self.limits) if rx_attr else 0.0
                        ry_val = SVGLength.parse(ry_attr, limits=self.limits).to_absolute(context_name="ry", limits=self.limits) if ry_attr else 0.0
                        if rx_val > 0.0 or ry_val > 0.0:
                            raise SVGImportError(
                                "Rounded rectangle in <clipPath> is unsupported in Milestone 1 (requires curve approximation)",
                                diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message="Rounded rect clip unsupported"),
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

        def build_element(
            elem: ET.Element,
            parent_style: ComputedStyle,
            cumulative_ctm: np.ndarray,
        ) -> Drawable | None:
            tag = extract_element_tag(elem.tag)
            tag_lower = tag.lower()

            # Skip non-visual elements in visible tree
            if tag_lower in ("defs", "lineargradient", "radialgradient", "clippath", "title", "desc", "metadata"):
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
                    ch_d = build_element(child, style, element_ctm)
                    if ch_d is not None:
                        children_drawables.append(ch_d)
                drawable = Group(
                    children=children_drawables,
                    transform=Transform.from_matrix(tf_mat),
                    opacity=style.opacity,
                    blend_mode=style.mix_blend_mode,
                    visible=True,  # Keeps group open so visibility:visible children can render
                )

            elif tag_lower == "path":
                d_str = elem.attrib.get("d", "")
                drawable = SVGPathParser.parse(d_str, fill_rule=style.fill_rule, limits=self.limits)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "rect":
                x = SVGLength.parse(elem.attrib.get("x", "0"), limits=self.limits).to_absolute(context_name="rect x", limits=self.limits)
                y = SVGLength.parse(elem.attrib.get("y", "0"), limits=self.limits).to_absolute(context_name="rect y", limits=self.limits)
                w = SVGLength.parse(elem.attrib.get("width", "0"), limits=self.limits).to_absolute(context_name="rect width", limits=self.limits)
                h = SVGLength.parse(elem.attrib.get("height", "0"), limits=self.limits).to_absolute(context_name="rect height", limits=self.limits)

                if w < 0 or h < 0:
                    raise SVGImportError("Rectangle dimensions must be non-negative")
                if w == 0 or h == 0:
                    return None  # SVG spec: zero-dimension rect is omitted

                rx_attr = elem.attrib.get("rx")
                ry_attr = elem.attrib.get("ry")

                if rx_attr is None and ry_attr is None:
                    drawable = Rectangle(position=Point(x, y), width=w, height=h)
                else:
                    rx_val = SVGLength.parse(rx_attr, limits=self.limits).to_absolute(context_name="rx", limits=self.limits) if rx_attr else None
                    ry_val = SVGLength.parse(ry_attr, limits=self.limits).to_absolute(context_name="ry", limits=self.limits) if ry_attr else None

                    if rx_val is not None and rx_val < 0:
                        raise SVGImportError("Negative rx is invalid")
                    if ry_val is not None and ry_val < 0:
                        raise SVGImportError("Negative ry is invalid")

                    eff_rx = rx_val if rx_val is not None else ry_val
                    eff_ry = ry_val if ry_val is not None else rx_val

                    # Clamp per W3C SVG used-value rules
                    eff_rx = min(eff_rx, w / 2.0)
                    eff_ry = min(eff_ry, h / 2.0)

                    if eff_rx <= 0 or eff_ry <= 0:
                        drawable = Rectangle(position=Point(x, y), width=w, height=h)
                    elif math.isclose(eff_rx, eff_ry, rel_tol=1e-7, abs_tol=1e-7):
                        drawable = RoundedRectangle(x=x, y=y, width=w, height=h, corner_radius=eff_rx)
                    else:
                        raise SVGImportError(
                            f"Elliptical rounded rectangle (rx={eff_rx}, ry={eff_ry}) is not supported in Milestone 1",
                            diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_ELLIPTICAL_ROUNDED_RECT", severity="error", message="Elliptical rounded rect unsupported"),
                        )
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "circle":
                cx = SVGLength.parse(elem.attrib.get("cx", "0"), limits=self.limits).to_absolute(context_name="circle cx", limits=self.limits)
                cy = SVGLength.parse(elem.attrib.get("cy", "0"), limits=self.limits).to_absolute(context_name="circle cy", limits=self.limits)
                r = SVGLength.parse(elem.attrib.get("r", "0"), limits=self.limits).to_absolute(context_name="circle r", limits=self.limits)
                if r < 0:
                    raise SVGImportError("Circle radius must be non-negative")
                if r == 0:
                    return None
                drawable = Circle(center=Point(cx, cy), radius=r)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "ellipse":
                cx = SVGLength.parse(elem.attrib.get("cx", "0"), limits=self.limits).to_absolute(context_name="ellipse cx", limits=self.limits)
                cy = SVGLength.parse(elem.attrib.get("cy", "0"), limits=self.limits).to_absolute(context_name="ellipse cy", limits=self.limits)
                rx = SVGLength.parse(elem.attrib.get("rx", "0"), limits=self.limits).to_absolute(context_name="ellipse rx", limits=self.limits)
                ry = SVGLength.parse(elem.attrib.get("ry", "0"), limits=self.limits).to_absolute(context_name="ellipse ry", limits=self.limits)
                if rx < 0 or ry < 0:
                    raise SVGImportError("Ellipse radii must be non-negative")
                if rx == 0 or ry == 0:
                    return None
                drawable = Ellipse(center=Point(cx, cy), radius_x=rx, radius_y=ry)
                drawable.transform = Transform.from_matrix(tf_mat)

            elif tag_lower == "line":
                x1 = SVGLength.parse(elem.attrib.get("x1", "0"), limits=self.limits).to_absolute(context_name="line x1", limits=self.limits)
                y1 = SVGLength.parse(elem.attrib.get("y1", "0"), limits=self.limits).to_absolute(context_name="line y1", limits=self.limits)
                x2 = SVGLength.parse(elem.attrib.get("x2", "0"), limits=self.limits).to_absolute(context_name="line x2", limits=self.limits)
                y2 = SVGLength.parse(elem.attrib.get("y2", "0"), limits=self.limits).to_absolute(context_name="line y2", limits=self.limits)
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
            drawable.visible = (style.visibility not in ("hidden", "collapse"))

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
                    paint_res = bind_paint_server(ref_id, drawable, defs_registry, effective_viewport, element_ctm)
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
            if hasattr(drawable, "stroke") and style.stroke and style.stroke.lower() != "none" and style.stroke_width > 0:
                # Check stroke transform compatibility
                scaled_w, scaled_dashes, scaled_offset = evaluate_stroke_compatibility(
                    style.stroke_width,
                    style.stroke_dasharray,
                    style.stroke_dashoffset,
                    element_ctm,
                    style.vector_effect,
                    element_tag=tag,
                )

                if style.stroke.startswith("url("):
                    m_url = RE_URL_REF.match(style.stroke)
                    if not m_url:
                        raise SVGImportError(
                            f"Unsupported or malformed stroke url reference: '{style.stroke}'",
                            diagnostic=SVGImportDiagnostic(code="SVG_MALFORMED_URL_REFERENCE", severity="error", message=f"Malformed stroke url '{style.stroke}'", attribute="stroke"),
                        )
                    ref_id = m_url.group(1)
                    paint_res = bind_paint_server(ref_id, drawable, defs_registry, effective_viewport, element_ctm)
                    if paint_res is not None:
                        st_kw = {
                            "width": scaled_w,
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
                m_clip = RE_URL_REF.match(style.clip_path)
                if not m_clip:
                    raise SVGImportError(
                        f"Unsupported or malformed clip-path syntax: '{style.clip_path}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_PATH", severity="error", message=f"Unsupported clip-path '{style.clip_path}'", attribute="clip-path"),
                    )
                clip_id = m_clip.group(1)
                if clip_id not in defs_registry.clip_templates:
                    raise SVGImportError(
                        f"Unresolved clipPath reference '#{clip_id}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNRESOLVED_REFERENCE", severity="error", message=f"Unresolved clipPath '#{clip_id}'"),
                    )
                cp_tmpl = defs_registry.clip_templates[clip_id]
                if cp_tmpl.element is None:
                    raise SVGImportError(
                        f"ClipPath '#{clip_id}' has no geometry",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message="ClipPath has no geometry"),
                    )

                # Build clip geometry child
                clip_style = SVGStyleResolver.resolve(cp_tmpl.element, cp_tmpl.computed_style, limits=self.limits)
                clip_d = build_element(cp_tmpl.element, cp_tmpl.computed_style, np.eye(3, dtype=np.float64))
                if clip_d is None:
                    raise SVGImportError(
                        f"ClipPath '#{clip_id}' geometry produced no drawable",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message="ClipPath geometry is empty"),
                    )

                # Compose transforms: M_child is clip_d's own local transform
                M_child = clip_d.transform.get_matrix(Point(0, 0))
                M_clip_base = cp_tmpl.transform @ M_child

                if cp_tmpl.clipPathUnits == "objectBoundingBox":
                    bbox = drawable.get_geometry_bounds()
                    M_obb = np.array([
                        [bbox.width, 0.0, bbox.x],
                        [0.0, bbox.height, bbox.y],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    M_clip_total = M_obb @ M_clip_base
                else:
                    # userSpaceOnUse: M_clip = M_clipPath @ M_child
                    M_clip_total = M_clip_base

                if isinstance(clip_d, Path):
                    clip_d.transform = Transform.from_matrix(M_clip_total)
                    clip_d.fill_rule = clip_style.clip_rule
                    drawable.clip = clip_d
                elif isinstance(clip_d, Rectangle) and np.allclose(M_clip_total, np.eye(3)):
                    drawable.clip = ClipRect(clip_d.position.x, clip_d.position.y, clip_d.width, clip_d.height)
                elif isinstance(clip_d, Rectangle):
                    p_clip = Path(fill_rule=clip_style.clip_rule)
                    p_clip.move_to(Point(clip_d.position.x, clip_d.position.y))
                    p_clip.line_to(Point(clip_d.position.x + clip_d.width, clip_d.position.y))
                    p_clip.line_to(Point(clip_d.position.x + clip_d.width, clip_d.position.y + clip_d.height))
                    p_clip.line_to(Point(clip_d.position.x, clip_d.position.y + clip_d.height))
                    p_clip.close()
                    p_clip.transform = Transform.from_matrix(M_clip_total)
                    drawable.clip = p_clip
                elif isinstance(clip_d, Polyline):
                    p_clip = Path(fill_rule=clip_style.clip_rule)
                    if clip_d.points:
                        p_clip.move_to(clip_d.points[0])
                        for pt in clip_d.points[1:]:
                            p_clip.line_to(pt)
                        p_clip.close()
                    p_clip.transform = Transform.from_matrix(M_clip_total)
                    drawable.clip = p_clip
                else:
                    raise SVGImportError(
                        f"Unsupported clip geometry '{type(clip_d).__name__}'",
                        diagnostic=SVGImportDiagnostic(code="SVG_UNSUPPORTED_CLIP_GEOMETRY", severity="error", message=f"Unsupported clip geometry '{type(clip_d).__name__}'"),
                    )

            return drawable

        # Build all top-level children
        root_style = SVGStyleResolver.resolve(root, limits=self.limits)
        for child in root:
            d = build_element(child, root_style, M_viewbox)
            if d is not None:
                root_drawables.append(d)

        # Attach to Scene default layer
        if not np.allclose(M_viewbox, np.eye(3)):
            # Wrap in synthetic root Group to preserve viewBox mapping
            root_group = Group(children=root_drawables, transform=Transform.from_matrix(M_viewbox))
            root_group.metadata["svg:root_viewbox"] = True
            scene.add(root_group)
        else:
            for d in root_drawables:
                scene.add(d)

        return SVGImportResult(scene=scene)
