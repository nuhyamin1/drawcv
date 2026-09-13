"""Enumerations for DrawCV."""

from enum import Enum


class LineType(Enum):
    """Line anti-aliasing / connectivity types."""
    AA = "AA"
    LINE_8 = "LINE_8"
    LINE_4 = "LINE_4"


class CapStyle(Enum):
    """Cap styles for stroke terminals."""
    BUTT = "butt"
    ROUND = "round"
    SQUARE = "square"


class JoinStyle(Enum):
    """Join styles for connected stroke segments."""
    MITER = "miter"
    ROUND = "round"
    BEVEL = "bevel"


class ArcClosure(Enum):
    """Closure type for an Arc primitive."""
    OPEN = "open"      # Open curve stroke
    CHORD = "chord"    # Straight line connecting endpoints
    PIE = "pie"        # Two line segments connecting endpoints to center (pie sector)


class FillRule(Enum):
    """Path fill rules for multi-contour topological interiors."""
    EVEN_ODD = "even_odd"  # Parity-based: odd number of crossings = inside
    NON_ZERO = "non_zero"  # Winding-based: net non-zero winding number = inside


class ArrowHeadStyle(Enum):
    """Visual style for arrowhead markers."""
    TRIANGLE = "triangle"  # Solid filled triangle
    OPEN = "open"          # V-stroke open head
    DIAMOND = "diamond"    # Four-vertex diamond marker
    CIRCLE = "circle"      # Circular terminal marker


class FontFamily(Enum):
    """Renderer-independent typography font faces (mapped to Hershey or system fonts)."""
    SIMPLEX = "simplex"
    PLAIN = "plain"
    DUPLEX = "duplex"
    COMPLEX = "complex"
    TRIPLEX = "triplex"
    COMPLEX_SMALL = "complex_small"
    SCRIPT_SIMPLEX = "script_simplex"
    SCRIPT_COMPLEX = "script_complex"


class ImageInterpolation(Enum):
    """Resampling interpolation filters for raster images."""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    AREA = "area"
    LANCZOS = "lanczos"


class BlurType(Enum):
    """Spatial blur kernel algorithms."""
    GAUSSIAN = "gaussian"
    BOX = "box"


class TextAlignment(Enum):
    """Horizontal text baseline alignment relative to anchor position."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class MaskMapping(Enum):
    """Coordinate mapping mode for grayscale masks."""
    FIT_BOUNDS = "fit_bounds"  # Stretches mask bilinearly over entity's pre-effect bounds
    ABSOLUTE = "absolute"      # Aligns mask 1:1 with entity local origin
