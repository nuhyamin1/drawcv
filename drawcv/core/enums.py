"""Enumerations for DrawCV."""

from enum import Enum


class LineType(Enum):
    """Line anti-aliasing / connectivity types."""
    AA = "AA"
    LINE_8 = "LINE_8"
    LINE_4 = "LINE_4"


class CapStyle(Enum):
    """Cap styles for stroke terminals.
    
    Note: Full vector cap rendering support is deferred to future phases.
    """
    BUTT = "butt"
    ROUND = "round"
    SQUARE = "square"


class JoinStyle(Enum):
    """Join styles for connected stroke segments.
    
    Note: Full vector join rendering support is deferred to future phases.
    """
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
