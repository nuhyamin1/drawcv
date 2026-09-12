"""DrawCV — A highly manipulable 2D drawing library built on OpenCV and NumPy."""

from drawcv.canvas import Canvas
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import (
    ArcClosure,
    ArrowHeadStyle,
    BlurType,
    CapStyle,
    FillRule,
    FontFamily,
    ImageInterpolation,
    JoinStyle,
    LineType,
    MaskMapping,
    TextAlignment,
)
from drawcv.core.exceptions import (
    DrawCVError,
    ObjectNotFoundError,
    RenderError,
    ValidationError,
)
from drawcv.core.geometry import Point, StrokePoint
from drawcv.core.path_processing import (
    catmull_rom_spline,
    chaikin_smooth,
    compute_path_length,
    rdp_simplify,
)
from drawcv.core.transform import Transform
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect
from drawcv.effects.effect import Effect
from drawcv.effects.mask import Mask
from drawcv.effects.shadow import ShadowEffect
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.positioning import (
    align_bottom,
    align_center_x,
    align_center_y,
    align_centers,
    align_left,
    align_right,
    align_top,
    distribute_horizontally,
    distribute_vertically,
    place_above,
    place_below,
    place_left_of,
    place_right_of,
)
from drawcv.renderer import OpenCVRenderer
from drawcv.scene import Scene
from drawcv.selection import Selection
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.image import ImageObject
from drawcv.shapes.line import Line
from drawcv.shapes.path import (
    Close,
    CubicTo,
    LineTo,
    MoveTo,
    Path,
    PathCommand,
    QuadraticTo,
    Subpath,
)
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.shapes.text import Text
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle

__version__ = "0.6.0"

__all__ = [
    "align_bottom",
    "align_center_x",
    "align_center_y",
    "align_centers",
    "align_left",
    "align_right",
    "align_top",
    "Arc",
    "ArcClosure",
    "Arrow",
    "ArrowHeadStyle",
    "BezierCurve",
    "BlurEffect",
    "BlurType",
    "BoundingBox",
    "Canvas",
    "CapStyle",
    "catmull_rom_spline",
    "chaikin_smooth",
    "Circle",
    "ClipPath",
    "ClipRect",
    "Close",
    "Color",
    "compute_path_length",
    "CubicTo",
    "distribute_horizontally",
    "distribute_vertically",
    "Drawable",
    "DrawCVError",
    "Effect",
    "Ellipse",
    "FillRule",
    "FillStyle",
    "FontFamily",
    "FreehandStroke",
    "Group",
    "ImageInterpolation",
    "ImageObject",
    "JoinStyle",
    "Layer",
    "Line",
    "LineTo",
    "LineType",
    "Mask",
    "MaskMapping",
    "MoveTo",
    "ObjectNotFoundError",
    "OpenCVRenderer",
    "Path",
    "PathCommand",
    "place_above",
    "place_below",
    "place_left_of",
    "place_right_of",
    "Point",
    "Polygon",
    "Polyline",
    "QuadraticTo",
    "rdp_simplify",
    "Rectangle",
    "RenderError",
    "RoundedRectangle",
    "Scene",
    "Selection",
    "ShadowEffect",
    "StrokePoint",
    "StrokeStyle",
    "Subpath",
    "Text",
    "TextAlignment",
    "Transform",
    "ValidationError",
]
