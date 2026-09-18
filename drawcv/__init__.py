"""DrawCV — A highly manipulable 2D drawing library built on OpenCV and NumPy."""

from drawcv.typography import (
    DictFontResolver,
    FontAsset,
    FontResolver,
    ResolvedFontDescriptor,
    TextLineMetrics,
    TextMetrics,
)
from drawcv.canvas import Canvas
from drawcv.styles.paint import (
    GradientStop,
    LinearGradient,
    RadialGradient,
    ConicGradient,
    ImagePaint,
    Paint,
    PaintLike,
    paint_from_dict,
)
from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import (
    ArcClosure,
    ArrowHeadStyle,
    BlendMode,
    BlurType,
    CapStyle,
    DisplacementChannel,
    EdgeDetectionMethod,
    FillRule,
    FontFamily,
    ImageInterpolation,
    JoinStyle,
    LineType,
    MaskMapping,
    PathBooleanOp,
    TextAlignment,
)
from drawcv.core.exceptions import (
    DrawCVError,
    InvalidFormatError,
    ObjectNotFoundError,
    PathBooleanError,
    RenderError,
    SerializationError,
    UnknownDrawableTypeError,
    UnsupportedVersionError,
    ValidationError,
)
from drawcv.core.geometry import Point, StrokePoint
from drawcv.history import (
    AddObjectCommand,
    Command,
    CompoundCommand,
    GroupCommand,
    HistoryManager,
    RemoveObjectCommand,
    ReorderCommand,
    StateEditCommand,
    TransformCommand,
    UngroupCommand,
)
from drawcv.serialization import (
    CURRENT_FORMAT_IDENTIFIER,
    CURRENT_SCHEMA_VERSION,
    SchemaMigrator,
    from_json,
    register_drawable_type,
    to_json,
)
from drawcv.core.path_processing import (
    catmull_rom_spline,
    chaikin_smooth,
    compute_path_length,
    rdp_simplify,
)
from drawcv.core.transform import Transform
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect
from drawcv.effects.color import (
    BrightnessContrastEffect,
    ColorMatrixEffect,
    GrayscaleEffect,
    HueShiftEffect,
    SaturationEffect,
    SepiaEffect,
)
from drawcv.effects.convolution import ConvolutionEffect
from drawcv.effects.displacement import DisplacementMapEffect
from drawcv.effects.edge import EdgeDetectionEffect
from drawcv.effects.effect import Effect
from drawcv.effects.emboss import EmbossEffect
from drawcv.effects.glow import GlowEffect
from drawcv.effects.mask import Mask
from drawcv.effects.noise import NoiseEffect
from drawcv.effects.shadow import ShadowEffect
from drawcv.effects.sharpen import SharpenEffect

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
    EllipticalArcTo,
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
from drawcv.shapes.text import Text, TextAnchor, TextRun
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle
from drawcv.animation import (
    AnimationTrack,
    EASING_FUNCTIONS,
    Timeline,
    Timing,
    VideoRenderer,
    ease_in,
    ease_in_out,
    ease_out,
    get_easing,
    lerp,
    lerp_bounds,
    lerp_color,
    lerp_point,
    lerp_transform,
    linear,
)
from drawcv.core.path_processing import (
    slice_bezier,
    slice_path,
    slice_polyline,
    slice_stroke_points,
)

__version__ = "0.9.0"

from drawcv.svg import SVGExporter, SVGExport, SVGFallback
from drawcv.svg_import import (
    SVGImporter,
    SVGImportResult,
    SVGImportDiagnostic,
    SVGImportError,
    SVGImportLimits,
)

__all__ = [
    "SVGExporter",
    "SVGExport",
    "SVGFallback",
    "SVGImporter",
    "SVGImportResult",
    "SVGImportDiagnostic",
    "SVGImportError",
    "SVGImportLimits",
    "FontAsset", "TextMetrics", "TextLineMetrics",
    "FontResolver", "DictFontResolver", "ResolvedFontDescriptor",
    "GradientStop", "LinearGradient", "RadialGradient",
    "AddObjectCommand",
    "align_bottom",
    "align_center_x",
    "align_center_y",
    "align_centers",
    "align_left",
    "align_right",
    "align_top",
    "AnimationTrack",
    "Arc",
    "ArcClosure",
    "Arrow",
    "ArrowHeadStyle",
    "BezierCurve",
    "BlendMode",
    "BlurEffect",
    "BlurType",
    "BoundingBox",
    "BrightnessContrastEffect",
    "Canvas",
    "CapStyle",
    "catmull_rom_spline",
    "chaikin_smooth",
    "Circle",
    "ClipPath",
    "ClipRect",
    "Close",
    "Color",
    "ColorMatrixEffect",
    "Command",
    "CompoundCommand",
    "compute_path_length",
    "ConicGradient",
    "ConvolutionEffect",
    "CubicTo",
    "CURRENT_FORMAT_IDENTIFIER",
    "CURRENT_SCHEMA_VERSION",
    "DisplacementChannel",
    "DisplacementMapEffect",
    "distribute_horizontally",
    "distribute_vertically",
    "Drawable",
    "DrawCVError",
    "ease_in",
    "ease_in_out",
    "ease_out",
    "EASING_FUNCTIONS",
    "EdgeDetectionEffect",
    "EdgeDetectionMethod",
    "Effect",
    "Ellipse",
    "EllipticalArcTo",
    "EmbossEffect",
    "FillRule",
    "FillStyle",
    "FontFamily",
    "FreehandStroke",
    "from_json",
    "get_easing",
    "GlowEffect",
    "GrayscaleEffect",
    "Group",
    "GroupCommand",
    "HistoryManager",
    "HueShiftEffect",
    "ImageInterpolation",
    "ImageObject",
    "ImagePaint",
    "InvalidFormatError",
    "JoinStyle",
    "Layer",
    "lerp",
    "lerp_bounds",
    "lerp_color",
    "lerp_point",
    "lerp_transform",
    "Line",
    "linear",
    "LineTo",
    "LineType",
    "Mask",
    "MaskMapping",
    "MoveTo",
    "NoiseEffect",
    "ObjectNotFoundError",
    "OpenCVRenderer",
    "Paint",
    "PaintLike",
    "paint_from_dict",
    "Path",
    "PathBooleanError",
    "PathBooleanOp",
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
    "register_drawable_type",
    "RemoveObjectCommand",
    "RenderError",
    "ReorderCommand",
    "RoundedRectangle",
    "Scene",
    "SchemaMigrator",
    "Selection",
    "SerializationError",
    "SaturationEffect",
    "SepiaEffect",
    "ShadowEffect",
    "SharpenEffect",
    "slice_bezier",
    "slice_path",
    "slice_polyline",
    "slice_stroke_points",
    "StateEditCommand",
    "StrokePoint",
    "StrokeStyle",
    "Subpath",
    "Text",
    "TextAlignment",
    "TextAnchor",
    "TextRun",
    "Timeline",
    "Timing",
    "to_json",
    "Transform",
    "TransformCommand",
    "UngroupCommand",
    "UnknownDrawableTypeError",
    "UnsupportedVersionError",
    "ValidationError",
    "VideoRenderer",
]
