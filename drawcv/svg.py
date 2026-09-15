"""Self-contained SVG export with explicit, subtree-scoped raster fallback."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path as FilePath
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from drawcv.core.alpha import unpremultiply
from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, BlendMode, FillRule
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry_utils import flatten_arc
from drawcv.effects.clipping import ClipPath, ClipRect
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.renderer import OpenCVRenderer, _IsolatedSurface
from drawcv.shapes.arc import Arc
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.line import Line
from drawcv.shapes.path import Path, MoveTo, LineTo, QuadraticTo, CubicTo, Close
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.styles.paint import LinearGradient

SVG_NS = "http://www.w3.org/2000/svg"


def _number(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValidationError("SVG coordinates must be finite")
    return format(value, ".15g")


def _color(color):
    return f"rgb({color.r},{color.g},{color.b})"


def _points(points, closed=False):
    points = list(points)
    if len(points) == 1:
        points.append(points[0])  # Explicit zero-length segment retains round/square caps.
    return ("M " + " L ".join(f"{_number(p.x)} {_number(p.y)}" for p in points)
            + (" Z" if closed else "")) if points else ""


def _matrix(matrix):
    return "matrix(" + " ".join(_number(matrix[r, c]) for r, c in
                                ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))) + ")"


@dataclass(frozen=True)
class SVGFallback:
    """One rasterized object or layer, including all of its descendants."""
    entity: str
    reason: str


@dataclass(frozen=True)
class SVGExport:
    """SVG document and immutable accounting of rasterized subtrees."""
    svg: str
    fallbacks: tuple[SVGFallback, ...]

    def save(self, filepath: str | FilePath) -> None:
        FilePath(filepath).write_text(self.svg, encoding="utf-8")


class SVGExporter:
    """Export the current frame; strict mode rejects any required PNG fallback.

    Coordinates are baked into world space, leaving screen-space stroke widths
    and dash lengths unchanged. Curved arc primitives become editable polylines
    using the raster renderer's 0.25 pixel target tolerance and sampling cap;
    Path/Bezier commands retain their curves.
    Instances hold configuration only and can be reused after failed exports.
    """
    def __init__(self, *, strict: bool = False):
        if not isinstance(strict, bool):
            raise ValidationError("strict must be a boolean")
        self.strict = strict

    def render(self, scene) -> SVGExport:
        from drawcv.scene import Scene
        if not isinstance(scene, Scene):
            raise ValidationError("SVG export requires a Scene")
        return _Writer(scene, self.strict).export()


class _Writer:
    def __init__(self, scene, strict):
        self.scene, self.strict = scene, strict
        self.root = ET.Element("svg", {"xmlns": SVG_NS, "version": "1.1",
            "width": str(scene.width), "height": str(scene.height),
            "viewBox": f"0 0 {scene.width} {scene.height}", "color-interpolation": "sRGB"})
        self.defs = ET.SubElement(self.root, "defs")
        self.fallbacks = []
        self.counter = 0

    def identifier(self):
        self.counter += 1
        return f"drawcv-{self.counter}"

    def export(self):
        bg = self.scene.background
        ET.SubElement(self.root, "rect", {"width": str(self.scene.width),
            "height": str(self.scene.height), "fill": _color(bg), "fill-opacity": _number(bg.a)})
        for layer in self.scene.layers:
            self.entity(layer, self.root)
        return SVGExport(ET.tostring(self.root, encoding="unicode"), tuple(self.fallbacks))

    def entity(self, entity, parent):
        if not entity.visible or entity.opacity <= 0:
            return
        progress = getattr(entity, "render_progress", 1.0)
        if progress <= 0:
            return
        source_id = entity.name if isinstance(entity, Layer) else entity.id
        if progress < 1 and getattr(entity, "supports_progressive_rendering", False):
            original = entity
            entity = original.slice_at_progress(progress)
            entity.transform = original.transform.copy()
            if entity.transform.pivot is None:
                object.__setattr__(entity.transform, "pivot", original.get_geometry_bounds().center)
            entity._parent, entity._layer = original._parent, original._layer
            entity.render_progress = 1.0
        reason = self.fallback_reason(entity)
        group = ET.SubElement(parent, "g", {"id": self.identifier(), "data-drawcv-id": source_id})
        blend_mode = getattr(entity, "blend_mode", BlendMode.NORMAL)
        if blend_mode != BlendMode.NORMAL:
            group.set("style", f"mix-blend-mode: {blend_mode.value.replace('_', '-')};")
        if reason:
            if self.strict:
                raise RenderError(f"SVG requires raster fallback for {source_id}: {reason}")
            self.fallbacks.append(SVGFallback(source_id, reason))
            group.set("data-drawcv-raster", reason)
            self.raster(entity, group)
            return
        group.set("opacity", _number(entity.opacity))
        if entity.clip is not None:
            clip = entity.clip
            points = clip.corners if isinstance(clip, ClipRect) else clip.points
            if not isinstance(entity, Layer):
                points = [entity.to_world(p) for p in points]
            ident = self.identifier()
            node = ET.SubElement(self.defs, "clipPath", {"id": ident, "clipPathUnits": "userSpaceOnUse"})
            ET.SubElement(node, "path", {"d": _points(points, True), "clip-rule": "evenodd"})
            group.set("clip-path", f"url(#{ident})")
        if isinstance(entity, (Layer, Group)):
            children = entity.objects if isinstance(entity, Layer) else entity.children
            for _, child in sorted(enumerate(children), key=lambda pair: (pair[1].z_index, pair[0])):
                self.entity(child, group)
        else:
            attrs = {"d": self.geometry(entity), "fill": "none", "stroke": "none"}
            fill = getattr(entity, "fill", None)
            if isinstance(entity, Arc) and entity.closure == ArcClosure.OPEN:
                fill = None
            if fill is not None and fill.enabled:
                attrs["fill"] = self.paint(fill.paint, entity)
                attrs["fill-opacity"] = _number(fill.opacity * (fill.paint.a if isinstance(fill.paint, Color) else 1))
                attrs["fill-rule"] = "nonzero" if getattr(entity, "fill_rule", None) == FillRule.NON_ZERO else "evenodd"
            stroke = getattr(entity, "stroke", None)
            if stroke is not None:
                attrs.update({"stroke": _color(stroke.color), "stroke-opacity": _number(stroke.opacity * stroke.color.a),
                    "stroke-width": _number(stroke.width), "stroke-linecap": stroke.cap_style.value,
                    "stroke-linejoin": stroke.join_style.value, "stroke-miterlimit": _number(stroke.miter_limit)})
                if stroke.dash_array:
                    attrs["stroke-dasharray"] = " ".join(map(_number, stroke.dash_array))
                    attrs["stroke-dashoffset"] = _number(stroke.dash_offset)
            ET.SubElement(group, "path", attrs)

    @staticmethod
    def fallback_reason(entity):
        if entity.effects:
            return "effects"
        if entity.mask is not None:
            return "mask"
        if entity.clip is not None and not isinstance(entity.clip, (ClipRect, ClipPath)):
            return "unsupported clip"
        if isinstance(entity, FreehandStroke) and entity.variable_width:
            return "variable-width stroke"
        supported = (Layer, Group, Line, Rectangle, Circle, Ellipse, Polygon, Polyline,
                     RoundedRectangle, Arc, BezierCurve, Path, FreehandStroke)
        if type(entity) not in supported:
            return f"{type(entity).__name__} rendering"
        return None

    def raster(self, entity, parent):
        class SubtreeRenderer(OpenCVRenderer):
            def _get_render_opacity(self, drawable):
                if drawable is entity and self._current_isolating_ancestor is not entity:
                    return float(entity.opacity)
                return super()._get_render_opacity(drawable)
        surface = _IsolatedSurface(self.scene.width, self.scene.height, alpha_output=True)
        SubtreeRenderer()._render_isolated(entity, surface)
        pixels = unpremultiply(surface.buffer)
        yy, xx = np.nonzero(pixels[..., 3])
        if not len(xx):
            return
        x, y, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
        ok, encoded = cv2.imencode(".png", pixels[y:bottom, x:right])
        if not ok:
            raise RenderError("Could not encode SVG fallback PNG")
        ET.SubElement(parent, "image", {"x": str(x), "y": str(y), "width": str(right-x),
            "height": str(bottom-y), "preserveAspectRatio": "none",
            "{http://www.w3.org/1999/xlink}href": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")})

    def paint(self, paint, entity):
        if isinstance(paint, Color):
            return _color(paint)
        ident = self.identifier()
        attrs = {"id": ident, "gradientUnits": "userSpaceOnUse", "spreadMethod": "pad",
                 "color-interpolation": "sRGB"}
        if paint.space == "object":
            attrs["gradientTransform"] = _matrix(entity.world_matrix)
        if isinstance(paint, LinearGradient):
            attrs.update(x1=_number(paint.start.x), y1=_number(paint.start.y),
                         x2=_number(paint.end.x), y2=_number(paint.end.y))
            node = ET.SubElement(self.defs, "linearGradient", attrs)
        else:
            attrs.update(cx=_number(paint.center.x), cy=_number(paint.center.y), r=_number(paint.radius))
            node = ET.SubElement(self.defs, "radialGradient", attrs)
        for stop in paint.stops:
            ET.SubElement(node, "stop", {"offset": _number(stop.position),
                "stop-color": _color(stop.color), "stop-opacity": _number(stop.color.a)})
        return f"url(#{ident})"

    @staticmethod
    def geometry(obj):
        def coords(p):
            p = obj.to_world(p)
            return f"{_number(p.x)} {_number(p.y)}"
        if isinstance(obj, Path):
            commands = []
            for subpath in obj.subpaths:
                for cmd in subpath.commands:
                    if isinstance(cmd, (MoveTo, LineTo)):
                        commands.append(("M " if isinstance(cmd, MoveTo) else "L ") + coords(cmd.point))
                    elif isinstance(cmd, QuadraticTo):
                        commands.append("Q " + coords(cmd.control) + " " + coords(cmd.end))
                    elif isinstance(cmd, CubicTo):
                        commands.append("C " + coords(cmd.control1) + " " + coords(cmd.control2) + " " + coords(cmd.end))
                    elif isinstance(cmd, Close):
                        commands.append("Z")
                    else:
                        raise RenderError("Unsupported path command")
                if subpath.closed and subpath.commands and not isinstance(subpath.commands[-1], Close):
                    commands.append("Z")
            return " ".join(commands)
        if isinstance(obj, BezierCurve):
            return "M " + coords(obj.p0) + (" C " if obj.p3 is not None else " Q ") + " ".join(
                coords(p) for p in (obj.p1, obj.p2, obj.p3) if p is not None)
        closed = True
        if isinstance(obj, Line):
            points, closed = [obj.start, obj.end], False
        elif isinstance(obj, Rectangle):
            if obj.width <= 0 or obj.height <= 0:
                return ""
            points = obj.corners
        elif isinstance(obj, Polygon):
            points = obj.vertices
        elif isinstance(obj, Polyline):
            points, closed = obj.points, obj.closed
        elif isinstance(obj, FreehandStroke):
            points, closed = obj.get_processed_points(), False
        else:
            tolerance = 0.25 / max(float(np.linalg.norm(obj.world_matrix[:2, :2], ord=2)), 1e-12)
            if isinstance(obj, (Circle, Ellipse)):
                rx, ry = (obj.radius, obj.radius) if isinstance(obj, Circle) else (obj.radius_x, obj.radius_y)
                if rx <= 0 or ry <= 0:
                    return ""
                points = flatten_arc(obj.center, rx, ry, 0, 360, tolerance=tolerance)
            else:
                points = obj.get_contour_points(tolerance=tolerance)
                if isinstance(obj, Arc):
                    closed = obj.closure != ArcClosure.OPEN
        return _points((obj.to_world(p) for p in points), closed)
