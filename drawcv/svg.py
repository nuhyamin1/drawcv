"""Self-contained SVG export with explicit, subtree-scoped raster fallback."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from pathlib import Path as FilePath
import unicodedata
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from drawcv.core.alpha import unpremultiply
from drawcv.core.color import Color
from drawcv.core.enums import ArcClosure, BlendMode, FillRule, ImageInterpolation
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import flatten_arc, transform_elliptical_arc
from drawcv.effects.clipping import ClipPath, ClipRect, get_clip_point_mapper
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.renderer import OpenCVRenderer, _IsolatedSurface
from drawcv.shapes.arc import Arc
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.line import Line
from drawcv.shapes.path import Path, MoveTo, LineTo, QuadraticTo, CubicTo, EllipticalArcTo, Close
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.shapes.text import Text, TextAnchor, TextRun
from drawcv.styles.paint import (
    ConicGradient,
    ImagePaint,
    LinearGradient,
    RadialGradient,
)

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


def _text_native_svg_fallback_reason(text: Text) -> str | None:
    if text.is_font_substituted:
        return "font substituted"
    if text.fonts is None and text.runs is None:
        return "Hershey font text"
    if text.font_family_name is None:
        return "missing font family name"
    if text.text_origin != "baseline":
        return f"text origin {text.text_origin}"
    if text.direction not in ("ltr", "rtl", "auto"):
        return f"text direction {text.direction}"
    if text.direction == "auto":
        full_text = text.text
        if any(unicodedata.bidirectional(c) in ("R", "AL", "RLE", "RLO") for c in full_text):
            return "text direction auto with RTL text"
    if text.background_fill is not None and text.background_fill.enabled and text.background_fill.opacity > 0.0:
        return "background plate"
    if text.padding > 0.0:
        return "text padding"
    if text.wrap_width is not None:
        return "wrapped text"
    if "\n" in text.text:
        return "multiline text"
    if text.runs is not None:
        for r in text.runs:
            if r.is_font_substituted:
                return "font substituted"
            if r.direction == "auto":
                return "run direction auto"
            if r.direction is not None and r.direction not in ("ltr", "rtl"):
                return f"run direction {r.direction}"

            has_semantic_font_override = (
                r.font_family_name is not None
                or r.font_weight is not None
                or r.font_style is not None
            )
            if has_semantic_font_override:
                if r.fonts is None:
                    return "semantic font override without run font asset"

            if r.fonts is not None and r.fonts != text.fonts:
                if not has_semantic_font_override:
                    return "run font asset override without SVG semantic descriptor"
    return None


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
            ident = self.identifier()
            node = ET.SubElement(self.defs, "clipPath", {"id": ident, "clipPathUnits": "userSpaceOnUse"})
            map_point = get_clip_point_mapper(clip, entity)

            if isinstance(clip, Path):
                def coords(p):
                    p_w = map_point(p)
                    return f"{_number(p_w.x)} {_number(p_w.y)}"

                owner_matrix = entity.world_matrix if hasattr(entity, "world_matrix") else np.eye(3)
                pivot = clip.transform.pivot or clip.get_geometry_bounds().center
                clip_matrix = clip.transform.get_matrix(default_pivot=pivot)
                clip_to_world_matrix = owner_matrix @ clip_matrix

                commands = []
                for subpath in clip.subpaths:
                    cur_loc = Point(0.0, 0.0)
                    for cmd in subpath.commands:
                        if isinstance(cmd, (MoveTo, LineTo)):
                            commands.append(("M " if isinstance(cmd, MoveTo) else "L ") + coords(cmd.point))
                            cur_loc = cmd.point
                        elif isinstance(cmd, QuadraticTo):
                            commands.append("Q " + coords(cmd.control) + " " + coords(cmd.end))
                            cur_loc = cmd.end
                        elif isinstance(cmd, CubicTo):
                            commands.append("C " + coords(cmd.control1) + " " + coords(cmd.control2) + " " + coords(cmd.end))
                            cur_loc = cmd.end
                        elif isinstance(cmd, EllipticalArcTo):
                            try:
                                _, p2_w, rx_w, ry_w, phi_w, large_w, sweep_w = transform_elliptical_arc(
                                    cur_loc, cmd.end, cmd.radius_x, cmd.radius_y, cmd.x_axis_rotation, cmd.large_arc, cmd.sweep, clip_to_world_matrix
                                )
                            except ValidationError as e:
                                raise RenderError(f"Cannot export EllipticalArcTo in clip under singular or near-singular transform: {e}") from e
                            commands.append(
                                f"A {_number(rx_w)} {_number(ry_w)} {_number(phi_w)} "
                                f"{1 if large_w else 0} {1 if sweep_w else 0} "
                                f"{_number(p2_w.x)} {_number(p2_w.y)}"
                            )
                            cur_loc = cmd.end
                        elif isinstance(cmd, Close):
                            commands.append("Z")
                        else:
                            raise RenderError("Unsupported path command in clip")
                    if subpath.commands and not isinstance(subpath.commands[-1], Close):
                        # Implicit closure for filled clip boundary in SVG
                        commands.append("Z")

                d_str = " ".join(commands)
                rule = "nonzero" if clip.fill_rule == FillRule.NON_ZERO else "evenodd"
                ET.SubElement(node, "path", {"d": d_str, "clip-rule": rule})
            else:
                points = clip.corners if isinstance(clip, ClipRect) else clip.points
                points = [map_point(p) for p in points]
                ET.SubElement(node, "path", {"d": _points(points, True), "clip-rule": "evenodd"})

            group.set("clip-path", f"url(#{ident})")
        if isinstance(entity, (Layer, Group)):
            children = entity.objects if isinstance(entity, Layer) else entity.children
            for _, child in sorted(enumerate(children), key=lambda pair: (pair[1].z_index, pair[0])):
                self.entity(child, group)
        elif isinstance(entity, Text):
            text_attrs = {
                # The enclosing group's clip is already in world coordinates.
                # Transform only the text, leaving that clip in the same space.
                "transform": _matrix(entity.world_matrix),
                "x": _number(entity.position.x),
                "y": _number(entity.position.y),
                "font-size": _number(entity.font_size),
            }
            if entity.fill_none:
                text_attrs["fill"] = "none"
            else:
                text_attrs["fill"] = _color(entity.color)
                eff_fill_op = entity.fill_opacity * entity.color.a
                if eff_fill_op < 1.0:
                    text_attrs["fill-opacity"] = _number(eff_fill_op)

            if entity.font_family_name is not None:
                text_attrs["font-family"] = entity.font_family_name
            if entity.font_weight is not None and str(entity.font_weight).lower() not in ("400", "normal"):
                text_attrs["font-weight"] = str(entity.font_weight)
            if entity.font_style is not None and str(entity.font_style).lower() != "normal":
                text_attrs["font-style"] = str(entity.font_style)
            if entity.text_anchor is not None and entity.text_anchor != TextAnchor.START:
                text_attrs["text-anchor"] = entity.text_anchor.value if isinstance(entity.text_anchor, TextAnchor) else str(entity.text_anchor)
            if entity.xml_space == "preserve":
                text_attrs["xml:space"] = "preserve"
            if entity.direction == "rtl":
                text_attrs["direction"] = "rtl"
            text_node = ET.SubElement(group, "text", text_attrs)
            if entity.runs:
                eff_root_dir = "rtl" if entity.direction == "rtl" else "ltr"
                for r in entity.runs:
                    tspan_attrs = {}
                    if r.x is not None:
                        tspan_attrs["x"] = _number(r.x)
                    if r.y is not None:
                        tspan_attrs["y"] = _number(r.y)
                    if r.dx != 0.0:
                        tspan_attrs["dx"] = _number(r.dx)
                    if r.dy != 0.0:
                        tspan_attrs["dy"] = _number(r.dy)
                    if r.font_size is not None and not math.isclose(r.font_size, entity.font_size):
                        tspan_attrs["font-size"] = _number(r.font_size)
                    if r.font_family_name is not None and r.font_family_name != entity.font_family_name:
                        tspan_attrs["font-family"] = r.font_family_name
                    if r.font_weight is not None and r.font_weight != entity.font_weight:
                        tspan_attrs["font-weight"] = str(r.font_weight)
                    if r.font_style is not None and r.font_style != entity.font_style:
                        tspan_attrs["font-style"] = str(r.font_style)
                    if r.fill_none:
                        tspan_attrs["fill"] = "none"
                    else:
                        if r.fill is not None:
                            run_fill_color = r.fill
                            if entity.fill_none or (run_fill_color.r, run_fill_color.g, run_fill_color.b) != (entity.color.r, entity.color.g, entity.color.b):
                                tspan_attrs["fill"] = _color(run_fill_color)
                        else:
                            run_fill_color = entity.color

                        run_prop_op = r.fill_opacity if r.fill_opacity is not None else entity.fill_opacity
                        run_eff_svg_fill_op = run_prop_op * run_fill_color.a
                        inherited_svg_fill_op = eff_fill_op if not entity.fill_none else 1.0
                        if not math.isclose(run_eff_svg_fill_op, inherited_svg_fill_op, abs_tol=1e-5):
                            tspan_attrs["fill-opacity"] = _number(run_eff_svg_fill_op)
                    if r.text_anchor is not None and r.text_anchor != entity.text_anchor:
                        tspan_attrs["text-anchor"] = r.text_anchor.value if isinstance(r.text_anchor, TextAnchor) else str(r.text_anchor)
                    if r.direction is not None and r.direction != eff_root_dir:
                        tspan_attrs["direction"] = r.direction
                    if r.xml_space is not None and r.xml_space != entity.xml_space:
                        tspan_attrs["xml:space"] = r.xml_space
                    tspan = ET.SubElement(text_node, "tspan", tspan_attrs)
                    tspan.text = r.text
            else:
                text_node.text = entity.text
        else:
            stroke = getattr(entity, "stroke", None)
            local = stroke is not None and stroke.space == "object"
            attrs = {"d": self.geometry(entity, local=local), "fill": "none", "stroke": "none"}
            if local:
                attrs["transform"] = _matrix(entity.world_matrix)
            fill = getattr(entity, "fill", None)
            if isinstance(entity, Arc) and entity.closure == ArcClosure.OPEN:
                fill = None
            if fill is not None and fill.enabled:
                attrs["fill"] = self.paint(fill.paint, entity, local=local)
                attrs["fill-opacity"] = _number(fill.opacity * (fill.paint.a if isinstance(fill.paint, Color) else 1))
                attrs["fill-rule"] = "nonzero" if getattr(entity, "fill_rule", None) == FillRule.NON_ZERO else "evenodd"
            stroke = getattr(entity, "stroke", None)
            if stroke is not None and getattr(stroke, "width", 1.0) > 0 and getattr(stroke, "opacity", 1.0) > 0:
                if isinstance(stroke.paint, Color):
                    stroke_color = _color(stroke.paint)
                    stroke_opacity = stroke.opacity * stroke.paint.a
                else:
                    stroke_color = self.paint(stroke.paint, entity, local=local)
                    stroke_opacity = stroke.opacity
                attrs.update({"stroke": stroke_color, "stroke-opacity": _number(stroke_opacity),
                    "stroke-width": _number(stroke.width), "stroke-linecap": stroke.cap_style.value,
                    "stroke-linejoin": stroke.join_style.value, "stroke-miterlimit": _number(stroke.miter_limit)})
                if stroke.space == "screen":
                    attrs["vector-effect"] = "non-scaling-stroke"
                if stroke.dash_array:
                    attrs["stroke-dasharray"] = " ".join(map(_number, stroke.dash_array))
                    attrs["stroke-dashoffset"] = _number(stroke.dash_offset)
            ET.SubElement(group, "path", attrs)

    @classmethod
    def _paint_fallback_reason(cls, paint):
        if isinstance(paint, Color):
            return None
        if isinstance(paint, ConicGradient):
            return "conic gradient paint"
        if isinstance(paint, ImagePaint):
            if paint.repeat != "repeat":
                return f"image paint {paint.repeat} repeat"
            if paint.interpolation != ImageInterpolation.LINEAR:
                return f"image paint {paint.interpolation.value} interpolation"
            return None
        if isinstance(paint, (LinearGradient, RadialGradient)):
            return None
        return f"unsupported paint {type(paint).__name__}"

    @classmethod
    def fallback_reason(cls, entity):
        if entity.effects:
            return "effects"
        if entity.mask is not None:
            return "mask"
        if entity.clip is not None and not isinstance(entity.clip, (ClipRect, ClipPath, Path)):
            return "unsupported clip"
        if isinstance(entity, FreehandStroke) and entity.variable_width:
            return "variable-width stroke"
        if isinstance(entity, Text):
            return _text_native_svg_fallback_reason(entity)
        fill = getattr(entity, "fill", None)
        if fill is not None and fill.enabled:
            reason = cls._paint_fallback_reason(fill.paint)
            if reason:
                return reason
        stroke = getattr(entity, "stroke", None)
        if stroke is not None and getattr(stroke, "width", 1.0) > 0 and getattr(stroke, "opacity", 1.0) > 0:
            reason = cls._paint_fallback_reason(stroke.paint)
            if reason:
                return f"stroke {reason}"
        supported = (Layer, Group, Line, Rectangle, Circle, Ellipse, Polygon, Polyline,
                     RoundedRectangle, Arc, BezierCurve, Path, FreehandStroke, Text)
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

    def paint(self, paint, entity, *, local=False):
        if isinstance(paint, Color):
            return _color(paint)
        ident = self.identifier()
        paint_mat = paint.transform.get_matrix(Point(0.0, 0.0))
        if local and paint.space == "object":
            M = paint_mat
        elif local:
            # World-space paint must cancel the transform on the local path.
            # Singular paths have no painted area; pseudoinverse keeps export finite.
            M = np.linalg.pinv(entity.world_matrix) @ paint_mat
        elif paint.space == "object":
            M = entity.world_matrix @ paint_mat
        else:
            M = paint_mat

        if isinstance(paint, (LinearGradient, RadialGradient)):
            attrs = {"id": ident, "gradientUnits": "userSpaceOnUse", "spreadMethod": paint.spread,
                     "color-interpolation": "sRGB"}
            if paint.space == "object" or not np.allclose(M, np.eye(3)):
                attrs["gradientTransform"] = _matrix(M)
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
        elif isinstance(paint, ImagePaint):
            h, w = paint.image.shape[:2]
            pw = float(w) * float(paint.scale[0])
            ph = float(h) * float(paint.scale[1])
            attrs = {
                "id": ident,
                "patternUnits": "userSpaceOnUse",
                "width": _number(pw),
                "height": _number(ph),
                "x": _number(paint.origin.x),
                "y": _number(paint.origin.y),
            }
            if paint.space == "object" or not np.allclose(M, np.eye(3)):
                attrs["patternTransform"] = _matrix(M)
            pattern = ET.SubElement(self.defs, "pattern", attrs)
            ok, encoded = cv2.imencode(".png", paint.image)
            if not ok:
                raise RenderError("Could not encode SVG ImagePaint PNG")
            img_attrs = {
                "width": _number(pw),
                "height": _number(ph),
                "preserveAspectRatio": "none",
                "{http://www.w3.org/1999/xlink}href": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii"),
            }
            if paint.opacity < 1.0:
                img_attrs["opacity"] = _number(paint.opacity)
            ET.SubElement(pattern, "image", img_attrs)
            return f"url(#{ident})"
        else:
            raise RenderError(f"Unsupported paint type for SVG export: {type(paint).__name__}")

    @staticmethod
    def geometry(obj, *, local=False):
        def coords(p):
            p = p if local else obj.to_world(p)
            return f"{_number(p.x)} {_number(p.y)}"
        if isinstance(obj, Path):
            commands = []
            for subpath in obj.subpaths:
                cur_loc = Point(0.0, 0.0)
                for cmd in subpath.commands:
                    if isinstance(cmd, (MoveTo, LineTo)):
                        commands.append(("M " if isinstance(cmd, MoveTo) else "L ") + coords(cmd.point))
                        cur_loc = cmd.point
                    elif isinstance(cmd, QuadraticTo):
                        commands.append("Q " + coords(cmd.control) + " " + coords(cmd.end))
                        cur_loc = cmd.end
                    elif isinstance(cmd, CubicTo):
                        commands.append("C " + coords(cmd.control1) + " " + coords(cmd.control2) + " " + coords(cmd.end))
                        cur_loc = cmd.end
                    elif isinstance(cmd, EllipticalArcTo):
                        try:
                            _, p2_w, rx_w, ry_w, phi_w, large_w, sweep_w = transform_elliptical_arc(
                                cur_loc, cmd.end, cmd.radius_x, cmd.radius_y, cmd.x_axis_rotation, cmd.large_arc, cmd.sweep, np.eye(3) if local else obj.world_matrix
                            )
                        except ValidationError as e:
                            raise RenderError(f"Cannot export EllipticalArcTo under singular or near-singular transform: {e}") from e
                        commands.append(
                            f"A {_number(rx_w)} {_number(ry_w)} {_number(phi_w)} "
                            f"{1 if large_w else 0} {1 if sweep_w else 0} "
                            f"{_number(p2_w.x)} {_number(p2_w.y)}"
                        )
                        cur_loc = cmd.end
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
        return _points((p if local else obj.to_world(p) for p in points), closed)
