"""OpenCV raster rendering engine for DrawCV scenes."""

from __future__ import annotations
import math
import cv2
import numpy as np

from drawcv.canvas import Canvas
from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import ArcClosure, ArrowHeadStyle, FillRule, LineType
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import evaluate_fill_rule_mask, flatten_arc
from drawcv.group import Group
from drawcv.scene import Scene
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.line import Line
from drawcv.shapes.path import Path
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle


class OpenCVRenderer:
    """Raster renderer translating DrawCV retained scene graphs into Canvas images using OpenCV.
    
    The renderer is the sole component responsible for OpenCV drawing primitives,
    color space conversion (RGB -> BGR), and mathematical alpha compositing.
    """

    def render(self, scene: Scene) -> Canvas:
        """Render the given Scene and return a new Canvas instance."""
        if not isinstance(scene, Scene):
            raise ValidationError(f"Expected Scene, got {type(scene).__name__}")

        canvas = Canvas(scene.width, scene.height)
        canvas.clear(scene.background)

        # 1. Iterate through layers in ascending z_order (creation order as tie-breaker)
        for layer in scene.layers:
            if not layer.visible or layer.opacity <= 0.0:
                continue

            # 2. Within each layer, sort top-level drawables by ascending (z_index, insertion_order)
            visible_items = [
                (idx, obj) for idx, obj in enumerate(layer.objects)
                if obj.visible and obj.opacity > 0.0
            ]
            sorted_items = sorted(visible_items, key=lambda item: (item[1].z_index, item[0]))

            for _, drawable in sorted_items:
                self._render_single(drawable, canvas)

        return canvas

    def _render_single(self, drawable: Drawable, canvas: Canvas) -> None:
        """Dispatch rendering for a single drawable."""
        if isinstance(drawable, Group):
            self._render_group(drawable, canvas)
        elif isinstance(drawable, Line):
            self._render_line(drawable, canvas)
        elif isinstance(drawable, Rectangle):
            self._render_rectangle(drawable, canvas)
        elif isinstance(drawable, Circle):
            self._render_circle(drawable, canvas)
        elif isinstance(drawable, Ellipse):
            self._render_ellipse(drawable, canvas)
        elif isinstance(drawable, Polygon):
            self._render_polygon(drawable, canvas)
        elif isinstance(drawable, Polyline):
            self._render_polyline(drawable, canvas)
        elif isinstance(drawable, RoundedRectangle):
            self._render_rounded_rectangle(drawable, canvas)
        elif isinstance(drawable, Arc):
            self._render_arc(drawable, canvas)
        elif isinstance(drawable, Arrow):
            self._render_arrow(drawable, canvas)
        elif isinstance(drawable, BezierCurve):
            self._render_bezier(drawable, canvas)
        elif isinstance(drawable, Path):
            self._render_path(drawable, canvas)
        else:
            raise RenderError(f"Unsupported drawable type: {type(drawable).__name__}")

    def _render_group(self, group: Group, canvas: Canvas) -> None:
        """Render visible children of a Group in deterministic order."""
        if not group.effective_visible or group.effective_opacity <= 0.0:
            return

        visible_children = [
            (idx, child) for idx, child in enumerate(group.children)
            if child.visible and child.opacity > 0.0
        ]
        sorted_children = sorted(visible_children, key=lambda item: (item[1].z_index, item[0]))

        for _, child in sorted_children:
            self._render_single(child, canvas)


    # -------------------------------------------------------------------------
    # Centralized Shape Renderers
    # -------------------------------------------------------------------------

    def _render_line(self, line: Line, canvas: Canvas) -> None:
        stroke = line.stroke
        if stroke is None or stroke.width <= 0:
            return

        eff_alpha = stroke.color.a * stroke.opacity * line.effective_opacity
        if eff_alpha <= 0.0:
            return

        w_p1 = line.to_world(line.start)
        w_p2 = line.to_world(line.end)
        p1 = (int(round(w_p1.x)), int(round(w_p1.y)))
        p2 = (int(round(w_p2.x)), int(round(w_p2.y)))
        thickness = max(1, int(round(stroke.width)))
        cv_line_type = self._get_cv_line_type(stroke.line_type)
        bgr = stroke.color.to_bgr()

        if eff_alpha >= 1.0:
            cv2.line(canvas.buffer, p1, p2, bgr, thickness, cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.line(mask, p1, p2, 255, thickness, cv_line_type)
            bounds = line.get_bounds()
            self._composite_mask(canvas, mask, bgr, eff_alpha, bounds)

    def _render_rectangle(self, rect: Rectangle, canvas: Canvas) -> None:
        if rect.width <= 0 or rect.height <= 0:
            return

        world_corners = [rect.to_world(c) for c in rect.corners]
        pts = np.array(
            [[int(round(c.x)), int(round(c.y))] for c in world_corners],
            dtype=np.int32
        ).reshape((-1, 1, 2))
        bounds = rect.get_bounds()

        # 1. Render Fill if enabled
        fill = rect.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * rect.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Render Stroke if specified (non-scaling screen thickness)
        stroke = rect.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * rect.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_circle(self, circle: Circle, canvas: Canvas) -> None:
        if circle.radius <= 0:
            return

        w_center = circle.to_world(circle.center)
        center_pt = (int(round(w_center.x)), int(round(w_center.y)))
        bounds = circle.get_bounds()

        sx = circle.transform.scale_x
        sy = circle.transform.scale_y
        r = float(circle.radius)
        is_uniform = math.isclose(sx, sy, rel_tol=1e-5)

        # 1. Render Fill if enabled
        fill = circle.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * circle.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if is_uniform:
                    scaled_r = max(1, int(round(r * sx)))
                    if fill_alpha >= 1.0:
                        cv2.circle(canvas.buffer, center_pt, scaled_r, fill_bgr, -1, cv2.LINE_AA)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.circle(mask, center_pt, scaled_r, 255, -1, cv2.LINE_AA)
                        self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
                else:
                    axes = (max(1, int(round(r * sx))), max(1, int(round(r * sy))))
                    angle = circle.transform.rotation
                    if fill_alpha >= 1.0:
                        cv2.ellipse(canvas.buffer, center_pt, axes, angle, 0, 360, fill_bgr, -1, cv2.LINE_AA)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.ellipse(mask, center_pt, axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
                        self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Render Stroke if specified (non-scaling screen thickness)
        stroke = circle.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * circle.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if is_uniform:
                    scaled_r = max(1, int(round(r * sx)))
                    if stroke_alpha >= 1.0:
                        cv2.circle(canvas.buffer, center_pt, scaled_r, stroke_bgr, thickness, cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.circle(mask, center_pt, scaled_r, 255, thickness, cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)
                else:
                    axes = (max(1, int(round(r * sx))), max(1, int(round(r * sy))))
                    angle = circle.transform.rotation
                    if stroke_alpha >= 1.0:
                        cv2.ellipse(canvas.buffer, center_pt, axes, angle, 0, 360, stroke_bgr, thickness, cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.ellipse(mask, center_pt, axes, angle, 0, 360, 255, thickness, cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_ellipse(self, ellipse: Ellipse, canvas: Canvas) -> None:
        if ellipse.radius_x <= 0 or ellipse.radius_y <= 0:
            return

        sampled = flatten_arc(ellipse.center, ellipse.radius_x, ellipse.radius_y, 0.0, 360.0, tolerance=0.5)
        world_pts = [ellipse.to_world(p) for p in sampled]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = ellipse.get_bounds()

        # 1. Fill
        fill = ellipse.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * ellipse.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = ellipse.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * ellipse.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_polygon(self, polygon: Polygon, canvas: Canvas) -> None:
        if len(polygon.vertices) < 3:
            return

        world_pts = [polygon.to_world(p) for p in polygon.vertices]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = polygon.get_bounds()

        # 1. Fill
        fill = polygon.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * polygon.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = polygon.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * polygon.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_polyline(self, polyline: Polyline, canvas: Canvas) -> None:
        if len(polyline.points) < 2:
            return

        world_pts = [polyline.to_world(p) for p in polyline.points]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = polyline.get_bounds()

        stroke = polyline.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * polyline.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=polyline.closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=polyline.closed, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_rounded_rectangle(self, rect: RoundedRectangle, canvas: Canvas) -> None:
        local_pts = rect.get_contour_points(tolerance=0.5)
        world_pts = [rect.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = rect.get_bounds()

        # 1. Fill
        fill = rect.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * rect.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = rect.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * rect.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_arc(self, arc: Arc, canvas: Canvas) -> None:
        local_pts = arc.get_contour_points(tolerance=0.5)
        world_pts = [arc.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = arc.get_bounds()
        is_closed = (arc.closure != ArcClosure.OPEN)

        # 1. Fill (only for CHORD or PIE)
        fill = arc.fill
        if is_closed and fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * arc.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = arc.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * arc.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if stroke_alpha >= 1.0:
                    cv2.polylines(canvas.buffer, [pts], isClosed=is_closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=is_closed, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_arrow(self, arrow: Arrow, canvas: Canvas) -> None:
        stroke = arrow.stroke
        if stroke is None or stroke.width <= 0:
            return

        stroke_alpha = stroke.color.a * stroke.opacity * arrow.effective_opacity
        if stroke_alpha <= 0.0:
            return

        w_start = arrow.to_world(arrow.start)
        w_end, head_pts = arrow.get_world_head_geometry()
        bounds = arrow.get_bounds()
        stroke_bgr = stroke.color.to_bgr()
        thickness = max(1, int(round(stroke.width)))
        cv_line_type = self._get_cv_line_type(stroke.line_type)

        # 1. Shaft line
        p1 = (int(round(w_start.x)), int(round(w_start.y)))
        p2 = (int(round(w_end.x)), int(round(w_end.y)))
        if stroke_alpha >= 1.0:
            cv2.line(canvas.buffer, p1, p2, stroke_bgr, thickness, cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.line(mask, p1, p2, 255, thickness, cv_line_type)
            self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

        # 2. Arrowhead
        fill = arrow.fill
        fill_alpha = (fill.color.a * fill.opacity * arrow.effective_opacity) if (fill and fill.enabled) else 0.0
        fill_bgr = fill.color.to_bgr() if fill else stroke_bgr

        if arrow.head_style in (ArrowHeadStyle.TRIANGLE, ArrowHeadStyle.DIAMOND):
            pts_head = np.array([[int(round(p.x)), int(round(p.y))] for p in head_pts], dtype=np.int32).reshape((-1, 1, 2))
            if fill_alpha > 0.0:
                if fill_alpha >= 1.0:
                    cv2.fillPoly(canvas.buffer, [pts_head], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts_head], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
            if stroke_alpha >= 1.0:
                cv2.polylines(canvas.buffer, [pts_head], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
            else:
                mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                cv2.polylines(mask, [pts_head], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

        elif arrow.head_style == ArrowHeadStyle.OPEN:
            pts_head = np.array([[int(round(p.x)), int(round(p.y))] for p in head_pts], dtype=np.int32).reshape((-1, 1, 2))
            if stroke_alpha >= 1.0:
                cv2.polylines(canvas.buffer, [pts_head], isClosed=False, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
            else:
                mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                cv2.polylines(mask, [pts_head], isClosed=False, color=255, thickness=thickness, lineType=cv_line_type)
                self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

        elif arrow.head_style == ArrowHeadStyle.CIRCLE:
            c = head_pts[0]
            r = max(1, int(round(head_pts[1].x)))
            c_int = (int(round(c.x)), int(round(c.y)))
            if fill_alpha > 0.0:
                if fill_alpha >= 1.0:
                    cv2.circle(canvas.buffer, c_int, r, fill_bgr, -1, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.circle(mask, c_int, r, 255, -1, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
            if stroke_alpha >= 1.0:
                cv2.circle(canvas.buffer, c_int, r, stroke_bgr, thickness, cv_line_type)
            else:
                mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                cv2.circle(mask, c_int, r, 255, thickness, cv_line_type)
                self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_bezier(self, bezier: BezierCurve, canvas: Canvas) -> None:
        stroke = bezier.stroke
        if stroke is None or stroke.width <= 0:
            return

        stroke_alpha = stroke.color.a * stroke.opacity * bezier.effective_opacity
        if stroke_alpha <= 0.0:
            return

        world_pts = bezier.flatten_world(tolerance=0.5)
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = bezier.get_bounds()
        stroke_bgr = stroke.color.to_bgr()
        thickness = max(1, int(round(stroke.width)))
        cv_line_type = self._get_cv_line_type(stroke.line_type)

        if stroke_alpha >= 1.0:
            cv2.polylines(canvas.buffer, [pts], isClosed=False, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv_line_type)
            self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_path(self, path: Path, canvas: Canvas) -> None:
        world_contours = path.flatten_world(tolerance=0.5)
        if not world_contours:
            return

        bounds = path.get_bounds()

        # 1. Fill using topological fill rule evaluator
        fill = path.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * path.effective_opacity
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                mask = evaluate_fill_rule_mask(
                    world_contours, path.fill_rule, canvas.width, canvas.height, supersample=2
                )
                self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = path.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * path.effective_opacity
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)

                for idx, contour in enumerate(world_contours):
                    if len(contour) < 2:
                        continue
                    is_closed = idx < len(path.subpaths) and path.subpaths[idx].closed
                    pts = np.array([[int(round(p.x)), int(round(p.y))] for p in contour], dtype=np.int32).reshape((-1, 1, 2))
                    if stroke_alpha >= 1.0:
                        cv2.polylines(canvas.buffer, [pts], isClosed=is_closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.polylines(mask, [pts], isClosed=is_closed, color=255, thickness=thickness, lineType=cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)


    # -------------------------------------------------------------------------
    # Utility Helpers
    # -------------------------------------------------------------------------

    def _get_cv_line_type(self, line_type: LineType) -> int:
        """Map LineType enum to OpenCV line connectivity constant."""
        if line_type == LineType.AA:
            return cv2.LINE_AA
        elif line_type == LineType.LINE_8:
            return cv2.LINE_8
        elif line_type == LineType.LINE_4:
            return cv2.LINE_4
        else:
            raise ValidationError(f"Unknown LineType: {line_type}")

    def _composite_mask(
        self,
        canvas: Canvas,
        mask: np.ndarray,
        color_bgr: tuple[int, int, int],
        alpha: float,
        bbox: BoundingBox | None = None
    ) -> None:
        """Perform mathematically exact alpha blending: dst = dst * (1 - a) + src * a."""
        if alpha <= 0.0:
            return

        if bbox is not None:
            margin = 3
            x1 = max(0, int(math.floor(bbox.left)) - margin)
            y1 = max(0, int(math.floor(bbox.top)) - margin)
            x2 = min(canvas.width, int(math.ceil(bbox.right)) + margin + 1)
            y2 = min(canvas.height, int(math.ceil(bbox.bottom)) + margin + 1)
        else:
            x1, y1, x2, y2 = 0, 0, canvas.width, canvas.height

        if x2 <= x1 or y2 <= y1:
            return

        sub_mask = mask[y1:y2, x1:x2]
        if not np.any(sub_mask):
            return

        coverage = (sub_mask.astype(np.float32) / 255.0) * float(alpha)
        color_vec = np.array(color_bgr, dtype=np.float32)

        sub_dst = canvas.buffer[y1:y2, x1:x2].astype(np.float32)
        blended = sub_dst * (1.0 - coverage[:, :, None]) + color_vec * coverage[:, :, None]
        canvas.buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
