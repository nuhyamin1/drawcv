"""OpenCV raster rendering engine for DrawCV scenes."""

from __future__ import annotations
import math
import cv2
import numpy as np
from typing import Any

from drawcv.canvas import Canvas
from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import (
    ArcClosure,
    ArrowHeadStyle,
    BlurType,
    FillRule,
    FontFamily,
    ImageInterpolation,
    LineType,
    MaskMapping,
    TextAlignment,
)
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import _FONT_FAMILY_TO_CV, evaluate_fill_rule_mask, flatten_arc
from drawcv.effects.blur import BlurEffect
from drawcv.effects.clipping import ClipPath, ClipRect
from drawcv.effects.mask import Mask
from drawcv.effects.shadow import ShadowEffect
from drawcv.group import Group
from drawcv.layer import Layer
from drawcv.scene import Scene
from drawcv.shapes.arc import Arc
from drawcv.shapes.arrow import Arrow
from drawcv.shapes.bezier import BezierCurve
from drawcv.shapes.circle import Circle
from drawcv.shapes.ellipse import Ellipse
from drawcv.shapes.freehand import FreehandStroke
from drawcv.shapes.image import ImageObject
from drawcv.shapes.line import Line
from drawcv.shapes.path import Path
from drawcv.shapes.polygon import Polygon
from drawcv.shapes.polyline import Polyline
from drawcv.shapes.rectangle import Rectangle
from drawcv.shapes.rounded_rectangle import RoundedRectangle
from drawcv.shapes.text import Text


class _IsolatedSurface:
    """Internal float32 BGRA offscreen render target for isolated compositing passes."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buffer = np.zeros((height, width, 4), dtype=np.float32)
        self.is_isolated = True


class OpenCVRenderer:
    """Raster renderer translating DrawCV retained scene graphs into Canvas images using OpenCV.
    
    The renderer is the sole component responsible for OpenCV drawing primitives,
    color space conversion (RGB -> BGR), and mathematical alpha compositing.
    """

    def __init__(self):
        self._current_isolating_ancestor: Any | None = None

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

            if self._needs_isolated_compositing(layer):
                self._render_isolated(layer, canvas)
            else:
                # 2. Within each layer, sort top-level drawables by ascending (z_index, insertion_order)
                visible_items = [
                    (idx, obj) for idx, obj in enumerate(layer.objects)
                    if obj.visible and obj.opacity > 0.0
                ]
                sorted_items = sorted(visible_items, key=lambda item: (item[1].z_index, item[0]))

                for _, drawable in sorted_items:
                    self._render_single(drawable, canvas)

        return canvas

    # -------------------------------------------------------------------------
    # Compositing & Isolation Dispatch
    # -------------------------------------------------------------------------

    def _needs_isolated_compositing(self, entity: Drawable | Layer) -> bool:
        """Determine whether an entity requires an isolated offscreen buffer pass."""
        if entity.clip is not None:
            return True
        if entity.mask is not None:
            return True
        if entity.effects and len(entity.effects) > 0:
            return True
        if isinstance(entity, (Group, Layer)) and entity.opacity < 1.0:
            return True
        return False

    def _get_render_opacity(self, drawable: Drawable) -> float:
        """Calculate effective opacity for rendering, respecting the current isolating ancestor."""
        if self._current_isolating_ancestor is drawable:
            return 1.0

        op = float(drawable.opacity)
        curr = drawable._parent
        while curr is not None and curr is not self._current_isolating_ancestor:
            op *= float(curr.opacity)
            curr = curr._parent

        if curr is None and self._current_isolating_ancestor is None and drawable._layer is not None:
            op *= float(drawable._layer.opacity)

        return float(op)

    def _is_direct_draw(self, canvas: Any, alpha: float) -> bool:
        """Check if direct uint8 buffer drawing is safe without mask compositing."""
        return (alpha >= 1.0) and not getattr(canvas, "is_isolated", False)

    def _render_single(self, drawable: Drawable, canvas: Canvas | _IsolatedSurface) -> None:
        """Dispatch rendering for a single drawable."""
        if not drawable.effective_visible:
            return

        if self._needs_isolated_compositing(drawable):
            self._render_isolated(drawable, canvas)
            return

        if isinstance(drawable, Group):
            self._render_group(drawable, canvas)
        else:
            self._render_single_primitive(drawable, canvas)

    def _render_group(self, group: Group, canvas: Canvas | _IsolatedSurface) -> None:
        """Render visible children of a Group in deterministic order."""
        if not group.effective_visible or group.effective_opacity <= 0.0:
            return

        if self._needs_isolated_compositing(group):
            self._render_isolated(group, canvas)
            return

        self._render_group_children(group, canvas)

    def _render_group_children(self, group: Group, canvas: Canvas | _IsolatedSurface) -> None:
        """Render group children without re-checking group-level isolation."""
        visible_children = [
            (idx, child) for idx, child in enumerate(group.children)
            if child.visible and child.opacity > 0.0
        ]
        sorted_children = sorted(visible_children, key=lambda item: (item[1].z_index, item[0]))

        for _, child in sorted_children:
            self._render_single(child, canvas)

    def _render_single_primitive(self, drawable: Drawable, canvas: Canvas | _IsolatedSurface) -> None:
        """Dispatch rendering for a leaf drawable without re-checking drawable-level isolation."""
        if isinstance(drawable, Line):
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
        elif isinstance(drawable, FreehandStroke):
            self._render_freehand(drawable, canvas)
        elif isinstance(drawable, ImageObject):
            self._render_image(drawable, canvas)
        elif isinstance(drawable, Text):
            self._render_text(drawable, canvas)
        else:
            raise RenderError(f"Unsupported drawable type: {type(drawable).__name__}")

    # -------------------------------------------------------------------------
    # Isolated Offscreen Compositing Engine & Effects Pipeline
    # -------------------------------------------------------------------------

    def _render_isolated(self, entity: Drawable | Layer, destination: Canvas | _IsolatedSurface) -> None:
        """Render an entity through an isolated offscreen buffer pass with effects pipeline."""
        if not entity.visible:
            return

        eff_bounds = entity.get_effect_bounds()
        pre_effect_bounds = entity.get_bounds()

        margin = 3
        x1 = max(0, int(math.floor(eff_bounds.left)) - margin)
        y1 = max(0, int(math.floor(eff_bounds.top)) - margin)
        x2 = min(destination.width, int(math.ceil(eff_bounds.right)) + margin + 1)
        y2 = min(destination.height, int(math.ceil(eff_bounds.bottom)) + margin + 1)

        if x2 <= x1 or y2 <= y1:
            return

        # 1. Allocate isolated float32 BGRA surface
        base_surface = _IsolatedSurface(destination.width, destination.height)

        # 2. Render base primitives into isolated surface with entity opacity withheld
        prev_ancestor = self._current_isolating_ancestor
        self._current_isolating_ancestor = entity
        try:
            if isinstance(entity, Layer):
                visible_items = [
                    (idx, obj) for idx, obj in enumerate(entity.objects)
                    if obj.visible and obj.opacity > 0.0
                ]
                sorted_items = sorted(visible_items, key=lambda item: (item[1].z_index, item[0]))
                for _, obj in sorted_items:
                    self._render_single(obj, base_surface)
            elif isinstance(entity, Group):
                self._render_group_children(entity, base_surface)
            else:
                self._render_single_primitive(entity, base_surface)
        finally:
            self._current_isolating_ancestor = prev_ancestor

        base_buffer = base_surface.buffer

        # Check if anything was rendered
        if not np.any(base_buffer[y1:y2, x1:x2, 3] > 0.0):
            return

        # 3. Effects Pipeline
        effects = getattr(entity, "effects", []) or []
        shadow_effect = next((e for e in effects if isinstance(e, ShadowEffect)), None)
        blur_effect = next((e for e in effects if isinstance(e, BlurEffect)), None)

        # a. Shadow Generation
        shadow_buffer: np.ndarray | None = None
        if shadow_effect is not None and shadow_effect.opacity > 0.0:
            sh_alpha = base_buffer[:, :, 3] * float(shadow_effect.color.a * shadow_effect.opacity)
            M_trans = np.float32([[1, 0, shadow_effect.offset_x], [0, 1, shadow_effect.offset_y]])
            shifted_alpha = cv2.warpAffine(
                sh_alpha, M_trans, (destination.width, destination.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
            )
            if shadow_effect.blur_radius > 0.0:
                sigma = float(shadow_effect.blur_radius)
                ksize = int(math.ceil(sigma * 3.0)) * 2 + 1
                blurred_alpha = cv2.GaussianBlur(shifted_alpha, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            else:
                blurred_alpha = shifted_alpha

            sh_b, sh_g, sh_r = shadow_effect.color.to_bgr()
            sh_color_vec = np.array([sh_b, sh_g, sh_r], dtype=np.float32)
            sh_rgb_pm = sh_color_vec * blurred_alpha[:, :, None]
            shadow_buffer = np.dstack([sh_rgb_pm, blurred_alpha])

        # b. Content Blur (Premultiplied throughout)
        if blur_effect is not None:
            k = blur_effect.kernel_size
            if blur_effect.blur_type == BlurType.GAUSSIAN:
                sig = blur_effect.sigma
                base_buffer[y1:y2, x1:x2] = cv2.GaussianBlur(
                    base_buffer[y1:y2, x1:x2], (k, k), sigmaX=sig, sigmaY=sig
                )
            else:
                base_buffer[y1:y2, x1:x2] = cv2.blur(
                    base_buffer[y1:y2, x1:x2], (k, k)
                )

        # c. Merge Shadow Behind Base
        if shadow_buffer is not None:
            sub_base = base_buffer[y1:y2, x1:x2]
            sub_shadow = shadow_buffer[y1:y2, x1:x2]
            base_rgb = sub_base[:, :, :3]
            base_a = sub_base[:, :, 3:4]
            sh_rgb = sub_shadow[:, :, :3]
            sh_a = sub_shadow[:, :, 3:4]

            # Porter-Duff: base OVER shadow
            merged_rgb = base_rgb + sh_rgb * (1.0 - base_a)
            merged_a = base_a + sh_a * (1.0 - base_a)
            sub_base[:, :, :3] = merged_rgb
            sub_base[:, :, 3] = merged_a[:, :, 0]

        # d. Mask Modulation (4-channel scale, mapped to pre-effect visual bounds)
        if getattr(entity, "mask", None) is not None:
            mask_obj = entity.mask
            full_mask = np.zeros((destination.height, destination.width), dtype=np.float32)
            if mask_obj.mapping == MaskMapping.FIT_BOUNDS:
                pb = pre_effect_bounds
                px1 = max(0, int(math.floor(pb.left)))
                py1 = max(0, int(math.floor(pb.top)))
                px2 = min(destination.width, int(math.ceil(pb.right)))
                py2 = min(destination.height, int(math.ceil(pb.bottom)))
                pw = px2 - px1
                ph = py2 - py1
                if pw > 0 and ph > 0:
                    cov = mask_obj.get_coverage(pw, ph)
                    full_mask[py1:py2, px1:px2] = cov
            else:
                full_mask = mask_obj.get_coverage(destination.width, destination.height)

            sub_mask = full_mask[y1:y2, x1:x2, None]
            base_buffer[y1:y2, x1:x2] *= sub_mask

        # e. Final Clip Stencil (Applied AFTER blur/shadow)
        if getattr(entity, "clip", None) is not None:
            clip_obj = entity.clip
            clip_mask = np.zeros((destination.height, destination.width), dtype=np.uint8)
            if isinstance(clip_obj, ClipRect):
                corners = [entity.to_world(c) for c in clip_obj.corners]
                pts = np.array([[int(round(p.x)), int(round(p.y))] for p in corners], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(clip_mask, [pts], 255)
            elif isinstance(clip_obj, ClipPath):
                pts_world = [entity.to_world(p) for p in clip_obj.points]
                pts = np.array([[int(round(p.x)), int(round(p.y))] for p in pts_world], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(clip_mask, [pts], 255)

            sub_clip = clip_mask[y1:y2, x1:x2]
            base_buffer[y1:y2, x1:x2][sub_clip == 0] = 0.0

        # f. Entity Opacity (4-channel scale)
        entity_op = float(entity.opacity)
        if entity_op < 1.0:
            base_buffer[y1:y2, x1:x2] *= entity_op

        # g. Destination Composite (premultiplied source-over)
        sub_src = base_buffer[y1:y2, x1:x2]
        src_rgb = sub_src[:, :, :3]
        src_a = sub_src[:, :, 3:4]

        if getattr(destination, "is_isolated", False):
            sub_dst = destination.buffer[y1:y2, x1:x2]
            dst_rgb = sub_dst[:, :, :3]
            dst_a = sub_dst[:, :, 3:4]
            sub_dst[:, :, :3] = src_rgb + dst_rgb * (1.0 - src_a)
            sub_dst[:, :, 3] = (src_a + dst_a * (1.0 - src_a))[:, :, 0]
        else:
            sub_dst = destination.buffer[y1:y2, x1:x2].astype(np.float32)
            blended = sub_dst * (1.0 - src_a) + src_rgb
            destination.buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Centralized Shape Renderers
    # -------------------------------------------------------------------------

    def _render_line(self, line: Line, canvas: Canvas | _IsolatedSurface) -> None:
        stroke = line.stroke
        if stroke is None or stroke.width <= 0:
            return

        eff_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(line)
        if eff_alpha <= 0.0:
            return

        w_p1 = line.to_world(line.start)
        w_p2 = line.to_world(line.end)
        p1 = (int(round(w_p1.x)), int(round(w_p1.y)))
        p2 = (int(round(w_p2.x)), int(round(w_p2.y)))
        thickness = max(1, int(round(stroke.width)))
        cv_line_type = self._get_cv_line_type(stroke.line_type)
        bgr = stroke.color.to_bgr()

        if self._is_direct_draw(canvas, eff_alpha):
            cv2.line(canvas.buffer, p1, p2, bgr, thickness, cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.line(mask, p1, p2, 255, thickness, cv_line_type)
            bounds = line.get_bounds()
            self._composite_mask(canvas, mask, bgr, eff_alpha, bounds)

    def _render_rectangle(self, rect: Rectangle, canvas: Canvas | _IsolatedSurface) -> None:
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
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(rect)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Render Stroke if specified (non-scaling screen thickness)
        stroke = rect.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(rect)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_circle(self, circle: Circle, canvas: Canvas | _IsolatedSurface) -> None:
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
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(circle)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if is_uniform:
                    scaled_r = max(1, int(round(r * sx)))
                    if self._is_direct_draw(canvas, fill_alpha):
                        cv2.circle(canvas.buffer, center_pt, scaled_r, fill_bgr, -1, cv2.LINE_AA)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.circle(mask, center_pt, scaled_r, 255, -1, cv2.LINE_AA)
                        self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
                else:
                    axes = (max(1, int(round(r * sx))), max(1, int(round(r * sy))))
                    angle = circle.transform.rotation
                    if self._is_direct_draw(canvas, fill_alpha):
                        cv2.ellipse(canvas.buffer, center_pt, axes, angle, 0, 360, fill_bgr, -1, cv2.LINE_AA)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.ellipse(mask, center_pt, axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
                        self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Render Stroke if specified (non-scaling screen thickness)
        stroke = circle.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(circle)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if is_uniform:
                    scaled_r = max(1, int(round(r * sx)))
                    if self._is_direct_draw(canvas, stroke_alpha):
                        cv2.circle(canvas.buffer, center_pt, scaled_r, stroke_bgr, thickness, cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.circle(mask, center_pt, scaled_r, 255, thickness, cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)
                else:
                    axes = (max(1, int(round(r * sx))), max(1, int(round(r * sy))))
                    angle = circle.transform.rotation
                    if self._is_direct_draw(canvas, stroke_alpha):
                        cv2.ellipse(canvas.buffer, center_pt, axes, angle, 0, 360, stroke_bgr, thickness, cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.ellipse(mask, center_pt, axes, angle, 0, 360, 255, thickness, cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_ellipse(self, ellipse: Ellipse, canvas: Canvas | _IsolatedSurface) -> None:
        if ellipse.radius_x <= 0 or ellipse.radius_y <= 0:
            return

        sampled = flatten_arc(ellipse.center, ellipse.radius_x, ellipse.radius_y, 0.0, 360.0, tolerance=0.5)
        world_pts = [ellipse.to_world(p) for p in sampled]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = ellipse.get_bounds()

        # 1. Fill
        fill = ellipse.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(ellipse)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = ellipse.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(ellipse)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_polygon(self, polygon: Polygon, canvas: Canvas | _IsolatedSurface) -> None:
        if len(polygon.vertices) < 3:
            return

        world_pts = [polygon.to_world(p) for p in polygon.vertices]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = polygon.get_bounds()

        # 1. Fill
        fill = polygon.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(polygon)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = polygon.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(polygon)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_polyline(self, polyline: Polyline, canvas: Canvas | _IsolatedSurface) -> None:
        if len(polyline.points) < 2:
            return

        world_pts = [polyline.to_world(p) for p in polyline.points]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = polyline.get_bounds()

        stroke = polyline.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(polyline)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=polyline.closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=polyline.closed, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_rounded_rectangle(self, rect: RoundedRectangle, canvas: Canvas | _IsolatedSurface) -> None:
        local_pts = rect.get_contour_points(tolerance=0.5)
        world_pts = [rect.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = rect.get_bounds()

        # 1. Fill
        fill = rect.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(rect)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = rect.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(rect)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_arc(self, arc: Arc, canvas: Canvas | _IsolatedSurface) -> None:
        local_pts = arc.get_contour_points(tolerance=0.5)
        world_pts = [arc.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = arc.get_bounds()
        is_closed = (arc.closure != ArcClosure.OPEN)

        # 1. Fill (only for CHORD or PIE)
        fill = arc.fill
        if is_closed and fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(arc)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = arc.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(arc)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts], isClosed=is_closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts], isClosed=is_closed, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_arrow(self, arrow: Arrow, canvas: Canvas | _IsolatedSurface) -> None:
        """Render an Arrow with shaft stroke and marker terminals."""
        stroke = arrow.stroke
        if stroke is None or stroke.width <= 0:
            return

        stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(arrow)
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
        if self._is_direct_draw(canvas, stroke_alpha):
            cv2.line(canvas.buffer, p1, p2, stroke_bgr, thickness, cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.line(mask, p1, p2, 255, thickness, cv_line_type)
            self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

        # 2. Arrowhead
        fill = arrow.fill
        fill_alpha = (fill.color.a * fill.opacity * self._get_render_opacity(arrow)) if (fill and fill.enabled) else 0.0
        fill_bgr = fill.color.to_bgr() if fill else stroke_bgr

        if arrow.head_style in (ArrowHeadStyle.TRIANGLE, ArrowHeadStyle.DIAMOND):
            pts_head = np.array([[int(round(p.x)), int(round(p.y))] for p in head_pts], dtype=np.int32).reshape((-1, 1, 2))
            if fill_alpha > 0.0:
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.fillPoly(canvas.buffer, [pts_head], fill_bgr, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts_head], 255, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
            if stroke_alpha > 0.0:
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.polylines(canvas.buffer, [pts_head], isClosed=True, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.polylines(mask, [pts_head], isClosed=True, color=255, thickness=thickness, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

        elif arrow.head_style == ArrowHeadStyle.OPEN:
            pts_head = np.array([[int(round(p.x)), int(round(p.y))] for p in head_pts], dtype=np.int32).reshape((-1, 1, 2))
            if stroke_alpha > 0.0:
                if self._is_direct_draw(canvas, stroke_alpha):
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
                if self._is_direct_draw(canvas, fill_alpha):
                    cv2.circle(canvas.buffer, c_int, r, fill_bgr, -1, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.circle(mask, c_int, r, 255, -1, cv2.LINE_AA)
                    self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)
            if stroke_alpha > 0.0:
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.circle(canvas.buffer, c_int, r, stroke_bgr, thickness, cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.circle(mask, c_int, r, 255, thickness, cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_bezier(self, bezier: BezierCurve, canvas: Canvas | _IsolatedSurface) -> None:
        stroke = bezier.stroke
        if stroke is None or stroke.width <= 0:
            return

        stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(bezier)
        if stroke_alpha <= 0.0:
            return

        world_pts = bezier.flatten_world(tolerance=0.5)
        if len(world_pts) < 2:
            return

        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        thickness = max(1, int(round(stroke.width)))
        cv_line_type = self._get_cv_line_type(stroke.line_type)
        stroke_bgr = stroke.color.to_bgr()
        bounds = bezier.get_bounds()

        if self._is_direct_draw(canvas, stroke_alpha):
            cv2.polylines(canvas.buffer, [pts], isClosed=False, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
        else:
            mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv_line_type)
            self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_path(self, path: Path, canvas: Canvas | _IsolatedSurface) -> None:
        world_contours = path.flatten_world(tolerance=0.5)
        if not world_contours:
            return

        bounds = path.get_bounds()

        # 1. Fill using topological fill rule evaluator
        fill = path.fill
        if fill is not None and fill.enabled and fill.opacity > 0.0:
            fill_alpha = fill.color.a * fill.opacity * self._get_render_opacity(path)
            if fill_alpha > 0.0:
                fill_bgr = fill.color.to_bgr()
                mask = evaluate_fill_rule_mask(
                    world_contours, path.fill_rule, canvas.width, canvas.height, supersample=2
                )
                self._composite_mask(canvas, mask, fill_bgr, fill_alpha, bounds)

        # 2. Stroke
        stroke = path.stroke
        if stroke is not None and stroke.width > 0 and stroke.opacity > 0.0:
            stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(path)
            if stroke_alpha > 0.0:
                stroke_bgr = stroke.color.to_bgr()
                thickness = max(1, int(round(stroke.width)))
                cv_line_type = self._get_cv_line_type(stroke.line_type)

                for idx, contour in enumerate(world_contours):
                    if len(contour) < 2:
                        continue
                    is_closed = idx < len(path.subpaths) and path.subpaths[idx].closed
                    pts = np.array([[int(round(p.x)), int(round(p.y))] for p in contour], dtype=np.int32).reshape((-1, 1, 2))
                    if self._is_direct_draw(canvas, stroke_alpha):
                        cv2.polylines(canvas.buffer, [pts], isClosed=is_closed, color=stroke_bgr, thickness=thickness, lineType=cv_line_type)
                    else:
                        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                        cv2.polylines(mask, [pts], isClosed=is_closed, color=255, thickness=thickness, lineType=cv_line_type)
                        self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)

    def _render_freehand(self, freehand: FreehandStroke, canvas: Canvas | _IsolatedSurface) -> None:
        """Render a FreehandStroke with constant width or variable width ribbon quads and round joints."""
        pts = freehand.get_processed_points()
        if len(pts) == 0:
            return

        stroke = freehand.stroke
        if stroke is None or stroke.width <= 0:
            return

        stroke_alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(freehand)
        if stroke_alpha <= 0.0:
            return

        world_pts = [freehand.to_world(Point(p.x, p.y)) for p in pts]
        stroke_bgr = stroke.color.to_bgr()
        cv_line_type = self._get_cv_line_type(stroke.line_type)
        bounds = freehand.get_bounds()

        if not freehand.variable_width:
            base_thickness = max(1, int(round(stroke.width)))
            cv_pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
            allow_direct = self._is_direct_draw(canvas, stroke_alpha)
            if allow_direct:
                cv2.polylines(canvas.buffer, [cv_pts], isClosed=False, color=stroke_bgr, thickness=base_thickness, lineType=cv_line_type)
            else:
                mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                cv2.polylines(mask, [cv_pts], isClosed=False, color=255, thickness=base_thickness, lineType=cv_line_type)
                self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)
        else:
            widths = freehand.get_point_widths(pts)
            if len(world_pts) == 1:
                r = max(1, int(round(widths[0] / 2.0)))
                center_pt = (int(round(world_pts[0].x)), int(round(world_pts[0].y)))
                if self._is_direct_draw(canvas, stroke_alpha):
                    cv2.circle(canvas.buffer, center_pt, r, stroke_bgr, -1, lineType=cv_line_type)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.circle(mask, center_pt, r, 255, -1, lineType=cv_line_type)
                    self._composite_mask(canvas, mask, stroke_bgr, stroke_alpha, bounds)
                return

            allow_direct = self._is_direct_draw(canvas, stroke_alpha)
            target_buffer = canvas.buffer if allow_direct else np.zeros((canvas.height, canvas.width), dtype=np.uint8)
            draw_color = stroke_bgr if allow_direct else 255

            for i in range(len(world_pts) - 1):
                p1 = world_pts[i]
                p2 = world_pts[i + 1]
                w1 = widths[i]
                w2 = widths[i + 1]

                dx = p2.x - p1.x
                dy = p2.y - p1.y
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue

                nx = -dy / seg_len
                ny = dx / seg_len

                half_w1 = w1 / 2.0
                half_w2 = w2 / 2.0

                q1 = (int(round(p1.x + nx * half_w1)), int(round(p1.y + ny * half_w1)))
                q2 = (int(round(p2.x + nx * half_w2)), int(round(p2.y + ny * half_w2)))
                q3 = (int(round(p2.x - nx * half_w2)), int(round(p2.y - ny * half_w2)))
                q4 = (int(round(p1.x - nx * half_w1)), int(round(p1.y - ny * half_w1)))

                quad = np.array([q1, q2, q3, q4], dtype=np.int32)
                cv2.fillConvexPoly(target_buffer, quad, draw_color, lineType=cv_line_type)

            for pt, w in zip(world_pts, widths):
                r = max(1, int(round(w / 2.0)))
                cv2.circle(target_buffer, (int(round(pt.x)), int(round(pt.y))), r, draw_color, -1, lineType=cv_line_type)

            if not allow_direct:
                self._composite_mask(canvas, target_buffer, stroke_bgr, stroke_alpha, bounds)

    def _render_image(self, img_obj: ImageObject, canvas: Canvas | _IsolatedSurface) -> None:
        """Render an ImageObject supporting crop, scaling, affine transforms, and source alpha."""
        render_opacity = self._get_render_opacity(img_obj)
        eff_alpha = img_obj.opacity * render_opacity if self._current_isolating_ancestor is not img_obj else 1.0
        if eff_alpha <= 0.0:
            return

        src_img = img_obj.image
        if img_obj.crop is not None:
            c = img_obj.crop
            cx1 = max(0, int(math.floor(c.left)))
            cy1 = max(0, int(math.floor(c.top)))
            cx2 = min(img_obj.source_width, int(math.ceil(c.right)))
            cy2 = min(img_obj.source_height, int(math.ceil(c.bottom)))
            src_img = src_img[cy1:cy2, cx1:cx2]

        dw = max(1, int(round(img_obj.display_width)))
        dh = max(1, int(round(img_obj.display_height)))
        cv_interp = self._get_cv_interpolation(img_obj.interpolation)

        if src_img.shape[1] != dw or src_img.shape[0] != dh:
            src_img = cv2.resize(src_img, (dw, dh), interpolation=cv_interp)

        # Normalize to BGR float32 and Alpha float32
        if src_img.ndim == 2:
            bgr_f = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR).astype(np.float32)
            alpha_f = np.ones((dh, dw), dtype=np.float32) * float(eff_alpha)
        elif src_img.shape[2] == 1:
            bgr_f = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR).astype(np.float32)
            alpha_f = np.ones((dh, dw), dtype=np.float32) * float(eff_alpha)
        elif src_img.shape[2] == 3:
            bgr_f = src_img.astype(np.float32)
            alpha_f = np.ones((dh, dw), dtype=np.float32) * float(eff_alpha)
        else:
            bgr_f = src_img[:, :, :3].astype(np.float32)
            alpha_f = (src_img[:, :, 3].astype(np.float32) / 255.0) * float(eff_alpha)

        # Premultiply
        pm_bgr = bgr_f * alpha_f[:, :, None]
        pm_bgra = np.dstack([pm_bgr, alpha_f])

        # Warp to world coordinates
        gx, gy = img_obj.position.x, img_obj.position.y
        gw, gh = float(dw), float(dh)
        src_pts = np.float32([[0, 0], [gw, 0], [0, gh]])
        dst_pts = np.float32([
            [img_obj.to_world(Point(gx, gy)).x, img_obj.to_world(Point(gx, gy)).y],
            [img_obj.to_world(Point(gx + gw, gy)).x, img_obj.to_world(Point(gx + gw, gy)).y],
            [img_obj.to_world(Point(gx, gy + gh)).x, img_obj.to_world(Point(gx, gy + gh)).y],
        ])
        M_warp = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(
            pm_bgra, M_warp, (canvas.width, canvas.height),
            flags=cv_interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        bounds = img_obj.get_bounds()
        x1 = max(0, int(math.floor(bounds.left)))
        y1 = max(0, int(math.floor(bounds.top)))
        x2 = min(canvas.width, int(math.ceil(bounds.right)) + 1)
        y2 = min(canvas.height, int(math.ceil(bounds.bottom)) + 1)

        if x2 <= x1 or y2 <= y1:
            return

        sub_warped = warped[y1:y2, x1:x2]
        src_rgb = sub_warped[:, :, :3]
        src_a = sub_warped[:, :, 3:4]

        if getattr(canvas, "is_isolated", False):
            sub_dst = canvas.buffer[y1:y2, x1:x2]
            dst_rgb = sub_dst[:, :, :3]
            dst_a = sub_dst[:, :, 3:4]
            sub_dst[:, :, :3] = src_rgb + dst_rgb * (1.0 - src_a)
            sub_dst[:, :, 3] = (src_a + dst_a * (1.0 - src_a))[:, :, 0]
        else:
            sub_dst = canvas.buffer[y1:y2, x1:x2].astype(np.float32)
            blended = sub_dst * (1.0 - src_a) + src_rgb
            canvas.buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    def _render_text(self, text_obj: Text, canvas: Canvas | _IsolatedSurface) -> None:
        """Render a Text drawable supporting background plate, multiline, and alignment."""
        render_opacity = self._get_render_opacity(text_obj)
        eff_alpha = text_obj.opacity * render_opacity if self._current_isolating_ancestor is not text_obj else 1.0
        if eff_alpha <= 0.0:
            return

        # 1. Background plate
        if text_obj.background_fill is not None and text_obj.background_fill.enabled and text_obj.background_fill.opacity > 0.0:
            gb = text_obj.get_geometry_bounds()
            if text_obj.background_radius > 0.0:
                plate_shape = RoundedRectangle(
                    x=gb.left,
                    y=gb.top,
                    width=gb.width,
                    height=gb.height,
                    corner_radius=text_obj.background_radius,
                    fill=text_obj.background_fill,
                    transform=text_obj.transform,
                    opacity=eff_alpha,
                )
                self._render_rounded_rectangle(plate_shape, canvas)
            else:
                plate_shape = Rectangle(
                    position=Point(gb.left, gb.top),
                    width=gb.width,
                    height=gb.height,
                    fill=text_obj.background_fill,
                    transform=text_obj.transform,
                    opacity=eff_alpha,
                )
                self._render_rectangle(plate_shape, canvas)

        # 2. Text Glyphs
        text_alpha = text_obj.color.a * eff_alpha
        if text_alpha <= 0.0 or not text_obj.text:
            return

        text_bgr = text_obj.color.to_bgr()
        font_face = self._get_cv_font(text_obj.font_family)
        metrics = text_obj.get_line_metrics()
        line_step = text_obj.get_text_bounds_dimensions()[2] * 1.4
        gb = text_obj.get_geometry_bounds()

        has_complex_transform = (
            abs(text_obj.transform.rotation) > 1e-4 or
            not math.isclose(text_obj.transform.scale_x, 1.0, rel_tol=1e-4) or
            not math.isclose(text_obj.transform.scale_y, 1.0, rel_tol=1e-4)
        )

        if not has_complex_transform:
            tx = text_obj.transform.translation_x
            ty = text_obj.transform.translation_y
            for i, (line_str, w_i, h_i, b_i) in enumerate(metrics):
                if not line_str:
                    continue
                if text_obj.alignment == TextAlignment.LEFT:
                    lx = gb.left + text_obj.padding
                elif text_obj.alignment == TextAlignment.CENTER:
                    lx = gb.left + (gb.width - w_i) / 2.0
                else:
                    lx = gb.right - text_obj.padding - w_i
                ly = gb.top + text_obj.padding + metrics[0][2] + i * line_step
                wx = int(round(lx + tx))
                wy = int(round(ly + ty))

                mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                cv2.putText(mask, line_str, (wx, wy), font_face, text_obj.font_scale, 255, text_obj.thickness, cv2.LINE_AA)
                self._composite_mask(canvas, mask, text_bgr, text_alpha, text_obj.get_bounds())
        else:
            local_w = int(math.ceil(gb.width)) + 10
            local_h = int(math.ceil(gb.height)) + 10
            local_mask = np.zeros((local_h, local_w), dtype=np.uint8)

            for i, (line_str, w_i, h_i, b_i) in enumerate(metrics):
                if not line_str:
                    continue
                if text_obj.alignment == TextAlignment.LEFT:
                    lx = text_obj.padding
                elif text_obj.alignment == TextAlignment.CENTER:
                    lx = (gb.width - w_i) / 2.0
                else:
                    lx = gb.width - text_obj.padding - w_i
                ly = text_obj.padding + metrics[0][2] + i * line_step
                cv2.putText(local_mask, line_str, (int(round(lx)), int(round(ly))), font_face, text_obj.font_scale, 255, text_obj.thickness, cv2.LINE_AA)

            src_pts = np.float32([[0, 0], [gb.width, 0], [0, gb.height]])
            dst_pts = np.float32([
                [text_obj.to_world(Point(gb.left, gb.top)).x, text_obj.to_world(Point(gb.left, gb.top)).y],
                [text_obj.to_world(Point(gb.right, gb.top)).x, text_obj.to_world(Point(gb.right, gb.top)).y],
                [text_obj.to_world(Point(gb.left, gb.bottom)).x, text_obj.to_world(Point(gb.left, gb.bottom)).y],
            ])
            M_warp = cv2.getAffineTransform(src_pts, dst_pts)
            world_mask = cv2.warpAffine(local_mask, M_warp, (canvas.width, canvas.height), flags=cv2.INTER_LINEAR)
            self._composite_mask(canvas, world_mask, text_bgr, text_alpha, text_obj.get_bounds())

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

    def _get_cv_font(self, font_family: FontFamily | str) -> int:
        """Map FontFamily enum to OpenCV Hershey font constant."""
        if isinstance(font_family, str):
            try:
                font_family = FontFamily(font_family.strip().lower())
            except ValueError:
                font_family = FontFamily.SIMPLEX
        elif not isinstance(font_family, FontFamily):
            font_family = FontFamily.SIMPLEX

        return _FONT_FAMILY_TO_CV.get(font_family, cv2.FONT_HERSHEY_SIMPLEX)

    def _get_cv_interpolation(self, interp: ImageInterpolation | str) -> int:
        """Map ImageInterpolation enum to OpenCV interpolation constant."""
        if isinstance(interp, str):
            try:
                interp = ImageInterpolation(interp.strip().lower())
            except ValueError:
                interp = ImageInterpolation.LINEAR
        elif not isinstance(interp, ImageInterpolation):
            interp = ImageInterpolation.LINEAR

        mapping = {
            ImageInterpolation.NEAREST: cv2.INTER_NEAREST,
            ImageInterpolation.LINEAR: cv2.INTER_LINEAR,
            ImageInterpolation.CUBIC: cv2.INTER_CUBIC,
            ImageInterpolation.AREA: cv2.INTER_AREA,
            ImageInterpolation.LANCZOS: cv2.INTER_LANCZOS4,
        }
        return mapping.get(interp, cv2.INTER_LINEAR)

    def _composite_mask(
        self,
        canvas: Canvas | _IsolatedSurface,
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

        if getattr(canvas, "is_isolated", False):
            sub_buf = canvas.buffer[y1:y2, x1:x2]
            src_a = coverage
            src_rgb = color_vec * src_a[:, :, None]
            dst_rgb = sub_buf[:, :, :3]
            dst_a = sub_buf[:, :, 3:4]

            sub_buf[:, :, :3] = src_rgb + dst_rgb * (1.0 - src_a[:, :, None])
            sub_buf[:, :, 3] = (src_a[:, :, None] + dst_a * (1.0 - src_a[:, :, None]))[:, :, 0]
        else:
            sub_dst = canvas.buffer[y1:y2, x1:x2].astype(np.float32)
            blended = sub_dst * (1.0 - coverage[:, :, None]) + color_vec * coverage[:, :, None]
            canvas.buffer[y1:y2, x1:x2] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
