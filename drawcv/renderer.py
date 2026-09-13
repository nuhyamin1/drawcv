"""OpenCV raster rendering engine for DrawCV scenes."""

from __future__ import annotations
import math
import cv2
import numpy as np
from typing import Any

from drawcv.canvas import Canvas
from drawcv.core.color import Color
from drawcv.core.paint_sampling import sample_gradient
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
from drawcv.core.alpha import premultiply, unpremultiply, clamp_premultiplied
from drawcv.core.transform import Transform
from drawcv.core.stroking import stroke_mask
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
    def __init__(self, width: int, height: int, *, alpha_output: bool = False):
        self.width = width
        self.height = height
        self.buffer = np.zeros((height, width, 4), dtype=np.float32)
        self.is_isolated = True
        self.alpha_output = alpha_output


class OpenCVRenderer:
    """Raster renderer translating DrawCV retained scene graphs into Canvas images using OpenCV.
    
    The renderer is the sole component responsible for OpenCV drawing primitives,
    color space conversion (RGB -> BGR), and mathematical alpha compositing.
    """

    def __init__(self):
        self._current_isolating_ancestor: Any | None = None

    def render(self, scene: Scene, *, alpha: bool = False) -> Canvas:
        """Render a Scene to legacy BGR, or opt-in straight BGRA with alpha=True.

        BGRA rendering uses premultiplied float surfaces until final quantization;
        scene.background alpha is honored without modifying authored scene state.
        """
        if not isinstance(scene, Scene):
            raise ValidationError(f"Expected Scene, got {type(scene).__name__}")

        if not isinstance(alpha, bool):
            raise ValidationError("alpha must be a boolean")
        paint_pipeline = alpha or self._requires_alpha_pipeline(scene)
        if paint_pipeline:
            canvas = _IsolatedSurface(scene.width, scene.height, alpha_output=True)
            background_alpha = scene.background.a if alpha else 1.0
            canvas.buffer[..., :3] = np.array(scene.background.to_bgr()) * background_alpha
            canvas.buffer[..., 3] = background_alpha
        else:
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

        if paint_pipeline:
            pixels = unpremultiply(canvas.buffer)
            return Canvas(scene.width, scene.height, pixels if alpha else pixels[..., :3], alpha=alpha)
        return canvas

    def render_drawable(self, drawable: Drawable, canvas: Canvas) -> None:
        """Render a single drawable directly onto an existing Canvas."""
        if canvas.has_alpha or self._requires_alpha_pipeline(drawable):
            surface = _IsolatedSurface(canvas.width, canvas.height, alpha_output=True)
            pixels = canvas.buffer if canvas.has_alpha else np.dstack([canvas.buffer, np.full((canvas.height, canvas.width), 255, np.uint8)])
            surface.buffer[:] = premultiply(pixels)
            self._render_single(drawable, surface)
            pixels = unpremultiply(surface.buffer)
            canvas.buffer[:] = pixels if canvas.has_alpha else pixels[..., :3]
        else:
            self._render_single(drawable, canvas)

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

        prog = getattr(drawable, "render_progress", 1.0)
        if prog <= 0.0:
            return
        if prog < 1.0 and getattr(drawable, "supports_progressive_rendering", False):
            sliced = drawable.slice_at_progress(prog)
            sliced.transform = drawable.transform.copy()
            if sliced.transform.pivot is None:
                # Assign directly so an authoritative matrix is not invalidated.
                object.__setattr__(sliced.transform, "pivot", drawable.get_geometry_bounds().center)
            sliced._parent, sliced._layer = drawable._parent, drawable._layer
            sliced.render_progress = 1.0  # Invariant: prevent recursive slicing
            self._render_single(sliced, canvas)
            return

        if self._needs_isolated_compositing(drawable) or (
            getattr(canvas, "alpha_output", False) and drawable.opacity < 1.0
        ):
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
        alpha_output = getattr(destination, "alpha_output", False)
        base_surface = _IsolatedSurface(destination.width, destination.height, alpha_output=alpha_output)

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
                blurred_alpha = cv2.GaussianBlur(shifted_alpha, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv2.BORDER_CONSTANT if alpha_output else cv2.BORDER_DEFAULT)
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
                    base_buffer[y1:y2, x1:x2], (k, k), sigmaX=sig, sigmaY=sig,
                    borderType=cv2.BORDER_CONSTANT if alpha_output else cv2.BORDER_DEFAULT
                )
            else:
                base_buffer[y1:y2, x1:x2] = cv2.blur(
                    base_buffer[y1:y2, x1:x2], (k, k),
                    borderType=cv2.BORDER_CONSTANT if alpha_output else cv2.BORDER_DEFAULT
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
                    if alpha_output:
                        # Fit to the full object bounds, then crop to the canvas;
                        # otherwise moving partly offscreen stretches the mask.
                        bx, by = int(math.floor(pb.left)), int(math.floor(pb.top))
                        bw = int(math.ceil(pb.right)) - bx
                        bh = int(math.ceil(pb.bottom)) - by
                        cov = mask_obj.get_coverage(bw, bh)[py1-by:py2-by, px1-bx:px2-bx]
                    else:
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
                corners = [c if isinstance(entity, Layer) else entity.to_world(c) for c in clip_obj.corners]
                pts = np.array([[int(round(p.x)), int(round(p.y))] for p in corners], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(clip_mask, [pts], 255)
            elif isinstance(clip_obj, ClipPath):
                pts_world = [p if isinstance(entity, Layer) else entity.to_world(p) for p in clip_obj.points]
                pts = np.array([[int(round(p.x)), int(round(p.y))] for p in pts_world], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(clip_mask, [pts], 255)

            sub_clip = clip_mask[y1:y2, x1:x2]
            base_buffer[y1:y2, x1:x2][sub_clip == 0] = 0.0

        # f. Entity Opacity (4-channel scale)
        entity_op = (self._get_render_opacity(entity)
                     if alpha_output and isinstance(entity, Drawable) else float(entity.opacity))
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

    def _render_line(self, line, canvas):
        self._stroke_contours(line, canvas, [([line.to_world(line.start), line.to_world(line.end)], False)])

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
        self._fill_polygon(rect, canvas, pts)

        self._stroke_contours(rect, canvas, [(world_corners, True)])

    def _render_circle(self, circle: Circle, canvas: Canvas | _IsolatedSurface) -> None:
        if circle.radius <= 0:
            return
        sampled = flatten_arc(circle.center, circle.radius, circle.radius, 0, 360,
                              tolerance=self._curve_tolerance(circle))
        world_pts = [circle.to_world(p) for p in sampled]
        pts = np.rint([[p.x, p.y] for p in world_pts]).astype(np.int32)
        bounds = circle.get_bounds()

        # 1. Render Fill if enabled
        self._fill_polygon(circle, canvas, pts)

        self._stroke_contours(circle, canvas, [(world_pts, True)])

    def _render_ellipse(self, ellipse: Ellipse, canvas: Canvas | _IsolatedSurface) -> None:
        if ellipse.radius_x <= 0 or ellipse.radius_y <= 0:
            return

        sampled = flatten_arc(ellipse.center, ellipse.radius_x, ellipse.radius_y, 0.0, 360.0,
                              tolerance=self._curve_tolerance(ellipse))
        world_pts = [ellipse.to_world(p) for p in sampled]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = ellipse.get_bounds()

        # 1. Fill
        self._fill_polygon(ellipse, canvas, pts)

        self._stroke_contours(ellipse, canvas, [(world_pts, True)])

    def _render_polygon(self, polygon: Polygon, canvas: Canvas | _IsolatedSurface) -> None:
        if len(polygon.vertices) < 3:
            return

        world_pts = [polygon.to_world(p) for p in polygon.vertices]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = polygon.get_bounds()

        # 1. Fill
        self._fill_polygon(polygon, canvas, pts)

        self._stroke_contours(polygon, canvas, [(world_pts, True)])

    def _render_polyline(self, polyline, canvas):
        self._stroke_contours(polyline, canvas, [([polyline.to_world(p) for p in polyline.points], polyline.closed)])

    def _render_rounded_rectangle(self, rect: RoundedRectangle, canvas: Canvas | _IsolatedSurface) -> None:
        local_pts = rect.get_contour_points(tolerance=self._curve_tolerance(rect))
        world_pts = [rect.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = rect.get_bounds()

        # 1. Fill
        self._fill_polygon(rect, canvas, pts)

        self._stroke_contours(rect, canvas, [(world_pts, True)])

    def _render_arc(self, arc: Arc, canvas: Canvas | _IsolatedSurface) -> None:
        local_pts = arc.get_contour_points(tolerance=self._curve_tolerance(arc))
        world_pts = [arc.to_world(p) for p in local_pts]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in world_pts], dtype=np.int32).reshape((-1, 1, 2))
        bounds = arc.get_bounds()
        is_closed = (arc.closure != ArcClosure.OPEN)

        # 1. Fill (only for CHORD or PIE)
        if is_closed:
            self._fill_polygon(arc, canvas, pts)

        self._stroke_contours(arc, canvas, [(world_pts, is_closed)])

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

        self._stroke_contours(arrow, canvas, [([w_start, w_end], False)])
        marker_stroke = stroke.copy()
        marker_stroke.dash_array = ()

        # 2. Arrowhead
        fill = arrow.fill
        if arrow.head_style in (ArrowHeadStyle.TRIANGLE, ArrowHeadStyle.DIAMOND):
            pts_head = np.array([[int(round(p.x)), int(round(p.y))] for p in head_pts], dtype=np.int32).reshape((-1, 1, 2))
            self._fill_polygon(arrow, canvas, pts_head)
            self._stroke_contours(arrow, canvas, [(head_pts, True)], style=marker_stroke)

        elif arrow.head_style == ArrowHeadStyle.OPEN:
            self._stroke_contours(arrow, canvas, [(head_pts, False)], style=marker_stroke)

        elif arrow.head_style == ArrowHeadStyle.CIRCLE:
            c = head_pts[0]
            r = max(1, int(round(head_pts[1].x)))
            c_int = (int(round(c.x)), int(round(c.y)))
            if fill is not None and fill.enabled and fill.opacity > 0:
                if isinstance(fill.paint, Color) and self._is_direct_draw(canvas, fill.color.a * fill.opacity * self._get_render_opacity(arrow)):
                    cv2.circle(canvas.buffer, c_int, r, fill.color.to_bgr(), -1, cv2.LINE_AA)
                else:
                    mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                    cv2.circle(mask, c_int, r, 255, -1, cv2.LINE_AA)
                    self._fill_mask(arrow, canvas, mask)
            ring = flatten_arc(c, r, r, 0, 360, tolerance=0.25)
            self._stroke_contours(arrow, canvas, [(ring, True)], style=marker_stroke)

    def _render_bezier(self, bezier, canvas):
        self._stroke_contours(bezier, canvas, [(bezier.flatten_world(tolerance=0.25), False)])

    def _render_path(self, path: Path, canvas: Canvas | _IsolatedSurface) -> None:
        contours = path.flatten_world(tolerance=0.5, include_closed=True)
        world_contours = [points for points, _ in contours]
        if not world_contours:
            return

        bounds = path.get_bounds()

        # 1. Fill using topological fill rule evaluator
        fill = path.fill
        if fill is not None and fill.enabled and fill.opacity > 0:
            mask = evaluate_fill_rule_mask(world_contours, path.fill_rule, canvas.width, canvas.height, supersample=2)
            self._fill_mask(path, canvas, mask)

        self._stroke_contours(path, canvas, contours)

    def _render_freehand(self, freehand, canvas):
        points = freehand.get_processed_points()
        if freehand.stroke is None:
            return
        world = self._world_points(freehand, points)
        widths = freehand.get_point_widths(points) if freehand.variable_width else None
        self._stroke_contours(freehand, canvas, [(world, False)], widths=widths)

    @staticmethod
    def _world_points(drawable, points):
        """Resolve invariant bounds/matrices once, without caching across calls.

        Keep each ancestor's matrix-vector operation and Point conversion in the
        same order as to_world. Combining matrices or vectorizing the dot products
        can change rounding at raster boundaries. Custom to_world methods retain
        their existing dispatch behavior.
        """
        if not points:
            return []
        matrices = []
        current = drawable
        while current is not None:
            if (getattr(current.to_world, "__func__", None) is not Drawable.to_world or
                    getattr(current.transform.transform_point, "__func__", None) is not Transform.transform_point):
                return [drawable.to_world(Point(p.x, p.y)) for p in points]
            pivot = current.transform.pivot
            if pivot is None:
                pivot = current.get_geometry_bounds().center
            matrices.append(current.transform.get_matrix(default_pivot=pivot))
            current = current._parent
        result = []
        for p in points:
            point = Point(p.x, p.y)
            for matrix in matrices:
                vector = matrix @ np.array([point.x, point.y, 1.0], dtype=np.float64)
                point = Point(vector[0], vector[1])
            result.append(point)
        return result

    def _render_image(self, img_obj: ImageObject, canvas: Canvas | _IsolatedSurface) -> None:
        """Render an ImageObject supporting crop, scaling, affine transforms, and source alpha."""
        render_opacity = self._get_render_opacity(img_obj)
        eff_alpha = (render_opacity if getattr(canvas, "alpha_output", False) else
                     (img_obj.opacity * render_opacity if self._current_isolating_ancestor is not img_obj else 1.0))
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

        alpha_output = getattr(canvas, "alpha_output", False)
        if alpha_output:
            # Discard hidden RGB before ANY filtering, including display resize.
            if src_img.ndim == 2 or src_img.shape[2] == 1:
                bgr = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                source = np.dstack([bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)])
            elif src_img.shape[2] == 3:
                source = np.dstack([src_img, np.full(src_img.shape[:2], 255, dtype=np.uint8)])
            else:
                source = src_img
            pm_bgra = premultiply(source) * eff_alpha
            if source.shape[1] != dw or source.shape[0] != dh:
                pm_bgra = cv2.resize(pm_bgra, (dw, dh), interpolation=cv_interp)
            pm_bgra = clamp_premultiplied(pm_bgra)
        else:
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

        if alpha_output:
            warped = clamp_premultiplied(warped)

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
        alpha_output = getattr(canvas, "alpha_output", False)
        eff_alpha = (render_opacity if alpha_output else
                     (text_obj.opacity * render_opacity if self._current_isolating_ancestor is not text_obj else 1.0))
        if eff_alpha <= 0.0:
            return

        # 1. Background plate
        if text_obj.background_fill is not None and text_obj.background_fill.enabled and text_obj.background_fill.opacity > 0.0:
            gb = text_obj.measure().paragraph_bounds if text_obj.fonts is not None else text_obj.get_geometry_bounds()
            if text_obj.background_radius > 0.0:
                plate_shape = RoundedRectangle(
                    x=gb.left,
                    y=gb.top,
                    width=gb.width,
                    height=gb.height,
                    corner_radius=text_obj.background_radius,
                    fill=text_obj.background_fill,
                    transform=Transform.from_matrix(text_obj.world_matrix) if alpha_output else text_obj.transform,
                    opacity=eff_alpha,
                )
                self._render_rounded_rectangle(plate_shape, canvas)
            else:
                plate_shape = Rectangle(
                    position=Point(gb.left, gb.top),
                    width=gb.width,
                    height=gb.height,
                    fill=text_obj.background_fill,
                    transform=Transform.from_matrix(text_obj.world_matrix) if alpha_output else text_obj.transform,
                    opacity=eff_alpha,
                )
                self._render_rectangle(plate_shape, canvas)

        # 2. Text Glyphs
        text_alpha = text_obj.color.a * eff_alpha
        if text_alpha <= 0.0 or not text_obj.text:
            return

        text_bgr = text_obj.color.to_bgr()
        if text_obj.fonts is not None:
            layout = text_obj._font_layout()
            if layout.ink is None:
                return
            metrics = text_obj.measure()
            origin = np.array([[1., 0., metrics.layout_bounds.x + layout.left],
                               [0., 1., metrics.layout_bounds.y + layout.top], [0., 0., 1.]])
            matrix = text_obj.world_matrix @ origin
            mask = cv2.warpAffine(layout.mask, matrix[:2], (canvas.width, canvas.height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self._composite_mask(canvas, mask, text_bgr, text_alpha)
            return
        font_face = self._get_cv_font(text_obj.font_family)
        metrics = text_obj.get_line_metrics()
        line_step = text_obj.get_text_bounds_dimensions()[2] * 1.4
        gb = text_obj.get_geometry_bounds()

        has_complex_transform = (
            abs(text_obj.transform.rotation) > 1e-4 or
            not math.isclose(text_obj.transform.scale_x, 1.0, rel_tol=1e-4) or
            not math.isclose(text_obj.transform.scale_y, 1.0, rel_tol=1e-4)
        )

        if alpha_output:
            has_complex_transform = not np.allclose(text_obj.world_matrix[:2, :2], np.eye(2), atol=1e-8, rtol=0)

        if not has_complex_transform:
            tx = text_obj.transform.translation_x
            ty = text_obj.transform.translation_y
            if alpha_output:
                tx, ty = text_obj.world_matrix[:2, 2]
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

    def _requires_alpha_pipeline(self, entity):
        # Font text, like gradient paint, uses the corrected pipeline in BGR too.
        if isinstance(entity, Text) and entity.fonts is not None:
            return True
        for name in ("fill", "background_fill"):
            fill = getattr(entity, name, None)
            if fill is not None and not isinstance(fill.paint, Color):
                return True
        for name in ("layers", "children", "objects"):
            children = getattr(entity, name, None)
            if children is not None:
                return any(self._requires_alpha_pipeline(child) for child in children)
        return False

    def _fill_polygon(self, drawable, canvas, points):
        fill = drawable.fill
        if fill is None or not fill.enabled or fill.opacity <= 0:
            return
        if isinstance(fill.paint, Color):
            alpha = fill.color.a * fill.opacity * self._get_render_opacity(drawable)
            if alpha <= 0:
                return
            if self._is_direct_draw(canvas, alpha):
                cv2.fillPoly(canvas.buffer, [points], fill.color.to_bgr(), cv2.LINE_AA)
                return
        mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)
        self._fill_mask(drawable, canvas, mask)

    def _fill_mask(self, drawable, canvas, mask):
        fill = drawable.fill
        opacity = fill.opacity * self._get_render_opacity(drawable)
        if isinstance(fill.paint, Color):
            self._composite_mask(canvas, mask, fill.color.to_bgr(), opacity * fill.color.a, drawable.get_bounds())
            return
        x, y, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            return
        region = np.s_[y:y+height, x:x+width]
        source = sample_gradient(fill.paint, drawable.world_matrix, width, height, origin=(x, y))
        source *= (mask[region].astype(np.float32) / 255 * opacity)[..., None]
        a = source[..., 3:4]
        destination = canvas.buffer[region]
        if getattr(canvas, "is_isolated", False):
            destination[..., :3] = source[..., :3] + destination[..., :3]*(1-a)
            destination[..., 3:4] = a + destination[..., 3:4]*(1-a)
        else:
            destination[:] = np.rint(source[..., :3] + destination*(1-a)).clip(0, 255).astype(np.uint8)

    def _curve_tolerance(self, drawable):
        scale = float(np.linalg.norm(drawable.world_matrix[:2, :2], ord=2))
        return 0.25 / max(scale, 1e-12)

    def _stroke_contours(self, drawable, canvas, contours, widths=None, style=None):
        stroke = style if style is not None else getattr(drawable, "stroke", None)
        if stroke is None:
            return
        alpha = stroke.color.a * stroke.opacity * self._get_render_opacity(drawable)
        if alpha <= 0:
            return
        samples = [([(p.x, p.y, widths[i] if widths is not None else stroke.width)
                     for i, p in enumerate(points)], closed) for points, closed in contours]
        mask = stroke_mask(canvas.width, canvas.height, samples, stroke,
                           self._get_cv_line_type(stroke.line_type))
        self._composite_mask(canvas, mask, stroke.color.to_bgr(), alpha)

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
            x1, y1, width, height = cv2.boundingRect(mask)
            x2, y2 = x1 + width, y1 + height

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
