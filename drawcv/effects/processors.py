"""Raster effect processors operating on float32 premultiplied BGRA buffers."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable, Type
import cv2
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.alpha import clamp_premultiplied
from drawcv.core.exceptions import ValidationError
from drawcv.core.enums import (
    BlurType,
    DisplacementChannel,
    EdgeDetectionMethod,
    ImageInterpolation,
)
from drawcv.core.paint_sampling import _get_cv_interpolation, sample_paint
from drawcv.effects.blur import BlurEffect
from drawcv.effects.color import (
    BrightnessContrastEffect,
    ColorMatrixEffect,
    GrayscaleEffect,
    HueShiftEffect,
    SaturationEffect,
    SepiaEffect,
)
from drawcv.effects.color_utils import (
    apply_spatial_straight_filter,
    apply_straight_color_op,
)
from drawcv.effects.convolution import ConvolutionEffect
from drawcv.effects.displacement import DisplacementMapEffect
from drawcv.effects.edge import EdgeDetectionEffect
from drawcv.effects.effect import Effect
from drawcv.effects.emboss import EmbossEffect
from drawcv.effects.gaussian import gaussian_kernel_size_for_radius
from drawcv.effects.glow import GlowEffect
from drawcv.effects.noise import NoiseEffect, hash_noise_coords
from drawcv.effects.shadow import ShadowEffect
from drawcv.effects.sharpen import SharpenEffect



@dataclass(frozen=True)
class RasterEffectStageContext:
    """Stage-specific execution context tracking canvas geometry, effect ROIs, and sampling support."""
    canvas_width: int
    canvas_height: int
    input_bounds: BoundingBox
    output_bounds: BoundingBox
    alpha_output: bool
    margin: int = 3
    sampling_padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    world_matrix: np.ndarray | None = None

    def input_roi(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) canvas coordinates bounding the input stage with guard margin."""
        b = self.input_bounds
        x1 = max(0, int(math.floor(b.left)) - self.margin)
        y1 = max(0, int(math.floor(b.top)) - self.margin)
        x2 = min(self.canvas_width, int(math.ceil(b.right)) + self.margin + 1)
        y2 = min(self.canvas_height, int(math.ceil(b.bottom)) + self.margin + 1)
        return (x1, y1, x2, y2)

    def output_roi(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) canvas coordinates bounding the output stage with guard margin."""
        b = self.output_bounds
        x1 = max(0, int(math.floor(b.left)) - self.margin)
        y1 = max(0, int(math.floor(b.top)) - self.margin)
        x2 = min(self.canvas_width, int(math.ceil(b.right)) + self.margin + 1)
        y2 = min(self.canvas_height, int(math.ceil(b.bottom)) + self.margin + 1)
        return (x1, y1, x2, y2)

    def requested_source_roi(self) -> tuple[int, int, int, int]:
        """Return requested/unclamped canvas coordinates (rx1, ry1, rx2, ry2) including sampling padding."""
        ox1, oy1, ox2, oy2 = self.output_roi()
        pl, pr, pt, pb = self.sampling_padding
        rx1 = ox1 - int(math.ceil(pl))
        ry1 = oy1 - int(math.ceil(pt))
        rx2 = ox2 + int(math.ceil(pr))
        ry2 = oy2 + int(math.ceil(pb))
        return (rx1, ry1, rx2, ry2)

    def clamped_source_roi(self) -> tuple[int, int, int, int]:
        """Return canvas-clamped coordinates (sx1, sy1, sx2, sy2) for slicing the actual canvas buffer."""
        rx1, ry1, rx2, ry2 = self.requested_source_roi()
        sx1 = max(0, rx1)
        sy1 = max(0, ry1)
        sx2 = min(self.canvas_width, rx2)
        sy2 = min(self.canvas_height, ry2)
        return (sx1, sy1, sx2, sy2)

    def source_roi(self) -> tuple[int, int, int, int]:
        """Backward-compatible convenience alias for clamped_source_roi."""
        return self.clamped_source_roi()

    def extract_padded_source(self, buffer: np.ndarray) -> np.ndarray:
        """Return canvas-clamped source slice without materializing out-of-canvas transparent borders.

        Sampling outside the canvas is mathematically zero and handled via BORDER_CONSTANT.
        """
        sx1, sy1, sx2, sy2 = self.clamped_source_roi()
        if sx2 <= sx1 or sy2 <= sy1:
            return np.zeros((0, 0, buffer.shape[2]), dtype=buffer.dtype)
        return buffer[sy1:sy2, sx1:sx2].copy()



_RASTER_PROCESSORS: dict[type[Effect], Callable[[Effect, np.ndarray, RasterEffectStageContext], np.ndarray]] = {}


def register_raster_processor(
    effect_cls: type[Effect],
    processor_fn: Callable[[Any, np.ndarray, RasterEffectStageContext], np.ndarray],
) -> None:
    """Register a raster processor callable for a concrete Effect class."""
    _RASTER_PROCESSORS[effect_cls] = processor_fn


def get_raster_processor(effect_cls: type[Effect]) -> Callable[[Any, np.ndarray, RasterEffectStageContext], np.ndarray]:
    """Retrieve the registered raster processor for an Effect class."""
    _ensure_builtin_processors()
    if effect_cls in _RASTER_PROCESSORS:
        return _RASTER_PROCESSORS[effect_cls]
    # Check inheritance hierarchy
    for registered_cls, processor in _RASTER_PROCESSORS.items():
        if issubclass(effect_cls, registered_cls):
            return processor
    raise NotImplementedError(f"No raster processor registered for effect class: {effect_cls.__name__}")


def process_blur(effect: BlurEffect, buffer: np.ndarray, ctx: RasterEffectStageContext) -> np.ndarray:
    """Execute spatial blur within stage output ROI."""
    ox1, oy1, ox2, oy2 = ctx.output_roi()
    if ox2 <= ox1 or oy2 <= oy1:
        return buffer

    k = effect.kernel_size
    if effect.blur_type == BlurType.GAUSSIAN:
        sig = effect.sigma
        buffer[oy1:oy2, ox1:ox2] = cv2.GaussianBlur(
            buffer[oy1:oy2, ox1:ox2],
            (k, k),
            sigmaX=sig,
            sigmaY=sig,
            borderType=cv2.BORDER_CONSTANT,
        )
    else:
        buffer[oy1:oy2, ox1:ox2] = cv2.blur(
            buffer[oy1:oy2, ox1:ox2],
            (k, k),
            borderType=cv2.BORDER_CONSTANT,
        )
    return buffer


def process_shadow(effect: ShadowEffect, buffer: np.ndarray, ctx: RasterEffectStageContext) -> np.ndarray:
    """Execute drop shadow within stage output ROI."""
    if effect.opacity <= 0.0 or effect.color.a <= 0.0:
        return buffer

    ix1, iy1, ix2, iy2 = ctx.input_roi()
    ox1, oy1, ox2, oy2 = ctx.output_roi()
    if ix2 <= ix1 or iy2 <= iy1 or ox2 <= ox1 or oy2 <= oy1:
        return buffer

    sub_in = buffer[iy1:iy2, ix1:ix2]
    sub_out = buffer[oy1:oy2, ox1:ox2]

    # Extract and scale alpha from input ROI
    sh_alpha_in = sub_in[..., 3] * float(effect.color.a * effect.opacity)

    # Calculate translation from input ROI space to output ROI space
    dx = float(ix1 - ox1) + float(effect.offset_x)
    dy = float(iy1 - oy1) + float(effect.offset_y)
    M_trans = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])

    out_w = ox2 - ox1
    out_h = oy2 - oy1
    shifted_alpha = cv2.warpAffine(
        sh_alpha_in,
        M_trans,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    if effect.blur_radius > 0.0:
        ksize = gaussian_kernel_size_for_radius(effect.blur_radius)
        blurred_alpha = cv2.GaussianBlur(
            shifted_alpha,
            (ksize, ksize),
            sigmaX=float(effect.blur_radius),
            sigmaY=float(effect.blur_radius),
            borderType=cv2.BORDER_CONSTANT,
        )
    else:
        blurred_alpha = shifted_alpha

    sh_b, sh_g, sh_r = effect.color.to_bgr()
    sh_color_vec = np.array([sh_b, sh_g, sh_r], dtype=np.float32)
    sh_rgb_pm = sh_color_vec * blurred_alpha[..., None]

    base_rgb = sub_out[..., :3]
    base_a = sub_out[..., 3:4]
    sh_a = blurred_alpha[..., None]

    # Porter-Duff: base OVER shadow
    merged_rgb = base_rgb + sh_rgb_pm * (1.0 - base_a)
    merged_a = base_a + sh_a * (1.0 - base_a)

    sub_out[..., :3] = merged_rgb
    sub_out[..., 3] = merged_a[..., 0]
    buffer[oy1:oy2, ox1:ox2] = sub_out
    return buffer


def process_glow(effect: GlowEffect, buffer: np.ndarray, ctx: RasterEffectStageContext) -> np.ndarray:
    """Execute outer glow within stage output ROI."""
    if effect.opacity <= 0.0 or effect.color.a <= 0.0 or effect.blur_radius <= 0.0:
        return buffer

    ix1, iy1, ix2, iy2 = ctx.input_roi()
    ox1, oy1, ox2, oy2 = ctx.output_roi()
    if ix2 <= ix1 or iy2 <= iy1 or ox2 <= ox1 or oy2 <= oy1:
        return buffer

    sub_in = buffer[iy1:iy2, ix1:ix2]
    sub_out = buffer[oy1:oy2, ox1:ox2]

    # Extract input alpha
    alpha_in = sub_in[..., 3]

    out_w = ox2 - ox1
    out_h = oy2 - oy1
    dx = float(ix1 - ox1)
    dy = float(iy1 - oy1)
    M_trans = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])

    # Map input alpha into stage output coordinate space
    aligned_alpha = cv2.warpAffine(
        alpha_in,
        M_trans,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    ksize = gaussian_kernel_size_for_radius(effect.blur_radius)
    blurred_alpha = cv2.GaussianBlur(
        aligned_alpha,
        (ksize, ksize),
        sigmaX=float(effect.blur_radius),
        sigmaY=float(effect.blur_radius),
        borderType=cv2.BORDER_CONSTANT,
    )

    # True Outer Glow: subtract input alpha so the glow radiates outward
    # and does not invade or alter the translucent interior
    outer_alpha = np.maximum(0.0, blurred_alpha - aligned_alpha) * float(effect.color.a * effect.opacity)
    np.clip(outer_alpha, 0.0, 1.0, out=outer_alpha)

    glow_b, glow_g, glow_r = effect.color.to_bgr()
    glow_color_vec = np.array([glow_b, glow_g, glow_r], dtype=np.float32)
    glow_rgb_pm = glow_color_vec * outer_alpha[..., None]
    glow_a = outer_alpha[..., None]

    base_rgb = sub_out[..., :3]
    base_a = sub_out[..., 3:4]

    # Porter-Duff: base OVER glow
    merged_rgb = base_rgb + glow_rgb_pm * (1.0 - base_a)
    merged_a = base_a + glow_a * (1.0 - base_a)

    sub_out[..., :3] = np.clip(merged_rgb, 0.0, 255.0)
    sub_out[..., 3] = np.clip(merged_a[..., 0], 0.0, 1.0)
    buffer[oy1:oy2, ox1:ox2] = sub_out
    return buffer


def process_brightness_contrast(
    effect: BrightnessContrastEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply brightness and contrast adjustments to straight color."""
    b = float(effect.brightness)
    c = float(effect.contrast)
    if b == 0.0 and c == 1.0:
        return buffer

    def _transform(bgr: np.ndarray) -> np.ndarray:
        return (bgr - 0.5) * c + 0.5 + b

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform)


def process_saturation(
    effect: SaturationEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply saturation adjustment using Rec.709 luma weights."""
    f = float(effect.factor)
    if f == 1.0:
        return buffer

    def _transform(bgr: np.ndarray) -> np.ndarray:
        luma = 0.0722 * bgr[..., 0:1] + 0.7152 * bgr[..., 1:2] + 0.2126 * bgr[..., 2:3]
        return luma + f * (bgr - luma)

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform)


def process_grayscale(
    effect: GrayscaleEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Convert straight color to grayscale using Rec.709 luma weighting."""
    intensity = float(effect.intensity)
    if intensity <= 0.0:
        return buffer

    def _transform(bgr: np.ndarray) -> np.ndarray:
        luma = 0.0722 * bgr[..., 0:1] + 0.7152 * bgr[..., 1:2] + 0.2126 * bgr[..., 2:3]
        gray = np.broadcast_to(luma, bgr.shape)
        return bgr * (1.0 - intensity) + gray * intensity

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform)


def process_sepia(
    effect: SepiaEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply sepia tone filtering."""
    intensity = float(effect.intensity)
    if intensity <= 0.0:
        return buffer

    def _transform(bgr: np.ndarray) -> np.ndarray:
        b = bgr[..., 0]
        g = bgr[..., 1]
        r = bgr[..., 2]
        sepia_b = 0.131 * b + 0.534 * g + 0.272 * r
        sepia_g = 0.168 * b + 0.686 * g + 0.349 * r
        sepia_r = 0.189 * b + 0.769 * g + 0.393 * r
        sepia = np.stack([sepia_b, sepia_g, sepia_r], axis=-1)
        return bgr * (1.0 - intensity) + sepia * intensity

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform)


def process_hue_shift(
    effect: HueShiftEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply canonical W3C hueRotate matrix transformation."""
    angle = float(effect.angle) % 360.0
    if angle == 0.0:
        return buffer

    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)

    # 3x3 RGB hueRotate matrix from SVG Filter Effects spec
    M = np.array([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ], dtype=np.float32)

    def _transform(bgr: np.ndarray) -> np.ndarray:
        rgb = bgr[..., ::-1]
        rgb_trans = rgb @ M.T
        return rgb_trans[..., ::-1]

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform)


def process_color_matrix(
    effect: ColorMatrixEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply 4x5 ColorMatrix transformation to straight RGB."""
    mat = np.array(effect.matrix, dtype=np.float32)

    def _transform(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        b = bgr[..., 0]
        g = bgr[..., 1]
        r = bgr[..., 2]
        a = alpha[..., 0]
        ones = np.ones_like(a)

        vec = np.stack([r, g, b, a, ones], axis=-1)

        r_new = vec @ mat[0]
        g_new = vec @ mat[1]
        b_new = vec @ mat[2]
        return np.stack([b_new, g_new, r_new], axis=-1)

    return apply_straight_color_op(buffer, ctx.output_roi(), _transform, include_alpha=True)


_BUILTINS_REGISTERED = False


def _ensure_builtin_processors() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    register_raster_processor(BlurEffect, process_blur)
    register_raster_processor(ShadowEffect, process_shadow)
    register_raster_processor(GlowEffect, process_glow)
    register_raster_processor(BrightnessContrastEffect, process_brightness_contrast)
    register_raster_processor(SaturationEffect, process_saturation)
    register_raster_processor(GrayscaleEffect, process_grayscale)
    register_raster_processor(SepiaEffect, process_sepia)
    register_raster_processor(HueShiftEffect, process_hue_shift)
    register_raster_processor(ColorMatrixEffect, process_color_matrix)
    register_raster_processor(ConvolutionEffect, process_convolution)
    register_raster_processor(SharpenEffect, process_sharpen)
    register_raster_processor(EmbossEffect, process_emboss)
    register_raster_processor(EdgeDetectionEffect, process_edge_detection)
    register_raster_processor(NoiseEffect, process_noise)
    register_raster_processor(DisplacementMapEffect, process_displacement_map)


def process_convolution(
    effect: ConvolutionEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply generalized discrete 2D spatial convolution across stage output ROI."""
    kernel_arr = np.array(effect.kernel, dtype=np.float32)
    factor = float(effect.factor)
    bias = float(effect.bias)

    def _convolve(src_straight: np.ndarray, target_box: tuple[int, int, int, int]) -> np.ndarray:
        tx1, ty1, tx2, ty2 = target_box
        filtered = cv2.filter2D(src_straight, -1, kernel_arr, borderType=cv2.BORDER_CONSTANT)
        target_filtered = filtered[ty1:ty2, tx1:tx2] * factor + bias
        return target_filtered

    return apply_spatial_straight_filter(buffer, ctx, _convolve)


def process_sharpen(
    effect: SharpenEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply unsharp masking spatial sharpen across stage output ROI."""
    if effect.amount <= 0.0 or effect.radius <= 0.0:
        return buffer

    ksize = gaussian_kernel_size_for_radius(effect.radius)
    sig = float(effect.radius)
    amt = float(effect.amount)

    def _sharpen(src_straight: np.ndarray, target_box: tuple[int, int, int, int]) -> np.ndarray:
        tx1, ty1, tx2, ty2 = target_box
        blurred = cv2.GaussianBlur(
            src_straight,
            (ksize, ksize),
            sigmaX=sig,
            sigmaY=sig,
            borderType=cv2.BORDER_CONSTANT,
        )
        orig = src_straight[ty1:ty2, tx1:tx2]
        bl = blurred[ty1:ty2, tx1:tx2]
        return orig + amt * (orig - bl)

    return apply_spatial_straight_filter(buffer, ctx, _sharpen)


def process_emboss(
    effect: EmbossEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply directional luminance gradient relief across stage output ROI."""
    rad = math.radians(effect.angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    strength = float(effect.strength)
    bias = float(effect.bias)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

    def _emboss(src_straight: np.ndarray, target_box: tuple[int, int, int, int]) -> np.ndarray:
        tx1, ty1, tx2, ty2 = target_box
        luma = 0.0722 * src_straight[..., 0] + 0.7152 * src_straight[..., 1] + 0.2126 * src_straight[..., 2]
        gx = cv2.filter2D(luma, -1, kx, borderType=cv2.BORDER_CONSTANT)
        gy = cv2.filter2D(luma, -1, ky, borderType=cv2.BORDER_CONSTANT)
        gx_t = gx[ty1:ty2, tx1:tx2]
        gy_t = gy[ty1:ty2, tx1:tx2]
        directional = cos_a * gx_t + sin_a * gy_t
        embossed = bias + strength * directional
        embossed = np.clip(embossed, 0.0, 1.0)
        return np.repeat(embossed[..., None], 3, axis=-1)

    return apply_spatial_straight_filter(buffer, ctx, _emboss)


def process_edge_detection(
    effect: EdgeDetectionEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply spatial edge detection filtering across stage output ROI."""
    method = effect.method
    strength = float(effect.strength)
    invert = effect.invert

    if method == EdgeDetectionMethod.SOBEL:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0
    else:
        kl = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32) / 8.0

    def _edge(src_straight: np.ndarray, target_box: tuple[int, int, int, int]) -> np.ndarray:
        tx1, ty1, tx2, ty2 = target_box
        luma = 0.0722 * src_straight[..., 0] + 0.7152 * src_straight[..., 1] + 0.2126 * src_straight[..., 2]
        if method == EdgeDetectionMethod.SOBEL:
            gx = cv2.filter2D(luma, -1, kx, borderType=cv2.BORDER_CONSTANT)
            gy = cv2.filter2D(luma, -1, ky, borderType=cv2.BORDER_CONSTANT)
            gx_t = gx[ty1:ty2, tx1:tx2]
            gy_t = gy[ty1:ty2, tx1:tx2]
            response = strength * np.sqrt(gx_t * gx_t + gy_t * gy_t)
        else:
            lap = cv2.filter2D(luma, -1, kl, borderType=cv2.BORDER_CONSTANT)
            lap_t = lap[ty1:ty2, tx1:tx2]
            response = strength * np.abs(lap_t)

        if invert:
            response = 1.0 - response
        response = np.clip(response, 0.0, 1.0)
        return np.repeat(response[..., None], 3, axis=-1)

    return apply_spatial_straight_filter(buffer, ctx, _edge)


def process_noise(
    effect: NoiseEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply spatially deterministic film grain / noise across stage output ROI."""
    if effect.amount <= 0.0:
        return buffer

    ox1, oy1, ox2, oy2 = ctx.output_roi()
    if ox2 <= ox1 or oy2 <= oy1:
        return buffer

    sub = buffer[oy1:oy2, ox1:ox2]
    alpha = sub[..., 3:4]
    valid_mask = (alpha > 1e-6)[..., 0]
    if not np.any(valid_mask):
        return buffer

    ys, xs = np.indices((oy2 - oy1, ox2 - ox1), dtype=np.int64)
    xs += ox1
    ys += oy1

    amt = float(effect.amount)
    seed = int(effect.seed)

    straight_bgr = np.zeros_like(sub[..., :3])
    alpha_valid = alpha[valid_mask]
    straight_bgr[valid_mask] = sub[valid_mask, :3] / (alpha_valid * 255.0)
    np.clip(straight_bgr, 0.0, 1.0, out=straight_bgr)

    if effect.monochrome:
        noise = hash_noise_coords(xs, ys, seed, channel=0)[..., None]
        straight_bgr[valid_mask] += amt * noise[valid_mask]
    else:
        nb = hash_noise_coords(xs, ys, seed, channel=0)
        ng = hash_noise_coords(xs, ys, seed, channel=1)
        nr = hash_noise_coords(xs, ys, seed, channel=2)
        noise_color = np.stack([nb, ng, nr], axis=-1)
        straight_bgr[valid_mask] += amt * noise_color[valid_mask]

    np.clip(straight_bgr, 0.0, 1.0, out=straight_bgr)
    sub[valid_mask, :3] = straight_bgr[valid_mask] * (alpha_valid * 255.0)
    sub[~valid_mask, :] = 0.0
    buffer[oy1:oy2, ox1:ox2] = sub
    return buffer


def process_displacement_map(
    effect: DisplacementMapEffect,
    buffer: np.ndarray,
    ctx: RasterEffectStageContext,
) -> np.ndarray:
    """Apply geometric displacement mapping across stage output ROI."""
    if float(effect.map.opacity) != 1.0:
        raise ValidationError(
            f"Displacement map ImagePaint opacity must be 1.0 to preserve displacement data, got {effect.map.opacity}"
        )

    ox1, oy1, ox2, oy2 = ctx.output_roi()
    w = ox2 - ox1
    h = oy2 - oy1
    if w <= 0 or h <= 0:
        return buffer

    world_mat = ctx.world_matrix if ctx.world_matrix is not None else np.eye(3, dtype=np.float32)
    map_samples = sample_paint(effect.map, world_mat, w, h, origin=(ox1, oy1))

    map_alpha = map_samples[..., 3]
    valid_map = map_alpha > 1e-6

    def _extract_channel(channel: DisplacementChannel) -> np.ndarray:
        if channel == DisplacementChannel.ALPHA:
            return np.clip(map_alpha, 0.0, 1.0)

        straight = np.full((h, w), 0.5, dtype=np.float32)
        if not np.any(valid_map):
            return straight

        denom = map_alpha[valid_map] * 255.0
        if channel == DisplacementChannel.BLUE:
            straight[valid_map] = map_samples[valid_map, 0] / denom
        elif channel == DisplacementChannel.GREEN:
            straight[valid_map] = map_samples[valid_map, 1] / denom
        elif channel == DisplacementChannel.RED:
            straight[valid_map] = map_samples[valid_map, 2] / denom
        elif channel == DisplacementChannel.LUMINANCE:
            b = map_samples[valid_map, 0] / denom
            g = map_samples[valid_map, 1] / denom
            r = map_samples[valid_map, 2] / denom
            straight[valid_map] = 0.0722 * b + 0.7152 * g + 0.2126 * r

        return np.clip(straight, 0.0, 1.0)

    cx = _extract_channel(effect.x_channel)
    cy = _extract_channel(effect.y_channel)

    dx = (2.0 * cx - 1.0) * float(effect.scale_x)
    dy = (2.0 * cy - 1.0) * float(effect.scale_y)

    ys, xs = np.indices((h, w), dtype=np.float32)
    x_dst = xs + float(ox1)
    y_dst = ys + float(oy1)

    source_global_x = x_dst - dx
    source_global_y = y_dst - dy

    map_x = source_global_x.astype(np.float32)
    map_y = source_global_y.astype(np.float32)

    cv_interp = _get_cv_interpolation(effect.interpolation)

    remapped = cv2.remap(
        buffer,
        map_x,
        map_y,
        interpolation=cv_interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )

    remapped = clamp_premultiplied(remapped)
    buffer[oy1:oy2, ox1:ox2] = remapped
    return buffer
