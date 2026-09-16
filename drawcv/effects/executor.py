"""Ordered effect execution coordinator."""

from __future__ import annotations
from typing import Sequence
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.effects.effect import Effect
from drawcv.effects.processors import RasterEffectStageContext, get_raster_processor


def execute_effects_pipeline(
    buffer: np.ndarray,
    effects: Sequence[Effect],
    base_bounds: BoundingBox,
    canvas_width: int,
    canvas_height: int,
    alpha_output: bool,
) -> tuple[np.ndarray, BoundingBox]:
    """Execute a sequence of retained effects left-to-right, propagating stage bounds and modifying buffer in-place."""
    current_bounds = base_bounds
    current_buffer = buffer

    for effect in effects:
        stage_output_bounds = effect.expand_bounds(current_bounds)
        stage_ctx = RasterEffectStageContext(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            input_bounds=current_bounds,
            output_bounds=stage_output_bounds,
            alpha_output=alpha_output,
        )
        processor = get_raster_processor(type(effect))
        current_buffer = processor(effect, current_buffer, stage_ctx)
        current_bounds = stage_output_bounds

    return current_buffer, current_bounds
