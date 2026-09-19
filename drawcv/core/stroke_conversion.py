"""Convert standard and freehand strokes to world-space filled geometry."""
import copy
import math

import numpy as np

from drawcv.core.color import Color
from drawcv.core.enums import FillRule
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import resolve_nonzero_contours
from drawcv.core.path_conversion import path_subpaths
from drawcv.core.stroking import _stroke_geometry
from drawcv.core.transform import Transform
from drawcv.shapes.path import Path
from drawcv.shapes.freehand import FreehandStroke
from drawcv.styles.fill import FillStyle


def stroke_to_path(obj, *, tolerance=0.25):
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValidationError("tolerance must be a finite positive number")
    freehand = isinstance(obj, FreehandStroke)
    geometry = None if freehand else Path(subpaths=path_subpaths(obj))
    result = Path(fill_rule=FillRule.NON_ZERO, name=obj.name, opacity=obj.opacity,
                  visible=obj.visible, locked=obj.locked, blend_mode=obj.blend_mode,
                  z_index=obj.z_index, tags=copy.deepcopy(obj.tags), metadata=copy.deepcopy(obj.metadata))
    style = getattr(obj, 'stroke', None)
    if style is None:
        return result
    matrix = obj.world_matrix
    paint = copy.deepcopy(style.paint)
    if not isinstance(paint, Color) and paint.space == 'object':
        paint.transform = Transform.from_matrix(matrix @ paint.transform.get_matrix(Point(0, 0)))
        paint.space = 'world'
    result.fill = FillStyle(paint=paint, opacity=style.opacity)
    if style.width <= 0:
        return result
    object_space = style.space == 'object'
    scale = float(np.linalg.norm(matrix[:2, :2], ord=2)) if object_space else 1.0
    if object_space and abs(np.linalg.det(matrix[:2, :2])) == 0:
        return result
    local_tolerance = tolerance / max(scale, 1e-12)
    if freehand:
        # Processing interpolates pressure/velocity as well as coordinates.
        # Evaluate widths before mapping geometry, exactly as in the renderer.
        points = obj.get_processed_points()
        widths = obj.get_point_widths(points)
        mapped = points if object_space else [obj.to_world(p.to_point()) for p in points]
        samples = [([(p.x, p.y, width) for p, width in zip(mapped, widths)], False)]
    else:
        contours = geometry.flatten_with_mapper(
            (lambda p: p) if object_space else obj.to_world,
            tolerance=min(local_tolerance, tolerance) / 2, include_closed=True)
        samples = [([(p.x, p.y, style.width) for p in points], loop)
                   for points, loop in contours]
    polygons = []

    def polygon(points):
        points = np.asarray(points, dtype=float)
        if object_space:
            points = points @ matrix[:2, :2].T + matrix[:2, 2]
        # All pieces must have the same winding so overlaps union, not cancel.
        area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
        if area == 0:
            return
        if area < 0:
            points = points[::-1]
        polygons.append([Point(float(x), float(y)) for x, y in points])

    def disk(p, radius):
        if radius <= 0:
            return
        angle = 2 * math.acos(np.clip(1 - local_tolerance / (2 * radius), -1, 1))
        count = max(12, math.ceil(2 * math.pi / max(angle, 1e-6)))
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        polygon(p + radius * np.column_stack((np.cos(angles), np.sin(angles))))

    _stroke_geometry(samples, style, polygon, disk)
    for contour in resolve_nonzero_contours(polygons):
        result.move_to(contour[0].x, contour[0].y)
        for point in contour[1:]:
            result.line_to(point.x, point.y)
        result.close()
    return result
