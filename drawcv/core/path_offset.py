"""Filled-region offsets using normalized boundaries and vector boolean cleanup."""
import copy
import math

import pathops

from drawcv.core.color import Color
from drawcv.core.enums import JoinStyle
from drawcv.core.exceptions import ValidationError, PathBooleanError
from drawcv.core.geometry import Point
from drawcv.core.path_boolean import drawcv_to_pathops, pathops_to_drawcv
from drawcv.core.transform import Transform
from drawcv.shapes.path import Path, MoveTo, Close
from drawcv.styles.fill import FillStyle
from drawcv.styles.stroke import StrokeStyle


def offset(path, distance, *, tolerance=.25, space='world', join_style=JoinStyle.ROUND, miter_limit=4.):
    for name, value in (('distance', distance), ('tolerance', tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError(f'{name} must be a finite number')
    if tolerance <= 0:
        raise ValidationError('tolerance must be positive')
    if space not in ('local', 'world'):
        raise ValidationError("space must be 'local' or 'world'")
    # Reuse style validation even for an empty or zero-distance operation.
    style = StrokeStyle(width=max(2*abs(distance), 1e-12), join_style=join_style, miter_limit=miter_limit)
    for contour in path.subpaths:
        drawing = any(not isinstance(cmd, (MoveTo, Close)) for cmd in contour.commands)
        if drawing and not (contour.closed or (contour.commands and isinstance(contour.commands[-1], Close))):
            raise ValidationError('offset() requires closed contours; use stroke_to_path() for open paths')
        # A MoveTo inside a Subpath can create an unclosed interior contour.
        active = False
        for cmd in contour.commands:
            if isinstance(cmd, MoveTo):
                if active:
                    raise ValidationError('offset() requires each contour to be closed')
                active = False
            elif isinstance(cmd, Close):
                active = False
            else:
                active = True
    flat = Path(fill_rule=path.fill_rule)
    mapper = path.to_world if space == 'world' else lambda p: p
    for points, _ in path.flatten_with_mapper(mapper, tolerance=tolerance/2, include_closed=True):
        if len(points) < 3:
            continue
        flat.move_to(points[0])
        for point in points[1:]:
            flat.line_to(point)
        flat.close()
    try:
        # Resolve holes, repeated contours and self-intersections BEFORE stroking:
        # internal edges of overlapping regions must not create erosion seams.
        region = pathops.simplify(drawcv_to_pathops(flat), fix_winding=True)
        boundary = pathops_to_drawcv(region)
        if distance != 0 and boundary.subpaths:
            boundary.stroke = style
            band = boundary.stroke_to_path(tolerance=tolerance/2)
            region = pathops.op(region, drawcv_to_pathops(band),
                                pathops.PathOp.UNION if distance > 0 else pathops.PathOp.DIFFERENCE,
                                fix_winding=True, keep_starting_points=False)
        result = pathops_to_drawcv(region)
    except (ValidationError, PathBooleanError):
        raise
    except Exception as error:
        raise PathBooleanError(f'Path offset failed: {error}') from error
    for name in ('name', 'visible', 'locked', 'opacity', 'blend_mode', 'z_index', 'tags', 'metadata'):
        setattr(result, name, copy.deepcopy(getattr(path, name)))
    result.fill = copy.deepcopy(path.fill) if path.fill is not None else FillStyle()
    if space == 'local':
        result.transform = Transform.from_matrix(path.world_matrix)
    else:
        paint = result.fill.paint
        if not isinstance(paint, Color) and paint.space == 'object':
            paint.transform = Transform.from_matrix(path.world_matrix @ paint.transform.get_matrix(Point(0, 0)))
            paint.space = 'world'
    return result
