"""Retained, editable vector tiles for fill and stroke paints."""
import copy
from dataclasses import dataclass, field
import math

import numpy as np

from drawcv.core.exceptions import ValidationError, RenderError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.styles.paint import Paint, ImagePaint


@dataclass(eq=False)
class VectorPattern(Paint):
    artwork: object
    width: float
    height: float
    spacing: tuple[float, float] = (0., 0.)
    origin: Point = field(default_factory=lambda: Point(0, 0))
    space: str = 'object'
    opacity: float = 1.
    transform: Transform = field(default_factory=Transform)

    def _validate(self):
        super()._validate()
        for name in ('width', 'height'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValidationError(f'Pattern {name} must be positive and finite')
        if not isinstance(self.spacing, (tuple, list)) or len(self.spacing) != 2:
            raise ValidationError('Pattern spacing must contain two non-negative numbers')
        for value in self.spacing:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValidationError('Pattern spacing must be finite and non-negative')
        if not math.isfinite(self.width+self.spacing[0]) or not math.isfinite(self.height+self.spacing[1]):
            raise ValidationError('Pattern repeat dimensions must be finite')
        if not isinstance(self.origin, Point):
            raise ValidationError('Pattern origin must be a Point')
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float)) or not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValidationError('Pattern opacity must be in [0, 1]')
        from drawcv.group import Group
        from drawcv.shapes.path import Path
        if not isinstance(self.artwork, (Path, Group)):
            raise ValidationError('Pattern artwork must be a Path or Group; convert shapes with to_path()')
        active = set()
        def visit(node):
            if id(node) in active:
                raise ValidationError('Cyclic pattern artwork is not supported')
            active.add(id(node))
            if not isinstance(node, (Path, Group)):
                raise ValidationError('Pattern groups must contain Paths or Groups')
            if node.effects or node.mask is not None:
                raise ValidationError('Pattern artwork effects and masks are not supported')
            if any(getattr(node, key, None) is not None for key in ('marker_start', 'marker_mid', 'marker_end')):
                raise ValidationError('Pattern artwork markers must be expanded to paths first')
            for key in ('fill', 'stroke'):
                style = getattr(node, key, None)
                if style is not None and isinstance(style.paint, (VectorPattern, ImagePaint)):
                    raise ValidationError('Nested patterns and raster paints are not supported in vector tiles')
            for child in getattr(node, 'children', []):
                visit(child)
            active.remove(id(node))
        visit(self.artwork)

    def tile(self):
        """Return isolated tile content with a local clip, without changing artwork."""
        self._validate()
        from drawcv.group import Group
        from drawcv.shapes.path import Path
        root = self.artwork
        memo = {id(value): None for value in (root._parent, root._layer, root._scene) if value is not None}
        art = copy.deepcopy(root, memo)
        art._parent = art._layer = art._scene = None
        art.transform = Transform.from_matrix(root.transform.get_matrix(root.get_geometry_bounds().center))
        def normalize(node):
            for key in ('fill', 'stroke'):
                style = getattr(node, key, None)
                if style is None:
                    continue
                style = style.copy()
                setattr(node, key, style)
                if key == 'stroke':
                    style.space = 'object'
                paint = style.paint
                if isinstance(paint, Paint) and paint.space == 'world':
                    paint.transform = Transform.from_matrix(np.linalg.pinv(node.world_matrix) @ paint.transform.get_matrix(Point(0,0)))
                    paint.space = 'object'
            for child in getattr(node, 'children', []):
                normalize(child)
        normalize(art)
        clip = Path().move_to(0,0).line_to(self.width,0).line_to(self.width,self.height).line_to(0,self.height).close()
        return Group(children=[art], clip=clip, opacity=self.opacity)

    def to_dict(self):
        self._validate()
        return dict(type='vector_pattern', artwork=self.artwork.to_dict(), width=self.width, height=self.height,
                    spacing=list(self.spacing), origin=self.origin.to_dict(), space=self.space, opacity=self.opacity,
                    transform=self.transform.to_dict())

    @classmethod
    def from_dict(cls, data):
        from drawcv.serialization.registry import get_drawable_deserializer
        artwork = data['artwork']
        return cls(get_drawable_deserializer(artwork['type'])(artwork), data['width'], data['height'],
                   tuple(data.get('spacing', (0,0))), Point.from_dict(data.get('origin', {'x':0, 'y':0})),
                   data.get('space', 'object'), data.get('opacity', 1.), Transform.from_dict(data['transform']) if data.get('transform') else Transform())


def sample_vector_pattern(paint, matrix, width, height, origin):
    from drawcv.scene import Scene
    from drawcv.renderer import OpenCVRenderer
    from drawcv.core.color import Color
    from drawcv.core.paint_sampling import sample_image
    paint._validate()
    transform = paint.transform.get_matrix(Point(0,0))
    effective = matrix @ transform if paint.space == 'object' else transform
    if np.linalg.det(effective[:2,:2]) == 0:
        return np.zeros((height, width, 4), dtype=np.float32)
    # Supersample at the largest affine stretch so rotation/shear do not blur
    # retained artwork into a fixed-resolution texture when zooming in.
    density = max(1., 2*float(np.linalg.norm(effective[:2,:2], ord=2)))
    pw, ph = paint.width+paint.spacing[0], paint.height+paint.spacing[1]
    if not math.isfinite(density) or pw*density > 8192 or ph*density > 8192:
        raise RenderError('Vector pattern tile exceeds raster working limit; reduce tile size or transform scale')
    rw, rh = max(1, math.ceil(pw*density)), max(1, math.ceil(ph*density))
    if rw > 8192 or rh > 8192 or rw*rh > 16_777_216:
        raise RenderError('Vector pattern tile exceeds raster working limit; reduce tile size or transform scale')
    tile = paint.tile()
    tile.transform = Transform.from_matrix(np.diag([rw/pw, rh/ph, 1.]))
    scene = Scene(rw, rh, background=Color(0,0,0,0))
    scene.add(tile)
    pixels = OpenCVRenderer().render(scene, alpha=True).buffer
    image = ImagePaint(pixels, origin=paint.origin, scale=(pw/rw, ph/rh), space=paint.space, transform=paint.transform.copy())
    return sample_image(image, matrix, width, height, origin)
