"""Editable bounded vector masks in object or world coordinates."""
import copy
from dataclasses import dataclass, field
import math

import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform
from drawcv.core.validation import validated_setattr


@dataclass
class VectorMask:
    artwork: object
    bounds: BoundingBox
    mode: str = 'alpha'
    space: str = 'object'
    opacity: float = 1.
    transform: Transform = field(default_factory=Transform)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_initialized', True)

    def __setattr__(self, name, value):
        validated_setattr(self, name, value)

    def _validate(self):
        from drawcv.group import Group
        from drawcv.shapes.path import Path
        from drawcv.styles.pattern import VectorPattern
        if self.mode not in ('alpha', 'luminance'):
            raise ValidationError("VectorMask mode must be 'alpha' or 'luminance'")
        if self.space not in ('object', 'world'):
            raise ValidationError("VectorMask space must be 'object' or 'world'")
        if not isinstance(self.bounds, BoundingBox) or self.bounds.width <= 0 or self.bounds.height <= 0:
            raise ValidationError('VectorMask bounds must be a positive BoundingBox')
        if not isinstance(self.transform, Transform):
            raise ValidationError('VectorMask transform must be a Transform')
        if isinstance(self.opacity, bool) or not isinstance(self.opacity, (int, float)) or not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValidationError('VectorMask opacity must be finite and in [0, 1]')
        active = set()
        def visit(node):
            if id(node) in active:
                raise ValidationError('Cyclic vector mask artwork is not supported')
            if not isinstance(node, (Path, Group)):
                raise ValidationError('VectorMask artwork must contain Paths or Groups')
            if node.mask is not None:
                raise ValidationError('Nested masks inside vector mask artwork are not supported')
            active.add(id(node))
            for name in ('fill', 'stroke'):
                style = getattr(node, name, None)
                if style is not None and isinstance(style.paint, VectorPattern):
                    style.paint._validate()
            for name in ('marker_start', 'marker_mid', 'marker_end'):
                marker = getattr(node, name, None)
                if marker is not None:
                    marker._validate()
                    visit(marker.path)
            for child in getattr(node, 'children', []):
                visit(child)
            active.remove(id(node))
        visit(self.artwork)

    def matrix(self, owner):
        matrix = self.transform.get_matrix(Point(0,0))
        if self.space == 'object':
            matrix = getattr(owner, 'world_matrix', np.eye(3)) @ matrix
        return matrix

    def content(self, owner):
        """Independent world-positioned artwork with a transformed bounds clip."""
        self._validate()
        from drawcv.group import Group
        from drawcv.shapes.path import Path
        root = self.artwork
        memo = {id(value): None for value in (root._parent, root._layer, root._scene) if value is not None}
        art = copy.deepcopy(root, memo)
        art._parent = art._layer = art._scene = None
        art.transform = Transform.from_matrix(root.transform.get_matrix(root.get_geometry_bounds().center))
        b = self.bounds
        clip = Path().move_to(b.left,b.top).line_to(b.right,b.top).line_to(b.right,b.bottom).line_to(b.left,b.bottom).close()
        return Group(children=[art], clip=clip, opacity=self.opacity, transform=Transform.from_matrix(self.matrix(owner)))

    def world_bounds(self, owner):
        matrix = self.matrix(owner)
        points = [matrix @ np.array([p.x,p.y,1.]) for p in self.bounds.corners]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return BoundingBox(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))

    def coverage(self, owner, width, height):
        from drawcv.scene import Scene
        from drawcv.renderer import OpenCVRenderer
        from drawcv.core.color import Color
        self._validate()
        if np.linalg.det(self.matrix(owner)[:2,:2]) == 0:
            return np.zeros((height,width),dtype=np.float32)
        scene = Scene(width,height,background=Color(0,0,0,0))
        scene.add(self.content(owner))
        rgba = OpenCVRenderer().render(scene,alpha=True).buffer.astype(np.float32)/255.
        alpha = rgba[...,3]
        if self.mode == 'alpha':
            return alpha
        # SVG mask luminance in explicitly selected sRGB, multiplied by alpha.
        return alpha*(.0722*rgba[...,0]+.7152*rgba[...,1]+.2126*rgba[...,2])

    def to_dict(self):
        self._validate()
        return dict(type='vector_mask', artwork=self.artwork.to_dict(), bounds=self.bounds.to_dict(),
                    mode=self.mode, space=self.space, opacity=self.opacity, transform=self.transform.to_dict())

    @classmethod
    def from_dict(cls, data):
        from drawcv.serialization.registry import get_drawable_deserializer
        artwork = data['artwork']
        return cls(get_drawable_deserializer(artwork['type'])(artwork), BoundingBox.from_dict(data['bounds']),
                   data.get('mode','alpha'), data.get('space','object'), data.get('opacity',1.),
                   Transform.from_dict(data['transform']) if data.get('transform') else Transform())
