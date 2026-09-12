"""Transform data model representing 2D affine transformations."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point


@dataclass
class Transform:
    """2D affine transformation model.
    
    Attributes:
        translation_x: Horizontal shift in pixels.
        translation_y: Vertical shift in pixels.
        rotation: Clockwise rotation in degrees.
        scale_x: Strictly positive horizontal scale factor (scale_x > 0).
        scale_y: Strictly positive vertical scale factor (scale_y > 0).
        pivot: Anchor point for rotation and scaling (None = dynamic intrinsic center).
    """
    translation_x: float = 0.0
    translation_y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    pivot: Point | None = None
    _matrix: np.ndarray | None = None

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_initialized", True)

    def _validate(self):
        for name in ("translation_x", "translation_y", "rotation", "scale_x", "scale_y"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Transform '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"Transform '{name}' must be finite, got {val}")

        if self.scale_x <= 0:
            raise ValidationError(f"Transform 'scale_x' must be strictly positive, got {self.scale_x}")
        if self.scale_y <= 0:
            raise ValidationError(f"Transform 'scale_y' must be strictly positive, got {self.scale_y}")

        if self.pivot is not None and not isinstance(self.pivot, Point):
            raise ValidationError(f"Transform 'pivot' must be a Point or None, got {type(self.pivot).__name__}")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if getattr(self, "_initialized", False):
            if not name.startswith("_"):
                # If explicit properties are mutated, invalidate explicit matrix cache
                object.__setattr__(self, "_matrix", None)
            self._validate()

    def is_identity(self) -> bool:
        """Check if this transform produces no visual modification."""
        return (
            self.translation_x == 0.0
            and self.translation_y == 0.0
            and self.rotation == 0.0
            and self.scale_x == 1.0
            and self.scale_y == 1.0
        )

    def is_translation_only(self) -> bool:
        """Check if this transform only contains translation."""
        return (
            self.rotation == 0.0
            and self.scale_x == 1.0
            and self.scale_y == 1.0
        )

    def get_effective_pivot(self, default_pivot: Point) -> Point:
        """Resolve pivot: returns explicit pivot if configured, otherwise default_pivot."""
        if self.pivot is not None:
            return self.pivot
        return default_pivot

    def get_matrix(self, default_pivot: Point | None = None) -> np.ndarray:
        """Compute the 3x3 affine transformation matrix M.
        
        M = T(tx, ty) * T(px, py) * R(theta) * S(sx, sy) * T(-px, -py)
        If an authoritative affine matrix was provided via from_matrix, it is returned directly.
        """
        if self._matrix is not None:
            return self._matrix.copy()

        p = self.pivot if self.pivot is not None else (default_pivot or Point(0.0, 0.0))
        px, py = p.x, p.y

        rad = math.radians(self.rotation)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # 1. Translate to origin relative to pivot: T(-px, -py)
        t_to_origin = np.array([
            [1.0, 0.0, -px],
            [0.0, 1.0, -py],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 2. Scale: S(sx, sy)
        s_matrix = np.array([
            [self.scale_x, 0.0,          0.0],
            [0.0,          self.scale_y, 0.0],
            [0.0,          0.0,          1.0]
        ], dtype=np.float64)

        # 3. Rotate clockwise: R(theta)
        r_matrix = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        # 4. Translate back from pivot: T(px, py)
        t_from_origin = np.array([
            [1.0, 0.0, px],
            [0.0, 1.0, py],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 5. Translation: T(tx, ty)
        t_trans = np.array([
            [1.0, 0.0, self.translation_x],
            [0.0, 1.0, self.translation_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        return t_trans @ t_from_origin @ r_matrix @ s_matrix @ t_to_origin

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Transform:
        """Create a Transform from a 3x3 affine matrix.
        
        The 3x3 matrix is kept authoritative in `_matrix` to preserve complete affine information
        (including shear), while also decomposing into canonical translation, rotation, and scale.
        """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            raise ValidationError(f"Expected 3x3 numpy array matrix, got {matrix}")

        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])

        a = float(matrix[0, 0])
        b = float(matrix[1, 0])
        c = float(matrix[0, 1])
        d = float(matrix[1, 1])

        sx = math.sqrt(a * a + b * b)
        sy = math.sqrt(c * c + d * d)
        if sx <= 0:
            sx = 1.0
        if sy <= 0:
            sy = 1.0

        rotation = math.degrees(math.atan2(b, a))

        tf = cls(
            translation_x=tx,
            translation_y=ty,
            rotation=rotation,
            scale_x=sx,
            scale_y=sy,
            pivot=None,
        )
        object.__setattr__(tf, "_matrix", np.array(matrix, dtype=np.float64))
        return tf

    def get_inverse_matrix(self, default_pivot: Point | None = None) -> np.ndarray:
        """Compute the 3x3 inverse affine transformation matrix M^-1."""
        m = self.get_matrix(default_pivot)
        return np.linalg.inv(m)

    def transform_point(self, point: Point, default_pivot: Point | None = None) -> Point:
        """Transform a Point using matrix M."""
        m = self.get_matrix(default_pivot)
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        res = m @ vec
        return Point(res[0], res[1])

    def inverse_transform_point(self, point: Point, default_pivot: Point | None = None) -> Point:
        """Inverse transform a Point using matrix M^-1."""
        inv_m = self.get_inverse_matrix(default_pivot)
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        res = inv_m @ vec
        return Point(res[0], res[1])

    def copy(self) -> Transform:
        """Create an independent copy of this Transform."""
        tf = Transform(
            translation_x=self.translation_x,
            translation_y=self.translation_y,
            rotation=self.rotation,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            pivot=self.pivot.copy() if self.pivot is not None else None,
        )
        if self._matrix is not None:
            object.__setattr__(tf, "_matrix", self._matrix.copy())
        return tf

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res: dict[str, Any] = {
            "translation_x": float(self.translation_x),
            "translation_y": float(self.translation_y),
            "rotation": float(self.rotation),
            "scale_x": float(self.scale_x),
            "scale_y": float(self.scale_y),
            "pivot": self.pivot.to_dict() if self.pivot is not None else None,
        }
        if self._matrix is not None:
            res["matrix"] = self._matrix.tolist()
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Transform:
        """Construct a Transform from a dictionary."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValidationError(f"Transform data must be a dict or None, got {type(data).__name__}")

        pivot_val = None
        if data.get("pivot") is not None:
            pivot_val = Point.from_dict(data["pivot"])

        tf = cls(
            translation_x=float(data.get("translation_x", 0.0)),
            translation_y=float(data.get("translation_y", 0.0)),
            rotation=float(data.get("rotation", 0.0)),
            scale_x=float(data.get("scale_x", 1.0)),
            scale_y=float(data.get("scale_y", 1.0)),
            pivot=pivot_val,
        )
        if "matrix" in data and data["matrix"] is not None:
            object.__setattr__(tf, "_matrix", np.array(data["matrix"], dtype=np.float64))
        return tf

