"""Freehand stroke shape representation and configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import math
from typing import Any
import uuid

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point, StrokePoint, distance_point_to_segment
import numpy as np
from drawcv.core.path_processing import (
    catmull_rom_spline,
    chaikin_smooth,
    compute_path_length,
    rdp_simplify,
)
from drawcv.styles.stroke import StrokeStyle


@dataclass(eq=False)
class FreehandStroke(Drawable):
    """Retained-mode freehand stroke preserving raw sampled points with modular processing.
    
    Attributes:
        points: Ordered list of original raw sampled StrokePoints (authoritative source data).
        stroke: Optional StrokeStyle for stroke rendering.
        smoothing: Optional smoothing algorithm (e.g. "chaikin").
        smoothing_iterations: Number of smoothing refinement iterations (>= 0).
        smoothing_ratio: Corner-cutting ratio for Chaikin algorithm (0.0 < ratio < 0.5).
        simplification: Optional simplification algorithm (e.g. "rdp").
        simplification_tolerance: Perpendicular distance threshold for RDP (>= 0.0).
        interpolation: Optional interpolation algorithm (e.g. "catmull_rom").
        interpolation_samples: Number of sample points per spline segment (>= 1).
        variable_width: Whether to vary stroke width along the path.
        width_mode: Width variation mode ("constant", "pressure", "velocity").
        min_width: Minimum rendered stroke width (>= 0.0).
        max_width: Maximum rendered stroke width (>= min_width).
        velocity_min: Lower bound for velocity normalization (>= 0.0).
        velocity_max: Upper bound for velocity normalization (> velocity_min).
    """
    supports_progressive_rendering: bool = True

    points: list[StrokePoint] = field(default_factory=list)
    stroke: StrokeStyle | None = None

    def slice_at_progress(self, progress: float) -> FreehandStroke:
        """Return a transient partial FreehandStroke sliced along arc length."""
        from drawcv.core.path_processing import slice_stroke_points
        p = max(0.0, min(1.0, float(progress)))
        sliced_pts = slice_stroke_points(self.points, p)
        if len(sliced_pts) < 2:
            if len(sliced_pts) == 1:
                sliced_pts = [sliced_pts[0], sliced_pts[0].copy()]
            else:
                sliced_pts = [StrokePoint(0.0, 0.0), StrokePoint(0.0, 0.0)]

        sliced = FreehandStroke(
            points=sliced_pts,
            stroke=self.stroke.copy() if self.stroke else None,
            smoothing=self.smoothing,
            smoothing_iterations=self.smoothing_iterations,
            smoothing_ratio=self.smoothing_ratio,
            simplification=self.simplification,
            simplification_tolerance=self.simplification_tolerance,
            interpolation=self.interpolation,
            interpolation_samples=self.interpolation_samples,
            variable_width=self.variable_width,
            width_mode=self.width_mode,
            min_width=self.min_width,
            max_width=self.max_width,
            velocity_min=self.velocity_min,
            velocity_max=self.velocity_max,
            name=self.name,
            visible=self.visible,
            locked=self.locked,
            opacity=self.opacity,
            z_index=self.z_index,
            transform=self.transform.copy(),
            clip=copy.deepcopy(self.clip),
            mask=copy.deepcopy(self.mask),
            effects=[copy.deepcopy(e) for e in self.effects],
        )
        sliced.render_progress = 1.0
        return sliced
    smoothing: str | None = None
    smoothing_iterations: int = 1
    smoothing_ratio: float = 0.25
    simplification: str | None = None
    simplification_tolerance: float = 1.0
    interpolation: str | None = None
    interpolation_samples: int = 8
    variable_width: bool = False
    width_mode: str = "pressure"
    min_width: float | None = None
    max_width: float | None = None
    velocity_min: float = 0.0
    velocity_max: float = 1000.0

    def __post_init__(self):
        super().__post_init__()
        self._normalize_and_validate()

    def _validate(self) -> None:
        super()._validate()
        self._normalize_and_validate()

    def _normalize_and_validate(self):
        # 1. Normalize points
        if not isinstance(self.points, list):
            raise ValidationError(f"FreehandStroke 'points' must be a list, got {type(self.points).__name__}")

        norm_points: list[StrokePoint] = []
        for idx, pt in enumerate(self.points):
            if isinstance(pt, StrokePoint):
                norm_points.append(pt)
            elif isinstance(pt, Point):
                norm_points.append(StrokePoint(pt.x, pt.y))
            elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                norm_points.append(StrokePoint(pt[0], pt[1]))
            else:
                raise ValidationError(f"Point at index {idx} cannot be converted to StrokePoint: {pt}")
        self.points = norm_points

        # Validate non-decreasing timestamps
        last_t: float | None = None
        for idx, pt in enumerate(self.points):
            if pt.timestamp is not None:
                if last_t is not None and pt.timestamp < last_t:
                    raise ValidationError(
                        f"Point timestamps must be monotonically non-decreasing: "
                        f"index {idx} (t={pt.timestamp}) < index {idx - 1} (t={last_t})"
                    )
                last_t = pt.timestamp

        # 2. Validate stroke style
        if self.stroke is not None and not isinstance(self.stroke, StrokeStyle):
            raise ValidationError(f"FreehandStroke 'stroke' must be a StrokeStyle or None, got {type(self.stroke).__name__}")

        # 3. Validate algorithms
        if self.smoothing is not None:
            if not isinstance(self.smoothing, str):
                raise ValidationError("smoothing must be a string or None")
            norm_s = self.smoothing.strip().lower()
            if norm_s not in ("chaikin",):
                raise ValidationError(f"Unknown smoothing algorithm: '{self.smoothing}' (supported: 'chaikin')")
            self.smoothing = norm_s

        if not isinstance(self.smoothing_iterations, int) or self.smoothing_iterations < 0:
            raise ValidationError(f"smoothing_iterations must be a non-negative integer, got {self.smoothing_iterations}")

        if not isinstance(self.smoothing_ratio, (int, float)) or not (0.0 < float(self.smoothing_ratio) < 0.5):
            raise ValidationError(f"smoothing_ratio must be between 0.0 and 0.5 exclusive, got {self.smoothing_ratio}")

        if self.simplification is not None:
            if not isinstance(self.simplification, str):
                raise ValidationError("simplification must be a string or None")
            norm_simp = self.simplification.strip().lower()
            if norm_simp not in ("rdp",):
                raise ValidationError(f"Unknown simplification algorithm: '{self.simplification}' (supported: 'rdp')")
            self.simplification = norm_simp

        if not isinstance(self.simplification_tolerance, (int, float)) or float(self.simplification_tolerance) < 0.0:
            raise ValidationError(f"simplification_tolerance must be non-negative, got {self.simplification_tolerance}")

        if self.interpolation is not None:
            if not isinstance(self.interpolation, str):
                raise ValidationError("interpolation must be a string or None")
            norm_interp = self.interpolation.strip().lower()
            if norm_interp not in ("catmull_rom",):
                raise ValidationError(f"Unknown interpolation algorithm: '{self.interpolation}' (supported: 'catmull_rom')")
            self.interpolation = norm_interp

        if not isinstance(self.interpolation_samples, int) or self.interpolation_samples < 1:
            raise ValidationError(f"interpolation_samples must be an integer >= 1, got {self.interpolation_samples}")

        # 4. Validate width mode and parameters
        if not isinstance(self.variable_width, bool):
            raise ValidationError("variable_width must be a boolean")

        if not isinstance(self.width_mode, str):
            raise ValidationError("width_mode must be a string")
        norm_wm = self.width_mode.strip().lower()
        if norm_wm not in ("constant", "pressure", "velocity"):
            raise ValidationError(f"Unknown width_mode: '{self.width_mode}' (supported: 'constant', 'pressure', 'velocity')")
        self.width_mode = norm_wm

        if self.min_width is not None:
            if not isinstance(self.min_width, (int, float)) or float(self.min_width) < 0.0:
                raise ValidationError(f"min_width must be non-negative, got {self.min_width}")
            self.min_width = float(self.min_width)

        if self.max_width is not None:
            if not isinstance(self.max_width, (int, float)):
                raise ValidationError(f"max_width must be numeric, got {self.max_width}")
            min_w = self.min_width if self.min_width is not None else 0.0
            if float(self.max_width) < min_w:
                raise ValidationError(f"max_width ({self.max_width}) cannot be less than min_width ({min_w})")
            self.max_width = float(self.max_width)

        if not isinstance(self.velocity_min, (int, float)) or float(self.velocity_min) < 0.0:
            raise ValidationError(f"velocity_min must be non-negative, got {self.velocity_min}")
        self.velocity_min = float(self.velocity_min)

        if not isinstance(self.velocity_max, (int, float)) or float(self.velocity_max) <= self.velocity_min:
            raise ValidationError(f"velocity_max ({self.velocity_max}) must be strictly greater than velocity_min ({self.velocity_min})")
        self.velocity_max = float(self.velocity_max)

    # -------------------------------------------------------------------------
    # Real-time Point Appending (Non-destructive source data)
    # -------------------------------------------------------------------------

    def add_point(
        self,
        pt: StrokePoint | Point | tuple[float, float],
        pressure: float | None = None,
        timestamp: float | None = None,
        velocity: float | None = None,
    ) -> StrokePoint:
        """Append a new sampled point during stroke capture without destroying existing points.
        
        If timestamp is provided but velocity is not, velocity is automatically computed as
        v = delta_distance / delta_time (with delta_time <= 0 protected to yield v = 0.0).
        """
        # Resolve x, y
        if isinstance(pt, StrokePoint):
            x, y = pt.x, pt.y
            p = pressure if pressure is not None else pt.pressure
            t = timestamp if timestamp is not None else pt.timestamp
            v = velocity if velocity is not None else pt.velocity
        elif isinstance(pt, Point):
            x, y = pt.x, pt.y
            p, t, v = pressure, timestamp, velocity
        elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
            x, y = pt[0], pt[1]
            p, t, v = pressure, timestamp, velocity
        else:
            raise ValidationError(f"Invalid point: {pt}")

        # Check timestamp monotonicity against previous point
        if t is not None and len(self.points) > 0:
            last_pt = self.points[-1]
            if last_pt.timestamp is not None:
                if t < last_pt.timestamp:
                    raise ValidationError(
                        f"Point timestamps must be monotonically non-decreasing: {t} < {last_pt.timestamp}"
                    )
                # Auto-calculate velocity if requested
                if v is None:
                    dt = t - last_pt.timestamp
                    if dt <= 0.0:
                        v = 0.0
                    else:
                        dist = math.hypot(x - last_pt.x, y - last_pt.y)
                        v = dist / dt

        new_pt = StrokePoint(x, y, pressure=p, timestamp=t, velocity=v)
        self.points.append(new_pt)
        return new_pt

    # -------------------------------------------------------------------------
    # Authoritative Processing Pipeline
    # -------------------------------------------------------------------------

    def get_processed_points(self) -> list[StrokePoint]:
        """Execute the authoritative 5-step processing pipeline without mutating raw source points.
        
        Order:
            Raw -> Optional RDP -> Optional Chaikin -> Optional Catmull-Rom
        """
        if len(self.points) <= 1:
            return [p.copy() for p in self.points]

        # 1. Start with copy of raw source points
        pts = [p.copy() for p in self.points]

        # 2. Optional RDP Simplification
        if self.simplification == "rdp" and self.simplification_tolerance > 0.0:
            pts = rdp_simplify(pts, self.simplification_tolerance)

        # 3. Optional Chaikin Smoothing
        if self.smoothing == "chaikin" and self.smoothing_iterations > 0:
            pts = chaikin_smooth(pts, self.smoothing_iterations, self.smoothing_ratio)

        # 4. Optional Catmull-Rom Spline Interpolation
        if self.interpolation == "catmull_rom" and self.interpolation_samples >= 1 and len(pts) >= 2:
            pts = catmull_rom_spline(pts, self.interpolation_samples)

        return pts

    def get_point_widths(self, points: list[StrokePoint] | None = None) -> list[float]:
        """Evaluate stroke widths for each point in the stroke according to width_mode."""
        if points is None:
            points = self.get_processed_points()

        base_w = self.stroke.width if self.stroke is not None else 1.0
        if not self.variable_width or self.width_mode == "constant" or len(points) == 0:
            return [base_w] * len(points)

        w_min = self.min_width if self.min_width is not None else 0.5 * base_w
        w_max = self.max_width if self.max_width is not None else 1.5 * base_w
        if w_max < w_min:
            w_max = w_min

        widths: list[float] = []
        if self.width_mode == "pressure":
            for pt in points:
                p = pt.pressure if pt.pressure is not None else 0.5
                widths.append(w_min + p * (w_max - w_min))
        elif self.width_mode == "velocity":
            v_span = max(1e-6, self.velocity_max - self.velocity_min)
            for pt in points:
                if pt.velocity is not None:
                    u = max(0.0, min(1.0, (pt.velocity - self.velocity_min) / v_span))
                else:
                    u = 0.5
                # Faster velocity produces thinner lines
                widths.append(w_max - u * (w_max - w_min))
        else:
            widths = [base_w] * len(points)

        return widths

    @property
    def length(self) -> float:
        """Cumulative length of the processed freehand path in local space."""
        return compute_path_length(self.get_processed_points())

    # -------------------------------------------------------------------------
    # Bounds Hierarchy (Accounting for Maximum Evaluated Width)
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds of raw point centerlines in local space."""
        if not self.points:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds in local space accounting for maximum stroke width across all processed points."""
        pts = self.get_processed_points()
        if not pts:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        widths = self.get_point_widths(pts)
        max_half_w = (max(widths) * (self.stroke.bounds_padding / self.stroke.width if self.stroke else 0.5)) if widths else (self.stroke.bounds_padding if self.stroke else 0.0)
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y).expand(max_half_w)

    def get_bounds(self) -> BoundingBox:
        """World-space visual AABB accounting for maximum stroke width under non-scaling stroke rule."""
        pts = self.get_processed_points()
        if not pts:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        M = self.world_matrix
        coords = np.array([[p.x, p.y, 1.0] for p in pts], dtype=np.float64).T
        world_coords = (M @ coords)[:2].T
        xs = world_coords[:, 0]
        ys = world_coords[:, 1]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        widths = self.get_point_widths(pts)
        max_half_w = (max(widths) * (self.stroke.bounds_padding / self.stroke.width if self.stroke else 0.5)) if widths else (self.stroke.bounds_padding if self.stroke else 0.0)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y).expand(max_half_w)

    def get_bounds_in_parent(self) -> BoundingBox:
        """Visual bounds in immediate parent's coordinate space."""
        pts = self.get_processed_points()
        if not pts:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        default_pivot = self.get_geometry_bounds().center
        local_M = self.transform.get_matrix(default_pivot=default_pivot)
        coords = np.array([[p.x, p.y, 1.0] for p in pts], dtype=np.float64).T
        parent_coords = (local_M @ coords)[:2].T
        xs = parent_coords[:, 0]
        ys = parent_coords[:, 1]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        widths = self.get_point_widths(pts)
        max_half_w = (max(widths) * (self.stroke.bounds_padding / self.stroke.width if self.stroke else 0.5)) if widths else (self.stroke.bounds_padding if self.stroke else 0.0)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y).expand(max_half_w)

    # -------------------------------------------------------------------------
    # Hit Testing & Anchors
    # -------------------------------------------------------------------------

    def contains_point(self, point_or_x: Point | float | int, y: float | int | None = None, tolerance: float = 4.0) -> bool:
        """Check if world point (x, y) or Point is within tolerance of the stroke path."""
        if isinstance(point_or_x, Point):
            world_p = point_or_x
        elif y is not None and isinstance(point_or_x, (int, float)) and isinstance(y, (int, float)):
            world_p = Point(point_or_x, y)
        else:
            raise ValidationError(f"Invalid point arguments: {point_or_x}, {y}")

        local_p = self.to_local(world_p)
        pts = self.get_processed_points()
        if not pts:
            return False

        widths = self.get_point_widths(pts)
        if len(pts) == 1:
            half_w = widths[0] / 2.0
            return local_p.distance_to(pts[0]) <= half_w + tolerance

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            dist = distance_point_to_segment(local_p, p1, p2)
            seg_half_w = max(widths[i], widths[i + 1]) / 2.0
            if dist <= seg_half_w + tolerance:
                return True

        return False

    def anchor(self, name: str) -> Point:
        """Retrieve shape-specific geometric anchor point mapped to world coordinates.
        
        Supported: 'start', 'end', 'midpoint', or standard BoundingBox anchors ('center', 'top_left', etc.).
        """
        norm = name.strip().lower()
        pts = self.get_processed_points()
        if norm == "start" and pts:
            return self.to_world(pts[0].to_point())
        if norm == "end" and pts:
            return self.to_world(pts[-1].to_point())
        if norm == "midpoint" and pts:
            if len(pts) == 1:
                return self.to_world(pts[0].to_point())
            half_len = self.length / 2.0
            cum_len = 0.0
            for i in range(len(pts) - 1):
                seg_d = pts[i].distance_to(pts[i + 1])
                if cum_len + seg_d >= half_len:
                    rem = half_len - cum_len
                    t = rem / seg_d if seg_d > 0.0 else 0.0
                    mid_x = pts[i].x + t * (pts[i + 1].x - pts[i].x)
                    mid_y = pts[i].y + t * (pts[i + 1].y - pts[i].y)
                    return self.to_world(Point(mid_x, mid_y))
                cum_len += seg_d
            return self.to_world(pts[-1].to_point())

        # Fallback to BoundingBox visual anchors
        return self.get_bounds().anchor(norm)

    def clone(self, new_id: bool = True) -> FreehandStroke:
        """Create an independent deep copy of the FreehandStroke preserving all raw points and settings."""
        return super().clone(new_id=new_id)  # type: ignore

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "points": [p.copy() for p in self.points],
            "stroke": self.stroke.copy() if self.stroke else None,
            "smoothing": self.smoothing,
            "smoothing_iterations": int(self.smoothing_iterations),
            "smoothing_ratio": float(self.smoothing_ratio),
            "simplification": self.simplification,
            "simplification_tolerance": float(self.simplification_tolerance),
            "interpolation": self.interpolation,
            "interpolation_samples": int(self.interpolation_samples),
            "variable_width": bool(self.variable_width),
            "width_mode": str(self.width_mode),
            "min_width": float(self.min_width) if self.min_width is not None else None,
            "max_width": float(self.max_width) if self.max_width is not None else None,
            "velocity_min": float(self.velocity_min),
            "velocity_max": float(self.velocity_max),
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "points" in state:
            self.points = [p.copy() if isinstance(p, StrokePoint) else StrokePoint.from_dict(p) for p in state["points"]]
        if "stroke" in state:
            st = state["stroke"]
            self.stroke = st.copy() if isinstance(st, StrokeStyle) else (StrokeStyle.from_dict(st) if st else None)
        if "smoothing" in state:
            self.smoothing = state["smoothing"]
        if "smoothing_iterations" in state:
            self.smoothing_iterations = int(state["smoothing_iterations"])
        if "smoothing_ratio" in state:
            self.smoothing_ratio = float(state["smoothing_ratio"])
        if "simplification" in state:
            self.simplification = state["simplification"]
        if "simplification_tolerance" in state:
            self.simplification_tolerance = float(state["simplification_tolerance"])
        if "interpolation" in state:
            self.interpolation = state["interpolation"]
        if "interpolation_samples" in state:
            self.interpolation_samples = int(state["interpolation_samples"])
        if "variable_width" in state:
            self.variable_width = bool(state["variable_width"])
        if "width_mode" in state:
            self.width_mode = str(state["width_mode"])
        if "min_width" in state:
            self.min_width = float(state["min_width"]) if state["min_width"] is not None else None
        if "max_width" in state:
            self.max_width = float(state["max_width"]) if state["max_width"] is not None else None
        if "velocity_min" in state:
            self.velocity_min = float(state["velocity_min"])
        if "velocity_max" in state:
            self.velocity_max = float(state["velocity_max"])
        self._invalidate_cache()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "freehand",
            "points": [p.to_dict() for p in self.points],
            "stroke": self.stroke.to_dict() if self.stroke else None,
            "smoothing": self.smoothing,
            "smoothing_iterations": int(self.smoothing_iterations),
            "smoothing_ratio": float(self.smoothing_ratio),
            "simplification": self.simplification,
            "simplification_tolerance": float(self.simplification_tolerance),
            "interpolation": self.interpolation,
            "interpolation_samples": int(self.interpolation_samples),
            "variable_width": bool(self.variable_width),
            "width_mode": str(self.width_mode),
            "min_width": float(self.min_width) if self.min_width is not None else None,
            "max_width": float(self.max_width) if self.max_width is not None else None,
            "velocity_min": float(self.velocity_min),
            "velocity_max": float(self.velocity_max),
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreehandStroke:
        """Construct a FreehandStroke from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        pts = [StrokePoint.from_dict(p) for p in data.get("points", [])]
        stroke = StrokeStyle.from_dict(data["stroke"]) if data.get("stroke") is not None else None
        return cls(
            points=pts,
            stroke=stroke,
            smoothing=data.get("smoothing"),
            smoothing_iterations=int(data.get("smoothing_iterations", 1)),
            smoothing_ratio=float(data.get("smoothing_ratio", 0.25)),
            simplification=data.get("simplification"),
            simplification_tolerance=float(data.get("simplification_tolerance", 1.0)),
            interpolation=data.get("interpolation"),
            interpolation_samples=int(data.get("interpolation_samples", 8)),
            variable_width=bool(data.get("variable_width", False)),
            width_mode=str(data.get("width_mode", "pressure")),
            min_width=float(data["min_width"]) if data.get("min_width") is not None else None,
            max_width=float(data["max_width"]) if data.get("max_width") is not None else None,
            velocity_min=float(data.get("velocity_min", 0.0)),
            velocity_max=float(data.get("velocity_max", 1000.0)),
            **base_kwargs,
        )


