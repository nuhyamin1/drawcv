"""Clipping boundaries (rectangular, legacy polygon, and retained vector paths)."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence, TYPE_CHECKING
import cv2
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.enums import FillRule
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import evaluate_fill_rule_mask, point_in_polygon

if TYPE_CHECKING:
    from drawcv.shapes.path import Path


@dataclass(frozen=True)
class ClipRect:
    """Rectangular clipping boundary defined in entity-local coordinates.
    
    Attributes:
        x: Left edge in entity-local space.
        y: Top edge in entity-local space.
        width: Width of the clipping box (>= 0).
        height: Height of the clipping box (>= 0).
    """
    x: float
    y: float
    width: float
    height: float

    def __init__(self, x: float | int, y: float | int, width: float | int, height: float | int):
        for name, val in (("x", x), ("y", y), ("width", width), ("height", height)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"ClipRect '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"ClipRect '{name}' must be a finite number, got {val}")

        if float(width) < 0.0 or float(height) < 0.0:
            raise ValidationError(f"ClipRect dimensions must be non-negative, got ({width}, {height})")

        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "height", float(height))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "rect",
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipRect:
        """Construct a ClipRect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"ClipRect data must be a dict, got {type(data).__name__}")
        return cls(data["x"], data["y"], data["width"], data["height"])

    @property
    def bounds(self) -> BoundingBox:
        """Local bounding box of the clip rectangle."""
        return BoundingBox(self.x, self.y, self.width, self.height)

    @property
    def corners(self) -> list[Point]:
        """Return 4 local corners: top-left, top-right, bottom-right, bottom-left."""
        return [
            Point(self.x, self.y),
            Point(self.x + self.width, self.y),
            Point(self.x + self.width, self.y + self.height),
            Point(self.x, self.y + self.height),
        ]


@dataclass(frozen=True)
class ClipPath:
    """Arbitrary vector path clipping boundary defined in entity-local coordinates.
    
    Attributes:
        points: Ordered sequence of local vertices forming a closed clipping boundary.
    """
    points: tuple[Point, ...]

    def __init__(self, points: Sequence[Point | tuple[float, float]]):
        if not isinstance(points, (list, tuple)):
            raise ValidationError(f"ClipPath points must be a sequence, got {type(points).__name__}")
        if len(points) < 3:
            raise ValidationError(f"ClipPath requires at least 3 points, got {len(points)}")

        norm_pts: list[Point] = []
        for idx, pt in enumerate(points):
            if isinstance(pt, Point):
                norm_pts.append(pt)
            elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                norm_pts.append(Point(pt[0], pt[1]))
            else:
                raise ValidationError(f"Point at index {idx} must be a Point or (x, y) tuple, got {pt}")

        object.__setattr__(self, "points", tuple(norm_pts))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "polygon",
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipPath:
        """Construct a ClipPath from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"ClipPath data must be a dict, got {type(data).__name__}")
        if "points" not in data:
            raise ValidationError("ClipPath data must contain 'points'")
        pts = [Point.from_dict(p) for p in data["points"]]
        return cls(pts)

    @property
    def bounds(self) -> BoundingBox:
        """Local bounding box of the clip path."""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)


# -----------------------------------------------------------------------------
# Retained Path Helper & Serialization Codecs
# -----------------------------------------------------------------------------

def _is_retained_path(obj: Any) -> bool:
    if obj is None:
        return False
    from drawcv.shapes.path import Path
    return isinstance(obj, Path)


def clip_to_dict(clip: Any | None) -> dict[str, Any] | None:
    """Serialize any supported clip geometry to a canonical plain JSON dictionary.

    Contract:
    Retained Path clips use live Python-object references at runtime. JSON serialization
    stores clip geometry/fill-rule/transform by value. Shared object aliasing between a
    clip and a separately registered scene drawable is therefore not preserved across
    serialization round-trips. Visual and geometric clipping semantics are preserved.
    """
    if clip is None:
        return None
    if isinstance(clip, ClipRect):
        return clip.to_dict()
    if isinstance(clip, ClipPath):
        return clip.to_dict()
    if _is_retained_path(clip):
        from drawcv.shapes.path import serialize_subpaths
        return {
            "type": "retained_path",
            "subpaths": serialize_subpaths(clip.subpaths),
            "fill_rule": clip.fill_rule.value if hasattr(clip.fill_rule, "value") else str(clip.fill_rule),
            "transform": clip.transform.to_dict(),
        }
    if hasattr(clip, "to_dict"):
        return clip.to_dict()
    raise ValidationError(f"Unsupported clip object for serialization: {type(clip).__name__}")


def clip_from_dict(data: dict[str, Any] | None) -> Any | None:
    """Deserialize a ClipRect, legacy ClipPath, or retained Path clip from dictionary representation."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValidationError(f"Clip data must be a dict or None, got {type(data).__name__}")

    clip_type = data.get("type", "rect")
    if clip_type == "rect":
        return ClipRect.from_dict(data)
    elif clip_type == "polygon":
        return ClipPath.from_dict(data)
    elif clip_type == "retained_path":
        from drawcv.shapes.path import Path, deserialize_subpaths
        from drawcv.core.transform import Transform
        subpaths = deserialize_subpaths(data.get("subpaths", []))
        fill_rule_val = FillRule(data["fill_rule"]) if "fill_rule" in data else FillRule.NON_ZERO
        tf = Transform.from_dict(data.get("transform", {}))
        return Path(subpaths=subpaths, fill_rule=fill_rule_val, transform=tf)
    elif clip_type == "path":
        # Defensive check for unmigrated 1.6 documents or direct dicts
        if "points" in data:
            return ClipPath.from_dict(data)
        elif "subpaths" in data:
            from drawcv.shapes.path import Path, deserialize_subpaths
            from drawcv.core.transform import Transform
            subpaths = deserialize_subpaths(data.get("subpaths", []))
            fill_rule_val = FillRule(data["fill_rule"]) if "fill_rule" in data else FillRule.NON_ZERO
            tf = Transform.from_dict(data.get("transform", {})) if "transform" in data else Transform()
            return Path(subpaths=subpaths, fill_rule=fill_rule_val, transform=tf)
        else:
            raise ValidationError("Ambiguous 'path' clip dictionary missing both 'points' and 'subpaths'")
    raise ValidationError(f"Unknown clip type: '{clip_type}'")


# -----------------------------------------------------------------------------
# Live-Identity Snapshot & Restoration Helpers
# -----------------------------------------------------------------------------

def capture_clip_state(clip: Any | None) -> Any:
    """Capture an in-place semantic state snapshot of an attached clip boundary."""
    if clip is None:
        return None
    if isinstance(clip, (ClipRect, ClipPath)):
        return copy.deepcopy(clip)
    if _is_retained_path(clip):
        return (
            clip,
            copy.deepcopy(clip.subpaths),
            clip.fill_rule,
            clip.transform.copy(),
        )
    return copy.deepcopy(clip)


def restore_clip_state(current_clip: Any | None, snapshot: Any) -> Any:
    """Restore clip state while preserving exact original Path reference."""
    if snapshot is None:
        return None
    if isinstance(snapshot, (ClipRect, ClipPath)):
        return copy.deepcopy(snapshot)
    if isinstance(snapshot, tuple) and len(snapshot) == 4:
        target_path, subpaths, fill_rule, transform = snapshot
        target_path.subpaths = copy.deepcopy(subpaths)
        target_path.fill_rule = fill_rule
        target_path.transform = transform.copy()
        return target_path
    return copy.deepcopy(snapshot)


# -----------------------------------------------------------------------------
# Centralized Clip Geometry Evaluator
# -----------------------------------------------------------------------------

def get_clip_point_mapper(clip: Any, owner: Any) -> Callable[[Point], Point]:
    """Return a point mapping function transforming clip-local coordinates to world coordinates.

    Sequential contract:
    clip geometry -> clip-local transform -> owner's sequential to_world() -> world
    Explicitly ignores clip's own _parent / scene tree attachments.
    """
    if _is_retained_path(clip):
        pivot = clip.transform.pivot or clip.get_geometry_bounds().center
        def map_retained_point(p: Point) -> Point:
            local = clip.transform.transform_point(p, default_pivot=pivot)
            if hasattr(owner, "to_world") and callable(owner.to_world):
                return owner.to_world(local)
            return local
        return map_retained_point
    else:
        def map_legacy_point(p: Point) -> Point:
            if hasattr(owner, "to_world") and callable(owner.to_world):
                return owner.to_world(p)
            return p
        return map_legacy_point


def evaluate_clip_contours(clip: Any, owner: Any, tolerance: float = 0.5) -> tuple[list[list[Point]], FillRule]:
    """Extract world-space closed contours and fill rule for any supported clip geometry."""
    mapper = get_clip_point_mapper(clip, owner)
    if isinstance(clip, ClipRect):
        pts = [mapper(c) for c in clip.corners]
        return ([pts], FillRule.EVEN_ODD)
    elif isinstance(clip, ClipPath):
        pts = [mapper(p) for p in clip.points]
        return ([pts], FillRule.EVEN_ODD)
    elif _is_retained_path(clip):
        if not clip.subpaths:
            return ([], clip.fill_rule)
        raw_contours = clip.flatten_with_mapper(mapper, tolerance=tolerance)
        closed_contours: list[list[Point]] = []
        for c in raw_contours:
            if not c or len(c) < 3:
                continue
            contour = list(c)
            if contour[0] != contour[-1]:
                contour.append(contour[0])
            closed_contours.append(contour)
        return (closed_contours, clip.fill_rule)
    return ([], FillRule.EVEN_ODD)


def evaluate_clip_coverage(clip: Any, owner: Any, width: int, height: int, tolerance: float = 0.5) -> np.ndarray:
    """Generate a float32 [0.0, 1.0] alpha coverage mask on canvas coordinates."""
    if width <= 0 or height <= 0:
        return np.zeros((max(0, height), max(0, width)), dtype=np.float32)

    if isinstance(clip, (ClipRect, ClipPath)):
        clip_mask = np.zeros((height, width), dtype=np.uint8)
        mapper = get_clip_point_mapper(clip, owner)
        points = clip.corners if isinstance(clip, ClipRect) else clip.points
        pts_world = [mapper(p) for p in points]
        pts = np.array([[int(round(p.x)), int(round(p.y))] for p in pts_world], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(clip_mask, [pts], 255)
        return clip_mask.astype(np.float32) / 255.0

    elif _is_retained_path(clip):
        contours, fill_rule = evaluate_clip_contours(clip, owner, tolerance=tolerance)
        if not contours:
            return np.zeros((height, width), dtype=np.float32)
        mask_u8 = evaluate_fill_rule_mask(contours, fill_rule, width, height, supersample=2)
        return mask_u8.astype(np.float32) / 255.0

    return np.zeros((height, width), dtype=np.float32)


def is_point_in_clip(clip: Any, owner: Any, world_point: Point) -> bool:
    """Test whether a world-space point lies inside the clip boundary."""
    if isinstance(clip, ClipRect):
        if hasattr(owner, "to_local") and callable(owner.to_local):
            local_pt = owner.to_local(world_point)
        else:
            local_pt = world_point
        return (clip.x <= local_pt.x <= clip.x + clip.width and
                clip.y <= local_pt.y <= clip.y + clip.height)

    elif isinstance(clip, ClipPath):
        if hasattr(owner, "to_local") and callable(owner.to_local):
            local_pt = owner.to_local(world_point)
        else:
            local_pt = world_point
        return point_in_polygon(local_pt, list(clip.points), include_boundary=True)

    elif _is_retained_path(clip):
        contours, fill_rule = evaluate_clip_contours(clip, owner, tolerance=0.5)
        if not contours:
            return False
        if fill_rule == FillRule.EVEN_ODD:
            inside_count = sum(1 for c in contours if len(c) >= 3 and point_in_polygon(world_point, c, include_boundary=True))
            return inside_count % 2 == 1
        else:  # FillRule.NON_ZERO
            winding = 0
            for c in contours:
                if len(c) >= 3 and point_in_polygon(world_point, c, include_boundary=True):
                    area = sum(c[i].x * c[(i + 1) % len(c)].y - c[(i + 1) % len(c)].x * c[i].y for i in range(len(c)))
                    winding += 1 if area >= 0 else -1
            return winding != 0

    return True
