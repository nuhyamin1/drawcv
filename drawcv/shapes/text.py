"""Retained-mode typography Text drawable for DrawCV."""

from __future__ import annotations
import copy
import math
from typing import Any
import uuid

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import FontFamily, TextAlignment
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import measure_text_size
from drawcv.core.transform import Transform
from drawcv.styles.fill import FillStyle


class Text(Drawable):
    """Retained-mode vector typography text drawable.
    
    Supports Hershey fonts, multi-line strings, alignment options,
    exact font metrics, and background plates (rectangular or rounded).
    """

    def __init__(
        self,
        text: str,
        position: Point | tuple[float, float] = Point(0, 0),
        color: Color = Color.black(),
        font_family: FontFamily = FontFamily.SIMPLEX,
        font_scale: float = 1.0,
        thickness: int = 1,
        alignment: TextAlignment = TextAlignment.LEFT,
        background_fill: FillStyle | Color | None = None,
        background_radius: float = 0.0,
        padding: float = 0.0,
        *,
        id: str | None = None,
        name: str | None = None,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
        z_index: int = 0,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform: Transform | None = None,
        clip: Any | None = None,
        mask: Any | None = None,
        effects: list[Any] | None = None,
    ):
        super_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "locked": locked,
            "opacity": opacity,
            "z_index": z_index,
            "clip": clip,
            "mask": mask,
        }
        if id is not None:
            super_kwargs["id"] = id
        if tags is not None:
            super_kwargs["tags"] = tags
        if metadata is not None:
            super_kwargs["metadata"] = metadata
        if transform is not None:
            super_kwargs["transform"] = transform
        if effects is not None:
            super_kwargs["effects"] = effects

        super().__init__(**super_kwargs)

        if not isinstance(text, str):
            raise ValidationError(f"Text 'text' must be a string, got {type(text).__name__}")
        self.text = text

        self.position = Point(position[0], position[1]) if isinstance(position, (tuple, list)) else position
        if not isinstance(self.position, Point):
            raise ValidationError(f"Text 'position' must be a Point, got {type(position).__name__}")

        if not isinstance(color, Color):
            raise ValidationError(f"Text 'color' must be a Color, got {type(color).__name__}")
        self.color = color

        if isinstance(font_family, str):
            try:
                self.font_family = FontFamily(font_family.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid FontFamily '{font_family}'")
        elif isinstance(font_family, FontFamily):
            self.font_family = font_family
        else:
            raise ValidationError(f"font_family must be FontFamily, got {type(font_family).__name__}")

        if not isinstance(font_scale, (int, float)) or isinstance(font_scale, bool) or float(font_scale) <= 0.0:
            raise ValidationError(f"font_scale must be a positive float, got {font_scale}")
        self.font_scale = float(font_scale)

        if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness < 1:
            raise ValidationError(f"thickness must be an integer >= 1, got {thickness}")
        self.thickness = int(thickness)

        if isinstance(alignment, str):
            try:
                self.alignment = TextAlignment(alignment.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid TextAlignment '{alignment}'")
        elif isinstance(alignment, TextAlignment):
            self.alignment = alignment
        else:
            raise ValidationError(f"alignment must be TextAlignment, got {type(alignment).__name__}")

        if isinstance(background_fill, Color):
            self.background_fill: FillStyle | None = FillStyle(color=background_fill)
        elif isinstance(background_fill, FillStyle) or background_fill is None:
            self.background_fill = background_fill
        else:
            raise ValidationError(f"background_fill must be FillStyle, Color, or None, got {type(background_fill).__name__}")

        if not isinstance(background_radius, (int, float)) or isinstance(background_radius, bool) or float(background_radius) < 0.0:
            raise ValidationError(f"background_radius must be a non-negative float, got {background_radius}")
        self.background_radius = float(background_radius)

        if not isinstance(padding, (int, float)) or isinstance(padding, bool) or float(padding) < 0.0:
            raise ValidationError(f"padding must be a non-negative float, got {padding}")
        self.padding = float(padding)

    # -------------------------------------------------------------------------
    # Metrics and Geometry
    # -------------------------------------------------------------------------

    def get_line_metrics(self) -> list[tuple[str, int, int, int]]:
        """Return list of (line_text, width, height, baseline) for each line."""
        lines = self.text.split("\n")
        metrics: list[tuple[str, int, int, int]] = []
        for line in lines:
            if not line:
                # Approximate empty line metrics using a dummy character
                _, h, b = measure_text_size("A", self.font_family, self.font_scale, self.thickness)
                metrics.append(("", 0, h, b))
            else:
                w, h, b = measure_text_size(line, self.font_family, self.font_scale, self.thickness)
                metrics.append((line, w, h, b))
        return metrics

    def get_text_bounds_dimensions(self) -> tuple[float, float, float]:
        """Compute (total_width, total_height, max_line_height)."""
        metrics = self.get_line_metrics()
        if not metrics:
            return 0.0, 0.0, 0.0

        widths = [m[1] for m in metrics]
        heights = [m[2] for m in metrics]
        baselines = [m[3] for m in metrics]

        max_width = float(max(widths)) if widths else 0.0
        max_height = float(max(heights)) if heights else 0.0
        max_baseline = float(max(baselines)) if baselines else 0.0

        if len(metrics) == 1:
            total_height = max_height + max_baseline
        else:
            line_step = max_height * 1.4
            total_height = (len(metrics) - 1) * line_step + max_height + max_baseline

        return max_width, total_height, max_height

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local coordinates (including padding if any)."""
        text_w, text_h, _ = self.get_text_bounds_dimensions()
        plate_w = text_w + 2.0 * self.padding
        plate_h = text_h + 2.0 * self.padding

        if self.alignment == TextAlignment.LEFT:
            plate_x = self.position.x
        elif self.alignment == TextAlignment.CENTER:
            plate_x = self.position.x - plate_w / 2.0
        else:  # TextAlignment.RIGHT
            plate_x = self.position.x - plate_w

        plate_y = self.position.y
        return BoundingBox(plate_x, plate_y, plate_w, plate_h)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds at identity transform."""
        return self.get_geometry_bounds()

    def get_bounds(self) -> BoundingBox:
        """World-space bounding box enclosing transformed text rectangle."""
        gb = self.get_geometry_bounds()
        corners = [
            Point(gb.left, gb.top),
            Point(gb.right, gb.top),
            Point(gb.right, gb.bottom),
            Point(gb.left, gb.bottom),
        ]
        world_pts = [self.to_world(p) for p in corners]
        xs = [p.x for p in world_pts]
        ys = [p.y for p in world_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    # -------------------------------------------------------------------------
    # Anchors, Hit-Testing & Cloning
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve a named geometric anchor point in world coordinates."""
        norm = name.strip().lower().replace("-", "_").replace(" ", "_")
        b = self.get_bounds()

        if norm == "center":
            return b.center
        elif norm == "top":
            return Point(b.left + b.width / 2.0, b.top)
        elif norm == "bottom":
            return Point(b.left + b.width / 2.0, b.bottom)
        elif norm == "left":
            return Point(b.left, b.top + b.height / 2.0)
        elif norm == "right":
            return Point(b.right, b.top + b.height / 2.0)
        elif norm in ("top_left", "topleft"):
            return b.top_left
        elif norm in ("top_right", "topright"):
            return b.top_right
        elif norm in ("bottom_left", "bottomleft"):
            return b.bottom_left
        elif norm in ("bottom_right", "bottomright"):
            return b.bottom_right
        else:
            valid = "center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right"
            raise ValidationError(f"Unknown anchor '{name}' for Text. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test whether a world-space point intersects the text bounding rectangle."""
        if not self.effective_visible:
            return False
        local_pt = self.to_local(world_point)
        gb = self.get_geometry_bounds()
        return (gb.left <= local_pt.x <= gb.right) and (gb.top <= local_pt.y <= gb.bottom)

    def clone(self, new_id: bool = True) -> Text:
        """Create a deep copy of this Text."""
        return Text(
            text=self.text,
            position=Point(self.position.x, self.position.y),
            color=copy.deepcopy(self.color),
            font_family=self.font_family,
            font_scale=self.font_scale,
            thickness=self.thickness,
            alignment=self.alignment,
            background_fill=copy.deepcopy(self.background_fill),
            background_radius=self.background_radius,
            padding=self.padding,
            id=str(uuid.uuid4()) if new_id else self.id,
            name=self.name,
            visible=self.visible,
            locked=self.locked,
            opacity=self.opacity,
            z_index=self.z_index,
            tags=set(self.tags),
            metadata=copy.deepcopy(self.metadata),
            transform=copy.deepcopy(self.transform),
            clip=copy.deepcopy(self.clip),
            mask=copy.deepcopy(self.mask),
            effects=copy.deepcopy(self.effects),
        )
