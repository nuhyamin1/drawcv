"""Retained-mode typography Text drawable for DrawCV."""

from __future__ import annotations
import copy
import math
from typing import Any
import uuid

from dataclasses import dataclass
from enum import Enum

from drawcv.core.bounds import BoundingBox
from drawcv.core.color import Color
from drawcv.core.drawable import Drawable
from drawcv.core.enums import FontFamily, TextAlignment
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.geometry_utils import measure_text_size
from drawcv.core.transform import Transform
from drawcv.styles.fill import FillStyle
from drawcv.typography import FontAsset, TextMetrics, TextLineMetrics


class TextAnchor(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


@dataclass(frozen=True)
class TextRun:
    """Immutable rich text run with optional per-span style and positioning overrides."""
    text: str
    fonts: tuple[FontAsset, ...] | None = None
    font_size: float | None = None
    font_family_name: str | None = None
    font_weight: str | int | None = None
    font_style: str | None = None
    fill: Color | None = None
    fill_none: bool = False
    fill_opacity: float | None = None
    x: float | None = None
    y: float | None = None
    dx: float = 0.0
    dy: float = 0.0
    text_anchor: TextAnchor | None = None
    direction: str | None = None
    xml_space: str | None = None
    is_font_substituted: bool = False

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValidationError(f"TextRun 'text' must be a string, got {type(self.text).__name__}")
        if self.fonts is not None:
            if isinstance(self.fonts, (list, tuple)):
                object.__setattr__(self, "fonts", tuple(self.fonts))
            if not self.fonts or not all(isinstance(f, FontAsset) for f in self.fonts):
                raise ValidationError("TextRun 'fonts' must be None or a nonempty sequence of FontAsset instances")
        if self.font_size is not None:
            if isinstance(self.font_size, bool) or not isinstance(self.font_size, (int, float)) or not math.isfinite(self.font_size) or self.font_size <= 0:
                raise ValidationError(f"TextRun font_size must be positive and finite, got {self.font_size}")
            if self.font_size > 4096:
                raise ValidationError("TextRun font_size cannot exceed 4096px")
        if self.fill is not None and not isinstance(self.fill, Color):
            raise ValidationError(f"TextRun fill must be Color or None, got {type(self.fill).__name__}")
        if not isinstance(self.fill_none, bool):
            raise ValidationError("TextRun fill_none must be a boolean")
        if self.fill_opacity is not None:
            if isinstance(self.fill_opacity, bool) or not isinstance(self.fill_opacity, (int, float)) or not (0.0 <= float(self.fill_opacity) <= 1.0):
                raise ValidationError(f"TextRun fill_opacity must be in range [0, 1], got {self.fill_opacity}")
        if self.text_anchor is not None:
            if isinstance(self.text_anchor, str):
                try:
                    object.__setattr__(self, "text_anchor", TextAnchor(self.text_anchor.strip().lower()))
                except ValueError:
                    raise ValidationError(f"Invalid TextAnchor '{self.text_anchor}'")
            elif not isinstance(self.text_anchor, TextAnchor):
                raise ValidationError(f"TextRun text_anchor must be TextAnchor or str, got {type(self.text_anchor).__name__}")
        if self.direction is not None and self.direction not in ("auto", "ltr", "rtl"):
            raise ValidationError("TextRun direction must be auto, ltr, or rtl")
        if self.xml_space is not None and self.xml_space not in ("default", "preserve"):
            raise ValidationError(f"TextRun xml_space must be 'default', 'preserve', or None, got '{self.xml_space}'")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text}
        if self.fonts is not None:
            d["fonts"] = [f.to_dict() for f in self.fonts]
        if self.font_size is not None:
            d["font_size"] = self.font_size
        if self.font_family_name is not None:
            d["font_family_name"] = self.font_family_name
        if self.font_weight is not None:
            d["font_weight"] = self.font_weight
        if self.font_style is not None:
            d["font_style"] = self.font_style
        if self.fill is not None:
            d["fill"] = self.fill.to_dict()
        if self.fill_none:
            d["fill_none"] = True
        if self.fill_opacity is not None:
            d["fill_opacity"] = self.fill_opacity
        if self.x is not None:
            d["x"] = self.x
        if self.y is not None:
            d["y"] = self.y
        if self.dx != 0.0:
            d["dx"] = self.dx
        if self.dy != 0.0:
            d["dy"] = self.dy
        if self.text_anchor is not None:
            d["text_anchor"] = self.text_anchor.value
        if self.direction is not None:
            d["direction"] = self.direction
        if self.xml_space is not None:
            d["xml_space"] = self.xml_space
        if self.is_font_substituted:
            d["is_font_substituted"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextRun:
        fonts = tuple(FontAsset.from_dict(f) for f in data["fonts"]) if data.get("fonts") is not None else None
        fill = Color.from_dict(data["fill"]) if data.get("fill") is not None else None
        ta = TextAnchor(data["text_anchor"]) if data.get("text_anchor") is not None else None
        f_op = float(data["fill_opacity"]) if "fill_opacity" in data and data["fill_opacity"] is not None else None
        return cls(
            text=str(data.get("text", "")),
            fonts=fonts,
            font_size=data.get("font_size"),
            font_family_name=data.get("font_family_name"),
            font_weight=data.get("font_weight"),
            font_style=data.get("font_style"),
            fill=fill,
            fill_none=bool(data.get("fill_none", False)),
            fill_opacity=f_op,
            x=data.get("x"),
            y=data.get("y"),
            dx=float(data.get("dx", 0.0)),
            dy=float(data.get("dy", 0.0)),
            text_anchor=ta,
            direction=data.get("direction"),
            xml_space=data.get("xml_space"),
            is_font_substituted=bool(data.get("is_font_substituted", False)),
        )


class Text(Drawable):
    """Retained-mode vector typography text drawable.
    
    Supports Hershey fonts, multi-line strings, alignment options,
    exact font metrics, rich positioned runs, and background plates (rectangular or rounded).
    """

    def __init__(
        self,
        text: str = "",
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
        fonts: tuple[FontAsset, ...] | None = None,
        font_size: float = 32,
        font_family_name: str | None = None,
        font_weight: str | int | None = None,
        font_style: str | None = None,
        runs: tuple[TextRun, ...] | list[TextRun] | None = None,
        text_origin: str = "top_left",
        text_anchor: TextAnchor | str | None = None,
        fill_opacity: float = 1.0,
        fill_none: bool = False,
        is_font_substituted: bool = False,
        wrap_width: float | None = None,
        line_spacing: float = 1.2,
        direction: str = "auto",
        xml_space: str = "default",
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
        timing: Any | None = None,
        render_progress: float = 1.0,
        **kwargs: Any,
    ):
        super_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "locked": locked,
            "opacity": opacity,
            "z_index": z_index,
            "clip": clip,
            "mask": mask,
            "timing": timing,
            "render_progress": float(render_progress),
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

        if runs is not None:
            runs_tuple = tuple(runs)
            if not text:
                text = "".join(r.text for r in runs_tuple)
            self.runs: tuple[TextRun, ...] | None = runs_tuple
        else:
            self.runs = None

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
        self.fonts = fonts
        self.font_size = font_size
        self.font_family_name = font_family_name
        self.font_weight = font_weight
        self.font_style = font_style
        self.text_origin = text_origin
        self.text_anchor = TextAnchor(text_anchor.strip().lower()) if isinstance(text_anchor, str) else text_anchor
        self.fill_opacity = float(fill_opacity)
        self.fill_none = bool(fill_none)
        self.is_font_substituted = bool(is_font_substituted)
        self.wrap_width = wrap_width
        self.line_spacing = line_spacing
        self.direction = direction
        if xml_space not in ("default", "preserve"):
            raise ValidationError(f"Text xml_space must be 'default' or 'preserve', got '{xml_space}'")
        self.xml_space = xml_space
        self._validate_font_options()
        object.__setattr__(self, "_font_initialized", True)

    def __setattr__(self, name, value):
        if name == "fonts" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if name == "runs" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if getattr(self, "_font_initialized", False) and name in (
            "fonts", "font_size", "font_family_name", "font_weight", "font_style",
            "runs", "text_origin", "text_anchor", "fill_opacity", "fill_none", "is_font_substituted",
            "wrap_width", "line_spacing", "direction", "xml_space", "text"
        ):
            candidate = copy.copy(self)
            object.__setattr__(candidate, name, value)
            candidate._validate_font_options()
        object.__setattr__(self, name, value)

    def _validate(self) -> None:
        super()._validate()
        self._validate_font_options()

    def _validate_font_options(self):
        if not isinstance(self.text, str):
            raise ValidationError("Text content must be a string")
        if self.text_origin not in ("top_left", "baseline"):
            raise ValidationError(f"text_origin must be 'top_left' or 'baseline', got '{self.text_origin}'")
        if self.text_anchor is not None:
            if isinstance(self.text_anchor, str):
                try:
                    object.__setattr__(self, "text_anchor", TextAnchor(self.text_anchor.strip().lower()))
                except ValueError:
                    raise ValidationError(f"Invalid TextAnchor '{self.text_anchor}'")
            elif not isinstance(self.text_anchor, TextAnchor):
                raise ValidationError(f"text_anchor must be TextAnchor or str, got {type(self.text_anchor).__name__}")
        if self.runs is not None:
            if not isinstance(self.runs, tuple) or not all(isinstance(r, TextRun) for r in self.runs):
                raise ValidationError("runs must be None or a sequence of TextRun instances")
        if isinstance(self.fill_opacity, bool) or not isinstance(self.fill_opacity, (int, float)) or not (0.0 <= float(self.fill_opacity) <= 1.0):
            raise ValidationError(f"fill_opacity must be in range [0, 1], got {self.fill_opacity}")
        if not isinstance(self.fill_none, bool):
            raise ValidationError("fill_none must be a boolean")
        if self.fonts is not None and (not isinstance(self.fonts, tuple) or not self.fonts
                                     or not all(isinstance(f, FontAsset) for f in self.fonts)):
            raise ValidationError("fonts must be None or a nonempty sequence of FontAsset values")
        for name, value in (("font_size", self.font_size), ("line_spacing", self.line_spacing), ("wrap_width", self.wrap_width)):
            if name == "wrap_width" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be positive and finite")
        if self.line_spacing < 1 or self.font_size > 4096:
            raise ValidationError("line_spacing must be >= 1 and font_size <= 4096px")
        if self.direction not in ("auto", "ltr", "rtl"):
            raise ValidationError("direction must be auto, ltr or rtl")
        if self.xml_space not in ("default", "preserve"):
            raise ValidationError("xml_space must be 'default' or 'preserve'")

    def _font_layout(self):
        self._validate_font_options()
        from drawcv.typography.layout import layout_text, layout_text_runs
        runs = self.runs
        if runs is not None or self.text_origin == "baseline":
            if runs is None:
                runs = (TextRun(
                    text=self.text,
                    fonts=self.fonts,
                    font_size=self.font_size,
                    font_family_name=self.font_family_name,
                    font_weight=self.font_weight,
                    font_style=self.font_style,
                    fill=None,
                    fill_none=self.fill_none,
                    fill_opacity=None,
                    text_anchor=self.text_anchor,
                    direction=self.direction,
                    xml_space=self.xml_space,
                    is_font_substituted=self.is_font_substituted,
                ),)
            return layout_text_runs(
                runs,
                self.fonts,
                self.font_size,
                self.wrap_width,
                self.alignment.value,
                self.direction,
                self.line_spacing,
                self.text_origin,
                self.text_anchor.value if self.text_anchor is not None else None,
                self.position.x,
                self.position.y,
                root_fill_opacity=self.fill_opacity,
                root_fill_none=self.fill_none,
            )
        return layout_text(
            self.text,
            self.fonts,
            self.font_size,
            self.wrap_width,
            self.alignment.value,
            self.direction,
            self.line_spacing,
        )

    def measure(self) -> TextMetrics:
        """Measure font text in local coordinates; Hershey retains its metric API."""
        if self.fonts is None and self.runs is None:
            raise ValidationError("measure() requires font text; use get_line_metrics() for Hershey")
        layout = self._font_layout()
        plate_w = layout.width + 2 * self.padding
        if self.runs is not None or self.text_origin == "baseline":
            x = self.position.x
            y = self.position.y
        else:
            x = self.position.x - {TextAlignment.LEFT: 0, TextAlignment.CENTER: plate_w / 2.0,
                                    TextAlignment.RIGHT: plate_w}[self.alignment]
            y = self.position.y
        ox, oy = x + self.padding, y + self.padding

        def shifted(box):
            return BoundingBox(box.x + ox, box.y + oy, box.width, box.height) if box is not None else None

        return TextMetrics(
            max((l.advance_width for l in layout.lines), default=0),
            BoundingBox(ox, oy, layout.width, layout.height),
            BoundingBox(x, y, plate_w, layout.height + 2 * self.padding),
            shifted(layout.ink),
            tuple(TextLineMetrics(l.text, l.advance_width, l.baseline + oy,
                                  shifted(l.line_box), shifted(l.ink_bounds)) for l in layout.lines),
        )

    # -------------------------------------------------------------------------

    def get_line_metrics(self) -> list[tuple[str, int, int, int]]:
        """Return list of (line_text, width, height, baseline) for each line."""
        if self.fonts is not None:
            layout = self._font_layout()
            return [(line.text, line.advance_width, layout.ascent, layout.descent) for line in layout.lines]
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
        if self.fonts is not None:
            layout = self._font_layout()
            return layout.width, layout.height, layout.ascent + layout.descent
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
        if self.fonts is not None or self.runs is not None:
            metrics = self.measure()
            return metrics.paragraph_bounds.union(metrics.ink_bounds) if metrics.ink_bounds else metrics.paragraph_bounds
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
        if self.fonts is not None or self.runs is not None:
            # Bilinear resampling can contribute one local pixel beyond ink.
            gb = BoundingBox(gb.x-1, gb.y-1, gb.width+2, gb.height+2)
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
            fonts=self.fonts,
            font_size=self.font_size,
            font_family_name=self.font_family_name,
            font_weight=self.font_weight,
            font_style=self.font_style,
            runs=copy.deepcopy(self.runs),
            text_origin=self.text_origin,
            text_anchor=self.text_anchor,
            fill_opacity=self.fill_opacity,
            fill_none=self.fill_none,
            is_font_substituted=self.is_font_substituted,
            wrap_width=self.wrap_width,
            line_spacing=self.line_spacing,
            direction=self.direction,
            xml_space=self.xml_space,
            timing=copy.deepcopy(self.timing),
            render_progress=self.render_progress,
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

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "fonts": self.fonts,
            "font_size": self.font_size,
            "font_family_name": self.font_family_name,
            "font_weight": self.font_weight,
            "font_style": self.font_style,
            "runs": copy.deepcopy(self.runs),
            "text_origin": self.text_origin,
            "text_anchor": self.text_anchor.value if hasattr(self.text_anchor, "value") else str(self.text_anchor) if self.text_anchor is not None else None,
            "fill_opacity": float(self.fill_opacity),
            "fill_none": bool(self.fill_none),
            "is_font_substituted": bool(self.is_font_substituted),
            "wrap_width": self.wrap_width,
            "line_spacing": self.line_spacing,
            "direction": self.direction,
            "xml_space": self.xml_space,
            "text": self.text,
            "position": self.position.copy(),
            "color": self.color.copy(),
            "font_family": self.font_family.value if hasattr(self.font_family, "value") else str(self.font_family),
            "font_scale": float(self.font_scale),
            "thickness": int(self.thickness),
            "alignment": self.alignment.value if hasattr(self.alignment, "value") else str(self.alignment),
            "background_fill": self.background_fill.copy() if isinstance(self.background_fill, FillStyle) else None,
            "background_radius": float(self.background_radius),
            "padding": float(self.padding),
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        for name in (
            "fonts", "font_size", "font_family_name", "font_weight", "font_style",
            "runs", "text_origin", "fill_opacity", "fill_none", "is_font_substituted",
            "wrap_width", "line_spacing", "direction", "xml_space"
        ):
            if name in state:
                setattr(self, name, state[name])
        if "text_anchor" in state:
            ta = state["text_anchor"]
            self.text_anchor = TextAnchor(ta) if isinstance(ta, str) else ta
        if "text" in state:
            self.text = str(state["text"])
        if "position" in state:
            self.position = state["position"].copy() if isinstance(state["position"], Point) else Point.from_dict(state["position"])
        if "color" in state:
            c = state["color"]
            self.color = c.copy() if isinstance(c, Color) else Color.from_dict(c)
        if "font_family" in state:
            ff = state["font_family"]
            self.font_family = FontFamily(ff) if isinstance(ff, str) else ff
        if "font_scale" in state:
            self.font_scale = float(state["font_scale"])
        if "thickness" in state:
            self.thickness = int(state["thickness"])
        if "alignment" in state:
            al = state["alignment"]
            self.alignment = TextAlignment(al) if isinstance(al, str) else al
        if "background_fill" in state:
            bf = state["background_fill"]
            self.background_fill = bf.copy() if isinstance(bf, FillStyle) else (FillStyle.from_dict(bf) if bf else None)
        if "background_radius" in state:
            self.background_radius = float(state["background_radius"])
        if "padding" in state:
            self.padding = float(state["padding"])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        res = self._base_to_dict()
        res.update({
            "type": "text",
            "text": self.text,
            "position": self.position.to_dict(),
            "color": self.color.to_dict(),
            "font_family": self.font_family.value if hasattr(self.font_family, "value") else str(self.font_family),
            "font_scale": float(self.font_scale),
            "thickness": int(self.thickness),
            "alignment": self.alignment.value if hasattr(self.alignment, "value") else str(self.alignment),
            "background_fill": self.background_fill.to_dict() if isinstance(self.background_fill, FillStyle) else None,
            "background_radius": float(self.background_radius),
            "padding": float(self.padding),
        })
        if self.fonts is not None or (self.font_size, self.wrap_width, self.line_spacing, self.direction) != (32, None, 1.2, 'auto'):
            res.update(fonts=[f.to_dict() for f in self.fonts] if self.fonts is not None else None, font_size=self.font_size,
                       wrap_width=self.wrap_width, line_spacing=self.line_spacing, direction=self.direction)
        if self.runs is not None:
            res["runs"] = [r.to_dict() for r in self.runs]
        if self.font_family_name is not None:
            res["font_family_name"] = self.font_family_name
        if self.font_weight is not None:
            res["font_weight"] = self.font_weight
        if self.font_style is not None:
            res["font_style"] = self.font_style
        if self.text_origin != "top_left":
            res["text_origin"] = self.text_origin
        if self.text_anchor is not None:
            res["text_anchor"] = self.text_anchor.value if hasattr(self.text_anchor, "value") else str(self.text_anchor)
        if self.fill_opacity != 1.0:
            res["fill_opacity"] = self.fill_opacity
        if self.fill_none:
            res["fill_none"] = True
        if self.xml_space != "default":
            res["xml_space"] = self.xml_space
        if self.is_font_substituted:
            res["is_font_substituted"] = True
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Text:
        """Construct a Text object from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        pos = Point.from_dict(data["position"]) if "position" in data else Point(0.0, 0.0)
        color = Color.from_dict(data["color"]) if "color" in data else Color.black()
        font_family = FontFamily(data["font_family"]) if "font_family" in data else FontFamily.SIMPLEX
        alignment = TextAlignment(data["alignment"]) if "alignment" in data else TextAlignment.LEFT
        bg_fill = FillStyle.from_dict(data["background_fill"]) if data.get("background_fill") is not None else None
        runs = tuple(TextRun.from_dict(r) for r in data["runs"]) if data.get("runs") is not None else None
        ta = TextAnchor(data["text_anchor"]) if data.get("text_anchor") is not None else None
        return cls(
            text=str(data.get("text", "")),
            position=pos,
            color=color,
            font_family=font_family,
            font_scale=float(data.get("font_scale", 1.0)),
            thickness=int(data.get("thickness", 1)),
            alignment=alignment,
            background_fill=bg_fill,
            background_radius=float(data.get("background_radius", 0.0)),
            padding=float(data.get("padding", 0.0)),
            fonts=tuple(FontAsset.from_dict(f) for f in data["fonts"]) if data.get("fonts") is not None else None,
            font_size=data.get("font_size", 32),
            font_family_name=data.get("font_family_name"),
            font_weight=data.get("font_weight"),
            font_style=data.get("font_style"),
            runs=runs,
            text_origin=data.get("text_origin", "top_left"),
            text_anchor=ta,
            fill_opacity=float(data.get("fill_opacity", 1.0)),
            fill_none=bool(data.get("fill_none", False)),
            is_font_substituted=bool(data.get("is_font_substituted", False)),
            wrap_width=data.get("wrap_width"),
            line_spacing=data.get("line_spacing", 1.2),
            direction=data.get("direction", "auto"),
            xml_space=data.get("xml_space", "default"),
            **base_kwargs,
        )
