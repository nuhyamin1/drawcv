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
from drawcv.typography import FontAsset, TextMetrics, TextLineMetrics


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
        fonts: tuple[FontAsset, ...] | None = None,
        font_size: float = 32,
        wrap_width: float | None = None,
        line_spacing: float = 1.2,
        direction: str = "auto",
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
        self.wrap_width = wrap_width
        self.line_spacing = line_spacing
        self.direction = direction
        self._validate_font_options()
        object.__setattr__(self, "_font_initialized", True)

    def __setattr__(self, name, value):
        if name == "fonts" and isinstance(value, (list, tuple)):
            value = tuple(value)
        if getattr(self, "_font_initialized", False) and name in (
            "fonts", "font_size", "wrap_width", "line_spacing", "direction", "text"
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

    def _font_layout(self):
        self._validate_font_options()
        from drawcv.typography.layout import layout_text
        return layout_text(self.text, self.fonts, self.font_size, self.wrap_width,
                           self.alignment.value, self.direction, self.line_spacing)

    def measure(self) -> TextMetrics:
        """Measure font text in local coordinates; Hershey retains its metric API."""
        if self.fonts is None:
            raise ValidationError("measure() requires font text; use get_line_metrics() for Hershey")
        layout = self._font_layout()
        plate_w = layout.width + 2*self.padding
        x = self.position.x - {TextAlignment.LEFT: 0, TextAlignment.CENTER: plate_w/2,
                                TextAlignment.RIGHT: plate_w}[self.alignment]
        y = self.position.y
        ox, oy = x+self.padding, y+self.padding
        def shifted(box):
            return BoundingBox(box.x+ox, box.y+oy, box.width, box.height) if box is not None else None
        return TextMetrics(max((l.advance_width for l in layout.lines), default=0),
            BoundingBox(ox, oy, layout.width, layout.height),
            BoundingBox(x, y, plate_w, layout.height+2*self.padding), shifted(layout.ink),
            tuple(TextLineMetrics(l.text, l.advance_width, l.baseline+oy,
                                  shifted(l.line_box), shifted(l.ink_bounds)) for l in layout.lines))


    # -------------------------------------------------------------------------
    # Metrics and Geometry
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
        if self.fonts is not None:
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
        if self.fonts is not None:
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
            fonts=self.fonts, font_size=self.font_size, wrap_width=self.wrap_width,
            line_spacing=self.line_spacing, direction=self.direction,
            timing=copy.deepcopy(self.timing), render_progress=self.render_progress,
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
            "fonts": self.fonts, "font_size": self.font_size, "wrap_width": self.wrap_width,
            "line_spacing": self.line_spacing, "direction": self.direction,
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
        for name in ("fonts", "font_size", "wrap_width", "line_spacing", "direction"):
            if name in state:
                setattr(self, name, state[name])
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
            font_size=data.get("font_size", 32), wrap_width=data.get("wrap_width"),
            line_spacing=data.get("line_spacing", 1.2), direction=data.get("direction", "auto"),
            **base_kwargs,
        )

