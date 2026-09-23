"""Text labels laid out inside or around a host drawable's box."""
from __future__ import annotations

import copy
import math
from typing import Any

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform

ALIGNS = ("left", "center", "right")
VERTICAL_ALIGNS = ("top", "middle", "bottom")
PLACEMENTS = ("inside", "above", "below", "left", "right")


class Label:
    """Retained text attached to a drawable and laid out against its local box.

    The box defaults to the host's local geometry bounds for inside placement and
    its stroke-inclusive local bounds outside, so captions clear the stroke. The
    label follows every host edit and transform. ``placement='inside'`` aligns
    the text within the box inset by ``padding``; ``above``/``below`` put it
    outside the box using ``align`` along the edge, and ``left``/``right`` use
    ``vertical_align``. Font text wraps to the available width when ``wrap``.

    Styling lives on ``content``, a Text whose position, alignment and wrap
    width are recomputed at each layout. Attributes not defined by Label are
    forwarded to it, so ``label.text``, ``label.color`` and ``label.font_size``
    read and write the text directly.
    """

    _OWN = ("content", "align", "vertical_align", "placement", "padding", "offset", "box", "wrap")

    def __init__(self, text: Any = "", *, align: str = "center", vertical_align: str = "middle",
                 placement: str = "inside", padding: float = 0.0, offset: Point | tuple = Point(0, 0),
                 box: BoundingBox | None = None, wrap: bool = True, **text_options: Any):
        from drawcv.shapes.text import Text
        if isinstance(text, Text):
            if text_options:
                raise ValidationError("Pass text options to the Text, not alongside a Text")
            content = text
        else:
            content = Text(text, **text_options)
        if isinstance(offset, (tuple, list)):
            offset = Point(offset[0], offset[1])
        for name, value in (("content", content), ("align", align), ("vertical_align", vertical_align),
                            ("placement", placement), ("padding", padding), ("offset", offset),
                            ("box", box), ("wrap", wrap)):
            object.__setattr__(self, name, value)
        self._validate()
        object.__setattr__(self, "_initialized", True)

    def __getattr__(self, name: str) -> Any:
        # Only reached for names Label does not define itself.
        if name.startswith("_") or "content" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.content, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in self._OWN:
            if isinstance(value, str) and name in ("align", "vertical_align", "placement"):
                value = value.strip().lower()
            if name == "offset" and isinstance(value, (tuple, list)):
                value = Point(value[0], value[1])
            candidate = copy.copy(self)
            object.__setattr__(candidate, name, value)
            candidate._validate()
            object.__setattr__(self, name, value)
        elif hasattr(self.content, name):
            setattr(self.content, name, value)
        else:
            raise AttributeError(f"Label has no attribute '{name}'")

    def __deepcopy__(self, memo):
        clone = object.__new__(Label)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            object.__setattr__(clone, name, copy.deepcopy(value, memo))
        return clone

    def _validate(self) -> None:
        from drawcv.shapes.text import Text
        if not isinstance(self.content, Text):
            raise ValidationError("Label content must be a Text")
        if self.content.runs is not None or self.content.text_origin != "top_left":
            raise ValidationError("Label text must be plain top_left text without runs")
        if not self.content.transform.is_identity():
            raise ValidationError("Label text must have an identity transform; transform the host instead")
        if self.content.label is not None:
            raise ValidationError("Label text cannot have its own label")
        for name, allowed in (("align", ALIGNS), ("vertical_align", VERTICAL_ALIGNS), ("placement", PLACEMENTS)):
            value = getattr(self, name)
            if isinstance(value, str):
                value = value.strip().lower()
                object.__setattr__(self, name, value)
            if value not in allowed:
                raise ValidationError(f"Label {name} must be one of {', '.join(allowed)}, got {value!r}")
        padding = self.padding
        if isinstance(padding, bool) or not isinstance(padding, (int, float)) or not math.isfinite(padding) or padding < 0:
            raise ValidationError("Label padding must be finite and non-negative")
        if not isinstance(self.offset, Point) or not (math.isfinite(self.offset.x) and math.isfinite(self.offset.y)):
            raise ValidationError("Label offset must be a finite Point")
        if self.box is not None and not isinstance(self.box, BoundingBox):
            raise ValidationError("Label box must be a BoundingBox in host-local coordinates or None")
        if not isinstance(self.wrap, bool):
            raise ValidationError("Label wrap must be a boolean")

    def resolve_box(self, host: Any) -> BoundingBox:
        """The host-local box this label is laid out against."""
        if self.box is not None:
            return self.box
        return host.get_geometry_bounds() if self.placement == "inside" else host.get_local_bounds()

    def layout(self, host: Any, matrix: Any = None) -> Any:
        """Return an independent Text positioned for host; matrix defaults to its world matrix."""
        from drawcv.core.enums import TextAlignment
        self._validate()
        box = self.resolve_box(host)
        pad = float(self.padding)
        text = self.content.clone(new_id=False)
        text.id = f"{host.id}-label"
        text.alignment = TextAlignment(self.align)
        text.position = Point(0, 0)
        if self.wrap and text.fonts is not None and self.placement in ("inside", "above", "below"):
            width = max(box.width - (2 * pad if self.placement == "inside" else 0), 1.0)
            # Wrap only on overflow, so text that fits keeps its natural width.
            if text.wrap_width is None or text.wrap_width > width:
                if text.measure().layout_bounds.width > width:
                    text.wrap_width = width
        size = text.measure().paragraph_bounds if text.fonts is not None else text.get_geometry_bounds()
        w, h = size.width, size.height

        # Padding larger than half the box collapses it onto its center line.
        px, py = min(pad, box.width / 2), min(pad, box.height / 2)
        inner = BoundingBox(box.x + px, box.y + py, box.width - 2 * px, box.height - 2 * py)
        if self.placement == "inside":
            x_box = y_box = inner
        else:
            x_box = y_box = box
        x = {"left": x_box.left, "center": x_box.center.x - w / 2, "right": x_box.right - w}[self.align]
        y = {"top": y_box.top, "middle": y_box.center.y - h / 2, "bottom": y_box.bottom - h}[self.vertical_align]
        if self.placement == "above":
            y = box.top - pad - h
        elif self.placement == "below":
            y = box.bottom + pad
        elif self.placement == "left":
            x = box.left - pad - w
        elif self.placement == "right":
            x = box.right + pad

        text.position = Point(x - size.x + self.offset.x, y - size.y + self.offset.y)
        text.transform = Transform.from_matrix(host.world_matrix if matrix is None else matrix)
        return text

    def to_dict(self) -> dict[str, Any]:
        self._validate()
        return {
            "content": self.content.to_dict(),
            "align": self.align,
            "vertical_align": self.vertical_align,
            "placement": self.placement,
            "padding": float(self.padding),
            "offset": self.offset.to_dict(),
            "box": self.box.to_dict() if self.box is not None else None,
            "wrap": self.wrap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Label:
        from drawcv.shapes.text import Text
        box = data.get("box")
        return cls(Text.from_dict(data["content"]), align=data.get("align", "center"),
                   vertical_align=data.get("vertical_align", "middle"),
                   placement=data.get("placement", "inside"), padding=data.get("padding", 0.0),
                   offset=Point.from_dict(data.get("offset", {"x": 0, "y": 0})),
                   box=BoundingBox.from_dict(box) if box is not None else None,
                   wrap=data.get("wrap", True))

    def __repr__(self) -> str:
        return (f"Label({self.content.text!r}, align={self.align!r}, vertical_align={self.vertical_align!r}, "
                f"placement={self.placement!r}, padding={self.padding!r})")


def label_instance(host: Any, matrix: Any = None) -> Any:
    """Independent positioned Text for host's label, or None."""
    label = getattr(host, "label", None)
    if label is None:
        return None
    if not isinstance(label, Label):
        raise ValidationError("Drawable label must be a Label or None")
    return label.layout(host, matrix)
