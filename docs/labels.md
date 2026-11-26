# Shape labels

A `Label` is text attached to a drawable. It is laid out against the host's box
each time the scene renders, so it stays aligned when the shape moves, resizes,
rotates or animates.

```python
from drawcv import Rectangle, Point, Label, Color, FillStyle

box = Rectangle(position=Point(40, 30), width=200, height=120,
                fill=FillStyle(color=Color(226, 236, 248)))
box.label = Label("Centered", color=Color(20, 40, 70), font_scale=0.9)  # both axes by default

box.label.vertical_align = "bottom"
box.label.padding = 10
box.label.text = "Changed"
box.label.color = Color(200, 0, 0)
box.label = None  # remove it
```

Any drawable except `Text` can carry a label: shapes, paths, arrows, freehand
strokes, images and groups. [examples/shape_labels.py](../examples/shape_labels.py)
renders every option.

## Layout options

| Option | Values | Default |
| --- | --- | --- |
| `align` | `"left"`, `"center"`, `"right"` | `"center"` |
| `vertical_align` | `"top"`, `"middle"`, `"bottom"` | `"middle"` |
| `placement` | `"inside"`, `"above"`, `"below"`, `"left"`, `"right"` | `"inside"` |
| `padding` | Gap in local units; the box inset inside, the distance from the edge outside | `0` |
| `offset` | Final `Point` nudge in local units | `(0, 0)` |
| `box` | `BoundingBox` in host-local coordinates | geometry bounds inside; stroke-inclusive bounds outside |
| `wrap` | Wrap font text that overflows the available width | `True` |

`placement="inside"` aligns the text within the box inset by `padding` on both
axes. `above` and `below` place it outside the box and use `align` along that
edge. `left` and `right` place it beside the box and use `vertical_align`.
Outside placements measure `padding` from the outer edge of the stroke, so a
caption above a thick line never touches it. `align` also aligns the lines of
multi-line text with each other.

Use `box` to lay out against part of a shape, for example the lower strip of a card:
`Label("footer", box=BoundingBox(40, 110, 200, 40))`.

## Text styling

Every other `Text` option passes through the constructor and is available as an
attribute: Hershey `font_scale`/`thickness`/`font_family`, or retained font
`fonts`/`font_size`/`font_family_name`/`line_spacing`, plus `color`, `opacity`,
`background_fill` and so on. `padding` always means the label's layout padding;
to pad a background plate, pass a prepared Text, which works for any style:
`Label(Text("Hi", background_fill=Color(255, 240, 200), padding=4), align="left")`.
The label controls that Text's position, alignment and wrap width; it must use
the default `text_origin="top_left"`, no runs and an identity transform.

Font text wraps to the available width only when it would otherwise overflow.
Hershey text has no wrapping; use `\n` for explicit lines.

## Vertical centering

The label centers its line box, not its ink. This matches CSS and keeps text from
jumping when its characters change. With retained fonts, the line box is the
font's ascent plus descent. For Hershey text, the OpenCV-reported line box leaves
extra space above capital letters, so capitals sit a few pixels low. Use
`offset` for optical adjustments.

## Transforms, bounds and compositing

The label uses the host's complete world transform, like text inside an SVG
group. Rotation, translation, skew and scale all apply, and scaling the host
scales its text. Change `width`/`height` or the geometry instead of
`transform.scale` when the text size should stay fixed.

The label is part of the host's compositing unit, like path markers. Host opacity
applies once to the shape and label together, and the host's clip, mask, effects
and blend mode apply to its label. It renders after the host's own drawing, so it
sits on top of the fill and stroke.

`get_geometry_bounds()` and `get_bounds()` exclude the label, so alignment and
placement helpers keep working on the shape itself. `get_effect_bounds()` and
group/layer visual bounds include it, so an outside caption is never cropped.
During progressive reveal the label is laid out against the complete shape.

## Persistence, history and animation

Labels round-trip through JSON (schema **1.14**), `clone()`, `scene.edit` undo/redo
and observational rendering. Unlabeled drawables serialize exactly as before.
Numeric and color label properties animate by path:

```python
scene.animate(box, "label.padding", 0, 30, duration=1)
scene.animate(box, "label.color", Color(0, 0, 0), Color(200, 0, 0), duration=1)
```

## SVG export

Font labels with a `font_family_name` export as editable `<text>` inside the host
group and share its opacity and clip. Wrapped and explicit lines use positioned
`<tspan>` elements. Hershey labels and unsupported text styles use a reported PNG
fallback with entity ID `"<host id>-label"`. Imported SVG text becomes a separate
`Text` object.
