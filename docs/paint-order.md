# Paint order

`paint_order` sets the sequence a shape's fill, stroke and markers are drawn in,
like SVG `paint-order`.

```python
from drawcv import Rectangle, Point, FillStyle, StrokeStyle, Color

badge = Rectangle(position=Point(20, 20), width=120, height=60,
                  fill=FillStyle(color=Color(250, 200, 40)),
                  stroke=StrokeStyle(color=Color(30, 30, 30), width=10),
                  paint_order="stroke")   # stroke first, fill on top of its inner half
badge.paint_order = "normal"              # fill, stroke, markers
```

Values follow SVG: `"normal"`, or `fill`, `stroke` and `markers` in any order,
separated by spaces or commas. Steps you leave out follow in normal order, so
`"stroke"` means `"stroke fill markers"` and `"markers"` means
`"markers fill stroke"`. The value is stored normalized, and invalid values raise
`ValidationError` without changing the shape.

With `"stroke"` the fill covers the inner half of the stroke, so only the outer
half shows. This is useful for outlines that should not cut into thin shapes. On
arrows, `fill` is the arrowhead fill and `stroke` is the shaft and head outline.
Markers exist on paths only. Drawables without a fill or stroke ignore the setting.

Paint order is part of JSON (written only when not normal), cloning, undo/redo and
`to_path()`. SVG export writes the `paint-order` attribute; a path whose markers
are not painted last is written as separate fill, marker and stroke elements,
because SVG draws markers only after their path. SVG import keeps `paint-order`,
including inheritance from groups.
