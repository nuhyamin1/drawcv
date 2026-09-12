# Project: Highly Manipulable Drawing Library Built on OpenCV

Build a reusable Python drawing library from scratch using **OpenCV and NumPy as the raster rendering backend**.

The goal is NOT to build a GUI application.

The goal is to build a clean, extensible, object-oriented **2D drawing engine/library** where every drawn object remains addressable, editable, movable, transformable, hideable, reorderable, serializable, and re-renderable after creation.

The library should separate:

1. Scene representation
2. Geometry
3. Appearance/style
4. Transformations
5. Rendering
6. Object management
7. Serialization
8. Optional temporal/animation information

Do not couple the library to PyQt, Tkinter, Flask, browser APIs, or any particular application.

Use Python 3.12+, NumPy, and OpenCV.

---

# 1. Fundamental Architecture

Do NOT treat drawing as immediately modifying pixels permanently.

Use a retained-mode drawing architecture.

Instead of:

```python
cv2.line(image, ...)
```

the public API should create an object:

```python
line = Line(
    start=Point(100, 100),
    end=Point(500, 300),
    stroke=StrokeStyle(...)
)

scene.add(line)
```

The renderer later converts the scene into pixels:

```python
image = renderer.render(scene)
```

This is essential because objects must remain independently manipulable after creation.

Example:

```python
line.move(50, 20)
line.stroke.color = Color.red()
line.stroke.width = 8
line.rotate(15)

image = renderer.render(scene)
```

The original pixels should not be considered authoritative state.

The scene model is authoritative.

---

# 2. Package Structure

Use approximately this structure:

```text
opencv_draw/
│
├── __init__.py
│
├── canvas.py
├── scene.py
├── renderer.py
│
├── core/
│   ├── drawable.py
│   ├── geometry.py
│   ├── color.py
│   ├── transform.py
│   ├── bounds.py
│   └── enums.py
│
├── styles/
│   ├── stroke.py
│   ├── fill.py
│   └── shadow.py
│
├── shapes/
│   ├── line.py
│   ├── arrow.py
│   ├── rectangle.py
│   ├── rounded_rectangle.py
│   ├── circle.py
│   ├── ellipse.py
│   ├── arc.py
│   ├── polygon.py
│   ├── polyline.py
│   ├── bezier.py
│   ├── path.py
│   └── freehand.py
│
├── text/
│   └── text.py
│
├── effects/
│   ├── opacity.py
│   ├── blur.py
│   ├── shadow.py
│   └── mask.py
│
├── animation/
│   ├── timing.py
│   └── interpolation.py
│
├── serialization/
│   ├── json_encoder.py
│   └── json_decoder.py
│
└── tests/
```

The exact structure may be improved if there is a strong architectural reason.

---

# 3. Core Point Type

Create a reusable `Point` type:

```python
Point(
    x: float,
    y: float
)
```

Coordinates should internally support floating-point values even though OpenCV eventually rasterizes integer coordinates.

Support:

```python
point.translate(dx, dy)
point.distance_to(other)
point.copy()
```

Also support arithmetic where useful:

```python
p3 = p1 + p2
p2 = p1 * 2
```

---

# 4. Color Model

Create a `Color` abstraction rather than exposing raw BGR tuples throughout the public API.

Prefer a human-friendly API:

```python
Color(r=255, g=0, b=0)
```

The renderer may internally convert RGB → OpenCV BGR.

Support:

```python
Color.from_hex("#FF0000")
Color.from_rgb(255, 0, 0)
Color.black()
Color.white()
Color.red()
```

Properties:

```text
r
g
b
a
```

Alpha should use either:

```text
0.0–1.0
```

or:

```text
0–255
```

but choose one representation and enforce it consistently.

Prefer 0.0–1.0 for high-level opacity.

---

# 5. Bounding Box

Create:

```python
BoundingBox(
    x,
    y,
    width,
    height
)
```

Expose:

```text
left
right
top
bottom
center
top_left
top_right
bottom_left
bottom_right
```

Every drawable object must be able to calculate:

```python
object.get_bounds()
```

The bounds should account for stroke thickness when practical.

---

# 6. Base Drawable Object

Every drawable should derive from an abstract `Drawable`.

Required common properties:

```text
id
name
visible
locked
opacity
z_index
metadata
tags
transform
```

Recommended:

```python
class Drawable:
    id: str
    name: str | None
    visible: bool
    locked: bool
    opacity: float
    z_index: int
    tags: set[str]
    metadata: dict
```

Required methods:

```python
draw(renderer, canvas)
get_bounds()
move(dx, dy)
rotate(angle, pivot=None)
scale(sx, sy=None, pivot=None)
clone()
to_dict()
```

Objects should automatically receive UUIDs unless explicitly assigned an ID.

---

# 7. Geometry Properties

Do NOT force every drawable to contain every geometry property.

Different geometry classes should expose appropriate properties.

Possible geometry concepts include:

```text
x
y
width
height

start
end

center

radius
radius_x
radius_y

points
vertices

control_points

start_angle
end_angle

rotation

anchor
pivot

bounding_box
```

Use `Point` objects rather than separate `start_x`, `start_y` variables whenever possible.

Example:

```python
line.start.x
line.start.y

line.end.x
line.end.y
```

---

# 8. Stroke Style

Create a reusable `StrokeStyle`.

Properties:

```text
color
width
opacity

line_type
cap_style
join_style

dash_pattern
dash_offset

antialias

taper_start
taper_end

brush_shape
softness
spacing
```

Enums should be used where appropriate.

Example:

```python
StrokeStyle(
    color=Color.black(),
    width=4,
    opacity=1.0,
    cap="round",
    join="round",
    antialias=True
)
```

OpenCV's `LINE_AA` should be used for anti-aliased rendering when appropriate.

Native OpenCV drawing operations including lines, rectangles, polygons and text should be used where suitable, but the public API must remain independent of raw OpenCV calls.

---

# 9. Fill Style

Create:

```python
FillStyle
```

Properties:

```text
enabled
color
opacity
```

Design the API so future implementations can add:

```text
linear_gradient
radial_gradient
pattern
texture
```

without requiring major redesign.

---

# 10. Transform Model

Create a reusable `Transform`.

Properties:

```text
translation_x
translation_y

rotation

scale_x
scale_y

skew_x
skew_y

flip_x
flip_y

pivot
```

Ideally support an affine transformation matrix internally.

Support:

```python
obj.move(dx, dy)
obj.rotate(degrees)
obj.scale(sx, sy)
obj.flip_horizontal()
obj.flip_vertical()
```

Transforms should ideally be non-destructive.

For example, rotating an object should not necessarily permanently rewrite every source coordinate.

Preserve original/local geometry when reasonable and calculate world geometry during rendering.

---

# 11. Required Drawing Primitives

Implement at least:

## Pixel / Point

```python
Pixel
PointMarker
```

Properties:

```text
position
color
size
opacity
shape
```

---

## Line

Properties:

```text
start
end
stroke
```

Derived/read-only properties:

```text
length
angle
center
bounds
```

---

## Arrow

Properties:

```text
start
end

stroke

head_length
head_width
head_angle

start_head
end_head
```

Support:

```text
none
triangle
open
diamond
circle
```

where practical.

---

## Rectangle

Properties:

```text
position
width
height
stroke
fill
```

---

## Rounded Rectangle

Additional:

```text
corner_radius
```

Potentially support individual corner radii later.

---

## Circle

Properties:

```text
center
radius
stroke
fill
```

---

## Ellipse

Properties:

```text
center
radius_x
radius_y
rotation
stroke
fill
```

---

## Arc

Properties:

```text
center
radius_x
radius_y
start_angle
end_angle
rotation
stroke
```

---

## Polyline

Properties:

```text
points
closed=False
stroke
```

---

## Polygon

Properties:

```text
vertices
stroke
fill
```

---

# 12. Bézier Curves

Implement Bézier support rather than depending solely on OpenCV primitives.

At minimum implement:

```text
quadratic Bézier
cubic Bézier
```

Cubic:

```python
BezierCurve(
    start,
    control1,
    control2,
    end
)
```

Sample the mathematical curve into sufficiently dense points and render using OpenCV polylines.

Allow configurable sampling resolution.

---

# 13. General Path

Create a `Path` abstraction.

A path should support commands similar conceptually to:

```text
MoveTo
LineTo
QuadraticTo
CubicTo
ClosePath
```

Example:

```python
path = Path()

path.move_to(100, 100)
path.line_to(200, 100)
path.curve_to(...)
path.close()
```

This is important because complicated drawings should not require creating hundreds of unrelated Line objects.

---

# 14. Freehand Stroke

Create a `FreehandStroke`.

It should store the ORIGINAL sampled input points:

```python
[
    Point(...),
    Point(...),
    Point(...),
]
```

Do not immediately destroy the original points.

Optional per-point properties should be supported:

```python
StrokePoint(
    x,
    y,
    pressure=None,
    timestamp=None,
    velocity=None
)
```

Support:

```text
smoothing
simplification
interpolation
pressure-sensitive width
```

Initially implement a simple version.

Then add optional:

```text
Chaikin smoothing
Catmull-Rom interpolation
Ramer-Douglas-Peucker simplification
```

Keep these algorithms modular.

---

# 15. Text Object

Implement a basic Text drawable.

Properties:

```text
text
position
color
opacity

font_face
font_scale
thickness

alignment
rotation

background
padding
```

Initially OpenCV Hershey fonts are sufficient.

Do not over-engineer advanced typography in version 1.

Make the design extensible enough that Pillow/FreeType could be integrated later without changing the public scene model.

---

# 16. Canvas

Create a `Canvas`.

Properties:

```text
width
height
background_color
channels
```

Possible constructor:

```python
canvas = Canvas(
    width=1920,
    height=1080,
    background=Color.white()
)
```

Methods:

```python
canvas.clear()
canvas.resize()
canvas.to_numpy()
canvas.save(path)
```

The underlying output should ultimately be a NumPy array compatible with OpenCV.

---

# 17. Scene

Create a `Scene` responsible for managing drawable objects.

Example:

```python
scene = Scene(1920, 1080)

scene.add(circle)
scene.add(label)
scene.add(arrow)
```

Operations:

```python
scene.add(obj)
scene.remove(id)
scene.get(id)
scene.find(...)
scene.clear()

scene.move_to_front(id)
scene.move_to_back(id)
scene.move_forward(id)
scene.move_backward(id)

scene.show(id)
scene.hide(id)
scene.lock(id)
scene.unlock(id)
```

Support lookup by:

```text
id
name
tag
type
```

Examples:

```python
scene.get("123")
scene.find_by_tag("annotation")
scene.find_by_type(Circle)
```

---

# 18. Groups

Implement `Group` as a drawable container.

Example:

```python
group = Group([
    circle,
    arrow,
    text
])
```

Moving:

```python
group.move(100, 50)
```

should move all contained objects logically.

Support:

```text
nested groups
group opacity
group visibility
group transforms
group bounds
```

---

# 19. Layers

Implement layers separately from groups.

A layer represents rendering organization.

Example:

```text
background
main
annotations
foreground
```

Allow:

```python
scene.create_layer("annotations")
scene.add(obj, layer="annotations")
```

Layer properties:

```text
visible
locked
opacity
z_order
```

Do not confuse semantic grouping with rendering layers.

---

# 20. Opacity and Alpha Compositing

OpenCV drawing functions do not automatically provide a complete retained-mode alpha compositing system.

Implement proper compositing.

For objects with opacity less than 1:

1. Draw onto a temporary transparent or intermediate buffer.
2. Alpha blend onto the destination.

Use mathematically correct alpha blending.

Design this utility centrally so every drawable does not implement its own blending code.

---

# 21. Clipping

Design support for:

```text
clip rectangle
clip path
mask
```

Basic rectangular clipping can be implemented first.

Architecture should permit arbitrary masks later.

---

# 22. Masks

Support an optional grayscale NumPy/OpenCV mask associated with a drawable, group, or layer.

Convention:

```text
0 = completely hidden
255 = completely visible
intermediate = partial alpha
```

Do not duplicate mask logic throughout the codebase.

---

# 23. Image Drawable

Support placing an existing image onto the scene:

```python
ImageObject(
    image=np.ndarray,
    position=Point(...),
    width=...,
    height=...
)
```

Support:

```text
resize
crop
rotate
opacity
mask
```

---

# 24. Object Anchors

Every drawable should expose useful named anchors based on its bounds:

```text
center

top
bottom
left
right

top_left
top_right
bottom_left
bottom_right
```

Allow:

```python
obj.anchor("center")
obj.anchor("top_right")
```

This will make higher-level positioning much easier.

---

# 25. Relative Positioning Utilities

Implement utilities such as:

```python
align_left(a, b)
align_right(a, b)
align_top(a, b)
align_bottom(a, b)

align_center_x(a, b)
align_center_y(a, b)

place_above(a, b, gap=20)
place_below(a, b, gap=20)
place_left_of(a, b, gap=20)
place_right_of(a, b, gap=20)
```

These can initially calculate positions rather than creating permanent constraints.

---

# 26. Optional Constraint System

Design for—but do not necessarily fully implement in version 1—a future constraint system such as:

```python
arrow.start.attach_to(label, "bottom")
arrow.end.attach_to(circle, "center")
```

If the circle moves, the arrow endpoint could automatically update.

Keep architecture compatible with this possibility.

---

# 27. Hit Testing

Implement:

```python
obj.contains_point(x, y)
scene.hit_test(x, y)
```

Hit testing should respect z-order and return topmost matching objects first.

For lines, use distance-to-segment with tolerance based on stroke width.

For circles and rectangles, use their actual geometry.

This is necessary for future object selection/manipulation.

---

# 28. Selection

Create a lightweight selection abstraction:

```python
selection = scene.select(["id1", "id2"])
```

Operations:

```python
selection.move(...)
selection.rotate(...)
selection.scale(...)
selection.delete()
```

Do not implement GUI handles.

Only implement logical selection/manipulation.

---

# 29. Temporal Metadata

Allow every drawable to optionally contain temporal information even if the initial renderer only produces static frames.

Create:

```python
Timing(
    start_time=0.0,
    duration=1.0,
    delay=0.0,
    easing="linear"
)
```

Possible properties:

```text
start_time
delay
duration
end_time
speed
easing
```

Also support a drawing/reveal progress value:

```text
0.0 = invisible/not drawn
0.5 = halfway rendered
1.0 = completely rendered
```

For path-based objects, implement:

```python
obj.render_progress = 0.5
```

so only the first 50% of the geometric path is rendered.

This should work first for:

```text
Line
Polyline
Path
FreehandStroke
Arrow
```

This capability must be architecturally separate from application UI animation loops.

---

# 30. Interpolation

Create reusable interpolation helpers:

```python
lerp(a, b, t)
lerp_point(p1, p2, t)
lerp_color(c1, c2, t)
```

Potential easing functions:

```text
linear
ease_in
ease_out
ease_in_out
```

Keep animation mathematics separate from rendering.

---

# 31. Serialization

Every scene and drawable should be serializable to plain JSON-compatible data.

Example:

```python
data = scene.to_dict()
json.dump(data, ...)
```

And:

```python
scene = Scene.from_dict(data)
```

Store semantic object information rather than rasterized pixels wherever possible.

Example representation:

```json
{
  "type": "line",
  "id": "abc123",
  "start": {"x": 100, "y": 200},
  "end": {"x": 500, "y": 300},
  "stroke": {
    "color": "#FF0000",
    "width": 4
  }
}
```

Ensure serialization has a format/version field for future migration.

---

# 32. Undo/Redo Architecture

Implement a clean command/history system or snapshot-based system.

At minimum:

```python
scene.undo()
scene.redo()
```

Actions that should eventually be reversible include:

```text
add
delete
move
resize
rotate
restyle
reorder
group
ungroup
```

Avoid storing full rendered image snapshots as the primary history mechanism.

History should operate on scene state/object commands.

---

# 33. Renderer

Create a dedicated OpenCV renderer.

Example:

```python
renderer = OpenCVRenderer()

image = renderer.render(scene)
```

Renderer responsibilities:

```text
sort layers
sort z-index
resolve transforms
calculate geometry
rasterize shapes
perform clipping
perform alpha compositing
apply masks
render effects
return NumPy image
```

Do not let Scene perform low-level OpenCV rendering.

---

# 34. Rendering Quality

Use:

```python
cv2.LINE_AA
```

where appropriate.

Support configurable:

```text
LINE_8
LINE_4
LINE_AA
```

Default to anti-aliased output.

---

# 35. Coordinate Model

Use the conventional image coordinate system:

```text
origin = top-left

+x = right
+y = down
```

Document this clearly.

Floating-point coordinates are allowed in model geometry.

Only quantize/rasterize when necessary.

---

# 36. Normalized Coordinates

Optionally provide helpers:

```python
canvas.normalized_point(0.5, 0.5)
```

which converts to canvas coordinates.

For example:

```text
(0,0)   = top-left
(1,1)   = bottom-right
(0.5,0.5) = center
```

Do not replace pixel coordinates with normalized coordinates; support both.

---

# 37. Derived Geometry

Objects should expose computed properties where useful.

Line:

```text
length
angle
center
```

Rectangle:

```text
area
center
corners
```

Circle:

```text
diameter
area
circumference
```

Polygon:

```text
centroid
area
perimeter
```

These should generally be computed rather than stored redundantly.

---

# 38. Metadata

Allow arbitrary metadata:

```python
obj.metadata["anything"] = ...
```

Also support tags:

```python
obj.tags.add("important")
```

Keep metadata separate from geometry/rendering properties.

---

# 39. Validation

Validate public API inputs.

Examples:

```text
negative radius → error
negative width → error
opacity outside 0–1 → error
malformed color → error
invalid points → error
invalid transform → error
```

Prefer clear custom exceptions over obscure OpenCV errors.

---

# 40. Performance

Do not optimize prematurely.

However:

- avoid unnecessary full-frame copies
- cache geometry where reasonable
- invalidate caches after relevant property changes
- keep rendering pipeline modular
- use NumPy vectorization where sensible
- use OpenCV primitives for rasterization where appropriate

Correctness and architecture are more important than micro-optimization initially.

---

# 41. Tests

Create automated tests for:

```text
Point
Color
BoundingBox
Transforms
Line geometry
Circle geometry
Rectangle geometry
Scene management
Layer ordering
Grouping
Serialization/deserialization
Hit testing
Alpha blending
Path interpolation
Freehand smoothing
Undo/redo
```

Also create visual-render tests that generate PNGs into:

```text
examples/output/
```

Do not require interactive windows for tests.

---

# 42. Example Scripts

Create examples:

```text
examples/
├── basic_shapes.py
├── styles.py
├── transformations.py
├── groups.py
├── layers.py
├── freehand.py
├── paths.py
├── alpha.py
├── object_manipulation.py
├── serialization.py
└── progressive_drawing.py
```

Each example should save an image to disk.

---

# 43. Desired User API

The final library should eventually allow code roughly like:

```python
from opencv_draw import *

scene = Scene(
    width=1200,
    height=800,
    background=Color.white()
)

circle = Circle(
    center=Point(400, 300),
    radius=100,
    stroke=StrokeStyle(
        color=Color.black(),
        width=4
    ),
    fill=FillStyle(
        color=Color.from_hex("#8EC5FF"),
        opacity=0.5
    )
)

arrow = Arrow(
    start=Point(100, 300),
    end=Point(280, 300),
    stroke=StrokeStyle(
        color=Color.red(),
        width=5
    )
)

scene.add(circle)
scene.add(arrow)

circle.move(100, 50)
arrow.stroke.width = 8

renderer = OpenCVRenderer()

image = renderer.render(scene)

cv2.imwrite("output.png", image)
```

---

# 44. Progressive Drawing API

Eventually support:

```python
stroke = FreehandStroke(
    points=[...],
    stroke=StrokeStyle(...)
)

stroke.progress = 0.25
```

Rendering should show approximately the first quarter of the stroke.

Then:

```python
stroke.progress = 0.50
stroke.progress = 0.75
stroke.progress = 1.0
```

should progressively reveal the remaining path.

The implementation should calculate progress by path length rather than simply by number of points where practical.

---

# 45. Development Phases

Do NOT attempt every advanced feature simultaneously.

Implement incrementally.

## Phase 1 — Foundation

Implement:

```text
Point
Color
BoundingBox
Canvas
Drawable
Scene
OpenCVRenderer
Line
Rectangle
Circle
StrokeStyle
FillStyle
PNG export
```

Fully test these before continuing.

## Phase 2 — Manipulation

Add:

```text
move
scale
rotate
opacity
z-index
visibility
clone
hit testing
```

## Phase 3 — Advanced Geometry

Add:

```text
Ellipse
Arc
Polygon
Polyline
Arrow
Bézier
Path
RoundedRectangle
```

## Phase 4 — Scene System

Add:

```text
Groups
Layers
Selection
Tags
Metadata
Object lookup
Relative positioning
```

## Phase 5 — Freehand Engine

Add:

```text
FreehandStroke
StrokePoint
smoothing
simplification
interpolation
variable width
```

## Phase 6 — Compositing

Add:

```text
alpha
masks
clipping
images
basic effects
```

## Phase 7 — Persistence

Add:

```text
JSON serialization
JSON deserialization
format versioning
undo/redo
```

## Phase 8 — Temporal Drawing

Add:

```text
Timing
progressive rendering
path-length interpolation
easing
frame rendering
```

Do not move to the next major phase until the previous phase has working tests and example output.

---

# 46. Critical Design Principles

Follow these throughout the project.

### A. Retained-mode graphics

Drawing objects must remain data structures after rendering.

Do not make pixels the only representation of the drawing.

### B. OpenCV is the renderer, not the object model

Avoid leaking `cv2` implementation details throughout the public API.

### C. Geometry and appearance are separate

Changing color must not modify geometry.

Changing geometry must not modify style.

### D. Preserve source information

For example, keep original freehand points rather than storing only the rasterized result.

### E. Properties must be manipulable after creation

This should work:

```python
circle.radius = 200
circle.fill.color = Color.red()
circle.move(50, 20)
```

and re-render correctly.

### F. Prefer composition over giant classes

Use:

```text
Drawable
Transform
StrokeStyle
FillStyle
Timing
```

rather than placing hundreds of unrelated fields directly inside every object.

### G. Keep renderer replaceable

The object model should not fundamentally depend on OpenCV-specific types.

A different renderer should theoretically be possible later.

### H. Write readable code

Prefer explicit, well-documented architecture over clever abstractions.

Use type hints throughout.

---

# 47. Documentation

Create a README explaining:

```text
installation
architecture
coordinate system
scene model
drawables
styles
transformations
rendering
serialization
examples
```

Document every public class and important public method.

---

# 48. Initial Implementation Instruction

Begin with **Phase 1 only**.

Before implementing later phases:

1. Create the repository/package structure.
2. Implement the core abstractions.
3. Implement Line, Rectangle, and Circle.
4. Implement StrokeStyle and FillStyle.
5. Implement Scene.
6. Implement OpenCVRenderer.
7. Implement image export.
8. Write unit tests.
9. Create a visual demo producing `phase1_demo.png`.
10. Run the tests and fix failures.
11. Review the architecture for obvious coupling problems.

Do not prematurely implement advanced features from later phases merely because they appear in this specification.

However, make Phase 1's architecture compatible with those future requirements.

At the end, provide:

- resulting project tree
- architectural explanation
- public API examples
- test results
- known limitations
- recommendations for Phase 2