# Reliable edits and history

`StrokeStyle`, `FillStyle`, and `Transform` validate candidate property changes
before assigning them. Rejected changes preserve values and any authoritative
affine matrix. This also protects values referenced outside the drawable:

```python
from drawcv import StrokeStyle, ValidationError

stroke = StrokeStyle(width=5)
try:
    stroke.width = -1
except ValidationError:
    assert stroke.width == 5
```

Use scene edits for compound semantic changes:

```python
with scene.edit(line, name="Restyle connection"):
    line.stroke.width = 6
    line.stroke.dash_array = (16, 8)
    line.stroke.cap_style = CapStyle.BUTT

scene.undo()
scene.redo()
assert scene.get(line.id) is line
```

An unhandled exception rolls all target snapshots back. On normal exit, edited
raw dataclass fields are validated on a candidate and geometry is checked through
its serialization constructor before recording history.
Groups include descendant state, and redundant ancestor/descendant targets are
deduplicated. Failed edits do not add undo entries or discard the redo branch.

`move`, `rotate`, and `scale` restore the original transform if a later component
fails. Selection transforms restore earlier members too. Multi-axis relative
placement and distribution similarly roll back completed transform changes.
The same Transform reference is retained for these failed transform operations.

Use `scene.batch()` to combine **recorded commands** into one history step:

```python
with scene.batch("Arrange connections"):
    scene.move_object(line.id, 20, 0)
    with scene.edit(line):
        line.stroke.dash_offset = 4
```

Failed nested batches undo their recorded commands. Failed compound command
execute/undo/redo operations reverse completed children, and failed history replay
keeps its stack entry available. A custom Command must itself fail atomically and
provide a working inverse; a manager cannot reconstruct arbitrary external side
effects of a partially failing custom command.

## Audit boundaries

| Surface inspected | Result / supported editing route |
| --- | --- |
| StrokeStyle, FillStyle, Transform setters | Validate before commit |
| `Drawable.progress` | Already validates before assigning |
| Point, Color, BoundingBox, path commands, clip geometry | Immutable values; replace through constructors |
| Path builders and `FreehandStroke.add_point` | Validate new command/sample before appending |
| Drawable/selection transforms and placement | Roll back multi-step transform edits |
| Selection opacity and z-index setters | Reject invalid types/ranges before mutation |
| Scene edit/restyle and history batches | Validate edited serialized state; restore on failure |
| Raw shape/effect/layer attributes and mutable lists | Not universally intercepted; use valid constructors and `scene.edit` for drawable transactions |

This milestone does not replace all retained dataclasses with a new property system.
Direct list/geometry edits outside a scene transaction can still create invalid
state, and a batch cannot undo unrecorded assignments. `scene.edit` preserves live
Drawable identity; nested styles and transforms may be replaced during semantic
undo/redo, as in the existing history implementation. Avoid holding nested-value
references across history restoration when you need the currently attached style.
