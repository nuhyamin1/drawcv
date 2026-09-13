import cv2
from drawcv import Scene, Line, Point, Color, StrokeStyle, Timing

scene = Scene(width=600, height=400, background=Color.white())

# A line that takes 2 seconds to draw itself with smooth easing
my_line = Line(
    start=Point(50, 200),
    end=Point(550, 200),
    stroke=StrokeStyle(color=Color.blue(), width=6.0),
)
my_line.timing = Timing(duration=2.0, easing="ease_in_out")
scene.add(my_line)

# Render at t=0.5s (25% drawn)
frame_25 = scene.render_at_time(0.5)
frame_25.save("line_25pct.png")

# Render at t=1.0s (50% drawn)
frame_50 = scene.render_at_time(1.0)
frame_50.save("line_50pct.png")

# Render at t=2.0s (100% complete)
frame_100 = scene.render_at_time(2.0)
frame_100.save("line_100pct.png")

print("Rendered 3 progressive stages!")
