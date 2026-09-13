import cv2
import time

from drawcv import Scene, Line, Point, Color, StrokeStyle, Timing

scene = Scene(width=600, height=400, background=Color.white())

line = Line(
    start=Point(50, 200),
    end=Point(550, 200),
    stroke=StrokeStyle(color=Color.blue(), width=6.0),
)

line.timing = Timing(duration=2.0, easing="ease_in_out")
scene.add(line)

start = time.perf_counter()

while True:
    t = time.perf_counter() - start

    if t > 2.0:
        t = 2.0

    frame = scene.render_at_time(t)

    cv2.imshow("Progressive Drawing", frame.buffer)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if t >= 2.0:
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()