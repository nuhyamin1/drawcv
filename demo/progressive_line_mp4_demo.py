from drawcv import Scene, Line, Point, Color, StrokeStyle, Timing, VideoRenderer

scene = Scene(width=600, height=400, background=Color(20, 25, 35))

line = Line(
    start=Point(50, 200),
    end=Point(550, 200),
    stroke=StrokeStyle(color=Color(100, 200, 255), width=5.0),
)
line.timing = Timing(duration=2.0, easing="ease_in_out")
scene.add(line)

# Directly render an MP4 video file!
VideoRenderer.render_video(scene, "drawing_line.mp4", duration=2.0, fps=30)
print("Video saved to drawing_line.mp4!")
