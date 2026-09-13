import cv2
from drawcv import (
    Scene,
    OpenCVRenderer,
    Point,
    Color,
    Line,
    Polyline,
    Arrow,
    ArrowHeadStyle,
    StrokeStyle,
)

# 1. Create a canvas/scene
scene = Scene(width=800, height=600, background=Color.white())

# 2. Add a simple red line
line1 = Line(
    start=Point(50, 100),
    end=Point(400, 100),
    stroke=StrokeStyle(color=Color.red(), width=5.0),
)
scene.add(line1)

# 3. Add a translucent blue line
line2 = Line(
    start=Point(50, 180),
    end=Point(700, 180),
    stroke=StrokeStyle(color=Color(0, 120, 255, 0.6), width=10.0),
)
scene.add(line2)

# 4. Add a connected Polyline (zigzag)
poly = Polyline(
    points=[
        Point(50, 300),
        Point(150, 220),
        Point(250, 340),
        Point(350, 240),
        Point(450, 320),
    ],
    stroke=StrokeStyle(color=Color(30, 180, 80), width=4.0),
)
scene.add(poly)

# 5. Add a directed Arrow
arrow = Arrow(
    start=Point(50, 450),
    end=Point(450, 450),
    head_length=25,
    head_width=18,
    head_style=ArrowHeadStyle.TRIANGLE,
    stroke=StrokeStyle(color=Color(220, 50, 50), width=4.0),
)
scene.add(arrow)

# 6. Render the scene
renderer = OpenCVRenderer()
canvas = renderer.render(scene)

# 7. Save to an image file
canvas.save("my_lines.png")
print("Saved image to my_lines.png!")

# 8. Show it directly in an OpenCV desktop window
cv2.imshow("DrawCV - My Lines", canvas.buffer)
print("Press any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
