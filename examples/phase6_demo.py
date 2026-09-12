"""Comprehensive visual demonstration of DrawCV Phase 6: Compositing & Effects.

Demonstrates:
1. Isolated Group Compositing (eliminating intersection darkening)
2. Asymmetric Drop Shadows and Content Blur Effects
3. Vector Path Clipping (ClipPath and ClipRect)
4. Grayscale Transparency Masks (Mask with FIT_BOUNDS)
5. Retained ImageObject with Cropping, Scaling, and Rotation
6. Retained Typography (Text with alignments, background plates, and rounded corners)
"""

import math
import numpy as np

from drawcv import (
    BlurEffect,
    BlurType,
    BoundingBox,
    Canvas,
    Circle,
    ClipPath,
    ClipRect,
    Color,
    FillStyle,
    FontFamily,
    FreehandStroke,
    Group,
    ImageInterpolation,
    ImageObject,
    Line,
    Mask,
    MaskMapping,
    OpenCVRenderer,
    Path,
    Point,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    ShadowEffect,
    StrokePoint,
    StrokeStyle,
    Text,
    TextAlignment,
)


def create_demo_image(width: int = 160, height: int = 120) -> np.ndarray:
    """Generate a vibrant synthetic procedural test image with rich gradients and circular patterns."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Complex gradient pattern
            r = int(128 + 127 * math.sin(x / 18.0))
            g = int(128 + 127 * math.sin(y / 15.0 + 1.0))
            b = int(128 + 127 * math.cos((x + y) / 22.0))
            img[y, x] = [b, g, r] # BGR
    return img


def create_radial_mask(size: int = 180) -> np.ndarray:
    """Generate a smooth radial gradient vignette mask (255 at center, 0 at perimeter)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size / 2.0
    max_radius = size / 2.0
    for y in range(size):
        for x in range(size):
            dist = math.hypot(x - center, y - center)
            if dist >= max_radius:
                val = 0
            else:
                val = int(255 * (1.0 - (dist / max_radius)))
            mask[y, x] = val
    return mask


def make_star_points(center: Point, outer_r: float, inner_r: float, num_points: int = 5) -> list[Point]:
    """Generate vertices for a regular star polygon."""
    pts = []
    angle_step = math.pi / num_points
    for i in range(num_points * 2):
        r = outer_r if i % 2 == 0 else inner_r
        ang = i * angle_step - math.pi / 2.0
        pts.append(Point(center.x + r * math.cos(ang), center.y + r * math.sin(ang)))
    return pts


def build_phase6_scene() -> Scene:
    scene = Scene(width=1600, height=1000, background=Color(15, 23, 42)) # Deep navy slate

    # -------------------------------------------------------------------------
    # Header & Titles
    # -------------------------------------------------------------------------
    title = Text(
        "DrawCV — Phase 6: Compositing & Effects Pipeline",
        position=Point(60, 45),
        color=Color(255, 255, 255),
        font_family=FontFamily.DUPLEX,
        font_scale=1.1,
        thickness=2,
    )
    subtitle = Text(
        "Isolated Offscreen Buffers • Drop Shadows • Blur • Vector Clipping • Grayscale Masks • Typography • ImageObject",
        position=Point(60, 85),
        color=Color(148, 163, 184), # Slate 400
        font_family=FontFamily.SIMPLEX,
        font_scale=0.6,
        thickness=1,
    )
    scene.add(title)
    scene.add(subtitle)

    # -------------------------------------------------------------------------
    # Panel 1: Isolated vs Unisolated Group Compositing
    # -------------------------------------------------------------------------
    p1_plate = RoundedRectangle(
        x=60, y=130, width=460, height=390, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)), # Slate 800
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p1_title = Text("1. Isolated Group Compositing", position=Point(85, 155), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p1_desc = Text("Eliminates double-opacity accumulation in overlapping geometry", position=Point(85, 182), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p1_plate)
    scene.add(p1_title)
    scene.add(p1_desc)

    # Sub-card Left: Unisolated (Direct overlap darkening)
    lbl_uniso = Text("Unhybrid / Standard Overlap", position=Point(100, 215), color=Color(226, 232, 240), font_scale=0.48)
    lbl_uniso_sub = Text("Dark intersection artifact", position=Point(100, 233), color=Color(248, 113, 113), font_scale=0.42)
    c1 = Circle(center=Point(170, 320), radius=55, fill=FillStyle(color=Color(236, 72, 153)), opacity=0.5) # Pink
    c2 = Circle(center=Point(230, 320), radius=55, fill=FillStyle(color=Color(59, 130, 246)), opacity=0.5) # Blue
    scene.add(lbl_uniso)
    scene.add(lbl_uniso_sub)
    scene.add(c1)
    scene.add(c2)

    # Sub-card Right: Isolated Group (Uniform 0.5 opacity, no dark intersection)
    lbl_iso = Text("Isolated Group Pass", position=Point(320, 215), color=Color(226, 232, 240), font_scale=0.48)
    lbl_iso_sub = Text("Unified 0.5 boundary opacity", position=Point(320, 233), color=Color(74, 222, 128), font_scale=0.42)
    iso_group = Group(opacity=0.5)
    c3 = Circle(center=Point(360, 320), radius=55, fill=FillStyle(color=Color(236, 72, 153)), opacity=1.0)
    c4 = Circle(center=Point(420, 320), radius=55, fill=FillStyle(color=Color(59, 130, 246)), opacity=1.0)
    iso_group.add(c3)
    iso_group.add(c4)
    scene.add(lbl_iso)
    scene.add(lbl_iso_sub)
    scene.add(iso_group)

    p1_note = Text(
        "Right group blends as a single unified matte: pink and blue merge internally\nat full opacity before 50% source-over to background.",
        position=Point(85, 470),
        color=Color(148, 163, 184),
        font_scale=0.42,
    )
    scene.add(p1_note)

    # -------------------------------------------------------------------------
    # Panel 2: Drop Shadows & Content Blur Effects
    # -------------------------------------------------------------------------
    p2_plate = RoundedRectangle(
        x=570, y=130, width=460, height=390, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p2_title = Text("2. Post-Processing Effects", position=Point(595, 155), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p2_desc = Text("Drop shadows with asymmetric padding & premultiplied content blur", position=Point(595, 182), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p2_plate)
    scene.add(p2_title)
    scene.add(p2_desc)

    # Card with deep drop shadow
    shadow_card = RoundedRectangle(
        x=620, y=225, width=170, height=170, corner_radius=14,
        fill=FillStyle(color=Color(241, 245, 249)), # Light white/slate
        stroke=StrokeStyle(color=Color(203, 213, 225), width=1.5),
    )
    shadow_card.effects.append(ShadowEffect(offset_x=15.0, offset_y=20.0, blur_radius=16.0, color=Color(0, 0, 0, 0.65)))
    card_badge = Text("SHADOW CARD", position=Point(645, 260), color=Color(30, 41, 59), font_scale=0.48, thickness=1)
    card_sub = Text("offset=(15, 20)\nsigma=16.0", position=Point(645, 290), color=Color(100, 116, 139), font_scale=0.42)
    scene.add(shadow_card)
    scene.add(card_badge)
    scene.add(card_sub)

    # Shape with Content Blur
    blur_shape = RoundedRectangle(
        x=840, y=225, width=150, height=150, corner_radius=14,
        fill=FillStyle(color=Color(168, 85, 247)), # Purple
    )
    blur_shape.effects.append(BlurEffect(kernel_size=25, sigma=7.0, blur_type=BlurType.GAUSSIAN))
    scene.add(blur_shape)

    blur_label = Text("CONTENT BLUR", position=Point(855, 415), color=Color(192, 132, 252), font_scale=0.48)
    blur_sub = Text("Premultiplied Gaussian blur\n(kernel=25, sigma=7.0)", position=Point(855, 440), color=Color(148, 163, 184), font_scale=0.42)
    scene.add(blur_label)
    scene.add(blur_sub)

    # -------------------------------------------------------------------------
    # Panel 3: Vector Path & Rectangular Clipping
    # -------------------------------------------------------------------------
    p3_plate = RoundedRectangle(
        x=1080, y=130, width=460, height=390, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p3_title = Text("3. Vector & Rectangular Clipping", position=Point(1105, 155), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p3_desc = Text("Entity-local ClipPath (star polygon) & ClipRect stenciling", position=Point(1105, 182), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p3_plate)
    scene.add(p3_title)
    scene.add(p3_desc)

    # Star vector clip container
    star_pts = make_star_points(Point(1220, 315), outer_r=90, inner_r=42, num_points=5)
    star_group = Group()
    star_group.clip = ClipPath(points=star_pts)

    # Fill star background
    star_bg = Rectangle(position=Point(1110, 205), width=220, height=220, fill=FillStyle(color=Color(15, 23, 42)))
    star_group.add(star_bg)

    # Dense concentric colorful rings inside star
    for r, col in [
        (85, Color(239, 68, 68)),   # Red
        (70, Color(249, 115, 22)),  # Orange
        (55, Color(234, 179, 8)),   # Yellow
        (40, Color(34, 197, 94)),   # Green
        (25, Color(59, 130, 246)),  # Blue
    ]:
        star_group.add(Circle(center=Point(1220, 315), radius=r, fill=FillStyle(color=col)))

    scene.add(star_group)

    # Star outline marker to highlight the clip contour
    star_outline = Polygon(vertices=star_pts, stroke=StrokeStyle(color=Color(255, 255, 255), width=2.0), fill=None)
    scene.add(star_outline)

    # Sub-item: Rectangular clip box
    rect_clipped = Rectangle(
        position=Point(1370, 240), width=140, height=140,
        fill=FillStyle(color=Color(14, 165, 233)), # Sky Blue
    )
    rect_clipped.clip = ClipRect(x=1390, y=260, width=100, height=100)
    rect_clipped.effects.append(BlurEffect(kernel_size=17, sigma=5.0)) # Blurred, but stenciled hard
    scene.add(rect_clipped)

    rect_clip_border = Rectangle(
        position=Point(1390, 260), width=100, height=100,
        stroke=StrokeStyle(color=Color(255, 255, 255), width=1.5), fill=None
    )
    scene.add(rect_clip_border)

    p3_note = Text(
        "Star ClipPath cuts colorful concentric circles strictly at vector vertices.\nRight box blurs internally but clips sharp to [100x100] stencil.",
        position=Point(1105, 465),
        color=Color(148, 163, 184),
        font_scale=0.42,
    )
    scene.add(p3_note)

    # -------------------------------------------------------------------------
    # Panel 4: Grayscale Mask Vignettes
    # -------------------------------------------------------------------------
    p4_plate = RoundedRectangle(
        x=60, y=550, width=460, height=400, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p4_title = Text("4. Grayscale Alpha Masks", position=Point(85, 575), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p4_desc = Text("Smooth feathering & premultiplied 4-channel scaling without fringes", position=Point(85, 602), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p4_plate)
    scene.add(p4_title)
    scene.add(p4_desc)

    # Radial vignette mask
    vignette_mask_buf = create_radial_mask(size=200)
    vignette_mask = Mask(buffer=vignette_mask_buf, mapping=MaskMapping.FIT_BOUNDS)

    # Group with colorful shapes masked by the vignette
    vignette_group = Group(mask=vignette_mask)
    vignette_bg = Rectangle(position=Point(100, 640), width=180, height=180, fill=FillStyle(color=Color(99, 102, 241))) # Indigo
    vignette_group.add(vignette_bg)
    vignette_group.add(Circle(center=Point(190, 730), radius=55, fill=FillStyle(color=Color(244, 63, 94)))) # Rose
    vignette_group.add(Rectangle(position=Point(140, 680), width=100, height=100, fill=FillStyle(color=Color(234, 179, 8)), opacity=0.7)) # Amber
    scene.add(vignette_group)

    # Linear gradient mask on striped card
    linear_mask_buf = np.tile(np.linspace(255, 0, 180, dtype=np.uint8), (180, 1))
    linear_mask = Mask(buffer=linear_mask_buf, mapping=MaskMapping.FIT_BOUNDS)

    linear_group = Group(mask=linear_mask)
    linear_bg = Rectangle(position=Point(315, 640), width=175, height=180, fill=FillStyle(color=Color(16, 185, 129))) # Emerald
    linear_group.add(linear_bg)
    for y_offset in range(660, 810, 25):
        linear_group.add(Line(start=Point(325, y_offset), end=Point(480, y_offset), stroke=StrokeStyle(color=Color.white(), width=4.0)))
    scene.add(linear_group)

    mask_lbl1 = Text("Radial Feather", position=Point(135, 845), color=Color(226, 232, 240), font_scale=0.48)
    mask_lbl2 = Text("Horizontal Ramp", position=Point(345, 845), color=Color(226, 232, 240), font_scale=0.48)
    p4_note = Text("Both B, G, R and Alpha scale synchronously by mask factor m (Cpm' = m*Cpm, a' = m*a)", position=Point(85, 915), color=Color(148, 163, 184), font_scale=0.42)
    scene.add(mask_lbl1)
    scene.add(mask_lbl2)
    scene.add(p4_note)

    # -------------------------------------------------------------------------
    # Panel 5: ImageObject, Cropping, Scaling & Rotation
    # -------------------------------------------------------------------------
    p5_plate = RoundedRectangle(
        x=570, y=550, width=460, height=400, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p5_title = Text("5. Retained ImageObject", position=Point(595, 575), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p5_desc = Text("Sub-rectangle cropping, scaling, rotation & source alpha", position=Point(595, 602), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p5_plate)
    scene.add(p5_title)
    scene.add(p5_desc)

    raw_img = create_demo_image(width=160, height=120)

    # Image A: Full image scaled with drop shadow
    img_a = ImageObject(
        image=raw_img,
        position=Point(610, 645),
        width=170,
        height=130,
        interpolation=ImageInterpolation.CUBIC,
    )
    img_a.effects.append(ShadowEffect(offset_x=10, offset_y=12, blur_radius=8))
    scene.add(img_a)

    # Image B: Cropped center square, rotated by 15 deg
    crop_rect = BoundingBox(20, 20, 100, 80)
    img_b = ImageObject(
        image=raw_img,
        position=Point(830, 655),
        crop=crop_rect,
        width=150,
        height=120,
        interpolation=ImageInterpolation.LINEAR,
    )
    img_b.transform.rotation = 12.0
    img_b.effects.append(ShadowEffect(offset_x=8, offset_y=14, blur_radius=10))
    scene.add(img_b)

    img_lbl1 = Text("Scaled Full Image", position=Point(630, 805), color=Color(226, 232, 240), font_scale=0.46)
    img_lbl2 = Text("Cropped & 12° Rotated", position=Point(835, 805), color=Color(226, 232, 240), font_scale=0.46)
    scene.add(img_lbl1)
    scene.add(img_lbl2)

    p5_note = Text(
        "Image matrices are retained objects with affine transforms, sub-box crop,\nand full compatibility with shadows, masks, and clips.",
        position=Point(595, 890),
        color=Color(148, 163, 184),
        font_scale=0.42,
    )
    scene.add(p5_note)

    # -------------------------------------------------------------------------
    # Panel 6: Typography, Alignment & Rounded Background Plates
    # -------------------------------------------------------------------------
    p6_plate = RoundedRectangle(
        x=1080, y=550, width=460, height=400, corner_radius=16,
        fill=FillStyle(color=Color(30, 41, 59)),
        stroke=StrokeStyle(color=Color(51, 65, 85), width=1.5),
    )
    p6_title = Text("6. Typography & Text Plates", position=Point(1105, 575), color=Color(56, 189, 248), font_scale=0.65, thickness=1)
    p6_desc = Text("Renderer-independent fonts, alignments & rounded background plates", position=Point(1105, 602), color=Color(148, 163, 184), font_scale=0.45, thickness=1)
    scene.add(p6_plate)
    scene.add(p6_title)
    scene.add(p6_desc)

    # Alignment demonstration
    align_line = Line(start=Point(1310, 640), end=Point(1310, 750), stroke=StrokeStyle(color=Color(71, 85, 105), width=1.0))
    scene.add(align_line)

    t_left = Text("LEFT ALIGNED", position=Point(1310, 650), color=Color(248, 113, 113), alignment=TextAlignment.LEFT, font_scale=0.46)
    t_center = Text("CENTER ALIGNED", position=Point(1310, 685), color=Color(250, 204, 21), alignment=TextAlignment.CENTER, font_scale=0.46)
    t_right = Text("RIGHT ALIGNED", position=Point(1310, 720), color=Color(74, 222, 128), alignment=TextAlignment.RIGHT, font_scale=0.46)
    scene.add(t_left)
    scene.add(t_center)
    scene.add(t_right)

    # Badges with rounded background plates and drop shadows
    badge1 = Text(
        "STATUS: OPERATIONAL",
        position=Point(1120, 780),
        color=Color.white(),
        font_family=FontFamily.SIMPLEX,
        font_scale=0.45,
        thickness=1,
        background_fill=Color(16, 185, 129), # Emerald green
        background_radius=8.0,
        padding=8.0,
    )
    badge1.effects.append(ShadowEffect(offset_x=4, offset_y=6, blur_radius=6, color=Color(0, 0, 0, 0.4)))
    scene.add(badge1)

    badge2 = Text(
        "FPS: 60 • LATENCY: 0.4ms",
        position=Point(1120, 835),
        color=Color.white(),
        font_family=FontFamily.SIMPLEX,
        font_scale=0.45,
        thickness=1,
        background_fill=Color(59, 130, 246), # Blue
        background_radius=8.0,
        padding=8.0,
    )
    badge2.effects.append(ShadowEffect(offset_x=4, offset_y=6, blur_radius=6, color=Color(0, 0, 0, 0.4)))
    scene.add(badge2)

    p6_note = Text(
        "FontFamily abstracts OpenCV Hershey constants. background_radius\nenables rounded typography badges with precise metrics and padding.",
        position=Point(1105, 905),
        color=Color(148, 163, 184),
        font_scale=0.42,
    )
    scene.add(p6_note)

    return scene


def main():
    print("Building Phase 6 Scene...")
    scene = build_phase6_scene()

    print("Rendering with OpenCVRenderer (1600x1000)...")
    renderer = OpenCVRenderer()
    canvas = renderer.render(scene)

    out_path = "examples/output/phase6_demo.png"
    canvas.save(out_path)
    print(f"Phase 6 Demo successfully rendered and saved to {out_path}!")


if __name__ == "__main__":
    main()
