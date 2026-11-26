"""Cold group bounds and path sampling workloads for whiteboard scenes."""
from time import perf_counter

from drawcv import FontAsset, Group, Path, Point, Polyline, Text


def main():
    font = FontAsset.from_file("tests/assets/fonts/notosans.ttf")
    labels = [Text(str(i), fonts=(font,), text_origin="baseline", position=Point(i*10, 10))
              for i in range(20)]
    curves = [Polyline(points=[Point(i, (i*i) % 31 + k) for i in range(600)])
              for k in range(2)]
    group = Group(children=labels + curves)
    start = perf_counter()
    group.get_bounds()
    print(f"Group bounds: {(perf_counter()-start)*1000:.2f} ms")

    path = Path().move_to(0, 0).cubic_to(Point(0, 90), Point(200, -10), Point(150, 80))
    progresses = [i/999 for i in range(1000)]
    start = perf_counter()
    if hasattr(path, "points_at"):
        path.points_at(progresses)
    else:
        # Allows running the same fixture against older releases.
        [path.point_at(p) for p in progresses]
    print(f"1,000 path positions: {(perf_counter()-start)*1000:.2f} ms")


if __name__ == "__main__":
    main()
