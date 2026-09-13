"""Canvas representing the rendered raster output buffer."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from drawcv.core.color import Color
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.alpha import premultiply


class Canvas:
    """Canvas owning the rendered NumPy/OpenCV raster image.
    
    Attributes:
        width: Pixel width of the canvas.
        height: Pixel height of the canvas.
        has_alpha: True for opt-in straight-alpha BGRA, False for legacy BGR.
    """

    def __init__(self, width: int, height: int, buffer: np.ndarray | None = None, *, alpha: bool = False):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValidationError(f"Canvas width must be a positive integer, got {width}")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValidationError(f"Canvas height must be a positive integer, got {height}")

        self.width = width
        self.height = height
        if not isinstance(alpha, bool):
            raise ValidationError("alpha must be a boolean")
        self._has_alpha = alpha
        channels = 4 if alpha else 3

        if buffer is not None:
            if not isinstance(buffer, np.ndarray):
                raise ValidationError("Canvas buffer must be a NumPy ndarray")
            if buffer.shape != (height, width, channels):
                raise ValidationError(f"Buffer shape {buffer.shape} does not match canvas dimensions ({height}, {width}, {channels})")
            if buffer.dtype != np.uint8:
                raise ValidationError(f"Buffer dtype must be uint8, got {buffer.dtype}")
            self._buffer = buffer.copy()
        else:
            self._buffer = np.zeros((height, width, channels), dtype=np.uint8)

    @property
    def has_alpha(self) -> bool:
        """Whether the public buffer is straight-alpha uint8 BGRA."""
        return self._has_alpha

    @property
    def buffer(self) -> np.ndarray:
        """Writable uint8 BGR, or straight BGRA when alpha was requested."""
        return self._buffer

    def to_numpy(self) -> np.ndarray:
        """Return an independent BGR/BGRA copy without changing channel order."""
        return self._buffer.copy()

    def clear(self, color: Color) -> None:
        """Fill the canvas with a background color."""
        if not isinstance(color, Color):
            raise ValidationError(f"Expected Color, got {type(color).__name__}")
        b, g, r = color.to_bgr()
        if self.has_alpha:
            a = int(round(color.a * 255))
            self._buffer[:, :] = [b, g, r, a] if a else [0, 0, 0, 0]
        else:
            self._buffer[:, :] = [b, g, r]

    def flatten(self, background: Color) -> Canvas:
        """Return an opaque BGR copy composited over an opaque RGB Color."""
        if not isinstance(background, Color) or background.a != 1.0:
            raise ValidationError("flatten background must be an opaque Color")
        if not self.has_alpha:
            return Canvas(self.width, self.height, self.buffer)
        source = premultiply(self.buffer)
        bgr = source[..., :3] + np.array(background.to_bgr(), dtype=np.float32) * (1 - source[..., 3:4])
        return Canvas(self.width, self.height, np.rint(bgr).clip(0, 255).astype(np.uint8))

    def normalized_point(self, nx: float, ny: float) -> Point:
        """Convert normalized (0.0 to 1.0) coordinates to absolute canvas Point coordinates."""
        if not (0.0 <= nx <= 1.0) or not (0.0 <= ny <= 1.0):
            raise ValidationError(f"Normalized coordinates must be in [0.0, 1.0], got ({nx}, {ny})")
        return Point(nx * self.width, ny * self.height)

    def save(self, path: str | Path) -> None:
        """Save BGR to an image, or straight BGRA to PNG without alpha loss.

        Flatten alpha canvases against an explicit background for other formats.
        """
        file_path = Path(path)
        if self.has_alpha and file_path.suffix.lower() != ".png":
            raise ValidationError("Alpha canvases export to PNG; call flatten(background) for other formats")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(file_path), self._buffer)
        if not success:
            raise RenderError(f"Failed to write image to '{file_path}'")

    def __repr__(self) -> str:
        return f"Canvas(width={self.width}, height={self.height})"
