"""Canvas representing the rendered raster output buffer."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from drawcv.core.color import Color
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.core.geometry import Point


class Canvas:
    """Canvas owning the rendered NumPy/OpenCV raster image.
    
    Attributes:
        width: Pixel width of the canvas.
        height: Pixel height of the canvas.
    """

    def __init__(self, width: int, height: int, buffer: np.ndarray | None = None):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValidationError(f"Canvas width must be a positive integer, got {width}")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValidationError(f"Canvas height must be a positive integer, got {height}")

        self.width = width
        self.height = height

        if buffer is not None:
            if not isinstance(buffer, np.ndarray):
                raise ValidationError("Canvas buffer must be a NumPy ndarray")
            if buffer.shape != (height, width, 3):
                raise ValidationError(f"Buffer shape {buffer.shape} does not match canvas dimensions ({height}, {width}, 3)")
            if buffer.dtype != np.uint8:
                raise ValidationError(f"Buffer dtype must be uint8, got {buffer.dtype}")
            self._buffer = buffer.copy()
        else:
            self._buffer = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def buffer(self) -> np.ndarray:
        """Direct access to internal BGR uint8 NumPy buffer."""
        return self._buffer

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the rendered BGR image as a NumPy array."""
        return self._buffer.copy()

    def clear(self, color: Color) -> None:
        """Fill the canvas with a background color."""
        if not isinstance(color, Color):
            raise ValidationError(f"Expected Color, got {type(color).__name__}")
        b, g, r = color.to_bgr()
        self._buffer[:, :] = [b, g, r]

    def normalized_point(self, nx: float, ny: float) -> Point:
        """Convert normalized (0.0 to 1.0) coordinates to absolute canvas Point coordinates."""
        if not (0.0 <= nx <= 1.0) or not (0.0 <= ny <= 1.0):
            raise ValidationError(f"Normalized coordinates must be in [0.0, 1.0], got ({nx}, {ny})")
        return Point(nx * self.width, ny * self.height)

    def save(self, path: str | Path) -> None:
        """Save the canvas image to a file on disk (e.g. PNG, JPG)."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(file_path), self._buffer)
        if not success:
            raise RenderError(f"Failed to write image to '{file_path}'")

    def __repr__(self) -> str:
        return f"Canvas(width={self.width}, height={self.height})"
