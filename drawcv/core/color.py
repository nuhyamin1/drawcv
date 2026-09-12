"""Color representation and color space conversion utilities for DrawCV."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any

from drawcv.core.exceptions import ValidationError


@dataclass(frozen=True)
class Color:
    """Immutable RGBA Color with strict channel validation.
    
    Channels:
        r: red channel, integer in [0, 255]
        g: green channel, integer in [0, 255]
        b: blue channel, integer in [0, 255]
        a: alpha channel, float in [0.0, 1.0]
    """
    r: int
    g: int
    b: int
    a: float = 1.0

    def __init__(self, r: int, g: int, b: int, a: float | int = 1.0):
        # Validate RGB channels
        for name, val in (("r", r), ("g", g), ("b", b)):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValidationError(f"Color channel '{name}' must be an integer, got {type(val).__name__}")
            if not (0 <= val <= 255):
                raise ValidationError(f"Color channel '{name}' must be in range [0, 255], got {val}")

        # Validate alpha channel
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            raise ValidationError(f"Color alpha must be a float in range [0.0, 1.0], got {type(a).__name__}")
        a_float = float(a)
        if not (0.0 <= a_float <= 1.0):
            raise ValidationError(f"Color alpha must be in range [0.0, 1.0], got {a}")

        object.__setattr__(self, "r", int(r))
        object.__setattr__(self, "g", int(g))
        object.__setattr__(self, "b", int(b))
        object.__setattr__(self, "a", a_float)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: float | int = 1.0) -> Color:
        """Create a Color from RGB and optional alpha."""
        return cls(r, g, b, a)

    @classmethod
    def from_bgr(cls, b: int, g: int, r: int, a: float | int = 1.0) -> Color:
        """Create a Color from BGR and optional alpha."""
        return cls(r, g, b, a)

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        """Parse a hex string into a Color.
        
        Supported formats:
            '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA' (leading '#' is optional).
        """
        if not isinstance(hex_str, str):
            raise ValidationError(f"Hex color must be a string, got {type(hex_str).__name__}")
        
        cleaned = hex_str.strip().lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]+", cleaned):
            raise ValidationError(f"Invalid characters in hex color string: '{hex_str}'")

        if len(cleaned) == 3:  # RGB short
            r = int(cleaned[0] * 2, 16)
            g = int(cleaned[1] * 2, 16)
            b = int(cleaned[2] * 2, 16)
            return cls(r, g, b, 1.0)
        elif len(cleaned) == 4:  # RGBA short
            r = int(cleaned[0] * 2, 16)
            g = int(cleaned[1] * 2, 16)
            b = int(cleaned[2] * 2, 16)
            a = int(cleaned[3] * 2, 16) / 255.0
            return cls(r, g, b, a)
        elif len(cleaned) == 6:  # RRGGBB
            r = int(cleaned[0:2], 16)
            g = int(cleaned[2:4], 16)
            b = int(cleaned[4:6], 16)
            return cls(r, g, b, 1.0)
        elif len(cleaned) == 8:  # RRGGBBAA
            r = int(cleaned[0:2], 16)
            g = int(cleaned[2:4], 16)
            b = int(cleaned[4:6], 16)
            a = int(cleaned[6:8], 16) / 255.0
            return cls(r, g, b, a)
        else:
            raise ValidationError(f"Invalid hex color length ({len(cleaned)} chars): '{hex_str}'")

    @classmethod
    def black(cls) -> Color:
        return cls(0, 0, 0, 1.0)

    @classmethod
    def white(cls) -> Color:
        return cls(255, 255, 255, 1.0)

    @classmethod
    def red(cls) -> Color:
        return cls(255, 0, 0, 1.0)

    @classmethod
    def green(cls) -> Color:
        return cls(0, 255, 0, 1.0)

    @classmethod
    def blue(cls) -> Color:
        return cls(0, 0, 255, 1.0)

    @classmethod
    def yellow(cls) -> Color:
        return cls(255, 255, 0, 1.0)

    @classmethod
    def cyan(cls) -> Color:
        return cls(0, 255, 255, 1.0)

    @classmethod
    def magenta(cls) -> Color:
        return cls(255, 0, 255, 1.0)

    @classmethod
    def transparent(cls) -> Color:
        return cls(0, 0, 0, 0.0)

    def to_rgb(self) -> tuple[int, int, int]:
        """Return (r, g, b) tuple."""
        return (self.r, self.g, self.b)

    def to_rgba(self) -> tuple[int, int, int, float]:
        """Return (r, g, b, a) tuple."""
        return (self.r, self.g, self.b, self.a)

    def to_bgr(self) -> tuple[int, int, int]:
        """Return (b, g, r) tuple for OpenCV BGR."""
        return (self.b, self.g, self.r)

    def to_bgra(self) -> tuple[int, int, int, float]:
        """Return (b, g, r, a) tuple for OpenCV BGRA."""
        return (self.b, self.g, self.r, self.a)

    def to_hex(self, include_alpha: bool = False) -> str:
        """Format as hex string `#RRGGBB` or `#RRGGBBAA`."""
        if include_alpha:
            alpha_byte = int(round(self.a * 255))
            return f"#{self.r:02X}{self.g:02X}{self.b:02X}{alpha_byte:02X}"
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def with_alpha(self, a: float | int) -> Color:
        """Return a copy of this color with a new alpha value."""
        return Color(self.r, self.g, self.b, a)

    def __repr__(self) -> str:
        if self.a < 1.0:
            return f"Color(r={self.r}, g={self.g}, b={self.b}, a={self.a:.2f})"
        return f"Color(r={self.r}, g={self.g}, b={self.b})"


# Pre-instantiated standard color constants
Color.WHITE = Color(255, 255, 255, 1.0)
Color.BLACK = Color(0, 0, 0, 1.0)
Color.RED = Color(255, 0, 0, 1.0)
Color.GREEN = Color(0, 255, 0, 1.0)
Color.BLUE = Color(0, 0, 255, 1.0)
Color.YELLOW = Color(255, 255, 0, 1.0)
Color.CYAN = Color(0, 255, 255, 1.0)
Color.MAGENTA = Color(255, 0, 255, 1.0)
Color.TRANSPARENT = Color(0, 0, 0, 0.0)

