"""Base effect abstraction for DrawCV post-processing visual effects."""

from __future__ import annotations
from abc import ABC, abstractmethod


class Effect(ABC):
    """Abstract base class for post-processing visual effects applied to drawables, groups, or layers."""

    @abstractmethod
    def get_padding(self) -> tuple[float, float, float, float]:
        """Return asymmetric padding required by this effect in pixels: (left, right, top, bottom)."""
        pass
