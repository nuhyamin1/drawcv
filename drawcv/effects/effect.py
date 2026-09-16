"""Base effect abstraction for DrawCV post-processing visual effects."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from drawcv.core.bounds import BoundingBox


class Effect(ABC):
    """Abstract base class for post-processing visual effects applied to drawables, groups, or layers."""

    @property
    @abstractmethod
    def effect_type(self) -> str:
        """Stable serialized string identifier (e.g. 'blur', 'shadow', 'glow')."""
        pass

    @abstractmethod
    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Calculate outgoing visual bounds given incoming visual bounds."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        pass

    def _validate(self) -> None:
        """Validate effect parameters in-place. Subclasses override to enforce invariants."""
        pass
