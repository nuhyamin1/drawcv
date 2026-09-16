"""Base effect abstraction for DrawCV post-processing visual effects."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from drawcv.core.bounds import BoundingBox


class Effect(ABC):
    """Abstract base class for post-processing visual effects applied to drawables, groups, or layers."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        orig_post_init = getattr(cls, "__post_init__", None)

        def wrapped_post_init(self, *args, **kwargs):
            if orig_post_init is not None:
                orig_post_init(self, *args, **kwargs)
            else:
                self._validate()
            object.__setattr__(self, "_initialized", True)

        cls.__post_init__ = wrapped_post_init

    def __setattr__(self, name: str, value: Any) -> None:
        from drawcv.core.validation import validated_setattr
        validated_setattr(self, name, value)

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

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Return filter sampling support padding (left, right, top, bottom) in screen pixels.

        Defaults to (0.0, 0.0, 0.0, 0.0) so existing effects remain source-compatible.
        """
        return (0.0, 0.0, 0.0, 0.0)

    def _validate(self) -> None:
        """Validate effect parameters in-place. Subclasses override to enforce invariants."""
        pass
