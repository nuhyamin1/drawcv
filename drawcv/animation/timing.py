"""Timing specification model for temporal evaluation and progressive animation."""

from __future__ import annotations
import math
from typing import Any, Callable

from drawcv.animation.easing import get_easing
from drawcv.core.exceptions import SerializationError, ValidationError


class Timing:
    """Temporal timing and pacing configuration.

    Attributes:
        start_time: Start timestamp in seconds.
        duration: Active duration in seconds (duration >= 0).
        delay: Additional delay offset before starting (in seconds).
        speed: Playback speed multiplier (speed > 0).
        easing: Easing curve identifier (str) or runtime callable.
        loop: Whether the animation cycles infinitely.
    """

    def __init__(
        self,
        start_time: float | int = 0.0,
        duration: float | int = 1.0,
        delay: float | int = 0.0,
        speed: float | int = 1.0,
        easing: str | Callable[[float], float] = "linear",
        loop: bool = False,
    ):
        self._validate_params(start_time, duration, delay, speed, easing, loop)
        self.start_time = float(start_time)
        self.duration = float(duration)
        self.delay = float(delay)
        self.speed = float(speed)
        self.easing = easing
        self.loop = bool(loop)
        self._easing_fn = get_easing(easing)

    def _validate_params(
        self,
        start_time: Any,
        duration: Any,
        delay: Any,
        speed: Any,
        easing: Any,
        loop: Any,
    ) -> None:
        for name, val in (("start_time", start_time), ("duration", duration), ("delay", delay), ("speed", speed)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValidationError(f"Timing '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"Timing '{name}' must be finite, got {val}")

        if float(duration) < 0.0:
            raise ValidationError(f"Timing 'duration' must be non-negative, got {duration}")
        if float(speed) <= 0.0:
            raise ValidationError(f"Timing 'speed' must be strictly positive, got {speed}")
        if not isinstance(loop, bool):
            raise ValidationError(f"Timing 'loop' must be a boolean, got {type(loop).__name__}")

        if float(duration) == 0.0 and bool(loop):
            raise ValidationError("Looping animation cannot have duration=0 (infinite cycle rate)")

        if not isinstance(easing, str) and not callable(easing):
            raise ValidationError(f"Timing 'easing' must be a string or callable, got {type(easing).__name__}")

    @property
    def cycle_duration(self) -> float:
        """Effective duration of a single cycle accounting for speed."""
        return self.duration / self.speed

    @property
    def cycle_end_time(self) -> float:
        """Timestamp when the first cycle completes."""
        return self.start_time + self.delay + self.cycle_duration

    @property
    def end_time(self) -> float:
        """End timestamp (inf if looping, else cycle_end_time)."""
        if self.loop:
            return float("inf")
        return self.cycle_end_time

    def _compute_phase(self, time: float) -> tuple[bool, float]:
        """Compute whether active and the normalized phase u in [0, 1].

        Returns:
            (is_active, u)
        """
        t0 = self.start_time + self.delay
        if time < t0:
            return False, 0.0

        if self.duration == 0.0:
            return True, 1.0

        cd = self.cycle_duration
        elapsed = time - t0

        if not self.loop:
            if elapsed >= cd:
                return True, 1.0
            u = elapsed / cd
            return True, max(0.0, min(1.0, u))
        else:
            u = (elapsed % cd) / cd
            return True, max(0.0, min(1.0, u))

    def get_progress(self, time: float) -> float:
        """Normalized progress clamped to [0.0, 1.0] for progressive reveals."""
        is_active, u = self._compute_phase(time)
        if not is_active:
            return 0.0
        if u >= 1.0 and not self.loop:
            return 1.0
        eased = self._easing_fn(u)
        return float(max(0.0, min(1.0, eased)))

    def evaluate(self, time: float) -> float:
        """Raw eased value without [0.0, 1.0] clamp, allowing overshoot for property tracks."""
        is_active, u = self._compute_phase(time)
        if not is_active:
            return 0.0
        if u >= 1.0 and not self.loop:
            return 1.0
        return float(self._easing_fn(u))

    def to_dict(self) -> dict[str, Any]:
        """Serialize timing configuration to JSON dictionary."""
        if not isinstance(self.easing, str):
            raise SerializationError(
                "Custom callable easing cannot be serialized to JSON; use a registered string identifier"
            )
        return {
            "start_time": self.start_time,
            "duration": self.duration,
            "delay": self.delay,
            "speed": self.speed,
            "easing": self.easing,
            "loop": self.loop,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Timing:
        """Construct a Timing instance from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"Timing data must be a dict, got {type(data).__name__}")
        return cls(
            start_time=data.get("start_time", 0.0),
            duration=data.get("duration", 1.0),
            delay=data.get("delay", 0.0),
            speed=data.get("speed", 1.0),
            easing=data.get("easing", "linear"),
            loop=data.get("loop", False),
        )
