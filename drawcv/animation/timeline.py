"""Timeline multi-track manager for synchronized scene animations."""

from __future__ import annotations
from typing import Any, Callable

from drawcv.animation.timing import Timing
from drawcv.animation.track import AnimationTrack
from drawcv.core.exceptions import ValidationError


class Timeline:
    """Manages an ordered collection of AnimationTracks synchronized to a common timeline."""

    def __init__(self, tracks: list[AnimationTrack] | None = None):
        self.tracks: list[AnimationTrack] = list(tracks) if tracks is not None else []

    def add_track(self, track: AnimationTrack) -> AnimationTrack:
        """Add an AnimationTrack to the timeline."""
        if not isinstance(track, AnimationTrack):
            raise ValidationError(f"Expected AnimationTrack, got {type(track).__name__}")
        self.tracks.append(track)
        return track

    def animate(
        self,
        target: Any,
        property_path: str,
        start_value: Any,
        end_value: Any,
        duration: float | int = 1.0,
        easing: str | Callable[[float], float] = "linear",
        delay: float | int = 0.0,
        speed: float | int = 1.0,
        loop: bool = False,
    ) -> AnimationTrack:
        """Convenience method to construct, register, and return an AnimationTrack."""
        target_id = target.id if hasattr(target, "id") else str(target)
        timing = Timing(
            start_time=0.0,
            duration=duration,
            delay=delay,
            speed=speed,
            easing=easing,
            loop=loop,
        )
        track = AnimationTrack(
            target_id=target_id,
            property_path=property_path,
            start_value=start_value,
            end_value=end_value,
            timing=timing,
        )
        self.add_track(track)
        return track

    def evaluate(self, time: float, scene: Any) -> None:
        """Evaluate all tracks at the given timestamp against the provided scene graph.

        Tracks are evaluated in insertion order; later tracks targeting the same
        property will overwrite earlier ones.
        """
        for track in self.tracks:
            obj = scene.get(track.target_id)
            if obj is not None:
                track.evaluate(time, obj)

    @property
    def duration(self) -> float:
        """Maximum active endpoint across all tracks (inf if any track loops)."""
        if not self.tracks:
            return 0.0
        max_end = 0.0
        for track in self.tracks:
            if track.timing.loop:
                return float("inf")
            max_end = max(max_end, track.timing.end_time)
        return max_end

    def to_dict(self) -> dict[str, Any]:
        """Serialize timeline and all tracks to dictionary."""
        return {
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], scene: Any | None = None) -> Timeline:
        """Construct a Timeline from serialized dictionary representation."""
        if not isinstance(data, dict):
            raise ValidationError(f"Timeline data must be a dict, got {type(data).__name__}")
        track_list = [
            AnimationTrack.from_dict(td, scene=scene)
            for td in data.get("tracks", [])
        ]
        return cls(tracks=track_list)
