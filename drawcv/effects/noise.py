"""Spatially deterministic coordinate-hashed grain and noise effect."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.exceptions import ValidationError
from drawcv.core.validation import validate_integer, validate_real_number
from drawcv.effects.effect import Effect


def hash_noise_coords(x: np.ndarray, y: np.ndarray, seed: int, channel: int) -> np.ndarray:
    """Evaluate pure vectorized uint32 coordinate hash producing uniform pseudo-random noise in [-1.0, 1.0].

    Spatially invariant to ROI bounds, dimensions, evaluation order, and global RNG state.
    """
    x_u32 = (x.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    y_u32 = (y.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    seed_canonical = int(seed) % 0x100000000
    chan_canonical = int(channel) % 0x100000000
    seed_term = np.uint32((seed_canonical * 0x9E3779B9) & 0xFFFFFFFF)
    chan_term = np.uint32((chan_canonical * 0xC2B2AE3D) & 0xFFFFFFFF)

    h = (x_u32 * np.uint32(0x1B873593) ^
         y_u32 * np.uint32(0x85EBCA6B) ^
         seed_term ^
         chan_term)
    h ^= (h >> np.uint32(16))
    h *= np.uint32(0x85EBCA6B)
    h ^= (h >> np.uint32(13))
    h *= np.uint32(0xC2B2AE35)
    h ^= (h >> np.uint32(16))
    return (h.astype(np.float32) / 2147483647.5) - 1.0


@dataclass
class NoiseEffect(Effect):
    """Spatially deterministic film grain / noise post-processing effect.

    Generates pseudo-random straight-color perturbations as a pure function of:
        (canvas_x, canvas_y, seed, channel)

    Guarantees bit-identical output regardless of ROI cropping, rendering sequence,
    cloning, serialization round-trips, or global random state.
    Preserves original output pixel alpha. Transparent pixels remain strictly (0, 0, 0, 0).

    Attributes:
        amount: Maximum straight-color perturbation in range [0.0, 1.0] (default: 0.1).
        seed: Canonical integer seed (evaluated modulo 2^32, default: 0).
        monochrome: If True, applies identical delta across B, G, R. If False, generates independent channel noise.
    """
    amount: float = 0.1
    seed: int = 0
    monochrome: bool = True

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        self.amount = validate_real_number(self.amount, "amount")
        if not (0.0 <= self.amount <= 1.0):
            raise ValidationError(f"amount must be in range [0.0, 1.0], got {self.amount}")

        raw_seed = validate_integer(self.seed, "seed")
        self.seed = int(raw_seed) % 0x100000000

        if not isinstance(self.monochrome, bool):
            raise ValidationError(f"monochrome must be a boolean, got {type(self.monochrome).__name__}")

    @property
    def effect_type(self) -> str:
        return "noise"

    def expand_bounds(self, input_bounds: BoundingBox) -> BoundingBox:
        """Noise preserves visual bounds."""
        return input_bounds

    def get_sampling_padding(self) -> tuple[float, float, float, float]:
        """Pointwise noise requires no neighborhood sampling padding."""
        return (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation."""
        return {
            "type": "noise",
            "amount": float(self.amount),
            "seed": int(self.seed),
            "monochrome": bool(self.monochrome),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseEffect:
        """Construct a NoiseEffect from a dictionary."""
        if not isinstance(data, dict):
            raise ValidationError(f"NoiseEffect data must be a dict, got {type(data).__name__}")
        return cls(
            amount=data.get("amount", 0.1),
            seed=data.get("seed", 0),
            monochrome=data.get("monochrome", True),
        )
