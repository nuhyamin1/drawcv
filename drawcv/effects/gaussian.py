"""Gaussian kernel sizing and padding calculation helpers."""

from __future__ import annotations
import math


def gaussian_kernel_size_for_radius(radius: float) -> int:
    """Calculate the odd positive integer Gaussian kernel dimension for a given radius/sigma."""
    if radius <= 0.0:
        return 0
    return int(math.ceil(float(radius) * 3.0)) * 2 + 1


def gaussian_pad_for_radius(radius: float) -> int:
    """Calculate the support radius (half-kernel padding) for a Gaussian blur radius/sigma."""
    ksize = gaussian_kernel_size_for_radius(radius)
    return ksize // 2 if ksize > 0 else 0
