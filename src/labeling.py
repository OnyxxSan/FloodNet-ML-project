"""Utilities for converting FloodNet semantic masks into image-level labels."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_water_fraction(mask: np.ndarray, water_label: int = 1) -> float:
    """Return the fraction of pixels assigned to the water class."""
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr == water_label))


def build_image_label(mask: np.ndarray, threshold: float = 0.005, water_label: int = 1) -> int:
    """Convert a segmentation mask to a binary image label using a water fraction threshold."""
    water_fraction = compute_water_fraction(mask, water_label=water_label)
    return int(water_fraction > threshold)


def build_label_vector(masks: Iterable[np.ndarray], threshold: float = 0.005, water_label: int = 1) -> list[int]:
    """Convert a collection of masks to binary labels."""
    return [build_image_label(mask, threshold=threshold, water_label=water_label) for mask in masks]
