import numpy as np

from src.labeling import build_image_label, compute_water_fraction


def test_compute_water_fraction_all_water():
    mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    assert compute_water_fraction(mask, water_label=1) == 1.0


def test_compute_water_fraction_mixed_class():
    mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    assert abs(compute_water_fraction(mask, water_label=1) - 0.5) < 1e-9


def test_build_image_label_threshold():
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    assert build_image_label(mask, threshold=0.3, water_label=1) == 1
    assert build_image_label(mask, threshold=0.9, water_label=1) == 0
