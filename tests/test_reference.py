"""Alinhamento, modelo de referência e Otsu manual."""

import cv2
import numpy as np
import pytest

from pdi_defeitos.reference import ReferenceModel, align_to, circular_roi, otsu_threshold, translate
from pdi_defeitos.stats import object_disk


def _bottle(center=(64, 64), size=128):
    img = np.full((size, size), 255, np.uint8)
    cv2.circle(img, center, 40, 60, -1)
    cv2.circle(img, center, 25, 120, 3)  # um "anel"
    return img


def test_translate_integer_shift_and_mask_stays_binary():
    m = np.zeros((20, 20), np.uint8)
    m[5, 5] = 1
    out = translate(m, 3, 2, is_mask=True)
    assert out[7, 8] == 1 and out.sum() == 1
    assert set(np.unique(translate(m, 0.4, 0.6, is_mask=True))) <= {0, 1}


def test_align_to_moves_center_to_target():
    img = _bottle(center=(60, 67))
    aligned, _, (dx, dy) = align_to(img, (64.0, 64.0))
    cx, cy, _ = object_disk(aligned)
    assert (cx, cy) == (pytest.approx(64, abs=0.3), pytest.approx(64, abs=0.3))
    assert (dx, dy) == (pytest.approx(4, abs=0.3), pytest.approx(-3, abs=0.3))


def test_anomaly_map_formula_and_roi():
    rng = np.random.default_rng(0)
    stack = 100 + rng.normal(0, 3, (50, 32, 32))
    model = ReferenceModel.fit(stack, center=(16, 16), radius=10, roi_frac=1.0, eps=2.0)
    img = model.mean.copy()
    img[16, 16] += 50
    z = model.anomaly_map(img)
    assert z[16, 16] == pytest.approx(50 / (model.std[16, 16] + 2))
    assert z[0, 0] == 0  # fora da ROI
    assert model.roi.sum() == circular_roi((32, 32), (16, 16), 10).sum()


def test_otsu_matches_opencv_on_uint8():
    rng = np.random.default_rng(1)
    img = np.concatenate([rng.normal(60, 10, 3000), rng.normal(170, 15, 2000)]).clip(0, 255).astype(np.uint8)
    t_cv, _ = cv2.threshold(img.reshape(50, 100), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert otsu_threshold(img, bins=256, value_range=(0, 256)) == t_cv + 1


def test_otsu_on_real_values_separates_modes():
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(1, 0.2, 5000), rng.normal(6, 0.5, 500)])
    t = otsu_threshold(v)
    assert abs(int((v >= t).sum()) - 500) <= 5  # separa os dois modos (500 valores altos)
