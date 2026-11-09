"""Testes das medidas da Fase 1 com imagens sintéticas de resposta conhecida."""

import cv2
import numpy as np
import pytest

from pdi_defeitos import stats


def _disk_image(size=128, center=(64, 64), radius=40):
    """Disco escuro (50) sobre fundo branco (255), como uma garrafa vista de cima."""
    img = np.full((size, size), 255, np.uint8)
    cv2.circle(img, center, radius, 50, -1)
    return img


def test_noise_sigma_recovers_known_noise():
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), 128.0)
    assert stats.estimate_noise_sigma(flat.astype(np.uint8)) == pytest.approx(0.0, abs=1e-9)
    noisy = np.clip(flat + rng.normal(0, 5, flat.shape), 0, 255).astype(np.uint8)
    assert stats.estimate_noise_sigma(noisy) == pytest.approx(5.0, rel=0.1)


def test_object_disk_finds_center_and_radius():
    cx, cy, r = stats.object_disk(_disk_image(center=(70, 60), radius=40))
    assert (cx, cy) == (pytest.approx(70, abs=0.5), pytest.approx(60, abs=0.5))
    assert r == pytest.approx(40, rel=0.03)


def test_defect_geometry_square_and_two_components():
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 1  # quadrado 20x20 centrado em (49.5, 49.5)
    g = stats.defect_geometry(mask, center=(49.5, 49.5), radius=50)
    assert g["area_px"] == 400 and g["n_components"] == 1
    assert g["radial_pos"] == pytest.approx(0.0, abs=1e-9)
    mask[0:5, 0:5] = 1
    assert stats.defect_geometry(mask, (49.5, 49.5), 50)["n_components"] == 2


def test_defect_geometry_empty_mask():
    g = stats.defect_geometry(np.zeros((10, 10), np.uint8), (5, 5), 5)
    assert g["area_px"] == 0 and np.isnan(g["radial_pos"])


def test_ring_contrast_sign_and_color():
    lab = np.zeros((60, 60, 3), np.uint8)
    lab[..., 0], lab[..., 1], lab[..., 2] = 100, 128, 128
    mask = np.zeros((60, 60), np.uint8)
    mask[20:40, 20:40] = 1
    lab[20:40, 20:40] = (130, 131, 132)  # defeito mais claro e levemente colorido
    c = stats.ring_contrast(lab, mask, ring_px=5)
    assert c["dL"] == pytest.approx(30)
    assert c["dab"] == pytest.approx(5.0)  # hypot(3, 4)


def test_radial_profile_bins():
    rd = stats.radial_distance((101, 101), (50, 50), 50)
    values = (rd < 0.5).astype(float)
    prof = stats.radial_profile(values, rd, np.array([0.0, 0.4, 0.6, 1.0]), stat=np.mean)
    assert prof[0] == 1.0 and prof[2] == 0.0 and 0 < prof[1] < 1
