"""Filtros, Sobel à mão contra o OpenCV e bordas novas do Canny."""

import cv2
import numpy as np
import pytest

from pdi_defeitos.preprocess import (DENOISERS, canny_new_edges, convolve2d, denoise,
                                     sobel_magnitude, sobel_magnitude_numpy)


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    g = np.full((64, 64), 100, np.uint8)
    cv2.circle(g, (32, 32), 15, 200, -1)
    return np.clip(g + rng.normal(0, 5, g.shape), 0, 255).astype(np.uint8)


def test_convolve2d_matches_filter2d(img):
    k = np.arange(9, dtype=np.float64).reshape(3, 3)
    ref = cv2.filter2D(img.astype(np.float64), -1, k, borderType=cv2.BORDER_REFLECT_101)
    assert np.allclose(convolve2d(img, k), ref)


def test_sobel_numpy_matches_opencv(img):
    assert np.allclose(sobel_magnitude_numpy(img), sobel_magnitude(img))


def test_sobel_zero_on_flat_and_strong_on_edge():
    flat = np.full((20, 20), 80, np.uint8)
    assert sobel_magnitude(flat).max() == 0
    step = np.zeros((20, 20), np.uint8)
    step[:, 10:] = 100
    mag = sobel_magnitude(step)
    assert mag[10, 9:11].min() == pytest.approx(400)  # 4 x degrau de 100 no núcleo [1 2 1]


@pytest.mark.parametrize("method,params", [("none", {}), ("gaussian", {"sigma": 2}), ("median", {"ksize": 5}),
                                           ("bilateral", {"d": 9, "sigma_color": 30, "sigma_space": 3})])
def test_denoise_reduces_noise_and_keeps_shape(img, method, params):
    out = denoise(img, method, **params)
    assert out.shape == img.shape and out.dtype == np.uint8
    flat_region = (slice(2, 12), slice(2, 12))
    if method != "none":
        assert out[flat_region].std() < img[flat_region].std()
    assert method in DENOISERS


def test_denoise_rejects_unknown(img):
    with pytest.raises(ValueError):
        denoise(img, "wiener")


def test_canny_new_edges_ignores_reference_edges():
    ref = np.zeros((64, 64), np.uint8)
    cv2.circle(ref, (32, 32), 20, 200, -1)
    ref_edges = cv2.Canny(ref, 40, 80)
    img = ref.copy()
    img[5:15, 5:15] = 200  # objeto novo, longe do círculo
    new = canny_new_edges(img, ref_edges, high=80, ratio=2, tolerance_px=2)
    assert new[:20, :20].sum() > 0       # borda do quadrado novo aparece
    assert new[20:, 20:].sum() == 0      # borda do círculo (existe na referência) não aparece
