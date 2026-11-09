"""Métricas de segmentação com máscaras sintéticas de sobreposição conhecida."""

import numpy as np
import pandas as pd
import pytest

from pdi_defeitos import metrics


def _square(y, x, s=10, shape=(40, 40)):
    m = np.zeros(shape, np.uint8)
    m[y : y + s, x : x + s] = 1
    return m


def test_two_squares_known_overlap():
    pred, gt = _square(0, 0), _square(0, 5)  # sobreposição de 10x5 = 50 px
    tp, fp, fn = metrics.confusion(pred, gt)
    assert (tp, fp, fn) == (50, 50, 50)
    s = metrics.scores(tp, fp, fn)
    assert s["dice"] == pytest.approx(0.5) and s["iou"] == pytest.approx(1 / 3)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)


def test_roi_excludes_pixels():
    pred, gt = _square(0, 0), _square(0, 0)
    roi = np.zeros((40, 40), np.uint8)
    roi[:, :5] = 1
    assert metrics.confusion(pred, gt, roi) == (50, 0, 0)


def test_edge_cases():
    assert metrics.scores(0, 0, 0)["dice"] == 1.0          # nada previsto, nada real
    assert metrics.scores(0, 0, 30)["precision"] == 0.0    # previsão vazia numa defeituosa
    roi = np.ones((40, 40), np.uint8)
    good = metrics.image_metrics(_square(0, 0, 4), np.zeros((40, 40), np.uint8), roi)
    assert np.isnan(good["dice"]) and good["fp_frac"] == pytest.approx(16 / 1600)
    assert good["n_pred_components"] == 1 and good["empty_pred"] == 0


def test_summarize_mean_vs_aggregate():
    roi = np.ones((40, 40), np.uint8)
    big = metrics.image_metrics(_square(0, 0, 20), _square(0, 0, 20), roi)     # Dice 1, defeito grande
    small = metrics.image_metrics(np.zeros((40, 40), np.uint8), _square(30, 30, 2), roi)  # Dice 0, pequeno
    df = pd.DataFrame([big | {"g": "x"}, small | {"g": "x"}])
    s = metrics.summarize(df, by="g").loc["x"]
    assert s.dice_mean == pytest.approx(0.5)          # cada imagem pesa igual
    assert s.dice_agg > 0.95                           # agregada dominada pelo defeito grande
