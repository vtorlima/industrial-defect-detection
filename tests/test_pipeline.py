"""Protocolo de avaliação (`evaluate_map`) e alinhamento do `Dataset`, com dados sintéticos."""

import numpy as np
import pytest

from pdi_defeitos.pipeline import evaluate_map

Q = 99.5  # mesmo percentil do config.yaml


def _noise_maps(ds, seed=0):
    """Mapa de ruído uniforme em [0, 1) para calibração e desenvolvimento."""
    rng = np.random.default_rng(seed)
    ids = ds.ids("calibration") + ds.ids("development")
    return {s: rng.uniform(0, 1, ds.roi.shape) for s in ids}


def _separable_maps(ds, seed=0):
    """Ruído em [0, 1) e +10 dentro do defeito: todo pixel de defeito fica acima de todo pixel normal."""
    maps = _noise_maps(ds, seed)
    for s in ds.ids("development"):
        maps[s] = maps[s] + 10.0 * (ds.aligned_gt(s) > 0)
    return maps


def test_separable_map_gives_auroc_one_and_high_dice(make_dataset):
    ds = make_dataset()
    r = evaluate_map(ds, _separable_maps(ds), Q)
    assert r["auroc"] == pytest.approx(1.0)
    assert r["auroc_broken_large"] == pytest.approx(1.0)
    assert r["auroc_contamination"] == pytest.approx(1.0)
    assert r["thr"] < 1.0            # o limiar vem do ruído das boas, não do defeito
    assert r["recall"] == pytest.approx(1.0)
    assert r["dice"] > 0.9           # só sobra o ~0,5% de ruído acima do percentil 99,5
    assert r["sep"] > 1.0            # z mediano do defeito acima do limiar


def test_random_map_gives_auroc_near_half(make_dataset):
    ds = make_dataset()
    r = evaluate_map(ds, _noise_maps(ds, seed=1), Q)
    assert r["auroc"] == pytest.approx(0.5, abs=0.05)
    assert r["dice"] < 0.1


def test_separable_map_with_shift_uses_aligned_gt(make_dataset):
    """Com deslocamento, a GT comparada é a alinhada (a mesma usada para montar o mapa)."""
    ds = make_dataset(shift=(3.0, -2.0))
    r = evaluate_map(ds, _separable_maps(ds), Q)
    assert r["auroc"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)


def test_threshold_comes_only_from_calibration(make_dataset):
    ds = make_dataset()
    maps = _separable_maps(ds)
    base = evaluate_map(ds, maps, Q)
    inside = np.argwhere(ds.roi > 0)

    # Um pixel absurdo numa imagem de desenvolvimento: o limiar não se mexe.
    dev = ds.ids("development")[0]
    changed = dict(maps)
    changed[dev] = maps[dev].copy()
    changed[dev][tuple(inside[0])] = 1e6
    r = evaluate_map(ds, changed, Q)
    assert r["thr"] == base["thr"]
    assert r["cal_components"] == base["cal_components"]

    # Controle: o mesmo pixel numa imagem de calibração muda o limiar.
    cal = ds.ids("calibration")[0]
    changed = dict(maps)
    changed[cal] = maps[cal].copy()
    lowest = min(map(tuple, inside), key=lambda p: maps[cal][p])  # menor valor da ROI vira o maior
    changed[cal][lowest] = 1e6
    assert evaluate_map(ds, changed, Q)["thr"] > base["thr"]


def test_metrics_ignore_everything_outside_roi(make_dataset):
    ds = make_dataset()
    maps = _separable_maps(ds)
    base = evaluate_map(ds, maps, Q)
    outside = ds.roi == 0

    # Fora da ROI: mapa absurdo em todas as imagens e GT "defeituosa" em todo o fundo.
    dirty = {s: np.where(outside, 1e6, m) for s, m in maps.items()}
    for s in ds.ids("development"):
        ds.gt[s] = np.where(outside, 1, ds.gt[s]).astype(np.uint8)
    r = evaluate_map(ds, dirty, Q)
    assert r.keys() == base.keys()
    for k in base:
        assert r[k] == pytest.approx(base[k]), k


def test_metrics_do_change_inside_roi(make_dataset):
    """Controle do teste anterior: a mesma sujeira dentro da ROI muda o resultado."""
    ds = make_dataset()
    maps = _separable_maps(ds)
    base = evaluate_map(ds, maps, Q)
    dev = ds.ids("development")[0]
    normal = (ds.roi > 0) & (ds.gt[dev] == 0)
    dirty = dict(maps)
    dirty[dev] = np.where(normal, 1e6, maps[dev])  # todo o normal da ROI vira falso positivo
    r = evaluate_map(ds, dirty, Q)
    assert r["auroc"] < base["auroc"]
    assert r["precision"] < base["precision"]


def test_ids_filters_by_role_and_label(make_dataset):
    ds = make_dataset(n_cal=4, n_dev_per_type=2)
    assert len(ds.ids("calibration")) == 4
    assert len(ds.ids("development")) == 4
    assert ds.ids("development", label=0) == []
    assert ds.ids("calibration", label=0) == ds.ids("calibration")


@pytest.mark.parametrize("shift", [(3.0, -2.0), (0.4, 0.6), (-1.5, 2.5)])
def test_aligned_gt_stays_binary(make_dataset, shift):
    ds = make_dataset(shift=shift)
    for s in ds.ids("development"):
        out = ds.aligned_gt(s)
        assert out.dtype == np.uint8 and out.shape == ds.gt[s].shape
        assert set(np.unique(out)) <= {0, 1}


def test_aligned_gt_moves_mask_by_the_shift(make_dataset):
    ds = make_dataset(shift=(3.0, -2.0))  # dx = +3 colunas, dy = -2 linhas
    s = ds.ids("development")[0]
    assert np.array_equal(ds.aligned_gt(s), np.roll(ds.gt[s], (-2, 3), axis=(0, 1)))


def test_aligned_image_is_float_and_follows_the_mask(make_dataset):
    ds = make_dataset(shift=(3.0, -2.0))
    s = ds.ids("development")[0]
    out = ds.aligned(ds.gray[s], s)
    assert out.dtype == np.float64 and out.shape == ds.gray[s].shape
    assert np.all(out[ds.aligned_gt(s) > 0] == 200)  # o defeito (200) acompanha a máscara alinhada


def test_aligned_interpolates_but_aligned_gt_does_not(make_dataset):
    """Deslocamento subpixel: a imagem ganha níveis intermediários (bilinear), a máscara não."""
    ds = make_dataset(shift=(0.4, 0.6))
    s = ds.ids("development")[0]
    original = set(np.unique(ds.gray[s]).tolist())
    assert not set(np.unique(ds.aligned(ds.gray[s], s)).tolist()) <= original
    assert set(np.unique(ds.aligned_gt(s))) <= {0, 1}
