"""Testes da partição com um dataset sintético do mesmo tamanho do bottle (sem ler imagens)."""

import pandas as pd
import pytest

from pdi_defeitos.split import check_partition, load_partition, make_cv_folds, make_partition

COUNTS = {("train", "good"): 209, ("test", "good"): 20, ("test", "broken_large"): 20,
          ("test", "broken_small"): 22, ("test", "contamination"): 21}


@pytest.fixture
def samples():
    rows = []
    for (split, dtype), n in COUNTS.items():
        for i in range(n):
            rows.append({"sample_id": f"{split}/{dtype}/{i:03d}", "split": split, "defect_type": dtype,
                         "label": int(dtype != "good"), "image_path": f"/x/{split}/{dtype}/{i:03d}.png",
                         "mask_path": None if dtype == "good" else f"/x/gt/{dtype}/{i:03d}_mask.png"})
    return pd.DataFrame(rows)


def _part(samples, seed=0):
    return make_partition(samples, n_reference=120, n_calibration=30, n_dev_per_type=3, seed=seed)


def test_counts_per_role(samples):
    part = _part(samples)
    counts = part.role.value_counts()
    assert counts["reference"] == 120 and counts["calibration"] == 30 and counts["development"] == 9
    assert counts["evaluation"] == len(samples) - 159
    ev = part[part.role == "evaluation"]
    assert (ev.label == 0).sum() == 59 + 20 and (ev.label == 1).sum() == 54


def test_partition_is_valid_and_deterministic(samples):
    a, b = _part(samples, seed=0), _part(samples, seed=0)
    assert check_partition(a) == []
    assert a.role.tolist() == b.role.tolist()
    assert a.role.tolist() != _part(samples, seed=1).role.tolist()


def test_reference_and_calibration_only_train_good(samples):
    part = _part(samples)
    rc = part[part.role.isin(["reference", "calibration"])]
    assert (rc.split == "train").all() and (rc.label == 0).all()


def test_dev_has_each_defect_type(samples):
    dev = _part(samples).query("role == 'development'")
    assert dev.defect_type.value_counts().to_dict() == {"broken_large": 3, "broken_small": 3, "contamination": 3}


def test_check_partition_catches_leak(samples):
    part = _part(samples)
    part.loc[part.index[(part.role == "evaluation") & (part.label == 1)][0], "role"] = "reference"
    assert any("reference" in p for p in check_partition(part))


def test_cv_folds_are_stratified_and_complete(samples):
    ev = _part(samples).query("role == 'evaluation'")
    folds = make_cv_folds(ev, n_splits=5, seeds=[0, 1])
    for seed, f in folds.groupby("seed"):
        assert sorted(f.sample_id) == sorted(ev.sample_id)  # cada imagem em exatamente um fold de teste
        per_fold = f.merge(ev, on="sample_id").groupby(["fold", "defect_type"]).size().unstack()
        assert (per_fold.max() - per_fold.min()).max() <= 1  # tipos distribuídos igualmente


def test_load_partition_restores_paths(samples, tmp_path):
    part = _part(samples)
    path = tmp_path / "particao.csv"
    part[["sample_id", "split", "defect_type", "label", "role"]].to_csv(path, index=False)
    back = load_partition(path, samples)
    assert back.image_path.notna().all() and back.role.tolist() == part.role.tolist()
