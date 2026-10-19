import numpy as np

from pdi_defeitos.config import load_config
from pdi_defeitos.io import count_by_folder, list_samples, load_image, load_mask, load_sample


def test_list_samples_structure(fake_category):
    df = list_samples(fake_category)
    assert len(df) == 3 + 1 + 4
    assert set(df.split) == {"train", "test"}
    assert df.loc[df.defect_type == "good", "mask_path"].isna().all()
    assert df.loc[df.label == 1, "mask_path"].notna().all()


def test_load_image_modes(fake_category):
    path = fake_category / "train" / "good" / "000.png"
    rgb, bgr = load_image(path), load_image(path, mode="bgr")
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[..., 0], bgr[..., 2])  # RGB é o BGR invertido
    assert load_image(path, mode="gray").ndim == 2
    assert load_image(path, size=32).shape == (32, 32, 3)


def test_mask_is_binary_after_resize(fake_category):
    path = fake_category / "ground_truth" / "broken_large" / "000_mask.png"
    mask = load_mask(path, shape=(64, 64), size=50)  # tamanho que não divide 64
    assert set(np.unique(mask)) <= {0, 1}
    assert load_mask(None, shape=(64, 64)).sum() == 0


def test_load_sample_aligns_shapes(fake_category):
    row = list_samples(fake_category).query("label == 1").iloc[0]
    img, mask = load_sample(row, size=32)
    assert img.shape[:2] == mask.shape == (32, 32)
    assert mask.sum() > 0


def test_count_by_folder(fake_category):
    counts = count_by_folder(fake_category)
    assert counts["train/good"] == 3
    assert counts["ground_truth/contamination"] == 2


def test_config_resolves_paths():
    cfg = load_config()
    assert cfg["category"] == "bottle"
    assert cfg["category_dir"].name == "bottle"
    assert (cfg["root"] / "config.yaml").is_file()
