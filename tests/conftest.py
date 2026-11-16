"""Cria um mini dataset falso com a mesma estrutura do MVTec AD.

Os testes rodam sem baixar nada, em qualquer computador.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


@pytest.fixture
def fake_category(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec" / "bottle"
    rng = np.random.default_rng(0)
    for i in range(3):
        _write(root / "train" / "good" / f"{i:03d}.png", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    _write(root / "test" / "good" / "000.png", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for dtype in ("broken_large", "contamination"):
        for i in range(2):
            _write(root / "test" / dtype / f"{i:03d}.png", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            mask = np.zeros((64, 64), np.uint8)
            mask[10:30, 20:40] = 255
            _write(root / "ground_truth" / dtype / f"{i:03d}_mask.png", mask)
    return root


@pytest.fixture
def make_dataset():
    """Construtor de um `pipeline.Dataset` sintético (sem disco, sem MVTec).

    Garrafa = disco cinza em fundo branco; ROI = disco de raio 24 no centro (32, 32).
    Cada defeituosa do desenvolvimento tem um defeito quadrado de 12x12 px (GT em
    {0, 1}, como `io.load_mask`), sempre dentro da ROI mesmo depois do deslocamento.
    `shift`: deslocamento (dx, dy) aplicado a todas as amostras; o padrão (0, 0)
    faz `aligned_gt` devolver a própria GT.
    """
    import pandas as pd

    from pdi_defeitos.pipeline import Dataset
    from pdi_defeitos.reference import circular_roi

    def _make(n_cal: int = 4, n_dev_per_type: int = 2, types=("broken_large", "contamination"),
              shift: tuple[float, float] = (0.0, 0.0), size: int = 64) -> Dataset:
        center, radius = (size / 2, size / 2), 0.4 * size
        rows = [(f"train/good/{i:03d}", "train", "good", 0, "reference") for i in range(2)]
        rows += [(f"train/good/{i + 2:03d}", "train", "good", 0, "calibration") for i in range(n_cal)]
        rows += [(f"test/{t}/{i:03d}", "test", t, 1, "development") for t in types for i in range(n_dev_per_type)]
        part = pd.DataFrame(rows, columns=["sample_id", "split", "defect_type", "label", "role"])

        corners = [(20, 20), (24, 30), (30, 22), (28, 28)]  # canto superior esquerdo (linha, coluna)
        gray, gt = {}, {}
        for k, row in enumerate(part.itertuples()):
            img = np.full((size, size), 255, np.uint8)
            cv2.circle(img, (size // 2, size // 2), int(radius), 90, -1)
            mask = np.zeros((size, size), np.uint8)
            if row.label == 1:
                r, c = corners[k % len(corners)]
                mask[r : r + 12, c : c + 12] = 1
                img[mask > 0] = 200
            gray[row.sample_id], gt[row.sample_id] = img, mask
        return Dataset(
            part=part, gray=gray, gt=gt,
            shift={s: shift for s in part.sample_id},
            target=center, radius=radius,
            roi=circular_roi((size, size), center, 0.75 * radius),
        )

    return _make
