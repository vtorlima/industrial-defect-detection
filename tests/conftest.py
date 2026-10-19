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
