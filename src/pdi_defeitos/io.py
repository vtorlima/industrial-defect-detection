"""Leitura do dataset MVTec AD: listagem de amostras, imagens e máscaras.

Estrutura esperada de uma categoria (ex.: bottle):

    bottle/
      train/good/000.png ...
      test/good/000.png ...
      test/<tipo_defeito>/000.png ...
      ground_truth/<tipo_defeito>/000_mask.png ...

Lições da disciplina aplicadas aqui:
  * cv2 lê em BGR; por padrão devolvemos RGB (Guia PDI, seção 16).
  * Imagens são reduzidas com INTER_AREA; máscaras com INTER_NEAREST, porque
    interpolar uma máscara criaria valores entre 0 e 1 (Guia PDI, seção 6).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _imread(path: Path, flags: int) -> np.ndarray:
    """cv2.imread que funciona com acentos no caminho (problema comum no Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Não consegui ler a imagem: {path}")
    return img


def list_samples(category_dir: Path | str) -> pd.DataFrame:
    """Lista todas as imagens da categoria em um DataFrame.

    Colunas:
      sample_id   ex.: "test/broken_large/000"
      split       "train" ou "test"
      defect_type "good" ou o nome do defeito
      label       0 = normal, 1 = defeituosa
      image_path  caminho da imagem
      mask_path   caminho da máscara GT (None para imagens boas)
    """
    category_dir = Path(category_dir)
    if not category_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta da categoria não existe: {category_dir}. "
            "Baixe a categoria do MVTec AD e extraia nessa pasta."
        )

    rows = []
    for split in ("train", "test"):
        split_dir = category_dir / split
        if not split_dir.is_dir():
            continue
        for type_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = type_dir.name
            for img_path in sorted(type_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                mask_path = None
                if defect_type != "good":
                    candidate = category_dir / "ground_truth" / defect_type / f"{img_path.stem}_mask.png"
                    mask_path = candidate if candidate.is_file() else None
                rows.append(
                    {
                        "sample_id": f"{split}/{defect_type}/{img_path.stem}",
                        "split": split,
                        "defect_type": defect_type,
                        "label": int(defect_type != "good"),
                        "image_path": img_path,
                        "mask_path": mask_path,
                    }
                )
    return pd.DataFrame(rows)


def load_image(path: Path | str, size: int | None = None, mode: str = "rgb") -> np.ndarray:
    """Lê uma imagem como uint8.

    mode: "rgb" (padrão), "bgr" (convenção do OpenCV) ou "gray".
    size: se dado, redimensiona para size x size com INTER_AREA.
    """
    if mode == "gray":
        img = _imread(Path(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = _imread(Path(path), cv2.IMREAD_COLOR)
        if mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode != "bgr":
            raise ValueError(f"mode inválido: {mode!r} (use 'rgb', 'bgr' ou 'gray')")
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_mask(path: Path | str | None, shape: tuple[int, int], size: int | None = None) -> np.ndarray:
    """Lê a máscara GT como uint8 com valores {0, 1}.

    path=None (imagem boa) devolve uma máscara toda zero.
    shape: (altura, largura) da imagem original, usado quando path é None.
    size: se dado, redimensiona com INTER_NEAREST.
    """
    if path is None or (isinstance(path, float) and np.isnan(path)):
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        mask = (_imread(Path(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if size is not None:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask


def load_sample(row: pd.Series, size: int | None = None, mode: str = "rgb") -> tuple[np.ndarray, np.ndarray]:
    """Atalho: devolve (imagem, máscara) de uma linha de `list_samples`."""
    img = load_image(row["image_path"], size=size, mode=mode)
    mask = load_mask(row["mask_path"], shape=img.shape[:2], size=None)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, mask
