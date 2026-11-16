"""Peças compartilhadas pelos scripts a partir da Fase 4: carregar a partição com as
imagens, alinhar e avaliar um mapa de anomalia com o mesmo protocolo para todas as
variantes (uma mudança por vez, o resto fixo)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from . import metrics
from .io import list_samples, load_sample
from .reference import circular_roi, translate
from .split import load_partition
from .stats import object_disk


@dataclass
class Dataset:
    """Imagens em cinza (uint8, 256 px, NÃO alinhadas), máscaras GT e o deslocamento
    que alinha cada garrafa ao centro médio da referência."""

    part: pd.DataFrame
    gray: dict[str, np.ndarray]
    gt: dict[str, np.ndarray]
    shift: dict[str, tuple[float, float]]
    target: tuple[float, float]
    radius: float
    roi: np.ndarray = field(repr=False, default=None)

    def ids(self, role: str, label: int | None = None) -> list[str]:
        sel = self.part.role == role
        if label is not None:
            sel &= self.part.label == label
        return self.part.loc[sel, "sample_id"].tolist()

    def aligned(self, img: np.ndarray, sid: str) -> np.ndarray:
        """Translada uma imagem (já filtrada) de `sid` para o centro alvo, em float."""
        dx, dy = self.shift[sid]
        return translate(img, dx, dy).astype(np.float64)

    def aligned_gt(self, sid: str) -> np.ndarray:
        dx, dy = self.shift[sid]
        return translate(self.gt[sid], dx, dy, is_mask=True)


def load_dataset(cfg: dict) -> Dataset:
    size = cfg["image"]["size"]
    part = load_partition(cfg["root"] / "results" / "fase2" / "particao.csv", list_samples(cfg["category_dir"]))
    gray, gt, disk = {}, {}, {}
    for _, row in tqdm(part.iterrows(), total=len(part), desc="carregando"):
        img, mask = load_sample(row, size=size)
        gray[row.sample_id] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gt[row.sample_id] = mask
        disk[row.sample_id] = object_disk(gray[row.sample_id])
    ref = part.loc[part.role == "reference", "sample_id"]
    target = tuple(np.mean([disk[i][:2] for i in ref], axis=0))
    radius = float(np.mean([disk[i][2] for i in ref]))
    shift = {sid: (target[0] - d[0], target[1] - d[1]) for sid, d in disk.items()}
    roi = circular_roi((size, size), target, cfg["reference"]["roi_frac"] * radius)
    return Dataset(part, gray, gt, shift, target, radius, roi)


def evaluate_map(ds: Dataset, anomaly: dict[str, np.ndarray], percentile: float) -> dict:
    """Protocolo único para comparar mapas de anomalia (sem tocar na avaliação).

    anomaly: sample_id -> mapa (alinhado), para calibração e desenvolvimento.
    Limiar = percentil do mapa nas boas de calibração (dentro da ROI).

    Devolve:
      thr                  limiar calibrado
      auroc_<tipo>, auroc  AUROC de pixel nas defeituosas do desenvolvimento (independe do limiar)
      sep                  separação = z mediano dentro do defeito / limiar (> 1: defeito típico passa)
      dice/precision/recall médias no desenvolvimento com o limiar calibrado
      cal_components       componentes espúrios por imagem boa de calibração (ruído residual)
    """
    roi = ds.roi > 0
    cal = ds.ids("calibration")
    thr = float(np.percentile(np.concatenate([anomaly[s][roi] for s in cal]), percentile))
    out = {"thr": thr}
    out["cal_components"] = float(np.mean([metrics.n_components((anomaly[s] >= thr) & roi) for s in cal]))

    dev = ds.part[ds.part.role == "development"]
    zs, ys, per = [], [], []
    by_type: dict[str, tuple[list, list]] = {}
    inside = []
    for _, row in dev.iterrows():
        z, g = anomaly[row.sample_id], ds.aligned_gt(row.sample_id)
        zs.append(z[roi])
        ys.append(g[roi] > 0)
        by_type.setdefault(row.defect_type, ([], []))
        by_type[row.defect_type][0].append(z[roi])
        by_type[row.defect_type][1].append(g[roi] > 0)
        inside.append(np.median(z[(g > 0) & roi]))
        per.append(metrics.image_metrics(((z >= thr) & roi).astype(np.uint8), g, ds.roi))
    out["auroc"] = roc_auc_score(np.concatenate(ys), np.concatenate(zs))
    for t, (zz, yy) in sorted(by_type.items()):
        out[f"auroc_{t}"] = roc_auc_score(np.concatenate(yy), np.concatenate(zz))
    out["sep"] = float(np.median(inside) / thr)
    per = pd.DataFrame(per)
    for m in ("dice", "precision", "recall", "n_pred_components"):
        out[m] = per[m].mean()
    return out
