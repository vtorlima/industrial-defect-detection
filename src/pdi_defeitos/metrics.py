"""Métricas de segmentação: máscara prevista × ground truth, contadas pixel a pixel dentro da ROI.

    Dice = 2TP / (2TP + FP + FN)      IoU = TP / (TP + FP + FN)
    Precisão = TP / (TP + FP)         Recall = TP / (TP + FN)

Decisões (registradas no relatório):
  * Tudo é contado só dentro da ROI; o fundo branco inflaria os acertos.
  * Imagens boas têm GT vazia, então Dice não é definido. Nelas medimos a fração
    de pixels falsos positivos e o número de componentes espúrios.
  * Máscara prevista vazia numa defeituosa: precisão indefinida, tratada como 0
    (e contada em `empty_pred`). GT e previsão vazias: Dice e IoU = 1.
  * Reportamos a média por imagem (cada defeito pesa igual) e a agregada (soma
    TP, FP e FN de todas as imagens; dominada pelos defeitos grandes).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def confusion(pred: np.ndarray, gt: np.ndarray, roi: np.ndarray | None = None) -> tuple[int, int, int]:
    """Contagens (TP, FP, FN) de pixels, restritas à ROI se dada."""
    p, g = pred > 0, gt > 0
    if roi is not None:
        r = roi > 0
        p, g = p & r, g & r
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Dice, IoU, precisão e recall a partir das contagens."""
    if tp + fp + fn == 0:  # GT e previsão vazias: concordância perfeita
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    return {
        "dice": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def n_components(mask: np.ndarray) -> int:
    """Número de componentes conexos (vizinhança 8)."""
    n, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return n - 1


def image_metrics(pred: np.ndarray, gt: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    """Todas as métricas de uma imagem. Para boas, Dice/IoU/precisão/recall ficam NaN."""
    tp, fp, fn = confusion(pred, gt, roi)
    pred_roi = (pred > 0) & (roi > 0)
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_frac": fp / int((roi > 0).sum()),
        "n_pred_components": n_components(pred_roi),
        "empty_pred": int(not pred_roi.any()),
    }
    if (gt > 0).any():
        out |= scores(tp, fp, fn)
    else:
        out |= {"dice": np.nan, "iou": np.nan, "precision": np.nan, "recall": np.nan}
    return out


def summarize(per_image: pd.DataFrame, by: str = "defect_type") -> pd.DataFrame:
    """Resumo por grupo: médias por imagem e métricas agregadas (somando TP, FP, FN).

    per_image: uma linha por imagem com as colunas de `image_metrics` e a coluna `by`.
    """
    rows = []
    for name, g in per_image.groupby(by):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        agg = scores(tp, fp, fn) if (g.tp + g.fn).sum() else {}
        rows.append(
            {
                by: name,
                "n": len(g),
                "dice_mean": g.dice.mean(),
                "iou_mean": g.iou.mean(),
                "precision_mean": g.precision.mean(),
                "recall_mean": g.recall.mean(),
                "dice_agg": agg.get("dice", np.nan),
                "iou_agg": agg.get("iou", np.nan),
                "fp_frac_mean": g.fp_frac.mean(),
                "n_pred_components_mean": g.n_pred_components.mean(),
                "empty_pred": int(g.empty_pred.sum()),
            }
        )
    return pd.DataFrame(rows).set_index(by)
