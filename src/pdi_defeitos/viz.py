"""Figuras usadas na exploração e, depois, nas galerias de erros do relatório."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_sample


def draw_mask_contour(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), thickness: int = 1) -> np.ndarray:
    """Copia a imagem e desenha o contorno da máscara por cima (não cobre o defeito)."""
    out = img_rgb.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def gallery(df: pd.DataFrame, size: int, n_per_type: int = 5, seed: int = 0) -> plt.Figure:
    """Grade com uma linha por tipo (good primeiro) e o contorno da máscara GT em vermelho.

    df: saída de `io.list_samples`. As amostras de cada linha são sorteadas com `seed`.
    """
    types = ["good"] + sorted(t for t in df.defect_type.unique() if t != "good")
    fig, axes = plt.subplots(len(types), n_per_type, figsize=(2.2 * n_per_type, 2.3 * len(types)))
    for r, t in enumerate(types):
        rows = df[df.defect_type == t].sample(n=n_per_type, random_state=seed)
        for c, (_, row) in enumerate(rows.iterrows()):
            img, mask = load_sample(row, size=size)
            ax = axes[r, c]
            ax.imshow(draw_mask_contour(img, mask))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(t, fontsize=10)
            ax.set_title(row.sample_id.split("/", 1)[1], fontsize=7)
    fig.tight_layout()
    return fig


def variability_maps(mean_map: np.ndarray, std_map: np.ndarray, title_std: str = "desvio padrão por pixel") -> plt.Figure:
    """Imagem média e desvio padrão por pixel das garrafas boas, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    axes[0].imshow(mean_map, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("imagem média (boas)")
    im = axes[1].imshow(std_map, cmap="magma")
    axes[1].set_title(title_std)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="níveis de cinza")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def panel_grid(rows: list[list[np.ndarray]], row_titles: list[str], col_titles: list[str], cmaps=None, vmax=None) -> plt.Figure:
    """Grade de imagens: uma linha por exemplo, uma coluna por etapa (imagem, mapa, máscara...).

    cmaps/vmax: por coluna (None = padrão). Usada nas galerias de resultados e de erros.
    """
    n_r, n_c = len(rows), len(col_titles)
    cmaps = cmaps or [None] * n_c
    vmax = vmax or [None] * n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.3 * n_c, 2.3 * n_r), squeeze=False)
    for r, imgs in enumerate(rows):
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(im, cmap=cmaps[c], vmin=0 if vmax[c] is not None else None, vmax=vmax[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=8)
    fig.tight_layout()
    return fig
