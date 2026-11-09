"""Medidas da Fase 1 (exploração): o que os dados dizem antes de escolher qualquer técnica.

Cada função responde a uma pergunta que orienta uma decisão do plano:

  estimate_noise_sigma   Há ruído de sensor para o denoising remover?
  object_disk            A garrafa está sempre na mesma posição? (precisa alinhar?)
  defect_geometry        Qual o tamanho e a posição dos defeitos? (limita filtros,
                         elemento estruturante e área mínima de componente)
  ring_contrast          O defeito muda brilho, cor ou nenhum dos dois?
  radial_profile         Onde a garrafa boa varia naturalmente? (fonte de falsos positivos)

Todas trabalham com arrays NumPy e não leem arquivos; quem lê é `io`.
"""

from __future__ import annotations

import cv2
import numpy as np

# Núcleo de Immerkær (1996): diferença de dois Laplacianos. Anula superfícies
# suaves e deixa passar o ruído, por isso estima sigma sem confundir com bordas.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Estimativa rápida do desvio padrão do ruído gaussiano (método de Immerkær).

    sigma = sqrt(pi/2) * soma|I * N| / (6 (W-2)(H-2))

    Valores perto de 0,5 nível de cinza significam ruído desprezível.
    """
    g = gray.astype(np.float64)
    h, w = g.shape
    resp = cv2.filter2D(g, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).sum() / (6.0 * (w - 2) * (h - 2)))


def object_disk(gray: np.ndarray) -> tuple[float, float, float]:
    """Centro (x, y) e raio equivalente do objeto escuro sobre fundo claro.

    No bottle o fundo é branco (255) e a garrafa escura, então Otsu com
    THRESH_BINARY_INV separa os dois sem parâmetro manual. O centro é o
    centroide (momentos de ordem 1) e o raio vem da área: r = sqrt(A / pi).
    """
    _, obj = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.moments(obj, binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("Otsu não encontrou objeto na imagem.")
    return m["m10"] / m["m00"], m["m01"] / m["m00"], float(np.sqrt(m["m00"] / np.pi))


def radial_distance(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    """Mapa de distância de cada pixel ao centro, dividida pelo raio (0 = centro, 1 = raio)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return np.hypot(xx - center[0], yy - center[1]) / radius


def defect_geometry(mask: np.ndarray, center: tuple[float, float], radius: float) -> dict[str, float]:
    """Tamanho, número de componentes e posição radial de uma máscara binária.

    radius: raio de referência da garrafa (posição radial = distância / radius).
    Máscara vazia devolve área 0 e posição NaN.
    """
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    n_labels, _ = cv2.connectedComponents(mask, connectivity=8)
    out = {
        "area_px": area,
        "area_frac": area / mask.size,
        "n_components": n_labels - 1,
        "radial_pos": np.nan,
        "equiv_diameter_px": 2.0 * np.sqrt(area / np.pi),
    }
    if area:
        ys, xs = np.nonzero(mask)
        out["radial_pos"] = float(np.hypot(xs.mean() - center[0], ys.mean() - center[1]) / radius)
    return out


def ring_contrast(lab: np.ndarray, mask: np.ndarray, ring_px: int) -> dict[str, float]:
    """Diferença de brilho e de cor entre o defeito e um anel ao redor dele.

    lab: imagem em Lab do OpenCV (uint8: L em 0..255, a e b com 128 = neutro).
    O anel é a borda externa morfológica (mask dilatada) - mask, com largura ring_px.

    dL  > 0: defeito mais claro que a vizinhança.
    dab    : distância no plano (a, b), ou seja, mudança de cor sem contar brilho.
    """
    m = mask > 0
    if not m.any():
        return {"dL": np.nan, "dab": np.nan}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m
    lab = lab.astype(np.float64)
    d = lab[m].mean(axis=0) - lab[ring].mean(axis=0)
    return {"dL": float(d[0]), "dab": float(np.hypot(d[1], d[2]))}


def radial_profile(values: np.ndarray, rdist: np.ndarray, edges: np.ndarray, stat=np.median) -> np.ndarray:
    """Aplica `stat` aos valores de cada faixa radial [edges[i], edges[i+1])."""
    out = np.full(len(edges) - 1, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (rdist >= lo) & (rdist < hi)
        if sel.any():
            out[i] = stat(values[sel])
    return out
