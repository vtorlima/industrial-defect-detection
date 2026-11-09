"""Modelo de referência do normal (Fase 3): alinhamento, média e desvio por pixel, mapa de anomalia.

Ideia: com a garrafa sempre centrada e de cima, "defeito" = desvio do padrão normal
naquela posição. A diferença é dividida pela variação natural de cada pixel:

    z(x, y) = |I(x, y) - mu(x, y)| / (sigma(x, y) + eps)

Assim os anéis, que variam muito entre garrafas boas (Fase 1), pesam menos que o
fundo, que quase não varia. eps evita divisão por zero no fundo branco e impede
que pixels quase constantes gerem z enorme por diferenças de 1 nível.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .stats import object_disk


def translate(img: np.ndarray, dx: float, dy: float, is_mask: bool = False) -> np.ndarray:
    """Desloca a imagem (dx, dy) pixels. Subpixel com interpolação bilinear;
    máscaras com vizinho mais próximo para continuarem binárias. A borda que entra
    é preenchida com branco (cor do fundo) ou 0 (máscara)."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]
    if is_mask:
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def align_to(gray: np.ndarray, target: tuple[float, float], mask: np.ndarray | None = None):
    """Translada a imagem (e a máscara, se dada) para o centro da garrafa coincidir com `target`.

    Devolve (imagem_alinhada, mascara_alinhada_ou_None, (dx, dy)).
    """
    cx, cy, _ = object_disk(gray)
    dx, dy = target[0] - cx, target[1] - cy
    out_mask = translate(mask, dx, dy, is_mask=True) if mask is not None else None
    return translate(gray, dx, dy), out_mask, (dx, dy)


def circular_roi(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    """Máscara uint8 {0, 1} de um disco."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return (np.hypot(xx - center[0], yy - center[1]) <= radius).astype(np.uint8)


@dataclass
class ReferenceModel:
    """Média e desvio por pixel das garrafas boas de referência, já alinhadas."""

    mean: np.ndarray
    std: np.ndarray
    center: tuple[float, float]
    radius: float
    roi: np.ndarray
    eps: float

    @classmethod
    def fit(cls, images: np.ndarray, center: tuple[float, float], radius: float, roi_frac: float, eps: float):
        """images: pilha (n, h, w) em cinza, float, já alinhadas em `center`."""
        images = np.asarray(images, dtype=np.float64)
        roi = circular_roi(images.shape[1:], center, roi_frac * radius)
        return cls(images.mean(0), images.std(0), center, radius, roi, eps)

    def anomaly_map(self, gray: np.ndarray) -> np.ndarray:
        """Mapa z da imagem (já alinhada), zerado fora da ROI. Vetorizado: sem loops."""
        z = np.abs(gray.astype(np.float64) - self.mean) / (self.std + self.eps)
        return z * self.roi


def otsu_threshold(values: np.ndarray, bins: int = 256, value_range: tuple[float, float] | None = None) -> float:
    """Otsu implementado à mão (exercício da disciplina).

    Escolhe o corte que maximiza a variância entre classes
        sigma_B^2(k) = [mu_T w0(k) - mu(k)]^2 / (w0(k) [1 - w0(k)])
    com w0 = probabilidade acumulada e mu = média acumulada até a faixa k.

    Convenção: primeiro plano = valores >= limiar devolvido. Para uint8 com
    bins=256 e value_range=(0, 256), o resultado é o limiar do cv2.THRESH_OTSU + 1
    (o OpenCV marca valores > t; aqui marcamos >= t + 1, que é o mesmo conjunto).
    Funciona também com valores reais, como o mapa z.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if value_range is None:
        value_range = (float(values.min()), float(values.max()))
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    p = hist / hist.sum()
    levels = edges[:-1]                     # valor representativo de cada faixa
    w0 = np.cumsum(p)                       # peso da classe "abaixo do corte"
    mu = np.cumsum(p * levels)              # média acumulada (não normalizada)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0
    return float(edges[int(np.argmax(sigma_b2)) + 1])
