"""Pré-processamento (Fase 4): denoising e bordas.

Denoising. A Fase 1 mediu ruído de sensor desprezível (sigma ~ 0,6 nível), então
o objetivo aqui NÃO é remover ruído: é reduzir a diferença espúria entre a imagem
e a referência (desalinhamento residual de 1 a 3 px, reflexos, textura fina do
vidro) sem apagar o defeito. Candidatos e o que a teoria prevê (Guia PDI, 12 a 15):

  gaussian  passa-baixa linear; borra bordas de forma previsível, controlada por sigma.
            Favorito: desalinhamento de 1 a 2 px vira diferença pequena depois do borramento.
  median    não linear, ótimo contra ruído impulsivo (que não existe aqui); preserva
            bordas, inclusive as dos anéis desalinhados, que geram falsos positivos.
  bilateral suaviza só entre pixels de intensidade parecida; preserva bordas fortes,
            também as dos reflexos.

Limite superior: sigma do Gaussiano <= 3 px, bem abaixo do raio do menor defeito (~11 px).

Bordas. Quebras criam bordas que não existem na garrafa normal. Duas formas de usar:
  * diferença de magnitude do gradiente (Sobel) em relação à referência, normalizada
    como o mapa z (ver reference.ReferenceModel aplicado a pilhas de gradiente);
  * bordas novas do Canny: bordas da imagem fora das bordas da referência dilatadas
    (a dilatação dá tolerância ao desalinhamento residual).
"""

from __future__ import annotations

import cv2
import numpy as np

DENOISERS = ("none", "gaussian", "median", "bilateral")


def denoise(gray: np.ndarray, method: str, **p) -> np.ndarray:
    """Aplica o filtro `method` a uma imagem uint8 em cinza.

    gaussian: sigma (px)        median: ksize (ímpar)
    bilateral: d, sigma_color, sigma_space
    """
    if method == "none":
        return gray
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (0, 0), p["sigma"])
    if method == "median":
        return cv2.medianBlur(gray, p["ksize"])
    if method == "bilateral":
        return cv2.bilateralFilter(gray, p["d"], p["sigma_color"], p["sigma_space"])
    raise ValueError(f"filtro desconhecido: {method!r} (use um de {DENOISERS})")


# Núcleos de Sobel (Guia Segmentação, seção 8): derivada em uma direção,
# suavização [1 2 1] na outra.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlação 2D com borda refletida, implementada com NumPy (exercício da disciplina).

    Soma deslocamentos da imagem ponderados pelo núcleo: sem loops por pixel, só
    um loop pelos k*k coeficientes. Mesma convenção do cv2.filter2D (correlação).
    """
    k = kernel.shape[0] // 2
    padded = np.pad(img.astype(np.float64), k, mode="reflect")
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            out += kernel[dy, dx] * padded[dy : dy + h, dx : dx + w]
    return out


def sobel_magnitude_numpy(gray: np.ndarray) -> np.ndarray:
    """|grad I| = sqrt(Gx^2 + Gy^2) com o Sobel feito à mão."""
    return np.hypot(convolve2d(gray, SOBEL_X), convolve2d(gray, SOBEL_Y))


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """|grad I| com cv2.Sobel (versão usada no pipeline; conferida contra a NumPy nos testes)."""
    g = gray.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(gx, gy)


def canny_new_edges(gray: np.ndarray, ref_edges: np.ndarray, high: float, ratio: float, tolerance_px: int) -> np.ndarray:
    """Bordas do Canny presentes na imagem e ausentes na referência.

    ref_edges: bordas do Canny na imagem média da referência (uint8 {0, 255}).
    ratio: T_H / T_L (o guia recomenda entre 2:1 e 3:1).
    tolerance_px: raio da dilatação das bordas da referência (desalinhamento aceito).
    Devolve uint8 {0, 1}.
    """
    edges = cv2.Canny(gray, high / ratio, high, L2gradient=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    ref_dilated = cv2.dilate(ref_edges, k)
    return ((edges > 0) & (ref_dilated == 0)).astype(np.uint8)
