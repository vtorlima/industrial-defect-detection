"""Figuras usadas na exploração e, depois, nas galerias de erros do relatório."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

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


# --- Gráficos de resultados (figuras da apresentação) -------------------------------
# Só desenham: recebem tabelas já calculadas (os CSVs de results/) e não recalculam nada.

ACCENT = "#2a78d6"      # o item em destaque (a escolha)
CONTEXT = "#b4b2aa"     # os demais: contexto
INK = "#0b0b0b"         # texto principal
INK_2 = "#52514e"       # texto secundário
MUTED = "#898781"       # eixos
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def fmt_br(value: float, decimals: int = 2) -> str:
    """Número com vírgula decimal: fmt_br(0.8882) -> '0,89'."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _style_axis(ax: plt.Axes, title: str, note: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=20 if note else 8, fontweight="bold")
    if note:
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=MUTED, labelsize=8.5)


def _colors(n: int, highlight: int | None) -> list[str]:
    """Um destaque e o resto em cinza; sem destaque, tudo na cor de destaque."""
    if highlight is None:
        return [ACCENT] * n
    return [ACCENT if i == highlight else CONTEXT for i in range(n)]


def dot_panel(ax: plt.Axes, labels: list[str], values, title: str, highlight: int | None = None,
              xlim: tuple[float, float] | None = None, note: str | None = None, decimals: int = 2) -> None:
    """Pontos na horizontal, um por rótulo (de cima para baixo), com o valor ao lado.

    Para medidas em que a diferença importa mais que a distância ao zero (ex.: AUROC):
    o eixo pode começar fora do zero porque o ponto codifica posição, não comprimento.
    """
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    colors = _colors(len(values), highlight)
    _style_axis(ax, title, note)
    lo, hi = xlim or (values.min(), values.max())
    ax.hlines(y, lo, values, color=GRID, linewidth=1, zorder=1)
    ax.scatter(values, y, s=70, c=colors, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for i, v in enumerate(values):
        ax.annotate(fmt_br(v, decimals), (v, i), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold" if i == highlight else "normal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(values) - 0.5, -0.5)
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_br(v, 1))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _emphasize_tick(ax.get_yticklabels(), highlight)


def hbar_panel(ax: plt.Axes, labels: list[str], values, title: str, highlight: int | None = None,
               note: str | None = None, decimals: int = 2) -> None:
    """Barras horizontais a partir do zero, uma por rótulo (de cima para baixo), com o valor na ponta."""
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    _style_axis(ax, title, note)
    ax.barh(y, values, height=0.5, color=_colors(len(values), highlight), zorder=2)
    for i, v in enumerate(values):
        ax.annotate(fmt_br(v, decimals), (v, i), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold" if i == highlight else "normal")
    ax.set_xlim(0, values.max() * 1.22)
    ax.set_ylim(len(values) - 0.5, -0.5)
    ax.set_yticks(y, labels)
    tick_decimals = 0 if values.max() >= 5 else 1 if values.max() >= 0.5 else 2
    ax.xaxis.set_major_formatter(lambda v, _: fmt_br(v, tick_decimals))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=3)
    _emphasize_tick(ax.get_yticklabels(), highlight)


def column_panel(ax: plt.Axes, labels: list[str], values, title: str, note: str | None = None,
                 highlight: int | None = None, ymax: float | None = None, decimals: int = 2) -> None:
    """Colunas a partir do zero, uma por etapa, com o valor em cima."""
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))
    _style_axis(ax, title, note)
    ax.bar(x, values, width=0.45, color=_colors(len(values), highlight), zorder=2)
    for i, v in enumerate(values):
        ax.annotate(fmt_br(v, decimals), (i, v), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10, color=INK, fontweight="bold" if i == highlight else "normal")
    ax.set_ylim(0, ymax if ymax is not None else values.max() * 1.15)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=9, colors=INK_2)
    if ax.get_ylim()[1] > 2:  # contagens: marcas inteiras
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_formatter(lambda v, _: fmt_br(v, 1) if ax.get_ylim()[1] <= 2 else fmt_br(v, 0))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=3)
    _emphasize_tick(ax.get_xticklabels(), highlight)


def _emphasize_tick(ticklabels, highlight: int | None) -> None:
    for i, t in enumerate(ticklabels):
        t.set_color(INK if i == highlight else INK_2)
        t.set_fontweight("bold" if i == highlight else "normal")


def filter_choice(labels: list[str], auroc: dict[str, list[float]], components: list[float],
                  highlight: int, title: str, subtitle: str = "") -> plt.Figure:
    """Escolha do filtro: um painel de AUROC por coluna de `auroc` (título -> valores por
    filtro, todos no mesmo eixo) e um painel de componentes espúrios por garrafa boa.
    `highlight`: posição do filtro escolhido."""
    n = len(auroc) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.75 * n + 1.6, 0.42 * len(labels) + 2.1), sharey=True, facecolor=SURFACE)
    allv = np.concatenate([np.asarray(v, dtype=float) for v in auroc.values()])
    lo = np.floor(allv.min() * 10) / 10
    for ax, (name, vals) in zip(axes, auroc.items()):
        dot_panel(ax, labels, vals, name, highlight, xlim=(lo, 1.0), note="AUROC de pixel · maior é melhor")
    hbar_panel(axes[-1], labels, components, "Componentes espúrios", highlight,
               note="por garrafa boa (calibração) · menor é melhor")
    _titles(fig, title, subtitle)
    return fig


def progress(stages: list[str], panels: list[tuple[str, str, list[float], float | None]],
             title: str, subtitle: str = "") -> plt.Figure:
    """Progresso por etapa: um painel de colunas por métrica.

    panels: lista de (título, nota, valores por etapa, teto do eixo ou None)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.3), facecolor=SURFACE)
    for ax, (name, note, vals, ymax) in zip(np.atleast_1d(axes), panels):
        column_panel(ax, stages, vals, name, note, ymax=ymax)
    _titles(fig, title, subtitle)
    return fig


def _titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.86 if subtitle else 0.91), w_pad=2.5)
    fig.text(0.012, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=INK_2, va="top")


# --- Peças dos notebooks de apresentação (leem results/, não recalculam) -------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # cores categóricas, sempre nesta ordem


def fmt_table(df: pd.DataFrame, decimals: int = 2, mark=None, mark_text: str = "← escolhido") -> pd.DataFrame:
    """Tabela pronta para exibir: reais com `decimals` casas e vírgula decimal, NaN como "–",
    booleanos como sim/não; inteiros e textos ficam como estão.

    mark: rótulo do índice a destacar (recebe `mark_text` ao lado). Devolve um DataFrame de
    textos, que o GitHub mostra igual ao Jupyter (o Styler do pandas perde o estilo lá).
    """
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "sim" if v else "não"
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else fmt_br(v, decimals)
        return v

    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(cell).astype(object)
    if mark is not None:
        if mark not in out.index:
            raise KeyError(f"{mark!r} não está no índice da tabela")
        out.index = [f"{i}  {mark_text}" if i == mark else i for i in out.index]
        out.index.name = df.index.name
    return out


def hpanels(labels: list[str], panels: list[dict], highlight: int | None = None,
            title: str = "", subtitle: str = "") -> plt.Figure:
    """Linha de painéis horizontais com os mesmos rótulos (um por linha de cada painel).

    panels: dicionários com `title`, `values` e, opcionalmente, `note`, `kind` ("bar",
    o padrão: barras desde o zero; "dot": pontos, para quando a diferença importa mais
    que a distância ao zero), `xlim` (só "dot") e `decimals`.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 1.8, 0.42 * len(labels) + 2.1), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, p in zip(axes[0], panels):
        values = np.asarray(p["values"], dtype=float)
        if p.get("kind", "bar") == "dot":
            pad = 0.6 * (values.max() - values.min()) or 1.0
            xlim = p.get("xlim") or (values.min() - pad, values.max() + pad)
            dot_panel(ax, labels, values, p["title"], highlight, xlim=xlim, note=p.get("note"), decimals=p.get("decimals", 2))
        else:
            hbar_panel(ax, labels, values, p["title"], highlight, note=p.get("note"), decimals=p.get("decimals", 2))
    _titles(fig, title, subtitle)
    return fig


def stacked_hbar(table: pd.DataFrame, title: str = "", subtitle: str = "", xlabel: str = "imagens") -> plt.Figure:
    """Barras horizontais empilhadas de contagens: uma barra por linha de `table`, um
    segmento por coluna (até 4 colunas), com o total na ponta e legenda em cima."""
    if table.shape[1] > len(SERIES):
        raise ValueError(f"no máximo {len(SERIES)} colunas")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(table) + 2.0), facecolor=SURFACE)
    _style_axis(ax, "")
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for color, col in zip(SERIES, table.columns):
        vals = table[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=1.5, label=str(col), zorder=2)
        left += vals
    for i, total in enumerate(left):
        ax.annotate(fmt_br(total, 0), (total, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, left.max() * 1.1)
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_yticks(y, [str(i) for i in table.index])
    ax.tick_params(axis="y", labelsize=9.5, colors=INK_2)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=table.shape[1], frameon=False, fontsize=9,
              labelcolor=INK_2, handlelength=1.0, columnspacing=1.4, borderaxespad=0.2)
    _titles(fig, title, subtitle)
    return fig


def dataset_available(category_dir) -> bool:
    """True se a categoria do MVTec AD está no disco. As seções opcionais dos notebooks
    (exemplos visuais) usam isto para pular, com uma mensagem, quando o dataset não foi baixado."""
    from pathlib import Path

    ok = (Path(category_dir) / "train" / "good").is_dir()
    if not ok:
        print(f"Dataset não encontrado em {category_dir}.\n"
              "Esta seção é opcional e foi pulada; o resto do notebook não depende dela.\n"
              "Para vê-la, baixe o dataset (README, passo 5) e rode de novo.")
    return ok
