"""Figuras da apresentação da Fase 4, feitas só com os CSVs já commitados.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/figuras_fase4.py

Não recalcula nada e não precisa do dataset: lê results/fase3/{auroc_pixel,resumo}.csv
e results/fase4/{denoising,bordas}.csv e grava em results/fase4/:

  escolha_do_filtro.png  AUROC por filtro (geral e por tipo) e componentes espúrios
                         por garrafa boa, com o filtro escolhido em destaque.
  progresso.png          AUROC, Dice e componentes espúrios em três etapas:
                         baseline -> + Gaussiano sigma=2 -> + Sobel (máximo).

Antes de desenhar, confere se os CSVs das duas fases contam a mesma história
(baseline da Fase 3 = linha `none` da Fase 4; filtro escolhido = mapa `intensidade`).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # só grava arquivos; funciona sem tela

import pandas as pd

from pdi_defeitos import viz
from pdi_defeitos.config import find_repo_root

# Decisões da Fase 4 (config.yaml, seção `pipeline`), repetidas aqui porque este
# script lê apenas os CSVs.
CHOSEN_FILTER = "gaussian(sigma=2)"
CHOSEN_MAP = "maximo"
TOL = 5e-4  # os CSVs têm 4 casas decimais

FILTER_LABELS = {
    "none": "sem filtro",
    "gaussian(sigma=1)": "Gaussiano σ=1",
    "gaussian(sigma=2)": "Gaussiano σ=2",
    "gaussian(sigma=3)": "Gaussiano σ=3",
    "median(ksize=5)": "mediana 5×5",
    "bilateral(d=9,sigma_color=30,sigma_space=3)": "bilateral (d=9)",
}
AUROC_PANELS = {
    "auroc": "Geral",
    "auroc_broken_large": "broken_large",
    "auroc_broken_small": "broken_small",
    "auroc_contamination": "contamination",
}


def check_consistency(f3_auroc: pd.DataFrame, f3_resumo: pd.DataFrame, den: pd.DataFrame, bordas: pd.DataFrame) -> list[str]:
    """Diferenças entre CSVs que deveriam trazer o mesmo número (lista vazia = tudo certo)."""
    problems = []

    def same(name: str, a: float, b: float) -> None:
        if abs(a - b) > TOL:
            problems.append(f"{name}: {a:.4f} != {b:.4f}")

    none = den.loc["none"]
    base3 = f3_auroc[(f3_auroc.variante == "baseline") & (f3_auroc.role == "development")].auroc_pixel.iloc[0]
    same("AUROC do baseline (fase3/auroc_pixel.csv x fase4/denoising.csv)", base3, none.auroc)
    dev3 = f3_resumo[(f3_resumo.variante == "baseline") & (f3_resumo.role == "development")]
    dice3 = (dev3.dice_mean * dev3.n).sum() / dev3.n.sum()
    same("Dice do baseline (fase3/resumo.csv x fase4/denoising.csv)", dice3, none.dice)
    for col in ("auroc", "dice", "cal_components"):
        same(f"{col} do filtro escolhido (denoising.csv x bordas.csv)", den.loc[CHOSEN_FILTER, col], bordas.loc["intensidade", col])
    return problems


def main() -> int:
    root = find_repo_root()
    f3, f4 = root / "results" / "fase3", root / "results" / "fase4"
    den = pd.read_csv(f4 / "denoising.csv").set_index("filtro")
    bordas = pd.read_csv(f4 / "bordas.csv").set_index("mapa")
    problems = check_consistency(pd.read_csv(f3 / "auroc_pixel.csv"), pd.read_csv(f3 / "resumo.csv"), den, bordas)
    for p in problems:
        print(f"AVISO: CSVs inconsistentes. {p}", file=sys.stderr)

    # 1. Escolha do filtro
    labels = [FILTER_LABELS.get(f, f) for f in den.index]
    fig = viz.filter_choice(
        labels,
        {title: den[col].tolist() for col, title in AUROC_PANELS.items()},
        den.cal_components.tolist(),
        highlight=den.index.get_loc(CHOSEN_FILTER),
        title="Escolha do filtro de denoising: Gaussiano σ=2",
        subtitle="Conjunto de desenvolvimento (9 defeituosas). σ=3 empata na AUROC geral (diferença < 0,01); "
                 "no empate fica o σ menor, com menos risco para defeitos pequenos.",
    )
    fig.savefig(f4 / "escolha_do_filtro.png", dpi=200, metadata={"Software": None})

    # 2. Progresso: baseline -> + Gaussiano -> + Sobel
    stages = ["Baseline\n(sem filtro)", "+ Gaussiano σ=2\n(intensidade)", "+ Sobel\n(máximo)"]
    rows = [den.loc["none"], bordas.loc["intensidade"], bordas.loc[CHOSEN_MAP]]
    fig = viz.progress(
        stages,
        [
            ("AUROC de pixel", "maior é melhor", [r.auroc for r in rows], 1.0),
            ("Dice", "média por imagem · maior é melhor", [r.dice for r in rows], 1.0),
            ("Componentes espúrios", "por garrafa boa (calibração) · menor é melhor", [r.cal_components for r in rows], None),
        ],
        title="Progresso da Fase 4 no conjunto de desenvolvimento",
        subtitle="O Sobel melhora a AUROC, mas quase triplica os componentes espúrios: a morfologia (Fase 6) decide entre os dois mapas.",
    )
    fig.savefig(f4 / "progresso.png", dpi=200, metadata={"Software": None})

    print("Valores usados em progresso.png (baseline / + Gaussiano σ=2 / + Sobel):")
    for name, col in (("AUROC", "auroc"), ("Dice", "dice"), ("Componentes espúrios", "cal_components")):
        print(f"  {name:<21}" + " / ".join(viz.fmt_br(r[col]) for r in rows))
    print(f"Figuras em {f4}: escolha_do_filtro.png, progresso.png")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
