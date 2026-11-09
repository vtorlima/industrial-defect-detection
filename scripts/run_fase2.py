"""Fase 2 — partição supervisionada própria. Gera e confere a partição usada por todas as fases seguintes.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/run_fase2.py

Saídas em results/fase2/:
  particao.csv   sample_id, split, defect_type, label, role (reference, calibration,
                 development, evaluation)
  folds.csv      sample_id, seed, fold: fold de TESTE de cada imagem de avaliação,
                 para cada repetição da validação cruzada

Rodar de novo com a mesma semente gera arquivos idênticos. As fases seguintes leem
estes arquivos com split.load_partition; nunca sorteiam de novo.
"""

from __future__ import annotations

import sys

import pandas as pd

from pdi_defeitos.config import load_config
from pdi_defeitos.io import list_samples
from pdi_defeitos.split import PARTITION_COLUMNS, check_partition, make_cv_folds, make_partition


def main() -> int:
    cfg = load_config()
    p = cfg["partition"]
    out_dir = cfg["root"] / "results" / "fase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(cfg["category_dir"])
    part = make_partition(samples, p["n_reference"], p["n_calibration"], p["n_dev_per_type"], seed=cfg["seed"])
    problems = check_partition(part)
    if problems:
        print("PARTIÇÃO INVÁLIDA:\n  " + "\n  ".join(problems))
        return 1

    evaluation = part[part.role == "evaluation"]
    folds = make_cv_folds(evaluation, p["cv_splits"], p["cv_seeds"])

    part[PARTITION_COLUMNS].to_csv(out_dir / "particao.csv", index=False)
    folds.to_csv(out_dir / "folds.csv", index=False)

    # Resumo: papel x tipo
    table = pd.crosstab(part.role, part.defect_type, margins=True, margins_name="total")
    table = table.reindex(["reference", "calibration", "development", "evaluation", "total"])
    print(f"Partição (semente {cfg['seed']}):\n")
    print(table.to_string())

    # Conferência da estratificação: tipos por fold de teste, primeira semente
    first = folds[folds.seed == p["cv_seeds"][0]].merge(part[["sample_id", "defect_type"]], on="sample_id")
    print(f"\nFolds de teste da semente {p['cv_seeds'][0]} (imagens por tipo):\n")
    print(pd.crosstab(first.fold, first.defect_type).to_string())

    ev = evaluation.label.value_counts()
    print(f"\nAvaliação: {len(evaluation)} imagens, {ev.get(0, 0)} boas x {ev.get(1, 0)} defeituosas")
    print(f"{p['cv_splits']} folds x {len(p['cv_seeds'])} sementes = {p['cv_splits'] * len(p['cv_seeds'])} rodadas por experimento")
    print(f"\nSaídas em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
