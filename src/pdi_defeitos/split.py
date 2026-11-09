"""Partição supervisionada própria (Fase 2).

No MVTec AD as defeituosas só existem no teste (protocolo não supervisionado).
A spec pede uma partição própria. Cada imagem recebe exatamente um papel:

  reference    boas do treino usadas só para estimar média e desvio por pixel
  calibration  boas do treino usadas só para fixar limiares (taxa de falso positivo)
  development  poucas defeituosas por tipo, para olhar resultados e ajustar parâmetros;
               nunca entram nas métricas finais
  evaluation   todo o resto: métricas de segmentação e validação cruzada dos classificadores

Regras que evitam vazamento:
  * referência e calibração nunca entram na avaliação (uma imagem que ajudou a
    construir a média "se parece com a média" e daria resultado otimista);
  * a validação cruzada é estratificada pelo tipo (good e cada defeito), repetida
    com várias sementes, e os hiperparâmetros são escolhidos só nos folds de treino.

Aviso obrigatório no relatório: os números são de um experimento didático com
partição própria e não são comparáveis ao benchmark oficial do MVTec AD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROLES = ("reference", "calibration", "development", "evaluation")


def make_partition(
    samples: pd.DataFrame,
    n_reference: int,
    n_calibration: int,
    n_dev_per_type: int,
    seed: int,
) -> pd.DataFrame:
    """Devolve uma cópia de `samples` (saída de io.list_samples) com a coluna `role`.

    Referência e calibração são sorteadas entre as boas do treino; desenvolvimento
    sorteia `n_dev_per_type` imagens de cada tipo de defeito; o resto vai para avaliação.
    """
    rng = np.random.default_rng(seed)
    out = samples.copy()
    out["role"] = "evaluation"

    train_good = out.index[(out.split == "train") & (out.label == 0)].to_numpy()
    if n_reference + n_calibration >= len(train_good):
        raise ValueError(
            f"Referência ({n_reference}) + calibração ({n_calibration}) precisam deixar boas do treino "
            f"para a avaliação; há só {len(train_good)}."
        )
    shuffled = rng.permutation(train_good)
    out.loc[shuffled[:n_reference], "role"] = "reference"
    out.loc[shuffled[n_reference : n_reference + n_calibration], "role"] = "calibration"

    for dtype in sorted(out.loc[out.label == 1, "defect_type"].unique()):
        idx = out.index[out.defect_type == dtype].to_numpy()
        if n_dev_per_type >= len(idx):
            raise ValueError(f"Tipo {dtype} tem só {len(idx)} imagens; não sobra nada para avaliação.")
        out.loc[rng.choice(idx, size=n_dev_per_type, replace=False), "role"] = "development"

    return out


def make_cv_folds(evaluation: pd.DataFrame, n_splits: int, seeds: list[int]) -> pd.DataFrame:
    """Folds de validação cruzada estratificada pelo tipo, repetida para cada semente.

    Devolve formato longo: uma linha por (sample_id, seed) com o fold em que a
    amostra fica no TESTE. Para um fold k: treino = fold != k, teste = fold == k.
    """
    rows = []
    y = evaluation["defect_type"].to_numpy()
    ids = evaluation["sample_id"].to_numpy()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, test_idx) in enumerate(skf.split(ids, y)):
            rows += [{"sample_id": ids[i], "seed": seed, "fold": k} for i in test_idx]
    return pd.DataFrame(rows).sort_values(["seed", "sample_id"], ignore_index=True)


def check_partition(part: pd.DataFrame) -> list[str]:
    """Lista de problemas (vazia = partição válida). Usada pelo script e pelos testes."""
    problems = []
    if part.sample_id.duplicated().any():
        problems.append("sample_id repetido: alguma imagem aparece duas vezes.")
    if not set(part.role) <= set(ROLES):
        problems.append(f"papel desconhecido: {set(part.role) - set(ROLES)}")
    for role in ("reference", "calibration"):
        bad = part[(part.role == role) & ~((part.split == "train") & (part.label == 0))]
        if len(bad):
            problems.append(f"{role} contém imagens que não são boas do treino: {list(bad.sample_id)[:3]}")
    if (part[part.role == "development"].label == 0).any():
        problems.append("desenvolvimento contém imagens boas.")
    ev = part[part.role == "evaluation"]
    if ev.label.nunique() < 2:
        problems.append("avaliação não tem as duas classes.")
    return problems


PARTITION_COLUMNS = ["sample_id", "split", "defect_type", "label", "role"]


def load_partition(path: Path | str, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lê results/fase2/particao.csv. As fases seguintes usam isto, nunca sorteiam de novo.

    O CSV guarda só sample_id e papéis (caminhos mudam de computador para computador).
    Passe `samples` (saída de io.list_samples) para recuperar image_path e mask_path.
    """
    part = pd.read_csv(path)
    if samples is not None:
        part = part.merge(samples[["sample_id", "image_path", "mask_path"]], on="sample_id", how="left", validate="1:1")
        if part.image_path.isna().any():
            raise ValueError("Há sample_id na partição que não existe no dataset local.")
    return part
