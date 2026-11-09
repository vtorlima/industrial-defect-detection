"""Fase 3 — alinhamento, modelo de referência e baseline de segmentação.

Uso (da raiz do repositório, com o ambiente ativado; precisa de results/fase2/particao.csv):

    python scripts/run_fase3.py

Passos:
  1. Centro de cada garrafa (Otsu + momentos) e translação para o centro médio da referência.
  2. Média e desvio por pixel com as imagens de papel "reference".
  3. Mapa z = |I - mu| / (sigma + eps) em todas as outras imagens.
  4. Limiar global = percentil de z (ROI) nas imagens de "calibration".
  5. Métricas: Dice, IoU, precisão, recall nas defeituosas; pixels FP e componentes nas boas.

Variantes comparadas (uma mudança por vez):
  baseline        alinhado + limiar calibrado          <- a régua das próximas fases
  sem_alinhamento mesmo, sem translação                 (justifica o alinhamento)
  otsu_por_imagem alinhado + Otsu no mapa z de cada imagem (limiar sem calibração)

Decisões usam o conjunto de desenvolvimento. Os números da avaliação são registrados
como linha de base para os experimentos, nunca usados para escolher parâmetros.

Saídas em results/fase3/: metricas_por_imagem.csv, resumo.csv, alinhamento.csv,
exemplos_desenvolvimento.png
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pdi_defeitos import metrics, viz
from pdi_defeitos.config import load_config
from pdi_defeitos.io import list_samples, load_sample
from pdi_defeitos.reference import ReferenceModel, align_to, otsu_threshold
from pdi_defeitos.split import load_partition
from pdi_defeitos.stats import object_disk, radial_distance


def main() -> int:
    cfg = load_config()
    size, rc = cfg["image"]["size"], cfg["reference"]
    out_dir = cfg["root"] / "results" / "fase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    part = load_partition(cfg["root"] / "results" / "fase2" / "particao.csv", list_samples(cfg["category_dir"]))

    # 1. Carregar em cinza a 256 px e medir o centro de cada garrafa
    gray, gt, disk = {}, {}, {}
    for _, row in tqdm(part.iterrows(), total=len(part), desc="carregando"):
        img, mask = load_sample(row, size=size)
        gray[row.sample_id] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gt[row.sample_id] = mask
        disk[row.sample_id] = object_disk(gray[row.sample_id])

    ref_ids = part.loc[part.role == "reference", "sample_id"].tolist()
    target = tuple(np.mean([disk[i][:2] for i in ref_ids], axis=0))
    radius = float(np.mean([disk[i][2] for i in ref_ids]))

    def prepared(sid: str, align: bool):
        if not align:
            return gray[sid].astype(np.float64), gt[sid]
        g, m, _ = align_to(gray[sid], target, gt[sid])
        return g.astype(np.float64), m

    # 2. Modelos de referência com e sem alinhamento
    models = {}
    for align in (True, False):
        stack = np.stack([prepared(i, align)[0] for i in ref_ids])
        models[align] = ReferenceModel.fit(stack, target, radius, rc["roi_frac"], rc["eps"])

    # Efeito do alinhamento na variabilidade natural (critério da Fase 3)
    rdist = radial_distance((size, size), target, radius)
    rows = []
    for align, model in models.items():
        for name, (lo, hi) in {"fundo (r<0,5)": (0, 0.5), "anéis (0,5<=r<0,95)": (0.5, 0.95)}.items():
            sel = (rdist >= lo) & (rdist < hi)
            rows.append({"alinhado": align, "regiao": name, "std_mediano": np.median(models[align].std[sel]),
                         "std_p95": np.percentile(models[align].std[sel], 95)})
    align_tab = pd.DataFrame(rows)
    align_tab.to_csv(out_dir / "alinhamento.csv", index=False, float_format="%.3f")

    # 3-5. Para cada variante: limiar, máscaras e métricas
    variants = {"baseline": (True, "calibrado"), "sem_alinhamento": (False, "calibrado"), "otsu_por_imagem": (True, "otsu")}
    cal_ids = part.loc[part.role == "calibration", "sample_id"].tolist()
    test_part = part[part.role.isin(["development", "evaluation"])]
    per_image, thresholds, examples = [], {}, []
    for vname, (align, mode) in variants.items():
        model = models[align]
        roi = model.roi > 0
        cal_z = np.concatenate([model.anomaly_map(prepared(i, align)[0])[roi] for i in cal_ids])
        thr = float(np.percentile(cal_z, rc["threshold_percentile"]))
        thresholds[vname] = thr
        for _, row in test_part.iterrows():
            g, m = prepared(row.sample_id, align)
            z = model.anomaly_map(g)
            t = otsu_threshold(z[roi]) if mode == "otsu" else thr
            pred = ((z >= t) & roi).astype(np.uint8)
            rec = {"variante": vname, "sample_id": row.sample_id, "role": row.role, "split": row.split,
                   "defect_type": row.defect_type, "limiar": t, "z_p99": np.percentile(z[roi], 99)}
            rec |= metrics.image_metrics(pred, m, model.roi)
            per_image.append(rec)
            if vname == "baseline" and row.role == "development":
                examples.append((row.sample_id, g, z, pred, m))
    per_image = pd.DataFrame(per_image)
    per_image.to_csv(out_dir / "metricas_por_imagem.csv", index=False, float_format="%.5f")

    # Grupo de resumo: tipo de defeito; boas separadas em treino e teste (achado da Fase 2)
    per_image["grupo"] = np.where(per_image.defect_type == "good", "good_" + per_image.split, per_image.defect_type)
    summaries = []
    for (vname, role), g in per_image.groupby(["variante", "role"]):
        s = metrics.summarize(g, by="grupo").reset_index()
        s.insert(0, "role", role)
        s.insert(0, "variante", vname)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)

    # AUROC de pixel por variante e conjunto (independe do limiar)
    auc_rows = []
    for vname, (align, _) in variants.items():
        if vname == "otsu_por_imagem":
            continue  # mesmo mapa z do baseline
        model = models[align]
        roi = model.roi > 0
        for role in ("development", "evaluation"):
            ids = test_part.loc[(test_part.role == role) & (test_part.label == 1), "sample_id"]
            zs, ys = [], []
            for sid in ids:
                g, m = prepared(sid, align)
                zs.append(model.anomaly_map(g)[roi])
                ys.append(m[roi] > 0)
            auc_rows.append({"variante": vname, "role": role, "auroc_pixel": roc_auc_score(np.concatenate(ys), np.concatenate(zs))})
    auc = pd.DataFrame(auc_rows)
    summary.to_csv(out_dir / "resumo.csv", index=False, float_format="%.4f")
    auc.to_csv(out_dir / "auroc_pixel.csv", index=False, float_format="%.4f")

    # Figura: os 9 exemplos de desenvolvimento no baseline
    zmax = thresholds["baseline"] * 2
    fig = viz.panel_grid(
        [[g, np.minimum(z, zmax), p, m] for _, g, z, p, m in examples],
        [sid.split("/", 1)[1] for sid, *_ in examples],
        ["imagem alinhada", f"mapa z (limiar {thresholds['baseline']:.2f})", "máscara prevista", "GT"],
        cmaps=["gray", "magma", "gray", "gray"], vmax=[255, zmax, 1, 1],
    )
    fig.savefig(out_dir / "exemplos_desenvolvimento.png", dpi=100)

    # Resumo no terminal
    pd.set_option("display.width", 160)
    f = lambda v: f"{v:.3f}"
    print(f"\nCentro alvo ({size}px): ({target[0]:.2f}, {target[1]:.2f}); raio da garrafa {radius:.1f} px")
    print("\n1) Alinhamento: desvio por pixel entre as boas de referência")
    print(align_tab.to_string(index=False, float_format=f))
    print("\n2) Limiares (percentil {} de z nas boas de calibração):".format(rc["threshold_percentile"]),
          ", ".join(f"{k} {v:.2f}" for k, v in thresholds.items() if k != "otsu_por_imagem"))
    print("\n3) AUROC de pixel nas defeituosas")
    print(auc.to_string(index=False, float_format=f))
    cols = ["variante", "grupo", "n", "dice_mean", "iou_mean", "precision_mean", "recall_mean", "dice_agg", "fp_frac_mean", "n_pred_components_mean", "empty_pred"]
    for role, title in (("development", "DESENVOLVIMENTO (decisões)"), ("evaluation", "AVALIAÇÃO (registro, não usado para escolher)")):
        print(f"\n4) {title}")
        print(summary.loc[summary.role == role, cols].to_string(index=False, float_format=f))
    print(f"\nSaídas em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
