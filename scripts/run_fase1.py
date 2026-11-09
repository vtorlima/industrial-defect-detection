"""Fase 1 — exploração dos dados. Gera as tabelas e figuras que justificam as decisões do plano.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/run_fase1.py

Saídas em results/fase1/:
  amostras.csv               uma linha por imagem: ruído, posição da garrafa e, nas
                             defeituosas, área, componentes, posição radial e contraste
  resumo_por_tipo.csv        mínimo, mediana e máximo das medidas por tipo de defeito
  variabilidade_radial.csv   desvio padrão entre garrafas boas por faixa de raio
  galeria.png                5 exemplos por tipo com o contorno da máscara GT
  variabilidade.png          imagem média e desvio por pixel das boas do treino

Perguntas que cada saída responde estão em src/pdi_defeitos/stats.py.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pdi_defeitos import stats, viz
from pdi_defeitos.config import load_config
from pdi_defeitos.io import list_samples, load_image, load_sample


def main() -> int:
    cfg = load_config()
    size = cfg["image"]["size"]
    ex = cfg["exploration"]
    out_dir = cfg["root"] / "results" / "fase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = list_samples(cfg["category_dir"])
    print(f"{len(df)} imagens em {cfg['category_dir']}")

    # 1. Medidas por imagem
    rows, good_stack = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="medindo"):
        full_gray = load_image(row.image_path, mode="gray")  # ruído medido na resolução original
        img, mask = load_sample(row, size=size)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cx, cy, r = stats.object_disk(gray)
        rec = {
            "sample_id": row.sample_id,
            "split": row.split,
            "defect_type": row.defect_type,
            "noise_sigma_full": stats.estimate_noise_sigma(full_gray),
            "center_x": cx,
            "center_y": cy,
            "radius": r,
        }
        if row.label == 1:
            rec |= stats.defect_geometry(mask, (cx, cy), r)
            rec |= stats.ring_contrast(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), mask, ex["ring_px"])
        rows.append(rec)
        if row.split == "train":
            good_stack.append(gray.astype(np.float64))
    samples = pd.DataFrame(rows)
    samples.to_csv(out_dir / "amostras.csv", index=False, float_format="%.4f")

    # 2. Resumo por tipo de defeito
    cols = ["area_px", "area_frac", "equiv_diameter_px", "n_components", "radial_pos", "dL", "dab"]
    defects = samples[samples.defect_type != "good"]
    summary = defects.groupby("defect_type")[cols].agg(["min", "median", "max"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "n", defects.groupby("defect_type").size())
    summary.to_csv(out_dir / "resumo_por_tipo.csv", float_format="%.4f")

    # 3. Variabilidade natural das garrafas boas (treino), por faixa de raio
    stack = np.stack(good_stack)
    mean_map, std_map = stack.mean(0), stack.std(0)
    train = samples[samples.split == "train"]
    center = (train.center_x.mean(), train.center_y.mean())
    rdist = stats.radial_distance(mean_map.shape, center, train.radius.mean())
    edges = np.arange(0.0, 1.0 + 1e-9, ex["radial_bin"])
    radial = pd.DataFrame(
        {
            "r_min": edges[:-1],
            "r_max": edges[1:],
            "std_mediano": stats.radial_profile(std_map, rdist, edges),
            "std_p95": stats.radial_profile(std_map, rdist, edges, stat=lambda v: np.percentile(v, 95)),
            "cinza_medio": stats.radial_profile(mean_map, rdist, edges, stat=np.mean),
        }
    )
    radial.to_csv(out_dir / "variabilidade_radial.csv", index=False, float_format="%.3f")

    # 4. Figuras
    viz.gallery(df, size, n_per_type=ex["gallery_per_type"], seed=ex["gallery_seed"]).savefig(out_dir / "galeria.png", dpi=110)
    viz.variability_maps(mean_map, std_map).savefig(out_dir / "variabilidade.png", dpi=110)

    # 5. Resumo no terminal: os números que entram na seção "Dados" do relatório
    noise = samples.noise_sigma_full
    print(f"\nRuído (Immerkær, resolução original): média {noise.mean():.2f}, faixa {noise.min():.2f} a {noise.max():.2f} níveis")
    print(
        f"Centro da garrafa ({size}px): desvio x {train.center_x.std():.2f} px, y {train.center_y.std():.2f} px; "
        f"amplitude x {np.ptp(train.center_x):.1f} px, y {np.ptp(train.center_y):.1f} px"
    )
    print("\nVariabilidade entre boas por faixa de raio (0 = centro, 1 = borda da garrafa):")
    print(radial.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nDefeitos por tipo (mediana):")
    med = defects.groupby("defect_type")[cols].median()
    med.insert(0, "menor_area_px", defects.groupby("defect_type").area_px.min())
    print(med.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaídas em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
