"""Fase 4 — denoising e bordas, medidos contra o baseline da Fase 3.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/run_fase4.py

Protocolo (uma mudança por vez, o resto fixo; decisões só com calibração e desenvolvimento):
  A. Denoising: para cada filtro candidato, refazer a referência com imagens filtradas.
     Critério: AUROC de pixel no desenvolvimento. Empate (< decision_tie) -> o mais simples.
  B. Normalização de brilho por imagem (achado da Fase 2: boas do teste ~2 níveis mais claras).
     Adotada se não piorar a AUROC; falsos positivos nas boas da avaliação só são registrados.
  C. Bordas por Sobel: mapa z do gradiente, sozinho e combinado com o de intensidade.
     Adotado se melhorar broken sem piorar contamination.
  D. Bordas novas do Canny (razões 2:1 e 3:1): quantas caem dentro do defeito e quantas
     aparecem em garrafas boas. Informa se vale usá-las na segmentação (Fase 5).

Saídas em results/fase4/: denoising.csv, normalizacao.csv, bordas.csv, canny.csv,
mapas_desenvolvimento.png
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd

from pdi_defeitos import metrics, viz
from pdi_defeitos.config import load_config
from pdi_defeitos.pipeline import evaluate_map, load_dataset
from pdi_defeitos.preprocess import canny_new_edges, denoise, sobel_magnitude
from pdi_defeitos.reference import ReferenceModel

SIMPLICITY = ["none", "gaussian", "median", "bilateral"]  # ordem de desempate


def label(c: dict) -> str:
    extra = ",".join(f"{k}={v}" for k, v in c.items() if k != "method")
    return c["method"] + (f"({extra})" if extra else "")


def main() -> int:
    cfg = load_config()
    rc, ec, tie = cfg["reference"], cfg["edges"], cfg["decision_tie"]
    q = rc["threshold_percentile"]
    out_dir = cfg["root"] / "results" / "fase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg)
    roi = ds.roi > 0
    ref_ids = ds.ids("reference")
    used = ds.ids("calibration") + ds.ids("development") + ds.ids("evaluation", label=0)

    def prepared(c: dict, normalize: bool = False) -> dict[str, np.ndarray]:
        imgs = {s: ds.aligned(denoise(ds.gray[s], **c), s) for s in ref_ids + used}
        if normalize:  # tira o brilho médio da garrafa de cada imagem e põe o da referência
            ref_level = np.mean([imgs[s][roi].mean() for s in ref_ids])
            imgs = {s: im - im[roi].mean() + ref_level for s, im in imgs.items()}
        return imgs

    def zmaps(imgs: dict[str, np.ndarray], eps: float) -> dict[str, np.ndarray]:
        model = ReferenceModel.fit(np.stack([imgs[s] for s in ref_ids]), ds.target, ds.radius, rc["roi_frac"], eps)
        return {s: model.anomaly_map(imgs[s]) for s in used}

    # A. Denoising
    rows = []
    for c in cfg["denoise_candidates"]:
        r = evaluate_map(ds, zmaps(prepared(c), rc["eps"]), q)
        rows.append({"filtro": label(c), "method": c["method"], **r})
        print(f"  A {label(c):<45} AUROC {r['auroc']:.3f}  Dice {r['dice']:.3f}")
    den = pd.DataFrame(rows)
    den.to_csv(out_dir / "denoising.csv", index=False, float_format="%.4f", lineterminator="\n")
    best = den.auroc.max()
    tied = den[den.auroc >= best - tie].copy()
    tied["ordem"] = tied.method.map(SIMPLICITY.index)
    chosen_idx = tied.sort_values(["ordem", "filtro"]).index[0]
    chosen = cfg["denoise_candidates"][chosen_idx]

    # B. Normalização de brilho, com o filtro escolhido
    rows = []
    for normalize in (False, True):
        imgs = prepared(chosen, normalize)
        z = zmaps(imgs, rc["eps"])
        r = evaluate_map(ds, z, q)
        for split in ("train", "test"):  # registro: falsos positivos nas boas da avaliação
            ids = ds.part.loc[(ds.part.role == "evaluation") & (ds.part.label == 0) & (ds.part.split == split), "sample_id"]
            r[f"fp_frac_good_{split}"] = float(np.mean([((z[s] >= r["thr"]) & roi).sum() / roi.sum() for s in ids]))
        rows.append({"normalizar_brilho": normalize, **r})
    norm = pd.DataFrame(rows)
    norm.to_csv(out_dir / "normalizacao.csv", index=False, float_format="%.5f", lineterminator="\n")
    use_norm = bool(norm.loc[1, "auroc"] >= norm.loc[0, "auroc"] - tie)

    # C. Bordas por Sobel: z do gradiente e combinações
    imgs = prepared(chosen, use_norm)
    z_int = zmaps(imgs, rc["eps"])
    grads = {s: sobel_magnitude(im) for s, im in imgs.items()}
    z_grad = zmaps(grads, ec["grad_eps"])
    variants = {
        "intensidade": z_int,
        "gradiente": z_grad,
        "soma": {s: z_int[s] + z_grad[s] for s in used},
        "maximo": {s: np.maximum(z_int[s], z_grad[s]) for s in used},
    }
    edges = pd.DataFrame([{"mapa": k, **evaluate_map(ds, v, q)} for k, v in variants.items()])
    edges.to_csv(out_dir / "bordas.csv", index=False, float_format="%.4f", lineterminator="\n")
    base = edges.set_index("mapa").loc["intensidade"]
    ok = edges[(edges.auroc_contamination >= base.auroc_contamination - tie)
               & ((edges.auroc_broken_large + edges.auroc_broken_small) / 2 > (base.auroc_broken_large + base.auroc_broken_small) / 2 + tie)]
    chosen_map = ok.sort_values("auroc", ascending=False).mapa.iloc[0] if len(ok) else "intensidade"

    # D. Bordas novas do Canny (na imagem filtrada e alinhada, convertida para uint8)
    u8 = {s: np.clip(im, 0, 255).astype(np.uint8) for s, im in imgs.items()}
    ref_mean = np.clip(np.mean([imgs[s] for s in ref_ids], axis=0), 0, 255).astype(np.uint8)
    rows = []
    for ratio in ec["canny_ratios"]:
        ref_edges = cv2.Canny(ref_mean, ec["canny_high"] / ratio, ec["canny_high"], L2gradient=True)
        new = {s: canny_new_edges(u8[s], ref_edges, ec["canny_high"], ratio, ec["canny_tolerance_px"]) & ds.roi for s in used}
        dev = ds.part[ds.part.role == "development"]
        inside, total, hit = 0, 0, []
        per_type = {}
        for _, row in dev.iterrows():
            g = ds.aligned_gt(row.sample_id) > 0
            n = new[row.sample_id] > 0
            inside += int((n & g).sum())
            total += int(n.sum())
            hit.append((n & g).any())
            per_type.setdefault(row.defect_type, []).append((n & g).sum() / max(g.sum(), 1))
        cal = ds.ids("calibration")
        rows.append({
            "razao": f"{ratio}:1",
            "frac_bordas_novas_dentro_do_defeito": inside / max(total, 1),
            "defeitos_com_borda_nova": float(np.mean(hit)),
            **{f"cobertura_{t}": float(np.mean(v)) for t, v in sorted(per_type.items())},
            "px_borda_nova_por_boa_calib": float(np.mean([new[s].sum() for s in cal])),
            "componentes_por_boa_calib": float(np.mean([metrics.n_components(new[s]) for s in cal])),
        })
    canny = pd.DataFrame(rows)
    canny.to_csv(out_dir / "canny.csv", index=False, float_format="%.4f", lineterminator="\n")

    # Figura: desenvolvimento, mapas de intensidade, gradiente e o escolhido
    dev_ids = ds.ids("development")
    thr_i = edges.set_index("mapa").loc["intensidade", "thr"]
    thr_g = edges.set_index("mapa").loc["gradiente", "thr"]
    fig = viz.panel_grid(
        [[imgs[s], np.minimum(z_int[s], 2 * thr_i), np.minimum(z_grad[s], 2 * thr_g), ds.aligned_gt(s)] for s in dev_ids],
        [s.split("/", 1)[1] for s in dev_ids],
        [f"filtrada: {label(chosen)}", "z intensidade", "z gradiente (Sobel)", "GT"],
        cmaps=["gray", "magma", "magma", "gray"], vmax=[255, 2 * thr_i, 2 * thr_g, 1],
    )
    fig.savefig(out_dir / "mapas_desenvolvimento.png", dpi=100)

    # Relatório no terminal
    pd.set_option("display.width", 200)
    f = lambda v: f"{v:.3f}"
    cols = ["auroc", "auroc_broken_large", "auroc_broken_small", "auroc_contamination", "sep", "dice", "precision", "recall", "cal_components"]
    print("\nA) DENOISING (desenvolvimento; critério: AUROC)")
    print(den[["filtro"] + cols].to_string(index=False, float_format=f))
    print(f"   -> escolhido: {label(chosen)}  (melhor AUROC {best:.3f}; empate < {tie} resolvido pelo mais simples)")
    print("\nB) NORMALIZAÇÃO DE BRILHO (FP nas boas da avaliação: só registro)")
    print(norm[["normalizar_brilho", "auroc", "dice", "fp_frac_good_train", "fp_frac_good_test"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"   -> normalizar brilho: {'sim' if use_norm else 'não'}")
    print("\nC) BORDAS POR SOBEL")
    print(edges[["mapa"] + cols].to_string(index=False, float_format=f))
    print(f"   -> mapa escolhido: {chosen_map}")
    print("\nD) BORDAS NOVAS DO CANNY")
    print(canny.to_string(index=False, float_format=f))
    print(f"\nSaídas em {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
