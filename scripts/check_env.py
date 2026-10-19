"""Verifica se o ambiente está pronto para o projeto.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/check_env.py

Confere: versão do Python, as bibliotecas listadas em requirements.txt,
o pacote pdi_defeitos instalado em modo editável, o dataset baixado e a
leitura de uma imagem com máscara. Termina com "TUDO OK" ou com a lista do que falta.

A lista de bibliotecas vem do próprio requirements.txt: quando uma fase nova
acrescenta uma dependência lá, esta verificação passa a cobri-la sozinha.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pacotes cujo nome de import difere do nome no pip.
# Ao adicionar ao requirements.txt uma biblioteca nesse caso (ex.: scikit-learn -> sklearn),
# acrescente-a aqui também.
IMPORT_NAMES = {
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}

problems: list[str] = []


def check(ok: bool, label: str, fix: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FALHA'}] {label}")
    if not ok:
        problems.append(f"{label} -> {fix}" if fix else label)
    return ok


def requirements() -> list[str]:
    """Nomes dos pacotes listados em requirements.txt (sem versões nem comentários)."""
    names = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].lower())
    return names


def main() -> int:
    print("1. Python")
    v = sys.version_info
    check(v >= (3, 10), f"Python {v.major}.{v.minor}.{v.micro}", "instale Python 3.11 ou 3.12")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    check(in_venv, f"ambiente virtual ativo ({sys.prefix})", "ative o ambiente (Windows: .venv\\Scripts\\Activate.ps1; Mac/Linux: source .venv/bin/activate)")

    print("\n2. Bibliotecas do requirements.txt")
    for pip_name in requirements():
        module = IMPORT_NAMES.get(pip_name, pip_name.replace("-", "_"))
        try:
            importlib.import_module(module)
            check(True, f"{pip_name} {importlib.metadata.version(pip_name)}")
        except ImportError:
            check(False, pip_name, "pip install -r requirements.txt")

    print("\n3. Pacote do projeto")
    try:
        import pdi_defeitos
        from pdi_defeitos.config import load_config
        from pdi_defeitos.io import list_samples, load_sample

        installed_here = Path(pdi_defeitos.__file__).resolve().is_relative_to(ROOT)
        check(installed_here, f"pdi_defeitos importável de {Path(pdi_defeitos.__file__).parent}",
              "rode `pip install -e .` na raiz do repositório")
    except ImportError:
        check(False, "pdi_defeitos", "rode `pip install -e .` na raiz do repositório")
        return finish()

    print("\n4. Dataset")
    cfg = load_config(ROOT)
    category_dir = cfg["category_dir"]
    if not check(category_dir.is_dir(), f"pasta {category_dir}", "python scripts/download_data.py"):
        return finish()

    df = list_samples(category_dir)
    summary = df.groupby(["split", "defect_type"]).size()
    for (split, dtype), n in summary.items():
        exp = cfg["dataset"]["expected_counts"].get(f"{split}/{dtype}")
        check(exp is None or n == exp, f"{split}/{dtype}: {n}" + ("" if exp is None else f" (esperado {exp})"),
              "baixe de novo com --force")
    defective = df[df.label == 1]
    check(defective.mask_path.notna().all(), f"todas as {len(defective)} defeituosas têm máscara GT",
          "pasta ground_truth incompleta")

    print("\n5. Leitura de uma amostra")
    row = defective.iloc[0] if len(defective) else df.iloc[0]
    img, mask = load_sample(row, size=cfg["image"]["size"])
    check(img.shape[:2] == mask.shape and img.dtype.name == "uint8",
          f"{row.sample_id}: imagem {img.shape}, máscara {mask.shape}, pixels de defeito = {int(mask.sum())}")
    check(set(map(int, set(mask.ravel().tolist()))) <= {0, 1}, "máscara só tem valores 0 e 1 após redimensionar")

    return finish()


def finish() -> int:
    print()
    if problems:
        print("FALTA RESOLVER:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("TUDO OK — ambiente pronto para a Fase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
