"""Baixa e extrai a categoria do MVTec AD definida no config.yaml.

Uso (da raiz do repositório, com o ambiente ativado):

    python scripts/download_data.py
    python scripts/download_data.py --archive C:\\Users\\voce\\Downloads\\bottle.tar.xz

A segunda forma usa um arquivo que você já baixou pelo navegador
(útil se o download automático for bloqueado pela rede da faculdade).

Resultado: data/raw/mvtec/bottle/{train,test,ground_truth}/...
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdi_defeitos.config import load_config  # noqa: E402
from pdi_defeitos.io import count_by_folder  # noqa: E402


def download(url: str, dest: Path) -> None:
    """Baixa `url` para `dest` mostrando uma barra de progresso."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def extract(archive: Path, target_dir: Path) -> None:
    """Extrai o .tar.xz em `target_dir` com o filtro de segurança do Python."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        members = tar.getmembers()
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for member in tqdm(members, desc="Extraindo", unit="arq"):
            tar.extract(member, target_dir, **kwargs)


def main() -> int:
    cfg = load_config(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="usar um .tar.xz já baixado em vez de baixar")
    parser.add_argument("--force", action="store_true", help="apagar a pasta da categoria e extrair de novo")
    args = parser.parse_args()

    category, category_dir, data_dir = cfg["category"], cfg["category_dir"], cfg["data_dir"]

    if category_dir.is_dir() and not args.force:
        print(f"[ok] {category_dir} já existe. Use --force para baixar de novo.")
    else:
        if category_dir.is_dir():
            shutil.rmtree(category_dir)
        archive = args.archive
        if archive is None:
            archive = data_dir.parent / f"{category}.tar.xz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.is_file():
                print(f"Baixando {category} (~150 MB) de:\n  {cfg['dataset']['url']}")
                try:
                    download(cfg["dataset"]["url"], archive)
                except Exception as exc:  # rede bloqueada, link trocado etc.
                    print(
                        f"\n[erro] Download falhou: {exc}\n"
                        f"Baixe '{category}' manualmente em {cfg['dataset']['page']}\n"
                        f"e rode: python scripts/download_data.py --archive CAMINHO/{category}.tar.xz"
                    )
                    return 1
        if not archive.is_file():
            print(f"[erro] Arquivo não encontrado: {archive}")
            return 1
        print(f"Extraindo {archive.name} em {data_dir} ...")
        extract(archive, data_dir)
        if not category_dir.is_dir():
            print(f"[erro] Depois de extrair, não encontrei {category_dir}. "
                  f"Veja o que foi criado em {data_dir} e ajuste paths.data_dir no config.yaml.")
            return 1

    counts = count_by_folder(category_dir)
    expected = cfg["dataset"].get("expected_counts", {})
    print("\nConferência das contagens:")
    all_ok = True
    for key in sorted(set(counts) | set(expected)):
        got, exp = counts.get(key, 0), expected.get(key)
        ok = exp is None or got == exp
        all_ok &= ok
        print(f"  {'ok ' if ok else 'DIF'}  {key:<28} {got:>4}" + ("" if exp is None else f"  (esperado {exp})"))
    print("\nPronto." if all_ok else "\n[aviso] Alguma contagem difere do esperado; confira o download.")
    print(f"Licença do dataset: {cfg['dataset']['license']}.")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
