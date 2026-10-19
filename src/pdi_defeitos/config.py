"""Leitura do config.yaml e resolução de caminhos a partir da raiz do repositório."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Sobe a partir de `start` (ou da pasta atual) até achar o config.yaml.

    Assim o mesmo código funciona rodando da raiz ou de qualquer subpasta do repositório.
    """
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).is_file():
            return folder
    raise FileNotFoundError(
        f"Não encontrei {CONFIG_NAME} subindo a partir de {here}. "
        "Rode o código de dentro da pasta do repositório."
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    """Carrega o config.yaml e acrescenta caminhos absolutos prontos para uso.

    Chaves extras criadas:
      root         -> raiz do repositório
      data_dir     -> pasta com as categorias do MVTec (ex.: data/raw/mvtec)
      category_dir -> pasta da categoria escolhida (ex.: data/raw/mvtec/bottle)
    """
    root = find_repo_root(root)
    with open(root / CONFIG_NAME, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["root"] = root
    cfg["data_dir"] = (root / cfg["paths"]["data_dir"]).resolve()
    cfg["category_dir"] = cfg["data_dir"] / cfg["category"]
    return cfg
