"""Peças de apresentação do viz.py: formatação de tabelas e gráficos dos notebooks."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pdi_defeitos import viz


def test_fmt_br_uses_decimal_comma():
    assert viz.fmt_br(0.8882) == "0,89"
    assert viz.fmt_br(16.8667, 1) == "16,9"
    assert viz.fmt_br(3, 0) == "3"


def test_fmt_table_formats_by_type_and_keeps_the_original():
    df = pd.DataFrame({"auroc": [0.8296, np.nan], "n": [3, 17], "ok": [True, False], "nome": ["a", "b"]}, index=["x", "y"])
    out = viz.fmt_table(df)
    assert out.loc["x"].tolist() == ["0,83", 3, "sim", "a"]
    assert out.loc["y"].tolist() == ["–", 17, "não", "b"]
    assert df.loc["x", "auroc"] == 0.8296  # não altera a tabela de entrada
    assert viz.fmt_table(df, decimals=3).loc["x", "auroc"] == "0,830"


def test_fmt_table_marks_only_the_chosen_row():
    df = pd.DataFrame({"v": [1.0, 2.0]}, index=["a", "b"])
    out = viz.fmt_table(df, mark="b")
    assert list(out.index) == ["a", "b  ← escolhido"]
    with pytest.raises(KeyError):
        viz.fmt_table(df, mark="c")


def test_dataset_available(tmp_path, fake_category, capsys):
    assert viz.dataset_available(fake_category)
    assert capsys.readouterr().out == ""
    assert not viz.dataset_available(tmp_path / "nao_existe")
    assert "opcional" in capsys.readouterr().out  # pula com mensagem, sem erro


def test_hpanels_draws_one_axis_per_panel_and_highlights():
    fig = viz.hpanels(["a", "b", "c"], [{"title": "barras", "values": [1.0, 2.0, 3.0]},
                                        {"title": "pontos", "kind": "dot", "values": [0.8, 0.9, 0.85]}], highlight=1)
    assert len(fig.axes) == 2
    bars = fig.axes[0].patches
    assert [b.get_width() for b in bars] == [1.0, 2.0, 3.0]
    colors = [matplotlib.colors.to_hex(b.get_facecolor()) for b in bars]
    assert colors == [viz.CONTEXT, viz.ACCENT, viz.CONTEXT]
    plt.close(fig)


def test_stacked_hbar_totals_and_column_limit():
    table = pd.DataFrame({"good": [120, 79], "broken": [0, 17]}, index=["Referência", "Avaliação"])
    fig = viz.stacked_hbar(table)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["120", "96"]  # total na ponta de cada barra
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["good", "broken"]
    plt.close(fig)
    with pytest.raises(ValueError):
        viz.stacked_hbar(pd.DataFrame(np.ones((1, 5))))
