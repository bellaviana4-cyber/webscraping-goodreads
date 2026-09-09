from __future__ import annotations

r"""
Gera um relatório visual standalone em HTML para o projeto Goodreads.

Estrutura
---------
Capa | Metodologia | Visão Geral | Livros | Autores | Exploração | Apêndice

Uso
---
uv run python codigos/gerar_relatorio_goodreads_v2.py --abrir

Também é possível informar caminhos diferentes:
uv run python codigos/gerar_relatorio_goodreads_v2.py \
    --base-completa "caminho/goodreads_best_books.csv" \
    --base-dashboard "caminho/goodreads_dashboard.csv" \
    --base-parcial "caminho/goodreads_lista_parcial.csv" \
    --saida "caminho/relatorio_goodreads_visual_v2.html"
"""

import argparse
import json
import math
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs


# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================

CAMINHO_BASE_COMPLETA_PADRAO = r"C:\Users\bella\webscraping-goodreads\dados\goodreads_best_books.csv"
CAMINHO_BASE_DASHBOARD_PADRAO = r"C:\Users\bella\webscraping-goodreads\dados\goodreads_dashboard.csv"
CAMINHO_BASE_PARCIAL_PADRAO = r"C:\Users\bella\webscraping-goodreads\dados\goodreads_lista_parcial.csv"
CAMINHO_SAIDA_PADRAO = r"C:\Users\bella\webscraping-goodreads\dashboard\relatorio_goodreads_visual_v2.html"

AUTORA = "Isabella Viana"
TITULO = "Goodreads — Best Books Ever"
SUBTITULO = "Web scraping e análise descritiva da lista Best Books Ever"

# Paleta editorial própria desta versão: tons de papel, tinta, terracota e verde.
CORES = {
    "ink": "#1D2228",
    "ink_2": "#28313A",
    "paper": "#F7F3EA",
    "paper_2": "#EFE9DE",
    "card": "#FFFFFF",
    "terracotta": "#C96F4A",
    "terracotta_dark": "#A65336",
    "sage": "#4F6F64",
    "sage_light": "#88A397",
    "gold": "#D9A441",
    "blue": "#496780",
    "plum": "#76536F",
    "text": "#263238",
    "muted": "#6C767D",
    "border": "#E3DDD3",
    "white": "#FFFFFF",
    "soft": "#FAF8F3",
}

CHART_COLORS = [
    CORES["terracotta"],
    CORES["sage"],
    CORES["gold"],
    CORES["blue"],
    CORES["plum"],
    CORES["sage_light"],
]

COLUNAS_NUMERICAS = [
    "ranking",
    "book_id",
    "author_id",
    "avaliacao_media",
    "numero_avaliacoes",
    "score",
    "numero_votos_lista",
    "pagina_lista",
    "nota_ponderada",
]


# =============================================================================
# 1. LEITURA E PREPARAÇÃO
# =============================================================================


def detectar_separador(caminho: str | Path) -> str:
    with open(caminho, "r", encoding="utf-8-sig", errors="replace") as arquivo:
        primeira_linha = arquivo.readline()
    candidatos = {
        ",": primeira_linha.count(","),
        ";": primeira_linha.count(";"),
        "\t": primeira_linha.count("\t"),
        "|": primeira_linha.count("|"),
    }
    return max(candidatos, key=candidatos.get)


def ler_csv(caminho: str | Path, obrigatorio: bool = True) -> pd.DataFrame:
    path = Path(caminho)
    if not path.exists():
        if obrigatorio:
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()

    separador = detectar_separador(path)
    erros: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep=separador, encoding=encoding, low_memory=False)
            if len(df.columns) <= 1:
                raise ValueError("A leitura resultou em uma única coluna.")
            return df
        except Exception as exc:  # noqa: BLE001
            erros.append(f"{encoding}: {exc}")
    raise ValueError("Não foi possível ler o CSV. " + " | ".join(erros))


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for coluna in out.columns:
        if out[coluna].dtype == object:
            out[coluna] = out[coluna].astype("string").str.strip()
    for coluna in COLUNAS_NUMERICAS:
        if coluna in out.columns:
            out[coluna] = pd.to_numeric(out[coluna], errors="coerce")
    return out


def calcular_nota_ponderada(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    out = df.copy()
    rating = pd.to_numeric(out.get("avaliacao_media"), errors="coerce")
    votos = pd.to_numeric(out.get("numero_avaliacoes"), errors="coerce")

    C = float(rating.mean()) if rating.notna().any() else 0.0
    m = float(votos.quantile(0.75)) if votos.notna().any() else 0.0

    if m <= 0:
        out["nota_ponderada"] = rating
    else:
        out["nota_ponderada"] = (votos / (votos + m)) * rating + (m / (votos + m)) * C
    return out, C, m


def criar_faixas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "avaliacao_media" in out.columns:
        out["faixa_avaliacao"] = pd.cut(
            out["avaliacao_media"],
            bins=[-np.inf, 3.5, 4.0, 4.25, 4.5, np.inf],
            labels=["Até 3,5", "3,5 a 4,0", "4,0 a 4,25", "4,25 a 4,5", "Acima de 4,5"],
            include_lowest=True,
        ).astype("string")

    if "numero_avaliacoes" in out.columns:
        serie = pd.to_numeric(out["numero_avaliacoes"], errors="coerce")
        try:
            out["faixa_popularidade"] = pd.qcut(
                serie.rank(method="first"),
                q=4,
                labels=["Baixa", "Média", "Alta", "Muito alta"],
            ).astype("string")
        except ValueError:
            out["faixa_popularidade"] = "Não classificada"
    return out


def preparar_bases(
    caminho_completa: str,
    caminho_dashboard: str,
    caminho_parcial: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    completa = normalizar_colunas(ler_csv(caminho_completa, obrigatorio=True))
    dashboard = normalizar_colunas(ler_csv(caminho_dashboard, obrigatorio=False))
    parcial = normalizar_colunas(ler_csv(caminho_parcial, obrigatorio=False))

    obrigatorias = {"titulo", "autor", "avaliacao_media", "numero_avaliacoes"}
    faltantes = sorted(obrigatorias - set(completa.columns))
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes em goodreads_best_books.csv: {faltantes}")

    if "ranking" in completa.columns:
        completa = completa.sort_values("ranking", na_position="last")

    if "book_id" in completa.columns:
        completa = completa.drop_duplicates(subset="book_id", keep="first")
    else:
        completa = completa.drop_duplicates(subset=["titulo", "autor"], keep="first")

    completa = completa.reset_index(drop=True)
    completa, C, m = calcular_nota_ponderada(completa)
    completa = criar_faixas(completa)

    if dashboard.empty:
        dashboard = completa.copy()
    else:
        if "book_id" in dashboard.columns and "book_id" in completa.columns:
            extras = completa[["book_id", "nota_ponderada", "faixa_avaliacao", "faixa_popularidade"]].copy()
            dashboard = dashboard.drop(
                columns=[c for c in ["nota_ponderada", "faixa_avaliacao", "faixa_popularidade"] if c in dashboard.columns]
            ).merge(extras, on="book_id", how="left")
        else:
            dashboard, _, _ = calcular_nota_ponderada(dashboard)
            dashboard = criar_faixas(dashboard)

    return completa, dashboard, parcial, {"C": C, "m": m}


# =============================================================================
# 2. MÉTRICAS E TABELAS ANALÍTICAS
# =============================================================================


def safe_float(valor: Any) -> float | None:
    try:
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    except (TypeError, ValueError):
        return None


def fmt_int(valor: Any) -> str:
    numero = safe_float(valor)
    if numero is None:
        return "—"
    return f"{int(round(numero)):,}".replace(",", ".")


def fmt_dec(valor: Any, casas: int = 2) -> str:
    numero = safe_float(valor)
    if numero is None:
        return "—"
    return f"{numero:.{casas}f}".replace(".", ",")


def fmt_pct(valor: Any, casas: int = 1) -> str:
    numero = safe_float(valor)
    if numero is None:
        return "—"
    return f"{numero * 100:.{casas}f}%".replace(".", ",")


def fmt_compacto(valor: Any) -> str:
    numero = safe_float(valor)
    if numero is None:
        return "—"
    absoluto = abs(numero)
    if absoluto >= 1_000_000_000:
        return f"{numero / 1_000_000_000:.1f} bi".replace(".", ",")
    if absoluto >= 1_000_000:
        return f"{numero / 1_000_000:.1f} mi".replace(".", ",")
    if absoluto >= 1_000:
        return f"{numero / 1_000:.1f} mil".replace(".", ",")
    return fmt_int(numero)


def resumo_geral(df: pd.DataFrame) -> dict[str, Any]:
    avaliacao = pd.to_numeric(df["avaliacao_media"], errors="coerce")
    reviews = pd.to_numeric(df["numero_avaliacoes"], errors="coerce")
    retorno = {
        "livros": int(len(df)),
        "autores": int(df["autor"].nunique(dropna=True)),
        "avaliacao_media": safe_float(avaliacao.mean()),
        "avaliacao_mediana": safe_float(avaliacao.median()),
        "pct_4_mais": safe_float((avaliacao >= 4.0).mean()),
        "total_avaliacoes": safe_float(reviews.sum()),
        "mediana_avaliacoes": safe_float(reviews.median()),
        "votos_lista": safe_float(df["numero_votos_lista"].sum()) if "numero_votos_lista" in df.columns else None,
    }
    return retorno


def correlacao_spearman_sem_scipy(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    cols = [c for c in colunas if c in df.columns]
    base = df[cols].apply(pd.to_numeric, errors="coerce")
    return base.rank(method="average").corr(method="pearson")


def top_livros(df: pd.DataFrame, coluna: str, n: int = 10, crescente: bool = False) -> pd.DataFrame:
    if coluna not in df.columns:
        return pd.DataFrame()
    cols = [
        c
        for c in [
            "ranking", "titulo", "autor", "avaliacao_media", "numero_avaliacoes",
            "nota_ponderada", "score", "numero_votos_lista", "url_livro", "url_capa"
        ]
        if c in df.columns
    ]
    return (
        df[cols]
        .dropna(subset=[coluna])
        .sort_values(coluna, ascending=crescente)
        .head(n)
        .copy()
    )


def autores_resumo(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    agregacoes: dict[str, tuple[str, str]] = {
        "quantidade_livros": ("titulo", "count"),
        "avaliacao_media": ("avaliacao_media", "mean"),
        "nota_ponderada_media": ("nota_ponderada", "mean"),
        "total_avaliacoes": ("numero_avaliacoes", "sum"),
        "mediana_rank": ("ranking", "median") if "ranking" in base.columns else ("numero_avaliacoes", "count"),
    }
    agrupado = base.groupby("autor", dropna=True).agg(**agregacoes).reset_index()

    melhores = (
        base.sort_values(["nota_ponderada", "numero_avaliacoes"], ascending=[False, False])
        .drop_duplicates("autor")[["autor", "titulo"]]
        .rename(columns={"titulo": "livro_destaque"})
    )
    agrupado = agrupado.merge(melhores, on="autor", how="left")
    return agrupado


def completude(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for coluna in df.columns:
        validos = int(df[coluna].notna().sum())
        total = int(len(df))
        linhas.append({
            "variavel": coluna,
            "preenchidos": validos,
            "ausentes": total - validos,
            "completude": validos / total if total else 0,
        })
    return pd.DataFrame(linhas).sort_values("completude", ascending=True)


def rating_bands(df: pd.DataFrame) -> pd.DataFrame:
    ordem = ["Até 3,5", "3,5 a 4,0", "4,0 a 4,25", "4,25 a 4,5", "Acima de 4,5"]
    s = df["faixa_avaliacao"].fillna("Sem informação").astype(str)
    out = s.value_counts().rename_axis("faixa").reset_index(name="livros")
    out["ordem"] = out["faixa"].map({v: i for i, v in enumerate(ordem)}).fillna(99)
    return out.sort_values("ordem").drop(columns="ordem")


def insights_gerais(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    resumo = resumo_geral(df)
    corr = correlacao_spearman_sem_scipy(
        df,
        ["avaliacao_media", "numero_avaliacoes", "ranking", "score", "numero_votos_lista"],
    )

    corr_pop = None
    if {"avaliacao_media", "numero_avaliacoes"}.issubset(corr.columns):
        corr_pop = safe_float(corr.loc["avaliacao_media", "numero_avaliacoes"])

    corr_score = None
    if {"score", "numero_votos_lista"}.issubset(corr.columns):
        corr_score = safe_float(corr.loc["score", "numero_votos_lista"])

    top_author = autores_resumo(df).sort_values("quantidade_livros", ascending=False).head(1)
    if not top_author.empty:
        nome_autor = str(top_author.iloc[0]["autor"])
        n_autor = int(top_author.iloc[0]["quantidade_livros"])
        texto_autor = f"{nome_autor} aparece {n_autor} vezes na lista analisada."
    else:
        texto_autor = "Não foi possível identificar o autor mais recorrente."

    corr_pop_texto = "não calculada"
    if corr_pop is not None:
        corr_pop_texto = f"{corr_pop:+.2f}"
    corr_score_texto = "não calculada"
    if corr_score is not None:
        corr_score_texto = f"{corr_score:+.2f}"

    return [
        (
            "Concentração de boas avaliações",
            fmt_pct(resumo["pct_4_mais"]),
            "dos livros possuem avaliação média de pelo menos 4,0.",
        ),
        (
            "Popularidade típica",
            fmt_compacto(resumo["mediana_avaliacoes"]),
            "é a mediana do número de avaliações por livro.",
        ),
        (
            "Nota × popularidade",
            corr_pop_texto,
            "correlação de Spearman entre avaliação média e número de avaliações.",
        ),
        (
            "Score × votos da lista",
            corr_score_texto,
            "correlação de Spearman entre as duas métricas da lista, quando disponíveis.",
        ),
        (
            "Autor mais recorrente",
            "Destaque",
            texto_autor,
        ),
    ]


# =============================================================================
# 3. GRÁFICOS
# =============================================================================


def plot_layout(titulo: str, subtitulo: str = "", height: int = 360) -> dict[str, Any]:
    return {
        "height": height,
        "margin": {"l": 38, "r": 20, "t": 78, "b": 48},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, Segoe UI, Arial, sans-serif", "color": CORES["text"], "size": 12},
        "title": {
            "text": f"<b>{titulo}</b>" + (f"<br><span style='font-size:11px;color:{CORES['muted']}'>{subtitulo}</span>" if subtitulo else ""),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 15, "color": CORES["ink"]},
        },
        "hoverlabel": {"bgcolor": CORES["ink"], "font_color": "white", "bordercolor": CORES["ink"]},
        "showlegend": False,
    }


def estilizar_eixos(fig: go.Figure, xgrid: bool = False, ygrid: bool = True) -> go.Figure:
    fig.update_xaxes(
        showgrid=xgrid,
        gridcolor="#ECE6DB",
        zeroline=False,
        linecolor="#D9D2C7",
        tickfont={"color": CORES["muted"], "size": 10},
        title_font={"color": CORES["muted"], "size": 11},
    )
    fig.update_yaxes(
        showgrid=ygrid,
        gridcolor="#ECE6DB",
        zeroline=False,
        linecolor="#D9D2C7",
        tickfont={"color": CORES["muted"], "size": 10},
        title_font={"color": CORES["muted"], "size": 11},
    )
    return fig


def grafico_histograma(df: pd.DataFrame) -> go.Figure:
    serie = pd.to_numeric(df["avaliacao_media"], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=serie,
        nbinsx=22,
        marker={"color": CORES["terracotta"], "line": {"color": CORES["white"], "width": 1}},
        hovertemplate="Avaliação: %{x:.2f}<br>Livros: %{y}<extra></extra>",
    ))
    media = serie.mean()
    mediana = serie.median()
    fig.add_vline(x=media, line_width=2, line_dash="dash", line_color=CORES["sage"])
    fig.add_vline(x=mediana, line_width=2, line_dash="dot", line_color=CORES["blue"])
    fig.update_layout(**plot_layout("Distribuição das avaliações", "Linhas: média e mediana da base", 390))
    fig.update_xaxes(title="Avaliação média")
    fig.update_yaxes(title="Quantidade de livros")
    return estilizar_eixos(fig)


def grafico_donut_faixas(df: pd.DataFrame) -> go.Figure:
    bands = rating_bands(df)
    fig = go.Figure(go.Pie(
        labels=bands["faixa"],
        values=bands["livros"],
        hole=0.62,
        sort=False,
        marker={"colors": CHART_COLORS[: len(bands)]},
        textinfo="percent",
        textfont={"size": 11},
        hovertemplate="%{label}<br>%{value} livros · %{percent}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Faixas de avaliação", "Participação dos livros em cada intervalo", 390))
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "y": -0.14, "x": 0, "font": {"size": 10}},
        annotations=[{
            "text": f"<b>{len(df)}</b><br><span style='font-size:10px'>livros</span>",
            "x": 0.5, "y": 0.5, "showarrow": False,
            "font": {"size": 20, "color": CORES["ink"]},
        }],
    )
    return fig


def grafico_scatter_popularidade(df: pd.DataFrame) -> go.Figure:
    base = df[[c for c in ["titulo", "autor", "ranking", "avaliacao_media", "numero_avaliacoes", "nota_ponderada"] if c in df.columns]].dropna(
        subset=["avaliacao_media", "numero_avaliacoes"]
    ).copy()
    base["log_reviews"] = np.log10(base["numero_avaliacoes"].clip(lower=1))
    if "ranking" in base.columns:
        cor = base["ranking"]
        colorscale = [[0, CORES["gold"]], [0.5, CORES["terracotta"]], [1, CORES["sage"]]]
        colorbar = {"title": "Ranking", "thickness": 10, "len": 0.72, "tickfont": {"size": 9}}
    else:
        cor = CORES["terracotta"]
        colorscale = None
        colorbar = None

    sizes = pd.to_numeric(base.get("numero_avaliacoes"), errors="coerce").fillna(0)
    if sizes.max() > sizes.min():
        marker_size = 8 + 12 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
    else:
        marker_size = np.full(len(base), 10)

    hover = []
    for _, row in base.iterrows():
        hover.append(
            f"<b>{escape(str(row.get('titulo', '')))}</b><br>"
            f"{escape(str(row.get('autor', '')))}<br>"
            f"Avaliação: {fmt_dec(row.get('avaliacao_media'))}<br>"
            f"Avaliações: {fmt_int(row.get('numero_avaliacoes'))}<br>"
            f"Ranking: {fmt_int(row.get('ranking'))}"
        )

    fig = go.Figure(go.Scatter(
        x=base["numero_avaliacoes"],
        y=base["avaliacao_media"],
        mode="markers",
        marker={
            "size": marker_size,
            "color": cor,
            "colorscale": colorscale,
            "reversescale": False,
            "opacity": 0.78,
            "line": {"width": 0.7, "color": "white"},
            **({"colorbar": colorbar} if colorbar else {}),
        },
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Avaliação × popularidade", "Cada ponto representa um livro; tamanho acompanha o volume de avaliações", 430))
    fig.update_xaxes(type="log", title="Número de avaliações · escala log")
    fig.update_yaxes(title="Avaliação média")
    return estilizar_eixos(fig)


def grafico_rank_score(df: pd.DataFrame) -> go.Figure:
    if "ranking" not in df.columns or "score" not in df.columns:
        return go.Figure()
    base = df[["ranking", "score", "titulo", "autor"]].dropna().sort_values("ranking")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base["ranking"],
        y=base["score"],
        mode="lines+markers",
        line={"color": CORES["blue"], "width": 2},
        marker={"color": CORES["gold"], "size": 6, "line": {"color": "white", "width": 0.5}},
        customdata=np.stack([base["titulo"].astype(str), base["autor"].astype(str)], axis=-1),
        hovertemplate="Ranking #%{x}<br><b>%{customdata[0]}</b><br>%{customdata[1]}<br>Score: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Ranking × score da lista", "Leitura da variação do score ao longo das posições coletadas", 360))
    fig.update_xaxes(title="Posição no ranking", autorange="reversed")
    fig.update_yaxes(title="Score")
    return estilizar_eixos(fig)


def grafico_correlacao(df: pd.DataFrame) -> go.Figure:
    nomes = {
        "avaliacao_media": "Avaliação",
        "numero_avaliacoes": "Nº avaliações",
        "ranking": "Ranking",
        "score": "Score",
        "numero_votos_lista": "Votos da lista",
        "nota_ponderada": "Nota ponderada",
    }
    colunas = [c for c in nomes if c in df.columns]
    corr = correlacao_spearman_sem_scipy(df, colunas)
    if corr.empty:
        return go.Figure()
    labels = [nomes[c] for c in corr.columns]
    texto = [[f"{v:+.2f}" if pd.notna(v) else "—" for v in row] for row in corr.values]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        colorscale=[
            [0.0, "#496780"],
            [0.5, "#F7F3EA"],
            [1.0, "#C96F4A"],
        ],
        text=texto,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar={"title": "ρ", "thickness": 11, "len": 0.75},
    ))
    fig.update_layout(**plot_layout("Relações entre as métricas", "Correlação de Spearman; valores próximos de ±1 indicam associação monotônica mais forte", 430))
    fig.update_xaxes(side="bottom")
    return fig


def grafico_barras_livros(df_top: pd.DataFrame, coluna: str, titulo: str, subtitulo: str, formato: str = "decimal") -> go.Figure:
    if df_top.empty:
        return go.Figure()
    base = df_top.copy().sort_values(coluna, ascending=True)
    valores = base[coluna]
    labels = base["titulo"].astype(str)
    custom = np.stack([
        base["autor"].astype(str),
        base["avaliacao_media"].astype(float),
        base["numero_avaliacoes"].astype(float),
    ], axis=-1)
    if formato == "inteiro":
        text = [fmt_compacto(v) for v in valores]
        hover_value = "%{x:,.0f}"
    else:
        text = [fmt_dec(v, 3 if coluna == "nota_ponderada" else 2) for v in valores]
        hover_value = "%{x:.3f}" if coluna == "nota_ponderada" else "%{x:.2f}"

    fig = go.Figure(go.Bar(
        x=valores,
        y=labels,
        orientation="h",
        marker={"color": CORES["terracotta"], "line": {"color": "white", "width": 0.6}},
        text=text,
        textposition="outside",
        cliponaxis=False,
        customdata=custom,
        hovertemplate=(
            "<b>%{y}</b><br>%{customdata[0]}<br>"
            + f"Valor: {hover_value}<br>"
            + "Avaliação média: %{customdata[1]:.2f}<br>"
            + "Nº avaliações: %{customdata[2]:,.0f}<extra></extra>"
        ),
    ))
    fig.update_layout(**plot_layout(titulo, subtitulo, 440))
    fig.update_yaxes(automargin=True, title="")
    fig.update_xaxes(title="")
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


def grafico_autores_quantidade(autores: pd.DataFrame, n: int = 12) -> go.Figure:
    base = autores.sort_values(["quantidade_livros", "total_avaliacoes"], ascending=[False, False]).head(n).sort_values("quantidade_livros")
    fig = go.Figure(go.Bar(
        x=base["quantidade_livros"],
        y=base["autor"],
        orientation="h",
        marker={"color": CORES["sage"]},
        text=[fmt_int(v) for v in base["quantidade_livros"]],
        textposition="outside",
        customdata=np.stack([base["avaliacao_media"], base["total_avaliacoes"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Livros: %{x}<br>Avaliação média: %{customdata[0]:.2f}<br>Total de avaliações: %{customdata[1]:,.0f}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Autores com maior presença", "Quantidade de títulos do autor na lista coletada", 450))
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


def grafico_autores_popularidade(autores: pd.DataFrame, n: int = 12) -> go.Figure:
    base = autores.sort_values("total_avaliacoes", ascending=False).head(n).sort_values("total_avaliacoes")
    fig = go.Figure(go.Bar(
        x=base["total_avaliacoes"],
        y=base["autor"],
        orientation="h",
        marker={"color": CORES["gold"]},
        text=[fmt_compacto(v) for v in base["total_avaliacoes"]],
        textposition="outside",
        customdata=np.stack([base["quantidade_livros"], base["avaliacao_media"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Total de avaliações: %{x:,.0f}<br>Livros na lista: %{customdata[0]:.0f}<br>Avaliação média: %{customdata[1]:.2f}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Autores com maior alcance", "Soma do número de avaliações dos livros de cada autor", 450))
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


def grafico_autores_mapa(autores: pd.DataFrame) -> go.Figure:
    base = autores.dropna(subset=["quantidade_livros", "avaliacao_media", "total_avaliacoes"]).copy()
    if base.empty:
        return go.Figure()
    tamanho = np.sqrt(base["total_avaliacoes"].clip(lower=0))
    if tamanho.max() > tamanho.min():
        tamanho = 9 + 25 * (tamanho - tamanho.min()) / (tamanho.max() - tamanho.min())
    else:
        tamanho = np.full(len(base), 12)

    fig = go.Figure(go.Scatter(
        x=base["quantidade_livros"],
        y=base["avaliacao_media"],
        mode="markers",
        marker={
            "size": tamanho,
            "color": base["total_avaliacoes"],
            "colorscale": [[0, CORES["paper_2"]], [0.5, CORES["terracotta"]], [1, CORES["plum"]]],
            "opacity": 0.78,
            "line": {"color": "white", "width": 1},
            "colorbar": {"title": "Total de<br>avaliações", "thickness": 11, "len": 0.65},
        },
        customdata=np.stack([
            base["autor"].astype(str),
            base["total_avaliacoes"].astype(float),
            base["livro_destaque"].fillna("").astype(str),
        ], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Livros: %{x}<br>Avaliação média: %{y:.2f}<br>Total avaliações: %{customdata[1]:,.0f}<br>Destaque: %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Presença × avaliação dos autores", "Tamanho e cor dos pontos acompanham o alcance acumulado", 450))
    fig.update_xaxes(title="Quantidade de livros na lista")
    fig.update_yaxes(title="Avaliação média")
    return estilizar_eixos(fig)


def grafico_completude(df: pd.DataFrame) -> go.Figure:
    comp = completude(df).tail(15).sort_values("completude")
    fig = go.Figure(go.Bar(
        x=comp["completude"] * 100,
        y=comp["variavel"],
        orientation="h",
        marker={"color": CORES["blue"]},
        text=[f"{v*100:.0f}%" for v in comp["completude"]],
        textposition="inside",
        insidetextanchor="end",
        hovertemplate="%{y}<br>Completude: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**plot_layout("Completude das variáveis", "Percentual de registros preenchidos", 430))
    fig.update_xaxes(range=[0, 100], title="Completude (%)")
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


def fig_html(fig: go.Figure, div_id: str) -> str:
    if fig is None or not fig.data:
        return '<div class="empty-state">Visualização indisponível para a base atual.</div>'
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        div_id=div_id,
    )


# =============================================================================
# 4. COMPONENTES HTML
# =============================================================================


def icon(nome: str) -> str:
    icons = {
        "home": "⌂",
        "method": "◎",
        "overview": "◫",
        "books": "▤",
        "authors": "◉",
        "search": "⌕",
        "appendix": "≡",
        "arrow": "→",
        "star": "★",
    }
    return icons.get(nome, "•")


def kpi_card(valor: str, rotulo: str, legenda: str, classe: str = "terracotta") -> str:
    return f"""
    <article class="kpi {escape(classe)}">
      <div class="kpi-label">{escape(rotulo)}</div>
      <div class="kpi-value">{escape(valor)}</div>
      <div class="kpi-caption">{escape(legenda)}</div>
    </article>
    """


def chapter_header(numero: str, titulo: str, subtitulo: str) -> str:
    return f"""
    <header class="chapter-header">
      <div class="chapter-number">{escape(numero)}</div>
      <div>
        <div class="chapter-kicker">GOODREADS · ANÁLISE DESCRITIVA</div>
        <h1>{escape(titulo)}</h1>
        <p>{escape(subtitulo)}</p>
      </div>
    </header>
    """


def mini_table(df: pd.DataFrame, colunas: list[tuple[str, str]], max_rows: int = 10) -> str:
    if df.empty:
        return '<div class="empty-state">Sem registros para apresentar.</div>'
    cabecalho = "".join(f"<th>{escape(rotulo)}</th>" for _, rotulo in colunas)
    linhas = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for coluna, _ in colunas:
            valor = row.get(coluna)
            if pd.isna(valor):
                texto = "—"
            elif coluna in {"ranking", "quantidade_livros"}:
                texto = fmt_int(valor)
            elif coluna in {"avaliacao_media", "nota_ponderada", "nota_ponderada_media"}:
                texto = fmt_dec(valor, 3 if "ponderada" in coluna else 2)
            elif coluna in {"numero_avaliacoes", "total_avaliacoes", "score", "numero_votos_lista"}:
                texto = fmt_compacto(valor)
            else:
                texto = str(valor)
            cells.append(f"<td>{escape(texto)}</td>")
        linhas.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <div class="table-wrap compact-table-wrap">
      <table class="data-table compact-table">
        <thead><tr>{cabecalho}</tr></thead>
        <tbody>{''.join(linhas)}</tbody>
      </table>
    </div>
    """


def feature_book_card(label: str, livro: pd.Series | None, classe: str = "terracotta") -> str:
    if livro is None:
        return ""
    titulo = escape(str(livro.get("titulo", "—")))
    autor = escape(str(livro.get("autor", "—")))
    capa = str(livro.get("url_capa", "") or "")
    ranking = fmt_int(livro.get("ranking"))
    rating = fmt_dec(livro.get("avaliacao_media"))
    reviews = fmt_compacto(livro.get("numero_avaliacoes"))
    weighted = fmt_dec(livro.get("nota_ponderada"), 3)
    url = str(livro.get("url_livro", "") or "")

    cover_html = (
        f'<img src="{escape(capa)}" alt="Capa de {titulo}" loading="lazy" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        if capa and capa.lower() not in {"nan", "none", "<na>"}
        else ""
    )
    fallback_display = "none" if cover_html else "flex"
    link_open = f'<a class="book-open" href="{escape(url)}" target="_blank" rel="noopener">Abrir no Goodreads {icon("arrow")}</a>' if url and url.lower() not in {"nan", "none", "<na>"} else ""

    return f"""
    <article class="feature-book {escape(classe)}">
      <div class="feature-cover">
        {cover_html}
        <div class="cover-fallback" style="display:{fallback_display}"><span>{icon('books')}</span><small>GOODREADS</small></div>
      </div>
      <div class="feature-copy">
        <div class="feature-label">{escape(label)}</div>
        <h3>{titulo}</h3>
        <p class="feature-author">{autor}</p>
        <div class="feature-metrics">
          <span><b>{rating}</b> avaliação</span>
          <span><b>{reviews}</b> avaliações</span>
          <span><b>#{ranking}</b> ranking</span>
          <span><b>{weighted}</b> nota pond.</span>
        </div>
        {link_open}
      </div>
    </article>
    """


def home_html(meta: dict[str, Any], df: pd.DataFrame) -> str:
    resumo = resumo_geral(df)
    cards = [
        ("Metodologia", "Como os dados foram coletados, tratados e transformados em indicadores.", "metodologia", "method"),
        ("Visão Geral", "Panorama da base, distribuição das notas e relações entre as principais métricas.", "visao", "overview"),
        ("Livros", "Rankings de destaque por nota ponderada, popularidade e score da lista.", "livros", "books"),
        ("Autores", "Presença, avaliação média e alcance acumulado dos autores.", "autores", "authors"),
        ("Exploração", "Tabela pesquisável para consultar individualmente os livros coletados.", "exploracao", "search"),
        ("Apêndice", "Dicionário das variáveis, qualidade dos dados e regras de interpretação.", "apendice", "appendix"),
    ]
    cards_html = "".join(
        f"""
        <button class="home-nav-card" type="button" data-open="{aba}">
          <span class="home-nav-icon">{icon(icone)}</span>
          <div><h3>{escape(titulo)}</h3><p>{escape(desc)}</p></div>
          <span class="home-nav-arrow">{icon('arrow')}</span>
        </button>
        """
        for titulo, desc, aba, icone in cards
    )

    return f"""
    <section class="tab active" data-tab="home">
      <div class="cover-page">
        <div class="cover-noise"></div>
        <div class="cover-content">
          <div class="cover-eyebrow">PROJETO DE WEBSCRAPING · PYTHON</div>
          <h1>{escape(meta['titulo'])}</h1>
          <p class="cover-subtitle">{escape(meta['subtitulo'])}</p>
          <div class="cover-rule"></div>
          <p class="cover-description">Relatório visual e interativo construído a partir dos livros coletados na lista <i>Best Books Ever</i>, com foco em avaliação, popularidade, ranking e presença de autores.</p>
          <div class="cover-stats">
            <div><strong>{fmt_int(resumo['livros'])}</strong><span>livros</span></div>
            <div><strong>{fmt_int(resumo['autores'])}</strong><span>autores</span></div>
            <div><strong>{fmt_dec(resumo['avaliacao_media'])}</strong><span>avaliação média</span></div>
            <div><strong>{fmt_compacto(resumo['total_avaliacoes'])}</strong><span>avaliações</span></div>
          </div>
          <div class="cover-meta"><span>{escape(meta['autora'])}</span><span>Gerado em {escape(meta['data_processamento'])}</span></div>
        </div>
      </div>
      <div class="home-navigation">
        <div class="home-navigation-title"><span>Explore o relatório</span><p>Navegue pelos capítulos usando os cards abaixo ou a barra lateral.</p></div>
        <div class="home-nav-grid">{cards_html}</div>
      </div>
    </section>
    """


def metodologia_html(meta: dict[str, Any], df: pd.DataFrame, parcial: pd.DataFrame, params: dict[str, float]) -> str:
    comp = completude(df)
    n_colunas = len(df.columns)
    pct_comp = float(comp["completude"].mean()) if not comp.empty else 0
    duplicados = int(df.duplicated(subset="book_id").sum()) if "book_id" in df.columns else int(df.duplicated(subset=["titulo", "autor"]).sum())
    paginas = int(pd.to_numeric(df.get("pagina_lista"), errors="coerce").nunique()) if "pagina_lista" in df.columns else None
    parcial_n = len(parcial) if not parcial.empty else None

    return f"""
    <section class="tab" data-tab="metodologia">
      <div class="section stack">
        {chapter_header('01', 'Metodologia', 'Da requisição das páginas à preparação das bases utilizadas no relatório.')}

        <div class="method-flow">
          <div class="method-step"><span>01</span><div><b>Requisição</b><p>As páginas da lista são acessadas com <code>requests</code>, cabeçalho de navegador, timeout, tentativas e pausas entre requisições.</p></div></div>
          <div class="method-step"><span>02</span><div><b>Leitura do HTML</b><p>O conteúdo é interpretado com <code>BeautifulSoup</code> e os livros são identificados pelos elementos da tabela da lista.</p></div></div>
          <div class="method-step"><span>03</span><div><b>Estruturação</b><p>Título, autor, ranking, avaliação, volume de avaliações, score, votos e URLs são organizados em um DataFrame.</p></div></div>
          <div class="method-step"><span>04</span><div><b>Tratamento</b><p>São ajustados tipos numéricos, duplicidades e faixas auxiliares. A nota ponderada equilibra avaliação e popularidade.</p></div></div>
          <div class="method-step"><span>05</span><div><b>Entrega</b><p>As bases finais alimentam este relatório standalone, gerado como um único arquivo HTML com Plotly incorporado.</p></div></div>
        </div>

        <div class="grid g4">
          {kpi_card(fmt_int(len(df)), 'Registros finais', 'Quantidade de livros após organização e remoção de duplicidades.', 'terracotta')}
          {kpi_card(fmt_int(n_colunas), 'Variáveis', 'Número de colunas presentes na base completa usada no relatório.', 'sage')}
          {kpi_card(fmt_pct(pct_comp), 'Completude média', 'Média do percentual preenchido entre as variáveis da base.', 'gold')}
          {kpi_card(fmt_int(duplicados), 'Duplicidades finais', 'Duplicatas pelo identificador do livro ou pelo par título–autor.', 'blue')}
        </div>

        <div class="grid g2">
          <article class="content-card formula-card">
            <div class="card-kicker">NOTA PONDERADA</div>
            <h3>Como a comparação entre livros é estabilizada</h3>
            <div class="formula">WR = <span>v</span>/<small>v + m</small> · R &nbsp;+&nbsp; <span>m</span>/<small>v + m</small> · C</div>
            <div class="formula-legend">
              <p><b>R</b> avaliação média do livro</p>
              <p><b>v</b> número de avaliações do livro</p>
              <p><b>C</b> média geral da base = <strong>{fmt_dec(params['C'], 3)}</strong></p>
              <p><b>m</b> percentil 75 do nº de avaliações = <strong>{fmt_int(params['m'])}</strong></p>
            </div>
            <p class="note">A ponderação reduz o destaque de notas altas apoiadas em pouco volume e mantém a interpretação na mesma escala de avaliação.</p>
          </article>

          <article class="content-card source-card">
            <div class="card-kicker">RECORTE DA COLETA</div>
            <h3>O que efetivamente entra na análise</h3>
            <div class="source-list">
              <div><span>Fonte</span><b>Goodreads · Best Books Ever</b></div>
              <div><span>Páginas identificadas</span><b>{fmt_int(paginas) if paginas is not None else '—'}</b></div>
              <div><span>Registros no arquivo parcial</span><b>{fmt_int(parcial_n) if parcial_n is not None else '—'}</b></div>
              <div><span>Unidade de análise</span><b>Livro</b></div>
            </div>
            <p class="note">Gêneros e detalhes das páginas individuais não foram incorporados ao relatório porque a coleta desses campos não apresentou estabilidade suficiente.</p>
          </article>
        </div>

        <div class="callout"><b>Leitura correta.</b> O relatório descreve os livros presentes no recorte coletado da lista. As medidas de score e votos pertencem à dinâmica da própria lista do Goodreads; a avaliação média e o número de avaliações representam métricas gerais exibidas para cada livro.</div>
      </div>
    </section>
    """


def insight_cards_html(insights: list[tuple[str, str, str]]) -> str:
    return "".join(
        f"""
        <article class="insight-card">
          <span>{escape(titulo)}</span>
          <strong>{escape(valor)}</strong>
          <p>{escape(texto)}</p>
        </article>
        """
        for titulo, valor, texto in insights
    )


def visao_html(df: pd.DataFrame, graficos: dict[str, str]) -> str:
    resumo = resumo_geral(df)
    bands = rating_bands(df)
    bands["participacao"] = bands["livros"] / bands["livros"].sum() if bands["livros"].sum() else 0
    band_table = mini_table(bands, [("faixa", "Faixa"), ("livros", "Livros"), ("participacao", "Participação")], max_rows=10)
    # Corrige formatação específica de participação na tabela gerada acima por uma tabela dedicada simples.
    band_rows = "".join(
        f"<tr><td>{escape(str(r.faixa))}</td><td>{fmt_int(r.livros)}</td><td>{fmt_pct(r.participacao)}</td></tr>"
        for r in bands.itertuples()
    )
    band_table = f'<div class="table-wrap compact-table-wrap"><table class="data-table compact-table"><thead><tr><th>Faixa</th><th>Livros</th><th>Participação</th></tr></thead><tbody>{band_rows}</tbody></table></div>'

    return f"""
    <section class="tab" data-tab="visao">
      <div class="section stack">
        {chapter_header('02', 'Visão Geral', 'Uma leitura sintética da distribuição das avaliações, da popularidade e das relações entre as métricas disponíveis.')}

        <div class="grid g5">
          {kpi_card(fmt_int(resumo['livros']), 'Livros', 'Registros finais analisados.', 'terracotta')}
          {kpi_card(fmt_int(resumo['autores']), 'Autores', 'Autores distintos presentes na base.', 'sage')}
          {kpi_card(fmt_dec(resumo['avaliacao_media']), 'Avaliação média', 'Média simples das avaliações dos livros.', 'gold')}
          {kpi_card(fmt_compacto(resumo['total_avaliacoes']), 'Avaliações', 'Soma do volume de avaliações dos livros.', 'blue')}
          {kpi_card(fmt_pct(resumo['pct_4_mais']), 'Nota ≥ 4,0', 'Proporção dos livros com avaliação média de pelo menos 4.', 'plum')}
        </div>

        <div class="insight-strip">{insight_cards_html(insights_gerais(df))}</div>

        <div class="grid g2 visual-pair">
          <article class="content-card chart-card">{graficos['histograma']}</article>
          <article class="content-card chart-card">{graficos['donut']}</article>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['scatter_pop']}</div>
          <div class="split-aside"><div class="card-kicker">COMPLEMENTO</div><h3>Faixas de avaliação</h3><p>Distribuição tabular usada para complementar a leitura do histograma e do gráfico de participação.</p>{band_table}</div>
        </div>

        <div class="grid g2 visual-pair">
          <article class="content-card chart-card">{graficos['rank_score']}</article>
          <article class="content-card chart-card">{graficos['corr']}</article>
        </div>
      </div>
    </section>
    """


def livros_html(df: pd.DataFrame, graficos: dict[str, str]) -> str:
    top_weighted = top_livros(df, "nota_ponderada", 10)
    top_reviews = top_livros(df, "numero_avaliacoes", 10)
    top_score = top_livros(df, "score", 10) if "score" in df.columns else pd.DataFrame()

    rank1 = None
    if "ranking" in df.columns and df["ranking"].notna().any():
        rank1 = df.sort_values("ranking").iloc[0]
    weighted1 = top_weighted.iloc[0] if not top_weighted.empty else None
    reviews1 = top_reviews.iloc[0] if not top_reviews.empty else None
    score1 = top_score.iloc[0] if not top_score.empty else None

    cards = "".join([
        feature_book_card("#1 DA LISTA", rank1, "ink"),
        feature_book_card("MELHOR NOTA PONDERADA", weighted1, "terracotta"),
        feature_book_card("MAIOR POPULARIDADE", reviews1, "sage"),
        feature_book_card("MAIOR SCORE DA LISTA", score1, "gold"),
    ])

    tabela_weighted = mini_table(top_weighted, [("ranking", "Rank"), ("titulo", "Livro"), ("autor", "Autor"), ("nota_ponderada", "Nota pond."), ("numero_avaliacoes", "Avaliações")])
    tabela_reviews = mini_table(top_reviews, [("ranking", "Rank"), ("titulo", "Livro"), ("autor", "Autor"), ("numero_avaliacoes", "Avaliações"), ("avaliacao_media", "Nota")])
    tabela_score = mini_table(top_score, [("ranking", "Rank"), ("titulo", "Livro"), ("autor", "Autor"), ("score", "Score"), ("numero_votos_lista", "Votos")]) if not top_score.empty else '<div class="empty-state">Score não disponível.</div>'

    return f"""
    <section class="tab" data-tab="livros">
      <div class="section stack">
        {chapter_header('03', 'Livros', 'Destaques da lista considerando posição, nota ponderada, popularidade e métricas próprias do ranking.')}

        <div class="feature-grid">{cards}</div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['weighted']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Melhores por nota ponderada</h3><p>Ranking que combina avaliação média e quantidade de avaliações.</p>{tabela_weighted}</div>
        </div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['reviews']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Livros mais avaliados</h3><p>Popularidade medida pelo volume de avaliações acumuladas na plataforma.</p>{tabela_reviews}</div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['score']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Score da lista</h3><p>Leitura específica da lista <i>Best Books Ever</i>, complementada pelo número de votos recebidos.</p>{tabela_score}</div>
        </div>
      </div>
    </section>
    """


def autores_html(df: pd.DataFrame, graficos: dict[str, str]) -> str:
    autores = autores_resumo(df)
    top_count = autores.sort_values(["quantidade_livros", "total_avaliacoes"], ascending=[False, False]).head(12)
    top_pop = autores.sort_values("total_avaliacoes", ascending=False).head(12)

    min_livros = 3 if int(autores["quantidade_livros"].max()) >= 3 else 1
    top_quality = (
        autores[autores["quantidade_livros"] >= min_livros]
        .sort_values(["nota_ponderada_media", "total_avaliacoes"], ascending=[False, False])
        .head(12)
    )

    table_count = mini_table(top_count, [("autor", "Autor"), ("quantidade_livros", "Livros"), ("avaliacao_media", "Nota média"), ("total_avaliacoes", "Avaliações")])
    table_quality = mini_table(top_quality, [("autor", "Autor"), ("quantidade_livros", "Livros"), ("nota_ponderada_media", "Nota pond."), ("livro_destaque", "Destaque")])
    table_pop = mini_table(top_pop, [("autor", "Autor"), ("total_avaliacoes", "Avaliações"), ("quantidade_livros", "Livros"), ("avaliacao_media", "Nota média")])

    top = top_count.iloc[0] if not top_count.empty else None
    if top is not None:
        summary = f"{escape(str(top['autor']))} é o autor mais recorrente no recorte, com {fmt_int(top['quantidade_livros'])} livros e {fmt_compacto(top['total_avaliacoes'])} avaliações acumuladas."
    else:
        summary = "Não foi possível determinar o autor mais recorrente."

    return f"""
    <section class="tab" data-tab="autores">
      <div class="section stack">
        {chapter_header('04', 'Autores', 'Presença na lista, avaliação média e alcance dos autores a partir dos livros coletados.')}

        <div class="author-callout"><span>{icon('authors')}</span><div><b>Leitura de destaque</b><p>{summary}</p></div></div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['author_count']}</div>
          <div class="split-aside"><div class="card-kicker">PRESENÇA</div><h3>Autores com mais livros</h3><p>A quantidade de títulos mostra recorrência dentro da lista coletada.</p>{table_count}</div>
        </div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['author_scatter']}</div>
          <div class="split-aside"><div class="card-kicker">QUALIDADE</div><h3>Autores com melhor nota ponderada média</h3><p>Para reduzir comparações frágeis, a tabela prioriza autores com pelo menos {min_livros} livro(s) no recorte.</p>{table_quality}</div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['author_pop']}</div>
          <div class="split-aside"><div class="card-kicker">ALCANCE</div><h3>Autores mais avaliados</h3><p>Soma do volume de avaliações dos livros associados a cada autor.</p>{table_pop}</div>
        </div>
      </div>
    </section>
    """


def exploracao_html(df: pd.DataFrame) -> str:
    autores = sorted([str(x) for x in df["autor"].dropna().unique()])
    options_autores = "".join(f'<option value="{escape(a)}">{escape(a)}</option>' for a in autores)
    return f"""
    <section class="tab" data-tab="exploracao">
      <div class="section stack">
        {chapter_header('05', 'Exploração dos Dados', 'Consulta livre dos livros coletados com filtros, ordenação, pesquisa textual e paginação.')}

        <div class="explore-toolbar">
          <div class="control wide"><label>Pesquisar livro ou autor</label><input id="search-text" type="search" placeholder="Digite um título ou autor..."></div>
          <div class="control"><label>Autor</label><select id="filter-author"><option value="">Todos</option>{options_autores}</select></div>
          <div class="control"><label>Faixa de avaliação</label><select id="filter-rating"><option value="">Todas</option><option>Até 3,5</option><option>3,5 a 4,0</option><option>4,0 a 4,25</option><option>4,25 a 4,5</option><option>Acima de 4,5</option></select></div>
          <div class="control"><label>Popularidade</label><select id="filter-pop"><option value="">Todas</option><option>Baixa</option><option>Média</option><option>Alta</option><option>Muito alta</option></select></div>
          <div class="control"><label>Ordenar por</label><select id="sort-by"><option value="ranking_asc">Ranking</option><option value="nota_ponderada_desc">Nota ponderada</option><option value="avaliacao_media_desc">Avaliação média</option><option value="numero_avaliacoes_desc">Nº de avaliações</option><option value="score_desc">Score</option></select></div>
          <button id="clear-filters" type="button">Limpar filtros</button>
        </div>

        <div class="explore-summary"><b id="explore-count">—</b><span>registros encontrados</span><div class="explore-legend">Clique no título para abrir o Goodreads quando houver URL disponível.</div></div>

        <div class="table-wrap explore-table-wrap">
          <table class="data-table explore-table">
            <thead><tr><th>Ranking</th><th>Livro</th><th>Autor</th><th>Avaliação</th><th>Nº avaliações</th><th>Nota ponderada</th><th>Score</th><th>Votos</th></tr></thead>
            <tbody id="explore-body"></tbody>
          </table>
        </div>
        <div class="pagination"><button id="prev-page" type="button">← Anterior</button><span id="page-label">—</span><button id="next-page" type="button">Próxima →</button></div>
      </div>
    </section>
    """


def apendice_html(df: pd.DataFrame, graficos: dict[str, str], params: dict[str, float]) -> str:
    dicionario = [
        ("ranking", "Posição do livro na lista Best Books Ever no momento da coleta."),
        ("book_id", "Identificador numérico do livro extraído da URL do Goodreads."),
        ("titulo", "Título do livro exibido na lista."),
        ("autor", "Nome do autor exibido pelo Goodreads."),
        ("author_id", "Identificador numérico do autor extraído da URL."),
        ("url_livro", "Endereço da página do livro."),
        ("url_autor", "Endereço da página do autor."),
        ("url_capa", "Endereço da imagem de capa disponibilizada na listagem."),
        ("avaliacao_media", "Avaliação média geral do livro no Goodreads."),
        ("numero_avaliacoes", "Quantidade de avaliações registradas para o livro."),
        ("score", "Score atribuído ao livro dentro da lista Best Books Ever."),
        ("numero_votos_lista", "Quantidade de votos recebidos pelo livro especificamente na lista."),
        ("pagina_lista", "Página da lista em que o registro foi coletado."),
        ("nota_ponderada", "Avaliação ajustada pelo volume de avaliações usando a fórmula apresentada na metodologia."),
        ("faixa_avaliacao", "Categoria derivada a partir da avaliação média."),
        ("faixa_popularidade", "Quartil de popularidade derivado do número de avaliações."),
    ]
    rows = "".join(f"<tr><td><code>{escape(nome)}</code></td><td>{escape(desc)}</td></tr>" for nome, desc in dicionario)
    comp = completude(df)
    comp_rows = "".join(
        f"<tr><td><code>{escape(str(r.variavel))}</code></td><td>{fmt_pct(r.completude)}</td><td>{fmt_int(r.ausentes)}</td></tr>"
        for r in comp.sort_values("completude").itertuples()
    )

    return f"""
    <section class="tab" data-tab="apendice">
      <div class="section stack">
        {chapter_header('06', 'Apêndice', 'Documentação das variáveis, qualidade da base e regras que ajudam a reproduzir a análise.')}

        <div class="grid g2 visual-pair">
          <article class="content-card chart-card">{graficos['completude']}</article>
          <article class="content-card appendix-note">
            <div class="card-kicker">REPRODUTIBILIDADE</div>
            <h3>Parâmetros utilizados na nota ponderada</h3>
            <div class="appendix-big"><span>Média geral (C)</span><strong>{fmt_dec(params['C'], 4)}</strong></div>
            <div class="appendix-big"><span>Percentil 75 de avaliações (m)</span><strong>{fmt_int(params['m'])}</strong></div>
            <p>Esses valores são recalculados automaticamente a partir da base sempre que o script é executado.</p>
            <div class="callout small"><b>Importante.</b> Identificadores, URLs e página de coleta são campos de suporte e não devem ser interpretados como variáveis substantivas da análise.</div>
          </article>
        </div>

        <article class="content-card">
          <div class="card-kicker">DICIONÁRIO</div><h3>Descrição das variáveis</h3>
          <div class="table-wrap"><table class="data-table text-table"><thead><tr><th>Variável</th><th>Descrição</th></tr></thead><tbody>{rows}</tbody></table></div>
        </article>

        <article class="content-card">
          <div class="card-kicker">QUALIDADE</div><h3>Completude detalhada</h3>
          <div class="table-wrap"><table class="data-table"><thead><tr><th>Variável</th><th>Completude</th><th>Ausentes</th></tr></thead><tbody>{comp_rows}</tbody></table></div>
        </article>

        <div class="closing-note"><b>Escopo do relatório.</b> Esta entrega é descritiva e exploratória. Ela resume o recorte coletado do Goodreads e não pretende representar todo o catálogo da plataforma nem estabelecer relações causais entre popularidade, avaliação e posição na lista.</div>
      </div>
    </section>
    """


# =============================================================================
# 5. CSS
# =============================================================================

CSS = r"""
:root{
  --ink:#1D2228;--ink2:#28313A;--paper:#F7F3EA;--paper2:#EFE9DE;--card:#FFFFFF;
  --terracotta:#C96F4A;--terracotta-dark:#A65336;--sage:#4F6F64;--sage-light:#88A397;
  --gold:#D9A441;--blue:#496780;--plum:#76536F;--text:#263238;--muted:#6C767D;
  --border:#E3DDD3;--soft:#FAF8F3;--white:#FFFFFF;--shadow:0 12px 36px rgba(38,50,56,.08);
  --radius:18px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
html,body{margin:0;background:var(--paper);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif;-webkit-font-smoothing:antialiased}
button,input,select{font:inherit}.app{display:flex;min-height:100vh}.main{flex:1;min-width:0}.tab{display:none}.tab.active{display:block}

.sidebar{position:sticky;top:0;height:100vh;width:238px;flex:0 0 238px;background:var(--ink);color:#fff;padding:24px 15px 18px;display:flex;flex-direction:column;z-index:30;border-right:1px solid rgba(255,255,255,.06);transition:.18s ease}
.sidebar.collapsed{width:76px;flex-basis:76px}.sidebar-brand{padding:4px 10px 24px}.brand-mark{display:inline-flex;align-items:center;gap:10px;font-weight:900;letter-spacing:.03em}.brand-mark span:first-child{width:34px;height:34px;border:1px solid rgba(255,255,255,.22);border-radius:10px;display:grid;place-items:center;font-size:18px;background:linear-gradient(135deg,rgba(201,111,74,.34),rgba(79,111,100,.22))}.brand-copy{font-size:13px;line-height:1.1}.brand-copy small{display:block;font-weight:500;color:#9EAAA6;font-size:9px;letter-spacing:.12em;margin-top:4px}.sidebar.collapsed .brand-copy{display:none}
.sidebar-toggle{border:1px solid rgba(255,255,255,.12);background:#242B31;color:#fff;height:34px;border-radius:10px;cursor:pointer;margin:0 5px 18px;display:flex;align-items:center;justify-content:center;font-size:12px}.sidebar.collapsed .sidebar-toggle{width:42px}
.nav{display:flex;flex-direction:column;gap:6px}.nav-button{border:0;background:transparent;color:#AEB8B5;border-radius:12px;display:grid;grid-template-columns:30px 1fr;align-items:center;gap:9px;padding:10px 12px;cursor:pointer;text-align:left;font-size:12px;font-weight:750;min-height:45px}.nav-button:hover{background:#252D33;color:#fff}.nav-button.active{background:#30383D;color:#fff;box-shadow:inset 3px 0 0 var(--terracotta)}.nav-icon{font-size:18px;color:#D9A98E;text-align:center}.nav-label{line-height:1.2}.sidebar.collapsed .nav-button{grid-template-columns:1fr;place-items:center;padding:10px}.sidebar.collapsed .nav-label{display:none}.sidebar-footer{margin-top:auto;padding:16px 10px 0;border-top:1px solid rgba(255,255,255,.08);font-size:10px;color:#7F8A87;line-height:1.45}.sidebar.collapsed .sidebar-footer{display:none}

.cover-page{min-height:68vh;position:relative;overflow:hidden;background:radial-gradient(circle at 82% 18%,rgba(201,111,74,.24),transparent 30%),radial-gradient(circle at 20% 90%,rgba(79,111,100,.28),transparent 38%),linear-gradient(135deg,#151A1E 0%,#20272D 55%,#151A1E 100%);color:white;display:flex;align-items:center}
.cover-noise{position:absolute;inset:0;opacity:.10;background-image:linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px);background-size:44px 44px}.cover-content{position:relative;z-index:2;max-width:1180px;width:100%;margin:0 auto;padding:74px 58px 68px}.cover-eyebrow{font-size:11px;letter-spacing:.18em;color:#D8B39E;font-weight:800}.cover-content h1{font-family:Georgia,"Times New Roman",serif;font-size:58px;line-height:1.02;letter-spacing:-.035em;margin:14px 0 10px;max-width:850px;font-weight:600}.cover-subtitle{font-size:18px;color:#CDD6D2;margin:0;max-width:760px}.cover-rule{width:86px;height:3px;background:var(--terracotta);margin:30px 0}.cover-description{max-width:790px;font-size:14px;line-height:1.7;color:#BBC6C2}.cover-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;max-width:900px;margin-top:30px}.cover-stats>div{border-top:1px solid rgba(255,255,255,.16);padding-top:13px}.cover-stats strong{display:block;font-size:25px;font-weight:800}.cover-stats span{display:block;margin-top:3px;font-size:10px;text-transform:uppercase;letter-spacing:.10em;color:#A6B2AE}.cover-meta{display:flex;gap:24px;margin-top:40px;color:#87948F;font-size:11px}
.home-navigation{max-width:1220px;margin:0 auto;padding:36px 42px 56px}.home-navigation-title{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}.home-navigation-title span{font-family:Georgia,"Times New Roman",serif;font-size:23px;color:var(--ink)}.home-navigation-title p{margin:0;color:var(--muted);font-size:12px}.home-nav-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.home-nav-card{background:rgba(255,255,255,.72);border:1px solid var(--border);border-radius:16px;padding:18px;display:grid;grid-template-columns:38px 1fr auto;gap:13px;align-items:start;text-align:left;cursor:pointer;color:var(--text);box-shadow:0 6px 22px rgba(38,50,56,.04);transition:.16s ease}.home-nav-card:hover{transform:translateY(-2px);background:#fff;border-color:#D0C5B7;box-shadow:var(--shadow)}.home-nav-icon{width:36px;height:36px;border-radius:11px;background:var(--paper2);color:var(--terracotta);display:grid;place-items:center;font-size:17px}.home-nav-card h3{margin:1px 0 4px;font-size:13px;color:var(--ink)}.home-nav-card p{margin:0;font-size:11.5px;line-height:1.45;color:var(--muted)}.home-nav-arrow{color:var(--terracotta);font-size:18px}

.section{max-width:1380px;margin:0 auto;padding:38px 44px 64px}.stack>*+*{margin-top:22px}.chapter-header{display:grid;grid-template-columns:55px 1fr;gap:14px;align-items:start;padding-bottom:18px;border-bottom:1px solid var(--border)}.chapter-number{font-family:Georgia,"Times New Roman",serif;font-size:31px;color:var(--terracotta);line-height:1}.chapter-kicker{font-size:9px;letter-spacing:.15em;color:var(--sage);font-weight:900;margin-bottom:7px}.chapter-header h1{font-family:Georgia,"Times New Roman",serif;font-weight:600;font-size:34px;letter-spacing:-.02em;color:var(--ink);margin:0}.chapter-header p{font-size:13px;color:var(--muted);line-height:1.55;margin:7px 0 0;max-width:920px}

.grid{display:grid;gap:14px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g5{grid-template-columns:repeat(5,minmax(0,1fr))}.content-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px;min-width:0}.chart-card{padding:10px 14px 8px}
.kpi{background:#fff;border:1px solid var(--border);border-radius:15px;padding:16px;position:relative;overflow:hidden;min-height:132px;box-shadow:0 6px 22px rgba(38,50,56,.04)}.kpi:before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--terracotta)}.kpi.sage:before{background:var(--sage)}.kpi.gold:before{background:var(--gold)}.kpi.blue:before{background:var(--blue)}.kpi.plum:before{background:var(--plum)}.kpi-label{text-transform:uppercase;letter-spacing:.10em;font-size:9px;color:var(--muted);font-weight:900}.kpi-value{font-family:Georgia,"Times New Roman",serif;font-size:29px;color:var(--ink);margin-top:10px}.kpi-caption{font-size:11px;color:var(--muted);line-height:1.42;margin-top:6px}

.method-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.method-step{background:#fff;border:1px solid var(--border);border-radius:15px;padding:16px;min-height:165px;position:relative}.method-step>span{font-family:Georgia,serif;color:var(--terracotta);font-size:20px}.method-step b{display:block;margin-top:15px;color:var(--ink);font-size:12px}.method-step p{font-size:11.5px;color:var(--muted);line-height:1.52;margin:6px 0 0}.method-step:not(:last-child):after{content:"→";position:absolute;right:-11px;top:48%;z-index:2;color:#BDB4A7;font-size:18px}.card-kicker{font-size:9px;letter-spacing:.14em;color:var(--terracotta);font-weight:900}.content-card h3{margin:7px 0 10px;font-family:Georgia,"Times New Roman",serif;color:var(--ink);font-size:20px;font-weight:600}.formula-card{background:linear-gradient(145deg,#fff 0%,#FCF7F0 100%)}.formula{font-family:Georgia,"Times New Roman",serif;font-size:25px;color:var(--ink);padding:16px 0}.formula span{color:var(--terracotta)}.formula small{font-size:14px}.formula-legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.formula-legend p{margin:0;padding:9px 10px;background:rgba(239,233,222,.6);border-radius:9px;font-size:11px;color:var(--muted)}.formula-legend b{color:var(--terracotta);margin-right:4px}.formula-legend strong{color:var(--ink)}.source-list{display:grid;gap:0}.source-list>div{display:flex;justify-content:space-between;gap:15px;padding:11px 0;border-bottom:1px solid var(--border);font-size:11.5px}.source-list span{color:var(--muted)}.source-list b{color:var(--ink);text-align:right}.note{font-size:11.5px;line-height:1.55;color:var(--muted);margin-top:14px}.callout{background:#F0ECE3;border:1px solid #DDD3C5;border-left:4px solid var(--sage);padding:14px 16px;border-radius:11px;font-size:12px;line-height:1.55;color:var(--muted)}.callout b{color:var(--ink)}.callout.small{font-size:11px;margin-top:14px}

.insight-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.insight-card{padding:14px;background:rgba(255,255,255,.58);border:1px solid var(--border);border-radius:13px}.insight-card>span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.10em;color:var(--sage);font-weight:900}.insight-card strong{display:block;font-family:Georgia,"Times New Roman",serif;font-size:22px;color:var(--ink);margin:8px 0 4px}.insight-card p{font-size:10.5px;line-height:1.4;color:var(--muted);margin:0}.visual-pair{align-items:stretch}.split-card{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(330px,.75fr);gap:20px;align-items:center;padding:12px 14px 12px 20px}.split-card.reverse{grid-template-columns:minmax(330px,.75fr) minmax(0,1.65fr)}.split-card.reverse .split-chart{order:2}.split-card.reverse .split-aside{order:1;border-left:0;border-right:1px solid var(--border);padding-left:0;padding-right:20px}.split-chart{min-width:0}.split-aside{border-left:1px solid var(--border);padding-left:20px}.split-aside h3{font-size:18px}.split-aside>p{font-size:11.5px;color:var(--muted);line-height:1.5;margin:0 0 12px}

.feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.feature-book{display:grid;grid-template-columns:104px 1fr;background:#fff;border:1px solid var(--border);border-radius:18px;overflow:hidden;min-height:180px;box-shadow:var(--shadow);position:relative}.feature-book:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--terracotta)}.feature-book.sage:before{background:var(--sage)}.feature-book.gold:before{background:var(--gold)}.feature-book.ink:before{background:var(--ink)}.feature-cover{background:#EBE5DA;display:flex;align-items:center;justify-content:center;min-height:180px;overflow:hidden}.feature-cover img{width:100%;height:100%;object-fit:cover}.cover-fallback{width:100%;height:100%;align-items:center;justify-content:center;flex-direction:column;color:#8D8579;gap:8px;background:linear-gradient(160deg,#EEE6DA,#DDD1C1)}.cover-fallback span{font-size:30px}.cover-fallback small{font-size:8px;letter-spacing:.12em}.feature-copy{padding:16px 17px}.feature-label{font-size:8.5px;letter-spacing:.12em;color:var(--terracotta);font-weight:900}.feature-copy h3{font-size:17px;margin:7px 0 4px;line-height:1.2}.feature-author{margin:0;color:var(--muted);font-size:11px}.feature-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:13px}.feature-metrics span{font-size:9.5px;color:var(--muted);background:var(--soft);border-radius:8px;padding:7px}.feature-metrics b{display:block;color:var(--ink);font-size:11px}.book-open{display:inline-flex;margin-top:10px;color:var(--sage);text-decoration:none;font-size:10.5px;font-weight:800}.book-open:hover{text-decoration:underline}

.author-callout{display:flex;align-items:center;gap:14px;background:linear-gradient(135deg,#E9EFEA,#F8F4EC);border:1px solid #D6DDD8;border-radius:14px;padding:15px 18px}.author-callout>span{width:38px;height:38px;border-radius:12px;background:var(--sage);color:white;display:grid;place-items:center;font-size:18px}.author-callout b{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--sage)}.author-callout p{font-size:12px;color:var(--text);margin:4px 0 0}

.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:11px;background:#fff}.data-table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px}.data-table th,.data-table td{padding:10px 11px;border-bottom:1px solid #ECE6DD;text-align:right;white-space:nowrap}.data-table th{background:#F6F2EB;color:#746E65;font-size:8.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:900;position:sticky;top:0;z-index:2}.data-table th:first-child,.data-table td:first-child{text-align:left}.data-table tbody tr:last-child td{border-bottom:0}.data-table tbody tr:hover{background:#FBF8F2}.compact-table-wrap{max-height:322px}.compact-table{font-size:10px}.compact-table th,.compact-table td{padding:8px 9px}.compact-table td:nth-child(2),.compact-table td:nth-child(3){max-width:150px;overflow:hidden;text-overflow:ellipsis}.text-table th,.text-table td{text-align:left!important;white-space:normal;vertical-align:top;line-height:1.5}.text-table td:first-child{min-width:180px}

.explore-toolbar{display:grid;grid-template-columns:1.5fr repeat(4,minmax(150px,.72fr)) auto;gap:10px;align-items:end;background:#fff;border:1px solid var(--border);border-radius:16px;padding:15px;box-shadow:var(--shadow)}.control{display:flex;flex-direction:column;gap:5px}.control label{font-size:8.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:900}.control input,.control select{width:100%;height:39px;border:1px solid #D9D1C5;background:#FCFAF6;border-radius:10px;padding:0 10px;color:var(--text);font-size:11px}.control input:focus,.control select:focus{outline:2px solid rgba(201,111,74,.16);border-color:var(--terracotta)}#clear-filters{height:39px;border:0;background:var(--ink);color:#fff;border-radius:10px;padding:0 14px;font-size:10px;font-weight:800;cursor:pointer}.explore-summary{display:flex;align-items:baseline;gap:7px}.explore-summary b{font-family:Georgia,serif;font-size:23px;color:var(--ink)}.explore-summary>span{font-size:11px;color:var(--muted)}.explore-legend{margin-left:auto;font-size:10px;color:var(--muted)}.explore-table-wrap{max-height:620px}.explore-table td:nth-child(2){text-align:left;max-width:320px;white-space:normal;min-width:230px}.explore-table td:nth-child(3){text-align:left;min-width:140px}.explore-table a{color:var(--sage);font-weight:800;text-decoration:none}.explore-table a:hover{text-decoration:underline}.pagination{display:flex;justify-content:center;align-items:center;gap:12px}.pagination button{border:1px solid var(--border);background:#fff;border-radius:10px;padding:8px 12px;color:var(--text);cursor:pointer;font-size:10.5px}.pagination button:disabled{opacity:.4;cursor:default}.pagination span{font-size:10.5px;color:var(--muted)}

.appendix-note p{font-size:11.5px;color:var(--muted);line-height:1.55}.appendix-big{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--border)}.appendix-big span{font-size:11px;color:var(--muted)}.appendix-big strong{font-family:Georgia,serif;font-size:22px;color:var(--ink)}.closing-note{background:var(--ink);color:#C5CECA;border-radius:16px;padding:18px 20px;font-size:12px;line-height:1.6}.closing-note b{color:#fff}.empty-state{min-height:250px;display:grid;place-items:center;color:var(--muted);font-size:12px;background:#FBF8F2;border:1px dashed var(--border);border-radius:12px}

@media(max-width:1250px){.g5{grid-template-columns:repeat(3,minmax(0,1fr))}.insight-strip{grid-template-columns:repeat(3,minmax(0,1fr))}.method-flow{grid-template-columns:repeat(3,minmax(0,1fr))}.method-step:after{display:none}.explore-toolbar{grid-template-columns:repeat(3,minmax(0,1fr))}.control.wide{grid-column:span 2}}
@media(max-width:1050px){.home-nav-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.g4{grid-template-columns:repeat(2,minmax(0,1fr))}.split-card,.split-card.reverse{grid-template-columns:1fr}.split-card.reverse .split-chart,.split-card.reverse .split-aside{order:initial}.split-aside,.split-card.reverse .split-aside{border-left:0;border-right:0;border-top:1px solid var(--border);padding:18px 0 0}.feature-grid{grid-template-columns:1fr}}
@media(max-width:860px){.app{display:block}.sidebar{position:relative;width:100%!important;height:auto;flex-basis:auto!important;padding:12px}.sidebar-brand{padding:3px 8px 10px}.sidebar-toggle{display:none}.nav{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}.nav-button,.sidebar.collapsed .nav-button{grid-template-columns:1fr;place-items:center;padding:8px}.nav-label,.sidebar.collapsed .nav-label{display:block;font-size:9px;text-align:center}.sidebar-footer{display:none!important}.cover-content{padding:54px 26px}.cover-content h1{font-size:43px}.section,.home-navigation{padding-left:22px;padding-right:22px}.cover-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.g2{grid-template-columns:1fr}}
@media(max-width:650px){.nav{grid-template-columns:repeat(2,minmax(0,1fr))}.home-nav-grid,.g3,.g4,.g5,.insight-strip,.method-flow,.formula-legend{grid-template-columns:1fr}.cover-content h1{font-size:36px}.chapter-header{grid-template-columns:1fr}.chapter-number{font-size:22px}.feature-book{grid-template-columns:82px 1fr}.explore-toolbar{grid-template-columns:1fr}.control.wide{grid-column:span 1}.explore-legend{display:none}.cover-meta{flex-direction:column;gap:6px}.home-navigation-title{align-items:flex-start;flex-direction:column}}
@media print{.sidebar{display:none}.tab{display:block!important}.cover-page{min-height:90vh;break-after:page}.section{max-width:none;padding:20px}.content-card,.kpi,.feature-book{box-shadow:none;break-inside:avoid}.home-navigation{display:none}}
"""


# =============================================================================
# 6. JAVASCRIPT
# =============================================================================

JS = r"""
const DATA = window.GOODREADS_DATA || [];
let currentPage = 1;
const pageSize = 25;

function formatIntBR(v){
  if(v === null || v === undefined || v === '' || Number.isNaN(Number(v))) return '—';
  return Math.round(Number(v)).toLocaleString('pt-BR');
}
function formatDecBR(v, d=2){
  if(v === null || v === undefined || v === '' || Number.isNaN(Number(v))) return '—';
  return Number(v).toLocaleString('pt-BR',{minimumFractionDigits:d,maximumFractionDigits:d});
}
function esc(s){
  return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}[c]));
}
function activateTab(name){
  document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
  document.querySelectorAll('.nav-button').forEach(el => el.classList.toggle('active', el.dataset.target === name));
  window.scrollTo({top:0,behavior:'smooth'});
  setTimeout(() => { if(window.Plotly){ document.querySelectorAll('.tab.active .plotly-graph-div').forEach(d => Plotly.Plots.resize(d)); } }, 80);
}

document.querySelectorAll('.nav-button').forEach(btn => btn.addEventListener('click', () => activateTab(btn.dataset.target)));
document.querySelectorAll('[data-open]').forEach(btn => btn.addEventListener('click', () => activateTab(btn.dataset.open)));
const sb = document.querySelector('.sidebar');
const sbToggle = document.querySelector('.sidebar-toggle');
if(sbToggle){ sbToggle.addEventListener('click', () => sb.classList.toggle('collapsed')); }

function filteredData(){
  const text = (document.getElementById('search-text')?.value || '').trim().toLowerCase();
  const author = document.getElementById('filter-author')?.value || '';
  const rating = document.getElementById('filter-rating')?.value || '';
  const pop = document.getElementById('filter-pop')?.value || '';
  const sort = document.getElementById('sort-by')?.value || 'ranking_asc';
  let rows = DATA.filter(r => {
    const hay = `${r.titulo || ''} ${r.autor || ''}`.toLowerCase();
    return (!text || hay.includes(text)) && (!author || r.autor === author) && (!rating || r.faixa_avaliacao === rating) && (!pop || r.faixa_popularidade === pop);
  });
  const [field, dir] = sort.endsWith('_asc') ? [sort.slice(0,-4),'asc'] : [sort.slice(0,-5),'desc'];
  rows.sort((a,b) => {
    const av = a[field], bv = b[field];
    if(av === null || av === undefined) return 1;
    if(bv === null || bv === undefined) return -1;
    if(typeof av === 'number' && typeof bv === 'number') return dir === 'asc' ? av-bv : bv-av;
    return dir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });
  return rows;
}
function renderExplore(){
  const rows = filteredData();
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  currentPage = Math.min(currentPage, pages);
  const start = (currentPage-1)*pageSize;
  const shown = rows.slice(start,start+pageSize);
  const tbody = document.getElementById('explore-body');
  if(!tbody) return;
  tbody.innerHTML = shown.map(r => {
    const title = r.url_livro ? `<a href="${esc(r.url_livro)}" target="_blank" rel="noopener">${esc(r.titulo)}</a>` : esc(r.titulo);
    return `<tr><td>${formatIntBR(r.ranking)}</td><td>${title}</td><td>${esc(r.autor)}</td><td>${formatDecBR(r.avaliacao_media)}</td><td>${formatIntBR(r.numero_avaliacoes)}</td><td>${formatDecBR(r.nota_ponderada,3)}</td><td>${formatIntBR(r.score)}</td><td>${formatIntBR(r.numero_votos_lista)}</td></tr>`;
  }).join('');
  document.getElementById('explore-count').textContent = formatIntBR(rows.length);
  document.getElementById('page-label').textContent = `Página ${currentPage} de ${pages}`;
  document.getElementById('prev-page').disabled = currentPage <= 1;
  document.getElementById('next-page').disabled = currentPage >= pages;
}
['search-text','filter-author','filter-rating','filter-pop','sort-by'].forEach(id => {
  const el = document.getElementById(id);
  if(el) el.addEventListener(id === 'search-text' ? 'input' : 'change', () => { currentPage=1; renderExplore(); });
});
const clearBtn = document.getElementById('clear-filters');
if(clearBtn) clearBtn.addEventListener('click', () => {
  ['search-text','filter-author','filter-rating','filter-pop'].forEach(id => { const el=document.getElementById(id); if(el) el.value=''; });
  const sort=document.getElementById('sort-by'); if(sort) sort.value='ranking_asc';
  currentPage=1; renderExplore();
});
const prev=document.getElementById('prev-page'); if(prev) prev.addEventListener('click',()=>{if(currentPage>1){currentPage--;renderExplore();}});
const next=document.getElementById('next-page'); if(next) next.addEventListener('click',()=>{const p=Math.max(1,Math.ceil(filteredData().length/pageSize));if(currentPage<p){currentPage++;renderExplore();}});
renderExplore();
"""


# =============================================================================
# 7. MONTAGEM E SAÍDA
# =============================================================================


def json_para_script(objeto: Any) -> str:
    texto = json.dumps(objeto, ensure_ascii=False, allow_nan=False)
    return texto.replace("</", "<\\/")


def preparar_dados_exploracao(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        c for c in [
            "ranking", "titulo", "autor", "avaliacao_media", "numero_avaliacoes",
            "nota_ponderada", "score", "numero_votos_lista", "faixa_avaliacao",
            "faixa_popularidade", "url_livro"
        ] if c in df.columns
    ]
    temp = df[cols].copy()
    temp = temp.replace({np.nan: None, pd.NA: None})
    registros = []
    for row in temp.to_dict(orient="records"):
        item = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                item[k] = int(v)
            elif isinstance(v, (np.floating,)):
                item[k] = float(v) if math.isfinite(float(v)) else None
            elif pd.isna(v):
                item[k] = None
            else:
                item[k] = v
        registros.append(item)
    return registros


def render_html(completa: pd.DataFrame, dashboard: pd.DataFrame, parcial: pd.DataFrame, params: dict[str, float]) -> str:
    meta = {
        "titulo": TITULO,
        "subtitulo": SUBTITULO,
        "autora": AUTORA,
        "data_processamento": datetime.now().strftime("%d/%m/%Y"),
    }

    top_weighted = top_livros(completa, "nota_ponderada", 10)
    top_reviews = top_livros(completa, "numero_avaliacoes", 10)
    top_score = top_livros(completa, "score", 10) if "score" in completa.columns else pd.DataFrame()
    autores = autores_resumo(completa)

    graficos = {
        "histograma": fig_html(grafico_histograma(completa), "fig-hist"),
        "donut": fig_html(grafico_donut_faixas(completa), "fig-donut"),
        "scatter_pop": fig_html(grafico_scatter_popularidade(completa), "fig-scatter-pop"),
        "rank_score": fig_html(grafico_rank_score(completa), "fig-rank-score"),
        "corr": fig_html(grafico_correlacao(completa), "fig-corr"),
        "weighted": fig_html(grafico_barras_livros(top_weighted, "nota_ponderada", "Top 10 por nota ponderada", "Avaliação ajustada pelo volume de avaliações"), "fig-weighted"),
        "reviews": fig_html(grafico_barras_livros(top_reviews, "numero_avaliacoes", "Livros mais avaliados", "Volume acumulado de avaliações no Goodreads", "inteiro"), "fig-reviews"),
        "score": fig_html(grafico_barras_livros(top_score, "score", "Maiores scores da lista", "Métrica específica da lista Best Books Ever", "inteiro"), "fig-score") if not top_score.empty else '<div class="empty-state">Score não disponível para esta base.</div>',
        "author_count": fig_html(grafico_autores_quantidade(autores), "fig-author-count"),
        "author_pop": fig_html(grafico_autores_popularidade(autores), "fig-author-pop"),
        "author_scatter": fig_html(grafico_autores_mapa(autores), "fig-author-scatter"),
        "completude": fig_html(grafico_completude(completa), "fig-completude"),
    }

    sidebar_items = [
        ("home", "Capa", "home"),
        ("metodologia", "Metodologia", "method"),
        ("visao", "Visão Geral", "overview"),
        ("livros", "Livros", "books"),
        ("autores", "Autores", "authors"),
        ("exploracao", "Exploração", "search"),
        ("apendice", "Apêndice", "appendix"),
    ]
    nav = "".join(
        f'<button class="nav-button{" active" if i == 0 else ""}" data-target="{tab}" type="button"><span class="nav-icon">{icon(ico)}</span><span class="nav-label">{escape(label)}</span></button>'
        for i, (tab, label, ico) in enumerate(sidebar_items)
    )

    body = "".join([
        home_html(meta, completa),
        metodologia_html(meta, completa, parcial, params),
        visao_html(completa, graficos),
        livros_html(completa, graficos),
        autores_html(completa, graficos),
        exploracao_html(dashboard),
        apendice_html(completa, graficos, params),
    ])

    dados_json = json_para_script(preparar_dados_exploracao(dashboard))
    plotly_js = get_plotlyjs()

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{escape(TITULO)} · Relatório</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-brand"><div class="brand-mark"><span>G</span><div class="brand-copy">Goodreads<small>DATA REPORT</small></div></div></div>
    <button class="sidebar-toggle" type="button" title="Recolher barra lateral">☰</button>
    <nav class="nav">{nav}</nav>
    <div class="sidebar-footer">Projeto acadêmico<br>Web scraping · Python</div>
  </aside>
  <main class="main">{body}</main>
</div>
<script>{plotly_js}</script>
<script>window.GOODREADS_DATA={dados_json};</script>
<script>{JS}</script>
</body>
</html>"""


def gerar_relatorio(
    caminho_completa: str = CAMINHO_BASE_COMPLETA_PADRAO,
    caminho_dashboard: str = CAMINHO_BASE_DASHBOARD_PADRAO,
    caminho_parcial: str = CAMINHO_BASE_PARCIAL_PADRAO,
    caminho_saida: str = CAMINHO_SAIDA_PADRAO,
    abrir: bool = False,
) -> str:
    completa, dashboard, parcial, params = preparar_bases(
        caminho_completa,
        caminho_dashboard,
        caminho_parcial,
    )
    html = render_html(completa, dashboard, parcial, params)

    destino = Path(caminho_saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding="utf-8")

    print("[ok] Base completa:", Path(caminho_completa))
    print("[ok] Livros analisados:", len(completa))
    print("[ok] Autores distintos:", completa["autor"].nunique(dropna=True))
    print("[ok] Nota ponderada · C:", round(params["C"], 4), "| m:", round(params["m"], 0))
    print("[ok] Relatório gerado:", destino)

    if abrir:
        webbrowser.open(destino.resolve().as_uri())
    return str(destino)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera a versão visual v2 do relatório standalone do Goodreads.")
    parser.add_argument("--base-completa", default=CAMINHO_BASE_COMPLETA_PADRAO)
    parser.add_argument("--base-dashboard", default=CAMINHO_BASE_DASHBOARD_PADRAO)
    parser.add_argument("--base-parcial", default=CAMINHO_BASE_PARCIAL_PADRAO)
    parser.add_argument("--saida", default=CAMINHO_SAIDA_PADRAO)
    parser.add_argument("--abrir", action="store_true")
    args = parser.parse_args()

    gerar_relatorio(
        caminho_completa=args.base_completa,
        caminho_dashboard=args.base_dashboard,
        caminho_parcial=args.base_parcial,
        caminho_saida=args.saida,
        abrir=args.abrir,
    )
