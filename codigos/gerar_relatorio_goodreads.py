from __future__ import annotations

r"""
Gera um relatório visual standalone em HTML para o projeto Goodreads.

Estrutura
---------
Capa | Metodologia | Visão Geral | Livros | Autores | Exploração | Apêndice

"""

import argparse
import base64
import json
import math
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

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
CAMINHO_SAIDA_PADRAO = r"C:\Users\bella\webscraping-goodreads\dashboard\relatorio_goodreads.html"

AUTORA = "Isabella Viana Bambirra"
TITULO = "Goodreads — Best Books Ever"
SUBTITULO = "Web scraping e análise descritiva da lista Best Books Ever"
GOODREADS_FAVICON_URL = "https://www.goodreads.com/favicon.ico"
GOODREADS_FAVICON_FALLBACK_URI = GOODREADS_FAVICON_URL


def obter_goodreads_favicon_uri() -> str:
    """Baixa o favicon oficial atual do Goodreads e o incorpora ao HTML.

    O arquivo final continua standalone porque a imagem é convertida para data URI
    no momento da geração. Se a rede não estiver disponível durante a geração, mantém como fallback
    a própria URL oficial do favicon para não interromper a construção do relatório.
    """
    try:
        requisicao = Request(
            GOODREADS_FAVICON_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(requisicao, timeout=12) as resposta:
            conteudo = resposta.read()
        if not conteudo:
            raise ValueError("favicon vazio")
        return "data:image/x-icon;base64," + base64.b64encode(conteudo).decode("ascii")
    except Exception:
        return GOODREADS_FAVICON_FALLBACK_URI


GOODREADS_FAVICON_URI = obter_goodreads_favicon_uri()

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


def _normalizar_chave_texto(serie: pd.Series) -> pd.Series:
    """Normaliza texto apenas para identificar o mesmo livro em entradas repetidas."""
    return (
        serie.astype("string")
        .fillna("")
        .str.lower()
        .str.replace(r"[^\w]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def consolidar_livros(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Consolida ocorrências repetidas do mesmo título/autor.

    A lista do Goodreads pode apresentar mais de uma entrada para a mesma obra
    (por exemplo, edições distintas). Para o relatório, cada obra aparece uma
    única vez. Quando há repetição, é mantida a entrada com a melhor posição
    numérica no ranking; em caso de empate ou ranking ausente, usamos score,
    votos da lista e número de avaliações como critérios auxiliares.
    """
    if df.empty:
        return df.copy(), 0

    out = df.copy()
    n_inicial = len(out)

    titulo = _normalizar_chave_texto(out["titulo"])
    autor = _normalizar_chave_texto(out["autor"])
    out["_chave_obra"] = titulo + "||" + autor

    if "book_id" in out.columns:
        ids = out["book_id"].astype("string").fillna("")
        mascara_sem_chave = titulo.eq("") | autor.eq("")
        out.loc[mascara_sem_chave, "_chave_obra"] = "__book_id__" + ids.loc[mascara_sem_chave]

    def serie_ordem(nome: str, padrao: float) -> pd.Series:
        if nome not in out.columns:
            return pd.Series(padrao, index=out.index, dtype="float64")
        return pd.to_numeric(out[nome], errors="coerce").fillna(padrao)

    out["_ord_ranking"] = serie_ordem("ranking", np.inf)
    out["_ord_score"] = serie_ordem("score", -np.inf)
    out["_ord_votos"] = serie_ordem("numero_votos_lista", -np.inf)
    out["_ord_reviews"] = serie_ordem("numero_avaliacoes", -np.inf)

    out = out.sort_values(
        ["_ord_ranking", "_ord_score", "_ord_votos", "_ord_reviews"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    out = out.drop_duplicates(subset="_chave_obra", keep="first")

    if "book_id" in out.columns:
        out = out.drop_duplicates(subset="book_id", keep="first")

    auxiliares = ["_chave_obra", "_ord_ranking", "_ord_score", "_ord_votos", "_ord_reviews"]
    out = out.drop(columns=[c for c in auxiliares if c in out.columns])
    out = out.sort_values("ranking", na_position="last") if "ranking" in out.columns else out
    out = out.reset_index(drop=True)
    return out, n_inicial - len(out)


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

    n_completa_bruta = len(completa)
    completa, duplicatas_consolidadas = consolidar_livros(completa)
    completa, C, m = calcular_nota_ponderada(completa)
    completa = criar_faixas(completa)

    if dashboard.empty:
        dashboard = completa.copy()
    else:
        dashboard, _ = consolidar_livros(dashboard)
        dashboard, _, _ = calcular_nota_ponderada(dashboard)
        dashboard = criar_faixas(dashboard)

        dashboard["_join_titulo"] = _normalizar_chave_texto(dashboard["titulo"])
        dashboard["_join_autor"] = _normalizar_chave_texto(dashboard["autor"])
        completa_aux = completa.copy()
        completa_aux["_join_titulo"] = _normalizar_chave_texto(completa_aux["titulo"])
        completa_aux["_join_autor"] = _normalizar_chave_texto(completa_aux["autor"])
        colunas_recuperar = [
            c for c in [
                "ranking", "book_id", "url_livro", "url_capa", "avaliacao_media",
                "numero_avaliacoes", "nota_ponderada", "score", "numero_votos_lista",
                "faixa_avaliacao", "faixa_popularidade"
            ]
            if c in completa_aux.columns
        ]
        base_lookup = completa_aux[
            ["_join_titulo", "_join_autor"] + colunas_recuperar
        ].drop_duplicates(["_join_titulo", "_join_autor"])

        dashboard = dashboard.merge(
            base_lookup,
            on=["_join_titulo", "_join_autor"],
            how="left",
            suffixes=("", "_base"),
        )
        for coluna in colunas_recuperar:
            coluna_base = f"{coluna}_base"
            if coluna_base in dashboard.columns:
                if coluna in dashboard.columns:
                    dashboard[coluna] = dashboard[coluna].combine_first(dashboard[coluna_base])
                else:
                    dashboard[coluna] = dashboard[coluna_base]
                dashboard = dashboard.drop(columns=coluna_base)
        dashboard = dashboard.drop(columns=["_join_titulo", "_join_autor"])

    params = {
        "C": C,
        "m": m,
        "registros_brutos": float(n_completa_bruta),
        "livros_unicos": float(len(completa)),
        "duplicatas_consolidadas": float(duplicatas_consolidadas),
    }
    return completa, dashboard, parcial, params


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


def estatisticas_avaliacao(df: pd.DataFrame) -> dict[str, float | None]:
    serie = pd.to_numeric(df.get("avaliacao_media"), errors="coerce").dropna()
    if serie.empty:
        return {"min": None, "q1": None, "mediana": None, "q3": None, "max": None}
    return {
        "min": safe_float(serie.min()),
        "q1": safe_float(serie.quantile(0.25)),
        "mediana": safe_float(serie.median()),
        "q3": safe_float(serie.quantile(0.75)),
        "max": safe_float(serie.max()),
    }


def concentracao_autores(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    autores = autores_resumo(df).copy()
    total_livros = max(int(len(df)), 1)
    total_autores = max(int(len(autores)), 1)

    top10_livros = int(
        autores.sort_values(["quantidade_livros", "total_avaliacoes"], ascending=[False, False])
        .head(10)["quantidade_livros"]
        .sum()
    )
    autores_um_livro = int((autores["quantidade_livros"] == 1).sum())

    q = pd.to_numeric(autores["quantidade_livros"], errors="coerce").fillna(0)
    categorias = pd.Series(
        np.select(
            [q == 1, q == 2, q.between(3, 5), q.between(6, 10), q > 10],
            ["1 livro", "2 livros", "3 a 5 livros", "6 a 10 livros", "Mais de 10 livros"],
            default="Sem informação",
        ),
        index=autores.index,
        dtype="string",
    )
    ordem = ["1 livro", "2 livros", "3 a 5 livros", "6 a 10 livros", "Mais de 10 livros", "Sem informação"]
    dist = categorias.value_counts().reindex(ordem, fill_value=0).rename_axis("faixa").reset_index(name="autores")
    dist = dist[dist["autores"] > 0].copy()
    dist["participacao_autores"] = dist["autores"] / total_autores

    resumo = {
        "top10_share_livros": top10_livros / total_livros,
        "top10_livros": top10_livros,
        "autores_um_livro": autores_um_livro,
        "share_autores_um_livro": autores_um_livro / total_autores,
    }
    return resumo, dist


def insights_gerais(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    resumo = resumo_geral(df)
    top_author = autores_resumo(df).sort_values(
        ["quantidade_livros", "total_avaliacoes"], ascending=[False, False]
    ).head(1)

    if not top_author.empty:
        nome_autor = str(top_author.iloc[0]["autor"])
        n_autor = int(top_author.iloc[0]["quantidade_livros"])
        texto_autor = f"Aparece em {n_autor} livros do recorte analisado."
    else:
        nome_autor = "—"
        texto_autor = "Não foi possível identificar o autor mais recorrente."

    return [
        (
            "Concentração de boas avaliações",
            fmt_pct(resumo["pct_4_mais"]),
            "dos livros possuem avaliação média igual ou superior a 4,0.",
        ),
        (
            "Popularidade típica",
            fmt_compacto(resumo["mediana_avaliacoes"]),
            "é a mediana do número de avaliações recebidas por livro.",
        ),
        (
            "Autor mais recorrente",
            nome_autor,
            texto_autor,
        ),
    ]

def plot_layout(titulo: str, subtitulo: str = "", height: int = 360) -> dict[str, Any]:
    return {
        "height": height,
        "margin": {"l": 38, "r": 20, "t": 78, "b": 48},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Roboto, Segoe UI, Arial, sans-serif", "color": CORES["text"], "size": 12},
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
    fig.update_layout(**plot_layout("Como se distribuem as avaliações dos livros?", "Histograma das avaliações médias; as linhas indicam média e mediana", 390))
    if len(serie):
        xmin = max(0.0, float(serie.min()) - 0.08)
        xmax = min(5.0, float(serie.max()) + 0.08)
        fig.update_xaxes(title="Avaliação média", range=[xmin, xmax], dtick=0.2)
    else:
        fig.update_xaxes(title="Avaliação média")
    fig.update_yaxes(title="Quantidade de livros", rangemode="tozero")
    return estilizar_eixos(fig)


def grafico_donut_faixas(df: pd.DataFrame) -> go.Figure:
    bands = rating_bands(df).copy()
    total = float(bands["livros"].sum())
    bands["participacao"] = bands["livros"] / total if total else 0
    bands["pct_fmt"] = bands["participacao"].map(lambda v: fmt_pct(v, 1))
    bands["livros_fmt"] = bands["livros"].map(fmt_int)

    customdata = np.stack([bands["livros_fmt"], bands["pct_fmt"]], axis=-1) if len(bands) else None

    # O domínio da rosca é definido explicitamente para que o centro geométrico
    # possa ser reutilizado nas anotações do total e do rótulo.
    domain_x = [0.00, 0.58]
    domain_y = [0.05, 0.95]
    center_x = (domain_x[0] + domain_x[1]) / 2
    center_y = (domain_y[0] + domain_y[1]) / 2

    fig = go.Figure(go.Pie(
        labels=bands["faixa"],
        values=bands["livros"],
        hole=0.62,
        sort=False,
        domain={"x": domain_x, "y": domain_y},
        marker={"colors": CHART_COLORS[: len(bands)]},
        text=bands["pct_fmt"],
        textinfo="text",
        textfont={"size": 11},
        customdata=customdata,
        hovertemplate="%{label}<br>%{customdata[0]} livros · %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(**plot_layout(
        "Em quais faixas de avaliação os livros se concentram?",
        "Participação dos livros em cada intervalo de avaliação média",
        390,
    ))
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "v",
            "y": center_y,
            "yanchor": "middle",
            "x": 0.66,
            "xanchor": "left",
            "font": {"size": 11, "color": CORES["text"]},
            "itemsizing": "constant",
            "traceorder": "normal",
        },
        annotations=[
            {
                "text": f"<b>{fmt_int(len(df))}</b>",
                "xref": "paper", "yref": "paper",
                "x": center_x, "y": center_y + 0.035,
                "xanchor": "center", "yanchor": "middle",
                "showarrow": False,
                "font": {"size": 21, "color": CORES["ink"]},
            },
            {
                "text": "livros",
                "xref": "paper", "yref": "paper",
                "x": center_x, "y": center_y - 0.055,
                "xanchor": "center", "yanchor": "middle",
                "showarrow": False,
                "font": {"size": 10, "color": CORES["text"]},
            },
        ],
        margin={"l": 20, "r": 150, "t": 72, "b": 30},
    )
    return fig


def grafico_boxplot_avaliacao(df: pd.DataFrame) -> go.Figure:
    serie = pd.to_numeric(df.get("avaliacao_media"), errors="coerce").dropna()
    fig = go.Figure()
    if not serie.empty:
        fig.add_trace(go.Box(
            x=serie,
            orientation="h",
            boxpoints="outliers",
            marker={"color": CORES["terracotta"], "size": 5, "opacity": 0.65},
            line={"color": CORES["sage"], "width": 2},
            fillcolor="rgba(201,111,74,0.16)",
            hovertemplate="Avaliação média: %{x:.2f}<extra></extra>",
            name="Avaliação média",
        ))
    fig.update_layout(**plot_layout(
        "Quão concentradas estão as avaliações dos livros?",
        "Boxplot da avaliação média; a caixa representa o intervalo entre Q1 e Q3 e a linha interna indica a mediana",
        315,
    ))
    if len(serie):
        xmin = max(0.0, float(serie.min()) - 0.08)
        xmax = min(5.0, float(serie.max()) + 0.08)
        fig.update_xaxes(title="Avaliação média", range=[xmin, xmax], dtick=0.2)
    else:
        fig.update_xaxes(title="Avaliação média")
    fig.update_yaxes(showticklabels=False, title="")
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


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
    fig.update_layout(**plot_layout("Livros mais populares também são mais bem avaliados?", "Cada ponto representa um livro; o eixo de popularidade usa escala logarítmica", 430))
    positivos = pd.to_numeric(base["numero_avaliacoes"], errors="coerce").dropna()
    positivos = positivos[positivos > 0]
    if len(positivos):
        log_min = math.floor(math.log10(float(positivos.min())))
        log_max = math.ceil(math.log10(float(positivos.max())))
        fig.update_xaxes(type="log", title="Número de avaliações · escala log", range=[log_min, log_max])
    else:
        fig.update_xaxes(type="log", title="Número de avaliações · escala log")
    y = pd.to_numeric(base["avaliacao_media"], errors="coerce").dropna()
    if len(y):
        fig.update_yaxes(title="Avaliação média", range=[max(0, float(y.min()) - 0.08), min(5, float(y.max()) + 0.08)])
    else:
        fig.update_yaxes(title="Avaliação média")
    return estilizar_eixos(fig)


def grafico_rank_avaliacao(df: pd.DataFrame) -> go.Figure:
    if "ranking" not in df.columns or "avaliacao_media" not in df.columns:
        return go.Figure()
    cols = [c for c in ["ranking", "avaliacao_media", "titulo", "autor", "numero_avaliacoes"] if c in df.columns]
    base = df[cols].dropna(subset=["ranking", "avaliacao_media"]).sort_values("ranking")
    if base.empty:
        return go.Figure()

    custom_cols = [
        base["titulo"].astype(str),
        base["autor"].astype(str),
        pd.to_numeric(base.get("numero_avaliacoes"), errors="coerce").fillna(0),
    ]
    fig = go.Figure(go.Scatter(
        x=base["ranking"],
        y=base["avaliacao_media"],
        mode="markers",
        marker={
            "color": CORES["sage"],
            "size": 7,
            "opacity": 0.55,
            "line": {"color": "white", "width": 0.5},
        },
        customdata=np.stack(custom_cols, axis=-1),
        hovertemplate=(
            "Ranking #%{x}<br><b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            "Avaliação média: %{y:.2f}<br>Avaliações: %{customdata[2]:,.0f}<extra></extra>"
        ),
    ))
    fig.update_layout(**plot_layout(
        "Melhores posições no ranking têm avaliações maiores?",
        "Compara a posição na Best Books Ever com a avaliação média geral de cada livro",
        390,
    ))
    xmax = float(pd.to_numeric(base["ranking"], errors="coerce").max()) if len(base) else 1
    y = pd.to_numeric(base["avaliacao_media"], errors="coerce").dropna()
    fig.update_xaxes(title="Posição na lista", range=[0, xmax * 1.02])
    if len(y):
        fig.update_yaxes(title="Avaliação média", range=[max(0, float(y.min()) - 0.08), min(5, float(y.max()) + 0.08)])
    else:
        fig.update_yaxes(title="Avaliação média")
    return estilizar_eixos(fig, xgrid=False, ygrid=True)

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
    valores = pd.to_numeric(base[coluna], errors="coerce")
    labels = base["titulo"].astype(str).copy()

    repetidos = labels.duplicated(keep=False)
    labels.loc[repetidos] = labels.loc[repetidos] + " — " + base.loc[repetidos, "autor"].astype(str)

    custom = np.stack([
        base["autor"].astype(str),
        pd.to_numeric(base["avaliacao_media"], errors="coerce").fillna(np.nan),
        pd.to_numeric(base["numero_avaliacoes"], errors="coerce").fillna(0),
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
    fig.update_layout(**plot_layout(titulo, subtitulo, 385))
    fig.update_yaxes(automargin=True, title="")

    validos = valores.dropna()
    if len(validos):
        minimo = float(validos.min())
        maximo = float(validos.max())

        if coluna in {"nota_ponderada", "avaliacao_media"}:
            amplitude = max(maximo - minimo, 0.08)
            margem = max(amplitude * 0.35, 0.04)
            inicio = max(0.0, minimo - margem)
            fim = min(5.0, maximo + margem)
            if fim - inicio < 0.18:
                centro = (fim + inicio) / 2
                inicio = max(0.0, centro - 0.09)
                fim = min(5.0, centro + 0.09)
            fig.update_xaxes(range=[inicio, fim], tickformat=".2f", title="")
        else:
            fim = maximo * 1.18 if maximo > 0 else 1
            fig.update_xaxes(range=[0, fim], title="")

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
    fig.update_layout(**plot_layout("Quais autores aparecem mais vezes na lista?", "Quantidade de obras únicas de cada autor no recorte consolidado", 405))
    maximo = float(base["quantidade_livros"].max()) if len(base) else 1
    fig.update_xaxes(range=[0, maximo * 1.18])
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
    fig.update_layout(**plot_layout("Quais autores concentram o maior alcance?", "Soma do número de avaliações das obras únicas de cada autor", 405))
    maximo = float(base["total_avaliacoes"].max()) if len(base) else 1
    fig.update_xaxes(range=[0, maximo * 1.18])
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
    fig.update_layout(**plot_layout("Autores mais recorrentes também têm avaliações maiores?", "Cada ponto representa um autor; tamanho e cor acompanham o alcance acumulado", 450))
    max_livros = float(pd.to_numeric(base["quantidade_livros"], errors="coerce").max()) if len(base) else 1
    fig.update_xaxes(title="Quantidade de livros na lista", range=[0, max_livros * 1.08 + 0.5])
    y = pd.to_numeric(base["avaliacao_media"], errors="coerce").dropna()
    if len(y):
        fig.update_yaxes(title="Avaliação média", range=[max(0, float(y.min()) - 0.08), min(5, float(y.max()) + 0.08)])
    else:
        fig.update_yaxes(title="Avaliação média")
    return estilizar_eixos(fig)


def grafico_concentracao_autores(df: pd.DataFrame) -> go.Figure:
    _, dist = concentracao_autores(df)
    base = dist.iloc[::-1].copy()
    fig = go.Figure(go.Bar(
        x=base["autores"],
        y=base["faixa"],
        orientation="h",
        marker={"color": CORES["sage"]},
        text=[fmt_int(v) for v in base["autores"]],
        textposition="outside",
        customdata=np.stack([base["participacao_autores"].map(lambda v: fmt_pct(v, 1))], axis=-1),
        hovertemplate="%{y}<br>Autores: %{x}<br>Participação: %{customdata[0]}<extra></extra>",
    ))
    fig.update_layout(**plot_layout(
        "A presença na lista está concentrada em poucos autores?",
        "Número de autores segundo a quantidade de livros que possuem no recorte",
        360,
    ))
    maximo = float(base["autores"].max()) if len(base) else 1
    fig.update_xaxes(title="Quantidade de autores", range=[0, maximo * 1.16])
    fig.update_yaxes(title="")
    return estilizar_eixos(fig, xgrid=True, ygrid=False)


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
    paths = {
        "home": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M9 21v-7h6v7"/>',
        "method": '<path d="M9 3h6"/><path d="M10 3v6.5L5.5 18a2 2 0 0 0 1.8 3h9.4a2 2 0 0 0 1.8-3L14 9.5V3"/><path d="M8 15h8"/>',
        "overview": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20V7"/>',
        "books": '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v16H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H13v16h4.5A2.5 2.5 0 0 1 20 21.5z"/>',
        "authors": '<circle cx="9" cy="8" r="3"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0"/><circle cx="17" cy="9" r="2.5"/><path d="M15.5 14.5A5 5 0 0 1 21 20"/>',
        "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
        "appendix": '<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v5h5"/><path d="M9 13h6M9 17h6"/>',
        "arrow": '<path d="M5 12h14"/><path d="m14 7 5 5-5 5"/>',
        "star": '<path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
        "code": '<path d="m8 9-4 3 4 3"/><path d="m16 9 4 3-4 3"/><path d="m14 5-4 14"/>',
        "filter": '<path d="M4 5h16l-6 7v6l-4 2v-8z"/>',
        "sort": '<path d="M8 6h11M8 12h8M8 18h5"/><path d="m3 8 2-2 2 2M5 6v12"/>',
    }
    body = paths.get(nome, '<circle cx="12" cy="12" r="3"/>')
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" '
        'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )

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
        ("Visão Geral", "Panorama da base, distribuição das notas e relação entre posição, avaliação e popularidade.", "visao", "overview"),
        ("Livros", "Rankings de destaque por nota ponderada, popularidade e score da lista.", "livros", "books"),
        ("Autores", "Presença, avaliação média e alcance acumulado dos autores.", "autores", "authors"),
        ("Exploração", "Tabela pesquisável para consultar individualmente os livros coletados.", "exploracao", "search"),
        ("Apêndice", "Definições das métricas, variáveis e regras de ordenação.", "apendice", "appendix"),
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
          <div class="cover-g" aria-label="Goodreads"><img src="{GOODREADS_FAVICON_URI}" alt="Ícone do Goodreads"></div>
          <div class="cover-eyebrow">PROJETO DE WEBSCRAPING · PYTHON</div>
          <h1>{escape(meta['titulo'])}</h1>
          <p class="cover-subtitle">{escape(meta['subtitulo'])}</p>
          <div class="cover-rule"></div>
          <p class="cover-description">Relatório visual e interativo construído a partir dos livros coletados na lista <i>Best Books Ever</i>, com foco em avaliação, popularidade, posição no ranking e presença de autores.</p>
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
    duplicados = int(params.get("duplicatas_consolidadas", 0))
    paginas = int(pd.to_numeric(df.get("pagina_lista"), errors="coerce").nunique()) if "pagina_lista" in df.columns else None
    parcial_n = len(parcial) if not parcial.empty else None

    exemplo_html = escape(
        '<tr itemtype="http://schema.org/Book">\n'
        '  <td class="number">1</td>\n'
        '  <td>\n'
        '    <a class="bookTitle" href="/book/show/..."><span>Nome do livro</span></a>\n'
        '    <a class="authorName" href="/author/show/...">Nome do autor</a>\n'
        '    <span class="minirating">4.25 avg rating — 1,234,567 ratings</span>\n'
        '    <img class="bookCover" src="https://.../capa.jpg">\n'
        '    ... score: 1,234,567 ... 18,432 people voted ...\n'
        '  </td>\n'
        '</tr>'
    )

    seletores = [
        ("Bloco de cada livro", 'tr[itemtype="http://schema.org/Book"]', "Delimita cada registro da tabela da lista."),
        ("Ranking", "td.number", "Lê a posição do livro na lista."),
        ("Título e URL", "a.bookTitle", "Obtém o texto do título e o atributo href."),
        ("Autor e URL", "a.authorName", "Obtém o nome do autor e o link correspondente."),
        ("Capa", "img.bookCover", "Lê o atributo src da imagem."),
        ("Avaliação e nº de avaliações", "span.minirating", "Separa a avaliação média e a quantidade de avaliações com expressão regular."),
        ("Score e votos", "texto completo do bloco", "Localiza os padrões 'score:' e 'people voted' no texto do registro."),
    ]
    selector_rows = "".join(
        f"<tr><td>{escape(campo)}</td><td><code>{escape(seletor)}</code></td><td>{escape(desc)}</td></tr>"
        for campo, seletor, desc in seletores
    )

    return f"""
    <section class="tab" data-tab="metodologia">
      <div class="section stack">
        {chapter_header('01', 'Metodologia', 'Da requisição das páginas à preparação das bases utilizadas no relatório.')}

        <div class="method-flow">
          <div class="method-step"><span>01</span><div><b>Requisição</b><p>As páginas da lista são acessadas com <code>requests</code>, cabeçalho de navegador, timeout, tentativas e pausas entre requisições.</p></div></div>
          <div class="method-step"><span>02</span><div><b>Leitura do HTML</b><p>O conteúdo é interpretado com <code>BeautifulSoup</code> e cada livro é localizado dentro da tabela da lista.</p></div></div>
          <div class="method-step"><span>03</span><div><b>Extração</b><p>Seletores CSS e expressões regulares recuperam título, autor, ranking, avaliação, volume de avaliações, score, votos e URLs.</p></div></div>
          <div class="method-step"><span>04</span><div><b>Tratamento</b><p>São ajustados tipos numéricos, duplicidades e variáveis auxiliares. A nota ponderada equilibra avaliação e volume.</p></div></div>
          <div class="method-step"><span>05</span><div><b>Entrega</b><p>As bases finais alimentam este relatório standalone, gerado em um único arquivo HTML com gráficos Plotly incorporados.</p></div></div>
        </div>

        <div class="grid g4">
          {kpi_card(fmt_int(params.get('registros_brutos')), 'Registros coletados', 'Quantidade de linhas obtidas antes da consolidação de obras repetidas.', 'terracotta')}
          {kpi_card(fmt_int(len(df)), 'Livros únicos', 'Quantidade de obras após consolidar repetições do mesmo título e autor.', 'sage')}
          {kpi_card(fmt_int(duplicados), 'Entradas consolidadas', 'Registros repetidos removidos mantendo a melhor posição no ranking.', 'gold')}
          {kpi_card(fmt_int(n_colunas), 'Variáveis', 'Número de colunas presentes na base completa usada no relatório.', 'blue')}
        </div>

        <article class="content-card formula-card method-wide">
          <div class="card-kicker">NOTA PONDERADA</div>
          <h3>Como a comparação entre livros é estabilizada</h3>
          <p class="section-description">A avaliação média do Goodreads é útil, mas não considera o volume de avaliações. A nota ponderada aproxima livros com pouca evidência da média geral e permite comparar títulos com bases de avaliações muito diferentes de forma mais conservadora.</p>
          <div class="formula">WR = <span>v</span>/<small>v + m</small> · R &nbsp;+&nbsp; <span>m</span>/<small>v + m</small> · C</div>
          <div class="formula-legend">
            <p><b>R</b> avaliação média do livro</p>
            <p><b>v</b> número de avaliações do livro</p>
            <p><b>C</b> média geral da base = <strong>{fmt_dec(params['C'], 3)}</strong></p>
            <p><b>m</b> percentil 75 do nº de avaliações = <strong>{fmt_int(params['m'])}</strong></p>
          </div>
        </article>

        <article class="content-card source-card method-wide">
          <div class="card-kicker">RECORTE DA COLETA</div>
          <h3>O que efetivamente entra na análise</h3>
          <p class="section-description">O recorte parte dos registros recuperados diretamente das páginas da lista <i>Best Books Ever</i>. Como a lista pode trazer mais de uma entrada para a mesma obra — por exemplo, edições distintas — o relatório consolida registros com o mesmo título e autor. Nesses casos, é mantida a entrada com a melhor posição numérica no ranking, preservando uma única linha por obra para evitar duplicidade nos gráficos, rankings e na exploração.</p>
          <div class="source-list">
            <div><span>Fonte</span><b>Goodreads · Best Books Ever</b></div>
            <div><span>Páginas identificadas</span><b>{fmt_int(paginas) if paginas is not None else '—'}</b></div>
            <div><span>Registros no arquivo parcial</span><b>{fmt_int(parcial_n) if parcial_n is not None else '—'}</b></div>
            <div><span>Entradas repetidas consolidadas</span><b>{fmt_int(duplicados)}</b></div>
            <div><span>Unidade de análise</span><b>Obra única por título e autor</b></div>
          </div>
          <p class="note">Gêneros e detalhes das páginas individuais não foram incorporados porque a coleta desses campos não apresentou estabilidade suficiente.</p>
        </article>

        <article class="content-card method-wide">
          <div class="card-kicker">HTML DA PÁGINA</div>
          <h3>Exemplo simplificado da estrutura usada na coleta</h3>
          <p class="section-description">O scraping não depende da aparência visual da página, mas da estrutura dos elementos no HTML. O trecho abaixo representa os elementos relevantes observados em cada linha de livro.</p>
          <div class="code-example-grid">
            <div class="code-panel"><div class="code-panel-title">{icon('code')} Estrutura HTML simplificada</div><pre><code>{exemplo_html}</code></pre></div>
            <div class="selector-panel"><div class="code-panel-title">{icon('filter')} Como as informações são extraídas</div><div class="table-wrap"><table class="data-table selector-table"><thead><tr><th>Informação</th><th>Seletor / origem</th><th>Leitura</th></tr></thead><tbody>{selector_rows}</tbody></table></div></div>
          </div>
        </article>

        <div class="callout"><b>Leitura correta.</b> O relatório descreve os livros presentes no recorte coletado da lista. O <b>score</b> e os <b>votos da lista</b> pertencem à dinâmica da própria lista do Goodreads. Já a <b>avaliação média</b> e o <b>número de avaliações</b> são métricas gerais exibidas para cada livro.</div>
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
    stats = estatisticas_avaliacao(df)
    bands = rating_bands(df)
    bands["participacao"] = bands["livros"] / bands["livros"].sum() if bands["livros"].sum() else 0
    band_rows = "".join(
        f"<tr><td>{escape(str(r.faixa))}</td><td>{fmt_int(r.livros)}</td><td>{fmt_pct(r.participacao)}</td></tr>"
        for r in bands.itertuples()
    )
    band_table = f'<div class="table-wrap compact-table-wrap"><table class="data-table compact-table"><thead><tr><th>Faixa</th><th>Livros</th><th>Participação</th></tr></thead><tbody>{band_rows}</tbody></table></div>'

    stats_html = "".join([
        f'<div class="stat-mini"><span>Mínimo</span><strong>{fmt_dec(stats["min"])}</strong><small>menor avaliação média observada</small></div>',
        f'<div class="stat-mini"><span>Q1</span><strong>{fmt_dec(stats["q1"])}</strong><small>25% dos livros estão até este valor</small></div>',
        f'<div class="stat-mini"><span>Mediana</span><strong>{fmt_dec(stats["mediana"])}</strong><small>divide a base em duas metades</small></div>',
        f'<div class="stat-mini"><span>Q3</span><strong>{fmt_dec(stats["q3"])}</strong><small>75% dos livros estão até este valor</small></div>',
        f'<div class="stat-mini"><span>Máximo</span><strong>{fmt_dec(stats["max"])}</strong><small>maior avaliação média observada</small></div>',
    ])

    return f"""
    <section class="tab" data-tab="visao">
      <div class="section stack">
        {chapter_header('02', 'Visão Geral', 'Uma leitura sintética da distribuição das avaliações, da popularidade e da posição dos livros na lista.')}

        <div class="grid g4">
          {kpi_card(fmt_int(resumo['livros']), 'Livros', 'Quantidade de obras únicas analisadas.', 'terracotta')}
          {kpi_card(fmt_int(resumo['autores']), 'Autores', 'Quantidade de autores distintos presentes no recorte.', 'sage')}
          {kpi_card(fmt_dec(resumo['avaliacao_media']), 'Avaliação média', 'Média aritmética das avaliações médias dos livros.', 'gold')}
          {kpi_card(fmt_compacto(resumo['total_avaliacoes']), 'Avaliações', 'Soma do número de avaliações recebidas pelos livros.', 'blue')}
        </div>

        <div class="insight-strip">{insight_cards_html(insights_gerais(df))}</div>

        <div class="grid g2 visual-pair">
          <article class="content-card chart-card">{graficos['histograma']}<p class="chart-caption"><b>O que responde:</b> em quais valores as avaliações dos livros se concentram e quanto elas se dispersam. As linhas de média e mediana ajudam a perceber se a distribuição é aproximadamente equilibrada ou puxada por valores extremos.</p></article>
          <article class="content-card chart-card">{graficos['donut']}<p class="chart-caption"><b>O que responde:</b> quais intervalos concentram a maior parcela dos livros. Os percentuais são calculados sobre o total de obras únicas analisadas.</p></article>
        </div>

        <div class="content-card stats-context-card">
          <div class="stats-context-chart">{graficos['boxplot']}</div>
          <div class="stats-context-copy"><div class="card-kicker">RESUMO ESTATÍSTICO</div><h3>Onde está o centro e a dispersão das avaliações?</h3><p>O boxplot complementa o histograma ao destacar quartis, mediana e possíveis valores extremos. Os cinco indicadores abaixo permitem interpretar a distribuição sem depender apenas da média.</p><div class="stat-mini-grid">{stats_html}</div></div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['scatter_pop']}</div>
          <div class="split-aside"><div class="card-kicker">POPULARIDADE × AVALIAÇÃO</div><h3>Livros muito populares precisam ter notas maiores?</h3><p>Cada ponto representa um livro. O eixo horizontal usa escala logarítmica porque o número de avaliações varia de milhares a milhões; o eixo vertical mostra a avaliação média. A leitura permite localizar livros muito populares e bem avaliados, além de títulos com notas altas, mas menor alcance.</p><div class="table-title">Distribuição por faixa de avaliação</div>{band_table}</div>
        </div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['rank_rating']}</div>
          <div class="split-aside"><div class="card-kicker">POSIÇÃO NA LISTA</div><h3>Estar no topo da lista implica ter avaliação maior?</h3><p>O gráfico compara a posição ocupada na <i>Best Books Ever</i> com a avaliação média geral do livro. Ele permite verificar visualmente se os primeiros colocados também se concentram nas maiores notas ou se as duas medidas contam histórias diferentes.</p><div class="callout small"><b>Importante.</b> Ranking e avaliação média são medidas distintas: o primeiro descreve a posição dentro desta lista; a segunda resume as notas dadas ao livro no Goodreads.</div></div>
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
        {chapter_header('03', 'Livros', 'Destaques do recorte considerando posição na lista, avaliação ajustada, popularidade e score específico da Best Books Ever.')}

        <div class="feature-grid">{cards}</div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['weighted']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Melhores por nota ponderada</h3><p>Ordena os livros pela nota calculada neste projeto, que combina a avaliação média com o volume de avaliações. O objetivo é evitar que uma nota muito alta baseada em pouco volume seja comparada diretamente a títulos avaliados milhões de vezes.</p>{tabela_weighted}</div>
        </div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['reviews']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Livros mais avaliados</h3><p>Mostra os títulos com maior número de avaliações registradas no Goodreads. Essa medida é usada como indicador de alcance e popularidade, e não como medida de qualidade.</p>{tabela_reviews}</div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['score']}</div>
          <div class="split-aside"><div class="card-kicker">TOP 10</div><h3>Maiores scores da lista</h3><p>Apresenta a métrica de score exibida especificamente na lista <i>Best Books Ever</i>. Ela não está na escala de 0 a 5 e não deve ser confundida com a avaliação média do livro.</p>{tabela_score}</div>
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

    conc, dist = concentracao_autores(df)
    dist_rows = "".join(
        f"<tr><td>{escape(str(r.faixa))}</td><td>{fmt_int(r.autores)}</td><td>{fmt_pct(r.participacao_autores)}</td></tr>"
        for r in dist.itertuples()
    )
    dist_table = f'<div class="table-wrap compact-table-wrap"><table class="data-table compact-table"><thead><tr><th>Livros por autor</th><th>Autores</th><th>Participação</th></tr></thead><tbody>{dist_rows}</tbody></table></div>'

    return f"""
    <section class="tab" data-tab="autores">
      <div class="section stack">
        {chapter_header('04', 'Autores', 'Presença na lista, avaliação dos títulos e alcance acumulado dos autores presentes no recorte.')}

        <div class="author-callout"><span>{icon('authors')}</span><div><b>Leitura de destaque</b><p>{summary}</p></div></div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['author_concentration']}</div>
          <div class="split-aside"><div class="card-kicker">CONCENTRAÇÃO</div><h3>Como a recorrência dos autores se distribui?</h3><p>Os 10 autores com mais obras concentram <b>{fmt_pct(conc['top10_share_livros'])}</b> dos livros analisados. Ao mesmo tempo, <b>{fmt_pct(conc['share_autores_um_livro'])}</b> dos autores aparecem com apenas um livro. O gráfico e a tabela mostram como a recorrência se distribui entre todos os autores.</p>{dist_table}</div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['author_count']}</div>
          <div class="split-aside"><div class="card-kicker">PRESENÇA</div><h3>Detalhamento dos autores mais recorrentes</h3><p>O gráfico conta quantos títulos de cada autor aparecem no recorte. A tabela complementa essa presença com a avaliação média e o volume total de avaliações dos livros do autor.</p>{table_count}</div>
        </div>

        <div class="content-card split-card reverse">
          <div class="split-chart">{graficos['author_scatter']}</div>
          <div class="split-aside"><div class="card-kicker">PRESENÇA × AVALIAÇÃO × ALCANCE</div><h3>Leitura conjunta de recorrência, nota e alcance</h3><p>Cada ponto representa um autor. O eixo horizontal mostra quantos livros aparecem na lista; o vertical, a avaliação média desses títulos. O tamanho e a cor dos pontos acompanham o volume acumulado de avaliações, permitindo observar simultaneamente recorrência, qualidade média e alcance.</p>{table_quality}</div>
        </div>

        <div class="content-card split-card">
          <div class="split-chart">{graficos['author_pop']}</div>
          <div class="split-aside"><div class="card-kicker">ALCANCE</div><h3>Detalhamento do alcance acumulado por autor</h3><p>Soma o número de avaliações de todos os livros de cada autor presentes no recorte. Um valor alto pode refletir vários títulos populares ou poucos livros com alcance muito elevado.</p>{table_pop}</div>
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

        <div class="explore-summary"><b id="explore-count">—</b><span>registros encontrados</span><div class="explore-legend">Clique no título para abrir o Goodreads quando houver URL disponível.<span class="explore-color-legend"><i class="low"></i>menor <i class="mid"></i>intermediário <i class="high"></i>maior</span></div></div>

        <div class="table-wrap explore-table-wrap">
          <table class="data-table explore-table">
            <colgroup>
              <col class="col-ranking">
              <col class="col-livro">
              <col class="col-autor">
              <col class="col-avaliacao">
              <col class="col-n-avaliacoes">
              <col class="col-nota-ponderada">
              <col class="col-score">
              <col class="col-votos">
            </colgroup>
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
        ("avaliacao_media", "Avaliação média geral do livro exibida pelo Goodreads, na escala utilizada pela plataforma."),
        ("numero_avaliacoes", "Quantidade de avaliações registradas para o livro no Goodreads."),
        ("score", "Score exibido pelo Goodreads especificamente para o livro dentro da lista Best Books Ever. É uma métrica própria da lista e não uma nota de 0 a 5."),
        ("numero_votos_lista", "Quantidade de pessoas que votaram no livro especificamente na lista Best Books Ever."),
        ("pagina_lista", "Página da lista em que o registro foi coletado."),
        ("nota_ponderada", "Indicador calculado neste projeto a partir da avaliação média e do número de avaliações, usando a fórmula apresentada na metodologia."),
        ("faixa_avaliacao", "Categoria derivada da avaliação média para facilitar filtros e leitura da distribuição."),
        ("faixa_popularidade", "Quartil de popularidade derivado do número de avaliações."),
    ]
    rows = "".join(f"<tr><td><code>{escape(nome)}</code></td><td>{escape(desc)}</td></tr>" for nome, desc in dicionario)

    order_rules = [
        ("Ranking", "Crescente", "Exibe primeiro as menores posições numéricas. Assim, #1 aparece antes de #2, #3 e assim por diante."),
        ("Nota ponderada", "Decrescente", "Prioriza os maiores valores da nota calculada no projeto, equilibrando avaliação média e volume de avaliações."),
        ("Avaliação média", "Decrescente", "Prioriza os livros com maior avaliação média exibida pelo Goodreads, sem considerar diretamente o volume de avaliações."),
        ("Nº de avaliações", "Decrescente", "Prioriza os livros com maior quantidade de avaliações, sendo uma leitura de popularidade/alcance."),
        ("Score", "Decrescente", "Prioriza os maiores scores da lista Best Books Ever. A regra só usa a métrica específica da lista."),
    ]
    order_rows = "".join(
        f"<tr><td>{escape(nome)}</td><td>{escape(sentido)}</td><td>{escape(regra)}</td></tr>"
        for nome, sentido, regra in order_rules
    )

    return f"""
    <section class="tab" data-tab="apendice">
      <div class="section stack">
        {chapter_header('06', 'Apêndice', 'Definições das métricas, documentação das variáveis e regras utilizadas na exploração dos dados.')}

        <div class="grid g3 metric-definition-grid">
          <article class="content-card metric-definition"><div class="metric-icon">{icon('star')}</div><div class="card-kicker">GOODREADS</div><h3>Avaliação média</h3><p>É a média das notas atribuídas ao livro pelos usuários do Goodreads. Está na escala de avaliação da plataforma e é coletada diretamente da página. Um valor alto indica melhor avaliação média, mas não informa quantas pessoas avaliaram o livro.</p></article>
          <article class="content-card metric-definition"><div class="metric-icon">{icon('overview')}</div><div class="card-kicker">CALCULADA NO PROJETO</div><h3>Nota ponderada</h3><p>É uma medida derivada que combina a avaliação média com o número de avaliações. Livros com pouco volume são aproximados da média geral da base; livros com muito volume preservam mais de sua própria avaliação. Continua na mesma escala da avaliação média.</p></article>
          <article class="content-card metric-definition"><div class="metric-icon">{icon('books')}</div><div class="card-kicker">LISTA BEST BOOKS EVER</div><h3>Score</h3><p>É o valor exibido pelo Goodreads para o livro dentro da lista <i>Best Books Ever</i>. É uma métrica própria da lista, relacionada à votação nela, e não deve ser interpretada como avaliação de 0 a 5. O script apenas coleta o valor apresentado pelo site.</p></article>
        </div>

        <article class="content-card appendix-note">
          <div class="card-kicker">REPRODUTIBILIDADE</div>
          <h3>Parâmetros utilizados na nota ponderada</h3>
          <div class="grid g2 appendix-params">
            <div class="appendix-big"><span>Média geral (C)</span><strong>{fmt_dec(params['C'], 4)}</strong></div>
            <div class="appendix-big"><span>Percentil 75 de avaliações (m)</span><strong>{fmt_int(params['m'])}</strong></div>
          </div>
          <p>Esses valores são recalculados automaticamente a partir da base sempre que o script é executado.</p>
        </article>

        <article class="content-card">
          <div class="card-kicker">ORDENAÇÃO NA ABA EXPLORAÇÃO</div><h3>Regras das opções de ordenação</h3>
          <p class="section-description">Cada opção aplica uma regra diferente. Isso é importante porque ranking, avaliação, popularidade e score medem aspectos distintos do livro.</p>
          <div class="table-wrap"><table class="data-table text-table"><thead><tr><th>Opção</th><th>Sentido</th><th>Regra de leitura</th></tr></thead><tbody>{order_rows}</tbody></table></div>
        </article>

        <article class="content-card">
          <div class="card-kicker">CONSOLIDAÇÃO DE OBRAS</div><h3>Como o relatório trata entradas repetidas</h3>
          <p class="section-description">Uma mesma obra pode aparecer mais de uma vez na lista por diferenças de edição ou por registros distintos no Goodreads. Para evitar que o mesmo título seja contado duas vezes, o relatório cria uma chave normalizada a partir de <b>título + autor</b>. Quando há repetição, mantém a linha com a melhor posição numérica no ranking; score, votos da lista e número de avaliações funcionam apenas como critérios auxiliares em caso de empate ou posição ausente.</p>
          <div class="callout small"><b>Efeito prático.</b> Rankings, gráficos, estatísticas por autor e a aba Exploração passam a trabalhar com uma única observação por obra.</div>
        </article>

        <article class="content-card">
          <div class="card-kicker">DICIONÁRIO</div><h3>Descrição das variáveis</h3>
          <div class="table-wrap"><table class="data-table text-table"><thead><tr><th>Variável</th><th>Descrição</th></tr></thead><tbody>{rows}</tbody></table></div>
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
html,body{margin:0;background:var(--paper);color:var(--text);font-family:Roboto,"Segoe UI",Arial,sans-serif;-webkit-font-smoothing:antialiased}
button,input,select{font:inherit}.app{display:flex;min-height:100vh}.main{flex:1;min-width:0}.tab{display:none}.tab.active{display:block}

.sidebar{position:sticky;top:0;height:100vh;width:238px;flex:0 0 238px;background:var(--ink);color:#fff;padding:24px 15px 18px;display:flex;flex-direction:column;z-index:30;border-right:1px solid rgba(255,255,255,.06);transition:.18s ease}
.sidebar.collapsed{width:76px;flex-basis:76px}.sidebar-brand{padding:4px 10px 24px}.brand-mark{display:inline-flex;align-items:center;gap:10px;font-weight:900;letter-spacing:.03em}.brand-mark .brand-favicon{width:32px;height:32px;border:1px solid rgba(255,255,255,.18);border-radius:8px;display:grid;place-items:center;overflow:hidden;background:#F4EFE5}.brand-mark .brand-favicon img{width:100%;height:100%;object-fit:cover;image-rendering:auto}.brand-copy{font-size:13px;line-height:1.1}.brand-copy small{display:block;font-weight:500;color:#9EAAA6;font-size:9px;letter-spacing:.12em;margin-top:4px}.sidebar.collapsed .brand-copy{display:none}
.sidebar-toggle{border:1px solid rgba(255,255,255,.12);background:#242B31;color:#fff;height:34px;border-radius:10px;cursor:pointer;margin:0 5px 18px;display:flex;align-items:center;justify-content:center;font-size:12px}.sidebar.collapsed .sidebar-toggle{width:42px}
.nav{display:flex;flex-direction:column;gap:6px}.nav-button{border:0;background:transparent;color:#AEB8B5;border-radius:12px;display:grid;grid-template-columns:30px 1fr;align-items:center;gap:9px;padding:10px 12px;cursor:pointer;text-align:left;font-size:12px;font-weight:750;min-height:45px}.nav-button:hover{background:#252D33;color:#fff}.nav-button.active{background:#30383D;color:#fff;box-shadow:inset 3px 0 0 var(--terracotta)}.nav-icon{font-size:18px;color:#D9A98E;text-align:center}.nav-label{line-height:1.2}.sidebar.collapsed .nav-button{grid-template-columns:1fr;place-items:center;padding:10px}.sidebar.collapsed .nav-label{display:none}.sidebar-footer{margin-top:auto;padding:16px 10px 0;border-top:1px solid rgba(255,255,255,.08);font-size:10px;color:#7F8A87;line-height:1.45}.sidebar.collapsed .sidebar-footer{display:none}

.cover-page{min-height:68vh;position:relative;overflow:hidden;background:radial-gradient(circle at 82% 18%,rgba(201,111,74,.24),transparent 30%),radial-gradient(circle at 20% 90%,rgba(79,111,100,.28),transparent 38%),linear-gradient(135deg,#151A1E 0%,#20272D 55%,#151A1E 100%);color:white;display:flex;align-items:center}
.cover-noise{position:absolute;inset:0;opacity:.10;background-image:linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px);background-size:44px 44px}.cover-content{position:relative;z-index:2;max-width:1180px;width:100%;margin:0 auto;padding:74px 58px 68px}.cover-eyebrow{font-size:11px;letter-spacing:.18em;color:#D8B39E;font-weight:800}.cover-content h1{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:58px;line-height:1.02;letter-spacing:-.035em;margin:14px 0 10px;max-width:850px;font-weight:600}.cover-subtitle{font-size:18px;color:#CDD6D2;margin:0;max-width:760px}.cover-rule{width:86px;height:3px;background:var(--terracotta);margin:30px 0}.cover-description{max-width:790px;font-size:14px;line-height:1.7;color:#BBC6C2}.cover-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;max-width:900px;margin-top:30px}.cover-stats>div{border-top:1px solid rgba(255,255,255,.16);padding-top:13px}.cover-stats strong{display:block;font-size:25px;font-weight:800}.cover-stats span{display:block;margin-top:3px;font-size:10px;text-transform:uppercase;letter-spacing:.10em;color:#A6B2AE}.cover-meta{display:flex;gap:24px;margin-top:40px;color:#87948F;font-size:11px}
.tab[data-tab="home"]{min-height:100vh;background:linear-gradient(180deg,#151A1E 0%,#1D2429 58%,#171C20 100%);color:#fff}.home-navigation{max-width:1220px;margin:0 auto;padding:36px 42px 56px;background:transparent}.home-navigation-title{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}.home-navigation-title span{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:23px;color:#F6F3ED}.home-navigation-title p{margin:0;color:#9EAAA6;font-size:12px}.home-nav-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.home-nav-card{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.11);border-radius:16px;padding:18px;display:grid;grid-template-columns:38px 1fr auto;gap:13px;align-items:start;text-align:left;cursor:pointer;color:#E8EEEB;box-shadow:0 8px 26px rgba(0,0,0,.12);transition:.16s ease}.home-nav-card:hover{transform:translateY(-2px);background:rgba(255,255,255,.09);border-color:rgba(216,179,158,.35);box-shadow:0 12px 32px rgba(0,0,0,.18)}.home-nav-icon{width:36px;height:36px;border-radius:11px;background:rgba(201,111,74,.15);color:#D9A98E;display:grid;place-items:center;font-size:17px}.home-nav-card h3{margin:1px 0 4px;font-size:13px;color:#FFFFFF}.home-nav-card p{margin:0;font-size:11.5px;line-height:1.45;color:#AEB8B5}.home-nav-arrow{color:#D9A98E;font-size:18px}

.section{max-width:1380px;margin:0 auto;padding:38px 44px 64px}.stack>*+*{margin-top:22px}.chapter-header{display:grid;grid-template-columns:55px 1fr;gap:14px;align-items:start;padding-bottom:18px;border-bottom:1px solid var(--border)}.chapter-number{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:31px;color:var(--terracotta);line-height:1}.chapter-kicker{font-size:9px;letter-spacing:.15em;color:var(--sage);font-weight:900;margin-bottom:7px}.chapter-header h1{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-weight:600;font-size:34px;letter-spacing:-.02em;color:var(--ink);margin:0}.chapter-header p{font-size:13px;color:var(--muted);line-height:1.55;margin:7px 0 0;max-width:920px}

.grid{display:grid;gap:14px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g5{grid-template-columns:repeat(5,minmax(0,1fr))}.content-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px;min-width:0}.chart-card{padding:10px 14px 8px}
.kpi{background:#fff;border:1px solid var(--border);border-radius:15px;padding:16px;position:relative;overflow:hidden;min-height:132px;box-shadow:0 6px 22px rgba(38,50,56,.04)}.kpi:before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--terracotta)}.kpi.sage:before{background:var(--sage)}.kpi.gold:before{background:var(--gold)}.kpi.blue:before{background:var(--blue)}.kpi.plum:before{background:var(--plum)}.kpi-label{text-transform:uppercase;letter-spacing:.10em;font-size:9px;color:var(--muted);font-weight:900}.kpi-value{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:29px;color:var(--ink);margin-top:10px}.kpi-caption{font-size:11px;color:var(--muted);line-height:1.42;margin-top:6px}

.method-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.method-step{background:#fff;border:1px solid var(--border);border-radius:15px;padding:16px;min-height:165px;position:relative}.method-step>span{font-family:Roboto,"Segoe UI",Arial,sans-serif;color:var(--terracotta);font-size:20px}.method-step b{display:block;margin-top:15px;color:var(--ink);font-size:12px}.method-step p{font-size:11.5px;color:var(--muted);line-height:1.52;margin:6px 0 0}.method-step:not(:last-child):after{content:"→";position:absolute;right:-11px;top:48%;z-index:2;color:#BDB4A7;font-size:18px}.card-kicker{font-size:9px;letter-spacing:.14em;color:var(--terracotta);font-weight:900}.content-card h3{margin:7px 0 10px;font-family:Roboto,"Segoe UI",Arial,sans-serif;color:var(--ink);font-size:20px;font-weight:600}.formula-card{background:linear-gradient(145deg,#fff 0%,#FCF7F0 100%)}.formula{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:25px;color:var(--ink);padding:16px 0}.formula span{color:var(--terracotta)}.formula small{font-size:14px}.formula-legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.formula-legend p{margin:0;padding:9px 10px;background:rgba(239,233,222,.6);border-radius:9px;font-size:11px;color:var(--muted)}.formula-legend b{color:var(--terracotta);margin-right:4px}.formula-legend strong{color:var(--ink)}.source-list{display:grid;gap:0}.source-list>div{display:flex;justify-content:space-between;gap:15px;padding:11px 0;border-bottom:1px solid var(--border);font-size:11.5px}.source-list span{color:var(--muted)}.source-list b{color:var(--ink);text-align:right}.note{font-size:11.5px;line-height:1.55;color:var(--muted);margin-top:14px}.callout{background:#F0ECE3;border:1px solid #DDD3C5;border-left:4px solid var(--sage);padding:14px 16px;border-radius:11px;font-size:12px;line-height:1.55;color:var(--muted)}.callout b{color:var(--ink)}.callout.small{font-size:11px;margin-top:14px}

.insight-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.insight-card{padding:14px;background:rgba(255,255,255,.58);border:1px solid var(--border);border-radius:13px}.insight-card>span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.10em;color:var(--sage);font-weight:900}.insight-card strong{display:block;font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:22px;color:var(--ink);margin:8px 0 4px}.insight-card p{font-size:10.5px;line-height:1.4;color:var(--muted);margin:0}.visual-pair{align-items:stretch}.split-card{display:grid;grid-template-columns:minmax(0,1.18fr) minmax(430px,1fr);gap:22px;align-items:center;padding:14px 16px 14px 20px}.split-card.reverse{grid-template-columns:minmax(430px,1fr) minmax(0,1.18fr)}.split-card.reverse .split-chart{order:2}.split-card.reverse .split-aside{order:1;border-left:0;border-right:1px solid var(--border);padding-left:0;padding-right:20px}.split-chart{min-width:0}.split-aside{border-left:1px solid var(--border);padding-left:20px}.split-aside h3{font-size:18px}.split-aside>p{font-size:11.5px;color:var(--muted);line-height:1.5;margin:0 0 12px}

.feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.feature-book{display:grid;grid-template-columns:104px 1fr;background:#fff;border:1px solid var(--border);border-radius:18px;overflow:hidden;min-height:180px;box-shadow:var(--shadow);position:relative}.feature-book:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--terracotta)}.feature-book.sage:before{background:var(--sage)}.feature-book.gold:before{background:var(--gold)}.feature-book.ink:before{background:var(--ink)}.feature-cover{background:#EBE5DA;display:flex;align-items:center;justify-content:center;min-height:180px;overflow:hidden}.feature-cover img{width:100%;height:100%;object-fit:cover}.cover-fallback{width:100%;height:100%;align-items:center;justify-content:center;flex-direction:column;color:#8D8579;gap:8px;background:linear-gradient(160deg,#EEE6DA,#DDD1C1)}.cover-fallback span{font-size:30px}.cover-fallback small{font-size:8px;letter-spacing:.12em}.feature-copy{padding:16px 17px}.feature-label{font-size:8.5px;letter-spacing:.12em;color:var(--terracotta);font-weight:900}.feature-copy h3{font-size:17px;margin:7px 0 4px;line-height:1.2}.feature-author{margin:0;color:var(--muted);font-size:11px}.feature-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:13px}.feature-metrics span{font-size:9.5px;color:var(--muted);background:var(--soft);border-radius:8px;padding:7px}.feature-metrics b{display:block;color:var(--ink);font-size:11px}.book-open{display:inline-flex;margin-top:10px;color:var(--sage);text-decoration:none;font-size:10.5px;font-weight:800}.book-open:hover{text-decoration:underline}

.author-callout{display:flex;align-items:center;gap:14px;background:linear-gradient(135deg,#E9EFEA,#F8F4EC);border:1px solid #D6DDD8;border-radius:14px;padding:15px 18px}.author-callout>span{width:38px;height:38px;border-radius:12px;background:var(--sage);color:white;display:grid;place-items:center;font-size:18px}.author-callout b{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--sage)}.author-callout p{font-size:12px;color:var(--text);margin:4px 0 0}

.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:11px;background:#fff}.data-table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px}.data-table th,.data-table td{padding:10px 11px;border-bottom:1px solid #ECE6DD;text-align:right;white-space:nowrap}.data-table th{background:#F6F2EB;color:#746E65;font-size:8.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:900;position:sticky;top:0;z-index:2}.data-table th:first-child,.data-table td:first-child{text-align:left}.data-table tbody tr:last-child td{border-bottom:0}.data-table tbody tr:hover{background:#FBF8F2}.compact-table-wrap{max-height:360px}.compact-table{font-size:10.5px;min-width:620px}.compact-table th,.compact-table td{padding:9px 10px}.compact-table td:nth-child(2),.compact-table td:nth-child(3){white-space:normal;min-width:150px;max-width:240px;line-height:1.35}.compact-table td:first-child{min-width:58px}.text-table th,.text-table td{text-align:left!important;white-space:normal;vertical-align:top;line-height:1.5}.text-table td:first-child{min-width:180px}

.explore-toolbar{display:grid;grid-template-columns:1.5fr repeat(4,minmax(150px,.72fr)) auto;gap:10px;align-items:end;background:#fff;border:1px solid var(--border);border-radius:16px;padding:15px;box-shadow:var(--shadow)}.control{display:flex;flex-direction:column;gap:5px}.control label{font-size:8.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:900}.control input,.control select{width:100%;height:39px;border:1px solid #D9D1C5;background:#FCFAF6;border-radius:10px;padding:0 10px;color:var(--text);font-size:11px}.control input:focus,.control select:focus{outline:2px solid rgba(201,111,74,.16);border-color:var(--terracotta)}#clear-filters{height:39px;border:0;background:var(--ink);color:#fff;border-radius:10px;padding:0 14px;font-size:10px;font-weight:800;cursor:pointer}.explore-summary{display:flex;align-items:baseline;gap:7px}.explore-summary b{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:23px;color:var(--ink)}.explore-summary>span{font-size:11px;color:var(--muted)}.explore-legend{margin-left:auto;font-size:10px;color:var(--muted)}.explore-table-wrap{max-height:620px}.explore-table{table-layout:fixed;min-width:1180px}.explore-table .col-ranking{width:78px}.explore-table .col-livro{width:33%}.explore-table .col-autor{width:18%}.explore-table .col-avaliacao{width:105px}.explore-table .col-n-avaliacoes{width:130px}.explore-table .col-nota-ponderada{width:135px}.explore-table .col-score{width:110px}.explore-table .col-votos{width:95px}.explore-table th,.explore-table td{vertical-align:middle}.explore-table th:nth-child(1),.explore-table td:nth-child(1){text-align:center}.explore-table th:nth-child(2),.explore-table td:nth-child(2){text-align:left;white-space:normal;line-height:1.35}.explore-table th:nth-child(3),.explore-table td:nth-child(3){text-align:left;white-space:normal;line-height:1.35}.explore-table th:nth-child(n+4),.explore-table td:nth-child(n+4){text-align:center}.explore-table td.metric-gradient-cell{font-weight:850;border-left:1px solid rgba(255,255,255,.60);border-right:1px solid rgba(255,255,255,.60);transition:filter .12s ease}.explore-table tbody tr:hover td.metric-gradient-cell{filter:saturate(1.08) brightness(.98)}.explore-table a{color:var(--sage);font-weight:800;text-decoration:none}.explore-table a:hover{text-decoration:underline}.explore-color-legend{margin-left:14px;display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:9.5px;white-space:nowrap}.explore-color-legend i{display:inline-block;width:12px;height:8px;border-radius:999px}.explore-color-legend .low{background:#EFA19B}.explore-color-legend .mid{background:#F2DA83}.explore-color-legend .high{background:#86C7A1}.pagination{display:flex;justify-content:center;align-items:center;gap:12px}.pagination button{border:1px solid var(--border);background:#fff;border-radius:10px;padding:8px 12px;color:var(--text);cursor:pointer;font-size:10.5px}.pagination button:disabled{opacity:.4;cursor:default}.pagination span{font-size:10.5px;color:var(--muted)}

.appendix-note p{font-size:11.5px;color:var(--muted);line-height:1.55}.appendix-big{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--border)}.appendix-big span{font-size:11px;color:var(--muted)}.appendix-big strong{font-family:Roboto,"Segoe UI",Arial,sans-serif;font-size:22px;color:var(--ink)}.closing-note{background:var(--ink);color:#C5CECA;border-radius:16px;padding:18px 20px;font-size:12px;line-height:1.6}.closing-note b{color:#fff}.empty-state{min-height:250px;display:grid;place-items:center;color:var(--muted);font-size:12px;background:#FBF8F2;border:1px dashed var(--border);border-radius:12px}

@media(max-width:1250px){.g5{grid-template-columns:repeat(3,minmax(0,1fr))}.insight-strip{grid-template-columns:repeat(3,minmax(0,1fr))}.method-flow{grid-template-columns:repeat(3,minmax(0,1fr))}.method-step:after{display:none}.explore-toolbar{grid-template-columns:repeat(3,minmax(0,1fr))}.control.wide{grid-column:span 2}}

.cover-g{width:48px;height:48px;border:1px solid rgba(255,255,255,.22);border-radius:13px;display:grid;place-items:center;margin-bottom:26px;background:#F4EFE5;box-shadow:0 12px 30px rgba(0,0,0,.18);overflow:hidden}.cover-g img{width:100%;height:100%;object-fit:cover;image-rendering:auto}
.nav-icon svg{width:20px;height:20px;display:block}.home-nav-icon svg{width:19px;height:19px}.home-nav-arrow svg{width:17px;height:17px}.book-open svg{width:14px;height:14px;vertical-align:-2px}.author-callout>span svg{width:22px;height:22px}.code-panel-title svg,.metric-icon svg{width:20px;height:20px;display:block}
.method-wide{width:100%}.section-description{font-size:12px;line-height:1.6;color:var(--muted);margin:0 0 15px}.code-example-grid{display:grid;grid-template-columns:minmax(0,.95fr) minmax(0,1.35fr);gap:18px;align-items:stretch}.code-panel,.selector-panel{min-width:0}.code-panel-title{display:flex;align-items:center;gap:8px;font-size:10px;text-transform:uppercase;letter-spacing:.09em;font-weight:900;color:var(--sage);margin-bottom:10px}.code-panel pre{margin:0;background:#20262B;color:#E7ECEA;border-radius:12px;padding:18px;overflow:auto;min-height:100%;font-family:Consolas,"Courier New",monospace;font-size:11px;line-height:1.55}.code-panel code{font-family:inherit}.selector-table td:nth-child(2){font-family:Consolas,"Courier New",monospace}.table-title{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--sage);font-weight:900;margin:16px 0 8px}.chart-caption{font-size:11.5px;line-height:1.55;color:var(--muted);margin:0 10px 10px;border-top:1px solid var(--border);padding-top:12px}.chart-caption b{color:var(--ink)}
.metric-definition-grid{align-items:stretch}.metric-definition{position:relative;padding-top:54px}.metric-icon{position:absolute;top:17px;right:17px;width:34px;height:34px;border-radius:10px;background:var(--paper2);display:grid;place-items:center;color:var(--terracotta)}.metric-definition h3{margin-top:7px}.metric-definition p{font-size:11.5px;line-height:1.58;color:var(--muted);margin-bottom:0}.appendix-params .appendix-big{background:var(--soft);border:1px solid var(--border);border-radius:12px;padding:14px 16px}.plotly-graph-div{width:100%!important;min-height:320px}

@media(max-width:1050px){.home-nav-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.g4{grid-template-columns:repeat(2,minmax(0,1fr))}.split-card,.split-card.reverse{grid-template-columns:1fr}.split-card.reverse .split-chart,.split-card.reverse .split-aside{order:initial}.split-aside,.split-card.reverse .split-aside{border-left:0;border-right:0;border-top:1px solid var(--border);padding:18px 0 0}.feature-grid{grid-template-columns:1fr}}
@media(max-width:860px){.app{display:block}.sidebar{position:relative;width:100%!important;height:auto;flex-basis:auto!important;padding:12px}.sidebar-brand{padding:3px 8px 10px}.sidebar-toggle{display:none}.nav{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}.nav-button,.sidebar.collapsed .nav-button{grid-template-columns:1fr;place-items:center;padding:8px}.nav-label,.sidebar.collapsed .nav-label{display:block;font-size:9px;text-align:center}.sidebar-footer{display:none!important}.cover-content{padding:54px 26px}.cover-content h1{font-size:43px}.section,.home-navigation{padding-left:22px;padding-right:22px}.cover-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.g2{grid-template-columns:1fr}}
@media(max-width:650px){.nav{grid-template-columns:repeat(2,minmax(0,1fr))}.home-nav-grid,.g3,.g4,.g5,.insight-strip,.method-flow,.formula-legend{grid-template-columns:1fr}.cover-content h1{font-size:36px}.chapter-header{grid-template-columns:1fr}.chapter-number{font-size:22px}.feature-book{grid-template-columns:82px 1fr}.explore-toolbar{grid-template-columns:1fr}.control.wide{grid-column:span 1}.explore-legend{display:none}.cover-meta{flex-direction:column;gap:6px}.home-navigation-title{align-items:flex-start;flex-direction:column}}
@media print{.sidebar{display:none}.tab{display:block!important}.cover-page{min-height:90vh;break-after:page}.section{max-width:none;padding:20px}.content-card,.kpi,.feature-book{box-shadow:none;break-inside:avoid}.home-navigation{display:none}}

/* Contexto estatístico da avaliação */
.stats-context-card{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(430px,.85fr);gap:26px;align-items:center}
.stats-context-chart{min-width:0}
.stats-context-copy{border-left:1px solid var(--border);padding-left:24px}
.stats-context-copy h3{font-size:19px;margin:6px 0 8px;color:var(--ink)}
.stats-context-copy>p{font-size:12.5px;line-height:1.6;color:var(--muted);margin:0 0 14px}
.stat-mini-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}
.stat-mini{background:var(--paper2);border:1px solid var(--border);border-radius:11px;padding:11px 10px;min-width:0}
.stat-mini span{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.09em;font-weight:900;color:var(--muted)}
.stat-mini strong{display:block;font-size:19px;color:var(--ink);margin-top:5px;font-variant-numeric:tabular-nums}
.stat-mini small{display:block;font-size:9.5px;line-height:1.35;color:var(--muted);margin-top:4px}
@media(max-width:1100px){.stats-context-card{grid-template-columns:1fr}.stats-context-copy{border-left:0;border-top:1px solid var(--border);padding-left:0;padding-top:18px}.stat-mini-grid{grid-template-columns:repeat(5,minmax(100px,1fr));overflow-x:auto}}
@media(max-width:720px){.stat-mini-grid{grid-template-columns:repeat(2,minmax(0,1fr));overflow:visible}}
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
function metricBounds(field){
  const values = DATA.map(r => Number(r[field])).filter(Number.isFinite);
  if(!values.length) return {min:0,max:1};
  const min = Math.min(...values);
  const max = Math.max(...values);
  return {min,max: max > min ? max : min + 1};
}
const METRIC_BOUNDS = {
  avaliacao_media: metricBounds('avaliacao_media'),
  nota_ponderada: metricBounds('nota_ponderada')
};
function hexToRgb(hex){
  const h = hex.replace('#','');
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];
}
function mixColor(a,b,t){
  const ca=hexToRgb(a), cb=hexToRgb(b);
  const rgb=ca.map((v,i)=>Math.round(v+(cb[i]-v)*t));
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}
function redYellowGreenStyle(v, field){
  const n = Number(v);
  if(!Number.isFinite(n)) return '';
  const bounds = METRIC_BOUNDS[field] || {min:0,max:1};
  const t = Math.max(0, Math.min(1, (n - bounds.min) / (bounds.max - bounds.min)));
  const red = '#EFA19B';
  const yellow = '#F2DA83';
  const green = '#86C7A1';
  const background = t <= 0.5
    ? mixColor(red, yellow, t * 2)
    : mixColor(yellow, green, (t - 0.5) * 2);
  const text = t < 0.12 || t > 0.88 ? '#17302A' : '#24323A';
  return `background:${background};color:${text};`;
}
function resizeActivePlots(){
  if(!window.Plotly) return;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll('.tab.active .plotly-graph-div').forEach(d => {
      try { Plotly.Plots.resize(d); } catch(e) { console.warn('Falha ao redimensionar gráfico', e); }
    });
  }));
}
function activateTab(name){
  document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el.dataset.tab === name));
  document.querySelectorAll('.nav-button').forEach(el => el.classList.toggle('active', el.dataset.target === name));
  window.scrollTo({top:0,behavior:'smooth'});
  setTimeout(resizeActivePlots, 60);
  setTimeout(resizeActivePlots, 240);
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
    const ratingStyle = redYellowGreenStyle(r.avaliacao_media, 'avaliacao_media');
    const weightedStyle = redYellowGreenStyle(r.nota_ponderada, 'nota_ponderada');
    return `<tr><td>${formatIntBR(r.ranking)}</td><td>${title}</td><td>${esc(r.autor)}</td><td class="metric-gradient-cell" style="${ratingStyle}">${formatDecBR(r.avaliacao_media)}</td><td>${formatIntBR(r.numero_avaliacoes)}</td><td class="metric-gradient-cell" style="${weightedStyle}">${formatDecBR(r.nota_ponderada,3)}</td><td>${formatIntBR(r.score)}</td><td>${formatIntBR(r.numero_votos_lista)}</td></tr>`;
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
window.addEventListener('resize', resizeActivePlots);
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
        "rank_rating": fig_html(grafico_rank_avaliacao(completa), "fig-rank-rating"),
        "boxplot": fig_html(grafico_boxplot_avaliacao(completa), "fig-boxplot-rating"),
        "weighted": fig_html(grafico_barras_livros(top_weighted, "nota_ponderada", "Quais livros se destacam pela nota ponderada?", "Ranking que equilibra avaliação média e volume de avaliações"), "fig-weighted"),
        "reviews": fig_html(grafico_barras_livros(top_reviews, "numero_avaliacoes", "Quais livros concentram mais avaliações?", "Número de avaliações como indicador de alcance e popularidade", "inteiro"), "fig-reviews"),
        "score": fig_html(grafico_barras_livros(top_score, "score", "Quais livros têm maior score na Best Books Ever?", "Métrica específica da própria lista do Goodreads", "inteiro"), "fig-score") if not top_score.empty else '<div class="empty-state">Score não disponível para esta base.</div>',
        "author_concentration": fig_html(grafico_concentracao_autores(completa), "fig-author-concentration"),
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
<link rel="icon" href="{GOODREADS_FAVICON_URI}">
<style>{CSS}</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-brand"><div class="brand-mark"><span class="brand-favicon"><img src="{GOODREADS_FAVICON_URI}" alt="Goodreads"></span><div class="brand-copy">Goodreads<small>DATA REPORT</small></div></div></div>
    <button class="sidebar-toggle" type="button" title="Recolher barra lateral">☰</button>
    <nav class="nav">{nav}</nav>
    <div class="sidebar-footer">Projeto acadêmico<br>Web scraping · Python</div>
  </aside>
  <main class="main">{body}</main>
</div>
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
    parser = argparse.ArgumentParser(description="Gera a versão final v2 do relatório standalone do Goodreads.")
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
