import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from modelo_kmeans import segmentar

load_dotenv()

#nomeando  o site
st.set_page_config(
    page_title="Dashboard do Evento",
    page_icon="🎵",
    layout="wide"
)
#conexao com o banco
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


@st.cache_data(ttl=300)
def load_data():
    conn = get_connection()
    query = """
        SELECT
            f.id_fato,
            f.data_compra,
            f.valor,
            c.genero,
            c.data_de_nascimento,
            c.cep,
            m.metodo_de_pagamento,
            e.email
        FROM dw.fato_244 f
        JOIN dw.dim_cliente   c ON f.id_cliente   = c.id_cliente
        JOIN dw.dim_metodopag m ON f.id_metodopag = m.id_metodopag
        JOIN dw.dim_email     e ON f.id_email      = e.id_email
    """
    return pd.read_sql(query, conn)

#reconhcer bairros

@st.cache_data(show_spinner=False)
def cep_para_bairro(cep: str) -> str:
    try:
        cep_limpo = cep.replace("-", "").replace(" ", "").strip()
        if len(cep_limpo) != 8:
            return "Não identificado"
        r = requests.get(
            f"https://viacep.com.br/ws/{cep_limpo}/json/",
            timeout=5
        )
        data = r.json()
        if "erro" in data:
            return "Não identificado"
        bairro = data.get("bairro", "").strip()
        return bairro if bairro else "Não identificado"
    except Exception:
        return "Não identificado"


def enriquecer_bairros(df):
    ceps_unicos = df["cep"].dropna().unique()
    mapa = {}
    prog = st.progress(0, text="Consultando bairros via ViaCEP...")
    for i, cep in enumerate(ceps_unicos):
        mapa[cep] = cep_para_bairro(str(cep))
        prog.progress(
            (i + 1) / len(ceps_unicos),
            text=f"Consultando CEPs… {i+1}/{len(ceps_unicos)}"
        )
        time.sleep(0.05)
    prog.empty()
    df["bairro"] = df["cep"].map(mapa).fillna("Não identificado")
    return df

def processar(df):
    df = df.copy()

    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0)

    df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
    df["data"] = df["data_compra"].dt.date
    df["hora"] = df["data_compra"].dt.hour

    hoje = datetime.today()
    df["data_de_nascimento"] = pd.to_datetime(
        df["data_de_nascimento"], errors="coerce", dayfirst=True
    )
    df["idade"] = df["data_de_nascimento"].apply(
        lambda d: (hoje - d).days // 365 if pd.notna(d) else None
    )

    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 17, 24, 34, 44, 59, 150],
        labels=["< 18", "18–24", "25–34", "35–44", "45–59", "60+"]
    )

    df["tipo"] = df["valor"].apply(
        lambda v: "Pago" if v > 0 else "Gratuito"
    )

    df["genero"] = df["genero"].str.strip().str.capitalize()

    return df

# carrega e processa
with st.spinner("Carregando dados do banco..."):
    try:
        df_raw = load_data()
    except Exception as e:
        st.error(f"Erro ao conectar no banco: {e}")
        st.stop()

df = processar(df_raw)

CACHE_BAIRROS = "cache_bairros.csv"

if os.path.exists(CACHE_BAIRROS):
    cache = pd.read_csv(CACHE_BAIRROS)
    mapa = dict(zip(cache["cep"].astype(str), cache["bairro"]))
    df["bairro"] = df["cep"].astype(str).map(mapa).fillna("Não identificado")
else:
    df = enriquecer_bairros(df)
    cache_df = df[["cep", "bairro"]].drop_duplicates()
    cache_df.to_csv(CACHE_BAIRROS, index=False)


# header
st.title("🎵 Dashboard do Evento")
st.markdown("---")

# métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de inscritos",  len(df))
col2.metric("Ingressos pagos",     len(df[df["tipo"] == "Pago"]))
col3.metric("Ingressos gratuitos", len(df[df["tipo"] == "Gratuito"]))
col4.metric("Receita total",       f"R$ {df['valor'].sum():,.2f}")

#bairros

st.markdown("---")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("📍 Inscritos por bairro (DF)")
    bairros = (
        df[df["bairro"] != "Não identificado"]
        .groupby("bairro")
        .size()
        .reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False)
        .head(20)
    )
    fig_bairro = px.bar(
        bairros,
        x="quantidade",
        y="bairro",
        orientation="h",
        color="quantidade",
        color_continuous_scale="Blues",
        labels={"quantidade": "Inscritos", "bairro": "Bairro"},
    )
    fig_bairro.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
    )
    st.plotly_chart(fig_bairro, use_container_width=True)

with col_b:
    st.subheader("💳 Método de pagamento")
    metodos = df.groupby("metodo_de_pagamento").size().reset_index(name="quantidade")
    fig_met = px.pie(
        metodos,
        names="metodo_de_pagamento",
        values="quantidade",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_met.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
    fig_met.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_met, use_container_width=True)

    st.subheader("🎟️ Pago vs Gratuito")
    tipo_df = df.groupby("tipo").size().reset_index(name="quantidade")
    fig_tipo = px.pie(
        tipo_df,
        names="tipo",
        values="quantidade",
        hole=0.45,
        color_discrete_sequence=["#378ADD", "#9FE1CB"],
    )
    fig_tipo.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
    fig_tipo.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_tipo, use_container_width=True)

#idade

st.markdown("---")
st.subheader("📅 Inscrições por dia")
por_dia = df.groupby("data").size().reset_index(name="inscritos")
fig_tempo = px.area(
    por_dia, x="data", y="inscritos",
    labels={"data": "Data", "inscritos": "Inscritos"},
    color_discrete_sequence=["#378ADD"],
)
fig_tempo.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
st.plotly_chart(fig_tempo, use_container_width=True)

st.markdown("---")
col_c, col_d, col_e = st.columns(3)

with col_c:
    st.subheader("👥 Gênero")
    genero_df = df.groupby("genero").size().reset_index(name="quantidade")
    fig_gen = px.bar(
        genero_df, x="genero", y="quantidade",
        color="genero",
        color_discrete_sequence=["#378ADD", "#D4537E", "#888780"],
        labels={"genero": "", "quantidade": "Inscritos"},
    )
    fig_gen.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_gen, use_container_width=True)

with col_d:
    st.subheader("🎂 Faixa etária")
    faixa_df = df["faixa_etaria"].value_counts().sort_index().reset_index()
    faixa_df.columns = ["faixa", "quantidade"]
    fig_faixa = px.bar(
        faixa_df, x="faixa", y="quantidade",
        color="quantidade",
        color_continuous_scale="Blues",
        labels={"faixa": "Faixa etária", "quantidade": "Inscritos"},
    )
    fig_faixa.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_faixa, use_container_width=True)

with col_e:
    st.subheader("🕐 Horário das inscrições")
    hora_df = df.groupby("hora").size().reset_index(name="quantidade")
    fig_hora = px.bar(
        hora_df, x="hora", y="quantidade",
        color="quantidade",
        color_continuous_scale="Blues",
        labels={"hora": "Hora do dia", "quantidade": "Inscritos"},
    )
    fig_hora.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_hora, use_container_width=True)

st.markdown("---")
st.caption("Dashboard gerado com Streamlit + Plotly | Dados: PostgreSQL DW")

# ─── LINHA 5: ANÁLISE DE MERCADO — OUTROS GÊNEROS ────────────────────────────
st.markdown("---")
st.header("🎶 Análise de Mercado — Pagode, Sertanejo e Rock em Brasília")

@st.cache_data(ttl=300)
def load_outros_generos():
    conn = get_connection()
    return pd.read_sql("SELECT * FROM dw.eventos_outros_generos", conn)

df_outros = load_outros_generos()
df_outros["data"] = pd.to_datetime(df_outros["data"], errors="coerce")
df_outros["ano"] = df_outros["data"].dt.year
df_outros["mes"] = df_outros["data"].dt.month

col_o1, col_o2, col_o3 = st.columns(3)
col_o1.metric("Total de eventos", len(df_outros))
col_o2.metric("Eventos gratuitos", len(df_outros[df_outros["gratuito"] == "Sim"]))
col_o3.metric("Gêneros coletados", df_outros["genero"].nunique())

st.markdown("---")
col_p, col_q = st.columns(2)

with col_p:
    st.subheader("🎵 Distribuição por gênero")
    por_genero = df_outros["genero"].value_counts().reset_index()
    por_genero.columns = ["genero", "quantidade"]
    fig_og = px.pie(por_genero, names="genero", values="quantidade",
                    hole=0.45, color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623"])
    fig_og.update_layout(height=350)
    st.plotly_chart(fig_og, use_container_width=True)

with col_q:
    st.subheader("📅 Eventos por mês")
    por_mes = df_outros.groupby(["ano", "mes"]).size().reset_index(name="quantidade")
    por_mes["periodo"] = por_mes["ano"].astype(str) + "-" + por_mes["mes"].astype(str).str.zfill(2)
    fig_mes = px.bar(por_mes, x="periodo", y="quantidade",
                     color="quantidade", color_continuous_scale="Blues",
                     labels={"periodo": "Mês", "quantidade": "Eventos"})
    fig_mes.update_layout(coloraxis_showscale=False, height=350)
    st.plotly_chart(fig_mes, use_container_width=True)

st.subheader("📍 Eventos por local")
por_local = df_outros[df_outros["local"].notna() & (df_outros["local"] != "")]\
    .groupby("local").size().reset_index(name="quantidade")\
    .sort_values("quantidade", ascending=False).head(15)
fig_local = px.bar(por_local, x="quantidade", y="local", orientation="h",
                   color="quantidade", color_continuous_scale="Blues")
fig_local.update_layout(yaxis=dict(autorange="reversed"),
                        coloraxis_showscale=False, height=450)
st.plotly_chart(fig_local, use_container_width=True)

# ─── LINHA 6: IA — CLUSTERIZAÇÃO DE PERFIS ───────────────────────────────────
st.markdown("---")
st.header("🤖 IA — Segmentação de Perfis de Público (K-Means)")
st.markdown("Modelo de machine learning que agrupa os inscritos em perfis automaticamente com base em idade, valor pago e método de pagamento.")

df_clusters, resumo = segmentar(df)

col_k1, col_k2 = st.columns(2)

with col_k1:
    st.subheader("👥 Distribuição dos perfis")
    fig_cl = px.pie(
        resumo, names="perfil", values="quantidade",
        hole=0.45,
        color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623", "#27AE60"]
    )
    fig_cl.update_layout(height=350)
    st.plotly_chart(fig_cl, use_container_width=True)

with col_k2:
    st.subheader("📊 Idade vs Valor pago por perfil")
    fig_sc = px.scatter(
        df_clusters, x="idade", y="valor", color="perfil",
        color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623", "#27AE60"],
        labels={"idade": "Idade", "valor": "Valor (R$)", "perfil": "Perfil"},
        opacity=0.7,
    )
    fig_sc.update_layout(height=350)
    st.plotly_chart(fig_sc, use_container_width=True)

st.subheader("📋 Características de cada perfil")
tabela = resumo[["perfil", "quantidade", "idade_media", "valor_medio", "genero_predominante", "metodo_top", "bairro_top"]].copy()
tabela.columns = ["Perfil", "Inscritos", "Idade média", "Valor médio (R$)", "Gênero predominante", "Método de pagamento", "Bairro principal"]
st.dataframe(tabela, use_container_width=True, hide_index=True)

# ─── LINHA 4: ANÁLISE DE MERCADO ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 Análise de Mercado — Funk/Trap em Brasília")

@st.cache_data(ttl=300)
def load_mercado():
    conn = get_connection()
    query = "SELECT * FROM dw.eventos_mercado"
    return pd.read_sql(query, conn)

df_mercado = load_mercado()
df_mercado["data"] = pd.to_datetime(df_mercado["data"], errors="coerce")
df_mercado["ano"] = df_mercado["data"].dt.year
df_mercado["mes"] = df_mercado["data"].dt.month

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total de eventos", len(df_mercado))
col_m2.metric("Eventos gratuitos", len(df_mercado[df_mercado["gratuito"] == "Sim"]))
col_m3.metric("Regiões cobertas", df_mercado["bairro"].nunique())

st.markdown("---")
col_f, col_g = st.columns(2)

with col_f:
    st.subheader("📍 Eventos por região do DF")
    por_bairro = df_mercado[df_mercado["bairro"].notna() & (df_mercado["bairro"] != "")]\
        .groupby("bairro").size().reset_index(name="quantidade")\
        .sort_values("quantidade", ascending=False).head(15)
    fig_mb = px.bar(por_bairro, x="quantidade", y="bairro", orientation="h",
                    color="quantidade", color_continuous_scale="Blues")
    fig_mb.update_layout(yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False, height=400)
    st.plotly_chart(fig_mb, use_container_width=True)

with col_g:
    st.subheader("🎵 Eventos por gênero")
    por_genero = df_mercado["genero"].value_counts().reset_index()
    por_genero.columns = ["genero", "quantidade"]
    fig_mg = px.pie(por_genero, names="genero", values="quantidade",
                    hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_mg.update_layout(height=400)
    st.plotly_chart(fig_mg, use_container_width=True)

st.markdown("---")
col_h, col_i = st.columns(2)

st.subheader("📅 Dados de eventos coletados por ano")
por_ano = df_mercado.groupby("ano").size().reset_index(name="quantidade")
fig_ano = px.bar(por_ano, x="ano", y="quantidade",
                 color="quantidade", color_continuous_scale="Blues")
fig_ano.update_layout(coloraxis_showscale=False, height=350)
st.plotly_chart(fig_ano, use_container_width=True)

st.markdown("---")
st.subheader("🗺️ Mapa de Oportunidade: Perfis × Gêneros do Mercado")
st.markdown("Score de afinidade entre cada perfil do público e os gêneros musicais disponíveis no mercado — quanto maior, mais eventos existem nos bairros onde esse perfil mora.")

# Montar base de eventos
df_mercado_full = pd.concat([
    df_mercado.assign(dataset="Funk/Trap"),
    df_outros.assign(dataset="Pagode/Sertanejo/Rock")
], ignore_index=True)

# Normalizar bairro nos eventos (manter só os reconhecidos)
REGIOES_DF = [
    "Asa Norte", "Asa Sul", "Lago Sul", "Lago Norte", "Ceilândia",
    "Taguatinga", "Samambaia", "Gama", "Planaltina", "Sobradinho",
    "Santa Maria", "Guará", "Cruzeiro", "Noroeste", "Sudoeste",
    "Águas Claras", "Park Way",
]
eventos_com_bairro = df_mercado_full[
    df_mercado_full["bairro"].isin(REGIOES_DF)
][["genero", "bairro"]]

# Eventos por bairro × gênero
eventos_pivot = eventos_com_bairro.groupby(["bairro", "genero"]).size().reset_index(name="n_eventos")

# Inscritos com perfil e bairro identificado
df_com_bairro = df_clusters[
    (df_clusters["bairro"] != "Não identificado") &
    (df_clusters["bairro"].isin(REGIOES_DF))
][["perfil", "bairro"]]

# Inscritos por perfil × bairro
inscritos_pivot = df_com_bairro.groupby(["perfil", "bairro"]).size().reset_index(name="n_inscritos")

# Cruzamento: para cada perfil, somar eventos disponíveis nos bairros onde mora, por gênero
cruzamento = inscritos_pivot.merge(eventos_pivot, on="bairro", how="inner")
cruzamento["score"] = cruzamento["n_inscritos"] * cruzamento["n_eventos"]

heatmap_data = cruzamento.groupby(["perfil", "genero"])["score"].sum().reset_index()

# Filtrar top gêneros para não poluir
top_generos = (
    df_mercado_full["genero"].value_counts().head(6).index.tolist()
)
heatmap_data = heatmap_data[heatmap_data["genero"].isin(top_generos)]

if not heatmap_data.empty:
    pivot = heatmap_data.pivot(index="perfil", columns="genero", values="score").fillna(0)
    # Normalizar cada linha para % (0–100)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)

    import plotly.graph_objects as go
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot_pct.values,
        x=pivot_pct.columns.tolist(),
        y=pivot_pct.index.tolist(),
        colorscale="Blues",
        text=pivot_pct.values.round(1),
        texttemplate="%{text}%",
        showscale=True,
        colorbar=dict(title="Afinidade (%)"),
    ))
    fig_hm.update_layout(
        height=350,
        xaxis_title="Gênero musical no mercado",
        yaxis_title="Perfil do público",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Score = nº de inscritos do perfil no bairro × nº de eventos do gênero no mesmo bairro, normalizado por perfil. Fonte: Sympla Brasília (2025/2026)")
else:
    st.info("Dados insuficientes para gerar o mapa de oportunidade (poucos bairros em comum).")