import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os

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

if "bairro" not in df.columns:
    df = enriquecer_bairros(df)

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