import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import psycopg2
except Exception:
    psycopg2 = None
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from modelo_kmeans import segmentar
from modelo_previsao import ModeloOportunidade, ModeloPublico, preparar_mercado

load_dotenv()


def achar_arquivo(nome):
    """Procura o arquivo na pasta do dashboard e na pasta de cima (raiz do repositório)."""
    for pasta in [os.path.dirname(os.path.abspath(__file__)), os.getcwd(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]:
        caminho = os.path.join(pasta, nome)
        if os.path.exists(caminho):
            return caminho
    return None

st.set_page_config(
    page_title="Dashboard do Evento",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def get_connection():
    if psycopg2 is None:
        raise RuntimeError('psycopg2 indisponivel')
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

@st.cache_data
def load_data_csv():
    """
    Plano B quando o PostgreSQL não está ligado: lê a camada gold exportada em CSV.
    Cada linha do arquivo vem inteira entre aspas e separada por ';', por isso a leitura manual.
    """
    caminho = achar_arquivo("camadagoldtexto.csv")
    with open(caminho, encoding="utf-8") as f:
        linhas = [l.strip().strip('"') for l in f if l.strip()]
    cab = linhas[0].split(";")
    df_csv = pd.DataFrame([l.split(";") for l in linhas[1:]], columns=cab)
    df_csv = df_csv.apply(lambda c: c.str.strip())
    valor = (df_csv["Valor"].str.replace("R$", "", regex=False).str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False).str.strip())
    return pd.DataFrame({
        "id_fato": range(1, len(df_csv) + 1),
        "data_compra": df_csv["Data_Compra"],
        "valor": pd.to_numeric(valor, errors="coerce"),
        "genero": df_csv["Genero"],
        "data_de_nascimento": df_csv["Data_de_Nascimento"],
        "cep": df_csv["CEP"],
        "metodo_de_pagamento": df_csv["Metodo_de_pagamento"],
        "email": df_csv["Email"],
    })

@st.cache_data(show_spinner=False)
def cep_para_bairro(cep: str) -> str:
    try:
        cep_limpo = cep.replace("-", "").replace(" ", "").strip()
        if len(cep_limpo) != 8:
            return "Não identificado"
        r = requests.get(f"https://viacep.com.br/ws/{cep_limpo}/json/", timeout=5)
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
        prog.progress((i + 1) / len(ceps_unicos), text=f"Consultando CEPs… {i+1}/{len(ceps_unicos)}")
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
    df["data_de_nascimento"] = pd.to_datetime(df["data_de_nascimento"], errors="coerce", dayfirst=True)
    df["idade"] = df["data_de_nascimento"].apply(lambda d: (hoje - d).days // 365 if pd.notna(d) else None)
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 17, 24, 34, 44, 59, 150],
        labels=["< 18", "18–24", "25–34", "35–44", "45–59", "60+"]
    )
    df["tipo"] = df["valor"].apply(lambda v: "Pago" if v > 0 else "Gratuito")
    df["genero"] = df["genero"].str.strip().str.capitalize()
    return df

# ── Carrega dados ────────────────────────────────────────────────────────────
USANDO_CSV = False
with st.spinner("Carregando dados do banco..."):
    try:
        df_raw = load_data()
    except Exception as e:
        # sem banco, usa os CSVs do repositório pra o dashboard continuar funcionando
        if achar_arquivo("camadagoldtexto.csv"):
            df_raw = load_data_csv()
            USANDO_CSV = True
        else:
            st.error(f"Erro ao conectar no banco: {e}")
            st.stop()

df = processar(df_raw)

CACHE_BAIRROS = achar_arquivo("cache_bairros.csv") or "cache_bairros.csv"
if os.path.exists(CACHE_BAIRROS):
    cache = pd.read_csv(CACHE_BAIRROS)
    mapa = dict(zip(cache["cep"].astype(str), cache["bairro"]))
    df["bairro"] = df["cep"].astype(str).map(mapa).fillna("Não identificado")
else:
    df = enriquecer_bairros(df)
    cache_df = df[["cep", "bairro"]].drop_duplicates()
    cache_df.to_csv(CACHE_BAIRROS, index=False)

@st.cache_data(ttl=300)
def load_mercado():
    if USANDO_CSV:
        return pd.read_csv(achar_arquivo("eventos_funk_trap_brasilia.csv"))
    conn = get_connection()
    return pd.read_sql("SELECT * FROM dw.eventos_mercado", conn)

@st.cache_data(ttl=300)
def load_outros_generos():
    if USANDO_CSV:
        return pd.read_csv(achar_arquivo("eventos_outros_generos_brasilia.csv"))
    conn = get_connection()
    return pd.read_sql("SELECT * FROM dw.eventos_outros_generos", conn)

df_mercado = load_mercado()
df_mercado["data"] = pd.to_datetime(df_mercado["data"], errors="coerce")
df_mercado["ano"] = df_mercado["data"].dt.year
df_mercado["mes"] = df_mercado["data"].dt.month

df_outros = load_outros_generos()
df_outros["data"] = pd.to_datetime(df_outros["data"], errors="coerce")
df_outros["ano"] = df_outros["data"].dt.year
df_outros["mes"] = df_outros["data"].dt.month

df_clusters, resumo = segmentar(df)

# ── Carregar dados do bar ────────────────────────────────────────────────────
@st.cache_data
def load_bar():
    path = achar_arquivo("custo_bar_evento.csv")
    if path:
        return pd.read_csv(path)
    return pd.DataFrame()

df_bar = load_bar()

# Métricas derivadas do bar real
if not df_bar.empty:
    total_inscritos = len(df)
    bebidas_vendidas = df_bar[df_bar["categoria"] == "bebidas"]["saida"].sum()
    custo_bebidas = df_bar[df_bar["categoria"] == "bebidas"]["custo_total"].sum()
    custo_funcionarios = df_bar[df_bar["categoria"] == "funcionarios"]["custo_total"].sum()
    custo_extras = df_bar[df_bar["categoria"] == "despesas_extras"]["custo_total"].sum()
    custo_bar_total = custo_bebidas + custo_funcionarios + custo_extras

    consumo_por_pessoa = bebidas_vendidas / total_inscritos if total_inscritos > 0 else 1.5
    custo_bar_por_pessoa = custo_bar_total / total_inscritos if total_inscritos > 0 else 25.0

    # Ticket médio de bebida (custo unitário médio ponderado pelas vendas)
    df_beb = df_bar[(df_bar["categoria"] == "bebidas") & (df_bar["saida"] > 0)].copy()
    ticket_medio_bebida = (df_beb["custo_unitario"] * df_beb["saida"]).sum() / df_beb["saida"].sum() if len(df_beb) > 0 else 8.50
else:
    consumo_por_pessoa = 1.5
    custo_bar_por_pessoa = 25.0
    ticket_medio_bebida = 8.50
    custo_bar_total = 7706.0

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════
st.title("🎵 Dashboard do Evento")
st.caption("Use esse painel para planejar seu próximo evento: entenda quem é seu público, quando agir, qual gênero apostar e quanto cobrar.")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de inscritos",  len(df))
col2.metric("Ingressos pagos",     len(df[df["tipo"] == "Pago"]))
col3.metric("Ingressos gratuitos", len(df[df["tipo"] == "Gratuito"]))
col4.metric("Receita total",       f"R$ {df['valor'].sum():,.2f}")

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — QUEM É O MEU PÚBLICO?
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("👥 Quem é o meu público?")
st.caption("Perfil demográfico dos inscritos — onde moram, como pagaram e qual faixa etária predomina.")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader("📍 Inscritos por bairro (DF)")
    bairros = (
        df[df["bairro"] != "Não identificado"]
        .groupby("bairro").size().reset_index(name="quantidade")
        .sort_values("quantidade", ascending=False).head(20)
    )
    fig_bairro = px.bar(
        bairros, x="quantidade", y="bairro", orientation="h",
        color="quantidade", color_continuous_scale="Blues",
        labels={"quantidade": "Inscritos", "bairro": "Bairro"},
    )
    fig_bairro.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=10, b=0), height=500)
    st.plotly_chart(fig_bairro, use_container_width=True)

with col_b:
    st.subheader("💳 Método de pagamento")
    metodos = df.groupby("metodo_de_pagamento").size().reset_index(name="quantidade")
    fig_met = px.pie(metodos, names="metodo_de_pagamento", values="quantidade",
                     hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_met.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=240)
    fig_met.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_met, use_container_width=True)

    st.subheader("🎟️ Pago vs Gratuito")
    tipo_df = df.groupby("tipo").size().reset_index(name="quantidade")
    fig_tipo = px.pie(tipo_df, names="tipo", values="quantidade",
                      hole=0.45, color_discrete_sequence=["#378ADD", "#9FE1CB"])
    fig_tipo.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=240)
    fig_tipo.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_tipo, use_container_width=True)

st.markdown("")
col_c, col_d, col_e = st.columns(3)

with col_c:
    st.subheader("👤 Gênero")
    genero_df = df.groupby("genero").size().reset_index(name="quantidade")
    fig_gen = px.bar(genero_df, x="genero", y="quantidade", color="genero",
                     color_discrete_sequence=["#378ADD", "#D4537E", "#888780"],
                     labels={"genero": "", "quantidade": "Inscritos"})
    fig_gen.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig_gen, use_container_width=True)

with col_d:
    st.subheader("🎂 Faixa etária")
    faixa_df = df["faixa_etaria"].value_counts().sort_index().reset_index()
    faixa_df.columns = ["faixa", "quantidade"]
    fig_faixa = px.bar(faixa_df, x="faixa", y="quantidade",
                       color="quantidade", color_continuous_scale="Blues",
                       labels={"faixa": "Faixa etária", "quantidade": "Inscritos"})
    fig_faixa.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig_faixa, use_container_width=True)

with col_e:
    st.subheader("🕐 Horário das inscrições")
    hora_df = df.groupby("hora").size().reset_index(name="quantidade")
    fig_hora = px.bar(hora_df, x="hora", y="quantidade",
                      color="quantidade", color_continuous_scale="Blues",
                      labels={"hora": "Hora do dia", "quantidade": "Inscritos"})
    fig_hora.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=280)
    st.plotly_chart(fig_hora, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — QUANDO ORGANIZAR?
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("📣 Quando divulgar? — Comportamento de compra")
st.caption("A maioria das inscrições acontece na semana final. Abra as vendas cedo, mas concentre o marketing nos últimos 7 dias.")

por_dia = df.groupby("data").size().reset_index(name="inscritos")
fig_tempo = px.area(por_dia, x="data", y="inscritos",
                    labels={"data": "Data", "inscritos": "Inscritos"},
                    color_discrete_sequence=["#378ADD"])
fig_tempo.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
st.plotly_chart(fig_tempo, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — QUAL GÊNERO APOSTAR? (Funk/Trap)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🎧 Qual gênero tem mais espaço? — Funk e Trap em Brasília")
st.caption("Dados do Sympla sobre eventos de Funk/Trap em Brasília. Volume, frequência e regiões com mais oferta.")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total de eventos", len(df_mercado))
col_m2.metric("Eventos gratuitos", len(df_mercado[df_mercado["gratuito"] == "Sim"]))
col_m3.metric("Regiões cobertas", df_mercado["bairro"].nunique())

st.markdown("")
col_f, col_g = st.columns(2)

with col_f:
    st.subheader("📍 Eventos por região do DF")
    por_bairro = (df_mercado[df_mercado["bairro"].notna() & (df_mercado["bairro"] != "")]
                  .groupby("bairro").size().reset_index(name="quantidade")
                  .sort_values("quantidade", ascending=False).head(12))
    fig_mb = px.bar(por_bairro, x="quantidade", y="bairro", orientation="h",
                    color="quantidade", color_continuous_scale="Blues")
    fig_mb.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_mb, use_container_width=True)

with col_g:
    st.subheader("🎵 Top subgêneros (Funk/Trap)")
    por_genero_m = df_mercado["genero"].value_counts().head(6).reset_index()
    por_genero_m.columns = ["genero", "quantidade"]
    fig_mg = px.bar(por_genero_m, x="quantidade", y="genero", orientation="h",
                    color="quantidade", color_continuous_scale="Blues",
                    labels={"quantidade": "Eventos", "genero": "Subgênero"})
    fig_mg.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_mg, use_container_width=True)

st.subheader("📅 Eventos coletados por ano")
por_ano = df_mercado.groupby("ano").size().reset_index(name="quantidade")
fig_ano = px.bar(por_ano, x="ano", y="quantidade",
                 color="quantidade", color_continuous_scale="Blues",
                 labels={"ano": "Ano", "quantidade": "Eventos"})
fig_ano.update_layout(coloraxis_showscale=False, height=300)
st.plotly_chart(fig_ano, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — CONCORRÊNCIA (Pagode, Sertanejo, Rock)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🎶 Concorrência — Pagode, Sertanejo e Rock em Brasília")
st.caption("Outros gêneros disputam a mesma audiência. Veja frequência, locais mais usados e sazonalidade para identificar lacunas no calendário.")

col_o1, col_o2, col_o3 = st.columns(3)
col_o1.metric("Total de eventos", len(df_outros))
col_o2.metric("Eventos gratuitos", len(df_outros[df_outros["gratuito"] == "Sim"]))
col_o3.metric("Gêneros coletados", df_outros["genero"].nunique())

st.markdown("")
col_p, col_q = st.columns(2)

with col_p:
    st.subheader("🎵 Distribuição por gênero")
    por_genero_o = df_outros["genero"].value_counts().reset_index()
    por_genero_o.columns = ["genero", "quantidade"]
    fig_og = px.pie(por_genero_o, names="genero", values="quantidade",
                    hole=0.45, color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623"])
    fig_og.update_layout(height=320)
    st.plotly_chart(fig_og, use_container_width=True)

with col_q:
    st.subheader("📅 Eventos por mês")
    por_mes = df_outros.groupby(["ano", "mes"]).size().reset_index(name="quantidade")
    por_mes["periodo"] = por_mes["ano"].astype(str) + "-" + por_mes["mes"].astype(str).str.zfill(2)
    fig_mes = px.bar(por_mes.sort_values("periodo"), x="periodo", y="quantidade",
                     color="quantidade", color_continuous_scale="Blues",
                     labels={"periodo": "Mês", "quantidade": "Eventos"})
    fig_mes.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig_mes, use_container_width=True)

st.subheader("📍 Eventos por local")
por_local = (df_outros[df_outros["local"].notna() & (df_outros["local"] != "")]
             .groupby("local").size().reset_index(name="quantidade")
             .sort_values("quantidade", ascending=False).head(12))
fig_local = px.bar(por_local, x="quantidade", y="local", orientation="h",
                   color="quantidade", color_continuous_scale="Blues")
fig_local.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=400)
st.plotly_chart(fig_local, use_container_width=True)

st.subheader("📊 Sazonalidade por gênero")
por_mes_genero = (
    df_outros[df_outros["data"].notna()]
    .groupby(["ano", "mes", "genero"]).size().reset_index(name="quantidade")
)
por_mes_genero["periodo"] = (
    por_mes_genero["ano"].astype(str) + "-" +
    por_mes_genero["mes"].astype(str).str.zfill(2)
)
fig_mg_mes = px.bar(
    por_mes_genero.sort_values("periodo"),
    x="periodo", y="quantidade", color="genero", barmode="group",
    color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623"],
    labels={"periodo": "Mês", "quantidade": "Eventos", "genero": "Gênero"},
)
fig_mg_mes.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_mg_mes, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — QUANTO COBRAR? (K-Means)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("💰 Quanto cobrar? — Segmentação do público por IA")
st.caption("O modelo K-Means agrupa os inscritos em perfis distintos por idade, região, gênero e comportamento de pagamento. Use para decidir a política de preços do seu evento.")

col_k1, col_k2 = st.columns(2)

with col_k1:
    st.subheader("👥 Distribuição dos perfis")
    fig_cl = px.pie(resumo, names="perfil", values="quantidade", hole=0.45,
                    color_discrete_sequence=["#378ADD", "#D4537E", "#F5A623", "#27AE60"])
    fig_cl.update_layout(height=320)
    st.plotly_chart(fig_cl, use_container_width=True)

with col_k2:
    st.subheader("📊 Perfis: antecedência vs valor médio")
    fig_bar_perf = px.bar(
        resumo.melt(id_vars="perfil", value_vars=["dias_antes_media", "valor_medio"],
                    var_name="métrica", value_name="valor"),
        x="perfil", y="valor", color="métrica", barmode="group",
        color_discrete_map={"dias_antes_media": "#378ADD", "valor_medio": "#D4537E"},
        labels={"perfil": "Perfil", "valor": "Valor", "métrica": "Métrica"},
    )
    fig_bar_perf.for_each_trace(lambda t: t.update(
        name="Dias antes (média)" if t.name == "dias_antes_media" else "Valor médio (R$)"
    ))
    fig_bar_perf.update_layout(height=320, xaxis_tickangle=-15,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_bar_perf, use_container_width=True)

st.subheader("📋 Características de cada perfil")
tabela = resumo[["perfil", "quantidade", "idade_media", "dias_antes_media", "valor_medio",
                  "genero_predominante", "metodo_top", "bairro_top"]].copy()
tabela.columns = ["Perfil", "Inscritos", "Idade média", "Dias antes (média)", "Valor médio (R$)",
                   "Gênero predominante", "Método de pagamento", "Bairro principal"]
st.dataframe(tabela, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — MAPA DE OPORTUNIDADE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("🗺️ Mapa de Oportunidade: Perfis × Gêneros do Mercado")
st.caption("Cruzamento entre onde seu público mora e a oferta de eventos por gênero na mesma região.")

df_mercado_full = pd.concat([
    df_mercado.assign(dataset="Funk/Trap"),
    df_outros.assign(dataset="Pagode/Sertanejo/Rock")
], ignore_index=True)

REGIOES_DF = [
    "Asa Norte", "Asa Sul", "Lago Sul", "Lago Norte", "Ceilândia",
    "Taguatinga", "Samambaia", "Gama", "Planaltina", "Sobradinho",
    "Santa Maria", "Guará", "Cruzeiro", "Noroeste", "Sudoeste",
    "Águas Claras", "Park Way",
]

eventos_com_bairro = df_mercado_full[df_mercado_full["bairro"].isin(REGIOES_DF)][["genero", "bairro"]]
eventos_pivot = eventos_com_bairro.groupby(["bairro", "genero"]).size().reset_index(name="n_eventos")

df_com_bairro = df_clusters[
    (df_clusters["bairro"] != "Não identificado") &
    (df_clusters["bairro"].isin(REGIOES_DF))
][["perfil", "bairro"]]
inscritos_pivot = df_com_bairro.groupby(["perfil", "bairro"]).size().reset_index(name="n_inscritos")

cruzamento = inscritos_pivot.merge(eventos_pivot, on="bairro", how="inner")
cruzamento["score"] = cruzamento["n_inscritos"] * cruzamento["n_eventos"]
heatmap_data = cruzamento.groupby(["perfil", "genero"])["score"].sum().reset_index()

top_generos = df_mercado_full["genero"].value_counts().head(6).index.tolist()
heatmap_data = heatmap_data[heatmap_data["genero"].isin(top_generos)]

if not heatmap_data.empty:
    pivot = heatmap_data.pivot(index="perfil", columns="genero", values="score").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
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

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — SIMULADOR DE EVENTO
# ════════════════════════════════════════════════════════════════════════════
st.header("🧮 Simulador de Evento")
st.caption("Informe os parâmetros do seu evento e veja se o orçamento fecha com base nos dados reais do público.")

if USANDO_CSV:
    st.caption("ℹ️ Banco PostgreSQL indisponível: dados carregados dos CSVs do repositório (camada gold e eventos de mercado).")

# ── Modelos de previsão (treinados com os dados carregados acima) ────────────
eventos_mercado = preparar_mercado(df_mercado, df_outros)
modelo_oport = ModeloOportunidade().treinar(eventos_mercado)
modelo_publico = ModeloPublico().treinar(df_clusters)

# Antecedência mínima: o público compra em média X dias antes, então a data
# do evento precisa dar tempo de divulgar (mesma regra da dica de timing)
dias_compra = float((resumo["dias_antes_media"] * resumo["quantidade"]).sum() / resumo["quantidade"].sum())
antecedencia_min = max(int(dias_compra) + 7, 14)

# ════════════════════════════════════════════════════════════════════════════
# 8.1 — PREVISÃO DE DATA, GÊNERO E LOCAL
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🔮 Previsão de data, gênero e local")
st.caption("O modelo olha os eventos do mercado de Brasília e dá nota para cada combinação de data, gênero e local: "
           "quanto o gênero aparece no mercado, em que dia da semana ele costuma acontecer, quais casas recebem esse "
           "gênero e quantos eventos já estão marcados no mesmo dia (concorrência).")

hoje_ts = pd.Timestamp(datetime.today().date())
col_p1, col_p2 = st.columns([1, 2])
with col_p1:
    periodo = st.date_input(
        "📅 Período em que o evento pode acontecer",
        value=(hoje_ts.date(), (hoje_ts + pd.Timedelta(days=60)).date()),
        key="periodo_previsao",
    )
with col_p2:
    generos_escolhidos = st.multiselect("🎵 Gêneros que vocês topam fazer", modelo_oport.generos,
                                        default=modelo_oport.generos, key="generos_previsao")

if isinstance(periodo, (list, tuple)) and len(periodo) == 2 and generos_escolhidos:
    data_ini, data_fim = periodo
    data_minima = hoje_ts + pd.Timedelta(days=antecedencia_min)
    st.caption(f"Datas antes de **{data_minima:%d/%m/%Y}** ficam de fora: o público compra em média "
               f"{dias_compra:.0f} dias antes, então precisa de pelo menos {antecedencia_min} dias pra divulgar. "
               f"A concorrência só conta eventos já coletados (agenda até {eventos_mercado['data'].max():%d/%m/%Y}).")

    ranking = modelo_oport.recomendar(data_ini, data_fim, generos=generos_escolhidos,
                                      data_minima=data_minima, top=len(generos_escolhidos), um_por_genero=True)
    if ranking.empty:
        st.warning("Nenhuma combinação possível nesse período. Aumente o período ou escolha mais gêneros.")
        recomendacao = None
    else:
        melhor = ranking.iloc[0]
        st.success(f"🏆 **Recomendação:** {melhor['genero']} no **{melhor['local']}**, "
                   f"{melhor['dia_semana'].lower()} **{melhor['data']:%d/%m/%Y}**  \n{melhor['motivo']}")

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.markdown("**Melhor combinação de cada gênero**")
            fig_rank = px.bar(ranking.sort_values("nota"), x="nota", y="genero", orientation="h",
                              text=ranking.sort_values("nota")["nota"].map(lambda v: f"{v:.0f}"),
                              color="nota", color_continuous_scale="Blues",
                              hover_data={"local": True, "dia_semana": True},
                              labels={"nota": "Nota (0–100)", "genero": ""})
            fig_rank.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_rank, use_container_width=True)
        with col_r2:
            st.markdown("**Em que dia da semana cada gênero acontece**")
            dias = modelo_oport.dias_por_genero()
            dias = dias[dias["genero"].isin(generos_escolhidos)]
            pivot_dias = dias.pivot(index="genero", columns="dia", values="probabilidade")[
                ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]]
            fig_dias = go.Figure(data=go.Heatmap(z=(pivot_dias.values * 100).round(0), x=pivot_dias.columns,
                                                 y=pivot_dias.index, colorscale="Blues",
                                                 texttemplate="%{z:.0f}%", showscale=False))
            fig_dias.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_dias, use_container_width=True)

        genero_sim = st.selectbox("Ver os melhores locais (e a melhor data de cada um) para o gênero", ranking["genero"].tolist(),
                                  key="genero_detalhe")
        opcoes = modelo_oport.recomendar(data_ini, data_fim, generos=[genero_sim], data_minima=data_minima, top=5)
        tabela_op = opcoes[["data", "dia_semana", "local", "regiao_local", "nota", "motivo"]].copy()
        tabela_op["data"] = tabela_op["data"].dt.strftime("%d/%m/%Y")
        tabela_op.columns = ["Data", "Dia", "Local", "Região do local", "Nota", "Por quê"]
        st.dataframe(tabela_op, use_container_width=True, hide_index=True)
        recomendacao = opcoes.iloc[0] if not opcoes.empty else None
else:
    st.info("Escolha a data inicial e a final do período e pelo menos um gênero.")
    recomendacao = None

st.markdown("#### 💵 Parâmetros do evento")

# ── Inputs principais ────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
orcamento = col_s1.number_input("💰 Orçamento total (R$)", min_value=0, value=5000, step=500)
capacidade = col_s2.number_input("👥 Capacidade máxima", min_value=1, value=200, step=10)
# se o local recomendado tem região conhecida, já começa selecionado nela
regiao_rec = recomendacao["regiao_local"] if recomendacao is not None else "Não informado"
local_sim = col_s3.selectbox("📍 Local do evento", ["Plano Piloto", "Entorno"],
                             index=1 if regiao_rec == "Entorno" else 0)
preco_ingresso = col_s4.number_input("🎟️ Preço do ingresso (R$)", min_value=0, value=20, step=5)

# ── Montador de cardápio ──────────────────────────────────────────────────────
st.markdown("**🍹 Monte seu cardápio**")
vai_ter_bar = st.checkbox("Vai ter bar no evento?", value=True)

cardapio_resultado = pd.DataFrame(columns=["Drink", "Custo do drink (R$)", "Preço sugerido (R$)"])

if vai_ter_bar:

    # ── 1. Insumos ──────────────────────────────────────────────────────────
    st.markdown("**1️⃣ Insumos — o que você vai comprar**")
    insumos_default = pd.DataFrame({
        "Insumo": ["Smirnoff 1L", "Red Bull 250ml lata", "Energético 2L", "Heineken long neck 269ml", "Água 500ml", "Gelo (saco 5kg)"],
        "Tipo": ["garrafa", "lata", "garrafa", "lata", "garrafa", "saco"],
        "Volume (ml)": [1000, 250, 2000, 269, 500, 5000],
        "Custo (R$)": [39.99, 8.50, 20.00, 6.80, 1.09, 15.00],
    })
    insumos = st.data_editor(
        insumos_default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Insumo": st.column_config.TextColumn("Insumo", width="medium"),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["garrafa", "lata", "saco", "outro"]),
            "Volume (ml)": st.column_config.NumberColumn("Volume (ml)", min_value=1),
            "Custo (R$)": st.column_config.NumberColumn("Custo (R$)", min_value=0.0, format="R$ %.2f"),
        },
        key="tabela_insumos"
    )
    insumos = insumos.dropna(subset=["Insumo"])
    insumos["Custo por ml (R$)"] = insumos["Custo (R$)"] / insumos["Volume (ml)"]
    insumos_tipo = insumos.set_index("Insumo")["Tipo"].to_dict()
    insumos_custo_ml = insumos.set_index("Insumo")["Custo por ml (R$)"].to_dict()
    insumos_custo_unit = insumos.set_index("Insumo")["Custo (R$)"].to_dict()
    lista_insumos = insumos["Insumo"].tolist()

    # ── 2. Formulário de drink ───────────────────────────────────────────────
    st.markdown("**2️⃣ Monte seus drinks**")

    if "drinks_salvos" not in st.session_state:
        st.session_state.drinks_salvos = []

    with st.form("form_drink", clear_on_submit=True):
        nome_drink = st.text_input("Nome do drink", placeholder="Ex: Copão Smirnoff + Energético")
        st.caption("Selecione até 4 ingredientes e informe quantos ml vai usar (latas/sacos = campo ignorado, cobra a unidade inteira).")

        cols_ing = st.columns(4)
        ingredientes_form = []
        for i, col in enumerate(cols_ing):
            ing = col.selectbox(f"Ingrediente {i+1}", ["—"] + lista_insumos, key=f"ing_{i}")
            ml = col.number_input(f"Ml usado", min_value=0, value=0, step=10, key=f"ml_{i}")
            ingredientes_form.append((ing, ml))

        salvar = st.form_submit_button("➕ Adicionar drink")
        if salvar:
            erros = []
            if not nome_drink.strip():
                erros.append("❌ **Nome do drink** não pode ser vazio.")

            ingredientes_validos = [(ing, ml) for ing, ml in ingredientes_form if ing != "—"]
            if not ingredientes_validos:
                erros.append("❌ **Ingredientes** — selecione pelo menos um.")
            else:
                for ing, ml in ingredientes_validos:
                    tipo = insumos_tipo.get(ing, "garrafa")
                    if tipo not in ["lata", "saco"] and ml == 0:
                        erros.append(f"❌ **{ing}** é garrafa — informe quantos ml vai usar.")

            if erros:
                for e in erros:
                    st.error(e)
            else:
                custo_drink = 0.0
                detalhes = []
                for ing, ml in ingredientes_validos:
                    tipo = insumos_tipo.get(ing, "garrafa")
                    if tipo in ["lata", "saco"]:
                        custo = insumos_custo_unit.get(ing, 0)
                        detalhes.append(f"{ing} (1 unid)")
                    else:
                        custo = insumos_custo_ml.get(ing, 0) * ml
                        detalhes.append(f"{ing} {ml}ml")
                    custo_drink += custo
                st.session_state.drinks_salvos.append({
                    "Drink": nome_drink.strip(),
                    "Ingredientes": " + ".join(detalhes),
                    "Custo do drink (R$)": round(custo_drink, 2),
                    "Preço sugerido (R$)": round(custo_drink * 2.5, 2),
                })
                st.success(f"✅ **{nome_drink.strip()}** adicionado ao cardápio!")

    # Botão pra limpar
    if st.session_state.drinks_salvos:
        col_lp1, col_lp2 = st.columns([6, 1])
        with col_lp2:
            if st.button("🗑️ Limpar cardápio"):
                st.session_state.drinks_salvos = []
                st.rerun()

    # ── 3. Resumo ────────────────────────────────────────────────────────────
    st.markdown("**3️⃣ Resumo do cardápio**")
    if st.session_state.drinks_salvos:
        cardapio_resultado = pd.DataFrame(st.session_state.drinks_salvos)
        st.dataframe(cardapio_resultado, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum drink adicionado ainda. Use o formulário acima.")

st.markdown("")

if st.button("▶ Simular", use_container_width=False):
    # ── Dados reais do banco ─────────────────────────────────────────────────
    total_registros = len(df)
    pagantes = (df["valor"] > 0).sum() if "valor" in df.columns else 0
    taxa_pagantes = pagantes / total_registros if total_registros > 0 else 0.3

    # Ajusta taxa por local
    if "bairro" in df.columns:
        plano_bairros = ["Asa Norte", "Asa Sul", "Noroeste", "Sudoeste",
                         "Cruzeiro", "Lago Norte", "Lago Sul", "Park Way"]
        df_local = df[df["bairro"].isin(plano_bairros)] if local_sim == "Plano Piloto" else df[~df["bairro"].isin(plano_bairros)]
        if len(df_local) > 10:
            taxa_pagantes = (df_local["valor"] > 0).sum() / len(df_local) if "valor" in df_local.columns else taxa_pagantes

    # Previsão do perfil de público (modelo_previsao.ModeloPublico): reaproveita
    # o público real do Evento 244 ajustado pelo preço e pela região do novo evento
    previsao = modelo_publico.prever(preco_ingresso, local_sim, capacidade)
    if preco_ingresso > 0:
        taxa_pagantes = previsao["pct_pagantes"]

    # Público e pagantes esperados
    publico_esperado = int(capacidade * 0.65)
    pagantes_esperados = int(publico_esperado * taxa_pagantes)

    # ── Receitas ─────────────────────────────────────────────────────────────
    receita_ingresso = pagantes_esperados * preco_ingresso if preco_ingresso > 0 else 0.0

    if vai_ter_bar and len(cardapio_resultado) > 0:
        preco_medio_drink = cardapio_resultado["Preço sugerido (R$)"].mean()
        custo_medio_drink = cardapio_resultado["Custo do drink (R$)"].mean()
        drinks_por_pessoa = consumo_por_pessoa  # baseado no evento real
        receita_bar = round(publico_esperado * drinks_por_pessoa * preco_medio_drink, 2)
        custo_bar_estimado = round(publico_esperado * drinks_por_pessoa * custo_medio_drink, 2)
    else:
        receita_bar = 0.0
        custo_bar_estimado = 0.0

    receita_total = receita_ingresso + receita_bar
    custo_total_estimado = orcamento + custo_bar_estimado
    lucro = receita_total - custo_total_estimado

    # Ponto de equilíbrio (só ingresso)
    ponto_equilibrio = int(orcamento / preco_ingresso) if preco_ingresso > 0 else None

    # ── Veredicto ────────────────────────────────────────────────────────────
    if preco_ingresso == 0 and not vai_ter_bar:
        viabilidade = "gratuito"
    elif receita_total >= custo_total_estimado * 1.1:
        viabilidade = "viavel"
    elif receita_total >= custo_total_estimado * 0.8:
        viabilidade = "arriscado"
    else:
        viabilidade = "inviavel"

    # ── Perfil predominante para esse local ──────────────────────────────────
    try:
        perfil_local = resumo.copy()
        perfil_local["match_local"] = perfil_local["perfil"].str.contains(
            "Plano Piloto" if local_sim == "Plano Piloto" else "Entorno"
        )
        perfil_destaque = perfil_local[perfil_local["match_local"]].sort_values("quantidade", ascending=False)
        nome_perfil = perfil_destaque.iloc[0]["perfil"] if len(perfil_destaque) > 0 else resumo.sort_values("quantidade", ascending=False).iloc[0]["perfil"]
        dias_antes = perfil_destaque.iloc[0]["dias_antes_media"] if len(perfil_destaque) > 0 else resumo["dias_antes_media"].mean()
    except Exception:
        nome_perfil = "—"
        dias_antes = 0

    # ── Resultados ───────────────────────────────────────────────────────────
    st.markdown("#### 📊 Resultado da simulação")
    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    col_r1.metric("👥 Público esperado", f"{publico_esperado} pessoas")
    col_r2.metric("🎟️ Receita ingresso", f"R$ {receita_ingresso:,.0f}".replace(",", "."),
                  delta=f"{pagantes_esperados} pagantes")
    col_r3.metric("🍺 Receita bar", f"R$ {receita_bar:,.0f}".replace(",", "."),
                  delta=f"{consumo_por_pessoa:.1f} beb/pessoa" if vai_ter_bar else "sem bar")
    col_r4.metric("💵 Receita total", f"R$ {receita_total:,.0f}".replace(",", "."))
    col_r5.metric("📊 Resultado final", f"R$ {lucro:,.0f}".replace(",", "."),
                  delta="Lucro" if lucro >= 0 else "Prejuízo",
                  delta_color="normal" if lucro >= 0 else "inverse")

    st.markdown("")

    # Breakdown de custos
    with st.expander("📋 Ver breakdown de custos"):
        st.markdown(f"""
| Item | Valor |
|------|-------|
| Orçamento do evento | R$ {orcamento:,.0f} |
| Custo estimado do bar | R$ {custo_bar_estimado:,.0f} |
| **Custo total estimado** | **R$ {custo_total_estimado:,.0f}** |
| Receita de ingresso | R$ {receita_ingresso:,.0f} |
| Receita de bar | R$ {receita_bar:,.0f} |
| **Receita total** | **R$ {receita_total:,.0f}** |
| **Resultado** | **R$ {lucro:,.0f}** |
        """.replace(",", "."))
        if vai_ter_bar:
            st.caption(f"Custo bar estimado = {publico_esperado} pessoas × R$ {custo_bar_por_pessoa:.2f}/pessoa (baseado no seu evento real)")

    st.markdown("")

    # Veredicto visual
    if viabilidade == "viavel":
        st.success(f"✅ **Orçamento viável** — Receita total estimada de R$ {receita_total:,.0f} cobre os custos com margem.".replace(",", "."))
    elif viabilidade == "arriscado":
        st.warning(f"⚠️ **Cenário arriscado** — Receita cobre entre 80% e 110% dos custos. Reveja preço do ingresso ou reduza o orçamento.")
    elif viabilidade == "inviavel":
        if ponto_equilibrio and preco_ingresso > 0:
            st.error(f"❌ **Inviável** — Precisaria de pelo menos **{ponto_equilibrio} pagantes** só para cobrir o evento (sem bar). "
                     f"Estimativa atual: {pagantes_esperados} pagantes.")
        else:
            st.error("❌ **Inviável** — Reduza o orçamento, aumente o preço ou inclua bar.")
    else:
        st.info("ℹ️ **Evento gratuito sem bar** — Orçamento precisa ser coberto por patrocínio ou cotas.")

    # Dicas
    st.markdown("")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info(f"**🎯 Perfil predominante no local:**\n\n{nome_perfil}")
    with col_d2:
        st.info(f"**📅 Timing de divulgação:**\n\nEsse perfil compra em média **{dias_antes:.0f} dias antes**. "
                f"Comece a divulgar com pelo menos {max(int(dias_antes) + 7, 14)} dias de antecedência.")

    if ponto_equilibrio and preco_ingresso > 0:
        st.caption(f"Ponto de equilíbrio (só ingresso): {ponto_equilibrio} pagantes × R$ {preco_ingresso} = R$ {orcamento:,.0f}".replace(",", "."))

    # ════════════════════════════════════════════════════════════════════════
    # 8.2 — PREVISÃO DO PERFIL DE PÚBLICO
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("👥 Previsão do perfil de público")
    if recomendacao is not None:
        st.caption(f"Evento simulado: {recomendacao['genero']} · {recomendacao['local']} · "
                   f"{recomendacao['data']:%d/%m/%Y} · {local_sim} · ingresso R$ {preco_ingresso}")
    st.caption("Parte do público real do Evento 244 e ajusta pelo que muda: quem tem perfil de pagante pesa mais "
               "quando o ingresso é pago (Regressão Logística) e quem mora na mesma região do evento pesa mais.")

    col_pp1, col_pp2, col_pp3, col_pp4 = st.columns(4)
    col_pp1.metric("🎂 Idade média prevista", f"{previsao['idade_media']:.0f} anos")
    col_pp2.metric("🎟️ Pagantes previstos", f"{previsao['pct_pagantes']:.0%}" if preco_ingresso > 0 else "Gratuito")
    col_pp3.metric("📅 Compra em média", f"{previsao['dias_antes_medio']:.0f} dias antes")
    col_pp4.metric("⏰ Compra na última semana", f"{previsao['pct_ultima_semana']:.0%}")

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        fig_pf = px.bar(previsao["faixa_etaria"], x="faixa_etaria", y="percentual",
                        color="percentual", color_continuous_scale="Blues",
                        labels={"faixa_etaria": "Faixa etária", "percentual": "% do público"})
        fig_pf.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=0, r=0, t=30, b=0),
                             title="Faixa etária")
        st.plotly_chart(fig_pf, use_container_width=True)
    with col_g2:
        fig_pg = px.pie(previsao["genero"], names="genero", values="percentual", hole=0.45,
                        color_discrete_sequence=["#378ADD", "#D4537E", "#888780"])
        fig_pg.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), title="Gênero")
        st.plotly_chart(fig_pg, use_container_width=True)
    with col_g3:
        fig_pr = px.pie(previsao["regiao"], names="regiao", values="percentual", hole=0.45,
                        color_discrete_sequence=["#378ADD", "#9FE1CB", "#888780"])
        fig_pr.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), title="Onde mora")
        st.plotly_chart(fig_pr, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Perfis (K-Means) esperados no evento**")
        st.dataframe(previsao["perfis"].rename(columns={"perfil": "Perfil", "percentual": "% do público"}),
                     use_container_width=True, hide_index=True)
    with col_t2:
        st.markdown("**Bairros que mais devem aparecer**")
        st.dataframe(previsao["bairros"].rename(columns={"bairro": "Bairro", "percentual": "% do público"}),
                     use_container_width=True, hide_index=True)

    with st.expander("🔍 Como o modelo de público funciona"):
        st.markdown(f"A Regressão Logística estima a chance de cada pessoa do Evento 244 ser pagante. "
                    f"AUC na validação cruzada: **{modelo_publico.auc:.2f}** (0,5 = chute, 1,0 = perfeito).")
        st.dataframe(modelo_publico.coeficientes(), use_container_width=True, hide_index=True)
        st.caption("Coeficiente positivo = aumenta a chance de pagar (variáveis padronizadas). "
                   "Ingressos acima de R$ 35 são extrapolação: o Evento 244 só teve lotes de R$ 15 a R$ 35.")

st.markdown("---")
st.caption("Dashboard gerado com Streamlit + Plotly | Dados: PostgreSQL DW")