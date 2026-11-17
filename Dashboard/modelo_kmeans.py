"""
modelo_kmeans.py — Segmentação de perfis de público com K-Means
Features:
  - dias_antes_evento: quantos dias antes do evento a pessoa se inscreveu
  - pago: 1 se pagou ingresso, 0 se gratuito
  - idade
  - regiao: Plano Piloto=2, Entorno próximo=1, Periferia/outros=0
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTERS = 4
RANDOM_STATE = 42
DATA_EVENTO = pd.Timestamp("2025-09-27")

REGIAO_MAP = {
    "Asa Norte": 2, "Asa Sul": 2, "Noroeste": 2, "Sudoeste": 2,
    "Cruzeiro": 2, "Lago Norte": 2, "Lago Sul": 2, "Park Way": 2,
    "Guará": 1, "Taguatinga": 1, "Águas Claras": 1, "Sobradinho": 1,
    "Ceilândia": 0, "Samambaia": 0, "Planaltina": 0,
    "Santa Maria": 0, "Gama": 0, "Recanto das Emas": 0,
}


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ml = df.copy()

    df_ml["valor"] = pd.to_numeric(df_ml["valor"], errors="coerce").fillna(0)
    df_ml["idade"] = pd.to_numeric(df_ml["idade"], errors="coerce")
    df_ml["data_compra"] = pd.to_datetime(df_ml["data_compra"], errors="coerce")

    df_ml = df_ml.dropna(subset=["idade", "data_compra"]).reset_index(drop=True)

    # Dias antes do evento (negativo = depois, mas filtramos só antes)
    df_ml["dias_antes"] = (DATA_EVENTO - df_ml["data_compra"]).dt.days
    df_ml["dias_antes"] = df_ml["dias_antes"].clip(lower=0)

    # Pago binário
    df_ml["pago"] = (df_ml["valor"] > 0).astype(int)

    # Região numérica
    if "bairro" in df_ml.columns:
        df_ml["regiao_num"] = df_ml["bairro"].map(REGIAO_MAP).fillna(0.5)
    else:
        df_ml["regiao_num"] = 0.5

    features = df_ml[["dias_antes", "pago", "idade", "regiao_num"]].copy().fillna(0)
    return df_ml, features


def treinar_kmeans(features: pd.DataFrame) -> tuple[KMeans, np.ndarray]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    modelo = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = modelo.fit_predict(X)
    return modelo, labels


def nomear_cluster(row: pd.Series) -> str:
    antecipado = row["dias_antes_media"] >= 7
    pago = row["valor_medio"] > 0
    plano = row["regiao_media"] >= 1.5

    timing = "Antecipado" if antecipado else "Última hora"
    pagamento = "Pago" if pago else "Gratuito"
    local = "Plano Piloto" if plano else "Periferia"

    return f"Perfil {int(row['cluster']) + 1} — {timing} / {pagamento} / {local}"


def gerar_resumo(df_ml: pd.DataFrame) -> pd.DataFrame:
    resumo = df_ml.groupby("cluster").agg(
        quantidade=("cluster", "count"),
        idade_media=("idade", "mean"),
        valor_medio=("valor", "mean"),
        dias_antes_media=("dias_antes", "mean"),
        regiao_media=("regiao_num", "mean"),
        genero_predominante=("genero", lambda x: x.mode()[0] if not x.empty else "—"),
        metodo_top=("metodo_de_pagamento", lambda x: x.mode()[0] if not x.empty else "—"),
        bairro_top=("bairro", lambda x: x[x != "Não identificado"].mode()[0]
                    if not x[x != "Não identificado"].empty else "—"),
    ).reset_index()

    resumo["idade_media"] = resumo["idade_media"].round(1)
    resumo["valor_medio"] = resumo["valor_medio"].round(2)
    resumo["dias_antes_media"] = resumo["dias_antes_media"].round(1)
    resumo["perfil"] = resumo.apply(nomear_cluster, axis=1)
    return resumo


def segmentar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ml, features = preparar_features(df)
    _, labels = treinar_kmeans(features)
    df_ml["cluster"] = labels
    resumo = gerar_resumo(df_ml)
    df_ml["perfil"] = df_ml["cluster"].map(resumo.set_index("cluster")["perfil"])
    return df_ml, resumo