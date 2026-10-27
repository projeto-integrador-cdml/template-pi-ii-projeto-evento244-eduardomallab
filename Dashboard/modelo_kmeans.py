"""
modelo_kmeans.py — Segmentação de perfis de público com K-Means
Agrupa os inscritos do evento em 4 perfis com base em:
  - Idade
  - Valor pago
  - Gênero
  - Método de pagamento
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


N_CLUSTERS = 4
RANDOM_STATE = 42


def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recebe o DataFrame com os dados dos inscritos e retorna:
    - df_ml: DataFrame limpo com a coluna 'cluster' adicionada
    - features: DataFrame com as features usadas no modelo
    """
    df_ml = df.copy()

    # Garantir tipos numéricos
    df_ml["valor"] = pd.to_numeric(df_ml["valor"], errors="coerce").fillna(0)
    df_ml["idade"] = pd.to_numeric(df_ml["idade"], errors="coerce")
    df_ml = df_ml.dropna(subset=["idade"]).reset_index(drop=True)

    # Encode gênero: Masculino=1, Feminino=0, outros=0.5
    df_ml["genero_num"] = df_ml["genero"].map({"Masculino": 1, "Feminino": 0}).fillna(0.5)

    # One-hot encoding do método de pagamento
    metodos = pd.get_dummies(df_ml["metodo_de_pagamento"], prefix="pag")

    features = pd.concat([
        df_ml[["idade", "valor", "genero_num"]].reset_index(drop=True),
        metodos.reset_index(drop=True)
    ], axis=1).fillna(0)

    return df_ml, features


def treinar_kmeans(features: pd.DataFrame) -> tuple[KMeans, np.ndarray]:
    """
    Normaliza as features e treina o modelo K-Means.
    Retorna o modelo treinado e os labels de cluster.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    modelo = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = modelo.fit_predict(X)

    return modelo, labels


def nomear_cluster(row: pd.Series) -> str:
    """Gera um nome descritivo para cada cluster com base nas médias."""
    if row["idade_media"] < 25:
        faixa = "Jovem"
    elif row["idade_media"] < 40:
        faixa = "Adulto"
    else:
        faixa = "Sênior"

    pag = "Pago" if row["valor_medio"] > 0 else "Gratuito"
    return f"Perfil {int(row['cluster']) + 1} — {faixa} / {pag}"


def gerar_resumo(df_ml: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela resumo com características de cada cluster."""
    resumo = df_ml.groupby("cluster").agg(
        quantidade=("cluster", "count"),
        idade_media=("idade", "mean"),
        valor_medio=("valor", "mean"),
        genero_predominante=("genero", lambda x: x.mode()[0] if not x.empty else "—"),
        metodo_top=("metodo_de_pagamento", lambda x: x.mode()[0] if not x.empty else "—"),
        bairro_top=("bairro", lambda x: x[x != "Não identificado"].mode()[0]
                    if not x[x != "Não identificado"].empty else "—"),
    ).reset_index()

    resumo["idade_media"] = resumo["idade_media"].round(1)
    resumo["valor_medio"] = resumo["valor_medio"].round(2)
    resumo["perfil"] = resumo.apply(nomear_cluster, axis=1)

    return resumo


def segmentar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Função principal: recebe o DataFrame dos inscritos e retorna
    - df_ml: DataFrame com coluna 'cluster' e 'perfil'
    - resumo: tabela com características de cada perfil
    """
    df_ml, features = preparar_features(df)
    _, labels = treinar_kmeans(features)

    df_ml["cluster"] = labels
    resumo = gerar_resumo(df_ml)
    df_ml["perfil"] = df_ml["cluster"].map(resumo.set_index("cluster")["perfil"])

    return df_ml, resumo