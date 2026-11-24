"""
modelo_previsao.py — Previsões usadas dentro do Simulador de Evento

São duas partes:

1) Previsão de data, gênero e local
   A ideia é responder "qual combinação de data + gênero + local tem mais chance
   de dar certo?" usando os eventos de mercado que coletamos (Sympla, Shotgun etc).

   Usamos um modelo probabilístico no estilo Naive Bayes. Para cada combinação
   candidata calculamos:

       P(gênero) x P(dia da semana | gênero) x P(local | gênero)

   - P(gênero): quanto esse gênero aparece no mercado de Brasília (demanda)
   - P(dia | gênero): em quais dias da semana esse gênero costuma acontecer
   - P(local | gênero): quais casas recebem esse gênero
   Usamos suavização de Laplace pra não zerar combinações que só não apareceram
   na nossa amostra (ela é pequena).

   Depois aplicamos uma penalidade de CONCORRÊNCIA: se já tem evento agendado
   no mesmo dia, principalmente do mesmo gênero, a nota cai. E se o local já
   tem evento naquela data, a combinação é descartada.

2) Previsão do perfil de público
   A única base com dados de pessoas é a do nosso evento (Evento 244). Então a
   previsão parte desse público real e ajusta pelo que muda no novo evento:

   - Preço: treinamos uma Regressão Logística que estima a chance de cada pessoa
     ser pagante (pela idade, região, gênero e antecedência da compra). Num evento
     pago, quem tem perfil de pagante pesa mais. Quanto mais caro o ingresso,
     mais forte é esse filtro.
   - Local: quem mora na mesma região do evento (Plano Piloto ou Entorno) pesa
     mais do que quem mora longe.

   Com esses pesos a gente recalcula a cara do público: faixa etária, gênero,
   região, perfis do K-Means e quando as pessoas compram.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DIAS_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

BAIRROS_PLANO = ["Asa Norte", "Asa Sul", "Noroeste", "Sudoeste", "Cruzeiro",
                 "Lago Norte", "Lago Sul", "Park Way", "SAAN", "SCS", "Conic"]

PRECO_REFERENCIA = 15.0   # preço do ingresso do Evento 244
ALFA_LAPLACE = 1.0        # suavização: evita probabilidade zero
PENALIDADE_MESMO_GENERO = 0.6   # cada evento do mesmo gênero no dia multiplica a nota por 0,4
PENALIDADE_OUTROS = 0.15        # cada evento de outro gênero no dia tira bem menos
PESO_MESMA_REGIAO = 1.0
PESO_OUTRA_REGIAO = 0.5
PESO_REGIAO_DESCONHECIDA = 0.75


# ════════════════════════════════════════════════════════════════════════════
# Preparação dos eventos de mercado
# ════════════════════════════════════════════════════════════════════════════
def genero_macro(g) -> str:
    """Junta os subgêneros em 5 grupos. Ex.: 'Funk/Soul' -> 'Funk', 'Trap/Hip-Hop' -> 'Rap/Trap'."""
    if pd.isna(g):
        return "Outro"
    partes = [p.strip() for p in str(g).split("/") if p.strip() and p.strip() != "Festival"]
    principal = partes[0] if partes else "Outro"
    if principal in ("Rap", "Trap", "Hip-Hop"):
        return "Rap/Trap"
    return principal


# Mesmo lugar escrito de jeitos diferentes nas fontes
APELIDOS_LOCAL = {
    "Contexto Bar e Restaurante": "Contexto Bar",
    "GALPÃO 17 DF": "Galpão 17",
}


def limpar_local(local) -> str:
    """Tira o ' - Brasília, DF' que o Sympla coloca no fim do nome do local e junta nomes repetidos."""
    if pd.isna(local):
        return ""
    nome = str(local).replace(" - Brasília, DF", "").replace(", Brasília", "").strip()
    return APELIDOS_LOCAL.get(nome, nome)


def regiao_do_bairro(bairro) -> str:
    if pd.isna(bairro) or str(bairro).strip() in ("", "Não identificado"):
        return "Não informado"
    return "Plano Piloto" if str(bairro).strip() in BAIRROS_PLANO else "Entorno"


def preparar_mercado(df_mercado: pd.DataFrame, df_outros: pd.DataFrame) -> pd.DataFrame:
    """Junta as duas bases de mercado num formato só."""
    ev = pd.concat([df_mercado, df_outros], ignore_index=True)
    ev["data"] = pd.to_datetime(ev["data"], errors="coerce")
    ev = ev.dropna(subset=["data"]).copy()
    ev["genero_macro"] = ev["genero"].apply(genero_macro)
    ev["local_limpo"] = ev["local"].apply(limpar_local)
    ev["dia_semana"] = ev["data"].dt.dayofweek
    ev["regiao"] = ev["bairro"].apply(regiao_do_bairro)
    # "Brasília - DF" não é um local de verdade, é só a cidade
    ev.loc[ev["local_limpo"].isin(["Brasília - DF", "Brasília"]), "local_limpo"] = ""
    return ev


# ════════════════════════════════════════════════════════════════════════════
# PARTE 1 — Previsão de data, gênero e local
# ════════════════════════════════════════════════════════════════════════════
class ModeloOportunidade:
    """Naive Bayes de oportunidade: aprende com o mercado e dá nota pra cada combinação."""

    def __init__(self, min_eventos_local: int = 2):
        self.min_eventos_local = min_eventos_local

    def treinar(self, eventos: pd.DataFrame):
        self.eventos = eventos
        self.generos = sorted(g for g in eventos["genero_macro"].unique() if g != "Outro")
        n = len(eventos)
        k = len(self.generos)

        # P(gênero)
        cont_g = eventos["genero_macro"].value_counts()
        self.p_genero = {g: (cont_g.get(g, 0) + ALFA_LAPLACE) / (n + ALFA_LAPLACE * k) for g in self.generos}

        # P(dia da semana | gênero)
        self.p_dia = {}
        for g in self.generos:
            dias = eventos.loc[eventos["genero_macro"] == g, "dia_semana"].value_counts()
            total = dias.sum()
            self.p_dia[g] = {d: (dias.get(d, 0) + ALFA_LAPLACE) / (total + 7 * ALFA_LAPLACE) for d in range(7)}

        # Locais candidatos: só os que aparecem pelo menos min_eventos_local vezes
        com_local = eventos[eventos["local_limpo"] != ""]
        cont_local = com_local["local_limpo"].value_counts()
        self.locais = cont_local[cont_local >= self.min_eventos_local].index.tolist()
        p_local_geral = (cont_local[self.locais] / cont_local[self.locais].sum()).to_dict()

        # P(local | gênero) com suavização puxando pra popularidade geral do local
        self.p_local = {}
        self.n_local_genero = {}
        for g in self.generos:
            sub = com_local[(com_local["genero_macro"] == g) & com_local["local_limpo"].isin(self.locais)]
            cont = sub["local_limpo"].value_counts()
            total = cont.sum()
            self.n_local_genero[g] = cont.to_dict()
            self.p_local[g] = {l: (cont.get(l, 0) + ALFA_LAPLACE * p_local_geral[l]) / (total + ALFA_LAPLACE)
                               for l in self.locais}

        # Região de cada local (quando a base informa o bairro)
        reg = com_local[com_local["regiao"] != "Não informado"].groupby("local_limpo")["regiao"].agg(lambda x: x.mode()[0])
        self.regiao_local = reg.to_dict()
        return self

    def recomendar(self, data_inicio, data_fim, generos=None, data_minima=None, top=5,
                   um_por_genero=False) -> pd.DataFrame:
        """
        Testa todas as combinações (data x gênero x local) no período e devolve as melhores.
        um_por_genero=True devolve a melhor combinação de cada gênero (ranking de gêneros);
        senão devolve uma opção por local, com a melhor data de cada um.
        """
        datas = pd.date_range(pd.Timestamp(data_inicio), pd.Timestamp(data_fim), freq="D")
        if data_minima is not None:
            datas = datas[datas >= pd.Timestamp(data_minima)]
        generos = [g for g in (generos or self.generos) if g in self.generos]
        if len(datas) == 0 or not generos or not self.locais:
            return pd.DataFrame()

        ev = self.eventos
        # Concorrência: eventos que já estão marcados em cada dia
        agenda = ev[ev["data"].isin(datas)]
        mesmo_dia = agenda.groupby(["data", "genero_macro"]).size()
        total_dia = agenda.groupby("data").size()
        local_ocupado = set(zip(agenda["data"], agenda["local_limpo"]))

        linhas = []
        for g in generos:
            for d in datas:
                n_mesmo = int(mesmo_dia.get((d, g), 0))
                n_outros = int(total_dia.get(d, 0)) - n_mesmo
                fator_conc = (1 - PENALIDADE_MESMO_GENERO) ** n_mesmo * (1 - PENALIDADE_OUTROS) ** n_outros
                base = self.p_genero[g] * self.p_dia[g][d.dayofweek] * fator_conc
                for l in self.locais:
                    if (d, l) in local_ocupado:
                        continue  # o local já tem evento nesse dia
                    linhas.append({
                        "data": d, "genero": g, "local": l,
                        "score_bruto": base * self.p_local[g][l],
                        "eventos_mesmo_genero_no_dia": n_mesmo,
                        "outros_eventos_no_dia": n_outros,
                    })
        res = pd.DataFrame(linhas)
        if res.empty:
            return res
        # Nota de 0 a 100 relativa à melhor combinação encontrada
        res["nota"] = (100 * res["score_bruto"] / res["score_bruto"].max()).round(1)
        # Em caso de empate fica a data mais cedo (dá pra começar a vender antes)
        res = res.sort_values(["score_bruto", "data"], ascending=[False, True], kind="stable")
        if um_por_genero:
            res = res.drop_duplicates(subset=["genero"])
        else:
            # Uma opção por local, cada uma com a melhor data dele (fica mais útil pra comparar casas)
            res = res.drop_duplicates(subset=["local"])
        res = res.head(top).reset_index(drop=True)
        res["dia_semana"] = res["data"].dt.dayofweek.map(dict(enumerate(DIAS_SEMANA)))
        res["regiao_local"] = res["local"].map(self.regiao_local).fillna("Não informado")
        res["motivo"] = res.apply(self._explicar, axis=1)
        return res

    def _explicar(self, r) -> str:
        g, l = r["genero"], r["local"]
        pd_dia = self.p_dia[g][r["data"].dayofweek]
        n_lg = self.n_local_genero[g].get(l, 0)
        partes = [f"{r['dia_semana']} concentra {pd_dia:.0%} dos eventos de {g}"]
        if n_lg > 0:
            partes.append(f"{l} já recebeu {n_lg} evento(s) de {g}")
        else:
            partes.append(f"{l} é um dos locais mais usados do mercado")
        if r["eventos_mesmo_genero_no_dia"] == 0:
            partes.append(f"nenhum outro evento de {g} marcado nesse dia")
        else:
            partes.append(f"atenção: {r['eventos_mesmo_genero_no_dia']} evento(s) de {g} no mesmo dia")
        return " · ".join(partes)

    def probabilidades_genero(self) -> pd.DataFrame:
        return (pd.DataFrame({"genero": list(self.p_genero), "probabilidade": list(self.p_genero.values())})
                .sort_values("probabilidade", ascending=False))

    def dias_por_genero(self) -> pd.DataFrame:
        linhas = [{"genero": g, "dia": DIAS_SEMANA[d], "probabilidade": p}
                  for g in self.generos for d, p in self.p_dia[g].items()]
        return pd.DataFrame(linhas)


# ════════════════════════════════════════════════════════════════════════════
# PARTE 2 — Previsão do perfil de público
# ════════════════════════════════════════════════════════════════════════════
class ModeloPublico:
    """Estima como fica o público de um novo evento a partir do público real do Evento 244."""

    FEATURES = ["idade", "dias_antes", "plano_piloto", "feminino"]

    def treinar(self, df_clusters: pd.DataFrame):
        d = df_clusters.copy()
        d["regiao"] = d["bairro"].apply(regiao_do_bairro)
        d["plano_piloto"] = (d["regiao"] == "Plano Piloto").astype(int)
        d["feminino"] = d["genero"].astype(str).str.lower().str.startswith("f").astype(int)
        d = d.dropna(subset=["idade"]).reset_index(drop=True)
        self.base = d

        X, y = d[self.FEATURES], d["pago"]
        self.modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        # AUC por validação cruzada: mostra se o modelo separa pagante de não pagante melhor que o acaso (0,5)
        self.auc = float(cross_val_score(self.modelo, X, y, cv=5, scoring="roc_auc").mean()) if y.nunique() > 1 else float("nan")
        self.modelo.fit(X, y)
        self.base["prob_pagar"] = self.modelo.predict_proba(X)[:, 1]
        return self

    def coeficientes(self) -> pd.DataFrame:
        coef = self.modelo[-1].coef_[0]
        nomes = {"idade": "Idade", "dias_antes": "Dias de antecedência",
                 "plano_piloto": "Mora no Plano Piloto", "feminino": "Gênero feminino"}
        return pd.DataFrame({"Variável": [nomes[f] for f in self.FEATURES], "Efeito na chance de pagar": coef.round(3)})

    def prever(self, preco: float, regiao_evento: str, capacidade: int, ocupacao: float = 0.65) -> dict:
        d = self.base.copy()

        # Peso pelo preço: evento gratuito = todo mundo igual.
        # A intensidade cresce com a raiz do preço e fica limitada, porque nossos dados
        # só têm ingressos de R$ 15 a R$ 35 (acima disso é extrapolação).
        if preco > 0:
            intensidade = float(np.clip((preco / PRECO_REFERENCIA) ** 0.5, 0.5, 1.6))
            d["peso_preco"] = d["prob_pagar"] ** intensidade
        else:
            d["peso_preco"] = 1.0

        # Peso pela região: quem mora perto do evento tem mais chance de ir
        d["peso_regiao"] = np.where(d["regiao"] == "Não informado", PESO_REGIAO_DESCONHECIDA,
                                    np.where(d["regiao"] == regiao_evento, PESO_MESMA_REGIAO, PESO_OUTRA_REGIAO))
        d["peso"] = d["peso_preco"] * d["peso_regiao"]
        d["peso"] = d["peso"] / d["peso"].sum()

        def distribuicao(coluna, ordem=None):
            s = d.groupby(coluna, observed=True)["peso"].sum()
            if ordem is not None:
                s = s.reindex(ordem).fillna(0)
            return (s * 100).round(1).rename("percentual").reset_index()

        publico = int(capacidade * ocupacao)
        pct_pagantes = float((d["peso"] * d["prob_pagar"]).sum()) if preco > 0 else 0.0

        return {
            "publico_esperado": publico,
            "pagantes_esperados": int(round(publico * pct_pagantes)),
            "pct_pagantes": pct_pagantes,
            "idade_media": float((d["peso"] * d["idade"]).sum()),
            "dias_antes_medio": float((d["peso"] * d["dias_antes"]).sum()),
            "pct_ultima_semana": float(d.loc[d["dias_antes"] <= 7, "peso"].sum()),
            "faixa_etaria": distribuicao("faixa_etaria", ["< 18", "18–24", "25–34", "35–44", "45–59", "60+"]),
            "genero": distribuicao("genero"),
            "regiao": distribuicao("regiao"),
            "perfis": distribuicao("perfil").sort_values("percentual", ascending=False),
            "bairros": distribuicao("bairro").query("bairro != 'Não identificado'")
                       .sort_values("percentual", ascending=False).head(8),
            "metodo": distribuicao("metodo_de_pagamento").sort_values("percentual", ascending=False),
        }
