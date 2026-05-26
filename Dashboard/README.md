# 🎵 Dashboard de Análise de Evento Musical

Dashboard interativo desenvolvido como projeto integrador da disciplina de Business Intelligence, utilizando Python e Streamlit para visualização de dados de um evento musical realizado em Brasília/DF.

---

## 📊 Visão Geral

Este projeto implementa um pipeline completo de dados, desde a coleta até a visualização:

```
CSV do Evento → Pentaho (ETL) → PostgreSQL (Schema Estrela) → Streamlit (Dashboard)
```

---

## 🗂️ Estrutura do Projeto

```
Dashboard/
├── dashboard.py       # Aplicação principal
├── .env               # Credenciais do banco (não versionado)
├── .gitignore         # Arquivos ignorados pelo Git
├── requirements.txt   # Dependências do projeto
└── README.md          # Este arquivo
```

---

## 🗄️ Schema Estrela (Data Warehouse)

O banco de dados segue o modelo estrela com as seguintes tabelas:

```
                    ┌─────────────────┐
                    │   dim_cliente   │
                    │─────────────────│
                    │ id_cliente (PK) │
                    │ genero          │
                    │ data_nascimento │
                    │ cep             │
                    └────────┬────────┘
                             │
┌──────────────┐    ┌────────▼────────┐    ┌────────────────┐
│  dim_email   │    │   fato_244      │    │ dim_metodopag  │
│──────────────│    │─────────────────│    │────────────────│
│ id_email(PK) ├────│ id_fato (PK)    ├────│ id_metodopag   │
│ email        │    │ id_cliente (FK) │    │ metodo_pagament│
└──────────────┘    │ id_email (FK)   │    └────────────────┘
                    │ id_metodopag(FK)│
                    │ data_compra     │
                    │ valor           │
                    └─────────────────┘
```

---

## 📈 Dashboards Implementados

| Gráfico | Descrição |
|---|---|
| 📍 Inscritos por bairro | Top 20 bairros do DF com mais inscritos (via ViaCEP) |
| 💳 Método de pagamento | Distribuição entre PIX, cartão, Apple Pay e gratuito |
| 🎟️ Pago vs Gratuito | Proporção de ingressos pagos e gratuitos |
| 📅 Inscrições por dia | Evolução temporal das inscrições (ago–set 2025) |
| 👥 Gênero | Distribuição por gênero dos participantes |
| 🎂 Faixa etária | Agrupamento por faixas de idade |
| 🕐 Horário das inscrições | Pico de horário das inscrições durante o dia |

---

## 🔢 Resultados do Evento

> Dados reais extraídos do Data Warehouse

- **599** inscritos no total
- **155** ingressos pagos
- **444** ingressos gratuitos
- **R$ 3.405,00** de receita gerada
- **Asa Norte** foi o bairro com mais participantes
- **18–24 anos** foi a faixa etária predominante

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3.9+ | Linguagem principal |
| Streamlit | Framework de dashboard |
| Plotly Express | Visualizações interativas |
| Pandas | Manipulação de dados |
| psycopg2 | Conexão com PostgreSQL |
| python-dotenv | Gerenciamento de variáveis de ambiente |
| ViaCEP API | Enriquecimento de dados por CEP |
| Pentaho | ETL e normalização dos dados |
| PostgreSQL | Banco de dados (Data Warehouse) |

---

## ⚙️ Como Executar

### Pré-requisitos
- Python 3.9+
- PostgreSQL rodando localmente
- pgAdmin com o banco `gold` configurado

### Passo a passo

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/Dashboard.git
cd Dashboard
```

**2. Crie e ative o ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Configure as variáveis de ambiente**

Crie um arquivo `.env` na raiz do projeto:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gold
DB_USER=postgres
DB_PASSWORD=sua_senha
```

**5. Rode o dashboard**
```bash
streamlit run dashboard.py
```

Acesse em: **http://localhost:8501**

---

## 📦 Gerando o requirements.txt

```bash
pip freeze > requirements.txt
```

---

## 👨‍💻 Autor

Desenvolvido como projeto integrador de Business Intelligence.

> Pipeline completo: coleta de dados CSV → normalização no Pentaho → modelagem estrela no PostgreSQL → visualização com Streamlit.
