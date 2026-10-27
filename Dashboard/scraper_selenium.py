"""
scraper_selenium.py — Raspagem de eventos de Pagode, Sertanejo e Rock no Sympla Brasília
Usa Selenium para renderizar o JavaScript da página.
Salva os resultados em: eventos_outros_generos_brasilia.csv

Estrutura real do card Sympla (confirmada via inspeção):
  <a class="sympla-card ...">
    <h3 class="...">Nome do evento</h3>
    <p class="...">Local - Brasília, DF</p>
    <div class="...">Sábado, 17 de Out às 16:00</div>
  </a>
"""

import csv
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ── Configuração ──────────────────────────────────────────────────────────────

GENEROS = {
    "pagode":    "Pagode",
    "sertanejo": "Sertanejo",
    "rock":      "Rock",
}

ANOS_VALIDOS = {"2025", "2026"}

CAMPOS = [
    "nome_evento", "data", "horario_inicio", "horario_fim",
    "local", "endereco", "bairro", "genero", "line_up",
    "ingresso_preco", "gratuito", "status", "fonte"
]

REGIOES_DF = [
    "Asa Norte", "Asa Sul", "Lago Sul", "Lago Norte", "Ceilândia",
    "Taguatinga", "Samambaia", "Gama", "Planaltina", "Sobradinho",
    "Santa Maria", "Guará", "Cruzeiro", "Noroeste", "Sudoeste",
    "Águas Claras", "Park Way", "SAAN", "SCS", "Conic",
    "Riacho Fundo", "Vicente Pires", "Candangolândia",
]

MESES = {
    "jan": "01", "fev": "02", "mar": "03", "abr": "04",
    "mai": "05", "jun": "06", "jul": "07", "ago": "08",
    "set": "09", "out": "10", "nov": "11", "dez": "12",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def criar_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def extrair_bairro(texto: str) -> str:
    for r in REGIOES_DF:
        if r.lower() in texto.lower():
            return r
    return ""


def parse_data_hora(texto: str):
    """
    Converte texto como 'Sábado, 17 de Out às 16:00' em (data_iso, horario).
    Retorna (data_iso, horario) onde data_iso é YYYY-MM-DD e horario é HH:MM.
    """
    if not texto:
        return "", ""

    texto = texto.strip()

    # Extrair horário (HH:MM)
    hora_match = re.search(r"(\d{1,2}:\d{2})", texto)
    horario = hora_match.group(1) if hora_match else ""

    # Extrair data: "17 de Out" ou "17 Out"
    m = re.search(r"(\d{1,2})\s+de\s+([a-zA-Zç]+)", texto, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{1,2})\s+([a-zA-Zç]{3,})", texto, re.IGNORECASE)

    if m:
        dia = m.group(1).zfill(2)
        mes_nome = m.group(2).lower()[:3]
        mes = MESES.get(mes_nome, "")
        if not mes:
            return "", horario

        # Determinar ano: se já passou, provavelmente é do próximo ano
        hoje = datetime.today()
        ano_tentativa = hoje.year
        try:
            data_ev = datetime(ano_tentativa, int(mes), int(dia))
            # Se a data já passou por mais de 30 dias, assume próximo ano
            if (data_ev.date() - hoje.date()).days < -30:
                ano_tentativa += 1
        except ValueError:
            pass

        data_iso = f"{ano_tentativa}-{mes}-{dia}"
        return data_iso, horario

    return "", horario


def inferir_status(data_iso: str) -> str:
    if not data_iso:
        return "Encerrado"
    try:
        data_evento = datetime.strptime(data_iso, "%Y-%m-%d").date()
        return "Agendado" if data_evento >= datetime.today().date() else "Encerrado"
    except Exception:
        return "Encerrado"


# ── Scraping principal ────────────────────────────────────────────────────────

def raspar_sympla(driver, termo: str, genero_label: str, max_paginas: int = 5) -> list:
    eventos = []
    base_url = f"https://www.sympla.com.br/eventos/brasilia-df?s={termo}"
    vistos = set()

    for pagina in range(1, max_paginas + 1):
        url = base_url if pagina == 1 else f"{base_url}&page={pagina}"
        print(f"\n🔍 [{genero_label}] Página {pagina}: {url}")

        driver.get(url)
        time.sleep(4)  # aguarda JS renderizar

        # Scroll para ativar lazy-load
        for _ in range(4):
            driver.execute_script("window.scrollBy(0, 800)")
            time.sleep(0.8)
        time.sleep(1)

        # Selector confirmado via inspeção real da página
        cards = driver.find_elements(By.CSS_SELECTOR, "a[href*='/evento/']")
        print(f"  📦 {len(cards)} cards encontrados")

        novos_nesta_pagina = 0

        for card in cards:
            try:
                href = card.get_attribute("href") or ""
                if not href or "/evento/" not in href:
                    continue

                # Evitar duplicatas por URL
                if href in vistos:
                    continue
                vistos.add(href)

                # Extrair texto completo do card
                texto_card = card.text.strip()
                if not texto_card or len(texto_card) < 5:
                    continue

                linhas = [l.strip() for l in texto_card.split("\n") if l.strip()]
                if not linhas:
                    continue

                # Linha 0: nome do evento
                nome = linhas[0][:200]

                # Linha 1: local (ex: "ESKENTA BAR - Brasília, DF")
                local_texto = linhas[1] if len(linhas) > 1 else ""

                # Linha 2+: data/hora
                texto_data = " ".join(linhas[2:]) if len(linhas) > 2 else ""

                # Parse data e horário
                data_iso, horario = parse_data_hora(texto_data)

                # Filtro por ano
                ano_ev = data_iso[:4] if data_iso else ""
                if ano_ev and ano_ev not in ANOS_VALIDOS:
                    print(f"     ⏭️  Ignorado (ano {ano_ev}): {nome[:50]}")
                    continue

                # Extrair bairro do local
                bairro = extrair_bairro(local_texto)

                ev = {
                    "nome_evento":    nome,
                    "data":           data_iso,
                    "horario_inicio": horario,
                    "horario_fim":    "",
                    "local":          local_texto[:200],
                    "endereco":       local_texto[:200],
                    "bairro":         bairro,
                    "genero":         genero_label,
                    "line_up":        "",
                    "ingresso_preco": "",
                    "gratuito":       "Não",
                    "status":         inferir_status(data_iso),
                    "fonte":          "Sympla",
                }
                eventos.append(ev)
                novos_nesta_pagina += 1
                print(f"     ✅ {nome[:60]} | {data_iso} | {local_texto[:40]}")

            except Exception as e:
                print(f"     ⚠️  Erro no card: {e}")
                continue

        if novos_nesta_pagina == 0:
            print(f"  ⚠️  Nenhum evento novo na página {pagina}, encerrando paginação.")
            break

        print(f"  ✅ {novos_nesta_pagina} eventos novos nesta página")

    return eventos


# ── Salvar CSV ────────────────────────────────────────────────────────────────

def salvar_csv(eventos: list, arquivo: str):
    with open(arquivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)
        writer.writeheader()
        for ev in eventos:
            writer.writerow({k: ev.get(k, "") for k in CAMPOS})
    print(f"\n✅ {len(eventos)} eventos salvos em: {arquivo}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Scraper Sympla — Pagode / Sertanejo / Rock em Brasília")
    print("  Filtrando apenas eventos de 2025 e 2026")
    print("=" * 60)

    driver = criar_driver()
    todos = []

    try:
        for termo, label in GENEROS.items():
            print(f"\n{'='*40}")
            print(f"  Gênero: {label}")
            print(f"{'='*40}")
            eventos = raspar_sympla(driver, termo, label, max_paginas=5)
            todos.extend(eventos)
    finally:
        driver.quit()

    # Deduplicar por nome + data
    vistos = set()
    unicos = []
    for ev in todos:
        chave = (ev["nome_evento"].lower().strip(), ev["data"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(ev)

    print(f"\n{'='*60}")
    print(f"  Total coletado: {len(todos)}")
    print(f"  Após deduplicação: {len(unicos)}")
    print(f"{'='*60}")

    salvar_csv(unicos, "eventos_outros_generos_brasilia.csv")
    print("\nPronto! Execute:")
    print("  cat eventos_outros_generos_brasilia.csv | head -5")


if __name__ == "__main__":
    main()