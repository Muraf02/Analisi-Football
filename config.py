"""
Configurazione centrale del progetto football-analytics.
Le API key vanno messe come variabili d'ambiente (mai hardcoded nel codice).
"""

import os

# --- API KEYS (impostale come variabili d'ambiente sul tuo sistema) ---
# Windows (PowerShell):  $env:FOOTBALL_DATA_API_KEY="la_tua_chiave"
# Mac/Linux (bash):      export FOOTBALL_DATA_API_KEY="la_tua_chiave"
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")  # opzionale, per fase 3 (quote)

# --- LEGHE DA MONITORARE ---
# Codici competizione di football-data.org (v4)
LEAGUES = {
    "SA": {"name": "Serie A", "country": "Italy"},
    "PL": {"name": "Premier League", "country": "England"},
    "PD": {"name": "La Liga", "country": "Spain"},        # Primera Division
    "FL1": {"name": "Ligue 1", "country": "France"},
    "BL1": {"name": "Bundesliga", "country": "Germany"},
}

# Mappatura nomi squadre per Understat (nomi leggermente diversi, es. "Inter" -> "Inter")
# Verrà popolata/raffinata quando integriamo lo scraping xG (fase 2)
UNDERSTAT_LEAGUES = {
    "SA": "Serie_A",
    "PL": "EPL",
    "PD": "La_liga",
    "FL1": "Ligue_1",
    "BL1": "Bundesliga",
}

# --- DATABASE ---
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "football.db")

# --- API football-data.org ---
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# Rate limit piano gratuito: 10 richieste/minuto -> mettiamo un margine di sicurezza
FOOTBALL_DATA_REQUEST_DELAY_SECONDS = 7

# --- LOGGING ---
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
