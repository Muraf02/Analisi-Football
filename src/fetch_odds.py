"""
Recupera le quote reali dei bookmaker (mercati 1X2 e Over/Under) da
The Odds API per le prossime partite delle 5 leghe monitorate, e le
salva nel database.

IMPORTANTE: richiede una API key da https://the-odds-api.com (CON i
trattini nell'indirizzo — non confondere con theoddsapi.com, un servizio
diverso il cui piano gratuito non include il calcio).

Il piano gratuito di the-odds-api.com da' 500 crediti al mese. Ogni
chiamata a questo script consuma all'incirca:
    (numero di leghe) x (numero di mercati richiesti) crediti
cioè circa 5 leghe x 2 mercati = 10 crediti a esecuzione, quindi il
piano gratuito basta per circa 50 esecuzioni al mese — più che
sufficiente per un uso manuale (non automatico ogni 6 ore).

Uso:
    python src/fetch_odds.py
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    ODDS_API_KEY,
    ODDS_API_LEAGUES,
    ODDS_API_BASE_URL,
    ODDS_API_REGIONS,
    ODDS_API_MARKETS,
    LOG_FILE,
    LOG_DIR,
)
from src.db import get_connection

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _check_api_key():
    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY non impostata. Registrati gratuitamente su "
            "https://the-odds-api.com (CON i trattini nell'indirizzo) "
            "e imposta la variabile d'ambiente prima di eseguire lo script."
        )


def fetch_league_odds(odds_api_league_key):
    """Recupera le quote per tutte le partite in programma di una lega."""
    url = f"{ODDS_API_BASE_URL}/sports/{odds_api_league_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_API_REGIONS,
        "markets": ODDS_API_MARKETS,
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        logger.info(f"Crediti rimanenti sul piano The Odds API: {remaining}")
        print(f"  Crediti rimanenti questo mese: {remaining}")

    return resp.json()


def match_odds_to_database(league_code, events, conn):
    """
    Associa gli eventi restituiti da The Odds API alle partite già
    presenti nel database (per nome squadra + data), e salva le quote.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.match_id, m.utc_date, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ?
        """,
        (league_code,),
    )
    our_matches = cur.fetchall()

    matched = 0
    unmatched = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for event in events:
        event_date = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        home_team = event["home_team"]
        away_team = event["away_team"]

        found = None
        for row in our_matches:
            our_date = datetime.fromisoformat(row["utc_date"].replace("Z", "+00:00"))
            if abs((our_date - event_date).total_seconds()) > 24 * 3600:
                continue
            # Confronto semplice: The Odds API di solito usa nomi molto
            # simili a quelli ufficiali. Se in futuro emergono disallineamenti,
            # possiamo riusare la stessa logica di team_name_mapping.py
            if home_team.lower() in row["home_name"].lower() or row["home_name"].lower() in home_team.lower():
                if away_team.lower() in row["away_name"].lower() or row["away_name"].lower() in away_team.lower():
                    found = row
                    break

        if not found:
            unmatched += 1
            logger.warning(f"[{league_code}] Nessuna corrispondenza per {home_team} vs {away_team}")
            continue

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                market_key = market["key"]  # 'h2h' o 'totals'
                for outcome in market.get("outcomes", []):
                    # Normalizziamo il nome dell'esito in un formato semplice
                    if market_key == "h2h":
                        if outcome["name"] == home_team:
                            outcome_label = "1"
                        elif outcome["name"] == away_team:
                            outcome_label = "2"
                        else:
                            outcome_label = "X"
                        market_label = "1x2"
                    elif market_key == "totals":
                        point = outcome.get("point", "")
                        outcome_label = f"{outcome['name'].lower()}_{point}"
                        market_label = "over_under"
                    else:
                        continue

                    cur.execute(
                        """
                        INSERT INTO odds (match_id, bookmaker, market, outcome, odd_value, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (found["match_id"], bookmaker_name, market_label, outcome_label,
                         outcome["price"], now_iso),
                    )
        matched += 1

    conn.commit()
    logger.info(f"[{league_code}] Quote abbinate: {matched}, non trovate: {unmatched}")
    return matched, unmatched


def run():
    _check_api_key()
    conn = get_connection()
    total_matched = 0

    try:
        leagues = list(ODDS_API_LEAGUES.items())
        for i, (code, odds_key) in enumerate(leagues):
            print(f"Recupero quote per {code}...")
            try:
                events = fetch_league_odds(odds_key)
                m, u = match_odds_to_database(code, events, conn)
                total_matched += m
                print(f"  -> {m} partite con quote salvate, {u} non abbinate")
            except Exception as e:
                logger.error(f"[{code}] Errore recupero quote: {e}")
                print(f"  Errore: {e}")

            if i < len(leagues) - 1:
                time.sleep(2)

        print(f"\nCompletato. Totale partite con quote salvate: {total_matched}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
