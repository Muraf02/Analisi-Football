"""
Recupera i dati di Expected Goals (xG) da Understat.com e li associa
alle partite già presenti nel database (raccolte da football-data.org
tramite fetch_football_data.py, che va quindi eseguito prima di questo).

Understat non ha un'API ufficiale: i dati sono incorporati nell'HTML
della pagina di ogni lega, dentro una variabile JavaScript. Li estraiamo
con una tecnica di parsing testuale (regex + decodifica).

NOTA: essendo uno scraping (non un'API ufficiale), se Understat cambia
la struttura delle sue pagine questo script potrebbe smettere di
funzionare e andrà aggiornato. Per questo lo teniamo separato dal
recupero dati principale (football-data.org), che è più affidabile.

Uso:
    python src/fetch_understat.py
"""

import sys
import os
import re
import json
import time
import logging
from datetime import datetime, timezone

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import UNDERSTAT_LEAGUES, LOG_FILE, LOG_DIR
from src.db import get_connection
from src.team_name_mapping import match_team_name

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

UNDERSTAT_BASE_URL = "https://understat.com/league"

# Oltre questa differenza oraria tra le date, consideriamo che NON sia
# la stessa partita (protegge da falsi abbinamenti)
MAX_DATE_DIFFERENCE_HOURS = 36


def _extract_json_variable(html, variable_name):
    """
    Estrae e decodifica una variabile JS del tipo:
    var datesData = JSON.parse('\\x7B...\\x7D');
    presente nelle pagine di Understat.
    """
    pattern = rf"var\s+{variable_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if not match:
        raise ValueError(
            f"Variabile '{variable_name}' non trovata nella pagina. "
            "Understat potrebbe aver cambiato la struttura del sito."
        )

    raw = match.group(1)
    decoded = raw.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    return json.loads(decoded)


def fetch_league_season(league_understat_code, season):
    """Recupera tutte le partite (con xG) di una lega/stagione da Understat."""
    url = f"{UNDERSTAT_BASE_URL}/{league_understat_code}/{season}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _extract_json_variable(resp.text, "datesData")


def _parse_understat_date(date_str):
    """Understat fornisce datetime tipo '2025-08-16 15:00:00' (UTC)."""
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def match_and_store(league_code, understat_matches, conn):
    """
    Associa le partite Understat (per nome squadra + data) alle partite
    già presenti in 'matches', e salva l'xG in 'match_stats'.
    """
    cur = conn.cursor()

    cur.execute(
        """
        SELECT m.match_id, m.utc_date, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ? AND m.status = 'FINISHED'
        """,
        (league_code,),
    )
    our_matches = cur.fetchall()

    matched = 0
    unmatched = 0

    for um in understat_matches:
        if not um.get("isResult"):
            continue  # partita non ancora giocata, xG non disponibile

        u_home = um["h"]["title"]
        u_away = um["a"]["title"]
        u_date = _parse_understat_date(um["datetime"])
        home_xg = float(um["xG"]["h"])
        away_xg = float(um["xG"]["a"])

        found = None
        for row in our_matches:
            our_date = datetime.fromisoformat(row["utc_date"].replace("Z", "+00:00"))
            if abs((our_date - u_date).total_seconds()) > MAX_DATE_DIFFERENCE_HOURS * 3600:
                continue
            if match_team_name(u_home, row["home_name"]) and match_team_name(u_away, row["away_name"]):
                found = row
                break

        if not found:
            unmatched += 1
            logger.warning(
                f"[{league_code}] Nessuna corrispondenza per "
                f"'{u_home}' vs '{u_away}' ({u_date.date()})"
            )
            continue

        cur.execute(
            """
            INSERT INTO match_stats (match_id, home_xg, away_xg, source)
            VALUES (?, ?, ?, 'understat')
            ON CONFLICT(match_id) DO UPDATE SET
                home_xg=excluded.home_xg,
                away_xg=excluded.away_xg,
                source=excluded.source
            """,
            (found["match_id"], home_xg, away_xg),
        )
        matched += 1

    conn.commit()
    logger.info(f"[{league_code}] xG abbinati: {matched}, non trovati: {unmatched}")
    return matched, unmatched


def _current_season_start_year():
    """Le stagioni europee iniziano a metà anno: da luglio in poi consideriamo
    'iniziata' la stagione dell'anno corrente, altrimenti quella precedente."""
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def run_full_update():
    conn = get_connection()
    season = _current_season_start_year()

    total_matched = 0
    total_unmatched = 0

    try:
        leagues = list(UNDERSTAT_LEAGUES.items())
        for i, (code, understat_code) in enumerate(leagues):
            print(f"Recupero xG per {code} ({understat_code}), stagione {season}...")
            try:
                data = fetch_league_season(understat_code, season)
                m, u = match_and_store(code, data, conn)
                total_matched += m
                total_unmatched += u
                print(f"  -> {m} partite abbinate, {u} non trovate")
            except Exception as e:
                logger.error(f"[{code}] Errore recupero Understat: {e}")
                print(f"  Errore per {code}: {e}")

            if i < len(leagues) - 1:
                time.sleep(3)  # cortesia verso il server, non un vero rate limit

        print(f"\nCompletato. Totale xG abbinati: {total_matched}, non trovati: {total_unmatched}")
        if total_unmatched > 0:
            print(
                "Nota: alcune partite non sono state abbinate automaticamente "
                "(probabile differenza nei nomi delle squadre). Controlla "
                "logs/pipeline.log per i dettagli — si possono aggiungere "
                "mappature manuali in src/team_name_mapping.py."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    run_full_update()
