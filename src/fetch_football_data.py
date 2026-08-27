"""
Raccoglie squadre, calendario e risultati da football-data.org (v4)
per le leghe configurate in config.py, e li salva nel database SQLite.

Uso:
    python src/fetch_football_data.py

Richiede la variabile d'ambiente FOOTBALL_DATA_API_KEY impostata.
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_REQUEST_DELAY_SECONDS,
    LEAGUES,
    LOG_FILE,
    LOG_DIR,
)
from src.db import get_connection, init_db

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _headers():
    if not FOOTBALL_DATA_API_KEY:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY non impostata. "
            "Registrati gratuitamente su https://www.football-data.org/client/register "
            "e imposta la variabile d'ambiente prima di eseguire lo script."
        )
    return {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


def _get(url, params=None):
    """Wrapper per le chiamate GET con gestione errori e rate limiting."""
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    if resp.status_code == 429:
        logger.warning("Rate limit raggiunto, attendo 60 secondi...")
        time.sleep(60)
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_teams(league_code, conn, season=None):
    """Recupera e salva le squadre di una lega (di una specifica stagione,
    se indicata — utile per includere anche squadre poi retrocesse)."""
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{league_code}/teams"
    params = {}
    if season:
        params["season"] = season
    data = _get(url, params=params)
    teams = data.get("teams", [])

    cur = conn.cursor()
    for t in teams:
        cur.execute(
            """
            INSERT INTO teams (team_id, name, short_name, league_code, crest_url)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                name=excluded.name,
                short_name=excluded.short_name,
                league_code=excluded.league_code,
                crest_url=excluded.crest_url
            """,
            (t["id"], t["name"], t.get("shortName", t["name"]), league_code, t.get("crest")),
        )
    conn.commit()
    logger.info(f"[{league_code}] {len(teams)} squadre salvate/aggiornate")
    return len(teams)


def fetch_matches(league_code, conn, season=None):
    """Recupera calendario + risultati di una lega (stagione corrente se non specificata)."""
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{league_code}/matches"
    params = {}
    if season:
        params["season"] = season

    data = _get(url, params=params)
    matches = data.get("matches", [])

    cur = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()

    for m in matches:
        score = m.get("score", {})
        full_time = score.get("fullTime", {})
        half_time = score.get("halfTime", {})

        cur.execute(
            """
            INSERT INTO matches (
                match_id, league_code, season, matchday, utc_date, status,
                home_team_id, away_team_id, home_score, away_score,
                home_score_ht, away_score_ht, winner, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                status=excluded.status,
                home_score=excluded.home_score,
                away_score=excluded.away_score,
                home_score_ht=excluded.home_score_ht,
                away_score_ht=excluded.away_score_ht,
                winner=excluded.winner,
                last_updated=excluded.last_updated
            """,
            (
                m["id"],
                league_code,
                m["season"]["startDate"][:4],
                m.get("matchday"),
                m["utcDate"],
                m["status"],
                m["homeTeam"]["id"],
                m["awayTeam"]["id"],
                full_time.get("home"),
                full_time.get("away"),
                half_time.get("home"),
                half_time.get("away"),
                score.get("winner"),
                now_iso,
            ),
        )
    conn.commit()
    logger.info(f"[{league_code}] {len(matches)} partite salvate/aggiornate")
    return len(matches)


def run_full_update():
    """Esegue l'aggiornamento completo per tutte le leghe configurate."""
    init_db()
    conn = get_connection()
    total_matches = 0

    try:
        for i, (code, info) in enumerate(LEAGUES.items()):
            logger.info(f"Aggiornamento {info['name']} ({code})...")
            print(f"Aggiornamento {info['name']}...")

            fetch_teams(code, conn)
            time.sleep(FOOTBALL_DATA_REQUEST_DELAY_SECONDS)

            n = fetch_matches(code, conn)
            total_matches += n

            # Rispettiamo il rate limit del piano gratuito tra una lega e l'altra
            if i < len(LEAGUES) - 1:
                time.sleep(FOOTBALL_DATA_REQUEST_DELAY_SECONDS)

        print(f"\nCompletato. Totale partite aggiornate: {total_matches}")
        logger.info(f"Aggiornamento completo terminato. Totale partite: {total_matches}")
    except Exception as e:
        logger.error(f"Errore durante l'aggiornamento: {e}")
        print(f"Errore: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_full_update()
