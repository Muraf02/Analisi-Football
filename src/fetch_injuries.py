"""
Recupera infortuni e squalifiche da API-Football per le prossime partite
delle 5 leghe monitorate, e li salva nel database.

IMPORTANTE: richiede una API key gratuita da https://www.api-football.com
(registrazione gratuita, 100 richieste/giorno incluse, nessuna carta
richiesta). Il piano gratuito include già l'endpoint infortuni.

Come funziona: prima cerchiamo le partite in programma su API-Football
(che ha i SUOI ID interni, diversi dai nostri), le abbiniamo alle nostre
partite per nome squadra + data, poi per ciascuna partita abbinata
chiediamo la lista infortuni/squalifiche.

Consumo stimato: 1 chiamata per recuperare le partite di ciascuna lega
(5 chiamate) + 1 chiamata per ciascuna partita con infortuni disponibili
(tipicamente 20-40 per un aggiornamento completo) — ben dentro il limite
gratuito di 100/giorno per un uso manuale.

Uso:
    python src/fetch_injuries.py
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    API_FOOTBALL_KEY,
    API_FOOTBALL_BASE_URL,
    API_FOOTBALL_LEAGUE_IDS,
    LEAGUES,
    LOG_FILE,
    LOG_DIR,
)
from src.db import get_connection, init_db
from src.team_name_mapping import match_team_name

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}


def _check_api_key():
    if not API_FOOTBALL_KEY:
        raise RuntimeError(
            "API_FOOTBALL_KEY non impostata. Registrati gratuitamente su "
            "https://www.api-football.com e imposta la variabile d'ambiente "
            "prima di eseguire lo script."
        )


def _current_season_year():
    """API-Football etichetta la stagione con l'anno di inizio (es. 2026
    per la stagione 2026/27), come football-data.org."""
    now = datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def _get(endpoint, params):
    url = f"{API_FOOTBALL_BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    remaining = resp.headers.get("x-ratelimit-requests-remaining")
    if remaining is not None:
        logger.info(f"Richieste rimanenti oggi su API-Football: {remaining}")

    if data.get("errors"):
        logger.warning(f"API-Football ha segnalato un errore: {data['errors']}")

    return data


def fetch_upcoming_fixtures(api_league_id, season):
    """Recupera le prossime partite in programma per una lega (ID API-Football)."""
    data = _get("fixtures", {"league": api_league_id, "season": season, "next": 15})
    return data.get("response", [])


def fetch_injuries_for_fixture(fixture_id):
    """Recupera infortuni/squalifiche per una specifica partita."""
    data = _get("injuries", {"fixture": fixture_id})
    return data.get("response", [])


def match_fixtures_to_our_matches(league_code, fixtures, conn):
    """
    Abbina le partite di API-Football (per nome squadra + data) alle
    nostre partite già presenti nel database, restituendo un dizionario
    {nostro_match_id: fixture_id_di_api_football}.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.match_id, m.utc_date, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ?
              AND m.status NOT IN ('FINISHED', 'POSTPONED', 'SUSPENDED', 'CANCELED', 'CANCELLED', 'AWARDED')
        """,
        (league_code,),
    )
    our_matches = cur.fetchall()

    mapping = {}
    for fx in fixtures:
        fx_date = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
        home_team = fx["teams"]["home"]["name"]
        away_team = fx["teams"]["away"]["name"]

        for row in our_matches:
            our_date = datetime.fromisoformat(row["utc_date"].replace("Z", "+00:00"))
            if abs((our_date - fx_date).total_seconds()) > 24 * 3600:
                continue
            if match_team_name(home_team, row["home_name"]) and match_team_name(away_team, row["away_name"]):
                mapping[row["match_id"]] = fx["fixture"]["id"]
                break

    return mapping


def save_injuries(match_id, injuries_data, conn):
    cur = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Rimuoviamo i dati vecchi per questa partita prima di salvare quelli nuovi
    # (gli infortuni cambiano nel tempo, non vogliamo accumulare doppioni)
    cur.execute("DELETE FROM injuries WHERE match_id = ?", (match_id,))

    for entry in injuries_data:
        player = entry.get("player", {})
        team = entry.get("team", {})
        cur.execute(
            """
            INSERT INTO injuries (match_id, team_name, player_name, injury_type, reason, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                team.get("name", "Sconosciuta"),
                player.get("name", "Sconosciuto"),
                player.get("type", entry.get("type")),
                player.get("reason", entry.get("reason")),
                now_iso,
            ),
        )
    conn.commit()


def run():
    _check_api_key()
    init_db()
    conn = get_connection()
    season = _current_season_year()
    total_matches_with_injuries = 0
    total_players = 0

    try:
        leagues = list(API_FOOTBALL_LEAGUE_IDS.items())
        for i, (code, api_league_id) in enumerate(leagues):
            league_name = LEAGUES[code]["name"]
            print(f"Recupero infortuni per {league_name}...")

            try:
                fixtures = fetch_upcoming_fixtures(api_league_id, season)
                print(f"  -> {len(fixtures)} partite in programma trovate su API-Football")

                mapping = match_fixtures_to_our_matches(code, fixtures, conn)
                print(f"  -> {len(mapping)} abbinate alle nostre partite")

                for our_match_id, fixture_id in mapping.items():
                    time.sleep(1)  # cortesia verso il server
                    injuries = fetch_injuries_for_fixture(fixture_id)
                    if injuries:
                        save_injuries(our_match_id, injuries, conn)
                        total_matches_with_injuries += 1
                        total_players += len(injuries)

            except Exception as e:
                logger.error(f"[{code}] Errore recupero infortuni: {e}")
                print(f"  Errore: {e}")

            if i < len(leagues) - 1:
                time.sleep(1)

        print(
            f"\nCompletato. Partite con infortuni/squalifiche trovate: "
            f"{total_matches_with_injuries}, giocatori totali: {total_players}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    run()
