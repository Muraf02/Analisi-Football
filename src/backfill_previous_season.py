"""
Scarica UNA TANTUM i dati della stagione precedente (già conclusa) per
ciascuna lega, cosi' il modello ha una base storica sufficiente per fare
previsioni affidabili anche a inizio della nuova stagione.

A differenza di fetch_football_data.py (che gira ogni 6 ore per i dati
correnti), questo script va eseguito manualmente una volta sola — la
stagione passata non cambia più.

Uso:
    python src/backfill_previous_season.py
"""

import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LEAGUES, FOOTBALL_DATA_REQUEST_DELAY_SECONDS
from src.db import get_connection, init_db
from src.fetch_football_data import fetch_teams, fetch_matches


def previous_season_start_year():
    """Le stagioni europee iniziano a metà anno: restituisce l'anno di
    inizio della stagione PRECEDENTE a quella in corso."""
    now = datetime.now()
    current_season = now.year if now.month >= 7 else now.year - 1
    return current_season - 1


def run():
    init_db()
    conn = get_connection()
    season = previous_season_start_year()

    print(f"Scarico lo storico della stagione {season}/{season+1} (una tantum)...")
    total = 0

    try:
        leagues = list(LEAGUES.items())
        for i, (code, info) in enumerate(leagues):
            print(f"\n{info['name']} ({code})...")
            try:
                fetch_teams(code, conn, season=season)
                time.sleep(FOOTBALL_DATA_REQUEST_DELAY_SECONDS)

                n = fetch_matches(code, conn, season=season)
                print(f"  -> {n} partite della stagione {season}/{season+1} salvate")
                total += n
            except Exception as e:
                print(f"  Errore per {code}: {e}")

            if i < len(leagues) - 1:
                time.sleep(FOOTBALL_DATA_REQUEST_DELAY_SECONDS)

        print(f"\nCompletato. Totale partite storiche aggiunte: {total}")
        print(
            "\nQuesto script non deve girare automaticamente ogni 6 ore "
            "(la stagione passata non cambia) — puoi rilanciarlo manualmente "
            "in futuro solo se vuoi aggiungere un'altra stagione storica."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    run()
