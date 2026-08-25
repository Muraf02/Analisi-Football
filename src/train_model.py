"""
Addestra il modello Dixon-Coles sui dati reali presenti nel database
(raccolti da football-data.org) e genera previsioni per le partite in
programma.

Uso:
    python src/train_model.py
"""

import sys
import os
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LEAGUES
from src.db import get_connection
from src.model_poisson import DixonColesModel

MIN_MATCHES_REQUIRED = 30  # sotto questa soglia le stime sono troppo incerte


def load_finished_matches(league_code, conn):
    """Carica dal database le partite concluse di una lega, nel formato
    richiesto da DixonColesModel.fit()."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.utc_date, m.home_score, m.away_score,
               th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ? AND m.status = 'FINISHED'
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
        ORDER BY m.utc_date
        """,
        (league_code,),
    )
    rows = cur.fetchall()

    matches = []
    for row in rows:
        date = datetime.fromisoformat(row["utc_date"].replace("Z", "+00:00"))
        matches.append({
            "home_team": row["home_name"],
            "away_team": row["away_name"],
            "home_goals": row["home_score"],
            "away_goals": row["away_score"],
            "date": date,
        })
    return matches


def load_upcoming_matches(league_code, conn, limit=10):
    """Carica le prossime partite in programma (non ancora giocate).

    Invece di elencare esplicitamente gli stati "da giocare" (che secondo
    la documentazione di football-data.org possono essere SCHEDULED,
    TIMED, LIVE, IN_PLAY, PAUSED — e la lista esatta è cambiata nel tempo),
    escludiamo solo gli stati chiaramente "conclusi o annullati". Così il
    filtro resta corretto anche se l'API aggiunge/rinomina stati intermedi.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.utc_date, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ?
              AND m.status NOT IN ('FINISHED', 'POSTPONED', 'SUSPENDED', 'CANCELED', 'CANCELLED', 'AWARDED')
        ORDER BY m.utc_date
        LIMIT ?
        """,
        (league_code, limit),
    )
    return cur.fetchall()


def train_and_predict_league(league_code, league_name, conn):
    print(f"\n{'=' * 60}")
    print(f"{league_name} ({league_code})")
    print("=" * 60)

    matches = load_finished_matches(league_code, conn)
    print(f"Partite concluse disponibili: {len(matches)}")

    if len(matches) < MIN_MATCHES_REQUIRED:
        print(
            f"ATTENZIONE: meno di {MIN_MATCHES_REQUIRED} partite disponibili. "
            "Le stime con così pochi dati sono statisticamente poco affidabili "
            "(alta incertezza) — è normale a inizio stagione. Il modello "
            "diventerà via via più preciso con l'avanzare del campionato."
        )
        if len(matches) < 20:
            print("Troppo poche partite per addestrare il modello, salto questa lega.")
            return None

    model = DixonColesModel(time_decay_half_life_days=180)
    try:
        model.fit(matches)
    except Exception as e:
        print(f"Errore nell'addestramento: {e}")
        return None

    print(f"Modello addestrato. Vantaggio casalingo stimato: {model.home_advantage:.3f}")
    print(f"Parametro di correzione rho: {model.rho:.3f}")

    upcoming = load_upcoming_matches(league_code, conn)
    if upcoming:
        print(f"\nPrevisioni per le prossime {len(upcoming)} partite in programma:")
        for m in upcoming:
            try:
                pred = model.predict_match(m["home_name"], m["away_name"])
                print(
                    f"  {m['home_name']} vs {m['away_name']} "
                    f"({m['utc_date'][:10]}): "
                    f"1={pred['home_win']:.0%} X={pred['draw']:.0%} 2={pred['away_win']:.0%} "
                    f"| gol attesi {pred['home_expected_goals']:.1f}-{pred['away_expected_goals']:.1f}"
                )
            except ValueError:
                # squadra promossa/neopromossa senza storico sufficiente
                print(f"  {m['home_name']} vs {m['away_name']}: dati insufficienti per una stima")
    else:
        print("\nNessuna partita in programma trovata nel database per questa lega.")

    return model


def run():
    conn = get_connection()
    try:
        models = {}
        for code, info in LEAGUES.items():
            model = train_and_predict_league(code, info["name"], conn)
            if model:
                models[code] = model
        return models
    finally:
        conn.close()


if __name__ == "__main__":
    run()
