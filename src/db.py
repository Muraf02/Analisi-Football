"""
Gestione del database SQLite: schema e funzioni di accesso base.
SQLite è sufficiente per questo progetto (dati limitati, uso single-user).
"""

import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH


SCHEMA = """
-- Squadre
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,          -- id fornito da football-data.org
    name TEXT NOT NULL,
    short_name TEXT,
    league_code TEXT NOT NULL,
    UNIQUE(team_id)
);

-- Partite (risultati + calendario)
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,         -- id fornito da football-data.org
    league_code TEXT NOT NULL,
    season INTEGER NOT NULL,              -- anno di inizio stagione, es. 2025 per 2025/26
    matchday INTEGER,
    utc_date TEXT NOT NULL,               -- ISO datetime UTC
    status TEXT NOT NULL,                 -- SCHEDULED, LIVE, FINISHED, POSTPONED, ecc.
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_score INTEGER,                   -- NULL se non ancora giocata
    away_score INTEGER,
    home_score_ht INTEGER,                -- risultato primo tempo (se disponibile)
    away_score_ht INTEGER,
    winner TEXT,                          -- HOME_TEAM, AWAY_TEAM, DRAW, NULL
    last_updated TEXT,                    -- timestamp del nostro ultimo fetch
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- Statistiche avanzate per partita (xG, xGA, tiri, ecc.) - popolata in fase 2 (Understat)
CREATE TABLE IF NOT EXISTS match_stats (
    match_id INTEGER NOT NULL,
    home_xg REAL,
    away_xg REAL,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_possession REAL,
    away_possession REAL,
    source TEXT,                          -- es. 'understat'
    PRIMARY KEY (match_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- Quote bookmaker - popolata in fase 3 (The Odds API)
CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,                 -- es. '1X2', 'OVER_UNDER_2.5'
    outcome TEXT NOT NULL,                -- es. 'HOME', 'DRAW', 'AWAY', 'OVER', 'UNDER'
    odd_value REAL NOT NULL,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- Log delle esecuzioni della pipeline (per debug e monitoraggio aggiornamenti)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,               -- 'fetch_matches', 'fetch_stats', 'fetch_odds'
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,                          -- 'success', 'error'
    records_updated INTEGER,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches(league_code, season);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(utc_date);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
"""


def get_connection():
    """Restituisce una connessione al database, creando il file se non esiste."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inizializza lo schema del database. Sicuro da rilanciare (usa IF NOT EXISTS)."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        print(f"Database inizializzato correttamente in: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
