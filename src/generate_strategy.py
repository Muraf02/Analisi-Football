"""
Il pezzo che mette insieme tutto: confronta le probabilità stimate dal
modello Dixon-Coles con le quote VERE dei bookmaker (già scaricate da
fetch_odds.py), trova dove il modello vede un vantaggio genuino, e
genera proposte di scommessa (singole e combinate) che rientrano nel
range di quota desiderato — con il rischio calcolato onestamente tramite
simulazione.

Parametri (impostabili come variabili d'ambiente, o modificabili qui
sotto per un test rapido):
    STARTING_CAPITAL   es. 20
    TARGET_CAPITAL     es. 100
    ODDS_MIN           es. 1.5
    ODDS_MAX           es. 1.8

Uso:
    python src/generate_strategy.py
"""

import sys
import os
from datetime import datetime, timezone
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LEAGUES
from src.db import get_connection
from src.model_poisson import DixonColesModel
from src.markets import market_probabilities
from src.train_model import load_finished_matches, MIN_MATCHES_REQUIRED
from src.strategy_engine import simulate_strategy

STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", 20))
TARGET_CAPITAL = float(os.environ.get("TARGET_CAPITAL", 100))
ODDS_MIN = float(os.environ.get("ODDS_MIN", 1.5))
ODDS_MAX = float(os.environ.get("ODDS_MAX", 1.8))

# Solo questi due mercati hanno quote reali scaricate (fetch_odds.py
# richiede solo 'h2h' e 'totals' per risparmiare crediti API)
REAL_ODDS_MARKETS = {"1x2", "over_under"}


def get_best_real_odds(conn):
    """
    Per ogni partita/mercato/esito, prende la quota MIGLIORE (più alta)
    tra tutti i bookmaker disponibili — quella che otterresti davvero
    se controllassi più siti prima di puntare.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT match_id, market, outcome,
               MAX(odd_value) as best_odd,
               COUNT(DISTINCT bookmaker) as n_bookmakers
        FROM odds
        GROUP BY match_id, market, outcome
        """
    )
    result = {}
    for row in cur.fetchall():
        result.setdefault(row["match_id"], []).append({
            "market": row["market"],
            "outcome": row["outcome"],
            "best_odd": row["best_odd"],
            "n_bookmakers": row["n_bookmakers"],
        })
    return result


def get_upcoming_matches_with_info(league_code, conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.match_id, m.utc_date, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ?
              AND m.status NOT IN ('FINISHED', 'POSTPONED', 'SUSPENDED', 'CANCELED', 'CANCELLED', 'AWARDED')
        ORDER BY m.utc_date
        """,
        (league_code,),
    )
    return cur.fetchall()


def build_candidates(conn):
    """
    Per ogni lega, allena il modello e confronta le sue probabilità con
    le quote reali disponibili. Restituisce una lista di "candidati":
    ogni singolo esito (partita + mercato + esito) per cui abbiamo sia
    una stima del modello sia una quota reale, con il relativo vantaggio
    stimato (edge = probabilità modello - probabilità implicita nella
    quota reale).
    """
    real_odds_by_match = get_best_real_odds(conn)
    all_candidates = []

    for code, info in LEAGUES.items():
        matches = load_finished_matches(code, conn)
        if len(matches) < MIN_MATCHES_REQUIRED:
            print(f"[{info['name']}] Dati storici insufficienti, salto questa lega.")
            continue

        model = DixonColesModel(time_decay_half_life_days=180)
        try:
            model.fit(matches)
        except Exception as e:
            print(f"[{info['name']}] Errore addestramento modello: {e}")
            continue

        upcoming = get_upcoming_matches_with_info(code, conn)

        for m in upcoming:
            real_odds_for_match = real_odds_by_match.get(m["match_id"])
            if not real_odds_for_match:
                continue  # nessuna quota reale disponibile per questa partita

            try:
                score_matrix = model.score_matrix(m["home_name"], m["away_name"])
            except ValueError:
                continue  # squadra senza storico sufficiente (es. neopromossa)

            markets = market_probabilities(score_matrix)

            for real_odd_entry in real_odds_for_match:
                market_name = real_odd_entry["market"]
                outcome_name = real_odd_entry["outcome"]
                if market_name not in REAL_ODDS_MARKETS:
                    continue
                if market_name not in markets or outcome_name not in markets[market_name]:
                    continue

                model_prob = markets[market_name][outcome_name]
                real_odd = real_odd_entry["best_odd"]
                implied_prob = 1 / real_odd
                edge = model_prob - implied_prob

                all_candidates.append({
                    "league": info["name"],
                    "match": f"{m['home_name']} vs {m['away_name']}",
                    "date": m["utc_date"][:10],
                    "market": market_name,
                    "outcome": outcome_name,
                    "model_probability": round(model_prob, 4),
                    "real_odd": real_odd,
                    "n_bookmakers": real_odd_entry["n_bookmakers"],
                    "implied_probability": round(implied_prob, 4),
                    "edge": round(edge, 4),
                })

    return all_candidates


def find_singles_in_range(candidates, odds_min, odds_max):
    """Candidati singoli la cui quota reale cade nel range desiderato,
    ordinati dal miglior vantaggio stimato al peggiore."""
    in_range = [c for c in candidates if odds_min <= c["real_odd"] <= odds_max]
    return sorted(in_range, key=lambda c: -c["edge"])


def find_combos_in_range(candidates, odds_min, odds_max, n_legs=2, max_results=5, min_edge=0.0):
    """
    Cerca combinazioni di N partite DIVERSE la cui quota combinata
    (prodotto delle singole quote) cade nel range desiderato.
    Ordinate per vantaggio combinato stimato.
    """
    # Solo candidati con un minimo di vantaggio hanno senso da combinare
    good_candidates = [c for c in candidates if c["edge"] > min_edge]

    results = []
    for combo in combinations(good_candidates, n_legs):
        matches_involved = {c["match"] for c in combo}
        if len(matches_involved) < n_legs:
            continue  # evitiamo di combinare due esiti della STESSA partita

        combined_odd = 1.0
        combined_true_prob = 1.0
        combined_implied_prob = 1.0
        for c in combo:
            combined_odd *= c["real_odd"]
            combined_true_prob *= c["model_probability"]
            combined_implied_prob *= c["implied_probability"]

        if not (odds_min <= combined_odd <= odds_max):
            continue

        results.append({
            "legs": combo,
            "combined_odd": round(combined_odd, 3),
            "combined_true_probability": round(combined_true_prob, 4),
            "combined_implied_probability": round(combined_implied_prob, 4),
            "combined_edge": round(combined_true_prob - combined_implied_prob, 4),
        })

    results.sort(key=lambda r: -r["combined_edge"])
    return results[:max_results]


def winning_progression(starting_capital, target_capital, odd, stake_fraction=1.0, max_steps=15):
    """
    Calcola il percorso concreto SE VINCI SEMPRE (il caso migliore possibile,
    non una previsione) — quanto punti, quanto vinci, quanto capitale hai
    dopo ogni vittoria, e quanto ti manca all'obiettivo.

    Non è una previsione di quello che succederà (per quello serve
    simulate_strategy, che tiene conto anche delle sconfitte) — è una
    fotografia concreta e comprensibile di "cosa vince ogni singolo passo".
    """
    steps = []
    capital = starting_capital

    for step_num in range(1, max_steps + 1):
        if capital >= target_capital:
            break
        stake = round(capital * stake_fraction, 2)
        winnings = round(stake * (odd - 1), 2)
        capital_after = round(capital + winnings, 2)
        remaining = round(max(target_capital - capital_after, 0), 2)

        steps.append({
            "step": step_num,
            "capital_before": round(capital, 2),
            "stake": stake,
            "winnings": winnings,
            "capital_after": capital_after,
            "remaining_to_target": remaining,
            "target_reached": capital_after >= target_capital,
        })

        capital = capital_after
        if capital >= target_capital:
            break

    return steps


def build_json_output(candidates):
    """
    Prepara gli stessi risultati del report testuale, ma in formato JSON,
    così la pagina web (docs/index.html) può leggerli e mostrarli con
    un'interfaccia curata invece del testo grezzo dei log.
    """
    singles = find_singles_in_range(candidates, ODDS_MIN, ODDS_MAX)[:5]
    doubles = find_combos_in_range(candidates, ODDS_MIN, ODDS_MAX, n_legs=2)

    singles_out = []
    for c in singles:
        sim = simulate_strategy(
            STARTING_CAPITAL, TARGET_CAPITAL, c["real_odd"],
            c["model_probability"], stake_fraction=1.0, n_simulations=10000,
        )
        progression = winning_progression(STARTING_CAPITAL, TARGET_CAPITAL, c["real_odd"])
        singles_out.append({**c, "simulation": sim, "progression": progression})

    doubles_out = []
    for d in doubles:
        sim = simulate_strategy(
            STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"],
            d["combined_true_probability"], stake_fraction=1.0, n_simulations=10000,
        )
        progression = winning_progression(STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"])
        doubles_out.append({
            "legs": list(d["legs"]),
            "combined_odd": d["combined_odd"],
            "combined_true_probability": d["combined_true_probability"],
            "combined_implied_probability": d["combined_implied_probability"],
            "combined_edge": d["combined_edge"],
            "simulation": sim,
            "progression": progression,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "starting_capital": STARTING_CAPITAL,
            "target_capital": TARGET_CAPITAL,
            "odds_min": ODDS_MIN,
            "odds_max": ODDS_MAX,
        },
        "total_candidates": len(candidates),
        "positive_edge_count": len([c for c in candidates if c["edge"] > 0]),
        "singles": singles_out,
        "doubles": doubles_out,
    }


def save_json_output(candidates, path="docs/strategy.json"):
    import json
    output = build_json_output(candidates)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nRisultati salvati anche in formato web: {path}")


def print_report(candidates):
    print("=" * 100)
    print(f"STRATEGIA: {STARTING_CAPITAL}€ -> {TARGET_CAPITAL}€, range quota {ODDS_MIN}-{ODDS_MAX}")
    print("=" * 100)

    print(f"\nTotale esiti confrontati (modello vs quote reali): {len(candidates)}")
    positive_edge = [c for c in candidates if c["edge"] > 0]
    print(f"Di cui con vantaggio stimato positivo: {len(positive_edge)}")

    print("\n" + "-" * 100)
    print("OPZIONE A — Singole nel tuo range di quota")
    print("-" * 100)
    singles = find_singles_in_range(candidates, ODDS_MIN, ODDS_MAX)
    if not singles:
        print(f"Nessuna singola trovata con quota reale tra {ODDS_MIN} e {ODDS_MAX} al momento.")
    else:
        for c in singles[:5]:
            edge_label = f"+{c['edge']:.1%}" if c["edge"] > 0 else f"{c['edge']:.1%}"
            print(
                f"  [{c['league']}] {c['match']} ({c['date']}) — {c['market']} {c['outcome']} "
                f"@ {c['real_odd']} ({c['n_bookmakers']} bookmaker) | "
                f"modello: {c['model_probability']:.1%} vs mercato: {c['implied_probability']:.1%} "
                f"| vantaggio: {edge_label}"
            )
            sim = simulate_strategy(
                STARTING_CAPITAL, TARGET_CAPITAL, c["real_odd"],
                c["model_probability"], stake_fraction=1.0, n_simulations=10000,
            )
            print(
                f"      -> Con questa quota ripetuta: successo {sim['probability_reach_target']:.1%}, "
                f"rovina {sim['probability_bust']:.1%}"
            )
            progression = winning_progression(STARTING_CAPITAL, TARGET_CAPITAL, c["real_odd"])
            print(f"      -> Se vinci SUBITO questa scommessa (puntando tutto il capitale):")
            first = progression[0]
            print(
                f"         Punti {first['stake']}€ -> se vinci, incassi {first['winnings']}€ di vincita, "
                f"capitale diventa {first['capital_after']}€ (mancano {first['remaining_to_target']}€ all'obiettivo)"
            )
            if len(progression) > 1:
                print(
                    f"         Se vincessi SEMPRE (caso migliore, non una previsione): "
                    f"obiettivo raggiunto in {len(progression)} vittorie consecutive"
                )

    print("\n" + "-" * 100)
    print("OPZIONE B — Doppie che combinano due partite diverse nel tuo range")
    print("-" * 100)
    doubles = find_combos_in_range(candidates, ODDS_MIN, ODDS_MAX, n_legs=2)
    if not doubles:
        print(f"Nessuna doppia trovata con quota combinata tra {ODDS_MIN} e {ODDS_MAX} al momento.")
    else:
        for d in doubles:
            legs_desc = " + ".join(
                f"{c['match']} ({c['market']} {c['outcome']} @ {c['real_odd']})" for c in d["legs"]
            )
            edge_label = f"+{d['combined_edge']:.1%}" if d["combined_edge"] > 0 else f"{d['combined_edge']:.1%}"
            print(f"  {legs_desc}")
            print(
                f"      Quota combinata: {d['combined_odd']} | "
                f"modello: {d['combined_true_probability']:.1%} vs mercato: "
                f"{d['combined_implied_probability']:.1%} | vantaggio: {edge_label}"
            )
            sim = simulate_strategy(
                STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"],
                d["combined_true_probability"], stake_fraction=1.0, n_simulations=10000,
            )
            print(
                f"      -> Con questa combinazione ripetuta: successo {sim['probability_reach_target']:.1%}, "
                f"rovina {sim['probability_bust']:.1%}"
            )
            progression = winning_progression(STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"])
            first = progression[0]
            print(
                f"      -> Se vinci SUBITO questa doppia: punti {first['stake']}€ -> "
                f"incassi {first['winnings']}€, capitale diventa {first['capital_after']}€ "
                f"(mancano {first['remaining_to_target']}€ all'obiettivo)"
            )

    print("\n" + "=" * 100)
    print(
        "NOTA IMPORTANTE: un vantaggio stimato positivo significa che il modello "
        "vede quella probabilità come più alta di quanto implichi la quota — non "
        "è una garanzia di vincita. Il modello può sbagliare. Usa queste stime "
        "come UN input tra tanti nella tua decisione, non come una certezza."
    )


def run():
    conn = get_connection()
    try:
        candidates = build_candidates(conn)
        print_report(candidates)
        save_json_output(candidates)
    finally:
        conn.close()


if __name__ == "__main__":
    run()
