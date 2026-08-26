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
import time
from datetime import datetime, timezone, timedelta
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LEAGUES
from src.db import get_connection
from src.model_poisson import DixonColesModel
from src.markets import market_probabilities, combined_market_probabilities
from src.train_model import load_finished_matches, MIN_MATCHES_REQUIRED
from src.strategy_engine import simulate_strategy

STARTING_CAPITAL = float(os.environ.get("STARTING_CAPITAL", 20))
TARGET_CAPITAL = float(os.environ.get("TARGET_CAPITAL", 100))
ODDS_MIN = float(os.environ.get("ODDS_MIN", 1.5))
ODDS_MAX = float(os.environ.get("ODDS_MAX", 1.8))
N_LEGS = int(os.environ.get("N_LEGS", 2))  # quante partite combinare (1 = solo singole)

# Mercati per cui abbiamo quote reali (fetch_odds.py li richiede tutti)
REAL_ODDS_MARKETS = {"1x2", "over_under", "btts", "double_chance"}


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
    """
    Restituisce solo le partite della PROSSIMA giornata di campionato
    (non tutte le partite future disponibili), usando il numero di
    giornata (matchday) che football-data.org fornisce per ogni partita.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.match_id, m.utc_date, m.matchday, th.name as home_name, ta.name as away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.league_code = ?
              AND m.status NOT IN ('FINISHED', 'POSTPONED', 'SUSPENDED', 'CANCELED', 'CANCELLED', 'AWARDED')
        ORDER BY m.utc_date
        """,
        (league_code,),
    )
    rows = cur.fetchall()
    if not rows:
        return []

    matchdays = [r["matchday"] for r in rows if r["matchday"] is not None]
    if matchdays:
        next_matchday = min(matchdays)
        return [r for r in rows if r["matchday"] == next_matchday]

    # Fallback se il numero di giornata non è disponibile: prendiamo solo
    # le partite entro 7 giorni dalla più vicina (approssima la stessa idea)
    first_date = datetime.fromisoformat(rows[0]["utc_date"].replace("Z", "+00:00"))
    cutoff = first_date + timedelta(days=7)
    return [
        r for r in rows
        if datetime.fromisoformat(r["utc_date"].replace("Z", "+00:00")) <= cutoff
    ]


def recent_form(matches, team, before_date, n=5):
    """
    Restituisce la forma recente di una squadra come stringa (es. 'VVNPV',
    dalla più vecchia alla più recente), basata sulle ultime n partite
    PRIMA della data indicata.
    """
    team_matches = [
        m for m in matches
        if (m["home_team"] == team or m["away_team"] == team) and m["date"] < before_date
    ]
    team_matches.sort(key=lambda m: m["date"], reverse=True)
    recent = list(reversed(team_matches[:n]))  # dalla più vecchia alla più recente

    form = []
    for m in recent:
        if m["home_team"] == team:
            gf, ga = m["home_goals"], m["away_goals"]
        else:
            gf, ga = m["away_goals"], m["home_goals"]
        if gf > ga:
            form.append("V")
        elif gf == ga:
            form.append("N")
        else:
            form.append("P")
    return "".join(form) if form else "n/d"


def head_to_head_summary(matches, home_team, away_team, before_date, n=3):
    """Restituisce un riassunto testuale degli ultimi scontri diretti."""
    h2h = [
        m for m in matches
        if {m["home_team"], m["away_team"]} == {home_team, away_team} and m["date"] < before_date
    ]
    h2h.sort(key=lambda m: m["date"], reverse=True)
    h2h = h2h[:n]

    if not h2h:
        return "Nessuno scontro diretto recente disponibile nei dati storici."

    parts = []
    for m in h2h:
        parts.append(f"{m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']} ({m['date'].strftime('%Y-%m-%d')})")
    return " · ".join(parts)


def days_since_last_match(matches, team, before_date):
    """Giorni trascorsi dall'ultima partita giocata da una squadra
    (segnale di possibile stanchezza/vantaggio di riposo)."""
    team_matches = [
        m for m in matches
        if (m["home_team"] == team or m["away_team"] == team) and m["date"] < before_date
    ]
    if not team_matches:
        return None
    last_date = max(m["date"] for m in team_matches)
    return (before_date - last_date).days


def _team_appearance_counts(matches):
    """Conta quante partite storiche compaiono per ciascuna squadra —
    usato per il segnale di affidabilità (una neopromossa con poche
    partite giocate ha stime molto più incerte)."""
    from collections import Counter
    counts = Counter()
    for m in matches:
        counts[m["home_team"]] += 1
        counts[m["away_team"]] += 1
    return counts


def reliability_label(n_matches):
    """Etichetta di affidabilità in base a quante partite storiche
    abbiamo per una squadra. Soglie indicative, non scientifiche, ma
    utili come segnale d'allarme onesto."""
    if n_matches < 10:
        return "bassa", "pochissime partite storiche (probabile neopromossa o inizio stagione)"
    elif n_matches < 20:
        return "media", "un numero limitato di partite storiche"
    else:
        return "alta", "uno storico sufficiente"


def explain_candidate(c):
    """Genera una spiegazione testuale in linguaggio semplice per una
    proposta, così non resta solo un numero senza contesto."""
    home_rel, home_desc = reliability_label(c["home_team_matches"])
    away_rel, away_desc = reliability_label(c["away_team_matches"])
    overall_rel = "bassa" if "bassa" in (home_rel, away_rel) else (
        "media" if "media" in (home_rel, away_rel) else "alta"
    )

    parts = []
    parts.append(
        f"Il modello stima una probabilità del {c['model_probability']:.1%} per questo esito, "
        f"basandosi su {c['home_team_matches']} partite storiche della squadra di casa e "
        f"{c['away_team_matches']} della squadra in trasferta."
    )
    parts.append(
        f"La quota reale ({c['real_odd']}) implica invece una probabilità del "
        f"{c['implied_probability']:.1%} secondo il mercato — una differenza di "
        f"{c['edge']*100:+.1f} punti percentuali."
    )

    if c.get("home_form") or c.get("away_form"):
        parts.append(
            f"Forma recente (dalla più vecchia alla più recente): "
            f"casa {c.get('home_form', 'n/d')}, trasferta {c.get('away_form', 'n/d')} "
            f"(V=vittoria, N=pareggio, P=sconfitta)."
        )

    if c.get("head_to_head") and "Nessuno" not in c["head_to_head"]:
        parts.append(f"Scontri diretti recenti: {c['head_to_head']}.")

    rest_parts = []
    if c.get("home_rest_days") is not None:
        rest_parts.append(f"casa {c['home_rest_days']} giorni")
    if c.get("away_rest_days") is not None:
        rest_parts.append(f"trasferta {c['away_rest_days']} giorni")
    if rest_parts:
        rest_text = " e ".join(rest_parts)
        parts.append(f"Riposo dall'ultima partita: {rest_text}.")
        min_rest = min(
            [d for d in [c.get("home_rest_days"), c.get("away_rest_days")] if d is not None],
            default=None,
        )
        if min_rest is not None and min_rest <= 3:
            parts.append(
                "⚠️ Nota: una delle due squadre ha giocato pochissimi giorni fa — possibile "
                "affaticamento non pienamente catturato dal modello statistico."
            )

    if c.get("top_combined_markets"):
        combo_texts = []
        for item in c["top_combined_markets"]:
            combo_texts.append(f"{item['market']} ({item['probability']:.1%})")
        parts.append(
            "Mercati combinati sulla stessa partita più probabili secondo il modello "
            f"(stima, non verificata su una quota reale): {', '.join(combo_texts)}."
        )

    if overall_rel == "bassa":
        parts.append(
            f"⚠️ Affidabilità BASSA: almeno una delle due squadre ha {home_desc if home_rel=='bassa' else away_desc}. "
            "Un vantaggio stimato in questa condizione va preso con molta cautela — potrebbe riflettere "
            "un limite del modello più che un'opportunità reale."
        )
    elif overall_rel == "media":
        parts.append(
            "⚠️ Affidabilità MEDIA: lo storico disponibile è limitato, la stima potrebbe migliorare "
            "con l'avanzare della stagione."
        )
    else:
        parts.append("✓ Affidabilità ALTA: entrambe le squadre hanno uno storico sufficiente per una stima solida.")

    return " ".join(parts), overall_rel


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
        t_league_start = time.time()

        matches = load_finished_matches(code, conn)
        t_load = time.time()

        if len(matches) < MIN_MATCHES_REQUIRED:
            print(f"[{info['name']}] Dati storici insufficienti, salto questa lega.")
            continue

        model = DixonColesModel(time_decay_half_life_days=180)
        try:
            model.fit(matches)
        except Exception as e:
            print(f"[{info['name']}] Errore addestramento modello: {e}")
            continue
        t_fit = time.time()

        team_counts = _team_appearance_counts(matches)
        upcoming = get_upcoming_matches_with_info(code, conn)
        t_upcoming = time.time()

        print(
            f"[TIMING] {info['name']}: {len(matches)} partite storiche, "
            f"{len(set([m['home_team'] for m in matches] + [m['away_team'] for m in matches]))} squadre diverse, "
            f"{len(upcoming)} partite in programma | "
            f"caricamento={t_load-t_league_start:.2f}s addestramento={t_fit-t_load:.2f}s "
            f"prossima_giornata={t_upcoming-t_fit:.2f}s | "
            f"convergenza={model.fit_diagnostics['converged']} "
            f"iterazioni={model.fit_diagnostics['iterations']}"
        )

        for m in upcoming:
            real_odds_for_match = real_odds_by_match.get(m["match_id"])
            if not real_odds_for_match:
                continue  # nessuna quota reale disponibile per questa partita

            try:
                score_matrix = model.score_matrix(m["home_name"], m["away_name"])
            except ValueError:
                continue  # squadra senza storico sufficiente (es. neopromossa)

            markets = market_probabilities(score_matrix)
            combined_markets = combined_market_probabilities(score_matrix)
            # I 3 mercati combinati più probabili secondo il modello (solo
            # stima, nessuna quota reale disponibile per confrontarli)
            top_combined = sorted(combined_markets.items(), key=lambda x: -x[1])[:3]

            match_date = datetime.fromisoformat(m["utc_date"].replace("Z", "+00:00"))

            # Contesto calcolato UNA VOLTA per partita (non per ogni candidato)
            context = {
                "home_form": recent_form(matches, m["home_name"], match_date),
                "away_form": recent_form(matches, m["away_name"], match_date),
                "head_to_head": head_to_head_summary(matches, m["home_name"], m["away_name"], match_date),
                "home_rest_days": days_since_last_match(matches, m["home_name"], match_date),
                "away_rest_days": days_since_last_match(matches, m["away_name"], match_date),
                "top_combined_markets": [
                    {"market": k, "probability": v} for k, v in top_combined
                ],
            }

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

                candidate = {
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
                    "home_team_matches": team_counts.get(m["home_name"], 0),
                    "away_team_matches": team_counts.get(m["away_name"], 0),
                    **context,
                }
                explanation, reliability = explain_candidate(candidate)
                candidate["explanation"] = explanation
                candidate["reliability"] = reliability
                all_candidates.append(candidate)

    return all_candidates


def find_singles_in_range(candidates, odds_min, odds_max):
    """Candidati singoli la cui quota reale cade nel range desiderato,
    ordinati dal miglior vantaggio stimato al peggiore."""
    in_range = [c for c in candidates if odds_min <= c["real_odd"] <= odds_max]
    return sorted(in_range, key=lambda c: -c["edge"])


def find_combos_in_range(candidates, odds_min, odds_max, n_legs=2, max_results=5,
                          min_edge=0.0, pool_size=60, exclude_low_reliability=True):
    """
    Cerca combinazioni di N partite DIVERSE la cui quota combinata
    (prodotto delle singole quote) cade nel range desiderato.

    IMPORTANTE — come scegliamo quali candidati considerare: non basta
    prendere i candidati con il vantaggio migliore in assoluto — se hanno
    tutti quote troppo alte (o troppo basse) rispetto all'obiettivo,
    nessuna loro combinazione potrà mai cadere nel range desiderato
    (es. range 2.5-3.0 con 2 gambe: combinare due quote da 2.7 ciascuna dà
    una quota finale di ~7.3, troppo alta). Filtriamo quindi PRIMA ai
    candidati con una quota individuale matematicamente compatibile con
    l'obiettivo (vicina alla radice N-esima del range), e SOLO dopo
    scegliamo tra questi quelli con il vantaggio migliore.

    Anche qui: controllare TUTTE le combinazioni possibili tra centinaia
    di esiti è computazionalmente impossibile già con 4-5 gambe. Restringiamo
    quindi la ricerca ai `pool_size` candidati compatibili con il vantaggio
    migliore (escludendo di default quelli con affidabilità bassa).
    """
    good_candidates = [c for c in candidates if c["edge"] > min_edge]
    if exclude_low_reliability:
        good_candidates = [c for c in good_candidates if c.get("reliability") != "bassa"]

    if n_legs <= 1:
        pool = sorted(good_candidates, key=lambda c: -c["edge"])[:pool_size]
    else:
        # Quota "ideale" per singola gamba: la radice n_legs-esima del range,
        # con un margine di tolleranza (le gambe non devono avere per forza
        # tutte la stessa quota, basta che il PRODOTTO cada nel range)
        ideal_low = odds_min ** (1 / n_legs)
        ideal_high = odds_max ** (1 / n_legs)
        tol_low = max(1.01, ideal_low / 1.7)
        tol_high = ideal_high * 1.7

        compatible = [c for c in good_candidates if tol_low <= c["real_odd"] <= tol_high]
        compatible.sort(key=lambda c: -c["edge"])
        pool = compatible[:pool_size]

        # Fallback: se troppo pochi candidati compatibili, allarghiamo
        # includendo anche altri (meglio provare che restituire nulla)
        if len(pool) < n_legs * 2:
            used_ids = {id(c) for c in pool}
            extra = sorted(
                [c for c in good_candidates if id(c) not in used_ids],
                key=lambda c: -c["edge"],
            )
            pool = pool + extra[:max(0, pool_size - len(pool))]

        print(
            f"[TIMING][DEBUG combo] n_legs={n_legs} range={odds_min}-{odds_max} "
            f"quota_ideale_per_gamba={ideal_low:.2f}-{ideal_high:.2f} "
            f"(tolleranza {tol_low:.2f}-{tol_high:.2f}) -> "
            f"{len(compatible)} candidati compatibili, pool finale di {len(pool)}"
        )

    all_combos = []
    for combo in combinations(pool, n_legs):
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

        in_range = odds_min <= combined_odd <= odds_max
        distance = 0.0 if in_range else min(abs(combined_odd - odds_min), abs(combined_odd - odds_max))

        all_combos.append({
            "legs": combo,
            "n_legs": n_legs,
            "combined_odd": round(combined_odd, 3),
            "combined_true_probability": round(combined_true_prob, 4),
            "combined_implied_probability": round(combined_implied_prob, 4),
            "combined_edge": round(combined_true_prob - combined_implied_prob, 4),
            "in_range": in_range,
            "distance_from_range": round(distance, 3),
        })

    results = [c for c in all_combos if c["in_range"]]
    results.sort(key=lambda r: -r["combined_edge"])

    near_misses = []
    if not results:
        # Nessuna combinazione esatta trovata: mostriamo le combinazioni
        # REALI più vicine al range desiderato (non inventate), cosi'
        # l'utente sa se conviene allargare il range invece di restare al buio
        candidates_sorted = sorted(all_combos, key=lambda r: (r["distance_from_range"], -r["combined_edge"]))
        near_misses = candidates_sorted[:3]

    return results[:max_results], near_misses


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

    Mostra SOLO quello che l'utente ha richiesto (N_LEGS): se N_LEGS=1,
    sono singole; se N_LEGS>1, sono combinazioni — non entrambe insieme.
    """
    results, near_misses = find_combos_in_range(candidates, ODDS_MIN, ODDS_MAX, n_legs=N_LEGS)

    results_out = []
    for d in results:
        sim = simulate_strategy(
            STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"],
            d["combined_true_probability"], stake_fraction=1.0, n_simulations=10000,
        )
        progression = winning_progression(STARTING_CAPITAL, TARGET_CAPITAL, d["combined_odd"])
        results_out.append({
            "legs": list(d["legs"]),
            "n_legs": d["n_legs"],
            "combined_odd": d["combined_odd"],
            "combined_true_probability": d["combined_true_probability"],
            "combined_implied_probability": d["combined_implied_probability"],
            "combined_edge": d["combined_edge"],
            "simulation": sim,
            "progression": progression,
        })

    near_misses_out = [
        {
            "legs": list(nm["legs"]),
            "combined_odd": nm["combined_odd"],
            "combined_edge": nm["combined_edge"],
        }
        for nm in near_misses
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "starting_capital": STARTING_CAPITAL,
            "target_capital": TARGET_CAPITAL,
            "odds_min": ODDS_MIN,
            "odds_max": ODDS_MAX,
            "n_legs": N_LEGS,
        },
        "total_candidates": len(candidates),
        "positive_edge_count": len([c for c in candidates if c["edge"] > 0]),
        "results": results_out,
        "near_misses": near_misses_out,
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

    request_label = (
        f"Singola nel range di quota {ODDS_MIN}-{ODDS_MAX}"
        if N_LEGS == 1
        else f"Combinazione di {N_LEGS} partite con quota combinata {ODDS_MIN}-{ODDS_MAX}"
    )
    print("\n" + "-" * 100)
    print(f"LA TUA RICHIESTA — {request_label}")
    print("-" * 100)

    doubles, near_misses = find_combos_in_range(candidates, ODDS_MIN, ODDS_MAX, n_legs=N_LEGS)

    if not doubles:
        if N_LEGS == 1:
            print(f"Nessuna singola trovata con quota reale tra {ODDS_MIN} e {ODDS_MAX} al momento.")
        else:
            print(
                f"Nessuna combinazione da {N_LEGS} partite con vantaggio positivo trovata "
                f"con quota combinata esattamente tra {ODDS_MIN} e {ODDS_MAX} al momento."
            )
        if near_misses:
            print(f"\nLe combinazioni REALI più vicine al tuo range (per riferimento, non nel range richiesto):")
            for nm in near_misses:
                legs_desc = " + ".join(f"{c['match']} @ {c['real_odd']}" for c in nm["legs"])
                edge_label = f"+{nm['combined_edge']:.1%}" if nm["combined_edge"] > 0 else f"{nm['combined_edge']:.1%}"
                print(f"  {legs_desc} -> quota combinata {nm['combined_odd']} (vantaggio {edge_label})")
            print(
                "\nSuggerimento: se vuoi risultati in questo momento, prova ad allargare "
                "il range di quota, oppure un numero diverso di partite da combinare."
            )
    else:
        for d in doubles:
            legs_desc = " + ".join(
                f"{c['match']} ({c['market']} {c['outcome']} @ {c['real_odd']}, affidabilità {c['reliability']})"
                for c in d["legs"]
            )
            edge_label = f"+{d['combined_edge']:.1%}" if d["combined_edge"] > 0 else f"{d['combined_edge']:.1%}"
            print(f"  {legs_desc}")
            for c in d["legs"]:
                print(f"      · {c['match']}: {c['explanation']}")
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
                f"      -> Se vinci SUBITO questa combinazione: punti {first['stake']}€ -> "
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
        t0 = time.time()
        candidates = build_candidates(conn)
        t1 = time.time()
        print(f"\n[TIMING] build_candidates totale: {t1-t0:.2f}s ({len(candidates)} candidati)")

        print_report(candidates)
        t2 = time.time()
        print(f"[TIMING] print_report (ricerca combinazioni + simulazioni): {t2-t1:.2f}s")

        save_json_output(candidates)
        t3 = time.time()
        print(f"[TIMING] save_json_output: {t3-t2:.2f}s")
        print(f"[TIMING] TOTALE: {t3-t0:.2f}s")
    finally:
        conn.close()


if __name__ == "__main__":
    run()

        conn.close()


if __name__ == "__main__":
    run()
