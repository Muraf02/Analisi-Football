"""
Calcola le probabilità stimate dal modello Dixon-Coles su diversi mercati
di scommessa, non solo 1X2.

Tutti questi mercati si possono calcolare dalla stessa "matrice dei
risultati" (score_matrix) che il modello già produce — non serve un
modello diverso per ciascun mercato, basta sommare le celle giuste della
matrice.

Mercati coperti in questa fase (tutti derivabili dai gol):
- 1X2 (vittoria casa / pareggio / vittoria trasferta)
- Doppia chance (1X, X2, 12)
- Over/Under (2.5, 1.5, 3.5 gol totali)
- Both Teams To Score (Gol/NoGol)
- Risultato esatto

NON coperti (richiederebbero dati che il modello attuale non ha, es.
statistiche su cartellini o calci d'angolo):
- Mercati su cartellini, corner, tiri, falli
- Mercati sui singoli giocatori
"""

import numpy as np


def market_probabilities(score_matrix):
    """
    Data una matrice dei risultati (da DixonColesModel.score_matrix),
    restituisce le probabilità stimate su tutti i mercati coperti.
    """
    max_goals = score_matrix.shape[0] - 1

    home_win = float(np.sum(np.tril(score_matrix, -1)))
    draw = float(np.sum(np.diag(score_matrix)))
    away_win = float(np.sum(np.triu(score_matrix, 1)))

    result = {
        "1x2": {
            "1": round(home_win, 4),
            "X": round(draw, 4),
            "2": round(away_win, 4),
        },
        "double_chance": {
            "1X": round(home_win + draw, 4),
            "X2": round(draw + away_win, 4),
            "12": round(home_win + away_win, 4),
        },
        "over_under": {},
        "btts": {},
        "correct_score": {},
    }

    # Over/Under: per ogni soglia, sommiamo le celle della matrice dove
    # gol_casa + gol_trasferta supera (o no) la soglia
    for threshold in [1.5, 2.5, 3.5]:
        over = 0.0
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                if hg + ag > threshold:
                    over += score_matrix[hg][ag]
        result["over_under"][f"over_{threshold}"] = round(float(over), 4)
        result["over_under"][f"under_{threshold}"] = round(float(1 - over), 4)

    # Both Teams To Score: entrambe le squadre segnano almeno 1 gol
    btts_yes = 0.0
    for hg in range(1, max_goals + 1):
        for ag in range(1, max_goals + 1):
            btts_yes += score_matrix[hg][ag]
    result["btts"]["yes"] = round(float(btts_yes), 4)
    result["btts"]["no"] = round(float(1 - btts_yes), 4)

    # Risultato esatto: i 6 risultati più probabili, per non intasare l'output
    scores_flat = [
        ((hg, ag), score_matrix[hg][ag])
        for hg in range(max_goals + 1)
        for ag in range(max_goals + 1)
    ]
    scores_flat.sort(key=lambda x: -x[1])
    for (hg, ag), prob in scores_flat[:6]:
        result["correct_score"][f"{hg}-{ag}"] = round(float(prob), 4)

    return result


def combined_market_probabilities(score_matrix):
    """
    Calcola le probabilità di mercati COMBINATI sulla STESSA partita
    (es. "Vittoria Casa & Under 2.5 gol", quello che i bookmaker chiamano
    "bet builder" o "mercati combinati").

    A differenza delle combinazioni tra partite diverse (che assumono
    indipendenza tra gli eventi, un'approssimazione), questi valori sono
    calcolati ESATTAMENTE dalla stessa matrice dei risultati — non c'è
    nessuna assunzione di indipendenza da fare, perché derivano dalla
    stessa distribuzione di probabilità congiunta.

    NOTA: non abbiamo una fonte di quote reali per questi mercati
    specifici, quindi questi sono SOLO stime del modello — vanno mostrate
    come tali, senza calcolare un "vantaggio" verificato sul mercato.
    """
    max_goals = score_matrix.shape[0] - 1
    combos = {}

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = score_matrix[hg][ag]
            outcome_1x2 = "1" if hg > ag else ("X" if hg == ag else "2")
            total_goals = hg + ag
            btts_yes = hg >= 1 and ag >= 1

            for threshold in [1.5, 2.5, 3.5]:
                ou_key = f"over_{threshold}" if total_goals > threshold else f"under_{threshold}"
                key = f"{outcome_1x2}_{ou_key}"
                combos[key] = combos.get(key, 0.0) + p

            btts_key = f"{outcome_1x2}_btts_{'si' if btts_yes else 'no'}"
            combos[btts_key] = combos.get(btts_key, 0.0) + p

    return {k: round(v, 4) for k, v in combos.items()}


def all_market_outcomes(score_matrix):
    """
    Restituisce una lista piatta di TUTTI gli esiti calcolati, ciascuno
    con la sua probabilità — comoda per lo step successivo (cercare
    esiti la cui quota "equa" cade in un certo range).

    Formato di ogni elemento:
        {'market': '1x2', 'outcome': '1', 'probability': 0.55}
    """
    markets = market_probabilities(score_matrix)
    flat = []
    for market_name, outcomes in markets.items():
        for outcome_name, prob in outcomes.items():
            flat.append({
                "market": market_name,
                "outcome": outcome_name,
                "probability": prob,
                "fair_odd": round(1 / prob, 3) if prob > 0 else None,
            })
    return flat
