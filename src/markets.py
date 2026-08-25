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
