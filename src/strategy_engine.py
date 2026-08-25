"""
Motore di simulazione del rischio: dato un capitale di partenza, un
obiettivo, e una quota (o combinazione di quote), simula migliaia di
scenari possibili per stimare onestamente:
- la probabilità di raggiungere l'obiettivo
- la probabilità di andare in rovina (perdere il capitale)
- quante puntate servono in media

Perché la simulazione (Monte Carlo) invece di una formula unica: il
risultato di una sequenza di scommesse dipende da eventi casuali (vinci
o perdi ogni singola giocata), quindi non esiste "il" risultato — esiste
una DISTRIBUZIONE di risultati possibili. Simulare migliaia di volte
questa sequenza casuale è il modo standard in statistica per stimare
quella distribuzione quando non c'è una formula chiusa semplice
(specialmente con puntate proporzionali al capitale, che rendono la
matematica esatta molto più complessa di un semplice "lancio di moneta").
"""

import numpy as np


def simulate_strategy(
    starting_capital,
    target_capital,
    odd,
    true_win_probability,
    stake_fraction=1.0,
    max_bets=200,
    n_simulations=20000,
    min_stake=1.0,
):
    """
    Simula una strategia "ripeti la stessa giocata finché non raggiungi
    l'obiettivo o vai in rovina".

    Parametri:
        starting_capital: capitale di partenza (es. 20)
        target_capital: obiettivo (es. 100)
        odd: quota decimale della giocata (es. 1.65)
        true_win_probability: probabilità REALE stimata di vincere questa
            giocata (può differire da 1/odd se il modello vede un
            vantaggio o svantaggio rispetto al bookmaker)
        stake_fraction: quale frazione del capitale ATTUALE puntare ad
            ogni giocata. 1.0 = punti tutto ogni volta (rischio massimo,
            raggiungi l'obiettivo più in fretta se vinci, ma UNA sola
            sconfitta ti azzera). Valori più bassi (es. 0.3) sopravvivono
            meglio alle sconfitte ma richiedono più vittorie consecutive.
        max_bets: numero massimo di giocate per simulazione, oltre il
            quale la simulazione si ferma comunque (evita loop infiniti)
        n_simulations: quante sequenze casuali indipendenti simulare
            (più alto = stima più precisa, ma più lento)
        min_stake: sotto questa cifra si considera "rovina" (non ha più
            senso continuare a puntare)

    Restituisce un dizionario con le statistiche aggregate.
    """
    reached_target_count = 0
    busted_count = 0
    bets_to_reach_target = []
    final_capitals = []

    for _ in range(n_simulations):
        capital = starting_capital
        bets_placed = 0
        reached = False
        busted = False

        while bets_placed < max_bets:
            if capital >= target_capital:
                reached = True
                break

            stake = capital * stake_fraction
            if stake < min_stake:
                busted = True
                break

            bets_placed += 1
            win = np.random.random() < true_win_probability

            if win:
                capital = capital - stake + stake * odd
            else:
                capital -= stake

            if capital < min_stake:
                busted = True
                break

        final_capitals.append(capital)
        if reached:
            reached_target_count += 1
            bets_to_reach_target.append(bets_placed)
        if busted:
            busted_count += 1

    final_capitals = np.array(final_capitals)

    return {
        "probability_reach_target": round(reached_target_count / n_simulations, 4),
        "probability_bust": round(busted_count / n_simulations, 4),
        "probability_stuck_in_between": round(
            1 - (reached_target_count + busted_count) / n_simulations, 4
        ),
        "avg_bets_if_successful": (
            round(float(np.mean(bets_to_reach_target)), 1)
            if bets_to_reach_target else None
        ),
        "median_final_capital": round(float(np.median(final_capitals)), 2),
        "mean_final_capital": round(float(np.mean(final_capitals)), 2),
    }


def required_leg_odd(combined_odd, n_legs):
    """
    Se voglio ottenere una quota combinata `combined_odd` unendo `n_legs`
    eventi indipendenti con la STESSA quota ciascuno, che quota deve
    avere ogni singolo evento?

    Es: voglio una doppia che paghi 1.65 -> ogni singola gamba deve
    valere circa 1.28 (perché 1.28 x 1.28 ≈ 1.65)
    """
    return round(combined_odd ** (1 / n_legs), 3)


def compare_slip_structures(
    starting_capital,
    target_capital,
    target_odd_min,
    target_odd_max,
    max_legs=3,
    edge_scenarios=(0.0, 0.03),
    stake_fraction=1.0,
    n_simulations=20000,
):
    """
    Confronta diverse strutture di giocata (singola, doppia, tripla) che
    puntano tutte alla STESSA quota combinata finale (target_odd_min -
    target_odd_max), mostrando il rischio di ciascuna.

    edge_scenarios: lista di "vantaggi" da testare rispetto al mercato
        efficiente. 0.0 = nessun vantaggio (il modello non batte il
        bookmaker, probabilità vera = 1/quota, scenario onesto di
        base). 0.03 = il modello stima una probabilità reale superiore
        del 3% rispetto a quella implicita nella quota (scenario
        ottimistico, plausibile SOLO se confermato da un vero controllo
        con le quote reali di mercato — Fase successiva).

    Restituisce una lista di risultati, uno per ogni combinazione di
    (numero di gambe, scenario di vantaggio).
    """
    target_odd_mid = (target_odd_min + target_odd_max) / 2
    results = []

    for n_legs in range(1, max_legs + 1):
        leg_odd = required_leg_odd(target_odd_mid, n_legs)
        implied_prob_per_leg = 1 / leg_odd

        for edge in edge_scenarios:
            true_prob_per_leg = min(implied_prob_per_leg + edge, 0.99)
            # Probabilità combinata: prodotto delle probabilità delle
            # singole gambe (assumendo eventi indipendenti tra loro,
            # es. partite diverse in giorni/leghe diversi)
            true_prob_combined = true_prob_per_leg ** n_legs

            sim = simulate_strategy(
                starting_capital=starting_capital,
                target_capital=target_capital,
                odd=target_odd_mid,
                true_win_probability=true_prob_combined,
                stake_fraction=stake_fraction,
                n_simulations=n_simulations,
            )

            results.append({
                "structure": f"{n_legs} gamba/e da ~{leg_odd} l'una" if n_legs > 1
                             else f"Singola da {leg_odd}",
                "n_legs": n_legs,
                "leg_odd": leg_odd,
                "combined_odd": round(target_odd_mid, 3),
                "edge_assumption": edge,
                "true_probability_combined": round(true_prob_combined, 4),
                **sim,
            })

    return results
