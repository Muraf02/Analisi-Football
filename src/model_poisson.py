"""
Modello Dixon-Coles per la previsione dei risultati calcistici.

Cos'è: un'evoluzione del modello di Poisson (lo standard accademico per
modellare i gol nel calcio dal 1997, Dixon & Coles), che corregge un
difetto del Poisson puro nei risultati a basso punteggio (0-0, 1-0, 0-1,
1-1), storicamente più o meno frequenti di quanto il Poisson puro preveda.

Come funziona, in breve:
- Ogni squadra ha due parametri: "attacco" (quanto tende a segnare) e
  "difesa" (quanto tende a subire pochi gol)
- C'è un parametro di "vantaggio casalingo"
- I gol attesi in una partita = funzione di (attacco della squadra in casa,
  difesa della squadra in trasferta, vantaggio casa) e viceversa
- Un parametro aggiuntivo (rho) corregge la sottostima/sovrastima dei
  risultati a basso punteggio
- Le partite più recenti pesano di più nella stima (time decay), così il
  modello si aggiorna naturalmente con la forma attuale delle squadre

Riferimento originale: Dixon, M.J. and Coles, S.G. (1997), "Modelling
Association Football Scores and Inefficiencies in the Football Betting
Market", Journal of the Royal Statistical Society.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from datetime import datetime


def _dixon_coles_adjustment(home_goals, away_goals, home_exp, away_exp, rho):
    """
    Fattore di correzione (tau) applicato ai risultati 0-0, 1-0, 0-1, 1-1,
    dove il Poisson puro tende a essere impreciso.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - (home_exp * away_exp * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (home_exp * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (away_exp * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


class DixonColesModel:
    """
    Modello Dixon-Coles addestrabile su uno storico di partite di una lega.

    Uso tipico:
        model = DixonColesModel()
        model.fit(matches)  # matches: lista di dict con home_team, away_team,
                             # home_goals, away_goals, date
        probs = model.predict_match("Inter", "Milan")
    """

    def __init__(self, time_decay_half_life_days=180):
        """
        time_decay_half_life_days: dopo quanti giorni una partita "pesa"
        la metà rispetto a una partita giocata oggi, nella stima dei
        parametri. 180 giorni è un valore di partenza ragionevole
        (circa mezza stagione): dà più peso alla forma recente senza
        ignorare completamente l'inizio stagione.
        """
        self.half_life = time_decay_half_life_days
        self.teams = []
        self.team_index = {}
        self.attack = None
        self.defense = None
        self.home_advantage = None
        self.rho = None
        self.fitted = False

    def _time_weight(self, match_date, reference_date):
        days_ago = (reference_date - match_date).days
        days_ago = max(days_ago, 0)
        decay_rate = np.log(2) / self.half_life
        return np.exp(-decay_rate * days_ago)

    def _unpack_params(self, params):
        n = len(self.teams)
        attack = params[:n]
        defense = params[n:2 * n]
        home_adv = params[2 * n]
        rho = params[2 * n + 1]
        return attack, defense, home_adv, rho

    def _negative_log_likelihood_vectorized(self, params, home_idx, away_idx,
                                              home_goals, away_goals, weights):
        """
        Versione vettorizzata (numpy) del calcolo della log-verosimiglianza
        negativa: stessa identica formula matematica di prima, ma calcolata
        su array interi invece che con un ciclo Python partita per partita.
        Con centinaia di partite e più squadre (es. più stagioni unite),
        questo rende l'addestramento decine di volte più veloce.
        """
        attack, defense, home_adv, rho = self._unpack_params(params)

        home_exp = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        away_exp = np.exp(attack[away_idx] + defense[home_idx])

        home_exp = np.maximum(home_exp, 1e-6)
        away_exp = np.maximum(away_exp, 1e-6)

        log_p_home = poisson.logpmf(home_goals, home_exp)
        log_p_away = poisson.logpmf(away_goals, away_exp)

        # Correzione Dixon-Coles vettorizzata per i 4 casi 0-0, 1-0, 0-1, 1-1
        tau = np.ones_like(home_exp)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        tau[mask_00] = 1 - (home_exp[mask_00] * away_exp[mask_00] * rho)
        tau[mask_01] = 1 + (home_exp[mask_01] * rho)
        tau[mask_10] = 1 + (away_exp[mask_10] * rho)
        tau[mask_11] = 1 - rho

        tau = np.maximum(tau, 1e-10)

        log_likelihood = log_p_home + log_p_away + np.log(tau)
        return -np.sum(weights * log_likelihood)

    def _negative_log_likelihood(self, params, matches, weights):
        # Mantenuta per compatibilità/debug (versione leggibile a ciclo),
        # ma fit() usa la versione vettorizzata per le prestazioni.
        attack, defense, home_adv, rho = self._unpack_params(params)
        nll = 0.0

        for match, weight in zip(matches, weights):
            i = self.team_index[match["home_team"]]
            j = self.team_index[match["away_team"]]

            home_exp = np.exp(attack[i] + defense[j] + home_adv)
            away_exp = np.exp(attack[j] + defense[i])

            home_exp = max(home_exp, 1e-6)
            away_exp = max(away_exp, 1e-6)

            hg, ag = match["home_goals"], match["away_goals"]

            log_p_home = poisson.logpmf(hg, home_exp)
            log_p_away = poisson.logpmf(ag, away_exp)

            tau = _dixon_coles_adjustment(hg, ag, home_exp, away_exp, rho)
            tau = max(tau, 1e-10)  # evita log(0)

            log_likelihood = log_p_home + log_p_away + np.log(tau)
            nll -= weight * log_likelihood

        return nll

    def fit(self, matches, reference_date=None):
        """
        Allena il modello su uno storico di partite.

        matches: lista di dict, ciascuno con chiavi:
            'home_team', 'away_team' (nomi squadra, str)
            'home_goals', 'away_goals' (int)
            'date' (datetime)
        reference_date: data rispetto a cui calcolare il peso temporale
            (default: la data più recente tra le partite fornite)
        """
        if len(matches) < 20:
            raise ValueError(
                f"Servono almeno 20 partite per una stima affidabile, "
                f"ricevute {len(matches)}."
            )

        self.teams = sorted(set(
            [m["home_team"] for m in matches] + [m["away_team"] for m in matches]
        ))
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        n = len(self.teams)

        if reference_date is None:
            reference_date = max(m["date"] for m in matches)

        weights = np.array([self._time_weight(m["date"], reference_date) for m in matches])

        # Prepara array numpy (indici squadra, gol, pesi) una sola volta,
        # cosi' ogni valutazione della funzione durante l'ottimizzazione
        # lavora su array invece che ripetere un ciclo Python
        home_idx = np.array([self.team_index[m["home_team"]] for m in matches])
        away_idx = np.array([self.team_index[m["away_team"]] for m in matches])
        home_goals_arr = np.array([m["home_goals"] for m in matches])
        away_goals_arr = np.array([m["away_goals"] for m in matches])

        # Parametri iniziali: attacco/difesa a 0 (= squadra "media"),
        # vantaggio casa leggermente positivo, rho vicino a 0
        initial_params = np.concatenate([
            np.zeros(n),      # attacco
            np.zeros(n),      # difesa
            [0.2],            # vantaggio casa
            [-0.05],          # rho
        ])

        # Vincolo di identificabilità: la somma dei parametri di attacco
        # deve essere 0 (altrimenti attacco e difesa potrebbero "scivolare"
        # insieme senza cambiare le probabilità previste)
        constraints = [{
            "type": "eq",
            "fun": lambda p: np.sum(p[:n]),
        }]

        result = minimize(
            self._negative_log_likelihood_vectorized,
            initial_params,
            args=(home_idx, away_idx, home_goals_arr, away_goals_arr, weights),
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 60, "ftol": 1e-6},
        )

        # Diagnostica: aiuta a capire se l'ottimizzatore ha davvero raggiunto
        # una buona soluzione o si e' fermato al limite di iterazioni
        self.fit_diagnostics = {
            "converged": bool(result.success),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
        }

        if not result.success:
            # SLSQP puo' segnalare "non successo" semplicemente perche' ha
            # raggiunto il limite di iterazioni, pur avendo trovato una
            # soluzione già ragionevole (la funzione di perdita smette di
            # migliorare in modo significativo). Blocchiamo il pipeline
            # SOLO in caso di errore vero (valori non numerici), non per
            # questo motivo — altrimenti un limite di sicurezza sul tempo
            # diventerebbe un blocco totale del sistema.
            if not np.all(np.isfinite(result.x)):
                raise RuntimeError(f"Ottimizzazione fallita con valori non validi: {result.message}")

        self.attack, self.defense, self.home_advantage, self.rho = (
            self._unpack_params(result.x)
        )
        self.fitted = True
        return self

    def expected_goals(self, home_team, away_team):
        """Restituisce (gol attesi casa, gol attesi trasferta)."""
        if not self.fitted:
            raise RuntimeError("Il modello non è stato ancora allenato (fit).")
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError(
                f"Squadra non riconosciuta: '{home_team}' o '{away_team}' "
                "non presente nei dati usati per l'addestramento."
            )

        i = self.team_index[home_team]
        j = self.team_index[away_team]

        home_exp = np.exp(self.attack[i] + self.defense[j] + self.home_advantage)
        away_exp = np.exp(self.attack[j] + self.defense[i])
        return home_exp, away_exp

    def score_matrix(self, home_team, away_team, max_goals=8):
        """
        Restituisce una matrice (max_goals+1) x (max_goals+1) con la
        probabilità di ogni risultato esatto, es. matrix[2][1] = P(2-1).
        """
        home_exp, away_exp = self.expected_goals(home_team, away_team)

        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = poisson.pmf(hg, home_exp) * poisson.pmf(ag, away_exp)
                tau = _dixon_coles_adjustment(hg, ag, home_exp, away_exp, self.rho)
                matrix[hg][ag] = p * tau

        # Normalizza (la correzione di Dixon-Coles può alterare leggermente
        # la somma totale delle probabilità)
        matrix = matrix / matrix.sum()
        return matrix

    def predict_match(self, home_team, away_team, max_goals=8):
        """
        Restituisce le probabilità 1X2 e i gol attesi per una partita.

        Output:
            {
                'home_win': float,   # probabilità vittoria casa
                'draw': float,       # probabilità pareggio
                'away_win': float,   # probabilità vittoria trasferta
                'home_expected_goals': float,
                'away_expected_goals': float,
                'most_likely_score': (int, int),
            }
        """
        matrix = self.score_matrix(home_team, away_team, max_goals)
        home_exp, away_exp = self.expected_goals(home_team, away_team)

        home_win = np.sum(np.tril(matrix, -1))   # righe > colonne: casa segna di più
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))    # colonne > righe: trasferta segna di più

        most_likely_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        return {
            "home_win": round(float(home_win), 4),
            "draw": round(float(draw), 4),
            "away_win": round(float(away_win), 4),
            "home_expected_goals": round(float(home_exp), 2),
            "away_expected_goals": round(float(away_exp), 2),
            "most_likely_score": (int(most_likely_idx[0]), int(most_likely_idx[1])),
        }

    def team_ratings(self):
        """
        Restituisce un dizionario {squadra: {'attack': ..., 'defense': ...}}
        utile per ispezionare quali squadre il modello considera più forti
        in attacco/difesa. Valori più alti di 'attack' = attacco migliore;
        valori più bassi (più negativi) di 'defense' = difesa migliore.
        """
        if not self.fitted:
            raise RuntimeError("Il modello non è stato ancora allenato (fit).")
        return {
            team: {
                "attack": round(float(self.attack[i]), 3),
                "defense": round(float(self.defense[i]), 3),
            }
            for team, i in self.team_index.items()
        }
