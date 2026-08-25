"""
Gestisce la corrispondenza tra i nomi delle squadre usati da Understat
e quelli usati da football-data.org, che spesso sono scritti in modo
diverso (es. 'Inter' vs 'FC Internazionale Milano').

Strategia a più livelli:
1. Mappature manuali esplicite per i casi noti più comuni (le squadre
   principali delle 5 leghe monitorate)
2. Normalizzazione automatica (minuscolo, senza accenti, senza parole
   comuni come 'FC', 'AC', 'CF')
3. Similarità testuale come ultima risorsa

Se una partita non viene abbinata correttamente, comparirà nel file di
log (logs/pipeline.log): a quel punto si può aggiungere una nuova riga
a MANUAL_MAPPING qui sotto.
"""

import re
import unicodedata
from difflib import SequenceMatcher

# Mappature manuali: chiave = nome usato da Understat,
# valore = lista di nomi/varianti usati da football-data.org
MANUAL_MAPPING = {
    # Serie A
    "Inter": ["FC Internazionale Milano", "Inter"],
    "AC Milan": ["AC Milan", "Milan"],
    "Juventus": ["Juventus FC", "Juventus"],
    "Napoli": ["SSC Napoli", "Napoli"],
    "AS Roma": ["AS Roma", "Roma"],
    "Verona": ["Hellas Verona FC", "Hellas Verona"],

    # Premier League
    "Manchester United": ["Manchester United FC", "Man United"],
    "Manchester City": ["Manchester City FC", "Man City"],
    "Newcastle United": ["Newcastle United FC", "Newcastle"],
    "Tottenham": ["Tottenham Hotspur FC", "Tottenham"],
    "Wolverhampton Wanderers": ["Wolverhampton Wanderers FC", "Wolves"],
    "Nottingham Forest": ["Nottingham Forest FC", "Forest"],
    "Brighton": ["Brighton & Hove Albion FC", "Brighton Hove Albion"],

    # La Liga
    "Real Madrid": ["Real Madrid CF", "Real Madrid"],
    "Barcelona": ["FC Barcelona", "Barcelona"],
    "Atletico Madrid": ["Club Atlético de Madrid", "Atletico Madrid"],
    "Athletic Club": ["Athletic Club", "Athletic Bilbao"],
    "Real Sociedad": ["Real Sociedad de Fútbol", "Real Sociedad"],
    "Celta Vigo": ["RC Celta de Vigo", "Celta Vigo"],

    # Ligue 1
    "Paris Saint Germain": ["Paris Saint-Germain FC", "Paris Saint-Germain"],
    "Marseille": ["Olympique de Marseille", "Marseille"],
    "Lyon": ["Olympique Lyonnais", "Lyon"],
    "Monaco": ["AS Monaco FC", "Monaco"],
    "Saint-Etienne": ["AS Saint-Étienne", "Saint-Étienne"],

    # Bundesliga
    "Bayern Munich": ["FC Bayern München", "Bayern Munich"],
    "Borussia Dortmund": ["Borussia Dortmund", "Dortmund"],
    "RB Leipzig": ["RB Leipzig", "Leipzig"],
    "Bayer Leverkusen": ["Bayer 04 Leverkusen", "Leverkusen"],
    "Eintracht Frankfurt": ["Eintracht Frankfurt", "Frankfurt"],
    "Borussia M.Gladbach": ["Borussia Mönchengladbach", "Gladbach"],
}

# Parole comuni da ignorare nella normalizzazione (sigle societarie, ecc.)
_NOISE_WORDS = {
    "fc", "cf", "ac", "as", "ss", "ssc", "rc", "sv", "vfb", "vfl", "sc", "tsg",
    "club", "calcio", "football", "de", "futbol",
}


def _normalize(name):
    """Normalizza un nome squadra per il confronto: minuscolo, senza accenti,
    senza parole comuni tipo FC/AC/CF."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    words = [w for w in name.split() if w not in _NOISE_WORDS]
    return " ".join(words).strip()


def match_team_name(understat_name, football_data_name):
    """
    Restituisce True se i due nomi si riferiscono (con buona probabilità)
    alla stessa squadra.
    """
    # 1. Mappatura manuale esplicita
    if understat_name in MANUAL_MAPPING:
        if football_data_name in MANUAL_MAPPING[understat_name]:
            return True

    # 2. Confronto normalizzato esatto
    norm_u = _normalize(understat_name)
    norm_f = _normalize(football_data_name)
    if norm_u == norm_f:
        return True

    # 3. Un nome contenuto nell'altro (es. 'Inter' dentro 'Internazionale')
    if norm_u and norm_f and (norm_u in norm_f or norm_f in norm_u):
        return True

    # 4. Similarità testuale come ultima risorsa (soglia alta per evitare falsi positivi)
    similarity = SequenceMatcher(None, norm_u, norm_f).ratio()
    return similarity > 0.75
