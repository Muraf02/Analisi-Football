# Football Analytics — Fase 1: Pipeline Dati

Raccoglie automaticamente squadre, calendario e risultati di Serie A, Premier League,
La Liga, Ligue 1 e Bundesliga da football-data.org, salvandoli in un database SQLite
locale, pronto per le fasi successive (feature engineering, modello, strategia).

## Setup (una tantum)

### 1. Ottieni la API key gratuita

1. Vai su https://www.football-data.org/client/register
2. Registrati con la tua email (gratuito, nessuna carta richiesta)
3. Riceverai una API key via email — copiala

### 2. Installa Python e le dipendenze

Serve Python 3.9+. Poi, da terminale, dentro la cartella del progetto:

```bash
pip install -r requirements.txt
```

### 3. Imposta la API key come variabile d'ambiente

**Su Mac/Linux:**
```bash
export FOOTBALL_DATA_API_KEY="la_tua_chiave_qui"
```

**Su Windows (PowerShell):**
```powershell
$env:FOOTBALL_DATA_API_KEY="la_tua_chiave_qui"
```

> Nota: questo comando vale solo per la sessione di terminale corrente.
> Per renderlo permanente, aggiungilo al tuo `.bashrc`/`.zshrc` (Mac/Linux)
> o alle variabili d'ambiente di sistema (Windows). Se vuoi, nel prossimo
> step posso prepararti anche un file `.env` con caricamento automatico.

## Uso

### Aggiornamento singolo (manuale)

```bash
python src/fetch_football_data.py
```

Scarica squadre e partite di tutte e 5 le leghe e le salva/aggiorna nel database
(`data/football.db`).

### Aggiornamento automatico continuo

```bash
python src/scheduler.py
```

Aggiorna subito, poi ogni 6 ore, tenendo i dati sempre freschi. Lascialo aperto
in un terminale dedicato (o configuralo come servizio quando passeremo al deploy
su server).

## Struttura del database

- **teams**: anagrafica squadre per lega
- **matches**: calendario e risultati (aggiornati ad ogni run — le partite
  passano da `SCHEDULED` a `FINISHED` automaticamente)
- **match_stats**: statistiche avanzate (xG, tiri, possesso) — popolata in Fase 2
- **odds**: quote bookmaker — popolata in Fase 3
- **pipeline_runs**: log delle esecuzioni, per debug

## Verificare che funzioni

Dopo il primo aggiornamento, puoi controllare i dati così:

```bash
python -c "
from src.db import get_connection
conn = get_connection()
cur = conn.cursor()
cur.execute('SELECT league_code, COUNT(*) FROM matches GROUP BY league_code')
for row in cur.fetchall():
    print(row['league_code'], row['COUNT(*)'])
"
```

Dovresti vedere il conteggio partite per ciascuna delle 5 leghe.

## Limiti del piano gratuito football-data.org

- 10 richieste al minuto
- Dati storici limitati (in genere stagione corrente + poche precedenti)
- Nessuna statistica avanzata (xG, formazioni dettagliate) — per questo
  serviranno le fonti aggiuntive (Understat) nella Fase 2

## Prossimi step (non ancora implementati)

- **Fase 2**: integrazione xG/xGA da Understat (feature più predittive dei
  soli gol)
- **Fase 3**: integrazione quote bookmaker (The Odds API) per confronto
  probabilità modello vs mercato
- **Fase 4**: feature engineering (forma, Elo rating, rolling stats)
- **Fase 5**: modello predittivo (Poisson/Dixon-Coles + eventuale ML)
- **Fase 6**: modulo di risk management e target (simulazioni Monte Carlo,
  Kelly frazionato)
