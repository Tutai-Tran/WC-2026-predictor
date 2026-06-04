# WC-2026 Predictor: What's been built

A self-improving FIFA World Cup 2026 forecasting system: scrapes data, predicts every match (winner, score, scorers) and the full tournament (who reaches each round, who wins), keeps a continuously-updated Obsidian vault, and gets better as results arrive. Repo: github.com/Tutai-Tran/WC-2026-predictor.

## How to use

```bash
cd "/Users/tutaitran/Documents/own projects/WC-2026-predictor"
.venv/bin/python -m wc26.forecast        # full forecast -> DB + vault + console
.venv/bin/streamlit run wc26/app.py      # dashboard (http://localhost:8501)
.venv/bin/python -m wc26.update          # one self-improving refresh (scrape+news+recompute+forecast+vault)
bash scripts/install_autostart.sh        # auto-start dashboard + 3-hourly refresh on every login
bash scripts/stop.sh                     # stop everything + disable auto-start
```

The Obsidian vault is the folder `WC vault/` — open it in Obsidian via "Open folder as vault".

## The prediction model (honest probabilities, not certainties)

- **Elo** (`elo.py`): World Football Elo, replayed from a results ledger; self-maintained as results arrive.
- **Match model** (`model.py`): Elo difference -> expected goals -> Poisson + Dixon-Coles scoreline matrix -> W/D/L and top scorelines. Availability reduces a team's attacking expected goals.
- **Rules engine** (`rules.py`): exact 2026 rules — head-to-head-first group tie-breaks (with recursive re-application), best-8-of-12 third-placed teams.
- **Simulator** (`simulate.py`): Monte Carlo (50k) over groups -> the official 495-combination Annex C routing -> knockouts with extra time and penalty shootouts. Conditional re-simulation fixes already-played matches.
- **Goalscorers** (`scorers.py`): anytime-scorer probability = team goals x shrunk goal-share x minutes, plus a penalty-taker stream.
- **Calibration** (`backtest.py`): leak-free historical backtest; beats uniform/climatology/favourite baselines (test log loss 0.861), reliability slope ~1.0, with bootstrap CIs and a tournament-only slice.

## Data & the self-improving loop

- One-time seed (Wikipedia / martj42 results / eloratings / FIFA bracket), ingested into SQLite (`ingest.py`).
- **Live scraper** (`scrape.py`): ESPN public site JSON for completed friendly + World Cup results, deduped into the ledger; `recompute_elo` folds them into current ratings.
- **LLM news agent** (`news.py`): the local Claude CLI (your Max subscription) web-searches current injuries/suspensions per team and returns validated, source-quoted availability events that reduce a team's strength in the forecast.
- **Odds** (`odds.py`): The Odds API winner odds, devigged, as a prediction benchmark and a model+market blend (never betting).
- **Refresh loop** (`update.py`): sync vault overrides -> scrape results -> recompute Elo -> LLM news scan (rotating) -> log running calibration -> re-forecast -> regenerate vault -> log to memory.

## The vault (`WC vault/`)

- `_forecast.md` dashboard, `_bracket.md` (Mermaid knockout bracket), `_memory/CONTEXT.md` (running project memory, updated every task).
- `Countries/` (48, with snapshot + champion %, squad, and current injuries), `Groups/` (12), `Matches/{Group,Friendly,Knockout}` (dated, typed). Machine-written AUTO blocks; your HUMAN notes/overrides are never overwritten and feed back into the model.

## Dashboard (`app.py`, Streamlit)

8 tabs: Daily (matches by date), Champion & stages (with Monte Carlo CIs and a model+market blend), Matches (split by type), Bracket (graphviz diagram), Groups, Scorers, Availability (scraped injuries), Calibration (baselines, reliability, model-vs-market, friendly accuracy).

## Added in the overnight session (2026-06-05)

Vault memory system; match types/dates and a Daily view; 62 warm-up friendlies (predicted, and feeding Elo); knockout fixtures + a projected bracket that fills with winners; expanded scrapers (ESPN live results) and the LLM news agent; odds blend; user-friendly dashboard rebuild with the bracket diagram and Availability tab; auto-start on login; and three multi-agent test rounds (14 agents) with all findings fixed.

## Testing

65 unit/integration/property/regression tests (`tests/`), plus three rounds of adversarial multi-agent review. `.venv/bin/python -m pytest`.

## Honesty

Probabilities, not certainties. A single most-likely scoreline is usually ~10% likely; the top scorer pick misses about half the time. The model runs slightly hotter than the betting market on the very top teams (shown on the Calibration tab). Odds are used only as a prediction signal, never for betting.
