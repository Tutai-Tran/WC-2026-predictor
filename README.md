# WC-2026-predictor

FIFA World Cup 2026 prediction system. Forecasts match outcomes, scorelines, goalscorers, and tournament progression as calibrated probabilities, fed by scraped data and an LLM news agent, improving itself as results arrive.

Full plan and context: [PLAN.md](./PLAN.md).

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest          # run tests
.venv/bin/python -m wc26.forecast   # produce the full forecast
```

## Layout

- `wc26/` Python package (model, rules engine, simulator, ingest, vault generation, CLI)
- `tests/` unit + property tests (the rules engine is the highest-bug-risk surface)
- `data/raw/` scraped/seed data (results, Elo, groups, squads) ingested into SQLite
- `WC vault/` standalone Obsidian vault (one note per country/group/match), generated from the DB
- `wc26.db` SQLite source of truth (created on first run)

## Principles (see PLAN.md)

- Honest probabilities, not certainties. Every forecast carries a data-freshness/STALE banner.
- Works on manually entered data first; scrapers and the odds blend are accelerators on top.
- Elo updates fast from results; structural parameters change slowly and only when a significance test says so.
- Odds (free The Odds API) are a prediction signal and benchmark only, never betting.
