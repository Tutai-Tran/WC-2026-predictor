"""Project paths and FIFA World Cup 2026 structural constants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
VAULT = REPO_ROOT / "WC vault"
DB_PATH = REPO_ROOT / "wc26.db"

# 2026 format: 48 teams, 12 groups of 4.
GROUP_LETTERS = list("ABCDEFGHIJKL")
TEAMS_PER_GROUP = 4
N_TEAMS = 48

# Advancement: group winners + runners-up (24) + 8 best third-placed teams = 32 -> Round of 32.
N_GROUP_WINNERS = 12
N_RUNNERS_UP = 12
N_BEST_THIRDS = 8
KNOCKOUT_TEAMS = 32

HOSTS = {"United States", "Canada", "Mexico"}

# Tournament window (context for cadence, not a deadline).
TOURNAMENT_START = "2026-06-11"
GROUP_STAGE_END = "2026-06-27"
TOURNAMENT_END = "2026-07-19"


def tournament_mode(today: str | None = None) -> bool:
    """True while the tournament runs: faster news rotation, wider scrape window,
    and group-stage learning thresholds."""
    from datetime import date
    d = today or date.today().isoformat()
    return TOURNAMENT_START <= d <= TOURNAMENT_END


def group_stage_mode(today: str | None = None) -> bool:
    """True during the group stage, when the adoption gate runs with the lower
    (one-matchday) thresholds so the loop can learn inside the 72-match window."""
    from datetime import date
    d = today or date.today().isoformat()
    return TOURNAMENT_START <= d <= GROUP_STAGE_END

# Knockout stages in order.
KNOCKOUT_STAGES = ["R32", "R16", "QF", "SF", "Final"]

# Simulation defaults.
DEFAULT_SIM_RUNS = 50_000
DEFAULT_RNG_SEED = 20260611

# Model defaults (estimated from data in the backtest; these are only fallbacks).
# Host advantage: home advantage refit on 200 non-neutral FINALS matches (hosts at
# WC/continental finals, Nations League excluded) = 129.9 Elo. 2026 is discounted from
# that per host: Mexico keeps most of it (Azteca altitude, uniformly hostile crowd);
# the USA is discounted hardest (many "home" crowds are majority-away, e.g. vs Mexico
# or CONMEBOL sides); Canada sits between.
HOST_BUMP_ELO = 90.0              # generic fallback when a host has no specific value
HOST_BUMP_BY_COUNTRY = {"Mexico": 110.0, "United States": 75.0, "Canada": 90.0}
EXTRA_TIME_SCALE = 1.0 / 3.0      # ET goal expectation relative to 90 min
SHOOTOUT_FAVOURITE_TILT = 0.12    # fitted on 540 shootouts since 1990 (wc26.shootout:
                                  # favourites won 54.8%; slope 0.145, se 0.058; neutral-only
                                  # 0.212), shrunk toward the coin-flip prior for robustness

# Staleness: forecasts on inputs older than this are flagged STALE.
STALE_HOURS = 36

# News-LLM availability events expire after this many hours unless re-confirmed by a
# rescan (injuries heal, suspensions end; only human vault/manual entries persist).
NEWS_EVENT_MAX_AGE_HOURS = 72
