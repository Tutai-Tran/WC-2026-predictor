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
TOURNAMENT_END = "2026-07-19"

# Knockout stages in order.
KNOCKOUT_STAGES = ["R32", "R16", "QF", "SF", "Final"]

# Simulation defaults.
DEFAULT_SIM_RUNS = 50_000
DEFAULT_RNG_SEED = 20260611

# Model defaults (estimated from data in the backtest; these are only fallbacks).
DEFAULT_HOME_GOALS = 0.0          # neutral venue baseline; host bump applied per-venue
DEFAULT_HOST_BUMP = 0.12          # extra expected goals when a team plays in its host nation
EXTRA_TIME_SCALE = 1.0 / 3.0      # ET goal expectation relative to 90 min
SHOOTOUT_FAVOURITE_TILT = 0.05    # small tilt above 50/50 by relative strength

# Staleness: forecasts on inputs older than this are flagged STALE.
STALE_HOURS = 36
