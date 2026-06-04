"""SQLite source of truth: schema, connection, and migrations.

Design notes (see PLAN.md):
- Results live in an append-only ledger so Elo is replayable (recomputed from the
  ledger each run, never mutated in place); a corrected result self-heals.
- predictions and model_runs are append-only snapshots so calibration-over-time
  and a "why did this move" diff are possible.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    fifa_code   TEXT,
    confederation TEXT,
    group_letter TEXT,
    fifa_rank   INTEGER,
    is_host     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS players (
    id          INTEGER PRIMARY KEY,
    team_id     INTEGER NOT NULL REFERENCES teams(id),
    name        TEXT NOT NULL,
    position    TEXT,
    is_penalty_taker INTEGER NOT NULL DEFAULT 0,
    club_goals  REAL,
    minutes     REAL,
    goal_share  REAL,
    UNIQUE(team_id, name)
);

-- Fixtures and (optionally) their final scores.
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY,
    stage       TEXT NOT NULL,            -- 'group' or one of KNOCKOUT_STAGES
    group_letter TEXT,
    matchday    INTEGER,
    date_utc    TEXT,
    venue       TEXT,
    venue_country TEXT,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_goals  INTEGER,
    away_goals  INTEGER,
    played      INTEGER NOT NULL DEFAULT 0,
    source      TEXT,
    UNIQUE(stage, home_team_id, away_team_id, date_utc)
);

-- Append-only ledger of confirmed results (drives replayable Elo).
CREATE TABLE IF NOT EXISTS results_ledger (
    id          INTEGER PRIMARY KEY,
    match_id    INTEGER REFERENCES matches(id),
    played_on   TEXT,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    neutral     INTEGER NOT NULL DEFAULT 1,
    tournament  TEXT,
    source      TEXT,
    fetched_at  TEXT
);

-- Elo time series (valid_from lets us pick the as-of rating in backtests).
CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY,
    team_id     INTEGER NOT NULL REFERENCES teams(id),
    elo         REAL NOT NULL,
    valid_from  TEXT NOT NULL,
    source      TEXT,
    UNIQUE(team_id, valid_from)
);

CREATE TABLE IF NOT EXISTS availability_events (
    id          INTEGER PRIMARY KEY,
    team_id     INTEGER REFERENCES teams(id),
    player_name TEXT,
    status      TEXT,                      -- out / doubtful / fit / suspended
    expected_return TEXT,
    minutes_factor REAL,                   -- 0..1 expected share of minutes
    source      TEXT,
    source_quote TEXT,
    url         TEXT,
    confidence  REAL,
    fetched_at  TEXT,
    reviewed    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS manual_overrides (
    id          INTEGER PRIMARY KEY,
    team_id     INTEGER REFERENCES teams(id),
    player_name TEXT,
    kind        TEXT NOT NULL,             -- availability / result / note
    payload_json TEXT,
    created_at  TEXT
);

-- Append-only forecast snapshots.
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    scope       TEXT NOT NULL,             -- 'match' / 'team-stage' / 'champion'
    ref         TEXT,                      -- match id or team name
    payload_json TEXT NOT NULL,
    computed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL UNIQUE,
    ts          TEXT NOT NULL,
    git_sha     TEXT,
    config_json TEXT,
    input_hash  TEXT,
    rng_seed    INTEGER,
    model_version TEXT,
    metrics_json TEXT,
    data_completeness REAL
);

CREATE TABLE IF NOT EXISTS public_benchmark (
    id          INTEGER PRIMARY KEY,
    source      TEXT,
    scope       TEXT,
    payload_json TEXT,
    fetched_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_stage ON matches(stage);
CREATE INDEX IF NOT EXISTS idx_ratings_team ON ratings(team_id, valid_from);
CREATE INDEX IF NOT EXISTS idx_results_teams ON results_ledger(home_team, away_team, played_on);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with sane pragmas and row access by name."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if missing and stamp the schema version."""
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    return row["version"] if row else None
