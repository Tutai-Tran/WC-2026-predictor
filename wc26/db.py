"""SQLite source of truth: schema, connection, and migrations.

Design notes (see PLAN.md):
- Results live in an append-only ledger so Elo is replayable (recomputed from the
  ledger each run, never mutated in place); a corrected result self-heals.
- predictions and model_runs are append-only snapshots so calibration-over-time
  and a "why did this move" diff are possible.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

from . import config

SCHEMA_VERSION = 4

_SQLITE_MAGIC = b"SQLite format 3\x00"

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
    stage       TEXT NOT NULL,            -- 'friendly' | 'group' | R32/R16/QF/SF/Final
    match_no    INTEGER,                  -- official match number (knockouts)
    group_letter TEXT,
    matchday    INTEGER,
    date_utc    TEXT,
    venue       TEXT,
    venue_country TEXT,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_slot   TEXT,                     -- e.g. '1A','2B','3rd C/E/F/H/I' (knockouts)
    away_slot   TEXT,
    home_goals  INTEGER,
    away_goals  INTEGER,
    played      INTEGER NOT NULL DEFAULT 0,
    source      TEXT,
    UNIQUE(stage, home_team_id, away_team_id, date_utc),
    UNIQUE(match_no)
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

-- Self-learning loop -----------------------------------------------------
-- One PRE-MATCH prediction snapshot per match, frozen BEFORE its result is
-- folded into Elo, then graded in place once the match is played. This is the
-- leak-free basis for "why was this prediction wrong" analysis.
CREATE TABLE IF NOT EXISTS prediction_log (
    id            INTEGER PRIMARY KEY,
    match_id      INTEGER REFERENCES matches(id),
    stage         TEXT,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    date_utc      TEXT,
    p_home        REAL NOT NULL,
    p_draw        REAL NOT NULL,
    p_away        REAL NOT NULL,
    lambda_home   REAL,
    lambda_away   REAL,
    top_scoreline TEXT,
    pick          TEXT NOT NULL,          -- modal outcome at snapshot: home|draw|away
    inputs_json   TEXT,                   -- frozen inputs (elo gap, h2h, availability) for the post-mortem
    snapshot_at   TEXT NOT NULL,
    graded        INTEGER NOT NULL DEFAULT 0,
    home_goals    INTEGER,
    away_goals    INTEGER,
    actual        TEXT,                   -- home|draw|away
    correct       INTEGER,                -- 1 if pick==actual
    brier         REAL,
    log_loss      REAL,
    graded_at     TEXT,
    postmortem_status TEXT NOT NULL DEFAULT 'none',  -- none|pending|done|skipped|error
    UNIQUE(match_id)                      -- upsert while upcoming; frozen once graded
);

-- Structured LLM root-cause lessons for WRONG graded predictions.
CREATE TABLE IF NOT EXISTS postmortems (
    id            INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES prediction_log(id),
    match_id      INTEGER REFERENCES matches(id),
    factor        TEXT NOT NULL,          -- enum: elo_gap|home_advantage|availability|h2h|goal_volume|draw|variance|...
    direction     TEXT NOT NULL,          -- over|under (we over/under-weighted the factor)
    magnitude     REAL,
    suggested_segment TEXT,               -- global|friendly|group|knockout|host|mismatch|close_match
    confidence    REAL,
    evidence      TEXT,
    summary       TEXT,
    model_raw     TEXT,
    created_at    TEXT NOT NULL,
    applied       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(prediction_id, factor)
);

-- Phase 3b: every parameter-adjustment DECISION (adopted or rejected), so the
-- learning loop's changes to the algorithm are auditable and reversible. A row is
-- written each time the adoption gate evaluates a candidate against held-out data.
CREATE TABLE IF NOT EXISTS model_params_log (
    id              INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,
    factor          TEXT,                  -- the bias that motivated the change
    param           TEXT,                  -- which fitted parameter (c|gamma|home_adv_elo|rho|base_goals)
    direction       TEXT,                  -- up|down
    old_value       REAL,
    new_value       REAL,
    ll_new_before   REAL,                  -- log loss on the 2026 graded matches, before
    ll_new_after    REAL,                  -- ...and after the nudge (must improve to adopt)
    ll_hist_before  REAL,                  -- log loss on the historical held-out set, before
    ll_hist_after   REAL,                  -- ...and after (must not worsen beyond margin)
    n_eval          INTEGER,               -- number of 2026 graded matches used to validate
    adopted         INTEGER NOT NULL,      -- 1 if applied to fitted_params.json, 0 if rejected
    reason          TEXT,
    params_json     TEXT                   -- full fitted-params snapshot after this decision (rollback)
);

CREATE INDEX IF NOT EXISTS idx_predlog_grade ON prediction_log(graded, date_utc);
CREATE INDEX IF NOT EXISTS idx_predlog_pm ON prediction_log(graded, correct, postmortem_status);
CREATE INDEX IF NOT EXISTS idx_postmortems_applied ON postmortems(applied, factor, suggested_segment);
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
    # a second writer (manual run beside the loop) waits instead of failing instantly
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def is_valid_snapshot(path: str | Path) -> bool:
    """True only if `path` is a readable SQLite database that contains a forecast.

    Guards a hosted DB download against partial transfers, HTML error/redirect
    pages (e.g. a 404 served during the mid-publish ``--clobber`` window), and any
    other non-database bytes, so a bad download is never swapped in."""
    try:
        with open(path, "rb") as f:
            if f.read(16) != _SQLITE_MAGIC:
                return False
    except OSError:
        return False
    try:
        # quote so paths with spaces/special chars form a valid SQLite file: URI
        c = sqlite3.connect(f"file:{quote(str(Path(path).resolve()))}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        if c.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            return False
        return c.execute(
            "SELECT COUNT(*) FROM predictions WHERE scope='forecast'"
        ).fetchone()[0] > 0
    except sqlite3.DatabaseError:
        return False
    finally:
        c.close()


def install_snapshot(content: bytes, dest: str | Path | None = None) -> bool:
    """Atomically install a downloaded DB snapshot, returning True only if it took.

    Validates the bytes are a real forecast snapshot BEFORE swapping, and clears any
    stale ``-wal``/``-shm`` sidecars first: those belong to the previous file and,
    read against the freshly-swapped main DB, raise ``database disk image is
    malformed`` (the recurring hosted crash). A rejected download leaves the existing
    DB untouched."""
    dest = Path(dest) if dest is not None else config.DB_PATH
    tmp = str(dest) + ".download"
    Path(tmp).write_bytes(content)
    if not is_valid_snapshot(tmp):
        Path(tmp).unlink(missing_ok=True)
        return False
    for sfx in ("-wal", "-shm"):
        Path(str(dest) + sfx).unlink(missing_ok=True)
    os.replace(tmp, dest)                            # atomic swap of a validated file
    return True


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if missing and stamp the schema version."""
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] != SCHEMA_VERSION:                    # additive migration applied above; stamp it
        conn.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    return row["version"] if row else None
