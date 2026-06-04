"""Load scraped raw data (data/raw/*) into the SQLite source of truth.

Team names are the join key across sources. The scrapers were largely consistent;
only two names need aliasing to the Elo/squad sources.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, db

# groups.json name -> Elo seed name
ELO_ALIASES = {"Czech Republic": "Czechia", "Curacao": "Curaçao"}
# groups.json name -> squads.json key
SQUAD_ALIASES = {"Curacao": "Curaçao"}


def _load_json(name: str) -> dict:
    return json.loads((config.DATA_RAW / name).read_text())


def ingest_all(conn, raw_dir: Path | None = None) -> dict:
    """(Re)build core tables from data/raw. Returns a report with counts/misses."""
    global_raw = config.DATA_RAW
    if raw_dir is not None:
        config.DATA_RAW = raw_dir  # allow override in tests
    try:
        db.init_db(conn)
        report: dict = {}
        report["teams"] = _ingest_teams_and_groups(conn)
        report["elo"] = _ingest_elo(conn)
        report["fixtures"] = _ingest_fixtures(conn)
        report["players"] = _ingest_squads(conn)
        conn.commit()
        return report
    finally:
        config.DATA_RAW = global_raw


def _ingest_teams_and_groups(conn) -> dict:
    groups = _load_json("groups.json")["groups"]
    n = 0
    for letter, teams in groups.items():
        for t in teams:
            is_host = 1 if t.get("is_host") else 0
            conn.execute(
                """INSERT INTO teams (name, fifa_code, confederation, group_letter, fifa_rank, is_host)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                     fifa_code=excluded.fifa_code, confederation=excluded.confederation,
                     group_letter=excluded.group_letter, fifa_rank=excluded.fifa_rank,
                     is_host=excluded.is_host""",
                (t["name"], t.get("fifa_code"), t.get("confederation"),
                 letter, t.get("fifa_rank"), is_host),
            )
            n += 1
    return {"count": n, "groups": len(groups)}


def _team_id(conn, name: str) -> int | None:
    row = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()
    return row["id"] if row else None


def _ingest_elo(conn) -> dict:
    # Prefer self-replayed ratings (same scale the goal model is fitted on);
    # fall back to the eloratings.net seed before the backtest has been run.
    replayed = config.DATA_RAW / "replayed_elo.json"
    if replayed.exists():
        data = json.loads(replayed.read_text())
        src = "replay"
    else:
        data = _load_json("elo_seed.json")
        src = "elo_seed"
    ratings = data["ratings"]
    valid_from = data.get("updated", config.TOURNAMENT_START)
    misses = []
    n = 0
    for row in conn.execute("SELECT id, name FROM teams").fetchall():
        name = row["name"]
        elo = ratings.get(name) or ratings.get(ELO_ALIASES.get(name, ""))
        if elo is None:
            misses.append(name)
            continue
        conn.execute(
            """INSERT INTO ratings (team_id, elo, valid_from, source)
               VALUES (?,?,?,?)
               ON CONFLICT(team_id, valid_from) DO UPDATE SET elo=excluded.elo, source=excluded.source""",
            (row["id"], float(elo), valid_from, src),
        )
        n += 1
    return {"count": n, "misses": misses}


def _ingest_fixtures(conn) -> dict:
    matches = _load_json("fixtures.json")["matches"]
    n = 0
    for m in matches:
        hid = _team_id(conn, m["home"])
        aid = _team_id(conn, m["away"])
        if hid is None or aid is None:
            continue
        conn.execute(
            """INSERT INTO matches (stage, group_letter, matchday, date_utc, venue,
                                    venue_country, home_team_id, away_team_id, source)
               VALUES ('group',?,?,?,?,?,?,?, 'fixtures.json')
               ON CONFLICT(stage, home_team_id, away_team_id, date_utc) DO NOTHING""",
            (m.get("group"), m.get("matchday"), m.get("date_utc"),
             m.get("venue"), m.get("venue_country"), hid, aid),
        )
        n += 1
    return {"count": n}


def _ingest_squads(conn) -> dict:
    squads = _load_json("squads.json")["squads"]
    n = 0
    teams_done = 0
    for row in conn.execute("SELECT id, name FROM teams").fetchall():
        name = row["name"]
        players = squads.get(name) or squads.get(SQUAD_ALIASES.get(name, ""))
        if not players:
            continue
        teams_done += 1
        for p in players:
            goals = p.get("intl_goals")
            if goals is None:
                goals = p.get("club_goals_last")
            conn.execute(
                """INSERT INTO players (team_id, name, position, is_penalty_taker, club_goals)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(team_id, name) DO UPDATE SET
                     position=excluded.position, is_penalty_taker=excluded.is_penalty_taker,
                     club_goals=excluded.club_goals""",
                (row["id"], p["name"], p.get("position"),
                 1 if p.get("is_penalty_taker") else 0, goals),
            )
            n += 1
    return {"players": n, "teams_with_squad": teams_done}


def main() -> None:
    from . import db
    conn = db.connect()
    report = ingest_all(conn)
    print(report)


if __name__ == "__main__":
    main()
