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
        report["knockout"] = _ingest_knockout_fixtures(conn)
        report["friendlies"] = _ingest_friendlies(conn)
        conn.commit()
        return report
    except Exception:
        # don't leave a half-built DB if one sub-ingest fails partway
        conn.rollback()
        raise
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


_KO_MX = {"Guadalupe", "Mexico City", "Monterrey", "Guadalajara"}
_KO_CA = {"Toronto", "Vancouver"}


def _ko_country(city: str | None) -> str:
    if city in _KO_MX:
        return "Mexico"
    if city in _KO_CA:
        return "Canada"
    return "United States"


def _stage_for_match_no(n: int) -> str:
    if 73 <= n <= 88:
        return "R32"
    if 89 <= n <= 96:
        return "R16"
    if 97 <= n <= 100:
        return "QF"
    if 101 <= n <= 102:
        return "SF"
    if n == 104:
        return "Final"
    return "KO"


def _ingest_knockout_fixtures(conn) -> dict:
    """Round of 32 fixtures (with dates + slots) and the structural R16->Final tree."""
    bracket = _load_json("bracket.json")
    n = 0
    for m in bracket.get("r32", []):
        conn.execute(
            "INSERT INTO matches (stage, match_no, date_utc, venue, venue_country, "
            "home_slot, away_slot, source) VALUES ('R32',?,?,?,?,?,?, 'bracket') "
            "ON CONFLICT(match_no) DO NOTHING",
            (m["match"], m.get("date_utc"), m.get("venue"), _ko_country(m.get("city")),
             m["home_slot"], m["away_slot"]),
        )
        n += 1
    from .simulate import KNOCKOUT_TREE
    for match_no, (a, b) in KNOCKOUT_TREE.items():
        conn.execute(
            "INSERT INTO matches (stage, match_no, home_slot, away_slot, source) "
            "VALUES (?,?,?,?, 'bracket') ON CONFLICT(match_no) DO NOTHING",
            (_stage_for_match_no(match_no), match_no, f"W{a}", f"W{b}"),
        )
        n += 1
    return {"count": n}


def _elo_lookup() -> dict[str, float]:
    for fname in ("replayed_elo.json", "elo_seed.json"):
        path = config.DATA_RAW / fname
        if path.exists():
            return json.loads(path.read_text()).get("ratings", {})
    return {}


def _ensure_team(conn, name: str, elo: dict[str, float]) -> int:
    """Return team id, inserting a minimal (non-WC) team + rating if absent."""
    row = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO teams (name, is_host) VALUES (?, 0)", (name,))
    tid = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()["id"]
    rating = elo.get(name) or elo.get(ELO_ALIASES.get(name, "")) or 1500.0
    # seed with a PAST date so a later recompute (dated today) always sorts as latest
    conn.execute(
        "INSERT INTO ratings (team_id, elo, valid_from, source) VALUES (?,?,?,?) "
        "ON CONFLICT(team_id, valid_from) DO UPDATE SET elo=excluded.elo",
        (tid, float(rating), "2000-01-01", "friendly-opponent"),
    )
    return tid


def _ingest_friendlies(conn) -> dict:
    """Pre-tournament warm-up friendlies (predicted, and Elo-updating once played)."""
    path = config.DATA_RAW / "friendlies.json"
    if not path.exists():
        return {"count": 0, "note": "no friendlies.json yet"}
    data = json.loads(path.read_text())
    elo = _elo_lookup()
    n = 0
    for m in data.get("matches", []):
        if not m.get("home") or not m.get("away"):
            continue
        hid = _ensure_team(conn, m["home"], elo)
        aid = _ensure_team(conn, m["away"], elo)
        hs, as_ = m.get("home_score"), m.get("away_score")
        played = 1 if hs is not None and as_ is not None else 0
        conn.execute(
            "INSERT INTO matches (stage, date_utc, venue, home_team_id, away_team_id, "
            "home_goals, away_goals, played, source) VALUES ('friendly',?,?,?,?,?,?,?, 'friendlies.json') "
            "ON CONFLICT(stage, home_team_id, away_team_id, date_utc) DO NOTHING",
            (m.get("date_utc"), m.get("venue"), hid, aid, hs, as_, played),
        )
        if played:
            conn.execute(
                "INSERT INTO results_ledger (played_on, home_team, away_team, home_goals, "
                "away_goals, neutral, tournament, source, fetched_at) "
                "VALUES (?,?,?,?,?,1,'Friendly','friendlies.json',?)",
                (str(m.get("date_utc"))[:10], m["home"], m["away"], hs, as_, "ingest"),
            )
        n += 1
    return {"count": n}


def main() -> None:
    from . import db
    conn = db.connect()
    report = ingest_all(conn)
    print(report)


if __name__ == "__main__":
    main()
