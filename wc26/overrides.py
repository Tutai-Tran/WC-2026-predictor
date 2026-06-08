"""Manual data entry and parse-back of human vault overrides (two-tier vault).

The HUMAN `## Manual overrides` block in each country note is the human's source
of truth for availability judgements; this module parses those lines back into the
DB so they influence the forecast. It also provides manual result entry so a
matchday forecast never stalls when a scraper is unavailable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import config

# e.g.  - Vinicius: doubtful, 50% minutes, until 2026-06-20, source: Marca
#       - Neymar: out, until 2026-07-01, source: ESPN
_LINE = re.compile(r"^-\s*(?P<player>[^:]+):\s*(?P<status>out|doubtful|fit|suspended)\b(?P<rest>.*)$", re.I)
_MIN = re.compile(r"(\d+)\s*%\s*min", re.I)
_UNTIL = re.compile(r"until\s+(\d{4}-\d{2}-\d{2})", re.I)
_SOURCE = re.compile(r"source:\s*(.+?)\s*$", re.I)
_TEAM_FM = re.compile(r'(?m)^team:\s*"?([^"\n]+)"?\s*$')
_BLOCK = re.compile(r"<!-- WC26:HUMAN:override -->(.*?)(?=\n##\s|\Z)", re.S)


def parse_override_block(text: str) -> list[dict]:
    """Extract structured availability overrides from a country note's HUMAN block."""
    block = _BLOCK.search(text)
    if not block:
        return []
    out = []
    for raw in block.group(1).splitlines():
        line = raw.strip()
        m = _LINE.match(line)
        if not m:
            continue  # skips the italic example lines (they start with "_-")
        rest = m.group("rest")
        minutes = _MIN.search(rest)
        until = _UNTIL.search(rest)
        source = _SOURCE.search(rest)
        status = m.group("status").lower()
        out.append({
            "player": m.group("player").strip(),
            "status": status,
            "minutes_factor": (int(minutes.group(1)) / 100.0) if minutes
            else (0.0 if status in ("out", "suspended") else 0.5),
            "expected_return": until.group(1) if until else None,
            "source": source.group(1).strip() if source else "vault",
            "quote": line,
        })
    return out


def sync_vault_overrides(conn) -> dict:
    """Parse every country note's HUMAN override block into availability_events.

    Human overrides are authoritative, so prior vault-sourced events are replaced.
    """
    countries = config.VAULT / "Countries"
    conn.execute("DELETE FROM availability_events WHERE source LIKE 'vault%'")
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    if not countries.exists():
        return {"events": 0}
    for note in countries.glob("*.md"):
        text = note.read_text()
        team_m = _TEAM_FM.search(text)
        if not team_m:
            continue
        team = team_m.group(1).strip()
        row = conn.execute("SELECT id FROM teams WHERE name=?", (team,)).fetchone()
        if not row:
            continue
        for ov in parse_override_block(text):
            conn.execute(
                "INSERT INTO availability_events (team_id, player_name, status, "
                "expected_return, minutes_factor, source, source_quote, confidence, "
                "fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,?,?,1)",
                (row["id"], ov["player"], ov["status"], ov["expected_return"],
                 ov["minutes_factor"], "vault:" + ov["source"], ov["quote"], 1.0, now),
            )
            n += 1
    conn.commit()
    return {"events": n}


def add_result(conn, home: str, away: str, home_goals: int, away_goals: int,
               stage: str = "group", source: str = "manual",
               played_on: str | None = None, neutral: int = 1) -> int | None:
    """Record a played result: update the fixture and append to the results ledger.

    played_on should be the actual match date (so dedup and the chronological Elo
    replay are correct); tournament is derived from stage so the Elo K factor is
    right (friendlies use a smaller K than World Cup matches). Returns the bound
    fixture id, or None if no fixture matched (e.g. a knockout result, whose fixture
    is not pre-loaded)."""
    hid = conn.execute("SELECT id FROM teams WHERE name=?", (home,)).fetchone()
    aid = conn.execute("SELECT id FROM teams WHERE name=?", (away,)).fetchone()
    now = datetime.now(timezone.utc).isoformat()
    played_on = played_on or now[:10]
    tournament = "Friendly" if stage == "friendly" else "FIFA World Cup"
    match_id = None
    if hid and aid:
        m = conn.execute(
            "SELECT id FROM matches WHERE home_team_id=? AND away_team_id=? AND stage=?",
            (hid["id"], aid["id"], stage),
        ).fetchone()
        if m:                                       # fixture stored in the same orientation
            match_id = m["id"]
            conn.execute(
                "UPDATE matches SET home_goals=?, away_goals=?, played=1 WHERE id=?",
                (home_goals, away_goals, match_id),
            )
        elif stage != "friendly":
            # Neutral-venue tournament fixtures carry FIFA's nominal home/away, which can
            # be the reverse of how the result source (ESPN) reports the match. Bind to the
            # reversed fixture and swap the goals back into the fixture's orientation, so the
            # frozen pre-match snapshot still grades (the ledger keeps the source orientation,
            # which is irrelevant to the orientation-agnostic Elo replay). Restricted to
            # tournament stages: a group pairing is unique (each pair plays once), so the
            # reverse can never bind the wrong match. Friendlies are excluded because the same
            # pair can meet twice in a window and the host is the real home side anyway.
            rev = conn.execute(
                "SELECT id FROM matches WHERE home_team_id=? AND away_team_id=? AND stage=?",
                (aid["id"], hid["id"], stage),
            ).fetchone()
            if rev:
                match_id = rev["id"]
                conn.execute(
                    "UPDATE matches SET home_goals=?, away_goals=?, played=1 WHERE id=?",
                    (away_goals, home_goals, match_id),
                )
    conn.execute(
        "INSERT INTO results_ledger (match_id, played_on, home_team, away_team, "
        "home_goals, away_goals, neutral, tournament, source, fetched_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (match_id, played_on, home, away, home_goals, away_goals, neutral, tournament, source, now),
    )
    conn.commit()
    return match_id


def add_availability(conn, team: str, player: str, status: str,
                     minutes_factor: float = 0.0, source: str = "manual") -> None:
    """Manually record a single availability event (e.g. a late injury)."""
    row = conn.execute("SELECT id FROM teams WHERE name=?", (team,)).fetchone()
    if not row:
        raise ValueError(f"unknown team: {team}")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, source_quote, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,?,1)",
        (row["id"], player, status, minutes_factor, "manual:" + source,
         f"{player}: {status}", 1.0, now),
    )
    conn.commit()


def availability_multipliers(conn, team_players: dict[str, list[dict]]) -> dict[str, float]:
    """Per-team attacking-xG multiplier from current availability events.

    A missing/doubtful player removes (a fraction of) his share of the team's goals,
    bounded so a single event cannot collapse a team's attack.
    """
    from .scorers import team_goal_shares

    by_team_id = {}
    for r in conn.execute("SELECT id, name FROM teams"):
        by_team_id[r["id"]] = r["name"]

    out: dict[str, float] = {name: 1.0 for name in team_players}
    rows = conn.execute(
        "SELECT team_id, player_name, status, minutes_factor FROM availability_events"
    ).fetchall()
    reductions: dict[str, float] = {}
    for r in rows:
        team = by_team_id.get(r["team_id"])
        if team not in team_players:
            continue
        shares = team_goal_shares(team_players[team])
        share = shares.get(r["player_name"], 0.0)
        missing = 1.0 - (r["minutes_factor"] or 0.0)  # 'out' -> 1.0, doubtful 50% -> 0.5
        reductions[team] = reductions.get(team, 0.0) + share * missing
    for team, red in reductions.items():
        out[team] = max(0.4, 1.0 - min(0.6, red))  # cap total reduction at 60%
    return out
