"""LLM news agent: research current injuries/suspensions via the local Claude CLI
(your Max subscription, with web search), returning structured availability events.

Guards (per PLAN): the player must be in the official squad (enum), a source quote
is required, and a team's prior LLM-sourced events are replaced on each scan so they
never double-count. These events feed the availability -> attacking-xG channel.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

CLAUDE_BIN_CANDIDATES = ["/opt/homebrew/bin/claude", "/usr/local/bin/claude"]


def _claude_bin() -> str | None:
    env = os.environ.get("WC26_CLAUDE_BIN")
    if env and os.path.exists(env):
        return env
    for c in CLAUDE_BIN_CANDIDATES:
        if os.path.exists(c):
            return c
    return shutil.which("claude")


def _extract_json_array(text: str) -> list:
    text = text.replace("```json", "").replace("```", "").strip()
    # whole thing is the array
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    # otherwise scan for the first '[' that decodes to a list (robust to stray brackets)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            v, _ = decoder.raw_decode(text[i:])
            if isinstance(v, list):
                return v
        except Exception:
            continue
    return []


def parse_team_news(team: str, squad_names: list[str], timeout: int = 200) -> list[dict]:
    """Ask the Claude CLI to research current availability for a team's squad."""
    binary = _claude_bin()
    if not binary or not squad_names:
        return []
    prompt = (
        f"Research the current injury, suspension and availability situation for the {team} "
        f"men's national team ahead of the 2026 World Cup, using reliable web sources "
        f"(ESPN, BBC, official team channels, Transfermarkt, reputable press). "
        f"Consider ONLY these squad players: {', '.join(squad_names)}. "
        f'Return ONLY a JSON array (no prose). Each item: '
        f'{{"player","status":"out|doubtful|suspended","reason","source_url","source_quote"}}. '
        f"Omit players with no concern. If nothing found, return []."
    )
    try:
        res = subprocess.run(
            [binary, "-p", prompt, "--allowedTools", "WebSearch,WebFetch"],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return []
    squad = set(squad_names)
    out = []
    for e in _extract_json_array(res.stdout or ""):
        if not isinstance(e, dict):
            continue
        player = str(e.get("player", "")).strip()
        status = str(e.get("status", "")).lower().strip()
        quote = str(e.get("source_quote", "")).strip()
        if player not in squad or status not in ("out", "doubtful", "suspended") or not quote:
            continue
        out.append({"player": player, "status": status, "reason": e.get("reason", ""),
                    "url": e.get("source_url", ""), "quote": quote})
    return out


def _teams_to_scan(conn, limit: int):
    return conn.execute(
        "SELECT t.id, t.name, "
        "MAX(CASE WHEN ae.source LIKE 'news-llm%' THEN ae.fetched_at END) last "
        "FROM teams t LEFT JOIN availability_events ae ON ae.team_id=t.id "
        "WHERE t.group_letter IS NOT NULL GROUP BY t.id "
        "ORDER BY (last IS NULL) DESC, last ASC LIMIT ?",
        (limit,),
    ).fetchall()


def run_news_scan(conn, limit: int = 4) -> dict:
    """Scan a rotating batch of teams; write validated availability events."""
    now = datetime.now(timezone.utc).isoformat()
    scanned, total = [], 0
    for t in _teams_to_scan(conn, limit):
        squad = [r["name"] for r in conn.execute(
            "SELECT name FROM players WHERE team_id=?", (t["id"],))]
        events = parse_team_news(t["name"], squad[:26])
        # one transaction per team: if any write fails, roll back so we never
        # leave a team with its old events deleted and no new ones + no scan
        # marker (which would make the rotation re-scan it forever).
        n = 0
        try:
            conn.execute("DELETE FROM availability_events WHERE team_id=? AND source LIKE 'news-llm%'",
                         (t["id"],))
            for e in events:
                mf = 0.0 if e["status"] in ("out", "suspended") else 0.5
                conn.execute(
                    "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
                    "source, source_quote, url, confidence, fetched_at, reviewed) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0)",
                    (t["id"], e["player"], e["status"], mf, "news-llm", e["quote"], e["url"], 0.8, now),
                )
                n += 1
            # scan marker so rotation advances even when no concerns are found
            conn.execute(
                "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
                "source, source_quote, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,1)",
                (t["id"], "(scan-marker)", "fit", 1.0, "news-llm-marker", "scan", now),
            )
            conn.commit()
            total += n                     # only count events that were actually committed
            scanned.append(t["name"])
        except Exception:
            conn.rollback()
    return {"teams_scanned": scanned, "events": total}


def main() -> None:
    import sys
    from . import db
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    conn = db.connect()
    print(run_news_scan(conn, limit=limit))


if __name__ == "__main__":
    main()
