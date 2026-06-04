"""Live results scraping from ESPN's public site JSON (keyless, valid public data)
and the Elo-recompute that closes the self-improving loop.

ESPN's own site data layer (site.api.espn.com) is not a paid API; we read completed
match scores for friendlies and the World Cup, record genuinely new results into the
ledger, and recompute current Elo = base (results.csv) + the ledger of recent results.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, timedelta

from . import config, elo as elo_mod, overrides

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={ymd}"
LEAGUES = {"fifa.friendly": "Friendly", "fifa.world": "FIFA World Cup"}

# ESPN display name -> our team name
ESPN_ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea", "South Korea": "South Korea",
    "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curacao", "IR Iran": "Iran", "Türkiye": "Turkey", "Turkiye": "Turkey",
    "DR Congo": "DR Congo", "Cape Verde Islands": "Cape Verde",
}


def _norm(name: str) -> str:
    return ESPN_ALIASES.get(name, name)


def fetch_espn(league: str, ymd: str) -> list[dict]:
    """Completed matches for a league on a date (YYYYMMDD). [] on any error."""
    url = ESPN.format(league=league, ymd=ymd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 wc26"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception:
        return []
    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type", {})
        if not status.get("completed"):
            continue
        cs = comp.get("competitors", [])
        home = next((c for c in cs if c.get("homeAway") == "home"), None)
        away = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hg, ag = int(home.get("score")), int(away.get("score"))
        except (TypeError, ValueError):
            continue
        out.append({
            "date": (ev.get("date") or "")[:10],
            "home": _norm((home.get("team") or {}).get("displayName", "")),
            "away": _norm((away.get("team") or {}).get("displayName", "")),
            "hg": hg, "ag": ag, "tournament": LEAGUES.get(league, ""),
        })
    return out


def update_results(conn, days_back: int = 12) -> dict:
    """Record newly-completed friendly/World Cup results into the ledger + fixtures."""
    today = date.today()
    added = 0
    for d in range(days_back, -1, -1):
        day = today - timedelta(days=d)
        ymd = day.strftime("%Y%m%d")
        for league in LEAGUES:
            for m in fetch_espn(league, ymd):
                if not m["home"] or not m["away"]:
                    continue
                dup = conn.execute(
                    "SELECT 1 FROM results_ledger WHERE home_team=? AND away_team=? AND played_on=?",
                    (m["home"], m["away"], m["date"]),
                ).fetchone()
                if dup:
                    continue
                stage = "group" if league == "fifa.world" else "friendly"
                overrides.add_result(conn, m["home"], m["away"], m["hg"], m["ag"],
                                     stage=stage, source="espn")
                added += 1
    return {"added": added}


def recompute_elo(conn) -> dict:
    """Current Elo = base (results.csv replay) + the ledger of recent results,
    applied chronologically. Writes replayed_elo.json and updates the ratings table."""
    base_path = config.DATA_RAW / "base_elo.json"
    if not base_path.exists():
        return {"recomputed": False, "reason": "no base_elo.json (run backtest first)"}
    ratings = dict(json.loads(base_path.read_text()).get("ratings", {}))
    rows = conn.execute(
        "SELECT home_team h, away_team a, home_goals hg, away_goals ag, "
        "COALESCE(tournament,'Friendly') t, COALESCE(neutral,1) n FROM results_ledger "
        "WHERE home_goals IS NOT NULL ORDER BY played_on"
    ).fetchall()
    for r in rows:
        rh = ratings.get(r["h"], elo_mod.DEFAULT_ELO)
        ra = ratings.get(r["a"], elo_mod.DEFAULT_ELO)
        k = elo_mod.k_for_tournament(r["t"])
        home_adv = 0.0 if r["n"] else 60.0
        nh, na = elo_mod.update_pair(rh, ra, int(r["hg"]), int(r["ag"]), k=k, home_adv=home_adv)
        ratings[r["h"]], ratings[r["a"]] = nh, na

    today = date.today().isoformat()
    (config.DATA_RAW / "replayed_elo.json").write_text(json.dumps(
        {"updated": today, "source": "base_elo + results ledger",
         "ratings": {k: round(v, 1) for k, v in ratings.items()}}, indent=2))
    updated = 0
    for tr in conn.execute("SELECT id, name FROM teams").fetchall():
        if tr["name"] in ratings:
            conn.execute(
                "INSERT INTO ratings (team_id, elo, valid_from, source) VALUES (?,?,?,?) "
                "ON CONFLICT(team_id, valid_from) DO UPDATE SET elo=excluded.elo, source=excluded.source",
                (tr["id"], float(ratings[tr["name"]]), today, "recompute"),
            )
            updated += 1
    conn.commit()
    return {"recomputed": True, "ledger_matches": len(rows), "teams_updated": updated}


def main() -> None:
    from . import db
    conn = db.connect()
    print("results:", update_results(conn))
    print("elo:", recompute_elo(conn))


if __name__ == "__main__":
    main()
