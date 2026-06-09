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

# ESPN/source display name -> our canonical team name (also used to canonicalise
# base_elo spellings). Maps every observed spelling variant onto the teams-table name
# so a result is never silently dropped to a phantom rating or an ungraded fixture.
ESPN_ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea", "South Korea": "South Korea",
    "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curacao", "IR Iran": "Iran", "Türkiye": "Turkey", "Turkiye": "Turkey",
    "DR Congo": "DR Congo", "Cape Verde Islands": "Cape Verde",
    # observed in the live ledger as unmatched: qualified WC teams whose source spelling
    # differs from ours (would otherwise fail to grade their group matches), plus China.
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Congo DR": "DR Congo", "China": "China PR",
    # non-tournament opponents whose ESPN spelling differs from the results.csv name
    # (their friendlies feed an opponent's Elo; unmatched they'd be a phantom 1500)
    "Kyrgyz Republic": "Kyrgyzstan", "UAE": "United Arab Emirates",
    "Ireland": "Republic of Ireland", "Cabo Verde": "Cape Verde",
}


def _norm(name: str) -> str:
    return ESPN_ALIASES.get(name, name)


def parse_espn(data: dict, league: str = "") -> list[dict]:
    """Pure parser: ESPN scoreboard JSON -> list of completed results. Testable."""
    out = []
    for ev in (data or {}).get("events", []):
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


def fetch_espn(league: str, ymd: str) -> list[dict]:
    """Completed matches for a league on a date (YYYYMMDD). [] on any error."""
    url = ESPN.format(league=league, ymd=ymd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 wc26"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
    except Exception:
        return []
    return parse_espn(data, league)


def update_results(conn, days_back: int | None = None) -> dict:
    """Record newly-completed friendly/World Cup results into the ledger + fixtures.

    During the tournament the window widens to 30 days so a late ESPN correction
    (or a missed scrape day) is never silently lost outside the rescan horizon."""
    if days_back is None:
        days_back = 30 if config.tournament_mode() else 12
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
                                     stage=stage, source="espn", played_on=m["date"])
                added += 1
    return {"added": added}


def recompute_elo(conn) -> dict:
    """Current Elo = base (results.csv replay) + the ledger of recent results,
    applied chronologically. Writes replayed_elo.json and updates the ratings table."""
    base_path = config.DATA_RAW / "base_elo.json"
    if not base_path.exists():
        return {"recomputed": False, "reason": "no base_elo.json (run backtest first)"}
    # Canonicalise base_elo spellings through the same alias map (e.g. "Curaçao" -> our
    # "Curacao") so a qualified team isn't seeded under a name the ledger never matches.
    ratings: dict[str, float] = {}
    for name, elo in json.loads(base_path.read_text()).get("ratings", {}).items():
        ratings[_norm(name)] = elo
    known = set(ratings)                          # authoritative ~336-team name set
    rows = conn.execute(
        "SELECT home_team h, away_team a, home_goals hg, away_goals ag, "
        "COALESCE(tournament,'Friendly') t, COALESCE(neutral,1) n, played_on p FROM results_ledger "
        "WHERE home_goals IS NOT NULL ORDER BY played_on"
    ).fetchall()
    # Canonicalise ledger names so a result recorded under an ESPN spelling variant still
    # feeds the real team's Elo, AND de-duplicate by (home, away, date): once an alias is
    # added, a re-scrape can store a match already present under its old spelling, and that
    # match must be applied to Elo only once. Anything still unknown after aliasing is a
    # genuine gap (phantom DEFAULT_ELO, never grades) -> surface it loudly in the report.
    graded, seen = [], set()
    for r in rows:
        h, a = _norm(r["h"]), _norm(r["a"])
        key = (h, a, r["p"])
        if key in seen:
            continue
        seen.add(key)
        graded.append((h, a, int(r["hg"]), int(r["ag"]), r["t"], r["n"]))
    unmatched = sorted(({g[0] for g in graded} | {g[1] for g in graded}) - known)
    for h, a, hg, ag, tour, neutral in graded:
        rh = ratings.get(h, elo_mod.DEFAULT_ELO)
        ra = ratings.get(a, elo_mod.DEFAULT_ELO)
        k = elo_mod.k_for_tournament(tour)
        home_adv = 0.0 if neutral else 60.0
        nh, na = elo_mod.update_pair(rh, ra, hg, ag, k=k, home_adv=home_adv)
        ratings[h], ratings[a] = nh, na

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
    report = {"recomputed": True, "ledger_matches": len(rows), "teams_updated": updated}
    if unmatched:
        report["unmatched_teams"] = unmatched     # alias gap: add to ESPN_ALIASES
    return report


def main() -> None:
    from . import db
    conn = db.connect()
    print("results:", update_results(conn))
    print("elo:", recompute_elo(conn))


if __name__ == "__main__":
    main()
