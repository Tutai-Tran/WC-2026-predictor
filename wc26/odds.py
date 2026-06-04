"""The Odds API integration: World Cup winner (outright) odds.

Used ONLY as a prediction signal and calibration benchmark, never for betting.
Odds are devigged (overround removed) and compared to the model's champion
probabilities. The devigged market probabilities are stored in public_benchmark.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

from . import config, db

API = "https://api.the-odds-api.com/v4"
WINNER_SPORT = "soccer_fifa_world_cup_winner"

# The Odds API outright name -> our team name
NAME_ALIASES = {
    "USA": "United States", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "Czechia": "Czech Republic",
}


def _key() -> str | None:
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key
    env = config.REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("THE_ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_winner_odds(key: str | None = None, region: str = "eu"):
    """Return (events, requests_remaining). Raises on a missing key / HTTP error."""
    key = key or _key()
    if not key:
        raise RuntimeError("No THE_ODDS_API_KEY (set it in .env)")
    url = (f"{API}/sports/{WINNER_SPORT}/odds/?regions={region}"
           f"&markets=outrights&oddsFormat=decimal&apiKey={key}")
    with urllib.request.urlopen(url, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        return json.load(r), remaining


def devig_outrights(events: list) -> dict[str, float]:
    """Average decimal price per team across bookmakers, then implied probability
    with the overround removed (normalised to sum to 1)."""
    prices: dict[str, list[float]] = {}
    for event in events:
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "outrights":
                    continue
                for oc in mk.get("outcomes", []):
                    prices.setdefault(oc["name"], []).append(float(oc["price"]))
    implied = {t: 1.0 / (sum(ps) / len(ps)) for t, ps in prices.items()}
    total = sum(implied.values()) or 1.0
    return {NAME_ALIASES.get(t, t): v / total for t, v in implied.items()}


def store_benchmark(conn, market_probs: dict[str, float]) -> None:
    conn.execute(
        "INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) VALUES (?,?,?,?)",
        ("the-odds-api", "champion", json.dumps(market_probs),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def latest_market_champion(conn) -> dict[str, float]:
    row = conn.execute(
        "SELECT payload_json FROM public_benchmark WHERE source='the-odds-api' "
        "AND scope='champion' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row["payload_json"]) if row else {}


def model_vs_market(conn) -> list[dict]:
    market = latest_market_champion(conn)
    row = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    model = {t: p["champion"] for t, p in json.loads(row["payload_json"])["probs"].items()} if row else {}
    rows = []
    for team, mkt in sorted(market.items(), key=lambda kv: kv[1], reverse=True):
        if team in model:
            rows.append({"team": team, "market": round(mkt, 4), "model": round(model[team], 4),
                         "edge": round(model[team] - mkt, 4)})
    return rows


def refresh(conn=None) -> dict:
    conn = conn or db.connect()
    db.init_db(conn)
    events, remaining = fetch_winner_odds()
    market = devig_outrights(events)
    store_benchmark(conn, market)
    return {"teams": len(market), "requests_remaining": remaining,
            "comparison": model_vs_market(conn)}


def main() -> None:
    report = refresh()
    print(f"Odds API requests remaining: {report['requests_remaining']}  "
          f"({report['teams']} teams priced)")
    print(f"\n{'Team':<20}{'Market':>9}{'Model':>9}{'Edge':>9}")
    print("-" * 47)
    for r in report["comparison"][:16]:
        print(f"{r['team']:<20}{r['market']*100:8.1f}%{r['model']*100:8.1f}%{r['edge']*100:+8.1f}%")
    print("\n(Devigged market odds as a benchmark, not betting advice. "
          "Edge = model minus market; treat with caution.)")


if __name__ == "__main__":
    main()
