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
MATCH_SPORT = "soccer_fifa_world_cup"

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


def fetch_match_odds(key: str | None = None, region: str = "eu"):
    """Return (events, requests_remaining) for per-fixture 1X2 (h2h) match odds.

    Match markets are deeper and sharper than the outright market; used only as a
    research signal and CLV benchmark, never for betting. Raises on missing key /
    HTTP error. Uses more quota than the single outrights endpoint."""
    key = key or _key()
    if not key:
        raise RuntimeError("No THE_ODDS_API_KEY (set it in .env)")
    url = (f"{API}/sports/{MATCH_SPORT}/odds/?regions={region}"
           f"&markets=h2h&oddsFormat=decimal&apiKey={key}")
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (trusted API host)
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


def blend_probs(model: dict[str, float], market: dict[str, float],
                weight: float = 0.5) -> dict[str, float]:
    """Normalised model+market blend; weight = model share. Teams the market does not
    price keep their model probability (then everything renormalises)."""
    if not market:
        return {}
    blend = {t: (weight * mp + (1 - weight) * market[t]) if t in market else mp
             for t, mp in model.items()}
    total = sum(blend.values()) or 1.0
    return {t: v / total for t, v in blend.items()}


# --------------------------------------------------------------------------
# Per-fixture W/D/L (h2h) market: devig, persist, blend into published probs
# --------------------------------------------------------------------------

#: Defensible prior model share for the published-match log-pool blend until the
#: weight is fitted on historical odds (item 4). Market-heavy: the devigged closing
#: line is the strongest single signal, so the model gets the minority weight.
DEFAULT_MATCH_BLEND_WEIGHT = 0.35


def devig_h2h(events: list) -> dict[str, dict[str, float]]:
    """Per-fixture devigged 1X2 (h2h) probabilities, keyed "Home vs Away".

    Aggregates each fixture's three prices across books in PROBABILITY space (the
    edge.py consensus rule), then Shin-devigs the complete 3-way market so the
    overround is removed and the longshot is not over-stated. Returns
    {"Home vs Away": {"home","draw","away": prob}} with each fixture summing to 1.
    """
    from . import edge
    out: dict[str, dict[str, float]] = {}
    for event in events:
        home = NAME_ALIASES.get(event.get("home_team", ""), event.get("home_team", ""))
        away = NAME_ALIASES.get(event.get("away_team", ""), event.get("away_team", ""))
        if not home or not away:
            continue
        # per-selection list of decimal prices across all books for this fixture
        prices: dict[str, list[float]] = {}
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for oc in mk.get("outcomes", []):
                    name = NAME_ALIASES.get(oc["name"], oc["name"])
                    sel = "home" if name == home else "away" if name == away else "draw"
                    price = float(oc["price"])
                    if price > 1.0:
                        prices.setdefault(sel, []).append(price)
        if {"home", "draw", "away"} - set(prices):
            continue  # need a complete 3-way market to devig validly
        # consensus implied per selection (mean of 1/price), then represent as the
        # odds Shin expects; Shin renormalises and removes the margin
        implied = {sel: sum(1.0 / p for p in ps) / len(ps) for sel, ps in prices.items()}
        fair = edge.devig_shin({sel: 1.0 / q for sel, q in implied.items()})
        out[f"{home} vs {away}"] = {k: fair[k] for k in ("home", "draw", "away")}
    return out


def store_h2h_benchmark(conn, fixture_probs: dict[str, dict[str, float]]) -> None:
    """Persist devigged per-fixture h2h probabilities as scope='match' rows.

    One row per cycle holding the full {fixture: {home,draw,away}} map, mirroring
    the champion benchmark's append-only snapshot pattern."""
    if not fixture_probs:
        return
    conn.execute(
        "INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) VALUES (?,?,?,?)",
        ("the-odds-api", "match", json.dumps(fixture_probs),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def latest_market_h2h(conn) -> dict[str, dict[str, float]]:
    """Most recent devigged per-fixture h2h map, keyed "Home vs Away" ({} if none)."""
    row = conn.execute(
        "SELECT payload_json FROM public_benchmark WHERE source='the-odds-api' "
        "AND scope='match' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row["payload_json"]) if row else {}


def match_blend_weight(default: float = DEFAULT_MATCH_BLEND_WEIGHT) -> float:
    """Model share for the published-match blend: fitted blend_weight if present in
    fitted_params.json (item 4), else the defensible prior (market-heavy)."""
    fp = config.DATA_RAW / "fitted_params.json"
    if fp.exists():
        try:
            w = json.loads(fp.read_text()).get("blend_weight")
            if w is not None:
                return float(w)
        except (OSError, ValueError):
            pass
    return default


def blend_wdl(model_probs: dict[str, float], market_probs: dict[str, float],
              weight: float = DEFAULT_MATCH_BLEND_WEIGHT) -> dict[str, float]:
    """Log (geometric) opinion pool of a 3-way W/D/L model and market distribution.

    p_k proportional to model_k^weight * market_k^(1-weight), renormalised, where
    `weight` is the MODEL share. The log pool is the internally-consistent pool for
    a log-loss objective and, being market-heavy (weight ~0.35), cools a model-hot
    favourite toward the sharp market without ever producing a degenerate 0/1 prob.
    Returns the model probs unchanged if no market line is given.
    """
    if not market_probs:
        return dict(model_probs)
    keys = ("home", "draw", "away")
    eps = 1e-12
    pooled = {
        k: max(model_probs.get(k, 0.0), eps) ** weight
        * max(market_probs.get(k, 0.0), eps) ** (1.0 - weight)
        for k in keys
    }
    total = sum(pooled.values()) or 1.0
    return {k: v / total for k, v in pooled.items()}


def blended_champion(conn, weight: float = 0.5) -> dict[str, float]:
    """Champion probabilities blending the model with the devigged market (normalised).

    The market is the strongest single signal, so a blend is usually better calibrated
    than either alone. weight = model share. Returns {} if either side is missing."""
    market = latest_market_champion(conn)
    row = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not market or not row:
        return {}
    model = {t: p["champion"] for t, p in json.loads(row["payload_json"])["probs"].items()}
    return blend_probs(model, market, weight)


def market_movement(conn, days: int = 7) -> list[dict]:
    """Market re-evaluation signal: change in each team's devigged champion probability
    between the oldest snapshot inside the window and the latest one. A team the
    market is walking up (positive move) is one whose news/form the model may lag."""
    from datetime import timedelta
    row = conn.execute(
        "SELECT id, payload_json, fetched_at FROM public_benchmark WHERE source='the-odds-api' "
        "AND scope='champion' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return []
    latest = json.loads(row["payload_json"])
    cutoff = (datetime.fromisoformat(row["fetched_at"]) - timedelta(days=days)).isoformat()
    old_row = conn.execute(
        "SELECT id, payload_json FROM public_benchmark WHERE source='the-odds-api' "
        "AND scope='champion' AND fetched_at >= ? ORDER BY id ASC LIMIT 1", (cutoff,)
    ).fetchone()
    if not old_row or old_row["id"] == row["id"]:    # need two distinct snapshots
        return []
    old = json.loads(old_row["payload_json"])
    moves = [{"team": t, "market": round(p, 4), "move": round(p - old[t], 4)}
             for t, p in latest.items() if t in old]
    moves.sort(key=lambda m: abs(m["move"]), reverse=True)
    return moves


def refresh(conn=None) -> dict:
    conn = conn or db.connect()
    db.init_db(conn)
    events, remaining = fetch_winner_odds()
    market = devig_outrights(events)
    store_benchmark(conn, market)
    return {"teams": len(market), "requests_remaining": remaining,
            "comparison": model_vs_market(conn)}


def _within_horizon(events: list, hours: float) -> list:
    """Keep only events whose kickoff (`commence_time`) is within `hours` from now.

    Bounds the per-match h2h quota to fixtures that are about to start; events with
    no parseable kickoff are kept (the API only lists upcoming games)."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours)
    kept = []
    for ev in events:
        ct = ev.get("commence_time")
        if not ct:
            kept.append(ev)
            continue
        try:
            t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            kept.append(ev)
            continue
        if now <= t <= cutoff:
            kept.append(ev)
    return kept


def refresh_h2h(conn=None, *, horizon_hours: float = 48.0) -> dict:
    """Fetch, devig, and store per-fixture h2h W/D/L odds for near-term fixtures.

    Skips cleanly (no rows written) when THE_ODDS_API_KEY is absent, exactly like
    the champion path. Only fixtures kicking off within `horizon_hours` are priced,
    bounding the per-match quota. This is also what starts ACCUMULATING odds so the
    blend weight can be fitted later (item 4)."""
    conn = conn or db.connect()
    db.init_db(conn)
    if not _key():
        return {"skipped": "no THE_ODDS_API_KEY", "fixtures": 0}
    try:
        events, remaining = fetch_match_odds()
    except Exception as exc:                     # network/quota issues degrade cleanly
        return {"error": str(exc), "fixtures": 0}
    fixture_probs = devig_h2h(_within_horizon(events, horizon_hours))
    store_h2h_benchmark(conn, fixture_probs)
    return {"fixtures": len(fixture_probs), "requests_remaining": remaining}


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
