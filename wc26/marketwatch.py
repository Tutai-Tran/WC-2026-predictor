"""In-play market snapshotter for closing-line-value (CLV) and market-drift
RESEARCH. Paper only - never places a bet, never touches an account.

Disabled by default. When enabled (env WC26_INPLAY_ODDS=1) and only while a match
is in play, it snapshots the current aggregated market (best price + consensus
implied probability per selection) into `public_benchmark` with a timestamp, so
how the market moved during a match - and where it closed - can be studied later
against the model's pre-match line. It spends Odds API quota, so it is budgeted
per day. The snapshots are research data; nothing here is betting advice.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

from . import db, edge, schedule

SOURCE = "inplay"


def enabled() -> bool:
    """In-play snapshotting is opt-in via WC26_INPLAY_ODDS=1 (default off)."""
    return os.environ.get("WC26_INPLAY_ODDS", "").strip() in {"1", "true", "yes", "on"}


def store_snapshot(
    conn: sqlite3.Connection,
    agg: dict[str, edge.MarketLine],
    scope: str,
    ts: str | None = None,
) -> int:
    """Persist an aggregated market line as a timestamped snapshot. Returns the
    number of selections stored. Payload: {selection: [best_odds, implied, n_books]}."""
    ts = ts or datetime.now(timezone.utc).isoformat()
    payload = {k: [v.best_odds, v.consensus_implied, v.n_books] for k, v in agg.items()}
    conn.execute(
        "INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) VALUES (?,?,?,?)",
        (SOURCE, scope, json.dumps(payload), ts),
    )
    conn.commit()
    return len(payload)


def snapshots_today(conn: sqlite3.Connection, scope: str, day: str | None = None) -> int:
    """Count of in-play snapshots stored for `scope` on the given UTC day."""
    day = day or datetime.now(timezone.utc).date().isoformat()
    row = conn.execute(
        "SELECT COUNT(*) n FROM public_benchmark WHERE source=? AND scope=? "
        "AND substr(fetched_at,1,10)=?",
        (SOURCE, scope, day),
    ).fetchone()
    return row["n"] if row else 0


def latest_snapshot(conn: sqlite3.Connection, scope: str) -> dict[str, edge.MarketLine] | None:
    """The most recent in-play snapshot for `scope`, rebuilt as MarketLines."""
    row = conn.execute(
        "SELECT payload_json FROM public_benchmark WHERE source=? AND scope=? "
        "ORDER BY id DESC LIMIT 1",
        (SOURCE, scope),
    ).fetchone()
    if not row:
        return None
    raw = json.loads(row["payload_json"])
    return {k: edge.MarketLine(best_odds=v[0], consensus_implied=v[1], n_books=int(v[2]))
            for k, v in raw.items()}


def run_snapshot(
    conn: sqlite3.Connection,
    now: datetime,
    kickoffs: list[datetime],
    fetch_events,
    *,
    scope: str = "champion",
    market_key: str = "outrights",
    max_per_day: int = 24,
    force: bool = False,
) -> dict:
    """Take one in-play snapshot if (enabled OR force) AND a match is live AND the
    daily budget allows. `fetch_events` is a zero-arg callable returning Odds API
    events (injected so this is testable offline). Returns a status dict; never
    raises on a fetch error (best-effort)."""
    if not (force or enabled()):
        return {"snapshotted": False, "reason": "disabled"}
    if not schedule.is_live(now, kickoffs):
        return {"snapshotted": False, "reason": "no match in play"}
    if snapshots_today(conn, scope, now.date().isoformat()) >= max_per_day:
        return {"snapshotted": False, "reason": "daily budget reached"}
    try:
        events = fetch_events()
        agg = edge.aggregate_market(events, market_key)
    except Exception as exc:  # network/key/parse issues never break the loop
        return {"snapshotted": False, "reason": f"fetch error: {exc}"}
    if not agg:
        return {"snapshotted": False, "reason": "no market data"}
    n = store_snapshot(conn, agg, scope, now.isoformat())
    return {"snapshotted": True, "scope": scope, "selections": n}


def clv_table(
    conn: sqlite3.Connection,
    scope: str,
    entries: dict[str, float],
    method: str = "shin",
) -> list[dict]:
    """Closing-line value of paper entries vs the latest snapshot's devigged line.

    `entries` maps selection -> the devigged fair probability you took at entry.
    Positive CLV = you took a price implying a lower probability than where the
    market now sits (you beat the line). Latest snapshot is the closing proxy."""
    snap = latest_snapshot(conn, scope)
    if not snap:
        return []
    closing = edge.market_fair_probs(snap, method)
    rows = []
    for sel, entry_fair in entries.items():
        if sel not in closing:
            continue
        rows.append({
            "selection": sel,
            "entry_fair": round(entry_fair, 4),
            "closing_fair": round(closing[sel], 4),
            "clv": round(edge.clv(entry_fair, closing[sel]), 4),
        })
    rows.sort(key=lambda r: r["clv"], reverse=True)
    return rows


def main() -> None:
    """CLI: take one in-play snapshot if conditions allow (no-op otherwise)."""
    from datetime import datetime as _dt

    now = _dt.now(timezone.utc)
    conn = db.connect()
    kickoffs = schedule.upcoming_kickoffs(conn, now)

    def _fetch():
        from .odds import fetch_winner_odds
        events, _ = fetch_winner_odds()
        return events

    status = run_snapshot(conn, now, kickoffs, _fetch)
    print(f"{now.isoformat()} marketwatch: {status}")


if __name__ == "__main__":
    main()
