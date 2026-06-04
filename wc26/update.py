"""Fast self-improvement loop.

Run:  python -m wc26.update
Ensures base data, parses human vault overrides into the DB, logs the running
calibration of the previous forecast against any newly played results, then
re-runs the forecast (now conditioned on played matches + availability) and
regenerates the vault. Pre-tournament there are no results yet, so the calibration
log is simply empty until matches start.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import db, ingest, overrides, forecast, vaultgen, memory


def log_calibration(conn) -> dict:
    """Score the latest prior forecast's match predictions against played results."""
    snap = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not snap:
        return {"n": 0}
    matches = {(m["home"], m["away"]): m for m in json.loads(snap["payload_json"]).get("matches", [])}
    played = conn.execute(
        "SELECT h.name home, a.name away, m.home_goals hg, m.away_goals ag "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.played=1 AND m.home_goals IS NOT NULL"
    ).fetchall()
    import math
    briers, lls, n = [], [], 0
    for r in played:
        pred = matches.get((r["home"], r["away"]))
        if not pred:
            continue
        p = [pred["p_home"], pred["p_draw"], pred["p_away"]]
        outcome = 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)
        briers.append(sum((p[k] - (1.0 if k == outcome else 0.0)) ** 2 for k in range(3)))
        lls.append(-math.log(max(1e-12, p[outcome])))
        n += 1
    if n == 0:
        return {"n": 0}
    metrics = {"n": n, "brier": round(sum(briers) / n, 4), "log_loss": round(sum(lls) / n, 4),
               "computed_at": datetime.now(timezone.utc).isoformat()}
    conn.execute(
        "INSERT INTO predictions (run_id, scope, ref, payload_json, computed_at) VALUES (?,?,?,?,?)",
        ("calib-" + metrics["computed_at"], "calibration", "running",
         json.dumps(metrics), metrics["computed_at"]),
    )
    conn.commit()
    return metrics


def run(conn=None, n_runs: int = 50_000) -> dict:
    conn = conn or db.connect()
    db.init_db(conn)
    if conn.execute("SELECT COUNT(*) c FROM teams").fetchone()["c"] == 0:
        ingest.ingest_all(conn)
    overrides_report = overrides.sync_vault_overrides(conn)
    from . import scrape
    scraped = scrape.update_results(conn)        # pull newly-completed results
    elo_rep = scrape.recompute_elo(conn)         # fold them into current Elo
    calib = log_calibration(conn)                # score the prior forecast vs played results
    result = forecast.run_forecast(conn, n_runs=n_runs)
    vault = vaultgen.generate(conn, result)
    memory.log_task(
        "Automated refresh (update.py)",
        f"scraped results: {scraped}; elo: {elo_rep}; "
        f"overrides synced: {overrides_report.get('events', 0)} events; "
        f"running calibration: {calib}; vault: {vault}; run_id {result['run_id']}.",
    )
    return {"scraped": scraped, "elo": elo_rep, "overrides": overrides_report,
            "calibration": calib, "vault": vault, "run_id": result["run_id"]}


def main() -> None:
    report = run()
    print("update complete:", json.dumps({k: v for k, v in report.items() if k != "run_id"}))


if __name__ == "__main__":
    main()
