"""Self-learning loop: leak-free prediction log + grading (+ later: LLM post-mortems).

Each refresh, BEFORE new results are folded into Elo, we freeze the current
prediction for every still-upcoming match (`snapshot_upcoming`). When a match is
later played, we grade that frozen PRE-match prediction against the result
(`grade_newly_played`). This gives an honest, leak-free record of what we predicted
before kickoff vs what actually happened — the basis for analysing *why* a
prediction was wrong, without the current-Elo leakage that affects the friendlies
accuracy view.

Design note: snapshots reuse the latest forecast's own per-match probabilities (no
duplicated modelling), so the logged prediction is exactly what the dashboard showed.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from . import model


def _match_index(conn) -> dict:
    """(home, away, stage) -> (match_id, played, home_goals, away_goals) for group + friendly matches."""
    idx = {}
    for r in conn.execute(
        "SELECT m.id, m.stage, h.name home, a.name away, m.played, m.home_goals hg, m.away_goals ag "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage IN ('friendly', 'group')"
    ):
        idx[(r["home"], r["away"], r["stage"])] = (r["id"], r["played"], r["hg"], r["ag"])
    return idx


def _latest_forecast(conn) -> dict | None:
    row = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row["payload_json"]) if row else None


def snapshot_upcoming(conn) -> dict:
    """Freeze the current prediction for still-upcoming matches.

    LEAK-FREE CONTRACT: call this BEFORE new results are scraped/folded into Elo,
    so the snapshot reflects only pre-result information. Matches already played
    are skipped (their pre-match snapshot was frozen on an earlier cycle)."""
    payload = _latest_forecast(conn)
    if not payload:
        return {"snapshotted": 0}
    idx = _match_index(conn)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    items = ([("group", m) for m in payload.get("matches", [])]
             + [("friendly", f) for f in payload.get("friendlies", [])])
    for stage, m in items:
        rec = idx.get((m["home"], m["away"], stage))
        if not rec:
            continue
        match_id, played, _hg, _ag = rec
        if played:
            continue
        ph, pdw, pa = m["p_home"], m["p_draw"], m["p_away"]
        pick = model.outcome_label(ph, pdw, pa)
        inputs = {"lambda_home": m.get("lambda_home"), "lambda_away": m.get("lambda_away"),
                  "top_scoreline": m.get("top_scoreline")}
        conn.execute(
            "INSERT INTO prediction_log (match_id, stage, home_team, away_team, date_utc, "
            "p_home, p_draw, p_away, lambda_home, lambda_away, top_scoreline, pick, inputs_json, snapshot_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(match_id) DO UPDATE SET "
            "p_home=excluded.p_home, p_draw=excluded.p_draw, p_away=excluded.p_away, "
            "lambda_home=excluded.lambda_home, lambda_away=excluded.lambda_away, "
            "top_scoreline=excluded.top_scoreline, pick=excluded.pick, "
            "inputs_json=excluded.inputs_json, snapshot_at=excluded.snapshot_at "
            "WHERE prediction_log.graded=0",          # never overwrite a frozen, graded snapshot
            (match_id, stage, m["home"], m["away"], m.get("date"),
             ph, pdw, pa, m.get("lambda_home"), m.get("lambda_away"), m.get("top_scoreline"),
             pick, json.dumps(inputs), now),
        )
        n += 1
    conn.commit()
    return {"snapshotted": n}


def grade_newly_played(conn) -> dict:
    """Grade any frozen pre-match snapshot whose match is now played."""
    rows = conn.execute(
        "SELECT pl.id, pl.p_home, pl.p_draw, pl.p_away, pl.pick, m.home_goals hg, m.away_goals ag "
        "FROM prediction_log pl JOIN matches m ON m.id=pl.match_id "
        "WHERE pl.graded=0 AND m.played=1 AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL"
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    graded = wrong = 0
    for r in rows:
        actual = model.result_label(r["hg"], r["ag"])
        k = {"home": 0, "draw": 1, "away": 2}[actual]
        p = [r["p_home"], r["p_draw"], r["p_away"]]
        brier = sum((p[i] - (1.0 if i == k else 0.0)) ** 2 for i in range(3))
        ll = -math.log(max(1e-12, p[k]))
        ok = 1 if r["pick"] == actual else 0
        conn.execute(
            "UPDATE prediction_log SET graded=1, home_goals=?, away_goals=?, actual=?, correct=?, "
            "brier=?, log_loss=?, graded_at=?, "
            "postmortem_status=CASE WHEN ?=0 THEN 'pending' ELSE 'skipped' END WHERE id=?",
            (r["hg"], r["ag"], actual, ok, round(brier, 4), round(ll, 4), now, ok, r["id"]),
        )
        graded += 1
        wrong += (ok == 0)
    conn.commit()
    return {"graded": graded, "wrong": wrong}
