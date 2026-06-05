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

from . import config, model


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
    elo = {r["name"]: r["elo"] for r in conn.execute(
        "SELECT t.name, r.elo FROM teams t JOIN ratings r ON r.team_id=t.id "
        "WHERE r.valid_from=(SELECT MAX(valid_from) FROM ratings r2 WHERE r2.team_id=t.id)")}
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
        eh, ea = elo.get(m["home"]), elo.get(m["away"])
        # freeze the inputs that produced this prediction so the post-mortem is grounded
        inputs = {"lambda_home": m.get("lambda_home"), "lambda_away": m.get("lambda_away"),
                  "top_scoreline": m.get("top_scoreline"), "elo_home": eh, "elo_away": ea,
                  "elo_gap": (round(eh - ea, 1) if eh is not None and ea is not None else None)}
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


# --------------------------------------------------------------------------
# Phase 2: LLM root-cause post-mortems on WRONG predictions, then aggregation.
# The LLM only PROPOSES structured hypotheses (enum-validated); it never writes a
# model parameter. Aggregated biases are an audit trail + (phase 3) candidate fixes.
# --------------------------------------------------------------------------

_FACTORS = {"elo_gap", "home_advantage", "availability", "h2h", "goal_volume",
            "draw", "motivation", "tactical", "variance"}
_DIRECTIONS = {"over", "under"}
_SEGMENTS = {"global", "friendly", "group", "knockout", "host", "mismatch", "close_match"}


def _clamp01(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


def _validate_lessons(raw) -> list[dict]:
    """Keep only well-formed, enum-valid lessons (whitelist guard, like news.py)."""
    out = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        f = str(e.get("factor", "")).lower().strip()
        d = str(e.get("direction", "")).lower().strip()
        s = str(e.get("suggested_segment", "global")).lower().strip()
        ev = str(e.get("evidence", "")).strip()
        if f not in _FACTORS or d not in _DIRECTIONS or s not in _SEGMENTS or not ev:
            continue
        out.append({"factor": f, "direction": d, "magnitude": _clamp01(e.get("magnitude")),
                    "suggested_segment": s, "confidence": _clamp01(e.get("confidence")),
                    "evidence": ev[:500], "summary": str(e.get("summary", ""))[:200]})
    return out


def _analyze(row, timeout: int = 180):
    """Ask the local Claude CLI why this pre-match prediction was wrong.

    Returns a list of raw lesson dicts, or None on any failure (CLI missing,
    timeout, non-JSON output) so the caller can retry next cycle."""
    from .news import _claude_bin, _extract_json_array  # reuse the existing CLI-agent helpers
    binary = _claude_bin()
    if not binary:
        return None
    inp = json.loads(row["inputs_json"] or "{}")
    prompt = (
        "You are a football forecasting analyst doing a LEAK-FREE post-mortem on a wrong "
        "pre-match prediction. Explain why the model's modal pick missed and return structured "
        "lessons a quantitative model can act on.\n\n"
        f"Match: {row['home_team']} vs {row['away_team']} ({row['stage']}, {row['date_utc']}).\n"
        f"Our PRE-match prediction: home {row['p_home']:.0%} / draw {row['p_draw']:.0%} / "
        f"away {row['p_away']:.0%}; pick={row['pick']}; "
        f"expected goals {inp.get('lambda_home')}-{inp.get('lambda_away')}; "
        f"likely score {inp.get('top_scoreline')}; Elo gap (home-away) {inp.get('elo_gap')}.\n"
        f"ACTUAL RESULT: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} "
        f"(a {row['actual']} result).\n\n"
        "Return ONLY a JSON array of lesson objects, no prose. Each object:\n"
        '{"factor": one of '
        '["elo_gap","home_advantage","availability","h2h","goal_volume","draw","motivation","tactical","variance"], '
        '"direction": "over"|"under" (did the model OVER- or UNDER-weight this factor?), '
        '"magnitude": 0..1, "suggested_segment": one of '
        '["global","friendly","group","knockout","host","mismatch","close_match"], '
        '"confidence": 0..1, "evidence": short factual reason, "summary": one line}.\n'
        'If the miss looks like ordinary variance (a fair upset), return exactly '
        '[{"factor":"variance","direction":"over","magnitude":0.0,"suggested_segment":"global",'
        '"confidence":0.6,"evidence":"low-probability result, no model error","summary":"likely noise"}].'
    )
    try:
        import subprocess
        res = subprocess.run([binary, "-p", prompt], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    return _extract_json_array(res.stdout or "")


def run_postmortems(conn, limit: int = 2) -> dict:
    """Analyse up to `limit` of the most-confident WRONG predictions (rotating).

    Bounded LLM cost per cycle; rows stay pending/error and rotate into later cycles.
    Best-effort: a CLI outage never blocks the refresh."""
    rows = conn.execute(
        "SELECT * FROM prediction_log WHERE graded=1 AND correct=0 "
        "AND postmortem_status IN ('pending', 'error') ORDER BY log_loss DESC LIMIT ?",
        (limit,),
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    done = err = 0
    for r in rows:
        raw = _analyze(r)
        if raw is None:                              # CLI down/failed -> retry next cycle
            conn.execute("UPDATE prediction_log SET postmortem_status='error' WHERE id=?", (r["id"],))
            conn.commit()
            err += 1
            continue
        lessons = _validate_lessons(raw)
        try:
            for ln in lessons:
                conn.execute(
                    "INSERT INTO postmortems (prediction_id, match_id, factor, direction, magnitude, "
                    "suggested_segment, confidence, evidence, summary, model_raw, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(prediction_id, factor) DO UPDATE SET "
                    "direction=excluded.direction, magnitude=excluded.magnitude, "
                    "confidence=excluded.confidence, evidence=excluded.evidence, summary=excluded.summary",
                    (r["id"], r["match_id"], ln["factor"], ln["direction"], ln["magnitude"],
                     ln["suggested_segment"], ln["confidence"], ln["evidence"], ln["summary"],
                     json.dumps(raw), now),          # store the RAW model output for audit
                )
            conn.execute("UPDATE prediction_log SET postmortem_status='done' WHERE id=?", (r["id"],))
            conn.commit()
            done += 1
        except Exception:
            conn.rollback()
            err += 1
    return {"analyzed": done, "errors": err}


def aggregate_lessons(conn) -> dict:
    """Project the graded log + post-mortems into a learning summary (leak-free
    accuracy over time, per-segment accuracy, and ranked systematic biases).

    This is a regenerated projection (idempotent), the dashboard/vault read it. It
    does NOT change any model parameter — that's the guarded learner (phase 3)."""
    graded = conn.execute(
        "SELECT stage, correct, brier, log_loss FROM prediction_log WHERE graded=1"
    ).fetchall()
    n = len(graded)
    overall = {
        "n": n,
        "accuracy": round(sum(g["correct"] for g in graded) / n, 3) if n else None,
        "brier": round(sum(g["brier"] for g in graded) / n, 4) if n else None,
        "log_loss": round(sum(g["log_loss"] for g in graded) / n, 4) if n else None,
    }
    segments = {}
    for stage in ("group", "friendly"):
        sub = [g for g in graded if g["stage"] == stage]
        if sub:
            segments[stage] = {"n": len(sub),
                               "accuracy": round(sum(g["correct"] for g in sub) / len(sub), 3)}
    biases = []
    for r in conn.execute(
        "SELECT factor, suggested_segment seg, "
        "SUM(CASE WHEN direction='over' THEN confidence*magnitude ELSE -confidence*magnitude END) signed, "
        "SUM(confidence) wsum, COUNT(*) n "
        "FROM postmortems WHERE factor != 'variance' GROUP BY factor, suggested_segment"
    ):
        biases.append({
            "factor": r["factor"], "segment": r["seg"],
            "direction": "over" if (r["signed"] or 0) >= 0 else "under",
            "strength": round(abs(r["signed"] or 0) / max(1e-9, r["wsum"] or 0), 3),
            "n": r["n"],
        })
    biases.sort(key=lambda b: b["n"], reverse=True)
    return {"overall": overall, "segments": segments, "biases": biases,
            "computed_at": datetime.now(timezone.utc).isoformat()}

