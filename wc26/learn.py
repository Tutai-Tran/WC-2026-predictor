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
        # freeze the inputs that produced this prediction so the post-mortem is grounded;
        # mult_* and h2h_delta let the panel see an injury-reduced or h2h-nudged lambda
        # for what it was, instead of misattributing the miss to goal_volume
        inputs = {"lambda_home": m.get("lambda_home"), "lambda_away": m.get("lambda_away"),
                  "top_scoreline": m.get("top_scoreline"), "elo_home": eh, "elo_away": ea,
                  "elo_gap": (round(eh - ea, 1) if eh is not None and ea is not None else None),
                  "mult_home": m.get("mult_home", 1.0), "mult_away": m.get("mult_away", 1.0),
                  "h2h_delta": m.get("h2h_delta", 0.0)}
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


# A PANEL of independent analytical lenses. Each is a separate Claude CLI agent that
# looks at the same wrong match from a different angle, so a cause flagged by several
# lenses carries more weight than one lens's hunch (diverse-perspective robustness).
_PANEL = (
    ("quant", "You are a QUANTITATIVE model-calibration analyst. Focus on whether the "
              "rating-driven inputs were mis-weighted: elo_gap (favourite over/under-rated), "
              "goal_volume (too many/few goals expected), home_advantage, draw likelihood."),
    ("context", "You are a TEAM-CONTEXT analyst. Focus on situation, not maths: availability "
                "(injuries/suspensions/rotation), motivation and stakes (a dead rubber or an "
                "experimental friendly line-up), travel/fatigue. Map these to the factor enum."),
    ("tactical", "You are a TACTICAL football analyst. Focus on the on-pitch matchup: style "
                 "clash, game state, set-pieces, a red card. Be honest: if the result was simply "
                 "a fair low-probability upset with no model error, say so via the 'variance' factor."),
    ("draws", "You are a DRAW-PREDICTION analyst. Judge how likely a DRAW was vs the model's "
              "draw probability, in EITHER direction: weigh parity between the sides, "
              "low-scoring/cagey tendencies, defensive setups, and late-equaliser dynamics. "
              "Return factor 'draw' with direction 'under' if a draw was more likely than the "
              "model implied, or 'over' if less likely. Use 'goal_volume' if total goals drove it."),
    ("scoreline", "You are a SCORELINE/expected-goals analyst. Focus on the GOALS, not the "
                  "win/draw/loss pick: was the expected-goals total too high/low (goal_volume) or "
                  "the home/away split wrong, and would a tighter, more draw-prone scoreline "
                  "distribution have fit the actual score better? Map to 'goal_volume' and 'draw'."),
)

_LESSON_SCHEMA = (
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


def _match_brief(row) -> str:
    inp = json.loads(row["inputs_json"] or "{}")
    extras = []
    mh, ma = inp.get("mult_home", 1.0), inp.get("mult_away", 1.0)
    if (mh or 1.0) != 1.0 or (ma or 1.0) != 1.0:
        extras.append(f"availability xG multipliers already applied: home {mh}, away {ma}")
    if inp.get("h2h_delta"):
        extras.append(f"head-to-head supremacy nudge already applied: {inp['h2h_delta']}")
    extra = ("Adjustments baked into the prediction: " + "; ".join(extras) + ".\n") if extras else ""
    return (
        f"Match: {row['home_team']} vs {row['away_team']} ({row['stage']}, {row['date_utc']}).\n"
        f"Our PRE-match prediction: home {row['p_home']:.0%} / draw {row['p_draw']:.0%} / "
        f"away {row['p_away']:.0%}; pick={row['pick']}; "
        f"expected goals {inp.get('lambda_home')}-{inp.get('lambda_away')}; "
        f"likely score {inp.get('top_scoreline')}; Elo gap (home-away) {inp.get('elo_gap')}.\n"
        + extra +
        f"ACTUAL RESULT: {row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} "
        f"(a {row['actual']} result).\n\n"
    )


def _analyze_lens(row, lens_instruction: str, timeout: int = 180):
    """One panel agent (Claude CLI) analysing the miss from a single lens.

    Returns a list of raw lesson dicts, or None on any failure (CLI missing, timeout,
    non-JSON output) so the caller can treat that lens as absent this cycle."""
    from .news import _claude_bin, _extract_json_array  # reuse the existing CLI-agent helpers
    binary = _claude_bin()
    if not binary:
        return None
    prompt = (lens_instruction + " Do a LEAK-FREE post-mortem on this wrong pre-match "
              "prediction and return structured lessons a quantitative model can act on.\n\n"
              + _match_brief(row) + _LESSON_SCHEMA)
    try:
        import subprocess
        res = subprocess.run([binary, "-p", prompt], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    return _extract_json_array(res.stdout or "")


def _merge_panel(by_lens: dict[str, list]) -> list[dict]:
    """Merge several lenses' validated lessons into one consensus lesson per factor.

    A factor's net direction is the sign of summed signed confidence*magnitude across
    lenses; confidence is the mean of the agreeing lenses, boosted +25% per extra lens
    that agrees (capped at 1.0). So multi-agent agreement strengthens a factor and a
    lone hunch stays weak — the consensus signal the parameter gate later validates."""
    by_factor: dict[str, list] = {}
    for lens, lessons in by_lens.items():
        for ln in lessons:
            by_factor.setdefault(ln["factor"], []).append((lens, ln))
    merged = []
    for factor, items in by_factor.items():
        signed = sum((l["confidence"] * l["magnitude"]) * (1 if l["direction"] == "over" else -1)
                     for _, l in items)
        if signed > 0:
            direction = "over"
        elif signed < 0:
            direction = "under"
        else:                                         # weighted vote tied -> simple lens majority
            n_over = sum(1 for _, l in items if l["direction"] == "over")
            direction = "over" if n_over * 2 >= len(items) else "under"
        agree = [l for _, l in items if l["direction"] == direction]
        n_agree = len(agree)
        conf = min(1.0, (sum(l["confidence"] for l in agree) / n_agree) * (1.0 + 0.25 * (n_agree - 1)))
        mag = sum(l["magnitude"] for l in agree) / n_agree
        lenses = sorted({lens for lens, l in items if l["direction"] == direction})
        best = max(agree, key=lambda l: l["confidence"])
        merged.append({
            "factor": factor, "direction": direction,
            "magnitude": round(mag, 3), "confidence": round(conf, 3),
            "suggested_segment": best["suggested_segment"],
            "evidence": f"[{n_agree}/{len(_PANEL)} agents: {','.join(lenses)}] {best['evidence']}"[:500],
            "summary": best["summary"],
        })
    return merged


def _analyze_panel(row, timeout: int = 180):
    """Run the whole panel on one wrong match and merge to consensus lessons.

    Returns (merged_lessons, raw_by_lens) or None if EVERY lens failed (so the row
    rotates back in next cycle). Partial panels (some lenses down) still produce a
    result from whichever lenses answered."""
    by_lens, raw_by_lens = {}, {}
    for name, instruction in _PANEL:
        raw = _analyze_lens(row, instruction, timeout=timeout)
        if raw is None:
            continue
        raw_by_lens[name] = raw
        by_lens[name] = _validate_lessons(raw)
    if not raw_by_lens:                              # every lens failed -> retry next cycle
        return None
    return _merge_panel(by_lens), raw_by_lens


def run_postmortems(conn, limit: int = 2) -> dict:
    """Analyse up to `limit` of the most-confident WRONG predictions with the agent panel.

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
        result = _analyze_panel(r)
        if result is None:                           # whole panel down/failed -> retry next cycle
            conn.execute("UPDATE prediction_log SET postmortem_status='error' WHERE id=?", (r["id"],))
            conn.commit()
            err += 1
            continue
        lessons, raw_by_lens = result
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
                     json.dumps(raw_by_lens), now),   # store EVERY lens's raw output for audit
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


# --------------------------------------------------------------------------
# Phase 3a: turn quorum-reaching systematic biases into CANDIDATE parameter nudges.
# Candidates only — never auto-applied. The validated re-fit gate (phase 3b) decides
# adoption out-of-sample, so a transient bias never moves a live parameter. Frozen
# during the tournament until enough new matches accumulate to validate a change.
# --------------------------------------------------------------------------

_QUORUM = 8         # wrong matches tagging a factor before it becomes a candidate (guard vs anecdote)
_QUORUM_GROUP = 5   # during the group stage: ~11 wrong picks are expected across all 72 matches,
                    # so quorum 8 on a single factor would likely never fire inside the window

# factor -> (fitted param, direction implied when the model OVER-weighted that factor).
# This is exactly the set the validated gate (phase 3b) can tune, so the candidate audit
# trail never names a parameter the gate wouldn't actually change. Qualitative factors
# (h2h, availability, motivation, tactical) and home_advantage (inert under the neutral
# eval) have no auto-tunable knob and are recorded as biases only.
# Evidence is split by segment: friendly-tagged lessons tune only the friendly-specific
# rho_friendly (validated on graded friendlies), never the tournament parameters, so a
# friendly-only pattern can no longer over-generalise into the tournament model.
_PARAM_MAP = {
    "goal_volume": ("base_goals", "down"),   # over-predicted goals -> lower base expected goals
    "elo_gap":     ("c", "up"),              # over-rated favourites -> raise Elo-per-goal (less supremacy)
    "draw":        ("rho", "up"),            # over-predicted draws -> rho toward 0 (fewer draws)
}
_PARAM_MAP_FRIENDLY = {
    "draw": ("rho_friendly", "up"),
}
_FLIP = {"up": "down", "down": "up"}
_FRIENDLY_SEGMENTS = ("friendly",)


def gate_thresholds(today: str | None = None) -> tuple[int, int]:
    """(min graded matches, quorum) for the adoption gate; lower during the group
    stage so the loop can act inside the 72-match window, conservative otherwise."""
    if config.group_stage_mode(today):
        return _MIN_EVAL_GROUP, _QUORUM_GROUP
    return _MIN_EVAL, _QUORUM


def _quorum_factors(conn, quorum: int, pool: str) -> list[tuple[str, str, str]]:
    """Quorum-reaching (factor, param, direction) candidates for one evidence pool.

    Pools are keyed on the graded match's ACTUAL stage (objective), not the LLM's
    suggested_segment: pool='friendly' counts lessons from wrong friendlies and maps
    only to friendly params; pool='tournament' counts group/knockout lessons and maps
    to the tournament params. Matchday-3 group draws are excluded from the 'draw'
    factor: simultaneous final-round games where a draw can suit both sides are not
    representative evidence for the tournament-wide draw rate."""
    if pool == "friendly":
        stage_clause, param_map = "pl.stage = 'friendly'", _PARAM_MAP_FRIENDLY
    else:
        stage_clause, param_map = "pl.stage IN ('group', 'knockout')", _PARAM_MAP
    out = []
    for r in conn.execute(
        "SELECT pm.factor factor, "
        "SUM(CASE WHEN pm.direction='over' THEN pm.confidence*pm.magnitude "
        "ELSE -pm.confidence*pm.magnitude END) signed, COUNT(*) n "
        "FROM postmortems pm "
        "JOIN prediction_log pl ON pl.id = pm.prediction_id "
        "LEFT JOIN matches m ON m.id = pm.match_id "
        f"WHERE pm.factor != 'variance' AND {stage_clause} "
        "AND NOT (pm.factor = 'draw' AND pl.stage = 'group' AND COALESCE(m.matchday, 0) = 3) "
        "GROUP BY pm.factor HAVING n >= ?",
        (quorum,),
    ):
        mapped = param_map.get(r["factor"])
        if not mapped:
            continue                                  # qualitative factor — no fitted knob to nudge
        param, over_dir = mapped
        over_weighted = (r["signed"] or 0) >= 0
        out.append((r["factor"], param, over_dir if over_weighted else _FLIP[over_dir], r["n"]))
    return out


def propose_adjustments(conn, quorum: int | None = None) -> dict:
    """Turn quorum-reaching systematic biases into CANDIDATE parameter nudges.

    A factor only becomes a candidate once at least `quorum` wrong matches tagged it
    (so it is systematic, not a single upset). Friendly-tagged evidence is pooled
    separately and can only name friendly-specific parameters. Candidates are written
    for the audit trail only; they are NOT applied — adoption is decided by the
    validated re-fit gate out-of-sample. Writes data/raw/learned_adjustments.json."""
    if quorum is None:
        quorum = gate_thresholds()[1]
    candidates = []
    for pool in ("tournament", "friendly"):
        for factor, param, direction, n in _quorum_factors(conn, quorum, pool):
            candidates.append({
                "factor": factor, "param": param, "direction": direction,
                "evidence_n": n, "pool": pool, "applied": False,
                "note": "candidate only — adoption requires out-of-sample re-fit validation",
            })
    out = {"quorum": quorum, "candidates": candidates,
           "computed_at": datetime.now(timezone.utc).isoformat()}
    path = config.DATA_RAW / "learned_adjustments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------
# Phase 3b: VALIDATED adoption. Actually CHANGE the model's parameters from the
# learned biases — but only a change that lowers log loss on the 2026 results AND
# does not break calibration on the historical held-out set. That dual gate is what
# makes the loop get *more accurate* instead of chasing a single upset.
# --------------------------------------------------------------------------

_MIN_EVAL = 20        # graded TOURNAMENT matches required before the gate may change anything
_MIN_EVAL_GROUP = 12  # one full matchday during the group stage (first adoption ~June 13-14)
_MIN_EVAL_FRIENDLY = 12   # graded friendlies required before rho_friendly may move
_MIN_NEW_BETWEEN_ADOPT = 12   # newly graded matches a pool needs between two adoptions
_NET_MARGIN = 0.005   # (2026 log-loss improvement) - (historical regression) must beat this
_HIST_CAP = 0.01      # historical held-out log loss may never rise by more than this

# Per-step size and hard valid range for each gate-tunable parameter (ranges match the
# bounds backtest.fit() enforces, so an adopted value can never leave the sane region).
_GATE_STEP = {"c": 8.0, "base_goals": 0.06, "rho": 0.015, "rho_friendly": 0.015}
_GATE_RANGE = {"c": (50.0, 600.0), "base_goals": (1.5, 4.0), "rho": (-0.18, 0.0),
               "rho_friendly": (-0.30, 0.0)}


def _feats_2026(conn, stages: tuple[str, ...] = ("group", "knockout"),
                since: str | None = None):
    """Build a backtest-style feature set from graded predictions for the given stages
    (the new data the model must adapt to). Pre-match Elo comes from the frozen
    snapshot, so this stays leak-free. The default excludes friendlies: their
    draw/goal dynamics differ from the tournament; pass stages=('friendly',) to
    validate the friendly-specific parameters; pass `since` (an ISO timestamp) to keep
    only matches graded after it (rollback judges a change on post-adoption data only).
    Returns (feats, n) or None. Venues are treated as neutral — a constant that
    cancels in the current-vs-trial comparison for the tuned params (none of which
    interact with home_adv_elo)."""
    import numpy as np
    omap = {"home": 0, "draw": 1, "away": 2}
    eh, ea, gh, ga, out = [], [], [], [], []
    placeholders = ",".join("?" * len(stages))
    since_clause = "AND graded_at > ? " if since else ""
    params = stages + ((since,) if since else ())
    for r in conn.execute(
        "SELECT inputs_json, home_goals hg, away_goals ag, actual FROM prediction_log "
        "WHERE graded=1 AND home_goals IS NOT NULL AND away_goals IS NOT NULL "
        f"AND stage IN ({placeholders}) {since_clause}",
        params,
    ):
        inp = json.loads(r["inputs_json"] or "{}")
        e_h, e_a = inp.get("elo_home"), inp.get("elo_away")
        if e_h is None or e_a is None or r["actual"] not in omap:
            continue
        eh.append(float(e_h)); ea.append(float(e_a))
        gh.append(int(r["hg"])); ga.append(int(r["ag"])); out.append(omap[r["actual"]])
    n = len(out)
    if n == 0:
        return None
    return ({"elo_h": np.array(eh), "elo_a": np.array(ea), "neutral": np.ones(n, bool),
             "gh": np.array(gh), "ga": np.array(ga), "outcome": np.array(out)}, n)


def _tup(d, rho_key: str = "rho"):                    # (c, base, home_adv_elo, rho, gamma)
    return (d["c"], d["base_goals"], d.get("home_adv_elo", 60.0), d[rho_key], d.get("gamma", 0.0))


def _write_params(fp_path, fp: dict) -> None:
    import os
    tmp = fp_path.with_suffix(".json.tmp")            # atomic write: never leave a partial file
    with open(tmp, "w") as f:
        json.dump(fp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())                          # survive power loss: no zero-length params
    os.replace(tmp, fp_path)


def adopt_adjustments(conn, quorum: int | None = None) -> dict:
    """Apply ONE quorum-backed bias correction to the live fitted parameters per cycle,
    and only a nudge that improves out-of-sample log loss on the graded results so far
    WITHOUT worsening the historical held-out calibration. Tournament-segment evidence
    is validated on graded tournament matches and tunes the tournament params;
    friendly-segment evidence is validated on graded friendlies and may only tune
    rho_friendly. At most one parameter changes per cycle (each candidate judged
    against the same pre-cycle baseline, so they can't compound), every decision is
    logged to model_params_log for audit/rollback, and a self-correcting bias
    direction lets a later cycle step a bad change back."""
    from . import backtest as bt
    min_eval, q = gate_thresholds()
    if quorum is None:
        quorum = q
    fp_path = config.DATA_RAW / "fitted_params.json"
    if not fp_path.exists():
        return {"adopted": 0, "reason": "no fitted_params.json"}

    fp = json.loads(fp_path.read_text())
    for k in ("c", "base_goals", "rho", "rho_friendly"):   # never act on a corrupt params file
        v = fp.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return {"adopted": 0, "reason": f"corrupt fitted_params: {k}={v}"}

    try:                                              # leak-free historical held-out sets
        feats_all, _ratings = bt.prematch_pass(bt.read_results())
        recent = feats_all["year"] >= 2021
        feats_h = {key: val[recent] for key, val in feats_all.items()}
        fr_mask = recent & feats_all["is_friendly"]
        feats_h_friendly = {key: val[fr_mask] for key, val in feats_all.items()}
    except Exception as e:
        return {"adopted": 0, "reason": f"backtest load failed: {e}"}

    # (pool, eval feats requirement, eval set, historical guard, rho key, stages, params)
    pools = []
    new_t = _feats_2026(conn)
    pools.append(("tournament", min_eval, new_t, feats_h, "rho",
                  ("group", "knockout"), ("c", "base_goals", "rho")))
    new_f = _feats_2026(conn, stages=("friendly",))
    pools.append(("friendly", _MIN_EVAL_FRIENDLY, new_f, feats_h_friendly, "rho_friendly",
                  ("friendly",), ("rho_friendly",)))

    now = datetime.now(timezone.utc).isoformat()
    trials, ns, blocked = [], {}, []
    for pool, need, new, feats_hist, rho_key, stages, pool_params in pools:
        cands = [(f, p, d) for f, p, d, _n in _quorum_factors(conn, quorum, pool)]
        n_eval = new[1] if new else 0
        ns[pool] = n_eval
        if not cands:
            continue
        if new is None or n_eval < need:
            blocked.append(f"{pool}: {n_eval}/{need} graded matches")
            continue
        # ANTI-RATCHET: the same evidence must not justify a step every 3h cycle. A
        # pool may adopt again only after a matchday's worth of NEW graded matches
        # since its last adopted change (incremental refitting to the same 12 matches
        # would walk a parameter to its bound while each step passes the per-step gate).
        last = conn.execute(
            "SELECT created_at FROM model_params_log WHERE adopted=1 AND param IN "
            f"({','.join('?' * len(pool_params))}) ORDER BY id DESC LIMIT 1",
            pool_params,
        ).fetchone()
        if last:
            n_new = conn.execute(
                "SELECT COUNT(*) c FROM prediction_log WHERE graded=1 AND graded_at > ? "
                f"AND stage IN ({','.join('?' * len(stages))})",
                (last["created_at"], *stages),
            ).fetchone()["c"]
            if n_new < _MIN_NEW_BETWEEN_ADOPT:
                blocked.append(f"{pool}: {n_new}/{_MIN_NEW_BETWEEN_ADOPT} newly graded "
                               "since last adoption")
                continue
        feats_new = new[0]
        # FIXED pre-cycle baselines: every candidate is judged against the same starting
        # point, and at most ONE parameter changes this cycle, so candidates cannot
        # compound or corrupt each other's evaluation.
        base_new = bt.log_loss(feats_new, _tup(fp, rho_key))
        base_hist = bt.log_loss(feats_hist, _tup(fp, rho_key)) if len(feats_hist["outcome"]) else 0.0
        for factor, param, direction in cands:
            old_val = float(fp[param])
            lo, hi = _GATE_RANGE[param]
            step = _GATE_STEP[param] if direction == "up" else -_GATE_STEP[param]
            new_val = round(min(hi, max(lo, old_val + step)), 4)
            if new_val == old_val:                    # already at a bound; nothing to try
                continue
            trial = dict(fp); trial[param] = new_val
            ll_new = bt.log_loss(feats_new, _tup(trial, rho_key))
            ll_hist = bt.log_loss(feats_hist, _tup(trial, rho_key)) if len(feats_hist["outcome"]) else 0.0
            improve, regress = base_new - ll_new, ll_hist - base_hist
            ok = (ll_new < base_new and regress < _HIST_CAP
                  and improve - max(0.0, regress) > _NET_MARGIN)
            trials.append({"factor": factor, "param": param, "direction": direction,
                           "old": old_val, "new": new_val, "ll_new": ll_new, "ll_hist": ll_hist,
                           "base_new": base_new, "base_hist": base_hist, "n_eval": n_eval,
                           "net": improve - max(0.0, regress), "ok": ok})

    if not trials:
        if blocked:
            reason = "insufficient graded matches: " + "; ".join(blocked)
        elif ns.get("tournament", 0) < min_eval:
            reason = (f"insufficient graded matches "
                      f"(tournament: {ns.get('tournament', 0)}/{min_eval})")
        else:
            reason = f"no quorum candidates (quorum {quorum})"
        return {"adopted": 0, "reason": reason, "n": ns.get("tournament", 0),
                "n_friendly": ns.get("friendly", 0)}

    winner = max((t for t in trials if t["ok"]), key=lambda t: t["net"], default=None)
    for t in trials:
        chosen = t is winner
        conn.execute(
            "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, new_value, "
            "ll_new_before, ll_new_after, ll_hist_before, ll_hist_after, n_eval, adopted, reason, params_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, t["factor"], t["param"], t["direction"], t["old"], t["new"],
             round(t["base_new"], 4), round(t["ll_new"], 4),
             round(t["base_hist"], 4), round(t["ll_hist"], 4),
             t["n_eval"], 1 if chosen else 0,
             f"new {t['base_new']:.4f}->{t['ll_new']:.4f}; hist {t['base_hist']:.4f}->{t['ll_hist']:.4f}",
             json.dumps({**fp, t["param"]: t["new"]} if chosen else fp)),
        )
    conn.commit()
    if winner:
        fp[winner["param"]] = winner["new"]
        _write_params(fp_path, fp)                     # the forecast reads this next run
    return {"adopted": 1 if winner else 0, "n": ns.get("tournament", 0),
            "n_friendly": ns.get("friendly", 0),
            "decisions": [{"factor": t["factor"], "param": t["param"], "direction": t["direction"],
                           "adopted": t is winner,
                           "ll_new": [round(t["base_new"], 4), round(t["ll_new"], 4)]}
                          for t in trials]}


# --------------------------------------------------------------------------
# Phase 3c: automatic ROLLBACK. If an adopted nudge later proves harmful on the
# growing graded set, step it back without waiting for the bias loop to notice.
# --------------------------------------------------------------------------

_ROLLBACK_REGRESS = 0.02      # current params must be this much worse than the pre-change
_ROLLBACK_MIN_NEW = 8         # ...judged only once this many NEW matches graded since the change


def check_rollback(conn) -> dict:
    """Auto-revert the most recent adopted parameter change if, on the matches graded
    AFTER the change (only those: the pre-adoption matches selected the change, so
    scoring them again would bias the comparison toward keeping it), the old value
    clearly outperforms it. Logged like any other decision (factor='rollback'), so
    the audit trail stays complete."""
    from . import backtest as bt
    row = conn.execute(
        "SELECT * FROM model_params_log WHERE adopted=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row or row["factor"] == "rollback":
        return {"checked": False}
    fp_path = config.DATA_RAW / "fitted_params.json"
    if not fp_path.exists():
        return {"checked": False}
    fp = json.loads(fp_path.read_text())
    param = row["param"]
    if fp.get(param) != row["new_value"]:             # changed again since; nothing to judge
        return {"checked": False}
    stages = ("friendly",) if param == "rho_friendly" else ("group", "knockout")
    rho_key = "rho_friendly" if param == "rho_friendly" else "rho"
    n_since = conn.execute(
        "SELECT COUNT(*) c FROM prediction_log WHERE graded=1 AND graded_at > ? "
        f"AND stage IN ({','.join('?' * len(stages))})",
        (row["created_at"], *stages),
    ).fetchone()["c"]
    if n_since < _ROLLBACK_MIN_NEW:
        return {"checked": True, "rolled_back": False, "n_since": n_since}
    new = _feats_2026(conn, stages=stages, since=row["created_at"])
    if new is None:
        return {"checked": True, "rolled_back": False, "n_since": n_since}
    feats, n_eval = new
    old_fp = dict(fp); old_fp[param] = row["old_value"]
    ll_cur = bt.log_loss(feats, _tup(fp, rho_key))
    ll_old = bt.log_loss(feats, _tup(old_fp, rho_key))
    if ll_cur <= ll_old + _ROLLBACK_REGRESS:
        return {"checked": True, "rolled_back": False, "n_since": n_since}
    now = datetime.now(timezone.utc).isoformat()
    fp[param] = row["old_value"]
    _write_params(fp_path, fp)
    conn.execute(
        "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, new_value, "
        "ll_new_before, ll_new_after, ll_hist_before, ll_hist_after, n_eval, adopted, reason, params_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now, "rollback", param, _FLIP[row["direction"]], row["new_value"], row["old_value"],
         round(ll_cur, 4), round(ll_old, 4), None, None, n_eval, 1,
         f"auto-rollback: accuracy regression ({ll_cur:.4f} vs {ll_old:.4f} with old value) "
         f"over {n_since} newly graded matches",
         json.dumps(fp)),
    )
    conn.commit()
    return {"checked": True, "rolled_back": True, "param": param,
            "restored": row["old_value"], "n_since": n_since}

