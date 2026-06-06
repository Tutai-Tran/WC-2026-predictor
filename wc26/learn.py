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
    ("draws", "You are a DRAW SPECIALIST. The model has been systematically UNDER-predicting "
              "draws. Judge specifically how likely a DRAW was vs the model's draw probability: "
              "weigh parity between the sides, low-scoring/cagey tendencies, defensive setups, "
              "and late-equaliser dynamics. If a draw was more likely than the model implied, "
              "return factor 'draw' direction 'under'. Use 'goal_volume' if total goals drove it."),
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
    return (
        f"Match: {row['home_team']} vs {row['away_team']} ({row['stage']}, {row['date_utc']}).\n"
        f"Our PRE-match prediction: home {row['p_home']:.0%} / draw {row['p_draw']:.0%} / "
        f"away {row['p_away']:.0%}; pick={row['pick']}; "
        f"expected goals {inp.get('lambda_home')}-{inp.get('lambda_away')}; "
        f"likely score {inp.get('top_scoreline')}; Elo gap (home-away) {inp.get('elo_gap')}.\n"
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

_QUORUM = 8   # wrong matches tagging a factor before it becomes a candidate (guard vs anecdote)

# factor -> (fitted param, direction implied when the model OVER-weighted that factor).
# This is exactly the set the validated gate (phase 3b) can tune, so the candidate audit
# trail never names a parameter the gate wouldn't actually change. Qualitative factors
# (h2h, availability, motivation, tactical) and home_advantage (inert under the neutral
# eval) have no auto-tunable knob and are recorded as biases only.
_PARAM_MAP = {
    "goal_volume": ("base_goals", "down"),   # over-predicted goals -> lower base expected goals
    "elo_gap":     ("c", "up"),              # over-rated favourites -> raise Elo-per-goal (less supremacy)
    "draw":        ("rho", "up"),            # over-predicted draws -> rho toward 0 (fewer draws)
}
_FLIP = {"up": "down", "down": "up"}


def propose_adjustments(conn, quorum: int = _QUORUM) -> dict:
    """Turn quorum-reaching systematic biases into CANDIDATE parameter nudges.

    A factor only becomes a candidate once at least `quorum` wrong matches tagged it
    (so it is systematic, not a single upset). Candidates are written for the audit
    trail only; they are NOT applied — adoption is decided by the validated re-fit gate
    out-of-sample. Returns the candidate list and writes data/raw/learned_adjustments.json."""
    candidates = []
    for r in conn.execute(
        "SELECT factor, "
        "SUM(CASE WHEN direction='over' THEN confidence*magnitude "
        "ELSE -confidence*magnitude END) signed, COUNT(*) n "
        "FROM postmortems WHERE factor != 'variance' GROUP BY factor HAVING n >= ?",
        (quorum,),
    ):
        mapped = _PARAM_MAP.get(r["factor"])
        if not mapped:
            continue                                  # qualitative factor — no fitted knob to nudge
        param, over_dir = mapped
        over_weighted = (r["signed"] or 0) >= 0
        candidates.append({
            "factor": r["factor"], "param": param,
            "direction": over_dir if over_weighted else _FLIP[over_dir],
            "evidence_n": r["n"], "applied": False,
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
_NET_MARGIN = 0.005   # (2026 log-loss improvement) - (historical regression) must beat this
_HIST_CAP = 0.01      # historical held-out log loss may never rise by more than this

# Per-step size and hard valid range for each gate-tunable parameter (ranges match the
# bounds backtest.fit() enforces, so an adopted value can never leave the sane region).
_GATE_STEP = {"c": 8.0, "base_goals": 0.06, "rho": 0.015}
_GATE_RANGE = {"c": (50.0, 600.0), "base_goals": (1.5, 4.0), "rho": (-0.18, 0.0)}


def _feats_2026(conn):
    """Build a backtest-style feature set from the graded TOURNAMENT predictions (the
    new data the model must adapt to). Pre-match Elo comes from the frozen snapshot, so
    this stays leak-free. Friendlies are excluded: their draw/goal dynamics differ from
    the tournament we are tuning for. Returns (feats, n) or None. Venues are treated as
    neutral — a constant that cancels in the current-vs-trial comparison for the tuned
    params (c/base_goals/rho don't interact with home_adv_elo, which the gate never tunes)."""
    import numpy as np
    omap = {"home": 0, "draw": 1, "away": 2}
    eh, ea, gh, ga, out = [], [], [], [], []
    for r in conn.execute(
        "SELECT inputs_json, home_goals hg, away_goals ag, actual FROM prediction_log "
        "WHERE graded=1 AND home_goals IS NOT NULL AND away_goals IS NOT NULL "
        "AND stage IN ('group', 'knockout')"          # tournament distribution only
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


def adopt_adjustments(conn, quorum: int = _QUORUM) -> dict:
    """Apply ONE quorum-backed bias correction to the live fitted parameters per cycle,
    and only a nudge that improves out-of-sample log loss on the tournament results so
    far WITHOUT worsening the historical held-out calibration. At most one parameter
    changes per cycle (each candidate judged against the same pre-cycle baseline, so they
    can't compound), every decision is logged to model_params_log for audit/rollback, and
    a self-correcting bias direction lets a later cycle step a bad change back."""
    import os
    from . import backtest as bt
    fp_path = config.DATA_RAW / "fitted_params.json"
    if not fp_path.exists():
        return {"adopted": 0, "reason": "no fitted_params.json"}
    new = _feats_2026(conn)
    if new is None or new[1] < _MIN_EVAL:
        return {"adopted": 0, "reason": "insufficient graded matches", "n": (new[1] if new else 0)}
    feats_new, n_eval = new

    cands = []
    for r in conn.execute(
        "SELECT factor, SUM(CASE WHEN direction='over' THEN confidence*magnitude "
        "ELSE -confidence*magnitude END) signed, COUNT(*) n FROM postmortems "
        "WHERE factor != 'variance' GROUP BY factor HAVING n >= ?", (quorum,)
    ):
        m = _PARAM_MAP.get(r["factor"])
        if not m:
            continue
        param, over_dir = m
        over = (r["signed"] or 0) >= 0
        cands.append((r["factor"], param, over_dir if over else _FLIP[over_dir]))
    if not cands:
        return {"adopted": 0, "reason": "no quorum candidates", "n": n_eval}

    fp = json.loads(fp_path.read_text())
    for k in ("c", "base_goals", "rho"):              # never act on a corrupt params file
        v = fp.get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return {"adopted": 0, "reason": f"corrupt fitted_params: {k}={v}", "n": n_eval}

    try:                                              # leak-free historical held-out set
        feats_all, _ratings = bt.prematch_pass(bt.read_results())
        mask = feats_all["year"] >= 2021
        feats_h = {key: val[mask] for key, val in feats_all.items()}
    except Exception as e:
        return {"adopted": 0, "reason": f"backtest load failed: {e}", "n": n_eval}

    def tup(d):                                       # (c, base, home_adv_elo, rho, gamma)
        return (d["c"], d["base_goals"], d.get("home_adv_elo", 60.0), d["rho"], d.get("gamma", 0.0))

    # FIXED pre-cycle baselines: every candidate is judged against the same starting point,
    # and at most ONE parameter changes this cycle, so candidates cannot compound or corrupt
    # each other's evaluation. The next cycle continues from the new baseline.
    base_new = bt.log_loss(feats_new, tup(fp))
    base_hist = bt.log_loss(feats_h, tup(fp))
    now = datetime.now(timezone.utc).isoformat()

    trials = []
    for factor, param, direction in cands:
        old_val = float(fp[param])
        lo, hi = _GATE_RANGE[param]
        step = _GATE_STEP[param] if direction == "up" else -_GATE_STEP[param]
        new_val = round(min(hi, max(lo, old_val + step)), 4)
        if new_val == old_val:                        # already at a bound; nothing to try
            continue
        trial = dict(fp); trial[param] = new_val
        ll_new = bt.log_loss(feats_new, tup(trial))
        ll_hist = bt.log_loss(feats_h, tup(trial))
        improve, regress = base_new - ll_new, ll_hist - base_hist
        ok = (ll_new < base_new and regress < _HIST_CAP and improve - max(0.0, regress) > _NET_MARGIN)
        trials.append({"factor": factor, "param": param, "direction": direction, "old": old_val,
                       "new": new_val, "ll_new": ll_new, "ll_hist": ll_hist,
                       "net": improve - max(0.0, regress), "ok": ok})

    winner = max((t for t in trials if t["ok"]), key=lambda t: t["net"], default=None)
    for t in trials:
        chosen = t is winner
        conn.execute(
            "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, new_value, "
            "ll_new_before, ll_new_after, ll_hist_before, ll_hist_after, n_eval, adopted, reason, params_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, t["factor"], t["param"], t["direction"], t["old"], t["new"],
             round(base_new, 4), round(t["ll_new"], 4), round(base_hist, 4), round(t["ll_hist"], 4),
             n_eval, 1 if chosen else 0,
             f"new {base_new:.4f}->{t['ll_new']:.4f}; hist {base_hist:.4f}->{t['ll_hist']:.4f}",
             json.dumps({**fp, t["param"]: t["new"]} if chosen else fp)),
        )
    conn.commit()
    if winner:
        fp[winner["param"]] = winner["new"]
        tmp = fp_path.with_suffix(".json.tmp")        # atomic write: never leave a partial file
        tmp.write_text(json.dumps(fp, indent=2))
        os.replace(tmp, fp_path)                       # the forecast reads this next run -> more accurate
    return {"adopted": 1 if winner else 0, "n": n_eval,
            "decisions": [{"factor": t["factor"], "param": t["param"], "direction": t["direction"],
                           "adopted": t is winner, "ll_new": [round(base_new, 4), round(t["ll_new"], 4)]}
                          for t in trials]}

