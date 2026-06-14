"""Tests for the self-learning prediction log + grading (Phase 1)."""

from wc26 import db, ingest, forecast, learn


def _setup(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    forecast.run_forecast(conn, n_runs=200, seed=5)   # writes a forecast payload
    return conn


def test_snapshot_then_grade(tmp_path):
    conn = _setup(tmp_path)

    snap = learn.snapshot_upcoming(conn)
    # all 72 group matches are upcoming pre-tournament (+ any not-yet-played friendlies)
    assert snap["snapshotted"] >= 72
    assert conn.execute("SELECT COUNT(*) FROM prediction_log WHERE stage='group'").fetchone()[0] == 72

    # mark one group match played and grade
    row = conn.execute(
        "SELECT h.name h, a.name a FROM matches m "
        "JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage='group' LIMIT 1").fetchone()
    conn.execute(
        "UPDATE matches SET home_goals=3, away_goals=0, played=1 "
        "WHERE stage='group' AND home_team_id=(SELECT id FROM teams WHERE name=?) "
        "AND away_team_id=(SELECT id FROM teams WHERE name=?)", (row["h"], row["a"]))
    conn.commit()

    grade = learn.grade_newly_played(conn)
    assert grade["graded"] == 1

    pl = conn.execute(
        "SELECT graded, actual, correct, brier, log_loss, pick FROM prediction_log "
        "WHERE home_team=? AND away_team=?", (row["h"], row["a"])).fetchone()
    assert pl["graded"] == 1
    assert pl["actual"] == "home"                     # 3-0 is a home win
    assert pl["correct"] in (0, 1)
    assert pl["brier"] is not None and pl["log_loss"] is not None
    assert pl["correct"] == (1 if pl["pick"] == "home" else 0)


def test_snapshot_does_not_overwrite_graded(tmp_path):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    # grade one match
    row = conn.execute(
        "SELECT m.id, h.name h, a.name a FROM matches m "
        "JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage='group' LIMIT 1").fetchone()
    conn.execute("UPDATE matches SET home_goals=1, away_goals=2, played=1 WHERE id=?", (row["id"],))
    conn.commit()
    learn.grade_newly_played(conn)
    before = conn.execute("SELECT snapshot_at, graded, actual FROM prediction_log WHERE match_id=?",
                          (row["id"],)).fetchone()
    assert before["graded"] == 1 and before["actual"] == "away"

    # a later snapshot must NOT touch the frozen graded row
    learn.snapshot_upcoming(conn)
    after = conn.execute("SELECT snapshot_at, graded, actual FROM prediction_log WHERE match_id=?",
                         (row["id"],)).fetchone()
    assert after["snapshot_at"] == before["snapshot_at"]     # unchanged
    assert after["graded"] == 1 and after["actual"] == "away"


def test_grade_noop_when_nothing_played(tmp_path):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    assert learn.grade_newly_played(conn) == {"graded": 0, "wrong": 0}


def test_aggregate_lessons_empty(tmp_path):
    conn = _setup(tmp_path)
    agg = learn.aggregate_lessons(conn)                # no graded matches yet
    assert agg["overall"]["n"] == 0 and agg["overall"]["accuracy"] is None
    assert agg["biases"] == [] and agg["segments"] == {}


def _force_wrong(conn):
    """Mark one snapshot as a WRONG graded prediction, deterministically."""
    r = conn.execute("SELECT id, pick FROM prediction_log LIMIT 1").fetchone()
    wrong = "away" if r["pick"] != "away" else "home"
    hg, ag = (0, 2) if wrong == "away" else (2, 0)
    conn.execute("UPDATE prediction_log SET graded=1, correct=0, home_goals=?, away_goals=?, "
                 "actual=?, brier=0.9, log_loss=1.5, postmortem_status='pending' WHERE id=?",
                 (hg, ag, wrong, r["id"]))
    conn.commit()
    return r["id"]


def test_postmortems_write_and_aggregate(tmp_path, monkeypatch):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    pid = _force_wrong(conn)
    # the panel returns (merged_lessons, raw_by_lens); validation already happened per lens
    monkeypatch.setattr(learn, "_analyze_panel", lambda row, timeout=180: ([
        {"factor": "home_advantage", "direction": "over", "magnitude": 0.5,
         "suggested_segment": "host", "confidence": 0.8, "evidence": "host lost", "summary": "over-rated host"},
        {"factor": "variance", "direction": "over", "magnitude": 0.0, "suggested_segment": "global",
         "confidence": 0.6, "evidence": "upset", "summary": "noise"},
    ], {"quant": [{"factor": "home_advantage"}]}))
    rep = learn.run_postmortems(conn, limit=5)
    assert rep["analyzed"] == 1
    # 2 valid lessons written (the invalid one is filtered out)
    assert conn.execute("SELECT COUNT(*) FROM postmortems WHERE prediction_id=?", (pid,)).fetchone()[0] == 2
    assert conn.execute("SELECT postmortem_status FROM prediction_log WHERE id=?", (pid,)).fetchone()[0] == "done"

    agg = learn.aggregate_lessons(conn)
    factors = {b["factor"] for b in agg["biases"]}
    assert "home_advantage" in factors and "variance" not in factors   # variance never a bias
    assert agg["overall"]["n"] >= 1 and agg["overall"]["accuracy"] is not None


def _pm(conn, pid, factor, direction, conf=0.8, mag=0.6):
    conn.execute(
        "INSERT INTO postmortems (prediction_id, match_id, factor, direction, magnitude, "
        "suggested_segment, confidence, evidence, summary, model_raw, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (pid, pid, factor, direction, mag, "global", conf, "e", "s", "[]", "2026-06-05T00:00:00+00:00"))


def test_propose_adjustments_quorum(tmp_path):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)            # real prediction_log rows to satisfy the FK
    ids = [r[0] for r in conn.execute("SELECT id FROM prediction_log LIMIT 30").fetchall()]
    for pid in ids[:8]:                       # quorum reached, maps to a fitted knob -> candidate
        _pm(conn, pid, "goal_volume", "over")
    for pid in ids[8:11]:                     # sub-quorum -> dropped
        _pm(conn, pid, "home_advantage", "under")
    for pid in ids[11:20]:                    # quorum but qualitative (no fitted knob) -> dropped
        _pm(conn, pid, "h2h", "over")
    for pid in ids[20:30]:                    # variance is never a bias
        _pm(conn, pid, "variance", "over")
    conn.commit()

    out = learn.propose_adjustments(conn, quorum=8)
    cands = {c["factor"]: c for c in out["candidates"]}
    assert set(cands) == {"goal_volume"}                      # only the systematic, mappable one
    assert cands["goal_volume"]["param"] == "base_goals"      # the real gate-tunable knob
    assert cands["goal_volume"]["direction"] == "down"        # over-predicted goals -> lower base goals
    assert cands["goal_volume"]["applied"] is False           # candidates are never auto-applied


def test_propose_adjustments_empty(tmp_path):
    conn = _setup(tmp_path)
    out = learn.propose_adjustments(conn)                     # no post-mortems yet
    assert out["candidates"] == []


def test_postmortem_cli_down_marks_error(tmp_path, monkeypatch):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    pid = _force_wrong(conn)
    monkeypatch.setattr(learn, "_analyze_panel", lambda row, timeout=180: None)   # whole panel down
    rep = learn.run_postmortems(conn, limit=5)
    assert rep["errors"] == 1
    assert conn.execute("SELECT postmortem_status FROM prediction_log WHERE id=?", (pid,)).fetchone()[0] == "error"


# ----------------------------- multi-agent panel (Part A) -----------------------------

def test_merge_panel_consensus():
    """Two lenses agreeing on a factor outweigh a lone dissenter and boost confidence;
    a factor only one lens raises stays single-strength and labelled 1/3 agents."""
    by_lens = {
        "quant":    [{"factor": "draw", "direction": "under", "magnitude": 0.5, "confidence": 0.7,
                      "suggested_segment": "friendly", "evidence": "q", "summary": "q"}],
        "context":  [{"factor": "draw", "direction": "under", "magnitude": 0.6, "confidence": 0.8,
                      "suggested_segment": "friendly", "evidence": "c", "summary": "c"}],
        "tactical": [{"factor": "draw", "direction": "over", "magnitude": 0.3, "confidence": 0.4,
                      "suggested_segment": "friendly", "evidence": "t", "summary": "t"},
                     {"factor": "tactical", "direction": "under", "magnitude": 0.5, "confidence": 0.6,
                      "suggested_segment": "global", "evidence": "lone", "summary": "lone"}],
    }
    merged = {m["factor"]: m for m in learn._merge_panel(by_lens)}
    n = len(learn._PANEL)                                     # evidence tags against the panel size
    assert merged["draw"]["direction"] == "under"             # 2 under outvote 1 over
    assert merged["draw"]["confidence"] > 0.75                # consensus boost above the raw mean
    assert f"2/{n} agents" in merged["draw"]["evidence"]
    assert merged["tactical"]["direction"] == "under" and f"1/{n} agents" in merged["tactical"]["evidence"]


# --------------------------- validated adoption gate (Part B) -------------------------

def _grade_with_elo(conn, n, actual="draw", elo_away=1690):
    """Mark n snapshotted GROUP-stage rows graded, with pre-match Elo in inputs_json
    (the gate only evaluates tournament matches, so friendlies don't count)."""
    import json
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM prediction_log WHERE stage='group' LIMIT ?", (n,)).fetchall()]
    hg, ag = (1, 1) if actual == "draw" else ((2, 0) if actual == "home" else (0, 2))
    for i, pid in enumerate(ids):
        conn.execute(
            "UPDATE prediction_log SET graded=1, home_goals=?, away_goals=?, actual=?, correct=0, "
            "brier=0.5, log_loss=1.0, inputs_json=? WHERE id=?",
            (hg, ag, actual, json.dumps({"elo_home": 1700 + i, "elo_away": elo_away}), pid))
    conn.commit()
    return ids


def test_adopt_adjustments_insufficient_data(tmp_path):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)                              # snapshots exist, none graded
    rep = learn.adopt_adjustments(conn)
    assert rep["adopted"] == 0 and "insufficient" in rep["reason"]


def test_adopt_adjustments_no_quorum_candidates(tmp_path):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    _grade_with_elo(conn, 20)                                  # enough data, but no post-mortems
    rep = learn.adopt_adjustments(conn)
    assert rep["adopted"] == 0 and "no quorum" in rep["reason"] and rep["n"] == 20


def _arm_gate(tmp_path, monkeypatch, hist_ll):
    """Set up a conn with 12 graded matches + quorum on 'draw', isolate DATA_RAW, and
    stub the backtest so the historical log loss is `hist_ll(params)` and the 2026 log
    loss rewards a more-negative rho. Returns (conn, raw_dir)."""
    import json, shutil
    import numpy as np
    from wc26 import config, backtest as bt
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    ids = _grade_with_elo(conn, 20)
    for pid in ids[:8]:                                        # quorum on 'draw', under-weighted
        _pm(conn, pid, "draw", "under")
    conn.commit()

    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(config.DATA_RAW / "fitted_params.json", raw / "fitted_params.json")
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)
    monkeypatch.setattr(bt, "read_results", lambda *a, **k: "df")
    monkeypatch.setattr(bt, "prematch_pass", lambda df: (
        {"year": np.array([2022, 2023]), "is_friendly": np.array([False, False]),
         "outcome": np.array([0, 0])}, {}))
    # the 2026 eval feats carry 'elo_h'; the historical subset (from prematch_pass) does not.
    # 2026 log loss falls as rho goes more negative; historical is hist_ll(params).
    monkeypatch.setattr(bt, "log_loss",
                        lambda feats, params: (1.0 + params[3]) if "elo_h" in feats else hist_ll(params))
    return conn, raw


def test_adopt_adjustments_adopts_when_validated(tmp_path, monkeypatch):
    import json
    # historical log loss flat (no regression); 2026 improves as rho goes more negative -> ADOPT
    conn, raw = _arm_gate(tmp_path, monkeypatch, hist_ll=lambda p: 0.80)
    before = json.loads((raw / "fitted_params.json").read_text())["rho"]
    rep = learn.adopt_adjustments(conn)
    after = json.loads((raw / "fitted_params.json").read_text())["rho"]
    assert rep["adopted"] == 1
    assert after < before                                      # rho nudged down and written
    row = conn.execute("SELECT param, adopted FROM model_params_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["param"] == "rho" and row["adopted"] == 1


def test_adopt_adjustments_rejects_when_history_regresses(tmp_path, monkeypatch):
    import json
    # historical log loss worsens past the cap when rho drops -> REJECT despite a 2026 gain
    conn, raw = _arm_gate(tmp_path, monkeypatch, hist_ll=lambda p: 0.80 - p[3])
    before = json.loads((raw / "fitted_params.json").read_text())["rho"]
    rep = learn.adopt_adjustments(conn)
    after = json.loads((raw / "fitted_params.json").read_text())["rho"]
    assert rep["adopted"] == 0
    assert after == before                                     # live params untouched
    row = conn.execute("SELECT adopted FROM model_params_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["adopted"] == 0                                 # decision logged for audit


# --------------------- item 8: temperature-tunable, reachable gate ----------------------

def test_temperature_is_gate_tunable():
    """Item 8: temperature is now in the gate's tunable set with the specified bounds."""
    assert "temperature" in learn._GATE_STEP
    assert learn._GATE_STEP["temperature"] == 0.03            # +/-0.03 de-sharpening step
    assert learn._GATE_RANGE["temperature"] == (0.8, 2.0)     # matches fit_temperature bounds
    # the favourites-overconfident signal maps elo_gap -> raise temperature
    assert learn._PARAM_MAP_CALIB["elo_gap"] == ("temperature", "up")
    assert "temperature" in learn._CALIB_PARAMS


def test_gate_thresholds_calibration_vs_structural():
    """Item 8: calibration knobs reach quorum/min-eval sooner than structural params."""
    # structural (base_goals, rho) keep the conservative group-stage thresholds
    assert learn._MIN_EVAL_GROUP == 12 and learn._QUORUM_GROUP == 5
    # global calibration (temperature, small c) are reachable a touch sooner
    assert learn._MIN_EVAL_GROUP_CALIB == 8 and learn._QUORUM_GROUP_CALIB == 3
    # the thrash guards are UNCHANGED (do not loosen)
    assert learn._NET_MARGIN == 0.005 and learn._HIST_CAP == 0.01
    assert learn._MIN_NEW_BETWEEN_ADOPT == 12
    assert learn._MIN_EVAL == 20                              # non-group min eval untouched


def _arm_temp_gate(tmp_path, monkeypatch, hist_temp_ll, n_graded=8):
    """Arm the gate with n_graded graded GROUP matches + an 'elo_gap over' quorum (3),
    forcing group-stage mode. The 2026 temperature log loss FALLS as T rises (rewards
    de-sharpening); the historical temperature log loss is hist_temp_ll(T). Structural
    bt.log_loss is held flat so only the temperature candidate can move. Returns
    (conn, raw)."""
    import shutil
    import numpy as np
    from wc26 import config, backtest as bt
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    ids = _grade_with_elo(conn, n_graded, actual="home", elo_away=2050)   # elite-vs-elite
    for pid in ids[:3]:                                        # quorum_calib=3 on 'elo_gap', over
        _pm(conn, pid, "elo_gap", "over")
    conn.commit()

    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(config.DATA_RAW / "fitted_params.json", raw / "fitted_params.json")
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)
    monkeypatch.setattr("wc26.config.group_stage_mode", lambda today=None: True)
    monkeypatch.setattr(bt, "read_results", lambda *a, **k: "df")
    monkeypatch.setattr(bt, "prematch_pass", lambda df: (
        {"year": np.array([2022, 2023]), "is_friendly": np.array([False, False]),
         "outcome": np.array([0, 0])}, {}))
    # structural log loss flat (no structural candidate can improve -> only temperature moves)
    monkeypatch.setattr(bt, "log_loss", lambda feats, params: 0.80)
    # temperature path: 2026 (has elo_h) improves with T; historical is hist_temp_ll(T)
    monkeypatch.setattr(bt, "_temp_log_loss",
                        lambda feats, params, T: (1.0 - 0.5 * (T - 1.0)) if "elo_h" in feats
                        else hist_temp_ll(T))
    return conn, raw


def test_gate_adopts_temperature_after_8_overconfident_matches(tmp_path, monkeypatch):
    import json
    # historical temperature log loss FLAT -> de-sharpening passes _NET_MARGIN/_HIST_CAP -> ADOPT
    conn, raw = _arm_temp_gate(tmp_path, monkeypatch, hist_temp_ll=lambda T: 0.80)
    before = json.loads((raw / "fitted_params.json").read_text()).get("temperature", 1.0)
    rep = learn.adopt_adjustments(conn)
    after = json.loads((raw / "fitted_params.json").read_text())["temperature"]
    assert rep["adopted"] == 1
    assert after > before                                      # temperature raised (de-sharpened)
    assert round(after - before, 4) == learn._GATE_STEP["temperature"]
    row = conn.execute("SELECT param, direction, adopted FROM model_params_log "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    assert row["param"] == "temperature" and row["direction"] == "up" and row["adopted"] == 1
    # 8 graded matches is enough for the calibration knob but BELOW the structural min-eval
    assert rep["n"] == 8


def test_gate_rejects_temperature_when_history_regresses(tmp_path, monkeypatch):
    import json
    # historical temperature log loss RISES past _HIST_CAP as T grows -> REJECT despite 2026 gain
    conn, raw = _arm_temp_gate(tmp_path, monkeypatch,
                               hist_temp_ll=lambda T: 0.80 + 0.5 * (T - 1.0))
    before = json.loads((raw / "fitted_params.json").read_text()).get("temperature", 1.0)
    rep = learn.adopt_adjustments(conn)
    after = json.loads((raw / "fitted_params.json").read_text())["temperature"]
    assert rep["adopted"] == 0                                 # _HIST_CAP guard still works
    assert after == before                                     # live temperature untouched
    row = conn.execute("SELECT param, adopted FROM model_params_log "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    assert row["param"] == "temperature" and row["adopted"] == 0   # logged for audit


def test_rollback_reverts_regressing_temperature(tmp_path, monkeypatch):
    """A previously-adopted temperature step that REGRESSES on the matches graded after
    it must be auto-rolled back. Before the fix, check_rollback scored the change through
    _tup (which excludes temperature), so ll_cur==ll_old and a temperature adoption could
    never be reverted, silently denying it the fast safety net every other param has."""
    import json
    from wc26 import config, backtest as bt
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    ids = _grade_with_elo(conn, 9, actual="home", elo_away=2050)        # >= _ROLLBACK_MIN_NEW
    conn.executemany("UPDATE prediction_log SET graded_at=? WHERE id=?",
                     [("2026-06-13T00:00:00+00:00", pid) for pid in ids])
    conn.commit()

    raw = tmp_path / "raw"; raw.mkdir()
    fp = json.loads((config.DATA_RAW / "fitted_params.json").read_text())
    fp["temperature"] = 1.08
    (raw / "fitted_params.json").write_text(json.dumps(fp))
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)
    # current T=1.08 scores WORSE (higher) than old T=1.02 on the post-adoption matches.
    monkeypatch.setattr(bt, "_temp_log_loss", lambda feats, params, T: 0.5 + T)
    monkeypatch.setattr(bt, "log_loss", lambda feats, params: 0.5)      # _tup path must NOT decide this

    conn.execute(
        "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, new_value, "
        "n_eval, adopted, reason, params_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("2026-06-01T00:00:00+00:00", "elo_gap", "temperature", "up", 1.02, 1.08, 9, 1,
         "test adopt", json.dumps(fp)))
    conn.commit()

    rep = learn.check_rollback(conn)
    assert rep["rolled_back"] is True
    assert rep["param"] == "temperature" and rep["restored"] == 1.02
    assert json.loads((raw / "fitted_params.json").read_text())["temperature"] == 1.02
    rb = conn.execute("SELECT factor, ll_new_before, ll_new_after FROM model_params_log "
                      "ORDER BY id DESC LIMIT 1").fetchone()
    assert rb["factor"] == "rollback"
    # temperature-aware scoring (1.58 vs 1.52); the temperature-blind _tup code would tie these
    assert round(rb["ll_new_before"] - rb["ll_new_after"], 2) == 0.06


def test_structural_param_still_blocked_below_12_in_group_stage(tmp_path, monkeypatch):
    import json
    import numpy as np
    import shutil
    from wc26 import config, backtest as bt
    # 8 graded matches with a STRUCTURAL ('draw') quorum: the structural min-eval is 12,
    # so the structural candidate must NOT adopt at 8 (only calibration knobs may).
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    ids = _grade_with_elo(conn, 8)
    for pid in ids[:5]:                                        # quorum 5 on 'draw'
        _pm(conn, pid, "draw", "under")
    conn.commit()
    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(config.DATA_RAW / "fitted_params.json", raw / "fitted_params.json")
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)
    monkeypatch.setattr("wc26.config.group_stage_mode", lambda today=None: True)
    monkeypatch.setattr(bt, "read_results", lambda *a, **k: "df")
    monkeypatch.setattr(bt, "prematch_pass", lambda df: (
        {"year": np.array([2022, 2023]), "is_friendly": np.array([False, False]),
         "outcome": np.array([0, 0])}, {}))
    monkeypatch.setattr(bt, "log_loss",
                        lambda feats, params: (1.0 + params[3]) if "elo_h" in feats else 0.80)
    before = json.loads((raw / "fitted_params.json").read_text())["rho"]
    rep = learn.adopt_adjustments(conn)
    after = json.loads((raw / "fitted_params.json").read_text())["rho"]
    assert rep["adopted"] == 0                                 # 8 < structural _MIN_EVAL_GROUP (12)
    assert after == before
    assert "insufficient" in rep["reason"] and "8/12" in rep["reason"]
