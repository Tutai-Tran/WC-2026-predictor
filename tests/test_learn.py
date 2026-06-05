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
    monkeypatch.setattr(learn, "_analyze", lambda row, timeout=180: [
        {"factor": "home_advantage", "direction": "over", "magnitude": 0.5,
         "suggested_segment": "host", "confidence": 0.8, "evidence": "host lost", "summary": "over-rated host"},
        {"factor": "variance", "direction": "over", "magnitude": 0.0, "suggested_segment": "global",
         "confidence": 0.6, "evidence": "upset", "summary": "noise"},
        {"factor": "NONSENSE", "direction": "sideways"},   # invalid -> dropped by the guard
    ])
    rep = learn.run_postmortems(conn, limit=5)
    assert rep["analyzed"] == 1
    # 2 valid lessons written (the invalid one is filtered out)
    assert conn.execute("SELECT COUNT(*) FROM postmortems WHERE prediction_id=?", (pid,)).fetchone()[0] == 2
    assert conn.execute("SELECT postmortem_status FROM prediction_log WHERE id=?", (pid,)).fetchone()[0] == "done"

    agg = learn.aggregate_lessons(conn)
    factors = {b["factor"] for b in agg["biases"]}
    assert "home_advantage" in factors and "variance" not in factors   # variance never a bias
    assert agg["overall"]["n"] >= 1 and agg["overall"]["accuracy"] is not None


def test_postmortem_cli_down_marks_error(tmp_path, monkeypatch):
    conn = _setup(tmp_path)
    learn.snapshot_upcoming(conn)
    pid = _force_wrong(conn)
    monkeypatch.setattr(learn, "_analyze", lambda row, timeout=180: None)   # CLI unavailable
    rep = learn.run_postmortems(conn, limit=5)
    assert rep["errors"] == 1
    assert conn.execute("SELECT postmortem_status FROM prediction_log WHERE id=?", (pid,)).fetchone()[0] == "error"
