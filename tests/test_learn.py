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
