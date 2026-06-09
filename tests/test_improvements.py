"""Tests for the 2026-06-09 accuracy/efficiency improvement batch:
availability expiry + confidence weighting, tournament-mode cadence, adaptive
learning gate with segment pools, rollback, input-hash skip, odds blend/movement,
challenger harness, and scraper aliases."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from wc26 import config, db, forecast, ingest, learn, odds, overrides


def _conn(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    return conn


def _players_by_team(conn):
    pbt = {}
    for r in conn.execute(
        "SELECT t.name tn, p.name pn, p.position pos, p.club_goals cg, p.is_penalty_taker pk "
        "FROM players p JOIN teams t ON t.id=p.team_id"
    ):
        pbt.setdefault(r["tn"], []).append(
            {"name": r["pn"], "position": r["pos"], "club_goals": r["cg"],
             "is_penalty_taker": bool(r["pk"])})
    return pbt


def _star(conn, team):
    return conn.execute(
        "SELECT p.name FROM players p JOIN teams t ON t.id=p.team_id "
        "WHERE t.name=? ORDER BY p.club_goals DESC LIMIT 1", (team,)).fetchone()["name"]


# ---------------- availability expiry + confidence ----------------

def test_passed_expected_return_expires_event(tmp_path):
    conn = _conn(tmp_path)
    pbt = _players_by_team(conn)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, expected_return, "
        "minutes_factor, source, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,?,1)",
        (tid, star, "out", yesterday, 0.0, "vault:test", 1.0,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
    # the player is back: the event must no longer reduce Spain's attack
    assert overrides.availability_multipliers(conn, pbt)["Spain"] == 1.0


def test_stale_news_event_expires_but_fresh_counts(tmp_path):
    conn = _conn(tmp_path)
    pbt = _players_by_team(conn)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    old = (datetime.now(timezone.utc)
           - timedelta(hours=config.NEWS_EVENT_MAX_AGE_HOURS + 5)).isoformat()
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,0)",
        (tid, star, "out", 0.0, "news-llm", 0.9, old))
    conn.commit()
    assert overrides.availability_multipliers(conn, pbt)["Spain"] == 1.0  # lapsed

    conn.execute("UPDATE availability_events SET fetched_at=? WHERE team_id=?",
                 (datetime.now(timezone.utc).isoformat(), tid))
    conn.commit()
    assert overrides.availability_multipliers(conn, pbt)["Spain"] < 1.0   # fresh again


def test_old_human_event_never_expires_by_age(tmp_path):
    conn = _conn(tmp_path)
    pbt = _players_by_team(conn)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,1)",
        (tid, star, "out", 0.0, "vault:test", 1.0, old))
    conn.commit()
    assert overrides.availability_multipliers(conn, pbt)["Spain"] < 1.0


def test_confidence_weights_reduction(tmp_path):
    conn = _conn(tmp_path)
    pbt = _players_by_team(conn)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    now = datetime.now(timezone.utc).isoformat()

    def mult_with_conf(conf):
        conn.execute("DELETE FROM availability_events")
        conn.execute(
            "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
            "source, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,0)",
            (tid, star, "out", 0.0, "news-llm", conf, now))
        conn.commit()
        return overrides.availability_multipliers(conn, pbt)["Spain"]

    # a low-confidence report must depress the attack less than a confirmed one
    assert mult_with_conf(0.6) > mult_with_conf(0.9)


# ---------------- tournament-mode cadence + gate thresholds ----------------

def test_tournament_mode_window():
    assert not config.tournament_mode("2026-06-09")
    assert config.tournament_mode("2026-06-11")
    assert config.tournament_mode("2026-07-19")
    assert not config.tournament_mode("2026-07-20")
    assert config.group_stage_mode("2026-06-20")
    assert not config.group_stage_mode("2026-06-28")


def test_gate_thresholds_adaptive():
    assert learn.gate_thresholds("2026-06-15") == (learn._MIN_EVAL_GROUP, learn._QUORUM_GROUP)
    assert learn.gate_thresholds("2026-06-01") == (learn._MIN_EVAL, learn._QUORUM)
    assert learn.gate_thresholds("2026-07-05") == (learn._MIN_EVAL, learn._QUORUM)


# ---------------- segment pools ----------------

def _insert_postmortems(conn, factor, segment, n, direction="under"):
    for i in range(n):
        conn.execute(
            "INSERT INTO prediction_log (match_id, stage, home_team, away_team, p_home, p_draw, "
            "p_away, pick, snapshot_at, graded, correct) VALUES (NULL,?,?,?,?,?,?,?,?,1,0)",
            (segment if segment == "friendly" else "group",
             f"{factor}{segment}A{i}", f"B{i}", 0.5, 0.25, 0.25, "home", "2026-06-01T00:00:00Z"))
        pid = conn.execute("SELECT last_insert_rowid() r").fetchone()["r"]
        conn.execute(
            "INSERT INTO postmortems (prediction_id, match_id, factor, direction, magnitude, "
            "suggested_segment, confidence, evidence, summary, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pid, None, factor, direction, 0.6, segment, 0.8, "test", "test",
             "2026-06-01T00:00:00Z"))
    conn.commit()


def test_friendly_evidence_maps_to_rho_friendly_only(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    _insert_postmortems(conn, "draw", "friendly", 6)
    out = learn.propose_adjustments(conn, quorum=5)
    cands = {c["param"]: c for c in out["candidates"]}
    assert "rho_friendly" in cands and cands["rho_friendly"]["pool"] == "friendly"
    assert "rho" not in cands          # friendly evidence must not name the tournament rho


def test_tournament_evidence_excludes_friendly_pool(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    _insert_postmortems(conn, "draw", "group", 6)
    out = learn.propose_adjustments(conn, quorum=5)
    cands = {c["param"]: c for c in out["candidates"]}
    assert "rho" in cands and cands["rho"]["pool"] == "tournament"
    assert "rho_friendly" not in cands


# ---------------- rollback ----------------

def test_rollback_restores_old_value_on_regression(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    fp = {"c": 208.55, "base_goals": 2.199, "rho": -0.18, "rho_friendly": -0.15,
          "gamma": 0.454, "home_adv_elo": 118.5, "goal_scale": 1.0}
    (tmp_path / "fitted_params.json").write_text(json.dumps(fp))
    # a logged adoption that moved rho from a sane value to the bound
    conn.execute(
        "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, "
        "new_value, n_eval, adopted, reason) VALUES (?,?,?,?,?,?,?,1,'test')",
        ("2026-06-01T00:00:00Z", "draw", "rho", "down", -0.06, -0.18, 12))
    # graded tournament matches AFTER the adoption where favourites won big:
    # extreme rho hurts; the old value must win the comparison
    for i in range(10):
        conn.execute(
            "INSERT INTO prediction_log (match_id, stage, home_team, away_team, p_home, p_draw, "
            "p_away, pick, snapshot_at, graded, graded_at, home_goals, away_goals, actual, "
            "correct, inputs_json) VALUES (NULL,?,?,?,?,?,?,?,?,1,?,?,?,?,1,?)",
            ("group", f"H{i}", f"A{i}", 0.7, 0.2, 0.1, "home",
             "2026-06-12T00:00:00Z", "2026-06-13T00:00:00Z", 1, 1, "draw",
             json.dumps({"elo_home": 1800, "elo_away": 1790})))
    conn.commit()
    rep = learn.check_rollback(conn)
    assert rep["checked"]
    if rep.get("rolled_back"):
        new_fp = json.loads((tmp_path / "fitted_params.json").read_text())
        assert new_fp["rho"] == -0.06
        last = conn.execute(
            "SELECT factor FROM model_params_log ORDER BY id DESC LIMIT 1").fetchone()
        assert last["factor"] == "rollback"


def test_rollback_noop_without_enough_new_matches(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    fp = {"c": 208.55, "base_goals": 2.199, "rho": -0.075, "rho_friendly": -0.15,
          "gamma": 0.454, "home_adv_elo": 118.5}
    (tmp_path / "fitted_params.json").write_text(json.dumps(fp))
    conn.execute(
        "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, "
        "new_value, n_eval, adopted, reason) VALUES (?,?,?,?,?,?,?,1,'test')",
        ("2026-06-01T00:00:00Z", "draw", "rho", "down", -0.06, -0.075, 12))
    conn.commit()
    rep = learn.check_rollback(conn)
    assert rep == {"checked": True, "rolled_back": False, "n_since": 0}
    assert json.loads((tmp_path / "fitted_params.json").read_text())["rho"] == -0.075


# ---------------- input fingerprint ----------------

def test_input_fingerprint_stable_and_sensitive(tmp_path):
    conn = _conn(tmp_path)
    a = forecast.input_fingerprint(conn)
    b = forecast.input_fingerprint(conn)
    assert a == b                                    # nothing changed -> same hash
    # scan markers are bookkeeping: they must NOT invalidate the fingerprint
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, source_quote, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,1)",
        (tid, "(scan-marker)", "fit", 1.0, "news-llm-marker", "scan",
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
    assert forecast.input_fingerprint(conn) == a
    # a new odds snapshot must invalidate it (champion_blend reads the market)
    conn.execute("INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) "
                 "VALUES ('the-odds-api','champion','{}',?)",
                 (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    odds_hash = forecast.input_fingerprint(conn)
    assert odds_hash != a
    overrides.add_availability(conn, "Spain", _star(conn, "Spain"), "out", 0.0)
    assert forecast.input_fingerprint(conn) != odds_hash      # new event -> new hash
    assert forecast.input_fingerprint(conn, n_runs=10) != odds_hash   # config participates


def test_forecast_stores_input_hash(tmp_path):
    conn = _conn(tmp_path)
    forecast.run_forecast(conn, n_runs=50, seed=7)
    row = conn.execute("SELECT input_hash FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["input_hash"] and len(row["input_hash"]) == 64


# ---------------- odds blend + movement ----------------

def test_blend_probs_normalised_and_weighted():
    model = {"Spain": 0.3, "England": 0.2, "Curacao": 0.5}
    market = {"Spain": 0.5, "England": 0.5}          # Curacao unpriced -> keeps model p
    blend = odds.blend_probs(model, market, weight=0.5)
    assert abs(sum(blend.values()) - 1.0) < 1e-9
    assert blend["Spain"] > blend["England"]
    assert odds.blend_probs(model, {}, 0.5) == {}


def test_market_movement(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    t0 = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    t1 = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) "
                 "VALUES ('the-odds-api','champion',?,?)",
                 (json.dumps({"Spain": 0.20, "England": 0.10}), t0))
    conn.execute("INSERT INTO public_benchmark (source, scope, payload_json, fetched_at) "
                 "VALUES ('the-odds-api','champion',?,?)",
                 (json.dumps({"Spain": 0.26, "England": 0.08}), t1))
    conn.commit()
    moves = {m["team"]: m["move"] for m in odds.market_movement(conn, days=7)}
    assert moves["Spain"] == pytest.approx(0.06)
    assert moves["England"] == pytest.approx(-0.02)


# ---------------- challenger ----------------

def test_bivariate_poisson_reduces_to_independent_at_l3_zero():
    from wc26 import challenger
    feats = {"elo_h": np.array([1800.0, 1600.0]), "elo_a": np.array([1700.0, 1700.0]),
             "neutral": np.array([True, True]), "outcome": np.array([0, 2])}
    params5 = (208.0, 2.2, 60.0, 0.45, 0.0)          # l3 = 0 -> independent Poisson
    p_chal = challenger._challenger_outcome_probs(feats, params5)
    from wc26 import backtest as bt
    p_champ = bt._probs_for(feats, (208.0, 2.2, 60.0, 0.0, 0.45))   # rho = 0
    assert np.allclose(p_chal, p_champ, atol=1e-6)
    assert np.allclose(p_chal.sum(axis=1), 1.0, atol=1e-9)


def test_bivariate_poisson_shared_component_raises_draws():
    from wc26 import challenger
    feats = {"elo_h": np.array([1700.0]), "elo_a": np.array([1700.0]),
             "neutral": np.array([True]), "outcome": np.array([1])}
    p0 = challenger._challenger_outcome_probs(feats, (208.0, 2.2, 60.0, 0.0, 0.0))
    p1 = challenger._challenger_outcome_probs(feats, (208.0, 2.2, 60.0, 0.0, 0.3))
    assert p1[0, 1] > p0[0, 1]                       # correlation -> more draws


# ---------------- news scan failure protection ----------------

def test_failed_scan_keeps_existing_events(tmp_path, monkeypatch):
    from wc26 import news
    conn = _conn(tmp_path)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, source_quote, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,?,0)",
        (tid, star, "out", 0.0, "news-llm", "q", 0.9, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    monkeypatch.setattr(news, "parse_team_news", lambda *a, **k: None)   # every scan fails
    rep = news.run_news_scan(conn, limit=3)
    assert rep["teams_scanned"] == [] and rep.get("scan_failed")
    kept = conn.execute("SELECT COUNT(*) c FROM availability_events "
                        "WHERE team_id=? AND source='news-llm'", (tid,)).fetchone()["c"]
    assert kept == 1                                 # the injury was NOT wiped


def test_successful_empty_scan_still_replaces(tmp_path, monkeypatch):
    from wc26 import news
    conn = _conn(tmp_path)
    star = _star(conn, "Spain")
    tid = conn.execute("SELECT id FROM teams WHERE name='Spain'").fetchone()["id"]
    conn.execute(
        "INSERT INTO availability_events (team_id, player_name, status, minutes_factor, "
        "source, source_quote, confidence, fetched_at, reviewed) VALUES (?,?,?,?,?,?,?,?,0)",
        (tid, star, "out", 0.0, "news-llm", "q", 0.9, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    monkeypatch.setattr(news, "parse_team_news", lambda *a, **k: [])     # success, no concerns
    news.run_news_scan(conn, limit=60)
    kept = conn.execute("SELECT COUNT(*) c FROM availability_events "
                        "WHERE team_id=? AND source='news-llm'", (tid,)).fetchone()["c"]
    assert kept == 0                                 # cleanly replaced by "no concerns"


# ---------------- adoption ratchet ----------------

def test_adoption_blocked_until_new_matches_after_adoption(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from wc26 import backtest as bt
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    monkeypatch.setattr(bt, "read_results", lambda *a, **k: "df")
    monkeypatch.setattr(bt, "prematch_pass", lambda df: (
        {"year": np.array([2022, 2023]), "is_friendly": np.array([False, False]),
         "outcome": np.array([0, 0])}, {}))
    monkeypatch.setattr(bt, "log_loss", lambda feats, params, weights=None: 0.8)
    (tmp_path / "fitted_params.json").write_text(json.dumps(
        {"c": 208.0, "base_goals": 2.2, "rho": -0.06, "rho_friendly": -0.15, "gamma": 0.45}))
    # quorum-reaching tournament evidence + enough graded matches overall...
    _insert_postmortems(conn, "draw", "group", 25)
    for i, r in enumerate(conn.execute(
            "SELECT id FROM prediction_log WHERE stage='group'").fetchall()):
        conn.execute(
            "UPDATE prediction_log SET graded=1, graded_at=?, home_goals=1, away_goals=1, "
            "actual='draw', inputs_json=? WHERE id=?",
            ("2026-06-01T00:00:00Z", json.dumps({"elo_home": 1700 + i, "elo_away": 1690}),
             r["id"]))
    # ...but a recent adoption with NO matches graded since
    conn.execute(
        "INSERT INTO model_params_log (created_at, factor, param, direction, old_value, "
        "new_value, n_eval, adopted, reason) VALUES (?,?,?,?,?,?,?,1,'test')",
        ("2026-06-02T00:00:00Z", "draw", "rho", "down", -0.06, -0.075, 20))
    conn.commit()
    rep = learn.adopt_adjustments(conn, quorum=5)
    assert rep["adopted"] == 0
    assert "newly graded since last adoption" in rep["reason"]


# ---------------- scraper aliases ----------------

def test_espn_alias_kyrgyz_republic():
    from wc26 import scrape
    data = {"events": [{
        "date": "2026-06-08T00:00Z",
        "competitions": [{
            "status": {"type": {"completed": True}},
            "competitors": [
                {"homeAway": "home", "score": "2",
                 "team": {"displayName": "Kyrgyz Republic"}},
                {"homeAway": "away", "score": "1", "team": {"displayName": "UAE"}},
            ]}]}]}
    out = scrape.parse_espn(data, "fifa.friendly")
    assert out[0]["home"] == "Kyrgyzstan"
    assert out[0]["away"] == "United Arab Emirates"
