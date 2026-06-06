from wc26 import db, ingest, forecast
from wc26.model import outcome_label, result_label


def test_run_forecast_end_to_end(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    result = forecast.run_forecast(conn, n_runs=200, seed=5)

    assert len(result["probs"]) == 48
    assert len(result["matches"]) == 72
    assert len(result["golden_boot"]) > 0

    champ_sum = sum(p["champion"] for p in result["probs"].values())
    assert abs(champ_sum - 1.0) < 1e-9

    # every match carries outcome probs that sum to ~1 and scorer lists
    for m in result["matches"]:
        assert abs(m["p_home"] + m["p_draw"] + m["p_away"] - 1.0) < 0.02
        assert len(m["top_scorers_home"]) >= 1

    # append-only snapshot + run metadata persisted
    assert conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM model_runs").fetchone()["c"] == 1


def test_friendly_forecasts_use_stronger_draw_rho(tmp_path):
    """Friendlies get a more-negative Dixon-Coles rho (they under-predict draws), so every
    friendly carries MORE draw mass than at the competitive rho. WC/group matches keep the
    fitted rho and are unaffected."""
    import dataclasses
    from wc26 import model as m, h2h as h2h_mod
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    base = forecast.load_fitted_params() or m.ModelParams()
    h2h = h2h_mod.build_index(cutoff=h2h_mod.PRE_WARMUP_CUTOFF)
    comp = {(f["home"], f["away"]): f["p_draw"]
            for f in forecast.friendly_forecasts(conn, dataclasses.replace(base, rho=-0.06), h2h)}
    friendly = {(f["home"], f["away"]): f["p_draw"]
                for f in forecast.friendly_forecasts(conn, dataclasses.replace(base, rho=-0.15), h2h)}
    assert comp and friendly
    assert all(friendly[k] >= comp[k] - 1e-9 for k in comp)   # never LESS draw mass anywhere
    assert sum(friendly.values()) > sum(comp.values())        # and strictly more in aggregate
    assert base.rho_friendly <= base.rho                      # live config: friendly rho is stronger


def test_top_scoreline_never_contradicts_outcome(tmp_path):
    """The displayed likely score must always agree with the displayed W/D/L."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    result = forecast.run_forecast(conn, n_runs=200, seed=5)
    for m in result["matches"] + result["friendlies"]:
        sh, sa = map(int, m["top_scoreline"].split("-"))
        displayed = outcome_label(m["p_home"], m["p_draw"], m["p_away"])
        assert displayed == result_label(sh, sa), (
            f"{m.get('home')} vs {m.get('away')}: probs say {displayed}, "
            f"score {m['top_scoreline']} says {result_label(sh, sa)}"
        )


def test_played_group_match_carries_result(tmp_path):
    """A played group match exposes its result so the dashboard can judge our call."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    # mark one group fixture as played 3-0
    row = conn.execute(
        "SELECT h.name h, a.name a FROM matches m "
        "JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage='group' LIMIT 1").fetchone()
    conn.execute(
        "UPDATE matches SET home_goals=3, away_goals=0, played=1 "
        "WHERE stage='group' AND home_team_id=(SELECT id FROM teams WHERE name=?) "
        "AND away_team_id=(SELECT id FROM teams WHERE name=?)", (row["h"], row["a"]))
    conn.commit()
    result = forecast.run_forecast(conn, n_runs=200, seed=5)
    played = [m for m in result["matches"] if m["home"] == row["h"] and m["away"] == row["a"]]
    assert played and played[0]["result"] == "3-0"
    # unplayed group matches keep result None
    assert any(m["result"] is None for m in result["matches"])


def test_load_fitted_params_used(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    # calibrated c should be far from the un-calibrated fallback of 110
    assert t.params.c > 150
