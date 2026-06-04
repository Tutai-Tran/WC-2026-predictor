from wc26 import db, ingest, forecast


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


def test_load_fitted_params_used(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    # calibrated c should be far from the un-calibrated fallback of 110
    assert t.params.c > 150
