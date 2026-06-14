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


def test_group_forecast_falls_back_to_pure_model_without_odds(tmp_path):
    """No market line (the historical / pre-odds case): published W/D/L equals the raw
    model probs and the audit-trail p_*_model fields mirror them. This is what keeps
    the all-historical backtest byte-identical (item 1)."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    matches = forecast.group_match_forecasts(t, market={})   # explicitly no odds
    for fx in matches:
        assert fx["p_home"] == fx["p_home_model"]
        assert fx["p_draw"] == fx["p_draw_model"]
        assert fx["p_away"] == fx["p_away_model"]


def test_group_forecast_blends_published_probs_when_odds_present(tmp_path):
    """When a devigged line exists for a fixture, the PUBLISHED W/D/L is the log-pool
    blend (model is preserved untouched in p_*_model). With a market line set deliberately
    away from the model, the published probs must differ from the raw model probs and
    equal blend_wdl at the configured weight."""
    from wc26 import odds
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    base = forecast.group_match_forecasts(t, market={})       # raw model probs per fixture
    fx0 = base[0]
    key = f"{fx0['home']} vs {fx0['away']}"
    # a market line clearly distinct from the model so the blend must move the numbers
    line = {"home": 0.40, "draw": 0.30, "away": 0.30}
    blended = forecast.group_match_forecasts(t, market={key: line})
    target = next(m for m in blended if m["home"] == fx0["home"] and m["away"] == fx0["away"])
    # the raw model probs are preserved for the audit trail
    assert target["p_home_model"] == fx0["p_home"]
    assert target["p_draw_model"] == fx0["p_draw"]
    assert target["p_away_model"] == fx0["p_away"]
    # the published probs equal the log-pool of (model, market) at the live weight
    w = odds.match_blend_weight()
    expect = odds.blend_wdl(
        {"home": fx0["p_home"], "draw": fx0["p_draw"], "away": fx0["p_away"]}, line, w)
    assert target["p_home"] == round(expect["home"], 3)
    assert target["p_draw"] == round(expect["draw"], 3)
    assert target["p_away"] == round(expect["away"], 3)
    # and the published probs moved off the pure model (the blend actually fired)
    assert (target["p_home"], target["p_draw"], target["p_away"]) != (
        fx0["p_home"], fx0["p_draw"], fx0["p_away"])
    assert abs(target["p_home"] + target["p_draw"] + target["p_away"] - 1.0) < 0.01
    # fixtures WITHOUT a line stay pure model
    other = next(m for m in blended if (m["home"], m["away"]) != (fx0["home"], fx0["away"]))
    assert other["p_home"] == other["p_home_model"]


def test_load_fitted_params_used(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    # calibrated c should be far from the un-calibrated fallback of 110
    assert t.params.c > 150


def test_scoreline_distribution_fields_present_and_wellformed(tmp_path):
    """Each match carries the additive distribution fields, the top-5 is sorted desc
    with length 5, and the published W/D/L is byte-identical to recomputing it from
    the same matrix (the new fields are pure read-only aggregation)."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    result = forecast.run_forecast(conn, n_runs=200, seed=5)
    for fx in result["matches"] + result["friendlies"]:
        tops = fx["top_scorelines"]
        assert len(tops) == 5
        probs = [p for _, p in tops]
        assert probs == sorted(probs, reverse=True)            # sorted descending
        cum = sum(probs)
        assert 0.40 <= cum <= 0.70                             # top-5 mass in a sane band
        assert "-" in fx["expected_score"]
        dm = fx["derived_markets"]
        assert abs(dm["over_2_5"] + dm["under_2_5"] - 1.0) < 1e-9
        assert abs(dm["btts_yes"] + dm["btts_no"] - 1.0) < 1e-9
        # backward-compat: the old lone modal field is still there
        assert fx["top_scoreline"] is not None


def test_group_top5_cumulative_mass_in_gate_range(tmp_path):
    """Gate (ii): per-fixture top-5 cumulative mass sits in ~0.50-0.62."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    result = forecast.run_forecast(conn, n_runs=200, seed=5)
    masses = [sum(p for _, p in m["top_scorelines"]) for m in result["matches"]]
    assert len(masses) == 72
    mean_mass = sum(masses) / len(masses)
    assert 0.50 <= mean_mass <= 0.62
    assert all(0.45 <= x <= 0.67 for x in masses)


def test_headline_scoreline_set_expands_beyond_four(tmp_path):
    """Gate (i): the DISTINCT headline-scoreline set across the 72 group fixtures
    expands from the baseline of 4 modal clean-sheet cells to >= 10 once the full
    top-5 distribution is published (the union of all top-5 cells per fixture)."""
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    result = forecast.run_forecast(conn, n_runs=200, seed=5)
    distinct = set()
    for m in result["matches"]:
        for (i, j), _ in m["top_scorelines"]:
            distinct.add((i, j))
    assert len(result["matches"]) == 72
    assert len(distinct) >= 10


def test_wdl_unchanged_vs_recomputed_matrix(tmp_path):
    """Gate (iii): the published W/D/L equals outcome_probs recomputed from the SAME
    forecast call (identical venue/elo/mult/h2h inputs), proving the reporting change
    cannot move any W/D/L number."""
    from wc26 import model as m
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    matches = forecast.group_match_forecasts(t)
    by_pair = {(fx["home"], fx["away"]): fx for fx in t.group_fixtures}
    for fx in matches:
        src = by_pair[(fx["home"], fx["away"])]
        vc = src.get("venue_country")
        f = m.match_forecast(
            t.teams[fx["home"]]["elo"], t.teams[fx["away"]]["elo"], t.params,
            t.host_adv(fx["home"], vc), t.host_adv(fx["away"], vc),
            mult_a=t.mult(fx["home"]), mult_b=t.mult(fx["away"]),
            h2h_delta=t.h2h_delta(fx["home"], fx["away"]),
        )
        # the published W/D/L is the (matchup-conditionally) temperature-scaled triple from
        # the SAME match_forecast call (item 3 global temperature + item A supremacy-conditional
        # de-sharpen); the reporting change (top_scorelines/derived_markets, item 2) still
        # cannot move it, which is what this gate proves. Comparing against f["p_*"] keeps the
        # invariant robust to the conditional temperature without re-deriving its formula here.
        assert fx["p_home"] == round(f["p_home"], 3)
        assert fx["p_draw"] == round(f["p_draw"], 3)
        assert fx["p_away"] == round(f["p_away"], 3)
