import numpy as np
import pandas as pd

from wc26 import backtest


def test_prematch_pass_is_leak_free():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2000-02-01"]),
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "home_score": [1, 2], "away_score": [0, 0],
        "tournament": ["Friendly", "Friendly"], "neutral": [True, True],
    })
    feats, ratings = backtest.prematch_pass(df)
    # first ever match: both teams recorded at the default (no look-ahead)
    assert feats["elo_h"][0] == 1500.0 and feats["elo_a"][0] == 1500.0
    # second match: A's pre-match Elo reflects the first win
    assert feats["elo_h"][1] > 1500.0


def test_outcome_probs_rows_sum_to_one():
    la = np.array([1.8, 0.5, 3.0]); lb = np.array([1.1, 2.0, 0.3])
    p = backtest._outcome_probs(la, lb, rho=-0.06)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def _synthetic_goal_feats(seed=0, n=4000):
    """Synthetic prematch feats with a realistic ~2.7 mean total and Elo spread."""
    rng = np.random.default_rng(seed)
    elo_h = rng.normal(1700, 120, n)
    elo_a = rng.normal(1700, 120, n)
    neutral = rng.random(n) < 0.5
    # ground-truth lambdas around a ~2.7 base so the goal-aware fit has a target to hit
    sup = (elo_h - elo_a) / 200.0
    la = np.maximum(0.15, (2.7 + sup) / 2.0)
    lb = np.maximum(0.15, (2.7 - sup) / 2.0)
    gh = rng.poisson(la)
    ga = rng.poisson(lb)
    outcome = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    return {"elo_h": elo_h, "elo_a": elo_a, "neutral": neutral,
            "gh": gh, "ga": ga, "outcome": outcome,
            "year": np.full(n, 2015), "is_major": np.zeros(n, bool),
            "is_friendly": np.zeros(n, bool),
            "tournament": np.array([""] * n, dtype=object)}


def test_mean_total_goals_tracks_base_goals():
    """mean_total_goals rises monotonically with base_goals (the goal-aware knob)."""
    feats = _synthetic_goal_feats(seed=1)
    home = 40.0
    low = backtest.mean_total_goals(feats, (200.0, 2.0, home, -0.06, 0.0))
    high = backtest.mean_total_goals(feats, (200.0, 3.0, home, -0.06, 0.0))
    assert high - low > 0.9          # +1.0 to base lifts mean total by ~1.0
    assert 1.9 < low < 2.1 and 2.9 < high < 3.1


def test_refit_base_goals_raises_toward_real_goal_volume():
    """The goal-aware refit moves base_goals from a WDL-low value toward the value
    that matches the actual mean total, without exploding W/D/L log-loss (item 5)."""
    feats = _synthetic_goal_feats(seed=2)
    home = 40.0
    actual = float(np.mean(feats["gh"] + feats["ga"]))   # ~2.7 by construction
    # a deliberately goal-blind low base (mimics the 2.19 WDL optimum)
    params_low = backtest.model_mod.ModelParams(c=200.0, base_goals=2.1, rho=-0.06, gamma=0.0)
    mpt_low = backtest.mean_total_goals(
        feats, (200.0, 2.1, home, -0.06, 0.0))
    assert mpt_low < actual - 0.1                          # starts under-predicting
    new_base = backtest.refit_base_goals(feats, params_low, home, actual,
                                         lambda_tot=1.0)
    assert new_base > 2.1                                  # raised
    mpt_new = backtest.mean_total_goals(
        feats, (200.0, new_base, home, -0.06, 0.0))
    assert abs(mpt_new - actual) < 0.06                    # lands within the gate band
    # only base_goals changed; it stays in bounds
    assert 1.5 <= new_base <= 4.0


def test_refit_base_goals_lambda_tot_zero_is_wdl_optimum():
    """With lambda_tot=0 the goal-aware refit reduces to the pure W/D/L base_goals."""
    feats = _synthetic_goal_feats(seed=3)
    home = 40.0
    params = backtest.model_mod.ModelParams(c=200.0, base_goals=2.5, rho=-0.06, gamma=0.0)
    actual = float(np.mean(feats["gh"] + feats["ga"]))
    b0 = backtest.refit_base_goals(feats, params, home, actual, lambda_tot=0.0)
    # the WDL-only base must sit at the log-loss minimum, independent of the total target
    ll0 = backtest.log_loss(feats, (200.0, b0, home, -0.06, 0.0))
    ll_lo = backtest.log_loss(feats, (200.0, b0 - 0.2, home, -0.06, 0.0))
    ll_hi = backtest.log_loss(feats, (200.0, b0 + 0.2, home, -0.06, 0.0))
    assert ll0 <= ll_lo + 1e-6 and ll0 <= ll_hi + 1e-6


def test_run_report_has_totals_slice():
    """run(write=False) reports the totals slice: pred lands within +/-0.05 of actual
    and the goal-aware base_goals is recorded alongside the W/D/L one (item 5 gate)."""
    report = backtest.run(write=False)
    t = report["totals"]
    assert abs(t["mean_total_dev"]) <= 0.05                # gate: within +/-0.05
    assert t["base_goals_goal_aware"] >= t["base_goals_wdl"]
    assert t["model_btts"] > 0.434                         # moved up from baseline
    assert t["lambda_tot"] == backtest.LAMBDA_TOT
    # W/D/L test log-loss must not rise more than 0.004 over the frozen baseline
    assert report["test"]["log_loss"] <= 0.8597 + 0.004


def _overconfident_feats(seed=0, n=3000):
    """Synthetic feats where the model is OVERCONFIDENT on favourites: outcomes are
    drawn from a flatter distribution than the model's Elo gap implies, so the honest
    out-of-sample temperature must be > 1 (de-sharpening)."""
    rng = np.random.default_rng(seed)
    elo_h = rng.normal(1800, 60, n)
    elo_a = rng.normal(1800, 60, n)
    neutral = np.ones(n, bool)
    # true favourite edge is HALF of what the Elo gap suggests -> model over-rates it
    eff = (elo_h - elo_a)
    p_home = 0.34 + 0.0004 * eff      # deliberately tame slope (model would use a steeper one)
    p_home = np.clip(p_home, 0.05, 0.7)
    p_draw = np.full(n, 0.30)
    p_away = np.clip(1.0 - p_home - p_draw, 0.05, 0.9)
    u = rng.random(n)
    outcome = np.where(u < p_home, 0, np.where(u < p_home + p_draw, 1, 2))
    return {"elo_h": elo_h, "elo_a": elo_a, "neutral": neutral,
            "gh": np.zeros(n, int), "ga": np.zeros(n, int), "outcome": outcome,
            "year": np.full(n, 2015), "is_major": np.zeros(n, bool),
            "is_friendly": np.zeros(n, bool),
            "tournament": np.array([""] * n, dtype=object)}


def test_fit_temperature_desharpens_an_overconfident_model():
    """On a held-out fold where the model is overconfident, the fitted temperature is
    > 1 and lowers log-loss vs the uncalibrated probs (item 3 gate, leak-free)."""
    feats = _overconfident_feats(seed=7)
    params = (200.0, 2.5, 0.0, -0.06, 0.0)   # a confident Elo model on a tame-outcome fold
    T = backtest.fit_temperature(feats, params)
    assert T > 1.0                                   # de-sharpening direction
    assert backtest._T_MIN <= T <= backtest._T_MAX   # stays bounded
    ll = backtest.log_loss(feats, params)
    ll_cal = backtest._temp_log_loss(feats, params, T)
    assert ll_cal < ll                               # calibration improves the fold


def test_fit_temperature_bounds_are_08_20():
    # a pathological fold cannot push T outside [0.8, 2.0]
    feats = _overconfident_feats(seed=3, n=400)
    very_sharp = (40.0, 3.0, 0.0, -0.06, 0.0)        # huge supremacy -> wild overconfidence
    T = backtest.fit_temperature(feats, very_sharp)
    assert 0.8 <= T <= 2.0


def test_elite_overconfidence_gap_positive_when_favourite_overrated():
    feats = _overconfident_feats(seed=11)
    # lift both Elos above the 2000 floor so the slice is non-empty
    feats = {**feats, "elo_h": feats["elo_h"] + 250, "elo_a": feats["elo_a"] + 250}
    params = (200.0, 2.5, 0.0, -0.06, 0.0)
    gap, n = backtest.elite_overconfidence_gap(feats, params, 0.0)
    assert n > 0
    assert gap > 0.0                                 # model over-predicts favourite wins
    # de-sharpening shrinks the gap toward zero
    T = backtest.fit_temperature(feats, params)
    gap_cal, _ = backtest.elite_overconfidence_gap(feats, params, 0.0, T)
    assert abs(gap_cal) < abs(gap)


def test_run_temperature_is_out_of_sample_and_desharpens():
    """run() fits temperature on the held-out is_major fold (NOT train), so it learns a
    de-sharpening T>1 and the calibration slice shows the fold improving (item 3)."""
    report = backtest.run(write=False)
    cal = report["calibration"]
    assert cal["validation_fold"] == "is_major"
    assert cal["temperature"] > 1.0                          # de-sharpening, not the
    assert report["fitted"]["temperature"] == cal["temperature"]   # train-leak ~1.0 of old
    # on the held-out fold, calibrated log-loss is no worse than uncalibrated
    assert cal["fold_log_loss_calibrated"] <= cal["fold_log_loss"]
    # the elite gap moves toward zero (de-sharpening cools elite favourites)
    assert cal["elite_n"] > 0
    assert cal["elite_fav_gap_pp_calibrated"] < cal["elite_fav_gap_pp"]
    # GUARDRAIL: applying it must not regress the overall test log-loss past the cap
    assert report["test"]["log_loss"] <= 0.8597 + 0.004
    assert report["test"]["log_loss_calibrated"] <= 0.8597 + 0.004


def test_eff_temp_vec_only_touches_elite_close_matches():
    """T_eff equals the global temperature for ordinary matches and exceeds it only for
    elite-vs-elite close games (item A: the vectorised mirror of model.effective_temperature)."""
    feats = {
        "elo_h": np.array([2080.0, 1700.0, 2080.0]),
        "elo_a": np.array([2030.0, 1600.0, 1500.0]),
        "neutral": np.array([True, True, True]),
        "outcome": np.array([0, 0, 0]),
    }
    params = (200.0, 2.5, 0.0, -0.06, 0.0)
    t = backtest._eff_temp_vec(feats, params, 0.0, temperature=1.02, t_elite=1.5)
    assert t[0] > 1.02 + 0.3          # elite-vs-elite close -> meaningfully de-sharpened
    assert abs(t[1] - 1.02) < 1e-3    # ordinary match -> base temperature
    assert abs(t[2] - 1.02) < 1e-2    # elite but a big gap -> ~base (gate ~0)
    # t_elite=0 is an exact no-op
    t0 = backtest._eff_temp_vec(feats, params, 0.0, temperature=1.02, t_elite=0.0)
    assert np.allclose(t0, 1.02)


def test_fit_t_elite_desharpens_elite_and_is_bounded():
    """On an overconfident elite-vs-elite fold the fitted t_elite is positive (extra
    de-sharpening) and stays within [0, 1.5]; it shrinks the elite overconfidence gap."""
    feats = _overconfident_feats(seed=21)
    # push both Elos above the 2000 floor so the matches are elite-vs-elite and close
    feats = {**feats, "elo_h": feats["elo_h"] + 230, "elo_a": feats["elo_a"] + 230}
    params = (200.0, 2.5, 0.0, -0.06, 0.0)
    te = backtest.fit_t_elite(feats, params, 0.0, temperature=1.0)
    assert backtest._TE_MIN <= te <= backtest._TE_MAX
    assert te > 0.0                                  # de-sharpening direction
    gap0, n = backtest.elite_overconfidence_gap(feats, params, 0.0, 1.0)
    gap_te, _ = backtest.elite_overconfidence_gap(feats, params, 0.0, 1.0, t_elite=te)
    assert n > 0
    assert abs(gap_te) < abs(gap0)                   # gap shrinks toward zero


def test_t_elite_leaves_strictly_non_elite_matches_untouched():
    """The matchup-conditional de-sharpen is provably a no-op on strictly non-elite
    matches (min Elo well below the 2000 floor): T_eff == base temperature there, so the
    bulk calibration cannot move no matter how large t_elite is fit (item A invariant)."""
    feats = _overconfident_feats(seed=22)
    # keep only the strictly non-elite matches (min Elo < 1800, i.e. >= 5 logistic widths
    # below the 2000 floor: the elite gate is numerically ~0 there)
    keep = np.minimum(feats["elo_h"], feats["elo_a"]) < 1800.0
    sub = backtest._subset(feats, keep)
    assert len(sub["outcome"]) > 100                 # a meaningful non-elite population
    params = (200.0, 2.5, 0.0, -0.06, 0.0)
    t_base = backtest._eff_temp_vec(sub, params, 0.0, temperature=1.02, t_elite=0.0)
    t_big = backtest._eff_temp_vec(sub, params, 0.0, temperature=1.02, t_elite=1.5)
    assert np.allclose(t_base, t_big, atol=1e-2)     # base temperature everywhere
    # and the non-elite log-loss is unchanged by the largest in-bound t_elite
    ll0 = backtest._eff_temp_log_loss(sub, params, 0.0, 1.02, 0.0)
    ll1 = backtest._eff_temp_log_loss(sub, params, 0.0, 1.02, 1.5)
    assert abs(ll1 - ll0) < 1e-3


def test_run_supremacy_conditional_desharpen_gate():
    """item A gate, on the genuinely held-out is_major SCORE sub-fold: the elite-vs-elite
    favourite overconfidence gap drops under the supremacy-conditional de-sharpen, the
    full-test gap falls toward <= 3pp, and the overall test log-loss does not regress."""
    report = backtest.run(write=False)
    cal = report["calibration"]
    # t_elite is fitted, positive, and recorded in the fitted params
    assert cal["t_elite"] > 0.0
    assert report["fitted"]["t_elite"] == cal["t_elite"]
    assert backtest._TE_MIN <= cal["t_elite"] <= backtest._TE_MAX
    # full-test elite gap drops toward the <= 3pp target under the conditional de-sharpen
    assert cal["elite_fav_gap_pp_supremacy_cond"] < cal["elite_fav_gap_pp_calibrated"]
    assert cal["elite_fav_gap_pp_supremacy_cond"] <= 3.0
    # GENUINELY HELD-OUT gate: on the score sub-fold (disjoint from the fit sub-fold), the
    # gap is smaller under t_elite than under the global temperature alone
    assert cal["elite_holdout_n"] > 0
    assert (abs(cal["elite_holdout_gap_pp_supremacy_cond"])
            < abs(cal["elite_holdout_gap_pp_temperature"]))
    # GUARDRAIL: overall test log-loss (raw and as-shipped/calibrated) must not regress
    # past the frozen baseline + 0.004
    assert report["test"]["log_loss"] <= 0.8599 + 0.004
    assert report["test"]["log_loss_calibrated"] <= 0.8599 + 0.004


def test_run_t_elite_does_not_disturb_non_elite_calibration():
    """The shipped matchup-conditional temperature changes the overall test log-loss only
    negligibly vs the global temperature, because it touches only the small elite cluster."""
    report = backtest.run(write=False)
    # calibrated (T_eff) overall log-loss must be no worse than the uncalibrated within a
    # tight band -> the elite-only de-sharpen does not harm the bulk non-elite population
    assert report["test"]["log_loss_calibrated"] <= report["test"]["log_loss"] + 0.001


def test_fit_clamps_params_on_degenerate_data():
    # all home wins 3-0 pushes the optimizer toward an out-of-bounds (even negative c)
    n = 60
    df = pd.DataFrame({
        "date": pd.to_datetime(["2010-01-01"] * n),
        "home_team": [f"H{i % 6}" for i in range(n)],
        "away_team": [f"A{i % 6}" for i in range(n)],
        "home_score": [3] * n, "away_score": [0] * n,
        "tournament": ["Friendly"] * n, "neutral": [False] * n,
    })
    feats, _ = backtest.prematch_pass(df)
    params, home = backtest.fit(feats)
    assert 50.0 <= params.c <= 600.0       # never negative / out of bounds
    assert 1.5 <= params.base_goals <= 4.0
    assert 0.0 <= home <= 150.0
    assert 0.0 <= params.gamma <= 1.0      # supremacy->goals term stays bounded


# ---------------- corrected stage-split of base_goals ----------------

def _finals_edition_feats():
    """A synthetic 16-match finals edition (12 group + 4 knockout by date) plus some
    friendlies, so stage_masks has a clean group/knockout split to classify."""
    import numpy as np
    dates = pd.to_datetime(
        [f"2018-06-{10 + i:02d}" for i in range(12)]      # 12 group matches
        + [f"2018-07-{1 + i:02d}" for i in range(4)]       # 4 later knockout matches
        + ["2017-03-01", "2017-03-02"]                     # 2 friendlies (non-finals)
    )
    n = len(dates)
    df = pd.DataFrame({
        "date": dates,
        "home_team": [f"H{i}" for i in range(n)],
        "away_team": [f"A{i}" for i in range(n)],
        "home_score": [1] * n, "away_score": [1] * n,
        "tournament": ["FIFA World Cup"] * 16 + ["Friendly"] * 2,
        "neutral": [True] * n,
    })
    feats, _ = backtest.prematch_pass(df)
    return feats


def test_stage_masks_splits_finals_into_group_and_knockout():
    feats = _finals_edition_feats()
    group, ko = backtest.stage_masks(feats)
    # a 16-match edition -> 8 knockout matches (the last 8 by date), 8 group
    assert int(ko.sum()) == 8
    assert int(group.sum()) == 8
    # friendlies are neither group nor knockout
    assert not (group | ko)[-2:].any()
    # group and knockout never overlap
    assert not (group & ko).any()


def test_stage_total_bias_sign_and_delta_shift():
    feats = _synthetic_goal_feats(seed=3, n=2000)
    # mark the first 1000 as a single FIFA World Cup edition so half are group/half knockout
    feats["tournament"] = np.array(["FIFA World Cup"] * 1000 + [""] * 1000, dtype=object)
    feats["year"] = np.array([2018] * 1000 + [2015] * 1000)
    params, home = backtest.fit(feats)
    group, ko = backtest.stage_masks(feats)
    assert group.any() and ko.any()
    b0 = backtest.stage_total_bias(feats, params, home, group, 0.0)
    # a negative base delta lowers the predicted total -> lowers the (pred-actual) bias
    b_down = backtest.stage_total_bias(feats, params, home, group, -0.2)
    assert b_down < b0
    # an empty mask returns None
    assert backtest.stage_total_bias(feats, params, home, np.zeros(len(group), bool)) is None


def test_fit_stage_deltas_has_corrective_shape_and_is_bounded():
    feats = _synthetic_goal_feats(seed=5, n=3000)
    feats["tournament"] = np.array(["FIFA World Cup"] * 1500 + [""] * 1500, dtype=object)
    feats["year"] = np.array([2018] * 1500 + [2015] * 1500)
    params, home = backtest.fit(feats)
    dG, dK = backtest.fit_stage_deltas(feats, params, home)
    # mandated shape: group down (<=0), knockout up (>=0)
    assert dG <= 0.0
    assert dK >= 0.0
    # both capped at +/- the bound
    assert abs(dG) <= backtest._STAGE_DELTA_CAP
    assert abs(dK) <= backtest._STAGE_DELTA_CAP


def test_run_stage_split_gate():
    """The corrected stage-split must: lower the group total-goals bias toward 0 (|bias|
    strictly smaller), not worsen the knockout bias, and keep W/D/L within the guardrail."""
    report = backtest.run(write=False)
    t = report["totals"]
    gb, ga = t["group_bias_before"], t["group_bias_after"]
    kb, ka = t["knockout_bias_before"], t["knockout_bias_after"]
    # group bias is positive (over-predicts) and the split shrinks |bias| toward 0
    assert gb is not None and ga is not None
    assert abs(ga) < abs(gb)
    # knockout bias must not worsen
    assert kb is not None and ka is not None
    assert abs(ka) <= abs(kb) + 1e-9
    # corrective shape carried into the fitted params
    assert report["fitted"]["base_goals_group_delta"] <= 0.0
    assert report["fitted"]["base_goals_knockout_delta"] >= 0.0
    # the split barely touches W/D/L (totals-only knob): test log-loss within guardrail
    assert report["test"]["log_loss"] <= 0.8639
    assert report["test"]["log_loss"] <= 0.8599 + 0.004
