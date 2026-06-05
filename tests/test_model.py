import numpy as np

from wc26 import model


def test_equal_teams_symmetric():
    la, lb = model.match_lambdas(1500, 1500)
    assert abs(la - lb) < 1e-9
    m = model.scoreline_matrix(la, lb)
    p_home, p_draw, p_away = model.outcome_probs(m)
    assert abs(p_home - p_away) < 1e-6
    assert abs((p_home + p_draw + p_away) - 1.0) < 1e-6


def test_stronger_team_favoured():
    f = model.match_forecast(1900, 1500)
    assert f["p_home"] > f["p_away"]
    assert f["lambda_home"] > f["lambda_away"]


def test_supremacy_increases_with_diff():
    _, _ = model.match_lambdas(1500, 1500)
    la1, lb1 = model.match_lambdas(1600, 1500)
    la2, lb2 = model.match_lambdas(1800, 1500)
    assert (la2 - lb2) > (la1 - lb1) > 0


def test_matrix_sums_to_one_and_nonneg():
    m = model.scoreline_matrix(1.8, 1.1, max_goals=10, rho=-0.06)
    assert abs(m.sum() - 1.0) < 1e-9
    assert (m >= 0).all()


def test_top_scorelines_sorted():
    m = model.scoreline_matrix(1.5, 1.2)
    tops = model.top_scorelines(m, 3)
    probs = [p for _, p in tops]
    assert probs == sorted(probs, reverse=True)
    assert len(tops) == 3


def test_lambda_floor():
    la, lb = model.match_lambdas(1000, 2200)  # huge gap
    assert lb >= model.ModelParams().min_lambda


def test_h2h_delta_shifts_supremacy():
    base_a, base_b = model.match_lambdas(1500, 1500)
    up_a, up_b = model.match_lambdas(1500, 1500, h2h_delta=0.5)
    assert up_a > base_a and up_b < base_b
    assert abs((up_a - up_b) - (base_a - base_b) - 0.5) < 1e-9   # supremacy += delta
    # symmetric negative delta favours the away side
    dn_a, dn_b = model.match_lambdas(1500, 1500, h2h_delta=-0.5)
    assert dn_a < dn_b


def test_h2h_delta_zero_is_unchanged():
    assert model.match_lambdas(1700, 1500) == model.match_lambdas(1700, 1500, h2h_delta=0.0)


def test_goal_scale_lifts_total_not_supremacy():
    p1 = model.ModelParams(goal_scale=1.0)
    p2 = model.ModelParams(goal_scale=1.2)
    a1, b1 = model.match_lambdas(1700, 1500, p1)
    a2, b2 = model.match_lambdas(1700, 1500, p2)
    assert (a2 + b2) > (a1 + b1)                       # more total goals
    assert abs((a2 - b2) - (a1 - b1)) < 1e-9           # same supremacy (favourite unchanged)


def test_gamma_makes_mismatches_higher_scoring():
    p0 = model.ModelParams(gamma=0.0)
    pg = model.ModelParams(gamma=0.4)
    even0 = sum(model.match_lambdas(1500, 1500, p0))
    even_g = sum(model.match_lambdas(1500, 1500, pg))
    assert abs(even0 - even_g) < 1e-9                  # even match: no extra goals
    mismatch0 = sum(model.match_lambdas(1900, 1500, p0))
    mismatch_g = sum(model.match_lambdas(1900, 1500, pg))
    assert mismatch_g > mismatch0                      # bigger gap -> more total goals with gamma
    # and a bigger gap gets a bigger gamma bump than a smaller gap
    small = sum(model.match_lambdas(1600, 1500, pg)) - sum(model.match_lambdas(1600, 1500, p0))
    big = sum(model.match_lambdas(2000, 1500, pg)) - sum(model.match_lambdas(2000, 1500, p0))
    assert big > small


def test_likely_scoreline_consistent_with_outcome():
    # strong favourite: predicted outcome is a home win, so the likely score must
    # be a home win (home goals strictly greater), never a draw.
    f = model.match_forecast(1950, 1500)
    assert f["p_home"] == max(f["p_home"], f["p_draw"], f["p_away"])
    sh, sa = f["likely_scoreline"]
    assert sh > sa


def test_forecast_hit_outcome_correctness():
    # predicted home win (probs favour home), actual 2-0 home win -> hit
    assert model.forecast_hit(0.6, 0.25, 0.15, 2, 0) is True
    # predicted home win, actual 0-0 draw -> miss
    assert model.forecast_hit(0.6, 0.25, 0.15, 0, 0) is False
    # predicted away win, actual away win -> hit
    assert model.forecast_hit(0.2, 0.3, 0.5, 1, 3) is True
    assert model.result_label(1, 1) == "draw"
    assert model.result_label(0, 2) == "away"


def test_most_likely_scoreline_respects_region():
    m = model.scoreline_matrix(1.2, 1.2)        # even match, modal exact score is a draw
    (dh, da), _ = model.most_likely_scoreline(m, "draw")
    assert dh == da
    (hh, ha), _ = model.most_likely_scoreline(m, "home")
    assert hh > ha
    (ah, aa), _ = model.most_likely_scoreline(m, "away")
    assert ah < aa
