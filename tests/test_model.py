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
