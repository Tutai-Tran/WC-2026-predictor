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
