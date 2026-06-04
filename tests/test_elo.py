import pandas as pd

from wc26 import elo


def test_expected_symmetry_and_monotonic():
    assert abs(elo.expected(1500, 1500) - 0.5) < 1e-9
    assert elo.expected(1700, 1500) > 0.5
    assert elo.expected(1300, 1500) < 0.5
    # home advantage helps
    assert elo.expected(1500, 1500, home_adv=100) > 0.5


def test_gd_multiplier_monotonic():
    assert elo.gd_multiplier(0) == 1.0
    assert elo.gd_multiplier(1) == 1.0
    assert elo.gd_multiplier(2) == 1.5
    assert elo.gd_multiplier(4) > elo.gd_multiplier(3) > elo.gd_multiplier(2)


def test_update_zero_sum_and_direction():
    a, b = elo.update_pair(1500, 1500, 2, 0, k=40)
    assert a > 1500 and b < 1500
    assert abs((a - 1500) + (b - 1500)) < 1e-9  # zero-sum
    # upset: weak team beats strong team gains more
    weak_win_a, _ = elo.update_pair(1400, 1800, 1, 0, k=40)
    fav_win_a, _ = elo.update_pair(1800, 1400, 1, 0, k=40)
    assert (weak_win_a - 1400) > (fav_win_a - 1800)


def test_k_for_tournament():
    assert elo.k_for_tournament("Friendly") == 20.0
    assert elo.k_for_tournament("FIFA World Cup") == 60.0
    assert elo.k_for_tournament("FIFA World Cup qualification") == 40.0


def test_replay_rewards_winner():
    df = pd.DataFrame(
        {
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "home_score": [2, 1, 3],
            "away_score": [0, 0, 1],
            "neutral": [True, True, True],
            "tournament": ["Friendly", "Friendly", "Friendly"],
        }
    )
    ratings = elo.replay(df, seed={"A": 1500, "B": 1500})
    assert ratings["A"] > 1500 > ratings["B"]
    # deterministic
    ratings2 = elo.replay(df, seed={"A": 1500, "B": 1500})
    assert ratings == ratings2
