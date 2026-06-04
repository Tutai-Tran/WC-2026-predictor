from wc26 import scorers

PLAYERS = [
    {"name": "Striker", "position": "FW", "club_goals": 20, "is_penalty_taker": True},
    {"name": "Winger", "position": "FW", "club_goals": 8, "is_penalty_taker": False},
    {"name": "Mid", "position": "MF", "club_goals": 4, "is_penalty_taker": False},
    {"name": "Back", "position": "DF", "club_goals": 1, "is_penalty_taker": False},
    {"name": "Keeper", "position": "GK", "club_goals": 0, "is_penalty_taker": False},
]


def test_shares_sum_to_one():
    shares = scorers.team_goal_shares(PLAYERS)
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert shares["Striker"] > shares["Mid"] > shares["Back"] > shares["Keeper"]


def test_anytime_probs_bounded_and_increase_with_lambda():
    low = dict(scorers.anytime_scorer_probs(PLAYERS, 0.8))
    high = dict(scorers.anytime_scorer_probs(PLAYERS, 2.5))
    for name in low:
        assert 0.0 <= low[name] <= 1.0
        assert high[name] >= low[name]


def test_penalty_taker_bonus():
    a = [{"name": "T", "position": "FW", "club_goals": 10, "is_penalty_taker": True},
         {"name": "N", "position": "FW", "club_goals": 10, "is_penalty_taker": False}]
    probs = dict(scorers.anytime_scorer_probs(a, 1.5))
    assert probs["T"] > probs["N"]  # identical except penalties


def test_top_scorers_shape():
    top = scorers.top_scorers(PLAYERS, 1.8, n=3)
    assert len(top) == 3
    assert top[0]["player"] == "Striker"
    assert "p_anytime" in top[0]
