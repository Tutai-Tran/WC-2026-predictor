"""Tests for the head-to-head supremacy adjustment."""

import pandas as pd

from wc26 import h2h


def _m(year, home, away, hg, ag, neutral=True):
    return (year, home, away, hg, ag, neutral)


def test_no_history_is_zero():
    assert h2h.h2h_delta([], "A", "B") == 0.0


def test_positive_when_a_dominates_and_negative_for_b():
    meetings = [_m(2024, "A", "B", 3, 0), _m(2023, "A", "B", 2, 0), _m(2022, "B", "A", 0, 1)]
    da = h2h.h2h_delta(meetings, "A", "B")
    db = h2h.h2h_delta(meetings, "B", "A")
    assert da > 0
    assert abs(da + db) < 1e-12          # antisymmetric


def test_capped():
    # absurd, repeated blowouts must still be bounded by DELTA_CAP and stay strong
    meetings = [_m(2025, "A", "B", 12, 0) for _ in range(40)]
    d = h2h.h2h_delta(meetings, "A", "B")
    assert 0 < d <= h2h.DELTA_CAP + 1e-9
    assert d > 0.6                       # lots of one-sided history -> near the cap


def test_shrinkage_more_data_is_stronger():
    one = [_m(2025, "A", "B", 2, 0)]
    many = [_m(2025, "A", "B", 2, 0) for _ in range(10)]
    assert h2h.h2h_delta(many, "A", "B") > h2h.h2h_delta(one, "A", "B") > 0


def test_recency_weighting():
    # a recent B win should pull the delta below an old A win of equal margin
    recent_b = [_m(2026, "A", "B", 0, 2), _m(1980, "A", "B", 2, 0)]
    assert h2h.h2h_delta(recent_b, "A", "B") < 0


def test_venue_correction_discounts_home_wins():
    # same 1-0 margin: a neutral win counts for more than a home win
    home_win = [_m(2025, "A", "B", 1, 0, neutral=False)]
    neutral_win = [_m(2025, "A", "B", 1, 0, neutral=True)]
    assert h2h.h2h_delta(neutral_win, "A", "B") > h2h.h2h_delta(home_win, "A", "B")


def test_oriented_gd_symmetric_orientation():
    m = _m(2025, "A", "B", 3, 1, neutral=True)
    assert h2h._oriented_gd(m, "A", "B") == 2.0
    assert h2h._oriented_gd(m, "B", "A") == -2.0
    # a meeting that is not between exactly this pair is ignored, never mis-oriented
    assert h2h._oriented_gd(m, "A", "C") is None
    assert h2h._oriented_gd(m, "Z", "Y") is None


def test_build_index_skips_future_and_missing_scores():
    df = pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B", "home_score": 2,
         "away_score": 1, "neutral": True},
        {"date": "2026-12-31", "home_team": "A", "away_team": "B", "home_score": 5,
         "away_score": 0, "neutral": True},                       # after cutoff -> skipped
        {"date": "2024-02-02", "home_team": "A", "away_team": "B", "home_score": None,
         "away_score": None, "neutral": True},                    # no score -> skipped
    ])
    idx = h2h.build_index(df, cutoff="2026-06-05")
    meetings = idx[frozenset(("A", "B"))]
    assert len(meetings) == 1
    assert meetings[0][3:5] == (2, 1)


def test_friendly_cutoff_excludes_2026_self_leak():
    # A 2026 warm-up between A and B must not appear in the pre-2026 friendly index,
    # otherwise the friendly forecast would see its own result.
    df = pd.DataFrame([
        {"date": "2019-06-01", "home_team": "A", "away_team": "B", "home_score": 1,
         "away_score": 1, "neutral": True},
        {"date": "2026-05-28", "home_team": "A", "away_team": "B", "home_score": 4,
         "away_score": 0, "neutral": True},   # the warm-up we'd be predicting
    ])
    pre = h2h.build_index(df, cutoff=h2h.PRE_WARMUP_CUTOFF)
    meetings = pre[frozenset(("A", "B"))]
    assert len(meetings) == 1 and meetings[0][0] == 2019   # only the historical meeting


def test_delta_map_antisymmetric_and_zero_for_unknown():
    df = pd.DataFrame([
        {"date": "2024-01-01", "home_team": "A", "away_team": "B", "home_score": 3,
         "away_score": 0, "neutral": True},
    ])
    idx = h2h.build_index(df)
    dm = h2h.delta_map(["A", "B", "C"], idx)
    assert dm[("A", "B")] > 0
    assert abs(dm[("A", "B")] + dm[("B", "A")]) < 1e-12
    assert dm.get(("A", "C"), 0.0) == 0.0      # no shared history
