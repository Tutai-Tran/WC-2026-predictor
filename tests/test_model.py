import numpy as np
import pytest

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


def test_derived_markets_consistency_identities():
    m = model.scoreline_matrix(1.7, 1.1)
    dm = model.derived_markets(m)
    # over/under and BTTS yes/no are complementary
    assert abs(dm["over_2_5"] + dm["under_2_5"] - 1.0) < 1e-9
    assert abs(dm["btts_yes"] + dm["btts_no"] - 1.0) < 1e-9
    # the four total-goals bands partition the whole mass
    bands = dm["total_goals_bands"]
    assert abs(sum(bands.values()) - 1.0) < 1e-9
    # over 2.5 == P(3) + P(4+) (the bands above 2)
    assert abs(dm["over_2_5"] - (bands["3"] + bands["4+"])) < 1e-9
    # under 2.5 == P(0-1) + P(2)
    assert abs(dm["under_2_5"] - (bands["0-1"] + bands["2"])) < 1e-9
    # BTTS-yes identity: 1 - P(home=0) - P(away=0) + P(0,0); clean-sheet probs match
    # the marginal zero masses (home keeps a sheet iff away scores 0)
    p_home_zero = float(m[0, :].sum())
    p_away_zero = float(m[:, 0].sum())
    assert abs(dm["clean_sheet_home"] - p_away_zero) < 1e-9
    assert abs(dm["clean_sheet_away"] - p_home_zero) < 1e-9
    expected_btts = 1.0 - p_home_zero - p_away_zero + float(m[0, 0])
    assert abs(dm["btts_yes"] - expected_btts) < 1e-9
    # every probability is a valid [0,1] mass
    for v in (dm["over_2_5"], dm["under_2_5"], dm["btts_yes"], dm["btts_no"],
              dm["clean_sheet_home"], dm["clean_sheet_away"]):
        assert 0.0 <= v <= 1.0


def test_top_scorelines_sorted_desc_and_length_five():
    m = model.scoreline_matrix(1.6, 1.3)
    tops = model.top_scorelines(m, 5)
    assert len(tops) == 5
    probs = [p for _, p in tops]
    assert probs == sorted(probs, reverse=True)
    # the global modal cell is the first entry
    assert tops[0][1] == pytest.approx(float(m.max()))


def test_expected_score_rounds_lambdas():
    assert model.expected_score(1.4, 0.6) == (1, 1)
    assert model.expected_score(2.6, 0.9) == (3, 1)


def test_derived_markets_do_not_change_wdl():
    # W/D/L must be byte-identical whether or not derived_markets is computed
    # (same matrix in, same outcome probs out: the new aggregation is read-only)
    m = model.scoreline_matrix(1.7, 1.1)
    wdl_before = model.outcome_probs(m)
    _ = model.derived_markets(m)
    _ = model.top_scorelines(m, 5)
    wdl_after = model.outcome_probs(m)
    assert wdl_before == wdl_after


# ---------------- temperature scaling (item 3) ----------------

def test_temper_wdl_identity_at_one():
    # T=1 is an exact no-op
    p = (0.6, 0.25, 0.15)
    assert model.temper_wdl(*p, 1.0) == p


def test_temper_wdl_renormalises_to_one():
    for T in (0.85, 1.0, 1.3, 1.9):
        out = model.temper_wdl(0.55, 0.27, 0.18, T)
        assert abs(sum(out) - 1.0) < 1e-12
        assert all(0.0 <= v <= 1.0 for v in out)


def test_temper_wdl_above_one_desharpens_favourite():
    # T>1 flattens toward uniform: the favourite's prob falls, the laggards' rise
    ph, pdw, pa = 0.70, 0.20, 0.10
    h2, d2, a2 = model.temper_wdl(ph, pdw, pa, 1.4)
    assert h2 < ph                 # favourite cooled
    assert a2 > pa and d2 > pdw     # the other two outcomes pick up the mass


def test_temper_wdl_below_one_sharpens():
    ph, pdw, pa = 0.55, 0.27, 0.18
    h2, _, a2 = model.temper_wdl(ph, pdw, pa, 0.85)
    assert h2 > ph and a2 < pa      # T<1 sharpens (concentrates on the leader)


def test_temper_wdl_preserves_argmax():
    # no bias term, so the winner can never flip under any temperature in-bound
    ph, pdw, pa = 0.46, 0.30, 0.24
    for T in (0.8, 1.0, 1.5, 2.0):
        out = model.temper_wdl(ph, pdw, pa, T)
        assert out.index(max(out)) == 0


def test_match_forecast_temperature_cools_favourite_not_scorelines():
    sharp = model.match_forecast(1950, 1500, model.ModelParams(temperature=1.0))
    cool = model.match_forecast(1950, 1500, model.ModelParams(temperature=1.4))
    # the favourite's win prob is lower with T>1...
    assert cool["p_home"] < sharp["p_home"]
    # ...but the scoreline matrix (and therefore exact-score reporting) is untouched
    assert (sharp["matrix"] == cool["matrix"]).all()
    assert sharp["top_scorelines"] == cool["top_scorelines"]
    # the displayed winner is unchanged (argmax cannot flip)
    assert cool["p_home"] == max(cool["p_home"], cool["p_draw"], cool["p_away"])
