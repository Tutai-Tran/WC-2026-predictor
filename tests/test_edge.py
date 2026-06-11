"""Tests for the model-vs-market edge / paper value-betting research module."""

from __future__ import annotations

import pytest

from wc26 import edge


# --- responsible-gambling guardrails (tested invariants) --------------------

def test_no_live_betting_integration():
    assert edge.LIVE_BETTING_ENABLED is False
    edge._assert_paper_only()  # must not raise


def test_disclaimer_has_required_messaging():
    d = edge.DISCLAIMER.lower()
    assert "paper" in d and "not betting advice" in d.lower()
    assert "afford to lose" in d
    assert "0800-0177" in edge.DISCLAIMER or "begambleaware" in d  # a help resource


# --- implied probability / overround ----------------------------------------

def test_implied_prob():
    assert edge.implied_prob(2.0) == pytest.approx(0.5)
    assert edge.implied_prob(4.0) == pytest.approx(0.25)


def test_implied_prob_rejects_invalid():
    with pytest.raises(ValueError):
        edge.implied_prob(1.0)


def test_overround_margin_and_arb():
    assert edge.overround({"h": 2.1, "d": 3.4, "a": 3.6}) > 1.0
    assert edge.overround({"h": 2.2, "d": 3.8, "a": 4.2}) < 1.0


def test_overround_guards_bad_odds():
    assert edge.overround({}) == 0.0
    with pytest.raises(ValueError):
        edge.overround({"h": 1.0, "a": 2.0})  # price of 1.0 is degenerate


# --- devigging --------------------------------------------------------------

def test_devig_proportional_sums_to_one():
    fair = edge.devig_proportional({"h": 2.1, "d": 3.4, "a": 3.6})
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["h"] > fair["d"] and fair["h"] > fair["a"]


def test_devig_shin_sums_to_one_and_corrects_bias():
    odds = {"fav": 1.45, "mid": 4.2, "dog": 8.0}
    assert edge.overround(odds) > 1.0
    prop = edge.devig_proportional(odds)
    shin = edge.devig_shin(odds)
    assert sum(shin.values()) == pytest.approx(1.0, abs=1e-6)
    # favourite-longshot correction: longshot down, favourite up vs proportional
    assert shin["fav"] > prop["fav"]
    assert shin["dog"] < prop["dog"]


def test_devig_shin_large_margin_converges():
    # A high-margin book (overround ~1.15) exercises a larger z; output still sums
    # to 1 and keeps the correct directional property.
    odds = {"h": 1.3, "d": 3.8, "a": 4.0}
    assert edge.overround(odds) > 1.1
    shin = edge.devig_shin(odds)
    prop = edge.devig_proportional(odds)
    assert sum(shin.values()) == pytest.approx(1.0, abs=1e-6)
    assert shin["h"] > prop["h"]


def test_devig_no_margin_falls_back_to_proportional():
    odds = {"h": 2.2, "d": 3.8, "a": 4.2}
    assert edge.devig_shin(odds) == pytest.approx(edge.devig_proportional(odds))


def test_devig_unknown_method():
    with pytest.raises(ValueError):
        edge.devig({"h": 2.0, "a": 2.0}, method="nope")


# --- EV, Kelly, shrink ------------------------------------------------------

def test_ev_per_unit_sign():
    assert edge.ev_per_unit(0.6, 2.0) == pytest.approx(0.2)
    assert edge.ev_per_unit(0.4, 2.0) == pytest.approx(-0.2)
    assert edge.ev_per_unit(0.5, 2.0) == pytest.approx(0.0)


def test_kelly_fraction_known_value():
    assert edge.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_zero_when_not_value():
    assert edge.kelly_fraction(0.4, 2.0) == 0.0


def test_effective_shrink_grows_with_odds():
    s_short = edge.effective_shrink(2.0, 0.25, 0.18)
    s_long = edge.effective_shrink(50.0, 0.25, 0.18)
    assert s_long > s_short
    assert 0.0 <= s_short <= 0.95 and 0.0 <= s_long <= 0.95
    # clipped at 0.95 for very long prices
    assert edge.effective_shrink(1e6, 0.25, 0.18) == pytest.approx(0.95)


# --- value bets -------------------------------------------------------------

def _noshrink(**kw):
    return {"shrink": 0.0, "shrink_odds_scale": 0.0, **kw}


def test_value_bets_finds_positive_ev():
    model = {"h": 0.62, "d": 0.24, "a": 0.14}
    odds = {"h": 2.1, "d": 3.4, "a": 3.6}
    bets = edge.value_bets(model, odds, bankroll=1000, **_noshrink(min_edge=0.02))
    h = next(b for b in bets if b.selection == "h")
    assert h.ev > 0 and h.stake > 0
    assert h.stake <= 1000 * 0.05  # per-bet cap


def test_value_bets_caps_stake():
    model = {"h": 0.95, "a": 0.05}
    odds = {"h": 5.0, "a": 1.1}
    bets = edge.value_bets(model, odds, bankroll=1000, kelly_frac=1.0,
                           max_stake_frac=0.05, **_noshrink())
    h = next(b for b in bets if b.selection == "h")
    assert h.stake == pytest.approx(50.0)


def test_value_bets_max_odds_excludes_longshot():
    # The Colombia trap: model >> market on a 50/1 shot must be filtered by the cap.
    model = {"col": 0.046, "fav": 0.954}
    odds = {"col": 53.0, "fav": 1.02}
    market = {"col": 0.017, "fav": 0.983}
    bets = edge.value_bets(model, odds, market_probs=market, max_odds=26.0)
    assert all(b.selection != "col" for b in bets)
    # With the cap lifted it would otherwise appear:
    bets2 = edge.value_bets(model, odds, market_probs=market, max_odds=None,
                            **_noshrink())
    assert any(b.selection == "col" for b in bets2)


def test_value_bets_odds_scaled_shrink_tames_midlongshot():
    model = {"x": 0.12}
    odds = {"x": 12.0}
    market = {"x": 0.07}
    no = edge.value_bets(model, odds, market_probs=market, max_odds=None, **_noshrink())
    yes = edge.value_bets(model, odds, market_probs=market, max_odds=None,
                          shrink=0.25, shrink_odds_scale=0.18)
    # The odds-scaled shrink reduces EV (and may remove the bet entirely).
    if no and yes:
        assert yes[0].ev < no[0].ev
    else:
        assert no  # at minimum the unshrunk version found it


def test_value_bets_min_books_gate():
    model = {"h": 0.62, "d": 0.24, "a": 0.14}
    odds = {"h": 2.1, "d": 3.4, "a": 3.6}
    nb = {"h": 1, "d": 5, "a": 5}
    bets = edge.value_bets(model, odds, n_books=nb, min_books=2, **_noshrink())
    assert all(b.selection != "h" for b in bets)  # 'h' priced by too few books


def test_value_bets_total_exposure_cap():
    # Several juicy bets; their stakes must be scaled to the exposure cap.
    model = {"a": 0.6, "b": 0.6, "c": 0.6}
    odds = {"a": 2.5, "b": 2.5, "c": 2.5}
    market = {"a": 0.4, "b": 0.4, "c": 0.4}
    bets = edge.value_bets(model, odds, bankroll=1000, market_probs=market,
                           kelly_frac=1.0, max_stake_frac=0.5,
                           max_total_exposure_frac=0.25, **_noshrink())
    assert sum(b.stake for b in bets) == pytest.approx(250.0, abs=1.0)


def test_value_bets_uses_supplied_market_not_subset_devig():
    # market_probs passthrough: the fair line is the FULL-book devig, not a devig
    # of the priced subset (which would be biased).
    model = {"a": 0.5, "b": 0.5}
    odds = {"a": 2.5}  # only 'a' offered
    market = {"a": 0.30, "b": 0.70}  # full-book fair line
    bets = edge.value_bets(model, odds, market_probs=market, **_noshrink())
    a = next(b for b in bets if b.selection == "a")
    assert a.market_prob == pytest.approx(0.30)
    assert a.edge == pytest.approx(0.20)


def test_value_bets_empty_inputs():
    assert edge.value_bets({}, {"h": 2.0}) == []
    assert edge.value_bets({"h": 0.5}, {}) == []


# --- arbitrage / dutching ---------------------------------------------------

def test_arbitrage_detects_and_locks_payout():
    odds = {"x": 2.2, "y": 3.8, "z": 4.2}
    res = edge.arbitrage(odds, total_stake=100.0)
    assert res.is_arb and res.guaranteed_return > 0
    for sel, stake in res.stakes.items():
        assert stake * odds[sel] == pytest.approx(res.payout, rel=1e-4)
    assert sum(res.stakes.values()) == pytest.approx(100.0, abs=1e-2)


def test_arbitrage_negative_when_margin():
    res = edge.arbitrage({"h": 2.1, "d": 3.4, "a": 3.6})
    assert not res.is_arb and res.guaranteed_return < 0


def test_dutch_reproduces_ek2024_example():
    o1, o2 = 120 / 70, 120 / 17
    res = edge.dutch_for_target_payout({"exact1": o1, "over1.5": o2}, target_payout=120)
    assert res["stakes"]["exact1"] == pytest.approx(70, abs=0.5)
    assert res["stakes"]["over1.5"] == pytest.approx(17, abs=0.5)
    assert res["total_staked"] == pytest.approx(87, abs=1.0)
    assert res["guaranteed_profit"] == pytest.approx(33, abs=1.0)
    assert res["overround"] < 1.0


# --- closing-line value / paper ledger --------------------------------------

def test_clv_sign():
    # Took 0.18 fair, market closed at 0.22 fair -> beat the close (+CLV).
    assert edge.clv(0.18, 0.22) == pytest.approx(0.04)
    assert edge.clv(0.30, 0.25) == pytest.approx(-0.05)


def test_paper_ledger_summary():
    led = edge.PaperLedger()
    led.record(edge.PaperBet("a", 5.0, 0.18, 0.22, 10.0, closing_fair_prob=0.22, result=True))
    led.record(edge.PaperBet("b", 3.0, 0.34, 0.30, 10.0, closing_fair_prob=0.30, result=False))
    s = led.summary()
    assert s["n_bets"] == 2 and s["n_with_clv"] == 2
    assert s["mean_clv"] == pytest.approx(((0.22 - 0.18) + (0.30 - 0.34)) / 2, abs=1e-6)
    assert s["pct_positive_clv"] == pytest.approx(0.5)
    # pnl: bet a wins 10*(5-1)=40, bet b loses 10 -> net +30
    assert s["pnl"] == pytest.approx(30.0)


# --- market aggregation (probability space) ---------------------------------

def _two_book_events():
    return [{
        "bookmakers": [
            {"markets": [{"key": "outrights", "outcomes": [
                {"name": "Spain", "price": 5.0}, {"name": "Brazil", "price": 7.0}]}]},
            {"markets": [{"key": "outrights", "outcomes": [
                {"name": "Spain", "price": 6.0}, {"name": "Brazil", "price": 9.0}]}]},
        ]
    }]


def test_aggregate_market_best_and_consensus():
    agg = edge.aggregate_market(_two_book_events(), "outrights")
    assert agg["Spain"].best_odds == 6.0  # longest price across books
    assert agg["Spain"].n_books == 2
    # consensus implied = mean(1/5, 1/6) = 0.18333, ABOVE 1/mean(5.5)=0.18181:
    # averaging in probability space preserves more margin than averaging prices.
    assert agg["Spain"].consensus_implied == pytest.approx((0.2 + 1 / 6) / 2)
    assert agg["Spain"].consensus_implied > 1.0 / 5.5


def test_market_fair_probs_sums_to_one():
    agg = edge.aggregate_market(_two_book_events(), "outrights")
    fair = edge.market_fair_probs(agg)
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["Spain"] > fair["Brazil"]


def test_aggregate_drops_degenerate_price():
    events = [{"bookmakers": [{"markets": [{"key": "outrights", "outcomes": [
        {"name": "Spain", "price": 1.0}, {"name": "Spain", "price": 5.0}]}]}]}]
    agg = edge.aggregate_market(events, "outrights")
    assert agg["Spain"].n_books == 1 and agg["Spain"].best_odds == 5.0


# --- match value ------------------------------------------------------------

def test_match_value_finds_value_per_fixture():
    matches = [
        {"home": "Spain", "away": "Albania", "date": "2026-06-24",
         "p_home": 0.74, "p_draw": 0.18, "p_away": 0.08, "result": None},
    ]
    market = {"Spain vs Albania": {"home": 1.5, "draw": 4.2, "away": 7.0}}
    rows = edge.match_value(matches, market, **_noshrink(min_edge=0.02))
    assert rows and isinstance(rows[0], edge.MatchValueBet)
    home = next(r for r in rows if r.selection == "home")
    assert home.match == "Spain vs Albania" and home.ev > 0


def test_match_value_skips_unpriced_matches():
    matches = [{"home": "X", "away": "Y", "p_home": 0.6, "p_draw": 0.2,
                "p_away": 0.2, "result": None}]
    assert edge.match_value(matches, {}) == []


def test_match_odds_by_fixture_shapes_best_price():
    events = [{
        "home_team": "Spain", "away_team": "Albania",
        "bookmakers": [
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Spain", "price": 1.5}, {"name": "Draw", "price": 4.0},
                {"name": "Albania", "price": 6.5}]}]},
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Spain", "price": 1.55}, {"name": "Draw", "price": 4.2},
                {"name": "Albania", "price": 7.0}]}]},
        ]
    }]
    out = edge.match_odds_by_fixture(events)
    assert out["Spain vs Albania"] == {"home": 1.55, "draw": 4.2, "away": 7.0}
