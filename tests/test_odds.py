import math

from wc26 import odds


def _events(prices):
    return [{"bookmakers": [{"markets": [{"key": "outrights",
             "outcomes": [{"name": n, "price": p} for n, p in prices.items()]}]}]}]


def _h2h_event(home, away, prices, n_books=1):
    """One h2h event with `n_books` identical bookmakers pricing home/draw/away.

    `prices` maps the OUTCOME NAME (home team / 'Draw' / away team) to decimal odds."""
    bk = {"markets": [{"key": "h2h",
          "outcomes": [{"name": n, "price": p} for n, p in prices.items()]}]}
    return {"home_team": home, "away_team": away, "bookmakers": [bk] * n_books}


def test_devig_normalizes_to_one():
    p = odds.devig_outrights(_events({"Spain": 4.0, "France": 4.0, "Brazil": 4.0}))
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-9 for v in p.values())  # equal prices -> equal probs


def test_devig_applies_name_alias():
    p = odds.devig_outrights(_events({"USA": 2.0, "Spain": 2.0}))
    assert "United States" in p and "USA" not in p
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_devig_averages_across_bookmakers():
    events = [{"bookmakers": [
        {"markets": [{"key": "outrights", "outcomes": [{"name": "Spain", "price": 2.0}]}]},
        {"markets": [{"key": "outrights", "outcomes": [{"name": "Spain", "price": 4.0}]}]},
    ]}]
    # only one team -> normalises to 1.0; just assert no crash and it is present
    p = odds.devig_outrights(events)
    assert p["Spain"] == 1.0


# --------------------------------------------------------------------------
# devig_h2h: per-fixture 3-way devig (sums to 1, removes the vig)
# --------------------------------------------------------------------------

def test_devig_h2h_sums_to_one_and_keys_by_fixture():
    ev = _h2h_event("Spain", "France", {"Spain": 2.1, "Draw": 3.4, "France": 3.6})
    out = odds.devig_h2h([ev])
    assert set(out) == {"Spain vs France"}
    probs = out["Spain vs France"]
    assert set(probs) == {"home", "draw", "away"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_devig_h2h_removes_vig():
    # 1/2.0 + 1/4.0 + 1/4.0 = 1.0 booksum, but add a real margin via shorter prices
    ev = _h2h_event("A", "B", {"A": 1.9, "Draw": 3.6, "B": 3.6})
    raw = {"home": 1 / 1.9, "draw": 1 / 3.6, "away": 1 / 3.6}
    assert sum(raw.values()) > 1.0                       # a genuine overround exists
    out = odds.devig_h2h([ev])["A vs B"]
    assert abs(sum(out.values()) - 1.0) < 1e-9           # devigged to a true distribution
    assert all(out[k] < raw[k] for k in out)             # every fair prob below its implied


def test_devig_h2h_skips_incomplete_market():
    ev = _h2h_event("A", "B", {"A": 2.0, "Draw": 3.5})    # no away price -> not 3-way
    assert odds.devig_h2h([ev]) == {}


def test_devig_h2h_aggregates_across_books():
    ev = _h2h_event("A", "B", {"A": 2.0, "Draw": 3.5, "B": 4.0}, n_books=3)
    out = odds.devig_h2h([ev])["A vs B"]
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["home"] > out["away"]                      # shorter home price -> higher prob


# --------------------------------------------------------------------------
# blend_wdl: log (geometric) opinion pool
# --------------------------------------------------------------------------

def test_blend_wdl_weight_one_is_pure_model():
    model = {"home": 0.6, "draw": 0.25, "away": 0.15}
    market = {"home": 0.45, "draw": 0.30, "away": 0.25}
    out = odds.blend_wdl(model, market, weight=1.0)
    assert all(abs(out[k] - model[k]) < 1e-9 for k in model)


def test_blend_wdl_weight_zero_is_pure_market():
    model = {"home": 0.6, "draw": 0.25, "away": 0.15}
    market = {"home": 0.45, "draw": 0.30, "away": 0.25}
    out = odds.blend_wdl(model, market, weight=0.0)
    assert all(abs(out[k] - market[k]) < 1e-9 for k in market)


def test_blend_wdl_is_geometric_pool():
    model = {"home": 0.6, "draw": 0.25, "away": 0.15}
    market = {"home": 0.45, "draw": 0.30, "away": 0.25}
    w = 0.35
    out = odds.blend_wdl(model, market, weight=w)
    # closed-form: p_k proportional to m_k^w * q_k^(1-w), renormalised
    raw = {k: model[k] ** w * market[k] ** (1 - w) for k in model}
    z = sum(raw.values())
    for k in model:
        assert abs(out[k] - raw[k] / z) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_blend_wdl_cools_a_hot_favourite_toward_market():
    """A model-hot favourite blended at the market-heavy prior lands BETWEEN model and
    market and has its favourite cooled toward the (lower) market line."""
    model = {"home": 0.60, "draw": 0.25, "away": 0.15}
    market = {"home": 0.45, "draw": 0.30, "away": 0.25}
    out = odds.blend_wdl(model, market, weight=0.35)
    assert market["home"] < out["home"] < model["home"]   # favourite cooled, still between
    assert model["draw"] < out["draw"] < market["draw"]    # draw lifted toward market
    assert model["away"] < out["away"] < market["away"]    # underdog lifted toward market
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_blend_wdl_no_market_returns_model():
    model = {"home": 0.5, "draw": 0.3, "away": 0.2}
    out = odds.blend_wdl(model, {})
    assert out == model
    assert out is not model                                # a copy, not the same dict


def test_blend_wdl_default_weight_is_market_heavy():
    assert odds.DEFAULT_MATCH_BLEND_WEIGHT == 0.35         # model share; market gets 0.65


# --------------------------------------------------------------------------
# match_blend_weight reader: fitted param if present, else defensible prior
# --------------------------------------------------------------------------

def test_match_blend_weight_reads_fitted_param(tmp_path, monkeypatch):
    import json as _json

    from wc26 import config
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fitted_params.json").write_text(_json.dumps({"blend_weight": 0.28}))
    monkeypatch.setattr(config, "DATA_RAW", raw)
    assert odds.match_blend_weight() == 0.28


def test_match_blend_weight_falls_back_to_default(tmp_path, monkeypatch):
    from wc26 import config
    raw = tmp_path / "raw"
    raw.mkdir()                                            # no fitted_params.json
    monkeypatch.setattr(config, "DATA_RAW", raw)
    assert odds.match_blend_weight() == odds.DEFAULT_MATCH_BLEND_WEIGHT


# --------------------------------------------------------------------------
# store / read round-trip for the scope='match' benchmark
# --------------------------------------------------------------------------

def test_store_and_read_h2h_benchmark_roundtrip(tmp_path):
    from wc26 import db
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    fixtures = {"A vs B": {"home": 0.5, "draw": 0.3, "away": 0.2}}
    odds.store_h2h_benchmark(conn, fixtures)
    assert odds.latest_market_h2h(conn) == fixtures
    # an empty map writes nothing and the reader stays on the last good snapshot
    odds.store_h2h_benchmark(conn, {})
    assert odds.latest_market_h2h(conn) == fixtures


def test_latest_market_h2h_empty_when_none(tmp_path):
    from wc26 import db
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    assert odds.latest_market_h2h(conn) == {}


def test_refresh_h2h_skips_without_key(tmp_path, monkeypatch):
    from wc26 import db
    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    monkeypatch.setattr(odds, "_key", lambda: None)
    rep = odds.refresh_h2h(conn)
    assert rep["fixtures"] == 0 and "skipped" in rep
    assert odds.latest_market_h2h(conn) == {}              # nothing written
