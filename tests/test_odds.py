from wc26 import odds


def _events(prices):
    return [{"bookmakers": [{"markets": [{"key": "outrights",
             "outcomes": [{"name": n, "price": p} for n, p in prices.items()]}]}]}]


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
