from wc26 import scrape


def _event(home, hs, away, as_, completed=True):
    return {"date": "2026-06-07T18:00Z", "competitions": [{
        "status": {"type": {"completed": completed}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": home}, "score": hs},
            {"homeAway": "away", "team": {"displayName": away}, "score": as_},
        ]}]}


def test_parse_espn_completed_only():
    data = {"events": [_event("Brazil", "3", "Panama", "1"),
                       _event("Spain", "2", "Iraq", "0", completed=False)]}
    out = scrape.parse_espn(data, "fifa.friendly")
    assert len(out) == 1
    m = out[0]
    assert m["home"] == "Brazil" and m["away"] == "Panama" and m["hg"] == 3 and m["ag"] == 1
    assert m["tournament"] == "Friendly"


def test_parse_espn_applies_aliases():
    data = {"events": [_event("USA", "1", "Czechia", "1")]}
    out = scrape.parse_espn(data, "fifa.world")
    assert out[0]["home"] == "United States" and out[0]["away"] == "Czech Republic"
    assert out[0]["tournament"] == "FIFA World Cup"


def test_parse_espn_skips_bad_scores_and_empty():
    assert scrape.parse_espn({}, "fifa.world") == []
    bad = {"events": [_event("A", "x", "B", "2")]}
    assert scrape.parse_espn(bad, "fifa.world") == []
