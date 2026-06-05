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


def test_recompute_elo_flags_unmatched_team_names(tmp_path, monkeypatch):
    """A scraped name missing from base_elo (an ESPN alias gap) must be surfaced,
    not silently turned into a phantom rating that never grades."""
    import shutil
    from wc26 import db, config
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(config.DATA_RAW / "base_elo.json", raw / "base_elo.json")   # real 336-team set
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)                        # isolate side-effect write

    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO results_ledger (home_team, away_team, home_goals, away_goals, "
        "played_on, tournament, neutral) VALUES (?,?,?,?,?,?,?)",
        ("Brazil", "Notarealteam FC", 2, 0, "2026-06-12", "FIFA World Cup", 1))
    conn.commit()

    rep = scrape.recompute_elo(conn)
    assert rep["recomputed"] is True
    assert "Notarealteam FC" in rep.get("unmatched_teams", [])     # the alias gap is flagged
    assert "Brazil" not in rep.get("unmatched_teams", [])          # a recognised team is not


def test_recompute_elo_canonicalises_spelling_variants(tmp_path, monkeypatch):
    """An ESPN spelling variant (and base_elo's own diacritic spellings) must be
    canonicalised so the real team gets the Elo, not a phantom that never grades."""
    import json, shutil
    from wc26 import db, config
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(config.DATA_RAW / "base_elo.json", raw / "base_elo.json")
    monkeypatch.setattr("wc26.config.DATA_RAW", raw)

    conn = db.connect(tmp_path / "t.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO results_ledger (home_team, away_team, home_goals, away_goals, "
        "played_on, tournament, neutral) VALUES (?,?,?,?,?,?,?)",
        ("Bosnia-Herzegovina", "Curacao", 2, 0, "2026-06-02", "Friendly", 1))   # both variants
    conn.commit()

    rep = scrape.recompute_elo(conn)
    assert rep.get("unmatched_teams", []) == []                    # both canonicalised, none flagged
    replayed = json.loads((raw / "replayed_elo.json").read_text())["ratings"]
    assert "Bosnia and Herzegovina" in replayed                    # ESPN variant merged into canonical
    assert "Bosnia-Herzegovina" not in replayed
    assert "Curacao" in replayed                                   # base_elo's "Curaçao" canonicalised


def test_recompute_elo_dedupes_variant_and_canonical_rows(tmp_path, monkeypatch):
    """A match stored under BOTH its old spelling and the canonical name (which an
    aliased re-scrape can produce) must update Elo once, not twice."""
    import json, shutil
    from wc26 import db, config

    def _elo_after(rows):
        raw = tmp_path / f"run{len(rows)}"                   # unique DATA_RAW per call
        raw.mkdir()
        shutil.copy(config.DATA_RAW / "base_elo.json", raw / "base_elo.json")
        monkeypatch.setattr("wc26.config.DATA_RAW", raw)
        conn = db.connect(raw / "t.db")
        db.init_db(conn)
        for h, a in rows:
            conn.execute(
                "INSERT INTO results_ledger (home_team, away_team, home_goals, away_goals, "
                "played_on, tournament, neutral) VALUES (?,?,3,0,'2026-06-02','Friendly',1)", (h, a))
        conn.commit()
        scrape.recompute_elo(conn)
        return json.loads((raw / "replayed_elo.json").read_text())["ratings"]["Brazil"]

    once = _elo_after([("Brazil", "DR Congo")])
    twice = _elo_after([("Brazil", "DR Congo"), ("Brazil", "Congo DR")])   # same match, two spellings
    assert once == twice                                     # de-duplicated: applied once
