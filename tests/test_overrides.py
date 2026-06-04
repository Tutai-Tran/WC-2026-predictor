from wc26 import db, ingest, forecast, simulate, overrides

NOTE = """---
team: "Brazil"
---

## Manual overrides
<!-- WC26:HUMAN:override -->
_- Vinicius: doubtful, 50% minutes (example, ignored)_
- Neymar: out, until 2026-07-01, source: ESPN
- Raphinha: doubtful, 50% minutes, source: Marca

## Change log
<!-- WC26:AUTO:changelog START -->
"""


def _players_by_team(conn):
    pbt = {}
    for r in conn.execute(
        "SELECT t.name tn, p.name pn, p.position pos, p.club_goals cg, p.is_penalty_taker pk "
        "FROM players p JOIN teams t ON t.id=p.team_id"
    ):
        pbt.setdefault(r["tn"], []).append(
            {"name": r["pn"], "position": r["pos"], "club_goals": r["cg"],
             "is_penalty_taker": bool(r["pk"])})
    return pbt


def test_parse_override_block_skips_example():
    ovs = {o["player"]: o for o in overrides.parse_override_block(NOTE)}
    assert set(ovs) == {"Neymar", "Raphinha"}            # italic example ignored
    assert ovs["Neymar"]["status"] == "out" and ovs["Neymar"]["minutes_factor"] == 0.0
    assert ovs["Neymar"]["expected_return"] == "2026-07-01"
    assert abs(ovs["Raphinha"]["minutes_factor"] - 0.5) < 1e-9


def test_availability_multiplier(tmp_path):
    conn = db.connect(tmp_path / "t.db"); ingest.ingest_all(conn)
    pbt = _players_by_team(conn)
    assert overrides.availability_multipliers(conn, pbt)["Spain"] == 1.0
    star = conn.execute("SELECT p.name FROM players p JOIN teams t ON t.id=p.team_id "
                        "WHERE t.name='Spain' ORDER BY p.club_goals DESC LIMIT 1").fetchone()["name"]
    overrides.add_availability(conn, "Spain", star, "out", 0.0)
    mult = overrides.availability_multipliers(conn, pbt)["Spain"]
    assert 0.4 <= mult < 1.0


def test_add_result_marks_played_and_ledger(tmp_path):
    conn = db.connect(tmp_path / "t.db"); ingest.ingest_all(conn)
    before_p = conn.execute("SELECT COUNT(*) c FROM matches WHERE played=1").fetchone()["c"]
    before_l = conn.execute("SELECT COUNT(*) c FROM results_ledger").fetchone()["c"]
    fx = conn.execute(
        "SELECT h.name home, a.name away FROM matches m JOIN teams h ON h.id=m.home_team_id "
        "JOIN teams a ON a.id=m.away_team_id WHERE m.stage='group' LIMIT 1").fetchone()
    overrides.add_result(conn, fx["home"], fx["away"], 4, 0)
    after_p = conn.execute("SELECT COUNT(*) c FROM matches WHERE played=1").fetchone()["c"]
    after_l = conn.execute("SELECT COUNT(*) c FROM results_ledger").fetchone()["c"]
    assert after_p == before_p + 1  # the group fixture is now marked played
    assert after_l == before_l + 1  # one ledger row appended


def test_add_result_friendly_uses_correct_tournament_and_date(tmp_path):
    conn = db.connect(tmp_path / "t.db"); ingest.ingest_all(conn)
    overrides.add_result(conn, "Brazil", "Andorra", 3, 0, stage="friendly",
                         source="manual", played_on="2026-06-01")
    row = conn.execute(
        "SELECT tournament, played_on FROM results_ledger "
        "WHERE source='manual' AND home_team='Brazil' ORDER BY id DESC LIMIT 1").fetchone()
    assert row["tournament"] == "Friendly"      # not 'FIFA World Cup' (correct Elo K)
    assert row["played_on"] == "2026-06-01"     # real match date (dedup + replay order)


def test_sync_vault_overrides(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db"); ingest.ingest_all(conn)
    vault = tmp_path / "vault"
    (vault / "Countries").mkdir(parents=True)
    (vault / "Countries" / "Brazil.md").write_text(NOTE)
    monkeypatch.setattr(overrides.config, "VAULT", vault)
    rep = overrides.sync_vault_overrides(conn)
    assert rep["events"] == 2
    names = {r["player_name"] for r in conn.execute("SELECT player_name FROM availability_events")}
    assert names == {"Neymar", "Raphinha"}


def test_conditional_sim_respects_played_and_keeps_invariants(tmp_path):
    conn = db.connect(tmp_path / "t.db"); ingest.ingest_all(conn)
    t = forecast.load_tournament(conn)
    fx = t.group_fixtures[0]
    t.played[(fx["home"], fx["away"])] = (0, 5)
    probs = simulate.simulate(t, n_runs=300, seed=1)["probs"]
    assert abs(sum(p["champion"] for p in probs.values()) - 1.0) < 1e-9
    assert abs(sum(p["advance"] for p in probs.values()) - 32.0) < 1e-9
