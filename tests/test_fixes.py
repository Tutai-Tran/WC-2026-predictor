"""Regression tests for issues found by the adversarial test workflow."""

from wc26 import db, ingest, forecast, vaultgen
from wc26.rules import Match, rank_group

RANK = {"A": 1, "B": 2, "C": 3, "D": 4}


def test_head_to_head_reapplied_to_still_tied_pair():
    # A,B,C all 6 pts. Among them C is best on h2h GD; A and B stay level on
    # h2h pts/GD/GF, so h2h is REAPPLIED to {A,B}: A beat B -> A above B.
    # Correct FIFA 2026 order is C, A, B, D (not C, B, A, D via overall GD).
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 3, 1), Match("A", "C", 1, 4), Match("B", "C", 3, 2),
        Match("A", "D", 1, 0), Match("B", "D", 1, 0), Match("C", "D", 1, 0),
    ]
    assert rank_group(teams, matches, RANK) == ["C", "A", "B", "D"]


def test_load_tournament_uses_latest_rating(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    ingest.ingest_all(conn)
    tid = conn.execute("SELECT id FROM teams WHERE name='Brazil'").fetchone()["id"]
    conn.execute("INSERT INTO ratings(team_id, elo, valid_from, source) VALUES(?,?,?,?)",
                 (tid, 1000.0, "2000-01-01", "stale"))
    conn.commit()
    t = forecast.load_tournament(conn)
    assert t.teams["Brazil"]["elo"] > 1500  # latest replay, not the stale 1000


def test_vault_link_alias():
    assert vaultgen._link("Curacao") == "Curaçao"
    assert vaultgen._link("Brazil") == "Brazil"


def test_set_frontmatter_ignores_body_lines():
    note = "---\ntype: x\n---\n\n## My read\ngroup: my favourite is the death group\n"
    out = vaultgen.set_frontmatter(note, "group", '"A"')
    assert "group: my favourite is the death group" in out  # body untouched
    assert 'group: "A"' not in out


def test_set_frontmatter_updates_real_key():
    note = "---\ntype: x\ngroup: \n---\n\nbody\n"
    out = vaultgen.set_frontmatter(note, "group", '"A"')
    assert 'group: "A"' in out
    assert out.endswith("body\n")


def test_match_note_preserves_human_via_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(vaultgen, "MATCHES", tmp_path)
    result = {"matches": [{"group": "A", "home": "X", "away": "Y",
                           "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
                           "top_scoreline": "1-0", "top_scoreline_p": 0.1,
                           "top_scorers_home": [], "top_scorers_away": []}],
              "data_as_of": "2026-06-04", "n_runs": 100, "seed": 1}
    vaultgen.write_match_notes(result)
    f = tmp_path / "A - X vs Y.md"
    f.write_text(f.read_text() + "my private post-match note\n")
    vaultgen.write_match_notes(result)  # regenerate
    assert "my private post-match note" in f.read_text()


def test_replace_auto_block_survives_orphan_start():
    note = ("<!-- WC26:AUTO:snapshot START | x -->\norphan no end\n"
            "PRECIOUS HUMAN\n"
            "<!-- WC26:AUTO:snapshot START | y -->\nreal\n<!-- WC26:AUTO:snapshot END -->\n")
    out = vaultgen.replace_auto_block(note, "snapshot", "NEW")
    assert "PRECIOUS HUMAN" in out  # not swallowed across the orphan START
