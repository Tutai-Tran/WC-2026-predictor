from wc26 import vaultgen

NOTE = """---
type: wc-country
champion_prob:
updated: 2026-06-01
---

# Brazil

<!-- WC26:AUTO:snapshot START | generated old | source x -->
old machine content
<!-- WC26:AUTO:snapshot END -->

## My read
<!-- WC26:HUMAN:notes -->
My private analysis that must survive.

## Manual overrides
<!-- WC26:HUMAN:override -->
- Neymar: doubtful
"""


def test_replace_auto_block_updates_only_machine():
    out = vaultgen.replace_auto_block(NOTE, "snapshot", "NEW machine content", source="v1")
    assert "NEW machine content" in out
    assert "old machine content" not in out
    # human content untouched
    assert "My private analysis that must survive." in out
    assert "- Neymar: doubtful" in out


def test_replace_auto_block_noop_when_marker_absent():
    out = vaultgen.replace_auto_block(NOTE, "does_not_exist", "X")
    assert out == NOTE


def test_set_frontmatter():
    out = vaultgen.set_frontmatter(NOTE, "champion_prob", 0.123)
    assert "champion_prob: 0.123" in out
    # does not duplicate
    assert out.count("champion_prob:") == 1


def test_human_block_preserved_through_two_generations():
    once = vaultgen.replace_auto_block(NOTE, "snapshot", "gen1")
    twice = vaultgen.replace_auto_block(once, "snapshot", "gen2")
    assert "gen2" in twice and "gen1" not in twice
    assert "My private analysis that must survive." in twice


def test_scoreline_block_renders_distribution_not_single_prediction():
    m = {
        "top_scoreline": "1-0", "top_scoreline_p": 0.128,
        "top_scorelines": [[[1, 1], 0.138], [[1, 0], 0.128], [[0, 0], 0.113],
                           [[0, 1], 0.093], [[2, 0], 0.088]],
        "expected_score": "1-1",
        "derived_markets": {"over_2_5": 0.39, "under_2_5": 0.61,
                            "btts_yes": 0.46, "btts_no": 0.54,
                            "total_goals_bands": {"0-1": 0.3, "2": 0.3, "3": 0.2, "4+": 0.2},
                            "clean_sheet_home": 0.4, "clean_sheet_away": 0.35},
    }
    out = vaultgen._scoreline_block(m)
    # all five scorelines appear with probabilities
    for cell in ("1-1", "1-0", "0-0", "0-1", "2-0"):
        assert cell in out
    # the modal cell is framed as one possibility, not "the prediction"
    assert "Most likely single score: 1-0" in out
    assert "not a prediction" in out
    # side-markets surfaced
    assert "Over 2.5" in out and "BTTS yes" in out
    assert "Expected score: 1-1" in out
    # cumulative top-5 mass is reported
    assert "cumulative" in out
