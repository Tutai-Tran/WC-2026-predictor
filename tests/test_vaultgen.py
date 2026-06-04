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
