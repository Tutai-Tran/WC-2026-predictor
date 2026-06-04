from wc26.rules import (
    Match,
    Stats,
    compute_stats,
    rank_group,
    rank_thirds,
    select_best_thirds,
)

RANK = {"A": 1, "B": 2, "C": 3, "D": 4}


def test_clean_ranking_by_points():
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 1, 0), Match("A", "C", 1, 0), Match("A", "D", 1, 0),
        Match("B", "C", 1, 0), Match("B", "D", 1, 0), Match("C", "D", 1, 0),
    ]
    # A wins 3, B wins 2, C wins 1, D wins 0
    assert rank_group(teams, matches, RANK) == ["A", "B", "C", "D"]


def test_head_to_head_beats_overall_gd():
    # A and B both 6 pts. A beat B head-to-head, but B has far better overall GD.
    # 2026 rule: head-to-head first => A ranks above B.
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 1, 0),   # A beats B head-to-head
        Match("A", "C", 0, 3),   # A loses heavily
        Match("A", "D", 1, 0),
        Match("B", "C", 2, 0),
        Match("B", "D", 5, 0),   # B piles up GD
        Match("C", "D", 0, 1),
    ]
    order = rank_group(teams, matches, RANK)
    assert order[0] == "A" and order[1] == "B"  # A above B despite B's bigger GD


def test_three_way_tie_falls_through_to_overall_gd():
    # A,B,C form a perfect cycle with identical head-to-head (each 1-0), so
    # head-to-head is equal and overall GD (via results vs D) decides.
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 1, 0), Match("B", "C", 1, 0), Match("C", "A", 1, 0),
        Match("A", "D", 5, 0), Match("B", "D", 3, 0), Match("C", "D", 1, 0),
    ]
    order = rank_group(teams, matches, RANK)
    assert order == ["A", "B", "C", "D"]  # overall GD 5 > 3 > 1


def test_fifa_rank_is_final_tiebreaker():
    # Two identical teams (0-0) separated only by FIFA ranking (lower is better).
    order = rank_group(["X", "Y"], [Match("X", "Y", 0, 0)], {"X": 5, "Y": 10})
    assert order == ["X", "Y"]


def test_partial_resolution_reapplies_head_to_head():
    # A,B,C tied on points; among them A beats both B and C (so A separates first),
    # then B vs C decides the rest by their head-to-head.
    teams = ["A", "B", "C", "D"]
    matches = [
        Match("A", "B", 1, 0), Match("A", "C", 1, 0),  # A wins both h2h
        Match("B", "C", 2, 0),                          # B beats C h2h
        Match("A", "D", 0, 1), Match("B", "D", 0, 1), Match("C", "D", 0, 1),  # all lose to D
    ]
    # A,B,C each: 2 wins (vs the two of them they beat? ) -> compute: A bt B,C; lost D => 6
    # B: lost A, beat C, lost D => 3 ; C: lost A, lost B, lost D => 0 ; D: 9
    # So D first (9). Among rest A=6,B=3,C=0 distinct -> A,B,C.
    order = rank_group(teams, matches, RANK)
    assert order == ["D", "A", "B", "C"]


def test_compute_stats_basic():
    s = compute_stats(["A", "B"], [Match("A", "B", 2, 1)])
    assert s["A"].points == 3 and s["A"].gd == 1 and s["A"].gf == 2
    assert s["B"].points == 0 and s["B"].gd == -1


def test_select_best_thirds():
    thirds = {
        "G1": Stats(3, 4, 5, 2),   # gd 3
        "G2": Stats(3, 4, 3, 1),   # gd 2
        "G3": Stats(3, 3, 4, 4),   # gd 0, 3 pts
        "G4": Stats(3, 1, 1, 5),   # gd -4
        "G5": Stats(3, 6, 8, 2),   # gd 6 -> best
    }
    rank = {k: i + 1 for i, k in enumerate(thirds)}
    best = select_best_thirds(thirds, rank, n=3)
    assert best[0] == "G5"  # biggest GD
    assert "G4" not in best  # worst GD excluded
    assert len(best) == 3
