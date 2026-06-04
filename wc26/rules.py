"""FIFA World Cup 2026 tournament rules engine (deterministic, heavily tested).

Within-group ranking uses the 2026 order, which is HEAD-TO-HEAD FIRST (a change
from 2022). For teams level on total points:
  1. points in matches among the tied teams (head-to-head)
  2. goal difference in those head-to-head matches
  3. goals scored in those head-to-head matches
  4. overall goal difference (all group matches)
  5. overall goals scored
  6. fair play (no card data available; treated as equal and skipped)
  7. FIFA world ranking (unique, guarantees a total order)
If a multi-team tie only partially separates under head-to-head, criteria 1-3 are
re-applied to the still-tied subset (recursive), per the regulations.

Third-placed teams are ranked across groups by overall points, GD, GF, fair play,
then FIFA ranking; the best 8 advance.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from itertools import groupby

Match = namedtuple("Match", "home away home_goals away_goals")


@dataclass
class Stats:
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def compute_stats(
    teams: list[str],
    matches: list[Match],
    restrict_to: set[str] | None = None,
) -> dict[str, Stats]:
    """Aggregate stats for each team. If restrict_to is given, only count matches
    where BOTH teams are in that set (used for head-to-head mini-tables)."""
    stats = {t: Stats() for t in teams}
    for m in matches:
        if m.home_goals is None or m.away_goals is None:
            continue
        if restrict_to is not None and (m.home not in restrict_to or m.away not in restrict_to):
            continue
        if m.home not in stats or m.away not in stats:
            continue
        hg, ag = int(m.home_goals), int(m.away_goals)
        sh, sa = stats[m.home], stats[m.away]
        sh.played += 1
        sa.played += 1
        sh.gf += hg
        sh.ga += ag
        sa.gf += ag
        sa.ga += hg
        if hg > ag:
            sh.points += 3
        elif hg < ag:
            sa.points += 3
        else:
            sh.points += 1
            sa.points += 1
    return stats


def rank_group(
    teams: list[str],
    matches: list[Match],
    fifa_rank: dict[str, int],
) -> list[str]:
    """Return the teams ordered best to worst per the 2026 tie-break rules."""
    overall = compute_stats(teams, matches)

    def overall_break(grp: list[str]) -> list[str]:
        # criteria 4,5, (fair play skipped), 7. FIFA rank lower is better.
        return sorted(
            grp,
            key=lambda t: (overall[t].gd, overall[t].gf, -fifa_rank.get(t, 9999)),
            reverse=True,
        )

    def break_tie(block: list[str]) -> list[str]:
        """All teams in `block` share the same total points."""
        mini = compute_stats(block, matches, restrict_to=set(block))
        block_sorted = sorted(
            block,
            key=lambda t: (mini[t].points, mini[t].gd, mini[t].gf),
            reverse=True,
        )
        result: list[str] = []
        for _, grp_iter in groupby(
            block_sorted, key=lambda t: (mini[t].points, mini[t].gd, mini[t].gf)
        ):
            grp = list(grp_iter)
            if len(grp) == 1:
                result.extend(grp)
            elif len(grp) == len(block):
                # head-to-head did not separate anyone -> overall criteria
                result.extend(overall_break(grp))
            else:
                # partial separation -> re-apply head-to-head to the still-tied subset
                result.extend(break_tie(grp))
        return result

    # primary ordering: total points, with tie-break blocks resolved above
    teams_sorted = sorted(teams, key=lambda t: overall[t].points, reverse=True)
    ordered: list[str] = []
    for _, block_iter in groupby(teams_sorted, key=lambda t: overall[t].points):
        block = list(block_iter)
        ordered.extend(block if len(block) == 1 else break_tie(block))
    return ordered


def rank_thirds(
    third_stats: dict[str, Stats],
    fifa_rank: dict[str, int],
) -> list[str]:
    """Rank the third-placed teams across groups (best first)."""
    return sorted(
        third_stats,
        key=lambda t: (
            third_stats[t].points,
            third_stats[t].gd,
            third_stats[t].gf,
            -fifa_rank.get(t, 9999),
        ),
        reverse=True,
    )


def select_best_thirds(
    third_stats: dict[str, Stats],
    fifa_rank: dict[str, int],
    n: int = 8,
) -> list[str]:
    return rank_thirds(third_stats, fifa_rank)[:n]
