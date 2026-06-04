"""World Football Elo engine.

Elo is replayable: current ratings are a deterministic function of a seed plus an
ordered list of results, so a corrected result self-heals on the next recompute
(see PLAN.md). Update follows the World Football Elo conventions: a goal-difference
multiplier and a match-importance K factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_ELO = 1500.0


def expected(elo_a: float, elo_b: float, home_adv: float = 0.0) -> float:
    """Expected score (win probability + half draw) for team A."""
    return 1.0 / (1.0 + 10.0 ** (-((elo_a + home_adv) - elo_b) / 400.0))


def gd_multiplier(goal_diff: int) -> float:
    """World Football Elo goal-difference weighting."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def k_for_tournament(tournament: str | None) -> float:
    """Match-importance K factor."""
    t = (tournament or "").lower()
    if "friendly" in t:
        return 20.0
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if "qualif" in t:
        return 40.0
    if any(x in t for x in ("euro", "copa", "cup of nations", "asian cup",
                            "gold cup", "confederations", "nations league")):
        return 50.0
    return 40.0


def update_pair(
    elo_a: float,
    elo_b: float,
    goals_a: int,
    goals_b: int,
    k: float = 40.0,
    home_adv: float = 0.0,
) -> tuple[float, float]:
    """Return updated (elo_a, elo_b) after one match. Zero-sum."""
    we_a = expected(elo_a, elo_b, home_adv)
    if goals_a > goals_b:
        w_a = 1.0
    elif goals_a < goals_b:
        w_a = 0.0
    else:
        w_a = 0.5
    g = gd_multiplier(goals_a - goals_b)
    delta = k * g * (w_a - we_a)
    return elo_a + delta, elo_b - delta


@dataclass
class EloConfig:
    default_elo: float = DEFAULT_ELO
    home_adv: float = 60.0  # Elo points for the home team when not a neutral venue


def replay(
    results: pd.DataFrame,
    seed: dict[str, float] | None = None,
    config: EloConfig | None = None,
) -> dict[str, float]:
    """Replay an ordered results frame into current Elo ratings.

    Expected columns: home_team, away_team, home_score, away_score, neutral
    (optional: tournament, date). Rows are processed in the given order; sort by
    date beforehand for a chronological replay.
    """
    cfg = config or EloConfig()
    ratings: dict[str, float] = dict(seed or {})

    def get(team: str) -> float:
        return ratings.get(team, cfg.default_elo)

    has_tournament = "tournament" in results.columns
    has_neutral = "neutral" in results.columns

    for row in results.itertuples(index=False):
        home = getattr(row, "home_team")
        away = getattr(row, "away_team")
        hs = getattr(row, "home_score")
        as_ = getattr(row, "away_score")
        if home is None or away is None or pd.isna(hs) or pd.isna(as_):
            continue
        k = k_for_tournament(getattr(row, "tournament")) if has_tournament else 40.0
        neutral = bool(getattr(row, "neutral")) if has_neutral else False
        home_adv = 0.0 if neutral else cfg.home_adv
        new_home, new_away = update_pair(
            get(home), get(away), int(hs), int(as_), k=k, home_adv=home_adv
        )
        ratings[home] = new_home
        ratings[away] = new_away
    return ratings
