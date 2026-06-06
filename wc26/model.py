"""Elo-driven goal-expectation match model.

Expected goals come from the Elo rating difference (few global parameters), not
from per-team attack/defence fits, which overfit sparse international data. The
scoreline distribution is Poisson with an optional Dixon-Coles low-score
correction. Global parameters are tuned in the backtest; the defaults here are
sensible fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class ModelParams:
    c: float = 110.0          # Elo points per goal of supremacy
    base_goals: float = 2.6   # expected total goals in an even match
    rho: float = -0.06        # Dixon-Coles low-score correction (competitive matches)
    rho_friendly: float = -0.06   # stronger draw correction for friendlies (they under-predict draws)
    max_goals: int = 10
    min_lambda: float = 0.15
    goal_scale: float = 1.0   # leak-free multiplier on total goals (fit in the backtest)
    gamma: float = 0.0        # extra total goals per unit |supremacy| (mismatches score more)


def match_lambdas(
    elo_a: float,
    elo_b: float,
    params: ModelParams = ModelParams(),
    home_adv_elo_a: float = 0.0,
    home_adv_elo_b: float = 0.0,
    h2h_delta: float = 0.0,
) -> tuple[float, float]:
    """Expected goals (lambda_a, lambda_b) from the effective Elo difference.

    `h2h_delta` (goal units, positive favours team A) is an optional head-to-head
    adjustment added to the Elo-derived supremacy; see `wc26.h2h`.
    """
    eff_diff = (elo_a + home_adv_elo_a) - (elo_b + home_adv_elo_b)
    supremacy = eff_diff / params.c + h2h_delta
    # Total goals (la + lb) = base. goal_scale is a flat leak-free volume correction;
    # gamma * |supremacy| makes mismatches score more in total (the favourite runs up
    # the score), giving per-match variation in totals. Supremacy (la - lb) is the
    # only driver of the favourite, so the modal W/D/L pick is unchanged by either term.
    base = params.base_goals * params.goal_scale + params.gamma * abs(supremacy)
    la = (base + supremacy) / 2.0
    lb = (base - supremacy) / 2.0
    return max(params.min_lambda, la), max(params.min_lambda, lb)


def _dc_tau(matrix: np.ndarray, la: float, lb: float, rho: float) -> np.ndarray:
    """Apply the Dixon-Coles correction to the four low-score cells."""
    m = matrix.copy()
    m[0, 0] *= 1.0 - la * lb * rho
    m[0, 1] *= 1.0 + la * rho
    m[1, 0] *= 1.0 + lb * rho
    m[1, 1] *= 1.0 - rho
    return m


def scoreline_matrix(
    la: float, lb: float, max_goals: int = 10, rho: float = -0.06
) -> np.ndarray:
    """P(home=i, away=j) matrix, shape (max_goals+1, max_goals+1), sums to ~1."""
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, la)
    pa = poisson.pmf(goals, lb)
    matrix = np.outer(ph, pa)
    if rho:
        matrix = _dc_tau(matrix, la, lb, rho)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from a scoreline matrix."""
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    return p_home, p_draw, p_away


def top_scorelines(matrix: np.ndarray, n: int = 3) -> list[tuple[tuple[int, int], float]]:
    """The n most likely exact scorelines with their probabilities."""
    flat = np.argsort(matrix, axis=None)[::-1][:n]
    out = []
    for idx in flat:
        i, j = np.unravel_index(idx, matrix.shape)
        out.append(((int(i), int(j)), float(matrix[i, j])))
    return out


def outcome_label(p_home: float, p_draw: float, p_away: float) -> str:
    """The modal outcome ('home' | 'draw' | 'away') for a set of outcome probs."""
    return ("home", "draw", "away")[int(np.argmax([p_home, p_draw, p_away]))]


def result_label(home_goals: int, away_goals: int) -> str:
    """The actual outcome ('home' | 'draw' | 'away') of a finished match."""
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def forecast_hit(p_home: float, p_draw: float, p_away: float,
                 home_goals: int, away_goals: int) -> bool:
    """True if our modal W/D/L pick matched the actual result of a played match."""
    return outcome_label(p_home, p_draw, p_away) == result_label(home_goals, away_goals)


def most_likely_scoreline(
    matrix: np.ndarray, outcome: str | None = None
) -> tuple[tuple[int, int], float]:
    """Most likely exact scoreline, optionally restricted to one outcome region.

    Passing `outcome` ('home' | 'draw' | 'away') returns the most likely scoreline
    *consistent with that result*, so the headline score never contradicts the
    headline winner (e.g. a predicted home win shows 2-1, not a 1-1 draw)."""
    if outcome == "home":
        mask = np.tril(np.ones_like(matrix), -1)   # home goals > away goals
    elif outcome == "away":
        mask = np.triu(np.ones_like(matrix), 1)    # away goals > home goals
    elif outcome == "draw":
        mask = np.eye(matrix.shape[0])
    else:
        mask = np.ones_like(matrix)
    idx = int(np.argmax(matrix * mask))
    i, j = np.unravel_index(idx, matrix.shape)
    return (int(i), int(j)), float(matrix[i, j])


def match_forecast(
    elo_a: float,
    elo_b: float,
    params: ModelParams = ModelParams(),
    home_adv_elo_a: float = 0.0,
    home_adv_elo_b: float = 0.0,
    mult_a: float = 1.0,
    mult_b: float = 1.0,
    h2h_delta: float = 0.0,
) -> dict:
    """Full single-match forecast: outcome probs, top scorelines, lambdas.

    mult_a/mult_b scale each team's attacking expected goals (e.g. an availability
    adjustment when key players are out). `h2h_delta` adds a head-to-head supremacy
    nudge (see `wc26.h2h`)."""
    la, lb = match_lambdas(elo_a, elo_b, params, home_adv_elo_a, home_adv_elo_b, h2h_delta)
    la *= mult_a
    lb *= mult_b
    matrix = scoreline_matrix(la, lb, params.max_goals, params.rho)
    p_home, p_draw, p_away = outcome_probs(matrix)
    modal = outcome_label(p_home, p_draw, p_away)
    consistent, consistent_p = most_likely_scoreline(matrix, modal)
    return {
        "lambda_home": la,
        "lambda_away": lb,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "top_scorelines": top_scorelines(matrix, 3),
        "likely_scoreline": consistent,            # most likely score for the predicted result
        "likely_scoreline_p": consistent_p,
        "matrix": matrix,
    }
