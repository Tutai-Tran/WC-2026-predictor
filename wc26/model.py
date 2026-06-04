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
    rho: float = -0.06        # Dixon-Coles low-score correction
    max_goals: int = 10
    min_lambda: float = 0.15


def match_lambdas(
    elo_a: float,
    elo_b: float,
    params: ModelParams = ModelParams(),
    home_adv_elo_a: float = 0.0,
    home_adv_elo_b: float = 0.0,
) -> tuple[float, float]:
    """Expected goals (lambda_a, lambda_b) from the effective Elo difference."""
    eff_diff = (elo_a + home_adv_elo_a) - (elo_b + home_adv_elo_b)
    supremacy = eff_diff / params.c
    la = (params.base_goals + supremacy) / 2.0
    lb = (params.base_goals - supremacy) / 2.0
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


def match_forecast(
    elo_a: float,
    elo_b: float,
    params: ModelParams = ModelParams(),
    home_adv_elo_a: float = 0.0,
    home_adv_elo_b: float = 0.0,
) -> dict:
    """Full single-match forecast: outcome probs, top scorelines, lambdas."""
    la, lb = match_lambdas(elo_a, elo_b, params, home_adv_elo_a, home_adv_elo_b)
    matrix = scoreline_matrix(la, lb, params.max_goals, params.rho)
    p_home, p_draw, p_away = outcome_probs(matrix)
    return {
        "lambda_home": la,
        "lambda_away": lb,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "top_scorelines": top_scorelines(matrix, 3),
        "matrix": matrix,
    }
