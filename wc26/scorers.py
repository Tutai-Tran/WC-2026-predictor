"""Goalscorer model: who is most likely to score.

A player's share of his team's goals is estimated from his goal record, shrunk
toward a position/role prior (international per-player samples are tiny). In a
match where the team is expected to score `team_lambda` open-play goals, the
player's expected goals are share * team_lambda, plus a separate penalty stream
for the designated taker. P(anytime scorer) = 1 - exp(-player_goals).
"""

from __future__ import annotations

import math

POSITION_PRIOR = {"FW": 1.0, "MF": 0.45, "DF": 0.12, "GK": 0.02}
DEFAULT_PRIOR = 0.3
GOAL_WEIGHT = 0.6                 # how strongly the goal record shifts share off the prior
PENALTY_GOALS_PER_MATCH = 0.09    # expected converted-penalty goals for a team's taker


def _prior(position: str | None) -> float:
    return POSITION_PRIOR.get((position or "").upper(), DEFAULT_PRIOR)


def team_goal_shares(players: list[dict]) -> dict[str, float]:
    """Map player name -> share of team goals (sums to 1), shrunk to a role prior."""
    raw = {}
    for p in players:
        goals = p.get("club_goals") or 0  # ingest stored the goal signal here
        raw[p["name"]] = _prior(p.get("position")) + GOAL_WEIGHT * max(0.0, float(goals))
    total = sum(raw.values()) or 1.0
    return {name: w / total for name, w in raw.items()}


def anytime_scorer_probs(
    players: list[dict], team_lambda: float
) -> list[tuple[str, float]]:
    """Return [(player, P(scores at least once))] sorted descending."""
    shares = team_goal_shares(players)
    takers = [p["name"] for p in players if p.get("is_penalty_taker")]
    pen_bonus = (PENALTY_GOALS_PER_MATCH / len(takers)) if takers else 0.0
    out = []
    for p in players:
        mu = shares[p["name"]] * max(0.0, team_lambda)
        if p.get("is_penalty_taker"):
            mu += pen_bonus
        prob = 1.0 - math.exp(-mu)
        out.append((p["name"], prob))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


def top_scorers(players: list[dict], team_lambda: float, n: int = 3) -> list[dict]:
    probs = anytime_scorer_probs(players, team_lambda)[:n]
    return [{"player": name, "p_anytime": round(p, 3)} for name, p in probs]
