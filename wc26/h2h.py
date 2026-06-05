"""Head-to-head adjustment.

Elo captures a team's overall strength but not the specific way two teams have
historically matched up. This module turns the prior meetings between two teams
into a small *supremacy delta* (in goal units, the same scale used by
`model.match_lambdas`) that nudges the Elo-derived expectation toward the
documented head-to-head tendency.

The signal is the recency-weighted, venue-corrected mean goal difference over
past meetings, shrunk toward zero by the (effective) sample size, scaled by a
small global weight, and capped. With no shared history the delta is exactly
0.0, so the model falls back to pure Elo.

Design choices (kept deliberately conservative so the calibrated Elo model stays
the backbone, see PLAN.md / backtest):
- recency: each meeting is weighted by 0.5 ** (age_years / HALF_LIFE_YEARS).
- venue: a home win is worth less than the same margin on neutral ground, so
  each non-neutral meeting's goal difference is de-biased by HOME_GD.
- shrinkage: weight = n_eff / (n_eff + SHRINK_K); one meeting barely moves it.
- influence + cap: delta = clip(WEIGHT * shrink * mean_gd, ±DELTA_CAP).

Only matches strictly on/under CUTOFF with both scores present are used, so the
World Cup itself can never leak into its own head-to-head priors.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from . import config

HALF_LIFE_YEARS = 10.0   # a meeting's weight halves every 10 years
SHRINK_K = 4.0           # effective meetings needed before H2H reaches half strength
WEIGHT = 0.25            # fraction of the (shrunk) mean goal-diff that becomes supremacy
GD_CLAMP = 3.0           # clamp a single pair's mean goal-diff before scaling
DELTA_CAP = 0.75         # hard cap on the supremacy delta (goal units)
HOME_GD = 0.40           # goal-difference value of playing at home (de-biases non-neutral games)
CUTOFF = "2026-06-05"    # ignore anything after this (keeps the WC out of its own priors)
# The 2026 warm-up friendlies are themselves predicted by the model, and results.csv
# already contains them, so their H2H must exclude the 2026 window or a friendly would
# "see" its own result. Group/knockout matches are after CUTOFF, so they are unaffected.
PRE_WARMUP_CUTOFF = "2025-12-31"
ASOF_YEAR = 2026         # reference year for recency weighting


def _pair_key(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def build_index(df: pd.DataFrame | None = None, path=None, cutoff: str = CUTOFF) -> dict:
    """Index historical results by unordered team pair.

    Returns {frozenset({A, B}): [(year, home, away, home_goals, away_goals, neutral), ...]}.
    Rows missing a score, or dated after `cutoff`, are skipped.
    """
    if df is None:
        path = path or (config.DATA_RAW / "results.csv")
        df = pd.read_csv(path)
    d = df.copy()
    d["home_score"] = pd.to_numeric(d["home_score"], errors="coerce")
    d["away_score"] = pd.to_numeric(d["away_score"], errors="coerce")
    d = d.dropna(subset=["home_score", "away_score"])
    dates = pd.to_datetime(d["date"], errors="coerce")
    d = d[dates.notna()]
    dates = dates[dates.notna()]
    if cutoff:
        keep = dates <= pd.Timestamp(cutoff)
        d = d[keep.to_numpy()]
        dates = dates[keep]
    if "neutral" in d.columns:
        neutral = d["neutral"].astype(str).str.upper().isin(["TRUE", "1"]).to_numpy()
    else:
        neutral = [True] * len(d)

    index: dict[frozenset, list] = defaultdict(list)
    years = dates.dt.year.to_numpy()
    homes = d["home_team"].to_numpy()
    aways = d["away_team"].to_numpy()
    hg = d["home_score"].round().to_numpy(int)   # round() so a stray non-integer score is not silently truncated
    ag = d["away_score"].round().to_numpy(int)
    for i in range(len(d)):
        index[_pair_key(homes[i], aways[i])].append(
            (int(years[i]), homes[i], aways[i], int(hg[i]), int(ag[i]), bool(neutral[i]))
        )
    return dict(index)


def _oriented_gd(meeting: tuple, team_a: str, team_b: str) -> float | None:
    """Goal difference from team_a's perspective, de-biased for home advantage.

    Returns None if the meeting is not between exactly team_a and team_b, so a
    mis-keyed meeting can never be silently scored with the wrong orientation."""
    _year, home, away, hg, ag, neutral = meeting
    if home == team_a and away == team_b:
        gd = hg - ag
        if not neutral:
            gd -= HOME_GD          # team_a was at home; discount its home boost
    elif home == team_b and away == team_a:
        gd = ag - hg
        if not neutral:
            gd += HOME_GD          # opponent was at home; that hurt team_a, add it back
    else:
        return None
    return float(gd)


def h2h_delta(meetings: list, team_a: str, team_b: str, asof_year: int = ASOF_YEAR) -> float:
    """Supremacy delta (goal units) for team_a vs team_b from their past meetings.

    Positive favours team_a. Antisymmetric: swapping the teams negates the result.
    Returns 0.0 when there is no usable history.
    """
    if not meetings:
        return 0.0
    weights = []
    gds = []
    for m in meetings:
        gd = _oriented_gd(m, team_a, team_b)
        if gd is None:                 # meeting not between this exact pair; ignore
            continue
        age = max(0, asof_year - m[0])
        weights.append(0.5 ** (age / HALF_LIFE_YEARS))
        gds.append(gd)
    n_eff = sum(weights)
    if n_eff <= 0:
        return 0.0
    mean_gd = sum(w * g for w, g in zip(weights, gds)) / n_eff
    mean_gd = max(-GD_CLAMP, min(GD_CLAMP, mean_gd))
    shrink = n_eff / (n_eff + SHRINK_K)
    delta = WEIGHT * shrink * mean_gd
    return max(-DELTA_CAP, min(DELTA_CAP, delta))


def delta_map(teams, index: dict, asof_year: int = ASOF_YEAR) -> dict:
    """Precompute directed H2H deltas for every ordered pair among `teams`.

    Returns {(home_team, away_team): delta}. Only pairs with shared history are
    stored; a missing key means 0.0 (use `.get((h, a), 0.0)`).
    """
    teams = list(teams)
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            meetings = index.get(_pair_key(a, b))
            if not meetings:
                continue
            d = h2h_delta(meetings, a, b, asof_year)
            if d != 0.0:
                out[(a, b)] = d
                out[(b, a)] = -d
    return out
