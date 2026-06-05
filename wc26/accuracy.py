"""Backward-looking accuracy of our played-match predictions, broken down by
metric (outcome, favourite, over/under, both-teams-to-score, goal difference,
exact score). Pure functions over the forecast payload, used by the dashboard
scorecard and the tests.
"""

from __future__ import annotations

import math

from . import model as model_mod

# display order + labels
METRICS = [
    ("outcome", "Win / Draw / Loss"),
    ("favourite", "Favourite wins"),
    ("ou25", "Over / Under 2.5"),
    ("btts", "Both teams score"),
    ("gd", "Goal difference"),
    ("exact", "Exact score"),
]


def _poisson_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


def _split(score) -> tuple[int, int] | None:
    try:
        h, a = str(score).split("-")
        return int(h), int(a)
    except (ValueError, AttributeError):
        return None


def metrics(played: list[dict]) -> list[dict]:
    """Per-metric correct/total over played matches.

    Each item should carry p_home/p_draw/p_away, top_scoreline, result, and
    (for over/under + BTTS) lambda_home/lambda_away. Each metric counts only the
    matches for which it is computable, so a missing field never inflates a rate.
    """
    correct = {k: 0 for k, _ in METRICS}
    total = {k: 0 for k, _ in METRICS}

    for m in played:
        actual = _split(m.get("result"))
        if actual is None:
            continue
        ah, aa = actual
        ph, pd, pa = m.get("p_home"), m.get("p_draw"), m.get("p_away")

        if ph is not None and pd is not None and pa is not None:
            total["outcome"] += 1
            if model_mod.outcome_label(ph, pd, pa) == model_mod.result_label(ah, aa):
                correct["outcome"] += 1
            total["favourite"] += 1
            fav_home = ph >= pa
            if (fav_home and ah > aa) or (not fav_home and aa > ah):
                correct["favourite"] += 1

        pred = _split(m.get("top_scoreline"))
        if pred is not None:
            sh, sa = pred
            total["exact"] += 1
            if sh == ah and sa == aa:
                correct["exact"] += 1
            total["gd"] += 1
            if (sh - sa) == (ah - aa):
                correct["gd"] += 1

        lh, la = m.get("lambda_home"), m.get("lambda_away")
        if lh is not None and la is not None:
            total["ou25"] += 1
            p_over = 1.0 - _poisson_cdf(2, lh + la)
            if (p_over > 0.5) == ((ah + aa) > 2.5):
                correct["ou25"] += 1
            total["btts"] += 1
            p_btts = (1.0 - math.exp(-lh)) * (1.0 - math.exp(-la))
            if (p_btts > 0.5) == (ah >= 1 and aa >= 1):
                correct["btts"] += 1

    return [{"key": k, "label": label, "correct": correct[k], "total": total[k],
             "pct": (correct[k] / total[k] * 100.0) if total[k] else None}
            for k, label in METRICS]


def played_from_payload(payload: dict) -> list[dict]:
    """Collect finished matches (friendlies + group) from a forecast payload."""
    out = []
    for f in payload.get("friendlies", []):
        if f.get("played") and f.get("result"):
            out.append(f)
    for m in payload.get("matches", []):
        if m.get("result"):
            out.append(m)
    return out
