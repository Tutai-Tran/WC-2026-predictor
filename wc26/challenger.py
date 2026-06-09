"""Challenger-model harness: principled model selection for the self-improving loop.

Run:  python -m wc26.challenger
Fits a CHALLENGER scoreline model on the same leak-free training window the champion
(Poisson + Dixon-Coles) uses, evaluates both on the same held-out test set, and
reports log loss / Brier / RPS with a paired-bootstrap CI on the log-loss difference.
Report only: nothing is adopted automatically. A challenger that beats the champion
with a CI excluding zero is the evidence required to switch models (PLAN decision:
"each adopted only if it wins a proper out-of-sample significance test").

Challenger v1: bivariate Poisson (Karlis & Ntzoufras). Goals are X = A + C,
Y = B + C with A~Pois(l1), B~Pois(l2), C~Pois(l3): the shared component C induces
the positive scoreline correlation that the champion only patches in the four
low-score Dixon-Coles cells.
"""

from __future__ import annotations

import json
from math import lgamma

import numpy as np
from scipy.optimize import minimize

from . import backtest as bt, config

MAX_GOALS = bt.MAX_GOALS


def _biv_poisson_probs(l1, l2, l3, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(X=x, Y=y) for X=A+C, Y=B+C. Vectorised over matches: l1/l2/l3 are arrays of
    shape (M,) (l3 may also be a scalar). Returns (M, G+1, G+1), rows normalised."""
    G = max_goals
    l1 = np.asarray(l1, float)[:, None]
    l2 = np.asarray(l2, float)[:, None]
    l3 = np.broadcast_to(np.asarray(l3, float), (l1.shape[0],))[:, None]
    ks = np.arange(G + 1)
    lfact = np.array([lgamma(k + 1) for k in range(G + 1)])
    # P(A=a) etc. via log pmf, then convolve over the shared component C
    log_p1 = -l1 + ks[None, :] * np.log(np.maximum(l1, 1e-12)) - lfact[None, :]
    log_p2 = -l2 + ks[None, :] * np.log(np.maximum(l2, 1e-12)) - lfact[None, :]
    p1, p2 = np.exp(log_p1), np.exp(log_p2)                       # (M, G+1)
    pc = np.exp(-l3 + ks[None, :] * np.log(np.maximum(l3, 1e-12)) - lfact[None, :])
    M = p1.shape[0]
    joint = np.zeros((M, G + 1, G + 1))
    for k in range(G + 1):                                        # C = k
        a = p1[:, : G + 1 - k]                                    # A = x - k
        b = p2[:, : G + 1 - k]                                    # B = y - k
        joint[:, k:, k:] += pc[:, k, None, None] * a[:, :, None] * b[:, None, :]
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    return joint


def _challenger_outcome_probs(feats, params) -> np.ndarray:
    """W/D/L probabilities under the bivariate Poisson challenger.

    params = (c, base, home_adv_elo, gamma, l3). The Elo-to-goals mapping is shared
    with the champion (same supremacy structure); l3 moves goal mass into the shared
    component, so totals stay comparable: l1 = la - l3, l2 = lb - l3 (floored)."""
    c, base, home_adv_elo, gamma, l3 = params
    la, lb = bt._lambdas(feats["elo_h"], feats["elo_a"], feats["neutral"],
                         c, base, home_adv_elo, gamma)
    # per-match shared component: capped so weak attacks keep their marginal mean
    # (a flat floor on l1/l2 would silently inflate a minnow's expected goals)
    l3_eff = np.clip(np.minimum(np.minimum(la, lb) - 0.05, l3), 0.0, 0.6)
    l1 = la - l3_eff
    l2 = lb - l3_eff
    joint = _biv_poisson_probs(l1, l2, l3_eff)
    tri = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)
    p_home = (joint * tri[None]).sum(axis=(1, 2))
    p_draw = np.trace(joint, axis1=1, axis2=2)
    p_away = np.clip(1.0 - p_home - p_draw, 0.0, 1.0)
    return np.column_stack([p_home, p_draw, p_away])


def challenger_log_loss(feats, params, weights=None) -> float:
    p = _challenger_outcome_probs(feats, params)
    nll = -np.log(np.clip(p[np.arange(len(p)), feats["outcome"]], 1e-12, 1.0))
    if weights is None:
        return float(np.mean(nll))
    return float(np.average(nll, weights=weights))


def fit_challenger(feats_train, weights=None):
    """Fit (c, base, home_adv_elo, gamma, l3) by log loss on the training window.
    Pass the SAME decay weights the champion was fitted with for a fair comparison."""
    def obj(x):
        c = min(600.0, max(50.0, x[0]))
        base = min(4.0, max(1.5, x[1]))
        home = min(150.0, max(0.0, x[2]))
        gamma = min(1.0, max(0.0, x[3]))
        l3 = min(0.6, max(0.0, x[4]))
        return challenger_log_loss(feats_train, (c, base, home, gamma, l3), weights=weights)

    res = minimize(obj, x0=[208.0, 2.2, 118.0, 0.45, 0.10], method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-5, "maxiter": 400})
    x = res.x
    return (min(600.0, max(50.0, float(x[0]))), min(4.0, max(1.5, float(x[1]))),
            min(150.0, max(0.0, float(x[2]))), min(1.0, max(0.0, float(x[3]))),
            min(0.6, max(0.0, float(x[4]))))


def paired_bootstrap_diff(feats, champion_params, challenger_params,
                          n_boot: int = 400, seed: int = 12345):
    """95% CI for (champion log loss - challenger log loss) by paired resampling.
    Positive values favour the challenger."""
    rng = np.random.default_rng(seed)
    p_champ = bt._probs_for(feats, champion_params)
    p_chal = _challenger_outcome_probs(feats, challenger_params)
    idx = np.arange(len(p_champ))
    nll_champ = -np.log(np.clip(p_champ[idx, feats["outcome"]], 1e-12, 1.0))
    nll_chal = -np.log(np.clip(p_chal[idx, feats["outcome"]], 1e-12, 1.0))
    diff = nll_champ - nll_chal
    m = len(diff)
    boots = [diff[rng.integers(0, m, m)].mean() for _ in range(n_boot)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
            round(float(diff.mean()), 4))


def run(write: bool = True, train_until: int = 2021) -> dict:
    df = bt.read_results()
    feats, _ = bt.prematch_pass(df)
    feats = bt._subset(feats, feats["year"] >= 2006)
    train = bt._subset(feats, feats["year"] < train_until)
    test = bt._subset(feats, feats["year"] >= train_until)

    champ = json.loads((config.DATA_RAW / "fitted_params.json").read_text())
    champ_tuple = (champ["c"], champ["base_goals"], champ.get("home_adv_elo", 60.0),
                   champ["rho"], champ.get("gamma", 0.0))

    chal = fit_challenger(train, weights=bt.decay_weights(train, train_until))
    lo, hi, mean = paired_bootstrap_diff(test, champ_tuple, chal)
    report = {
        "challenger": "bivariate-poisson",
        "challenger_params": {"c": round(chal[0], 2), "base_goals": round(chal[1], 3),
                              "home_adv_elo": round(chal[2], 1), "gamma": round(chal[3], 3),
                              "lambda3": round(chal[4], 4)},
        "test": {
            "n": int(len(test["outcome"])),
            "champion_log_loss": round(bt.log_loss(test, champ_tuple), 4),
            "challenger_log_loss": round(challenger_log_loss(test, chal), 4),
            "diff_mean": mean,                      # >0 favours the challenger
            "diff_ci95": [lo, hi],
        },
        "verdict": ("challenger wins (CI excludes 0)" if lo > 0
                    else ("champion wins (CI excludes 0)" if hi < 0
                          else "no significant difference — keep the champion")),
        "note": "report only; switching models requires this CI to exclude zero "
                "in the challenger's favour",
    }
    if write:
        (config.DATA_RAW / "challenger_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
