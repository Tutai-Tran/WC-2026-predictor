"""Leak-free calibration of the Elo-to-goals model on historical internationals.

One chronological pass records each match's PRE-match Elo (no look-ahead) and also
yields current self-maintained ratings. We then fit the global goal-model
parameters (c, base_goals, home_adv_elo) by minimising log loss on a training
window and report Brier / log loss / RPS on a held-out later window. Fitted
parameters and current ratings are written to data/raw for the forecast to use.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config, elo as elo_mod, model as model_mod

MIN_LAMBDA = 0.15
MAX_GOALS = 10


def read_results(path=None, exclude_future_after: str = "2026-06-04") -> pd.DataFrame:
    path = path or (config.DATA_RAW / "results.csv")
    df = pd.read_csv(path)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= pd.Timestamp(exclude_future_after)]
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    else:
        df["neutral"] = False
    return df.sort_values("date").reset_index(drop=True)


def prematch_pass(df: pd.DataFrame, home_adv: float = 60.0):
    """Chronological replay. Returns (features dict of arrays, current ratings)."""
    ratings: dict[str, float] = {}
    n = len(df)
    eh = np.empty(n); ea = np.empty(n); neu = np.empty(n, bool)
    gh = df["home_score"].to_numpy(int); ga = df["away_score"].to_numpy(int)
    years = df["date"].dt.year.to_numpy()
    homes = df["home_team"].to_numpy(); aways = df["away_team"].to_numpy()
    tours = df["tournament"].to_numpy() if "tournament" in df else np.array([""] * n)
    neutral = df["neutral"].to_numpy(bool)

    for i in range(n):
        h, a = homes[i], aways[i]
        rh = ratings.get(h, elo_mod.DEFAULT_ELO)
        ra = ratings.get(a, elo_mod.DEFAULT_ELO)
        eh[i], ea[i], neu[i] = rh, ra, neutral[i]
        adv = 0.0 if neutral[i] else home_adv
        k = elo_mod.k_for_tournament(tours[i])
        nh, na = elo_mod.update_pair(rh, ra, int(gh[i]), int(ga[i]), k=k, home_adv=adv)
        ratings[h], ratings[a] = nh, na

    outcome = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))  # 0 home,1 draw,2 away
    feats = {"elo_h": eh, "elo_a": ea, "neutral": neu, "gh": gh, "ga": ga,
             "year": years, "outcome": outcome}
    return feats, ratings


def _lambdas(elo_h, elo_a, neutral, c, base, home_adv_elo):
    eff = (elo_h + np.where(neutral, 0.0, home_adv_elo)) - elo_a
    sup = eff / c
    la = np.maximum(MIN_LAMBDA, (base + sup) / 2.0)
    lb = np.maximum(MIN_LAMBDA, (base - sup) / 2.0)
    return la, lb


def _outcome_probs(la, lb, rho=-0.06):
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals[None, :], la[:, None])      # (M,G)
    pa = poisson.pmf(goals[None, :], lb[:, None])
    joint = ph[:, :, None] * pa[:, None, :]            # (M,G,G)
    if rho:
        joint[:, 0, 0] *= 1.0 - la * lb * rho
        joint[:, 0, 1] *= 1.0 + la * rho
        joint[:, 1, 0] *= 1.0 + lb * rho
        joint[:, 1, 1] *= 1.0 - rho
    joint /= joint.sum(axis=(1, 2), keepdims=True)
    tri = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)  # home>away
    p_home = (joint * tri[None]).sum(axis=(1, 2))
    p_draw = np.trace(joint, axis1=1, axis2=2)
    p_away = 1.0 - p_home - p_draw
    return np.column_stack([p_home, p_draw, p_away])


def _probs_for(feats, params):
    c, base, home_adv_elo, rho = params
    la, lb = _lambdas(feats["elo_h"], feats["elo_a"], feats["neutral"], c, base, home_adv_elo)
    return _outcome_probs(la, lb, rho)


def log_loss(feats, params) -> float:
    p = _probs_for(feats, params)
    actual = p[np.arange(len(p)), feats["outcome"]]
    return float(-np.mean(np.log(np.clip(actual, 1e-12, 1.0))))


def brier(feats, params) -> float:
    p = _probs_for(feats, params)
    y = np.zeros_like(p)
    y[np.arange(len(p)), feats["outcome"]] = 1.0
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def rps(feats, params) -> float:
    p = _probs_for(feats, params)
    y = np.zeros_like(p)
    y[np.arange(len(p)), feats["outcome"]] = 1.0
    cp = np.cumsum(p, axis=1)[:, :2]
    cy = np.cumsum(y, axis=1)[:, :2]
    # ranked probability score: normalise by (number of outcomes - 1) = 2
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1)) / 2.0)


def _subset(feats, mask):
    return {k: v[mask] for k, v in feats.items()}


def fit(feats_train, rho: float = -0.06):
    def obj(x):
        c, base, home = x
        c = min(600.0, max(50.0, c))
        base = min(4.0, max(1.5, base))
        home = min(150.0, max(0.0, home))
        return log_loss(feats_train, (c, base, home, rho))

    res = minimize(obj, x0=[200.0, 2.6, 60.0], method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-5, "maxiter": 400})
    # clamp the RETURNED point to the same bounds enforced inside the objective,
    # so a pathological optimum can never write an invalid (e.g. negative c) param
    c = min(600.0, max(50.0, float(res.x[0])))
    base = min(4.0, max(1.5, float(res.x[1])))
    home = min(150.0, max(0.0, float(res.x[2])))
    return model_mod.ModelParams(c=round(c, 2), base_goals=round(base, 3), rho=rho), home


def run(write: bool = True, train_until: int = 2021):
    df = read_results()
    feats, ratings = prematch_pass(df)
    recent = feats["year"] >= 2006
    feats = _subset(feats, recent)
    train = _subset(feats, feats["year"] < train_until)
    test = _subset(feats, feats["year"] >= train_until)

    params, home_adv_elo = fit(train)
    p = (params.c, params.base_goals, home_adv_elo, params.rho)
    report = {
        "fitted": {**asdict(params), "home_adv_elo": round(home_adv_elo, 1)},
        "train": {"n": int(len(train["outcome"])),
                  "log_loss": round(log_loss(train, p), 4),
                  "brier": round(brier(train, p), 4), "rps": round(rps(train, p), 4)},
        "test": {"n": int(len(test["outcome"])),
                 "log_loss": round(log_loss(test, p), 4),
                 "brier": round(brier(test, p), 4), "rps": round(rps(test, p), 4)},
        "baseline_uniform_logloss": round(float(np.log(3)), 4),
    }
    if write:
        (config.DATA_RAW / "fitted_params.json").write_text(json.dumps(report["fitted"], indent=2))
        (config.DATA_RAW / "replayed_elo.json").write_text(
            json.dumps({"updated": "2026-06-04", "source": "self-replay of results.csv",
                        "ratings": {k: round(v, 1) for k, v in ratings.items()}}, indent=2)
        )
    return report


def main():
    report = run()
    print(json.dumps(report, indent=2))
    print("\nFitted params and replayed Elo written to data/raw/.")


if __name__ == "__main__":
    main()
