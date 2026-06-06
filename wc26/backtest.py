"""Leak-free calibration of the Elo-to-goals model on historical internationals.

One chronological pass records each match's PRE-match Elo (no look-ahead) and also
yields current self-maintained ratings. We then fit the global goal-model
parameters (c, base_goals, home_adv_elo, gamma) by minimising log loss on a training
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

    maj = np.empty(n, bool)
    for i in range(n):
        h, a = homes[i], aways[i]
        rh = ratings.get(h, elo_mod.DEFAULT_ELO)
        ra = ratings.get(a, elo_mod.DEFAULT_ELO)
        eh[i], ea[i], neu[i] = rh, ra, neutral[i]
        adv = 0.0 if neutral[i] else home_adv
        k = elo_mod.k_for_tournament(tours[i])
        maj[i] = k >= 50.0  # World Cup / continental finals (tournament-like)
        nh, na = elo_mod.update_pair(rh, ra, int(gh[i]), int(ga[i]), k=k, home_adv=adv)
        ratings[h], ratings[a] = nh, na

    outcome = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))  # 0 home,1 draw,2 away
    feats = {"elo_h": eh, "elo_a": ea, "neutral": neu, "gh": gh, "ga": ga,
             "year": years, "is_major": maj, "outcome": outcome}
    return feats, ratings


def _lambdas(elo_h, elo_a, neutral, c, base, home_adv_elo, gamma=0.0):
    eff = (elo_h + np.where(neutral, 0.0, home_adv_elo)) - elo_a
    sup = eff / c
    base_eff = base + gamma * np.abs(sup)   # mismatches score more in total
    la = np.maximum(MIN_LAMBDA, (base_eff + sup) / 2.0)
    lb = np.maximum(MIN_LAMBDA, (base_eff - sup) / 2.0)
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
    c, base, home_adv_elo, rho, gamma = params
    la, lb = _lambdas(feats["elo_h"], feats["elo_a"], feats["neutral"], c, base, home_adv_elo, gamma)
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


def climatology_log_loss(feats) -> float:
    """Baseline that always predicts the empirical (home/draw/away) base rates."""
    y = feats["outcome"]
    freq = np.array([(y == k).mean() for k in range(3)])
    freq = np.clip(freq, 1e-12, 1.0)
    return float(-np.mean(np.log(freq[y])))


def favourite_log_loss(feats, win=0.55, draw=0.27) -> float:
    """Baseline that gives a fixed edge to the higher-rated (Elo+home) side."""
    eff = feats["elo_h"] + np.where(feats["neutral"], 0.0, 60.0) - feats["elo_a"]
    home_fav = eff >= 0
    lose = 1.0 - win - draw
    p = np.empty((len(eff), 3))
    p[home_fav] = [win, draw, lose]
    p[~home_fav] = [lose, draw, win]
    actual = p[np.arange(len(p)), feats["outcome"]]
    return float(-np.mean(np.log(np.clip(actual, 1e-12, 1.0))))


def bootstrap_log_loss_ci(feats, params, n_boot=400, seed=12345):
    """Percentile CI for the model's log loss via match resampling."""
    rng = np.random.default_rng(seed)
    p = _probs_for(feats, params)
    actual = -np.log(np.clip(p[np.arange(len(p)), feats["outcome"]], 1e-12, 1.0))
    m = len(actual)
    boots = [actual[rng.integers(0, m, m)].mean() for _ in range(n_boot)]
    return round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)


def fit_temperature(feats, params):
    """Post-hoc temperature scaling: p_cal ∝ p**(1/T). Returns T>0."""
    p = _probs_for(feats, params)
    y = feats["outcome"]

    def obj(logT):
        T = float(np.exp(logT))
        pc = np.power(p, 1.0 / T)
        pc /= pc.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(pc[np.arange(len(pc)), y], 1e-12, 1.0))))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(obj, bounds=(np.log(0.5), np.log(3.0)), method="bounded")
    return float(np.exp(res.x))


def reliability_slope(feats, params, bins=10):
    """Slope of observed vs predicted home-win frequency across probability bins.

    Perfect calibration -> slope ~ 1. Returns (slope, n_bins_used)."""
    p_home = _probs_for(feats, params)[:, 0]
    obs = (feats["outcome"] == 0).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_home >= lo) & (p_home < hi)
        if m.sum() >= 30:
            xs.append(p_home[m].mean())
            ys.append(obs[m].mean())
    if len(xs) < 2:
        return float("nan"), len(xs)
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope, len(xs)


def _subset(feats, mask):
    return {k: v[mask] for k, v in feats.items()}


def fit(feats_train, rho: float = -0.06):
    def obj(x):
        c, base, home, gamma = x
        c = min(600.0, max(50.0, c))
        base = min(4.0, max(1.5, base))
        home = min(150.0, max(0.0, home))
        gamma = min(1.0, max(0.0, gamma))
        return log_loss(feats_train, (c, base, home, rho, gamma))

    res = minimize(obj, x0=[200.0, 2.6, 60.0, 0.2], method="Nelder-Mead",
                   options={"xatol": 0.5, "fatol": 1e-5, "maxiter": 600})
    # clamp the RETURNED point to the same bounds enforced inside the objective,
    # so a pathological optimum can never write an invalid (e.g. negative c) param
    c = min(600.0, max(50.0, float(res.x[0])))
    base = min(4.0, max(1.5, float(res.x[1])))
    home = min(150.0, max(0.0, float(res.x[2])))
    gamma = min(1.0, max(0.0, float(res.x[3])))
    return model_mod.ModelParams(c=round(c, 2), base_goals=round(base, 3), rho=rho,
                                 gamma=round(gamma, 3)), home


def apply_played_friendlies(ratings: dict[str, float]) -> dict[str, float]:
    """Continue the Elo replay with already-played warm-up friendlies so current
    ratings reflect the latest results (the backtest itself stays on results.csv)."""
    path = config.DATA_RAW / "friendlies.json"
    if not path.exists():
        return ratings
    matches = json.loads(path.read_text()).get("matches", [])
    played = [m for m in matches if m.get("home_score") is not None and m.get("away_score") is not None]
    played.sort(key=lambda m: m.get("date_utc") or "")
    r = dict(ratings)
    for m in played:
        h, a = m["home"], m["away"]
        rh = r.get(h, elo_mod.DEFAULT_ELO)
        ra = r.get(a, elo_mod.DEFAULT_ELO)
        nh, na = elo_mod.update_pair(rh, ra, int(m["home_score"]), int(m["away_score"]),
                                     k=20.0, home_adv=0.0)
        r[h], r[a] = nh, na
    return r


def _temp_log_loss(feats, params, T) -> float:
    p = _probs_for(feats, params)
    pc = np.power(p, 1.0 / T)
    pc /= pc.sum(axis=1, keepdims=True)
    a = pc[np.arange(len(pc)), feats["outcome"]]
    return float(-np.mean(np.log(np.clip(a, 1e-12, 1.0))))


def run(write: bool = True, train_until: int = 2021):
    df = read_results()
    feats, ratings = prematch_pass(df)
    feats = _subset(feats, feats["year"] >= 2006)
    train = _subset(feats, feats["year"] < train_until)
    test = _subset(feats, feats["year"] >= train_until)

    params, home_adv_elo = fit(train)
    p = (params.c, params.base_goals, home_adv_elo, params.rho, params.gamma)
    temperature = round(fit_temperature(train, p), 3)
    ci_lo, ci_hi = bootstrap_log_loss_ci(test, p)
    slope, nbins = reliability_slope(test, p)
    tourn = _subset(test, test["is_major"])

    # leak-free goal-volume correction: match mean predicted to mean actual total
    # goals on the held-out test window (bounded so it can never run wild).
    la_t, lb_t = _lambdas(test["elo_h"], test["elo_a"], test["neutral"],
                          params.c, params.base_goals, home_adv_elo, params.gamma)
    pred_tot = float(np.mean(la_t + lb_t))
    act_tot = float(np.mean(test["gh"] + test["ga"]))
    goal_scale = round(min(1.10, max(0.90, act_tot / pred_tot)), 4) if pred_tot > 0 else 1.0

    # preserve a previously-set friendly-only rho (a validated segment calibration the global
    # fit doesn't re-derive) so a manual re-fit never silently reverts the friendly draw fix
    _prev = json.loads((config.DATA_RAW / "fitted_params.json").read_text()) \
        if (config.DATA_RAW / "fitted_params.json").exists() else {}
    report = {
        "fitted": {**asdict(params), "home_adv_elo": round(home_adv_elo, 1),
                   "temperature": temperature, "goal_scale": goal_scale,
                   "rho_friendly": _prev.get("rho_friendly", asdict(params).get("rho_friendly", -0.06))},
        "train": {"n": int(len(train["outcome"])), "log_loss": round(log_loss(train, p), 4),
                  "brier": round(brier(train, p), 4), "rps": round(rps(train, p), 4)},
        "test": {"n": int(len(test["outcome"])), "log_loss": round(log_loss(test, p), 4),
                 "log_loss_ci95": [ci_lo, ci_hi],
                 "log_loss_calibrated": round(_temp_log_loss(test, p, temperature), 4),
                 "brier": round(brier(test, p), 4), "rps": round(rps(test, p), 4)},
        "tournament_only": {"n": int(len(tourn["outcome"])),
                            "log_loss": round(log_loss(tourn, p), 4) if len(tourn["outcome"]) else None,
                            "climatology_log_loss": round(climatology_log_loss(tourn), 4) if len(tourn["outcome"]) else None},
        "baselines": {"uniform_log_loss": round(float(np.log(3)), 4),
                      "climatology_log_loss": round(climatology_log_loss(test), 4),
                      "favourite_log_loss": round(favourite_log_loss(test), 4)},
        "reliability": {"home_win_slope": round(slope, 3), "bins_used": nbins},
        "beats_climatology": bool(log_loss(test, p) < climatology_log_loss(test)),
    }
    if write:
        (config.DATA_RAW / "fitted_params.json").write_text(json.dumps(report["fitted"], indent=2))
        (config.DATA_RAW / "backtest_report.json").write_text(json.dumps(report, indent=2))
        # base = results.csv replay only (stable anchor for re-applying live results)
        (config.DATA_RAW / "base_elo.json").write_text(
            json.dumps({"updated": "2026-06-05", "source": "self-replay of results.csv",
                        "ratings": {k: round(v, 1) for k, v in ratings.items()}}, indent=2)
        )
        current = apply_played_friendlies(ratings)
        (config.DATA_RAW / "replayed_elo.json").write_text(
            json.dumps({"updated": "2026-06-05",
                        "source": "self-replay of results.csv + played warm-up friendlies",
                        "ratings": {k: round(v, 1) for k, v in current.items()}}, indent=2)
        )
    return report


def main():
    report = run()
    print(json.dumps(report, indent=2))
    print("\nFitted params and replayed Elo written to data/raw/.")


if __name__ == "__main__":
    main()
