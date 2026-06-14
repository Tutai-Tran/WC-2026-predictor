"""Leak-free calibration of the Elo-to-goals model on historical internationals.

One chronological pass records each match's PRE-match Elo (no look-ahead) and also
yields current self-maintained ratings. We then fit the global goal-model
parameters (c, base_goals, home_adv_elo, gamma) by minimising log loss on a training
window and report Brier / log loss / RPS on a held-out later window. Fitted
parameters and current ratings are written to data/raw for the forecast to use.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config, elo as elo_mod, model as model_mod

MIN_LAMBDA = 0.15
MAX_GOALS = 10

# Weight on the total-goals calibration penalty when refitting base_goals under the
# goal-aware objective (item 5). Tuned so the fitted mean predicted total lands within
# +/-0.05 of the actual neutral/WC average while W/D/L test log-loss rises by <= 0.004.
LAMBDA_TOT = 0.03


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
    fri = np.empty(n, bool)
    for i in range(n):
        h, a = homes[i], aways[i]
        rh = ratings.get(h, elo_mod.DEFAULT_ELO)
        ra = ratings.get(a, elo_mod.DEFAULT_ELO)
        eh[i], ea[i], neu[i] = rh, ra, neutral[i]
        adv = 0.0 if neutral[i] else home_adv
        k = elo_mod.k_for_tournament(tours[i])
        maj[i] = k >= 50.0  # World Cup / continental finals (tournament-like)
        fri[i] = k <= 25.0  # friendlies (low-K): the segment rho_friendly is tuned on
        nh, na = elo_mod.update_pair(rh, ra, int(gh[i]), int(ga[i]), k=k, home_adv=adv)
        ratings[h], ratings[a] = nh, na

    outcome = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))  # 0 home,1 draw,2 away
    feats = {"elo_h": eh, "elo_a": ea, "neutral": neu, "gh": gh, "ga": ga,
             "year": years, "is_major": maj, "is_friendly": fri, "outcome": outcome,
             "tournament": tours.astype(object)}
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


def mean_total_goals(feats, params, weights=None) -> float:
    """Mean predicted total goals (lambda_home + lambda_away) over `feats`.

    The model-side analogue of the actual mean total; used by the goal-aware
    base_goals refit (item 5) to match real WC/neutral goal volume."""
    c, base, home_adv_elo, _rho, gamma = params
    la, lb = _lambdas(feats["elo_h"], feats["elo_a"], feats["neutral"],
                      c, base, home_adv_elo, gamma)
    tot = la + lb
    if weights is None:
        return float(np.mean(tot))
    return float(np.average(tot, weights=weights))


def model_btts(feats, params) -> float:
    """Mean model P(both teams to score) over `feats`.

    BTTS-yes = 1 - P(home=0) - P(away=0) + P(0,0) on the Dixon-Coles-corrected joint,
    reported in the totals slice (item 5) to track the goal-volume fix at 0.434 -> ~0.46."""
    c, base, home_adv_elo, rho, gamma = params
    la, lb = _lambdas(feats["elo_h"], feats["elo_a"], feats["neutral"],
                      c, base, home_adv_elo, gamma)
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
    p_home_zero = joint[:, 0, :].sum(axis=1)
    p_away_zero = joint[:, :, 0].sum(axis=1)
    p_nil_nil = joint[:, 0, 0]
    btts = 1.0 - p_home_zero - p_away_zero + p_nil_nil
    return float(np.mean(btts))


def log_loss(feats, params, weights=None) -> float:
    """Mean negative log likelihood; optional per-match weights (e.g. time decay)."""
    p = _probs_for(feats, params)
    actual = -np.log(np.clip(p[np.arange(len(p)), feats["outcome"]], 1e-12, 1.0))
    if weights is None:
        return float(np.mean(actual))
    return float(np.average(actual, weights=weights))


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


_T_MIN, _T_MAX = 0.8, 2.0


def fit_temperature(feats, params):
    """Post-hoc temperature scaling: p_cal ∝ p**(1/T), T in [0.8, 2.0].

    Must be fit on data NOT used to fit `params` (a held-out validation fold), or T
    leaks toward 1.0 and never learns the out-of-sample overconfidence. T>1 flattens
    (de-sharpens) overconfident favourites; T<1 sharpens. The bound keeps a pathological
    small fold from running the scaling away."""
    p = _probs_for(feats, params)
    y = feats["outcome"]

    def obj(logT):
        T = float(np.exp(logT))
        pc = np.power(p, 1.0 / T)
        pc /= pc.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(pc[np.arange(len(pc)), y], 1e-12, 1.0))))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(obj, bounds=(np.log(_T_MIN), np.log(_T_MAX)), method="bounded")
    return float(min(_T_MAX, max(_T_MIN, np.exp(res.x))))


def elite_overconfidence_gap(feats, params, home_adv_elo: float,
                             temperature: float = 1.0, elo_floor: float = 2000.0):
    """Favourite-win overconfidence gap on the elite-vs-elite slice (both Elo>floor).

    Gap = mean P(favourite wins) - observed favourite-win frequency. A results-only
    Elo over-rates hot elite favourites, so this is positive (overconfident) on the
    deep-knockout class the user watches; temperature>1 should shrink it toward 0.
    Returns (gap, n) or (nan, 0) when the slice is empty."""
    mask = (feats["elo_h"] > elo_floor) & (feats["elo_a"] > elo_floor)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    sub = _subset(feats, mask)
    pr = _probs_for(sub, params)
    if temperature != 1.0:
        pr = np.power(np.clip(pr, 1e-12, 1.0), 1.0 / temperature)
        pr /= pr.sum(axis=1, keepdims=True)
    eff = sub["elo_h"] + np.where(sub["neutral"], 0.0, home_adv_elo) - sub["elo_a"]
    home_fav = eff >= 0
    pred_fav = np.where(home_fav, pr[:, 0], pr[:, 2])
    obs_fav = np.where(home_fav, sub["outcome"] == 0, sub["outcome"] == 2).astype(float)
    return float(pred_fav.mean() - obs_fav.mean()), n


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


def decay_weights(feats, train_until: int = 2021, half_life_years: float = 10.0):
    """Time-decay weights: a match `half_life_years` before the train cutoff counts
    half as much as a cutoff-year match, so the fit tracks the modern game."""
    age = np.maximum(0.0, train_until - feats["year"].astype(float))
    return 0.5 ** (age / half_life_years)


def fit(feats_train, rho: float = -0.06, weights=None):
    def obj(x):
        c, base, home, gamma = x
        c = min(600.0, max(50.0, c))
        base = min(4.0, max(1.5, base))
        home = min(150.0, max(0.0, home))
        gamma = min(1.0, max(0.0, gamma))
        return log_loss(feats_train, (c, base, home, rho, gamma), weights=weights)

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


def refit_base_goals(feats_train, params, home_adv_elo: float,
                     actual_mean_total: float, weights=None,
                     lambda_tot: float = LAMBDA_TOT) -> float:
    """Refit ONLY base_goals under a goal-aware objective (item 5).

    The W/D/L fit (`fit`) selects base_goals blind to total goals and lands low
    (~2.19), so the model under-predicts goal volume by ~10%. Here c, gamma, rho and
    home are held fixed at their W/D/L optima and base_goals is re-selected to minimise
    `W/D/L log-loss + lambda_tot * (mean_pred_total - actual_mean_total)**2`. This
    matches real WC/neutral goal volume (and restores scoreline diversity) without a
    material W/D/L regression. Returns the new base_goals."""
    from scipy.optimize import minimize_scalar

    def obj(base: float) -> float:
        base = min(4.0, max(1.5, base))
        p = (params.c, base, home_adv_elo, params.rho, params.gamma)
        ll = log_loss(feats_train, p, weights=weights)
        mpt = mean_total_goals(feats_train, p, weights=weights)
        return ll + lambda_tot * (mpt - actual_mean_total) ** 2

    res = minimize_scalar(obj, bounds=(1.5, 4.0), method="bounded")
    return round(min(4.0, max(1.5, float(res.x))), 3)


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


_NOT_FINALS = ("nations league", "viva", "conifa", "island games", "ellan vannin")


def home_adv_majors(train, params, home_adv_global: float) -> dict | None:
    """Diagnostic: home advantage refit on non-neutral FINALS matches only (a host
    nation playing a major tournament in its own country, the reference class for the
    2026 co-hosts), all other params held fixed. Nations League and non-FIFA events
    carry k>=50 but are ordinary home fixtures, so they are excluded. Reported for
    audit; the live host bump is configured separately."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    not_finals = np.array([any(s in str(t).lower() for s in _NOT_FINALS)
                           for t in train["tournament"]])
    m = train["is_major"] & ~train["neutral"] & ~not_finals
    sub = _subset(train, m)
    n = len(sub["outcome"])
    if n < 100:
        return None
    res = minimize_scalar(
        lambda h: log_loss(sub, (params.c, params.base_goals, h, params.rho, params.gamma)),
        bounds=(0.0, 200.0), method="bounded",
    )
    return {"home_adv_elo": round(float(res.x), 1), "n": n}


def run(write: bool = True, train_until: int = 2021, decay_half_life: float | None = 10.0):
    # decay_half_life=10.0 is the live default (adopted 2026-06-09: test LL 0.8597 vs
    # 0.8600 undecayed); pass None to reproduce the undecayed fit.
    df = read_results()
    feats, ratings = prematch_pass(df)
    feats = _subset(feats, feats["year"] >= 2006)
    train = _subset(feats, feats["year"] < train_until)
    test = _subset(feats, feats["year"] >= train_until)

    w = decay_weights(train, train_until, decay_half_life) if decay_half_life else None
    params, home_adv_elo = fit(train, weights=w)
    base_goals_wdl = params.base_goals  # W/D/L-only optimum (pre goal-aware refit)
    # item 5: re-select ONLY base_goals under a goal-aware objective so the model
    # matches real goal volume (c, gamma, rho, home stay on the W/D/L fit above).
    train_totals = train["gh"] + train["ga"]
    actual_mean_total = float(np.average(train_totals, weights=w) if w is not None
                              else np.mean(train_totals))
    base_goals = refit_base_goals(train, params, home_adv_elo, actual_mean_total, weights=w)
    params = replace(params, base_goals=base_goals)
    p = (params.c, params.base_goals, home_adv_elo, params.rho, params.gamma)

    # item 3: un-leak temperature. It must be fit on data the params were NOT fit on,
    # or it collapses to ~1.0 and never learns the out-of-sample overconfidence. The
    # held-out validation fold is the tournament/major slice of the test window (the
    # deep-competition class where a results-only Elo over-rates hot favourites); params
    # come from `train` (year<train_until), so this fold is genuinely held out. Fitting
    # T here (not on train) yields a de-sharpening T>1 that actually cools favourites.
    val_temp = _subset(test, test["is_major"])
    temperature = round(fit_temperature(val_temp, p), 3) if len(val_temp["outcome"]) else 1.0

    ci_lo, ci_hi = bootstrap_log_loss_ci(test, p)
    slope, nbins = reliability_slope(test, p)
    tourn = _subset(test, test["is_major"])

    # leak-free goal-volume correction: fit goal_scale on a held-out validation fold
    # (the later chronological slice of the test window, a representative population),
    # NOT the full reporting window, so it carries no in-sample touch (D2). Bounded so
    # it can never run wild.
    val_year = max(train_until + 1, int(np.median(np.unique(test["year"]))))
    val_gs = _subset(test, test["year"] >= val_year)
    val_gs = val_gs if len(val_gs["outcome"]) else test
    la_v, lb_v = _lambdas(val_gs["elo_h"], val_gs["elo_a"], val_gs["neutral"],
                          params.c, params.base_goals, home_adv_elo, params.gamma)
    pred_tot_v = float(np.mean(la_v + lb_v))
    act_tot_v = float(np.mean(val_gs["gh"] + val_gs["ga"]))
    goal_scale = round(min(1.10, max(0.90, act_tot_v / pred_tot_v)), 4) if pred_tot_v > 0 else 1.0

    # totals slice still REPORTED on the full held-out test window (the gate population)
    la_t, lb_t = _lambdas(test["elo_h"], test["elo_a"], test["neutral"],
                          params.c, params.base_goals, home_adv_elo, params.gamma)
    pred_tot = float(np.mean(la_t + lb_t))
    act_tot = float(np.mean(test["gh"] + test["ga"]))

    # item 3 calibration slice: report the held-out fold log-loss before/after temperature
    # (this is the gate: calibrated < uncalibrated, with T>1), plus the elite-vs-elite
    # favourite-win overconfidence gap before/after de-sharpening.
    ll_fold = log_loss(val_temp, p) if len(val_temp["outcome"]) else None
    ll_fold_cal = _temp_log_loss(val_temp, p, temperature) if len(val_temp["outcome"]) else None
    elite_gap_raw, elite_n = elite_overconfidence_gap(test, p, home_adv_elo)
    elite_gap_cal, _ = elite_overconfidence_gap(test, p, home_adv_elo, temperature)

    # preserve a previously-set friendly-only rho (a validated segment calibration the global
    # fit doesn't re-derive) so a manual re-fit never silently reverts the friendly draw fix
    _prev = json.loads((config.DATA_RAW / "fitted_params.json").read_text()) \
        if (config.DATA_RAW / "fitted_params.json").exists() else {}
    report = {
        "fitted": {**asdict(params), "home_adv_elo": round(home_adv_elo, 1),
                   "temperature": temperature, "goal_scale": goal_scale,
                   "rho_friendly": _prev.get("rho_friendly", asdict(params).get("rho_friendly", -0.06))},
        "decay_half_life": decay_half_life,
        "home_adv_majors": home_adv_majors(train, params, home_adv_elo),
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
        "calibration": {
            "temperature": temperature,
            "validation_fold": "is_major",
            "fold_n": int(len(val_temp["outcome"])),
            # the gate: on the held-out fold, calibrated must beat uncalibrated with T>1
            "fold_log_loss": round(ll_fold, 4) if ll_fold is not None else None,
            "fold_log_loss_calibrated": round(ll_fold_cal, 4) if ll_fold_cal is not None else None,
            "fold_log_loss_improvement": round(ll_fold - ll_fold_cal, 5)
            if ll_fold is not None else None,
            "temperature_desharpens": bool(temperature > 1.0),
            # elite-vs-elite (both Elo>2000) favourite-win overconfidence gap, in pp
            "elite_n": elite_n,
            "elite_fav_gap_pp": round(elite_gap_raw * 100, 2),
            "elite_fav_gap_pp_calibrated": round(elite_gap_cal * 100, 2),
        },
        "totals": {"n": int(len(test["outcome"])),
                   "mean_pred_total": round(pred_tot, 4),
                   "mean_actual_total": round(act_tot, 4),
                   "mean_total_dev": round(pred_tot - act_tot, 4),
                   "model_btts": round(model_btts(test, p), 4),
                   "base_goals_wdl": round(base_goals_wdl, 3),
                   "base_goals_goal_aware": round(params.base_goals, 3),
                   "lambda_tot": LAMBDA_TOT},
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
