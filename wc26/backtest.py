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
# matchup-conditional de-sharpen (item A): t_elite is bounded so a small/pathological
# sub-fold cannot run the elite cooling away. 0 is a no-op; positive de-sharpens.
_TE_MIN, _TE_MAX = 0.0, 1.5


def _eff_temp_vec(feats, params, home_adv_elo: float,
                  temperature: float, t_elite: float) -> np.ndarray:
    """Per-match effective temperature T_eff = temperature + t_elite*g(matchup).

    Vectorised mirror of `model.effective_temperature`: g rises only for elite-vs-elite
    CLOSE games (both Elo high, |supremacy| small) and is ~0 elsewhere, so T_eff equals
    the global temperature for every ordinary match."""
    n = len(feats["outcome"])
    if t_elite == 0.0:
        return np.full(n, temperature)
    eff = feats["elo_h"] + np.where(feats["neutral"], 0.0, home_adv_elo) - feats["elo_a"]
    sup = eff / params[0]                              # supremacy in goal units (eff/c)
    min_elo = np.minimum(feats["elo_h"], feats["elo_a"])
    elite = 1.0 / (1.0 + np.exp(-(min_elo - model_mod._ELITE_ELO_MID) / model_mod._ELITE_ELO_WIDTH))
    close = 1.0 / (1.0 + np.exp(-(model_mod._ELITE_SUP_MID - np.abs(sup)) / model_mod._ELITE_SUP_WIDTH))
    return temperature + t_elite * (elite * close)


def _eff_temp_log_loss(feats, params, home_adv_elo: float,
                       temperature: float, t_elite: float) -> float:
    """Log-loss after applying the per-match matchup-conditional temperature T_eff."""
    p = _probs_for(feats, params)
    t = _eff_temp_vec(feats, params, home_adv_elo, temperature, t_elite)[:, None]
    pc = np.power(np.clip(p, 1e-12, 1.0), 1.0 / t)
    pc /= pc.sum(axis=1, keepdims=True)
    a = pc[np.arange(len(pc)), feats["outcome"]]
    return float(-np.mean(np.log(np.clip(a, 1e-12, 1.0))))


def fit_t_elite(feats, params, home_adv_elo: float, temperature: float):
    """Fit the elite-vs-elite de-sharpen height `t_elite` on `feats` (a held-out sub-fold).

    Holds the global `temperature` fixed and adds matchup-conditional de-sharpening that is
    non-zero ONLY for elite-vs-elite close games. Minimises log-loss over the WHOLE sub-fold
    (so it cannot help itself by hurting non-elite matches: g~0 there leaves them at the base
    temperature). Bounded to [0, 1.5]. Must be fit on data the params were NOT fit on, and
    scored on a further held-out sub-fold (the gate), so the elite-gap reduction is genuine."""
    from scipy.optimize import minimize_scalar

    def obj(t_elite: float) -> float:
        t_elite = min(_TE_MAX, max(_TE_MIN, t_elite))
        return _eff_temp_log_loss(feats, params, home_adv_elo, temperature, t_elite)

    res = minimize_scalar(obj, bounds=(_TE_MIN, _TE_MAX), method="bounded")
    return float(min(_TE_MAX, max(_TE_MIN, res.x)))


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
                             temperature: float = 1.0, elo_floor: float = 2000.0,
                             t_elite: float = 0.0):
    """Favourite-win overconfidence gap on the elite-vs-elite slice (both Elo>floor).

    Gap = mean P(favourite wins) - observed favourite-win frequency. A results-only
    Elo over-rates hot elite favourites, so this is positive (overconfident) on the
    deep-knockout class the user watches; temperature>1 should shrink it toward 0.
    A non-zero `t_elite` adds the matchup-conditional de-sharpen (item A), which targets
    exactly this slice. Returns (gap, n) or (nan, 0) when the slice is empty."""
    mask = (feats["elo_h"] > elo_floor) & (feats["elo_a"] > elo_floor)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    sub = _subset(feats, mask)
    pr = _probs_for(sub, params)
    t = _eff_temp_vec(sub, params, home_adv_elo, temperature, t_elite)
    if not np.allclose(t, 1.0):
        pr = np.power(np.clip(pr, 1e-12, 1.0), 1.0 / t[:, None])
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


# Finals tournaments with a group-stage-then-knockout structure: the reference class
# for WC 2026. (Nations League / qualifiers are round-robin and excluded; their goal
# volume is already carried by the global base_goals.)
FINALS_TOURNAMENTS = (
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "AFC Asian Cup",
    "African Cup of Nations",
    "Gold Cup",
    "Confederations Cup",
)

# Knockout-match count per finals edition, keyed by the edition's total match count
# (single-elimination bracket incl. the third-place play-off). Lets us split each
# edition into group (earlier) vs knockout (later) matches by date without a stage
# label in results.csv. Editions with an unlisted size fall back to "last 25%".
_KO_COUNT_BY_EDITION_SIZE = {
    16: 8, 26: 11, 31: 15, 32: 16, 38: 14, 51: 15, 52: 16, 64: 16, 104: 32,
}


def _ko_count(n_matches: int) -> int:
    """Knockout-match count for a finals edition of `n_matches` total matches."""
    return _KO_COUNT_BY_EDITION_SIZE.get(n_matches, max(1, round(n_matches * 0.25)))


def stage_masks(feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """(group_mask, knockout_mask) over `feats` for finals-tournament matches.

    Within each finals edition (tournament + year), the LATER `_ko_count` matches by
    chronological order are the knockout rounds and the rest are the group stage. Both
    masks are False for non-finals matches (friendlies/qualifiers/round-robins), which
    use the unshifted global base_goals. Leak-free: the schedule position of a match is
    known pre-kickoff."""
    tours = feats["tournament"]
    years = feats["year"]
    n = len(years)
    is_finals = np.array([str(t) in FINALS_TOURNAMENTS for t in tours], dtype=bool)
    ko = np.zeros(n, dtype=bool)
    # `feats` is already in chronological order (prematch_pass sorts by date), so the
    # position of a row within its edition is its chronological rank.
    editions: dict[tuple, list[int]] = {}
    for i in range(n):
        if is_finals[i]:
            editions.setdefault((str(tours[i]), int(years[i])), []).append(i)
    for idxs in editions.values():
        kc = _ko_count(len(idxs))
        for j in idxs[-kc:]:           # the last kc matches of the edition are knockout
            ko[j] = True
    group = is_finals & ~ko
    return group, ko


def stage_total_bias(feats: dict, params, home_adv_elo: float, mask: np.ndarray,
                     base_delta: float = 0.0) -> float | None:
    """mean predicted total - mean actual total on the `mask` subset, with an optional
    additive `base_delta` on base_goals. Positive = the model OVER-predicts total goals.

    Returns None for an empty subset. Used to fit/gate the per-stage base-goals split."""
    if not mask.any():
        return None
    sub = _subset(feats, mask)
    la, lb = _lambdas(sub["elo_h"], sub["elo_a"], sub["neutral"],
                      params.c, params.base_goals + base_delta, home_adv_elo, params.gamma)
    pred = float(np.mean(la + lb))
    act = float(np.mean(sub["gh"] + sub["ga"]))
    return pred - act


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


# Bounds on the per-stage base-goals split. Group totals are lowered (delta <= 0),
# knockout totals raised (delta >= 0); |delta| <= 0.30 goals keeps the correction a
# modest nudge around the global anchor that a small/noisy finals subset cannot run wild.
_STAGE_DELTA_CAP = 0.30

# Held-out reference biases of the global goal-aware base_goals on the finals group vs
# knockout subsets (mean model_pred total - mean actual total): GROUP over-predicts (+),
# KNOCKOUT under-predicts (-). The mandated split shape is the negative of these (group
# down, knockout up); they seed a stage with no training finals matches.
_REF_GROUP_BIAS = 0.124
_REF_KNOCKOUT_BIAS = -0.053


def fit_stage_deltas(feats_train: dict, params, home_adv_elo: float) -> tuple[float, float]:
    """Fit the (group, knockout) base-goals deltas: group totals down, knockout totals up.

    The global goal-aware `base_goals` averages group and knockout goal volume, so it
    OVER-predicts group totals and UNDER-predicts knockout totals. The additive delta that
    zeros a totals bias of `b` is exactly `-b` (the delta enters the total linearly), so we
    fit each stage on its TRAINING finals subset to drive that stage's totals bias to zero.
    The result is clamped to the mandated corrective shape (group delta <= 0 down, knockout
    delta >= 0 up) and capped at +/-`_STAGE_DELTA_CAP`, so the split only ever helps the
    stage it targets and a noisy/contrarian sub-sample cannot flip its sign. c/gamma/rho/home
    are held fixed (a totals-only knob), so the favourite/W-D-L pick is untouched. When a
    stage has no training finals matches, the held-out reference correction is used instead.
    Returns (group_delta, knockout_delta), rounded."""
    group_mask, ko_mask = stage_masks(feats_train)

    def _delta(mask: np.ndarray, ref_bias: float, lo: float, hi: float) -> float:
        train_bias = stage_total_bias(feats_train, params, home_adv_elo, mask)
        bias = ref_bias if train_bias is None else train_bias
        return round(min(hi, max(lo, -bias)), 3)

    group_delta = _delta(group_mask, _REF_GROUP_BIAS, -_STAGE_DELTA_CAP, 0.0)
    knockout_delta = _delta(ko_mask, _REF_KNOCKOUT_BIAS, 0.0, _STAGE_DELTA_CAP)
    return group_delta, knockout_delta


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

    # corrected stage-split of base_goals: the goal-aware anchor above averages group and
    # knockout goal volume, so it OVER-predicts group totals and UNDER-predicts knockout
    # totals. Fit a per-stage additive correction (group down, knockout up) on the historical
    # finals subsets of the TRAINING window; the deltas move totals only (never supremacy),
    # so W/D/L is untouched. Reported/gated on the held-out finals subsets below.
    group_delta, knockout_delta = fit_stage_deltas(train, params, home_adv_elo)
    params = replace(params, base_goals_group_delta=group_delta,
                     base_goals_knockout_delta=knockout_delta)
    p = (params.c, params.base_goals, home_adv_elo, params.rho, params.gamma)

    # item 3: un-leak temperature. It must be fit on data the params were NOT fit on,
    # or it collapses to ~1.0 and never learns the out-of-sample overconfidence. The
    # held-out validation fold is the tournament/major slice of the test window (the
    # deep-competition class where a results-only Elo over-rates hot favourites); params
    # come from `train` (year<train_until), so this fold is genuinely held out. Fitting
    # T here (not on train) yields a de-sharpening T>1 that actually cools favourites.
    val_temp = _subset(test, test["is_major"])
    temperature = round(fit_temperature(val_temp, p), 3) if len(val_temp["outcome"]) else 1.0

    # item A: supremacy-conditional de-sharpen. A single global temperature cannot reach
    # the elite-vs-elite favourite overconfidence, so fit an EXTRA height `t_elite` that is
    # added only for elite-vs-elite close games (T_eff = temperature + t_elite*g). To make
    # the gate genuinely out-of-sample (the in-sample-gate weakness flagged in batch 1), the
    # is_major fold is split temporally: t_elite is FIT on the earlier sub-fold and the elite
    # gap is SCORED on the later, held-out sub-fold. Both are disjoint from `train`.
    te_split_year = int(np.median(np.unique(val_temp["year"]))) if len(val_temp["outcome"]) else 0
    te_fit = _subset(val_temp, val_temp["year"] < te_split_year)
    te_score = _subset(val_temp, val_temp["year"] >= te_split_year)
    if not len(te_fit["outcome"]) or not len(te_score["outcome"]):
        te_fit = te_score = val_temp   # degenerate split: fall back to the whole fold
    t_elite = (round(fit_t_elite(te_fit, p, home_adv_elo, temperature), 3)
               if len(te_fit["outcome"]) else 0.0)

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

    # corrected stage-split gate: per-stage total-goals bias on the HELD-OUT finals subsets,
    # before (no delta) and after (with the fitted delta). GROUP |bias| must shrink toward 0;
    # KNOCKOUT |bias| must not worsen. group/knockout split classified by schedule position.
    test_group_mask, test_ko_mask = stage_masks(test)
    group_bias_before = stage_total_bias(test, params, home_adv_elo, test_group_mask, 0.0)
    group_bias_after = stage_total_bias(test, params, home_adv_elo, test_group_mask, group_delta)
    ko_bias_before = stage_total_bias(test, params, home_adv_elo, test_ko_mask, 0.0)
    ko_bias_after = stage_total_bias(test, params, home_adv_elo, test_ko_mask, knockout_delta)

    # item 3 calibration slice: report the held-out fold log-loss before/after temperature
    # (this is the gate: calibrated < uncalibrated, with T>1), plus the elite-vs-elite
    # favourite-win overconfidence gap before/after de-sharpening.
    ll_fold = log_loss(val_temp, p) if len(val_temp["outcome"]) else None
    ll_fold_cal = _temp_log_loss(val_temp, p, temperature) if len(val_temp["outcome"]) else None
    # full-test elite gap (continuity with the batch-1 report): raw -> global-temperature ->
    # +supremacy-conditional t_elite. The headline item-A gate is the HELD-OUT sub-fold below.
    elite_gap_raw, elite_n = elite_overconfidence_gap(test, p, home_adv_elo)
    elite_gap_cal, _ = elite_overconfidence_gap(test, p, home_adv_elo, temperature)
    elite_gap_te, _ = elite_overconfidence_gap(test, p, home_adv_elo, temperature,
                                               t_elite=t_elite)
    # item A gate: elite gap on the genuinely held-out SCORE sub-fold (disjoint from the
    # FIT sub-fold t_elite was fit on), global-temperature baseline vs +t_elite.
    elite_gap_score_base, elite_score_n = elite_overconfidence_gap(
        te_score, p, home_adv_elo, temperature)
    elite_gap_score_te, _ = elite_overconfidence_gap(
        te_score, p, home_adv_elo, temperature, t_elite=t_elite)

    # preserve a previously-set friendly-only rho (a validated segment calibration the global
    # fit doesn't re-derive) so a manual re-fit never silently reverts the friendly draw fix
    _prev = json.loads((config.DATA_RAW / "fitted_params.json").read_text()) \
        if (config.DATA_RAW / "fitted_params.json").exists() else {}
    report = {
        "fitted": {**asdict(params), "home_adv_elo": round(home_adv_elo, 1),
                   "temperature": temperature, "t_elite": t_elite, "goal_scale": goal_scale,
                   "rho_friendly": _prev.get("rho_friendly", asdict(params).get("rho_friendly", -0.06))},
        "decay_half_life": decay_half_life,
        "home_adv_majors": home_adv_majors(train, params, home_adv_elo),
        "train": {"n": int(len(train["outcome"])), "log_loss": round(log_loss(train, p), 4),
                  "brier": round(brier(train, p), 4), "rps": round(rps(train, p), 4)},
        "test": {"n": int(len(test["outcome"])), "log_loss": round(log_loss(test, p), 4),
                 "log_loss_ci95": [ci_lo, ci_hi],
                 # calibrated log-loss uses the SHIPPED matchup-conditional temperature
                 # (global temperature for ordinary matches, +t_elite for elite-vs-elite),
                 # so this is the real guardrail on the published behaviour
                 "log_loss_calibrated": round(
                     _eff_temp_log_loss(test, p, home_adv_elo, temperature, t_elite), 4),
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
            # supremacy-conditional de-sharpen height (item A); 0 = global-temperature only
            "t_elite": t_elite,
            # elite-vs-elite (both Elo>2000) favourite-win overconfidence gap, in pp, on the
            # FULL test slice: raw -> global temperature -> +supremacy-conditional t_elite
            "elite_n": elite_n,
            "elite_fav_gap_pp": round(elite_gap_raw * 100, 2),
            "elite_fav_gap_pp_calibrated": round(elite_gap_cal * 100, 2),
            "elite_fav_gap_pp_supremacy_cond": round(elite_gap_te * 100, 2),
            # item A gate: the elite gap on the GENUINELY HELD-OUT score sub-fold (disjoint
            # from the fit sub-fold t_elite was fit on) must drop under the supremacy-conditional
            # de-sharpen vs the global-temperature baseline
            "elite_holdout_n": elite_score_n,
            "elite_holdout_gap_pp_temperature": round(elite_gap_score_base * 100, 2),
            "elite_holdout_gap_pp_supremacy_cond": round(elite_gap_score_te * 100, 2),
        },
        "totals": {"n": int(len(test["outcome"])),
                   "mean_pred_total": round(pred_tot, 4),
                   "mean_actual_total": round(act_tot, 4),
                   "mean_total_dev": round(pred_tot - act_tot, 4),
                   "model_btts": round(model_btts(test, p), 4),
                   "base_goals_wdl": round(base_goals_wdl, 3),
                   "base_goals_goal_aware": round(params.base_goals, 3),
                   "lambda_tot": LAMBDA_TOT,
                   # corrected stage-split of base_goals (group down, knockout up). The
                   # *_bias_* fields are the gate: mean predicted total - mean actual total
                   # on the held-out finals subsets, before vs after the per-stage delta.
                   "base_goals_group_delta": group_delta,
                   "base_goals_knockout_delta": knockout_delta,
                   "group_n": int(test_group_mask.sum()),
                   "knockout_n": int(test_ko_mask.sum()),
                   "group_bias_before": round(group_bias_before, 4)
                   if group_bias_before is not None else None,
                   "group_bias_after": round(group_bias_after, 4)
                   if group_bias_after is not None else None,
                   "knockout_bias_before": round(ko_bias_before, 4)
                   if ko_bias_before is not None else None,
                   "knockout_bias_after": round(ko_bias_after, 4)
                   if ko_bias_after is not None else None},
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
