"""Penalty-shootout tilt, estimated from history instead of assumed.

Run:  python -m wc26.shootout
The knockout simulator tilts a shootout toward the stronger side by
SHOOTOUT_FAVOURITE_TILT * (2*E - 1), where E is the Elo expected score. This module
estimates that tilt from every recorded shootout (martj42 shootouts.csv joined to
the pre-match Elo replay): regression through the origin of (won - 0.5) on (2E - 1).
Writes data/raw/shootout_report.json; config.SHOOTOUT_FAVOURITE_TILT should track
the estimate (shootouts are historically close to a coin flip, so expect a small one).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest as bt, config, elo as elo_mod


def run(write: bool = True, min_year: int = 1990) -> dict:
    df = bt.read_results()
    feats, _ = bt.prematch_pass(df)
    # index pre-match Elo by (date, home, away); shootouts.csv shares the source keys
    key = {}
    dates = df["date"].dt.date.astype(str).to_numpy()
    homes, aways = df["home_team"].to_numpy(), df["away_team"].to_numpy()
    for i in range(len(df)):
        key[(dates[i], homes[i], aways[i])] = (feats["elo_h"][i], feats["elo_a"][i])

    so = pd.read_csv(config.DATA_RAW / "shootouts.csv")
    so = so[pd.to_datetime(so["date"]).dt.year >= min_year]
    x, y = [], []
    for _, r in so.iterrows():
        elos = key.get((r["date"], r["home_team"], r["away_team"]))
        if not elos or r["winner"] not in (r["home_team"], r["away_team"]):
            continue
        eh, ea = elos
        e_home = elo_mod.expected(eh, ea)            # neutral: no home advantage in pens
        x.append(2.0 * e_home - 1.0)
        y.append((1.0 if r["winner"] == r["home_team"] else 0.0) - 0.5)
    x, y = np.array(x), np.array(y)
    n = len(x)
    if n < 50 or float(np.sum(x * x)) == 0.0:
        return {"n": n, "error": "not enough joined shootouts to estimate a tilt"}
    slope = float(np.sum(x * y) / np.sum(x * x))     # OLS through the origin
    # standard error of the through-origin slope for a sanity band
    resid = y - slope * x
    se = float(np.sqrt(np.sum(resid**2) / max(1, n - 1) / np.sum(x * x)))
    has_fav = x != 0                                 # equal-Elo pairings have no favourite
    fav_won = float(np.mean((y[has_fav] > 0) == (x[has_fav] > 0))) if has_fav.any() else 0.5
    report = {
        "n": n, "since": min_year,
        "favourite_won_rate": round(fav_won, 4),
        "tilt_estimate": round(slope, 4),
        "tilt_se": round(se, 4),
        "configured_tilt": config.SHOOTOUT_FAVOURITE_TILT,
        "note": "tilt = slope of (won-0.5) on (2E-1); shootouts are nearly a coin flip, "
                "so the configured tilt should sit inside estimate ± 2*se",
    }
    if write:
        (config.DATA_RAW / "shootout_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
