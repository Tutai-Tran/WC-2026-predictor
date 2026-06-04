"""Orchestrator and CLI: build the full World Cup 2026 forecast.

Run:  python -m wc26.forecast [--runs N] [--seed S]
Produces analytic per-match forecasts for the group stage and a Monte Carlo
forecast of progression and the champion, stores an append-only snapshot, and
prints a readable summary. Probabilities, never certainties.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

from . import config, db, model as model_mod
from .simulate import Tournament, simulate

HOSTS = config.HOSTS


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def load_tournament(conn, params: model_mod.ModelParams | None = None) -> Tournament:
    teams: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT t.name n, t.group_letter g, t.fifa_rank fr, r.elo e "
        "FROM teams t JOIN ratings r ON r.team_id=t.id"
    ):
        teams[r["n"]] = {"elo": r["e"], "group": r["g"], "fifa_rank": r["fr"]}

    groups: dict[str, list[str]] = {}
    for r in conn.execute("SELECT name, group_letter FROM teams ORDER BY group_letter, name"):
        groups.setdefault(r["group_letter"], []).append(r["name"])

    fixtures: list[dict] = []
    for r in conn.execute(
        "SELECT m.group_letter grp, h.name home, a.name away, m.venue_country vc "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id "
        "JOIN teams a ON a.id=m.away_team_id WHERE m.stage='group'"
    ):
        fixtures.append({"group": r["grp"], "home": r["home"], "away": r["away"],
                         "venue_country": r["vc"]})

    r32 = json.loads((config.DATA_RAW / "bracket.json").read_text())["r32"]
    routing = json.loads((config.DATA_RAW / "third_place_routing.json").read_text())["combinations"]

    return Tournament(
        teams=teams, groups=groups, group_fixtures=fixtures,
        r32=r32, routing=routing,
        params=params or model_mod.ModelParams(),
    )


def _host_adv(team: str, venue_country: str | None, bump: float) -> float:
    return bump if venue_country and team == venue_country else 0.0


def group_match_forecasts(t: Tournament) -> list[dict]:
    out = []
    for fx in t.group_fixtures:
        h, a = fx["home"], fx["away"]
        vc = fx.get("venue_country")
        f = model_mod.match_forecast(
            t.teams[h]["elo"], t.teams[a]["elo"], t.params,
            _host_adv(h, vc, t.host_bump), _host_adv(a, vc, t.host_bump),
        )
        (sh, sa), sp = f["top_scorelines"][0]
        out.append({
            "group": fx["group"], "home": h, "away": a,
            "p_home": round(f["p_home"], 3), "p_draw": round(f["p_draw"], 3),
            "p_away": round(f["p_away"], 3),
            "top_scoreline": f"{sh}-{sa}", "top_scoreline_p": round(sp, 3),
        })
    return out


def run_forecast(conn, n_runs: int = config.DEFAULT_SIM_RUNS,
                 seed: int = config.DEFAULT_RNG_SEED) -> dict:
    t = load_tournament(conn)
    sim = simulate(t, n_runs=n_runs, seed=seed)
    matches = group_match_forecasts(t)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    ts = datetime.now(timezone.utc).isoformat()
    data_as_of = json.loads((config.DATA_RAW / "elo_seed.json").read_text()).get("updated")

    payload = {
        "probs": sim["probs"], "matches": matches,
        "data_as_of": data_as_of, "n_runs": n_runs, "seed": seed,
    }
    conn.execute(
        "INSERT INTO predictions (run_id, scope, ref, payload_json, computed_at) "
        "VALUES (?,?,?,?,?)",
        (run_id, "forecast", "all", json.dumps(payload), ts),
    )
    conn.execute(
        "INSERT INTO model_runs (run_id, ts, git_sha, config_json, rng_seed, "
        "model_version, data_completeness) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(run_id) DO NOTHING",
        (run_id, ts, _git_sha(), json.dumps(t.params.__dict__), seed,
         "elo-goal-v1", 1.0),
    )
    conn.commit()
    return {"run_id": run_id, "data_as_of": data_as_of, **payload}


def _print_summary(result: dict) -> None:
    probs = result["probs"]
    champ = sorted(probs.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    print("\n" + "=" * 60)
    print(f"  WORLD CUP 2026 FORECAST  (runs={result['n_runs']}, seed={result['seed']})")
    print(f"  Data as of: {result['data_as_of']}  |  probabilities, not certainties")
    print("=" * 60)
    print("\nChampion probability (top 16):")
    for name, p in champ[:16]:
        print(f"  {name:<22} win {p['champion']*100:5.1f}%   "
              f"final {p['final']*100:5.1f}%   SF {p['sf']*100:5.1f}%   "
              f"QF {p['qf']*100:5.1f}%")
    print("\nSample group-match forecasts (first 6):")
    for m in result["matches"][:6]:
        print(f"  [{m['group']}] {m['home']:<16} vs {m['away']:<16}  "
              f"W {m['p_home']*100:4.0f}% / D {m['p_draw']*100:4.0f}% / "
              f"L {m['p_away']*100:4.0f}%   most likely {m['top_scoreline']} "
              f"({m['top_scoreline_p']*100:.0f}%)")
    total = sum(p["champion"] for p in probs.values())
    print(f"\n(Champion probabilities sum to {total:.3f}; should be ~1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description="World Cup 2026 forecast")
    ap.add_argument("--runs", type=int, default=config.DEFAULT_SIM_RUNS)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_RNG_SEED)
    ap.add_argument("--db", type=str, default=None)
    args = ap.parse_args()
    conn = db.connect(args.db)
    db.init_db(conn)
    result = run_forecast(conn, n_runs=args.runs, seed=args.seed)
    _print_summary(result)


if __name__ == "__main__":
    main()
