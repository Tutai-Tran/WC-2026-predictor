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
from collections import defaultdict
from datetime import datetime, timezone

from . import config, db, model as model_mod, scorers as scorers_mod
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


def load_fitted_params() -> model_mod.ModelParams | None:
    fp = config.DATA_RAW / "fitted_params.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return model_mod.ModelParams(
        c=d.get("c", 219.0), base_goals=d.get("base_goals", 2.6), rho=d.get("rho", -0.06),
        max_goals=d.get("max_goals", 10), min_lambda=d.get("min_lambda", 0.15),
    )


def load_tournament(conn, params: model_mod.ModelParams | None = None) -> Tournament:
    if params is None:
        params = load_fitted_params()
    teams: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT t.name n, t.group_letter g, t.fifa_rank fr, r.elo e "
        "FROM teams t JOIN ratings r ON r.team_id=t.id "
        "WHERE r.valid_from = (SELECT MAX(r2.valid_from) FROM ratings r2 WHERE r2.team_id=t.id)"
    ):
        teams[r["n"]] = {"elo": r["e"], "group": r["g"], "fifa_rank": r["fr"]}

    for r in conn.execute(
        "SELECT t.name tn, p.name pn, p.position pos, p.club_goals cg, p.is_penalty_taker pk "
        "FROM players p JOIN teams t ON t.id=p.team_id"
    ):
        if r["tn"] in teams:
            teams[r["tn"]].setdefault("players", []).append(
                {"name": r["pn"], "position": r["pos"], "club_goals": r["cg"],
                 "is_penalty_taker": bool(r["pk"])}
            )

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
            "lambda_home": round(f["lambda_home"], 2), "lambda_away": round(f["lambda_away"], 2),
            "top_scoreline": f"{sh}-{sa}", "top_scoreline_p": round(sp, 3),
            "top_scorers_home": scorers_mod.top_scorers(t.teams[h].get("players", []), f["lambda_home"]),
            "top_scorers_away": scorers_mod.top_scorers(t.teams[a].get("players", []), f["lambda_away"]),
        })
    return out


def golden_boot(t: Tournament, top_n: int = 15) -> list[dict]:
    """Expected group-stage goals per player (a 'most likely to score' ranking)."""
    # key by (team, player) so distinct same-named players are not merged
    exp_goals: dict[tuple[str, str], float] = defaultdict(float)
    for fx in t.group_fixtures:
        h, a, vc = fx["home"], fx["away"], fx.get("venue_country")
        la, lb = model_mod.match_lambdas(
            t.teams[h]["elo"], t.teams[a]["elo"], t.params,
            _host_adv(h, vc, t.host_bump), _host_adv(a, vc, t.host_bump),
        )
        for team, players, lam in ((h, t.teams[h].get("players", []), la),
                                   (a, t.teams[a].get("players", []), lb)):
            shares = scorers_mod.team_goal_shares(players)
            for p in players:
                exp_goals[(team, p["name"])] += shares[p["name"]] * lam
    ranked = sorted(exp_goals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"player": name, "team": team, "exp_group_goals": round(g, 2)}
            for (team, name), g in ranked]


def run_forecast(conn, n_runs: int = config.DEFAULT_SIM_RUNS,
                 seed: int = config.DEFAULT_RNG_SEED) -> dict:
    t = load_tournament(conn)
    sim = simulate(t, n_runs=n_runs, seed=seed)
    matches = group_match_forecasts(t)
    boot = golden_boot(t)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    ts = datetime.now(timezone.utc).isoformat()
    data_as_of = json.loads((config.DATA_RAW / "elo_seed.json").read_text()).get("updated")

    payload = {
        "probs": sim["probs"], "matches": matches, "golden_boot": boot,
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
    print("\nSample group-match forecasts (first 5):")
    for m in result["matches"][:5]:
        print(f"  [{m['group']}] {m['home']} vs {m['away']}  "
              f"W {m['p_home']*100:.0f}% / D {m['p_draw']*100:.0f}% / L {m['p_away']*100:.0f}%  "
              f"likely {m['top_scoreline']} ({m['top_scoreline_p']*100:.0f}%)")
        sc = ", ".join(f"{s['player']} {s['p_anytime']*100:.0f}%"
                       for s in (m['top_scorers_home'] + m['top_scorers_away'])[:4])
        print(f"        scorers: {sc}")
    print("\nMost likely scorers (expected group-stage goals):")
    for b in result["golden_boot"][:10]:
        print(f"  {b['player']:<22} ({b['team']:<14}) {b['exp_group_goals']:.2f}")
    total = sum(p["champion"] for p in probs.values())
    print(f"\n(Champion probabilities sum to {total:.3f}; should be ~1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description="World Cup 2026 forecast")
    ap.add_argument("--runs", type=int, default=config.DEFAULT_SIM_RUNS)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_RNG_SEED)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--no-vault", action="store_true", help="skip writing the WC vault")
    args = ap.parse_args()
    conn = db.connect(args.db)
    db.init_db(conn)
    # self-bootstrap: ingest from data/raw if the database is empty
    if conn.execute("SELECT COUNT(*) c FROM teams").fetchone()["c"] == 0:
        from . import ingest
        print("Empty database; ingesting from data/raw ...")
        ingest.ingest_all(conn)
    result = run_forecast(conn, n_runs=args.runs, seed=args.seed)
    _print_summary(result)
    if not args.no_vault:
        from . import vaultgen
        counts = vaultgen.generate(conn, result)
        print(f"\nWC vault updated: {counts}")


if __name__ == "__main__":
    main()
