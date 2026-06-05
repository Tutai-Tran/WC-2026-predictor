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

from . import config, db, h2h as h2h_mod, model as model_mod, scorers as scorers_mod
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
        goal_scale=d.get("goal_scale", 1.0), gamma=d.get("gamma", 0.0),
    )


def load_tournament(conn, params: model_mod.ModelParams | None = None,
                    h2h_index: dict | None = None) -> Tournament:
    if params is None:
        params = load_fitted_params()
    if h2h_index is None:
        h2h_index = h2h_mod.build_index()
    teams: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT t.name n, t.group_letter g, t.fifa_rank fr, r.elo e "
        "FROM teams t JOIN ratings r ON r.team_id=t.id "
        "WHERE t.group_letter IS NOT NULL "
        "AND r.valid_from = (SELECT MAX(r2.valid_from) FROM ratings r2 WHERE r2.team_id=t.id) "
        "ORDER BY t.name"   # deterministic team order -> deterministic h2h delta_map construction
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
    for r in conn.execute(
        "SELECT name, group_letter FROM teams WHERE group_letter IS NOT NULL "
        "ORDER BY group_letter, name"
    ):
        groups.setdefault(r["group_letter"], []).append(r["name"])

    fixtures: list[dict] = []
    for r in conn.execute(
        "SELECT m.group_letter grp, h.name home, a.name away, m.venue_country vc, m.date_utc d "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id "
        "JOIN teams a ON a.id=m.away_team_id WHERE m.stage='group'"
    ):
        fixtures.append({"group": r["grp"], "home": r["home"], "away": r["away"],
                         "venue_country": r["vc"], "date": r["d"]})

    r32 = json.loads((config.DATA_RAW / "bracket.json").read_text())["r32"]
    routing = json.loads((config.DATA_RAW / "third_place_routing.json").read_text())["combinations"]

    from . import overrides
    players_by_team = {name: d.get("players", []) for name, d in teams.items()}
    attack_mult = overrides.availability_multipliers(conn, players_by_team)

    played: dict = {}
    for r in conn.execute(
        "SELECT h.name home, a.name away, m.home_goals hg, m.away_goals ag "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage='group' AND m.played=1 AND m.home_goals IS NOT NULL"
    ):
        played[(r["home"], r["away"])] = (int(r["hg"]), int(r["ag"]))

    h2h_map = h2h_mod.delta_map(teams.keys(), h2h_index)

    return Tournament(
        teams=teams, groups=groups, group_fixtures=fixtures,
        r32=r32, routing=routing,
        params=params or model_mod.ModelParams(),
        attack_mult=attack_mult, played=played, h2h=h2h_map,
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
            mult_a=t.mult(h), mult_b=t.mult(a), h2h_delta=t.h2h_delta(h, a),
        )
        ph, pdw, pa = round(f["p_home"], 3), round(f["p_draw"], 3), round(f["p_away"], 3)
        # pick the headline score from the matrix region of the modal outcome computed
        # on the *rounded* probs, so it can never contradict the W/D/L shown next to it
        (sh, sa), sp = model_mod.most_likely_scoreline(f["matrix"], model_mod.outcome_label(ph, pdw, pa))
        res = t.played.get((h, a))
        out.append({
            "group": fx["group"], "home": h, "away": a, "date": fx.get("date"),
            "p_home": ph, "p_draw": pdw, "p_away": pa,
            "lambda_home": round(f["lambda_home"], 2), "lambda_away": round(f["lambda_away"], 2),
            "top_scoreline": f"{sh}-{sa}", "top_scoreline_p": round(sp, 3),
            "result": (f"{res[0]}-{res[1]}" if res else None),
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
            _host_adv(h, vc, t.host_bump), _host_adv(a, vc, t.host_bump), t.h2h_delta(h, a),
        )
        la *= t.mult(h)
        lb *= t.mult(a)
        for team, players, lam in ((h, t.teams[h].get("players", []), la),
                                   (a, t.teams[a].get("players", []), lb)):
            shares = scorers_mod.team_goal_shares(players)
            for p in players:
                exp_goals[(team, p["name"])] += shares[p["name"]] * lam
    ranked = sorted(exp_goals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"player": name, "team": team, "exp_group_goals": round(g, 2)}
            for (team, name), g in ranked]


def friendly_forecasts(conn, params: model_mod.ModelParams,
                       h2h_index: dict | None = None) -> list[dict]:
    """Predict pre-tournament warm-up friendlies (any teams with a known Elo).

    Uses an H2H index built only from pre-2026 history so a played warm-up never
    leaks its own result into its own forecast (keeps it an honest accuracy check)."""
    if h2h_index is None:
        h2h_index = h2h_mod.build_index(cutoff=h2h_mod.PRE_WARMUP_CUTOFF)
    out = []
    for r in conn.execute(
        "SELECT m.date_utc d, h.name home, a.name away, m.home_goals hg, m.away_goals ag, m.played p, "
        "(SELECT elo FROM ratings WHERE team_id=h.id ORDER BY valid_from DESC LIMIT 1) he, "
        "(SELECT elo FROM ratings WHERE team_id=a.id ORDER BY valid_from DESC LIMIT 1) ae "
        "FROM matches m JOIN teams h ON h.id=m.home_team_id JOIN teams a ON a.id=m.away_team_id "
        "WHERE m.stage='friendly' ORDER BY m.date_utc"
    ):
        if r["he"] is None or r["ae"] is None:
            continue
        meetings = h2h_index.get(frozenset((r["home"], r["away"]))) or []
        delta = h2h_mod.h2h_delta(meetings, r["home"], r["away"])
        f = model_mod.match_forecast(r["he"], r["ae"], params, h2h_delta=delta)
        ph, pdw, pa = round(f["p_home"], 3), round(f["p_draw"], 3), round(f["p_away"], 3)
        (sh, sa), _ = model_mod.most_likely_scoreline(f["matrix"], model_mod.outcome_label(ph, pdw, pa))
        out.append({
            "date": r["d"], "home": r["home"], "away": r["away"],
            "p_home": ph, "p_draw": pdw, "p_away": pa, "top_scoreline": f"{sh}-{sa}",
            "lambda_home": round(f["lambda_home"], 2), "lambda_away": round(f["lambda_away"], 2),
            "played": bool(r["p"]),
            "result": (f'{r["hg"]}-{r["ag"]}' if r["p"] else None),
        })
    return out


def bracket_projection(conn, probs: dict) -> dict[str, str]:
    """Most-likely team for each group-position slot (1X = winner, 2X = runner-up).

    Reads the (conditional) simulation probabilities, so as real group results are
    entered the projection sharpens toward the actual qualifiers."""
    groups: dict[str, list[str]] = {}
    for r in conn.execute("SELECT name, group_letter FROM teams WHERE group_letter IS NOT NULL"):
        groups.setdefault(r["group_letter"], []).append(r["name"])
    proj: dict[str, str] = {}
    for g, teams in groups.items():
        winner = max(teams, key=lambda t: probs[t]["win_group"])
        proj[f"1{g}"] = winner
        rest = [t for t in teams if t != winner]
        proj[f"2{g}"] = max(rest, key=lambda t: probs[t]["top2"] - probs[t]["win_group"])
    return proj


def knockout_fixtures(conn, proj: dict[str, str]) -> list[dict]:
    """All knockout fixtures with dates, slot labels, and projected teams."""
    out = []
    for r in conn.execute(
        "SELECT match_no, stage, date_utc, venue, venue_country, home_slot, away_slot, "
        "home_goals hg, away_goals ag, played FROM matches "
        "WHERE stage IN ('R32','R16','QF','SF','Final') ORDER BY match_no"
    ):
        out.append({
            "match_no": r["match_no"], "stage": r["stage"], "date": r["date_utc"],
            "venue": r["venue"], "home_slot": r["home_slot"], "away_slot": r["away_slot"],
            "home_proj": proj.get(r["home_slot"]), "away_proj": proj.get(r["away_slot"]),
            "result": (f'{r["hg"]}-{r["ag"]}' if r["played"] else None),
        })
    return out


def run_forecast(conn, n_runs: int = config.DEFAULT_SIM_RUNS,
                 seed: int = config.DEFAULT_RNG_SEED) -> dict:
    h2h_index = h2h_mod.build_index()
    params = load_fitted_params() or model_mod.ModelParams()
    t = load_tournament(conn, params=params, h2h_index=h2h_index)
    sim = simulate(t, n_runs=n_runs, seed=seed)
    matches = group_match_forecasts(t)
    boot = golden_boot(t)
    # friendlies predict the 2026 warm-ups, so they get a leak-free (pre-2026) H2H index
    friendly_h2h = h2h_mod.build_index(cutoff=h2h_mod.PRE_WARMUP_CUTOFF)
    friendlies = friendly_forecasts(conn, t.params, friendly_h2h)
    proj = bracket_projection(conn, sim["probs"])
    knockout = knockout_fixtures(conn, proj)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    ts = datetime.now(timezone.utc).isoformat()
    data_as_of = json.loads((config.DATA_RAW / "elo_seed.json").read_text()).get("updated")

    payload = {
        "probs": sim["probs"], "matches": matches, "golden_boot": boot,
        "friendlies": friendlies, "knockout": knockout, "bracket_slots": proj,
        "data_as_of": data_as_of, "n_runs": n_runs, "seed": seed,
        "goal_calibration": {"goal_scale": params.goal_scale, "gamma": params.gamma},
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
