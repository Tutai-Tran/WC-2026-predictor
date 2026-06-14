"""Orchestrator and CLI: build the full World Cup 2026 forecast.

Run:  python -m wc26.forecast [--runs N] [--seed S]
Produces analytic per-match forecasts for the group stage and a Monte Carlo
forecast of progression and the champion, stores an append-only snapshot, and
prints a readable summary. Probabilities, never certainties.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

from . import (
    config, db, h2h as h2h_mod, model as model_mod, odds as odds_mod,
    scorers as scorers_mod,
)
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
    try:
        d = json.loads(fp.read_text())
    except (OSError, ValueError) as e:
        # a corrupt params file must degrade to defaults, not kill every future
        # refresh; the backtest rewrites the file on the next manual run
        print(f"WARNING: fitted_params.json unreadable ({e}); using model defaults")
        return None
    return model_mod.ModelParams(
        c=d.get("c", 219.0), base_goals=d.get("base_goals", 2.6), rho=d.get("rho", -0.06),
        rho_friendly=d.get("rho_friendly", d.get("rho", -0.06)),
        max_goals=d.get("max_goals", 10), min_lambda=d.get("min_lambda", 0.15),
        goal_scale=d.get("goal_scale", 1.0), gamma=d.get("gamma", 0.0),
        # un-leaked temperature (item 3): fit out-of-sample in the backtest and now
        # actually read here, so the de-sharpening reaches the published W/D/L probs
        temperature=d.get("temperature", 1.0),
        # supremacy-conditional de-sharpen (item A): extra elite-vs-elite cooling on top
        # of temperature, fit on a held-out sub-fold; 0.0 (a no-op) when absent
        t_elite=d.get("t_elite", 0.0),
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

    ko_venues = {r["match_no"]: r["venue_country"] for r in conn.execute(
        "SELECT match_no, venue_country FROM matches "
        "WHERE stage IN ('R16','QF','SF','Final') AND venue_country IS NOT NULL "
        "AND venue_country != ''"
    )}

    return Tournament(
        teams=teams, groups=groups, group_fixtures=fixtures,
        r32=r32, routing=routing,
        params=params or model_mod.ModelParams(),
        attack_mult=attack_mult, played=played, h2h=h2h_map, ko_venues=ko_venues,
    )


def group_match_forecasts(t: Tournament,
                          market: dict[str, dict[str, float]] | None = None) -> list[dict]:
    market = market or {}
    blend_w = odds_mod.match_blend_weight() if market else None
    out = []
    for fx in t.group_fixtures:
        h, a = fx["home"], fx["away"]
        vc = fx.get("venue_country")
        f = model_mod.match_forecast(
            t.teams[h]["elo"], t.teams[a]["elo"], t.params,
            t.host_adv(h, vc), t.host_adv(a, vc),
            mult_a=t.mult(h), mult_b=t.mult(a), h2h_delta=t.h2h_delta(h, a),
        )
        ph_m, pdw_m, pa_m = round(f["p_home"], 3), round(f["p_draw"], 3), round(f["p_away"], 3)
        # live market blend (item 1): if a devigged line exists for this fixture, the
        # PUBLISHED W/D/L is a log-pool of model+market (cools hot favourites toward the
        # sharp line); the raw model probs are always kept for the post-mortem audit
        # trail. No line (all historical data, most group games pre-odds) -> pure model.
        line = market.get(f"{h} vs {a}")
        if line:
            blended = odds_mod.blend_wdl(
                {"home": ph_m, "draw": pdw_m, "away": pa_m}, line, blend_w)
            ph, pdw, pa = (round(blended["home"], 3), round(blended["draw"], 3),
                           round(blended["away"], 3))
        else:
            ph, pdw, pa = ph_m, pdw_m, pa_m
        # pick the headline score from the matrix region of the modal outcome computed
        # on the *rounded* probs, so it can never contradict the W/D/L shown next to it
        (sh, sa), sp = model_mod.most_likely_scoreline(f["matrix"], model_mod.outcome_label(ph, pdw, pa))
        tops = model_mod.top_scorelines(f["matrix"], 5)
        es = model_mod.expected_score(f["lambda_home"], f["lambda_away"])
        res = t.played.get((h, a))
        out.append({
            "group": fx["group"], "home": h, "away": a, "date": fx.get("date"),
            "p_home": ph, "p_draw": pdw, "p_away": pa,
            # raw pre-blend model probs, always present for leak-free grading
            "p_home_model": ph_m, "p_draw_model": pdw_m, "p_away_model": pa_m,
            "lambda_home": round(f["lambda_home"], 2), "lambda_away": round(f["lambda_away"], 2),
            # adjustments baked into the lambdas, frozen so post-mortems can see them
            "mult_home": round(t.mult(h), 3), "mult_away": round(t.mult(a), 3),
            "h2h_delta": round(t.h2h_delta(h, a), 3),
            "top_scoreline": f"{sh}-{sa}", "top_scoreline_p": round(sp, 3),
            # full scoreline distribution: the modal cell holds <=16% mass, so publish
            # the top-5 with their probabilities rather than implying a single prediction
            "top_scorelines": [[[i, j], round(p, 4)] for (i, j), p in tops],
            "expected_score": f"{es[0]}-{es[1]}",
            "derived_markets": model_mod.derived_markets(f["matrix"]),
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
            t.host_adv(h, vc), t.host_adv(a, vc), t.h2h_delta(h, a),
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
        (sh, sa), sp = model_mod.most_likely_scoreline(f["matrix"], model_mod.outcome_label(ph, pdw, pa))
        tops = model_mod.top_scorelines(f["matrix"], 5)
        es = model_mod.expected_score(f["lambda_home"], f["lambda_away"])
        out.append({
            "date": r["d"], "home": r["home"], "away": r["away"],
            "p_home": ph, "p_draw": pdw, "p_away": pa, "top_scoreline": f"{sh}-{sa}",
            "top_scoreline_p": round(sp, 3),
            "top_scorelines": [[[i, j], round(p, 4)] for (i, j), p in tops],
            "expected_score": f"{es[0]}-{es[1]}",
            "derived_markets": model_mod.derived_markets(f["matrix"]),
            "lambda_home": round(f["lambda_home"], 2), "lambda_away": round(f["lambda_away"], 2),
            "h2h_delta": round(delta, 3),
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


def input_fingerprint(conn, n_runs: int = config.DEFAULT_SIM_RUNS,
                      seed: int = config.DEFAULT_RNG_SEED) -> str:
    """Cheap, stable hash of everything the forecast depends on: latest ratings,
    active availability events, played matches, and the fitted parameters. If two
    refreshes share a fingerprint, re-running the 50k-sim forecast would reproduce
    the previous payload bit-for-bit (fixed seed), so the run can be skipped."""
    import hashlib
    h = hashlib.sha256()
    h.update(f"runs={n_runs};seed={seed};".encode())
    fp = config.DATA_RAW / "fitted_params.json"
    h.update(fp.read_bytes() if fp.exists() else b"-")
    for r in conn.execute(
        "SELECT t.name, r.elo FROM teams t JOIN ratings r ON r.team_id=t.id "
        "WHERE r.valid_from=(SELECT MAX(valid_from) FROM ratings r2 WHERE r2.team_id=t.id) "
        "ORDER BY t.name"
    ):
        h.update(f"{r['name']}={r['elo']:.2f};".encode())
    # only events that currently INFLUENCE the forecast: scan markers are bookkeeping
    # (they change every news cycle without moving a forecast), and judging expiry via
    # _event_active here means an event crossing its 72h/return-date boundary changes
    # the fingerprint, so a "skip" can never freeze an injury past its expiry
    from datetime import datetime, timedelta, timezone

    from .overrides import _event_active
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    cutoff = (now - timedelta(hours=config.NEWS_EVENT_MAX_AGE_HOURS)).isoformat()
    for r in conn.execute(
        "SELECT team_id, player_name, status, minutes_factor, expected_return, source, "
        "confidence, fetched_at FROM availability_events "
        "WHERE source != 'news-llm-marker' ORDER BY team_id, player_name, id"
    ):
        if _event_active(r, today, cutoff):
            h.update(repr((r["team_id"], r["player_name"], r["status"],
                           r["minutes_factor"], r["confidence"])).encode())
    for r in conn.execute(
        "SELECT id, home_goals, away_goals, played FROM matches ORDER BY id"
    ):
        h.update(repr(tuple(r)).encode())
    # the market enters the published payload via champion_blend and the per-match
    # W/D/L blend (item 1), so new odds must invalidate the fingerprint or the blend
    # would go stale on skipped cycles
    row = conn.execute(
        "SELECT id FROM public_benchmark WHERE source='the-odds-api' AND scope='champion' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    h2h_row = conn.execute(
        "SELECT id FROM public_benchmark WHERE source='the-odds-api' AND scope='match' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    h.update(f"odds={row['id'] if row else None};".encode())
    h.update(f"odds_h2h={h2h_row['id'] if h2h_row else None};".encode())
    return h.hexdigest()


def run_forecast(conn, n_runs: int = config.DEFAULT_SIM_RUNS,
                 seed: int = config.DEFAULT_RNG_SEED,
                 input_hash: str | None = None) -> dict:
    h2h_index = h2h_mod.build_index()
    params = load_fitted_params() or model_mod.ModelParams()
    t = load_tournament(conn, params=params, h2h_index=h2h_index)
    sim = simulate(t, n_runs=n_runs, seed=seed)
    # live h2h market blend (item 1): pure model when no line exists for a fixture, so
    # the all-historical backtest is unchanged; live group games blend once odds open
    market_h2h = odds_mod.latest_market_h2h(conn)
    matches = group_match_forecasts(t, market_h2h)
    boot = golden_boot(t)
    # friendlies predict the 2026 warm-ups, so they get a leak-free (pre-2026) H2H index AND a
    # stronger Dixon-Coles rho (validated: friendlies systematically under-predict draws because
    # they over-produce goals; competitive/WC matches stay at the fitted rho and are untouched).
    friendly_h2h = h2h_mod.build_index(cutoff=h2h_mod.PRE_WARMUP_CUTOFF)
    friendly_params = dataclasses.replace(t.params, rho=t.params.rho_friendly)
    friendlies = friendly_forecasts(conn, friendly_params, friendly_h2h)
    proj = bracket_projection(conn, sim["probs"])
    knockout = knockout_fixtures(conn, proj)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    ts = datetime.now(timezone.utc).isoformat()
    # live ratings come from replayed_elo.json (base + results-ledger replay, dated
    # by the last scrape); fall back to the static seed only if it is missing
    data_as_of = next(
        (json.loads((config.DATA_RAW / f).read_text()).get("updated")
         for f in ("replayed_elo.json", "elo_seed.json")
         if (config.DATA_RAW / f).exists()),
        None,
    )

    # model+market champion blend: the devigged market is the strongest single external
    # signal, so the blend is published alongside (never instead of) the pure model.
    market = odds_mod.latest_market_champion(conn)
    blend = odds_mod.blend_probs({tm: p["champion"] for tm, p in sim["probs"].items()}, market)

    payload = {
        "probs": sim["probs"], "matches": matches, "golden_boot": boot,
        "friendlies": friendlies, "knockout": knockout, "bracket_slots": proj,
        "champion_blend": blend, "blend_weight": 0.5,
        "data_as_of": data_as_of, "n_runs": n_runs, "seed": seed,
        "goal_calibration": {"goal_scale": params.goal_scale, "gamma": params.gamma},
    }
    conn.execute(
        "INSERT INTO predictions (run_id, scope, ref, payload_json, computed_at) "
        "VALUES (?,?,?,?,?)",
        (run_id, "forecast", "all", json.dumps(payload), ts),
    )
    conn.execute(
        "INSERT INTO model_runs (run_id, ts, git_sha, config_json, input_hash, rng_seed, "
        "model_version, data_completeness) VALUES (?,?,?,?,?,?,?,?) "
        "ON CONFLICT(run_id) DO NOTHING",
        (run_id, ts, _git_sha(), json.dumps(t.params.__dict__),
         input_hash or input_fingerprint(conn, n_runs, seed), seed,
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
