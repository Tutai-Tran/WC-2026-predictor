"""Monte Carlo tournament simulation for the 2026 format.

Group stage -> standings (rules.rank_group) -> winners, runners-up, and the 8 best
third-placed teams routed to the Round of 32 via the official Annex C table ->
knockouts with extra time and penalty shootouts -> champion. Aggregated over many
runs into stage-by-stage and champion probabilities.

Determinism: pass a fixed seed. The same seed + same inputs reproduce the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, elo as elo_mod, model as model_mod
from .rules import Match, compute_stats, rank_group, select_best_thirds

# Knockout tree (verified from bracket.json progression). Each entry:
# match_no -> (source_match_a, source_match_b). R32 (73-88) resolved from groups.
KNOCKOUT_TREE: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}
R16_MATCHES = range(89, 97)
QF_MATCHES = range(97, 101)
SF_MATCHES = (101, 102)
FINAL_MATCH = 104

_MX_CITIES = {"Guadalupe", "Mexico City", "Monterrey", "Guadalajara"}
_CA_CITIES = {"Toronto", "Vancouver"}


def _r32_country(city: str | None) -> str:
    if city in _MX_CITIES:
        return "Mexico"
    if city in _CA_CITIES:
        return "Canada"
    return "United States"


def decide_knockout(a, ea, aa, b, eb, ab, params, rng, mult_a=1.0, mult_b=1.0,
                    h2h_delta=0.0) -> str:
    """Resolve a knockout match (no draws): 90 min, then extra time, then a
    near-coin-flip shootout with a small strength tilt. Pure and testable."""
    la, lb = model_mod.match_lambdas(ea, eb, params, aa, ab, h2h_delta)
    la *= mult_a
    lb *= mult_b
    ga, gb = rng.poisson(la), rng.poisson(lb)
    if ga != gb:
        return a if ga > gb else b
    ga += rng.poisson(la * config.EXTRA_TIME_SCALE)
    gb += rng.poisson(lb * config.EXTRA_TIME_SCALE)
    if ga != gb:
        return a if ga > gb else b
    # a shootout is a near coin-flip on skill alone; home/venue advantage does not
    # carry into penalties, so tilt on raw Elo (no host bump).
    p_a = 0.5 + config.SHOOTOUT_FAVOURITE_TILT * (2 * elo_mod.expected(ea, eb) - 1)
    p_a = min(0.65, max(0.35, p_a))
    return a if rng.random() < p_a else b


@dataclass
class Tournament:
    teams: dict[str, dict]              # name -> {elo, group, fifa_rank}
    groups: dict[str, list[str]]        # letter -> [team names]
    group_fixtures: list[dict]          # {group, home, away, venue_country}
    r32: list[dict]                     # bracket.json r32 entries
    routing: dict[str, dict]            # combo -> {host_slot: group_letter}
    params: model_mod.ModelParams = field(default_factory=model_mod.ModelParams)
    host_bump: float = config.HOST_BUMP_ELO
    attack_mult: dict = field(default_factory=dict)     # team -> attacking-xG multiplier
    played: dict = field(default_factory=dict)          # (home, away) -> (gh, ga) for completed group matches
    h2h: dict = field(default_factory=dict)             # (home, away) -> head-to-head supremacy delta

    def fifa_ranks(self) -> dict[str, int]:
        return {n: (d.get("fifa_rank") or 9999) for n, d in self.teams.items()}

    def mult(self, team: str) -> float:
        return self.attack_mult.get(team, 1.0)

    def h2h_delta(self, home: str, away: str) -> float:
        return self.h2h.get((home, away), 0.0)


_STAGES = ["win_group", "top2", "advance", "r16", "qf", "sf", "final", "champion"]


def simulate(tournament: Tournament, n_runs: int = 10_000, seed: int = config.DEFAULT_RNG_SEED) -> dict:
    t = tournament
    if not t.teams or len(t.groups) != len(config.GROUP_LETTERS) or any(
        len(v) < 3 for v in t.groups.values()
    ):
        raise ValueError(
            "simulate() needs 12 groups of >=3 teams with ratings; "
            "the database looks empty or un-ingested. Run: python -m wc26.ingest"
        )
    rng = np.random.default_rng(seed)
    counts = {name: dict.fromkeys(_STAGES, 0) for name in t.teams}
    fifa = t.fifa_ranks()

    # group fixtures grouped by letter for speed
    fixtures_by_group: dict[str, list[dict]] = {g: [] for g in t.groups}
    for fx in t.group_fixtures:
        fixtures_by_group[fx["group"]].append(fx)

    def adv(team: str, venue_country: str | None) -> float:
        return t.host_bump if venue_country and team == venue_country else 0.0

    def sample_goals(home: str, away: str, venue_country: str | None) -> tuple[int, int]:
        if (home, away) in t.played:                  # conditional sim: fix completed matches
            return t.played[(home, away)]
        la, lb = model_mod.match_lambdas(
            t.teams[home]["elo"], t.teams[away]["elo"], t.params,
            adv(home, venue_country), adv(away, venue_country), t.h2h_delta(home, away),
        )
        la *= t.mult(home)
        lb *= t.mult(away)
        return int(rng.poisson(la)), int(rng.poisson(lb))

    def knockout_winner(a: str, b: str, venue_country: str | None) -> str:
        return decide_knockout(
            a, t.teams[a]["elo"], adv(a, venue_country),
            b, t.teams[b]["elo"], adv(b, venue_country), t.params, rng,
            mult_a=t.mult(a), mult_b=t.mult(b), h2h_delta=t.h2h_delta(a, b),
        )

    for _ in range(n_runs):
        winners: dict[str, str] = {}
        runners: dict[str, str] = {}
        thirds: dict[str, str] = {}            # group letter -> third team
        third_stats = {}
        for g, teams in t.groups.items():
            matches = []
            for fx in fixtures_by_group[g]:
                gh, ga = sample_goals(fx["home"], fx["away"], fx.get("venue_country"))
                matches.append(Match(fx["home"], fx["away"], gh, ga))
            order = rank_group(teams, matches, fifa)
            winners[g], runners[g], thirds[g] = order[0], order[1], order[2]
            stats = compute_stats(teams, matches)
            third_stats[order[2]] = stats[order[2]]
            counts[order[0]]["win_group"] += 1
            counts[order[0]]["top2"] += 1
            counts[order[1]]["top2"] += 1

        # best 8 thirds and their group letters
        best_thirds = select_best_thirds(third_stats, fifa, n=config.N_BEST_THIRDS)
        third_team_to_group = {thirds[g]: g for g in t.groups}
        qualifying_letters = sorted(third_team_to_group[tt] for tt in best_thirds)
        combo = "".join(qualifying_letters)
        routing_map = t.routing.get(combo, {})
        third_by_group = {g: thirds[g] for g in qualifying_letters}

        def resolve_slot(slot: str, home_slot: str) -> str | None:
            if slot.startswith("3rd"):
                letter = routing_map.get(home_slot)
                return third_by_group.get(letter) if letter else None
            pos, g = slot[0], slot[1]
            return winners[g] if pos == "1" else runners[g]

        # Round of 32
        W: dict[int, str] = {}
        advanced: set[str] = set()
        for m in t.r32:
            home = resolve_slot(m["home_slot"], m["home_slot"])
            away = resolve_slot(m["away_slot"], m["home_slot"])
            if home is None or away is None:
                continue
            advanced.add(home)
            advanced.add(away)
            W[m["match"]] = knockout_winner(home, away, _r32_country(m.get("city")))
        for team in advanced:
            counts[team]["advance"] += 1

        # Knockout tree (neutral for host advantage from R16 on)
        for match_no, (sa, sb) in KNOCKOUT_TREE.items():
            a, b = W.get(sa), W.get(sb)
            if a is None or b is None:
                continue
            W[match_no] = knockout_winner(a, b, None)

        for m in t.r32:                              # winners of the 16 R32 matches reached the last 16
            mid = m["match"]
            if mid in W:
                counts[W[mid]]["r16"] += 1
        for mno in R16_MATCHES:                       # winners of R16 reached the last 8
            if mno in W:
                counts[W[mno]]["qf"] += 1
        for mno in QF_MATCHES:                        # winners of QF reached the last 4
            if mno in W:
                counts[W[mno]]["sf"] += 1
        for mno in SF_MATCHES:                        # winners of SF reached the final
            if mno in W:
                counts[W[mno]]["final"] += 1
        if FINAL_MATCH in W:
            counts[W[FINAL_MATCH]]["champion"] += 1

    probs = {
        name: {s: counts[name][s] / n_runs for s in _STAGES} for name in t.teams
    }
    return {"n_runs": n_runs, "seed": seed, "probs": probs}
