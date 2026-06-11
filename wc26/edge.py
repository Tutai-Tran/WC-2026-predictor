"""Model-vs-market edge analysis and paper value-betting research.

RESEARCH / PAPER ONLY. This module compares the model's calibrated probabilities
against devigged bookmaker odds to study market efficiency and *hypothetical*
expected value on a fictional bankroll. It never places a bet, never touches a
real account, never logs in to or automates any operator, and is not betting
advice. The patterns it surfaces (value, arbitrage, dutching) are routinely
voided and limited by operators.

Two correctness rules drive the whole module:

1. Expected value is computed against the *best raw offered* decimal odds (which
   carry the bookmaker margin), never against devigged "fair" probabilities.
2. The market *signal* (the fair line used for the probability edge and for
   shrinking the model) is built by aggregating in PROBABILITY space across books
   (so the genuine booksum/overround is preserved) and devigging that, never by
   averaging decimal prices (which destroys the margin by Jensen's inequality).
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------
# Responsible-gambling guardrails (see also docs/EDGE_RESEARCH.md)
# --------------------------------------------------------------------------

#: This module is analysis-only by design. There is no live-betting integration,
#: no account credentials, and no order-placement path. Do not add one.
LIVE_BETTING_ENABLED = False

DISCLAIMER = (
    "PAPER / RESEARCH ONLY - NOT BETTING ADVICE.\n"
    "This tool models hypothetical value on a FICTIONAL bankroll. It does not "
    "place bets, connect to any account, or automate any operator, and must "
    "never be wired to do so.\n"
    "Positive expected value still LOSES over small samples (high variance). The "
    "model is overconfident on extreme favourites and longshots. Operators void, "
    "limit, and close accounts for the patterns shown here.\n"
    "Never stake money you cannot afford to lose. If gambling stops being fun, "
    "seek help: Loket Kansspel 0800-0177 (NL), BeGambleAware.org, or your "
    "national gambling helpline."
)


def _assert_paper_only() -> None:
    """Self-attesting invariant: this module is paper/research only."""
    assert not LIVE_BETTING_ENABLED, (
        "edge.py is paper/research only; live betting is not supported by design."
    )


# --------------------------------------------------------------------------
# Implied probability, overround, devigging
# --------------------------------------------------------------------------

def implied_prob(decimal_odds: float) -> float:
    """Bookmaker-implied probability of a single decimal price (includes margin)."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def overround(odds: dict[str, float]) -> float:
    """Sum of implied probabilities across a market. >1 = margin, <1 = arbable."""
    if not odds:
        return 0.0
    if any(o <= 1.0 for o in odds.values()):
        raise ValueError(f"all decimal odds must be > 1.0; got {odds}")
    return sum(1.0 / o for o in odds.values())


def devig_proportional(odds: dict[str, float]) -> dict[str, float]:
    """Multiplicative (proportional) devig: fair_i = (1/o_i) / overround."""
    book = {k: 1.0 / v for k, v in odds.items()}
    total = sum(book.values()) or 1.0
    return {k: v / total for k, v in book.items()}


def devig_implied(implied: dict[str, float]) -> dict[str, float]:
    """Proportional devig of a dict of implied probabilities (sum normalised to 1)."""
    total = sum(implied.values()) or 1.0
    return {k: v / total for k, v in implied.items()}


def devig_shin(odds: dict[str, float], max_z: float = 0.4) -> dict[str, float]:
    """Shin (1992) devig: removes margin assuming a fraction ``z`` of insider money,
    which down-weights longshots relative to favourites and tends to be better
    calibrated than the proportional method. Solves z in [0, max_z] so fair
    probabilities sum to 1 by bisection. Reduces to proportional with no margin."""
    q = {k: 1.0 / v for k, v in odds.items()}
    b = sum(q.values())

    def fair_for_z(z: float) -> dict[str, float]:
        out = {}
        for k, qi in q.items():
            num = math.sqrt(z * z + 4.0 * (1.0 - z) * qi * qi / b) - z
            out[k] = num / (2.0 * (1.0 - z))
        return out

    # f(z) = sum(fair) - 1 decreases in z; if even z=0 sums <=1 (no margin), bail.
    if sum(fair_for_z(0.0).values()) <= 1.0:
        return devig_proportional(odds)
    lo, hi = 0.0, max_z
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sum(fair_for_z(mid).values()) > 1.0:
            lo = mid
        else:
            hi = mid
    fair = fair_for_z((lo + hi) / 2.0)
    total = sum(fair.values()) or 1.0
    return {k: v / total for k, v in fair.items()}


def devig(odds: dict[str, float], method: str = "proportional") -> dict[str, float]:
    """Devig a market by name. method = 'proportional' | 'shin'."""
    if method == "shin":
        return devig_shin(odds)
    if method == "proportional":
        return devig_proportional(odds)
    raise ValueError(f"unknown devig method: {method!r}")


# --------------------------------------------------------------------------
# Expected value and staking
# --------------------------------------------------------------------------

def ev_per_unit(prob: float, decimal_odds: float) -> float:
    """Expected profit per 1 unit staked at `decimal_odds` given true `prob`.
    EV = prob * odds - 1; positive iff prob > 1/odds."""
    return prob * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll (0 if not +EV).
    f* = (p*b - q)/b, b = odds - 1, q = 1 - p."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (prob * b - (1.0 - prob)) / b)


def effective_shrink(decimal_odds: float, base: float, odds_scale: float) -> float:
    """Shrink toward the market that GROWS with price: base + odds_scale*ln(odds),
    clipped to [0, 0.95]. Long prices (where the model is least calibrated) are
    pulled hardest toward the market line; mid prices least."""
    return max(0.0, min(0.95, base + odds_scale * math.log(max(decimal_odds, 1.0001))))


@dataclass(frozen=True)
class ValueBet:
    selection: str
    odds: float          # best raw offered decimal odds (EV basis)
    model_prob: float    # model probability after shrink toward market
    market_prob: float   # devigged market (fair) probability
    edge: float          # model_prob - market_prob (probability edge)
    ev: float            # expected profit per unit at the offered odds
    n_books: int         # number of books pricing this selection
    stake: float         # paper stake in (fictional) currency units


def value_bets(
    model_probs: dict[str, float],
    offered_odds: dict[str, float],
    *,
    market_probs: dict[str, float] | None = None,
    n_books: dict[str, int] | None = None,
    bankroll: float = 1000.0,
    kelly_frac: float = 0.25,
    min_edge: float = 0.02,
    min_ev: float = 0.0,
    shrink: float = 0.25,
    shrink_odds_scale: float = 0.18,
    max_odds: float | None = 26.0,
    min_books: int = 1,
    max_stake_frac: float = 0.05,
    max_total_exposure_frac: float = 0.25,
    devig_method: str = "proportional",
) -> list[ValueBet]:
    """Find positive-EV selections (paper) and size them with fractional Kelly.

    Signal vs price are kept separate:
      offered_odds  the BEST raw offered decimal odds per selection (EV basis).
      market_probs  the devigged fair line over the FULL market (signal). If None,
                    it is derived by devigging `offered_odds` -- only valid when
                    `offered_odds` is itself a complete, mutually-exclusive market
                    (e.g. a 3-way 1X2). For partial markets (outrights) the caller
                    MUST pass `market_probs` devigged from the full book.

    Tail and risk guards:
      shrink/shrink_odds_scale  odds-scaled shrink of the model toward the market.
      max_odds      drop selections priced longer than this (longshot trap guard).
      min_books     require at least this many books pricing the selection.
      min_edge/min_ev  probability-edge and EV gates.
      kelly_frac    fraction of full Kelly (0.25 = quarter-Kelly).
      max_stake_frac per-bet stake cap as a fraction of bankroll.
      max_total_exposure_frac  pro-rata scale ALL stakes down if their sum exceeds
                    this fraction of bankroll (a portfolio exposure cap; a coarse
                    proxy for the correlation between simultaneous bets).
    """
    common = {k: offered_odds[k] for k in offered_odds if k in model_probs}
    if not common:
        return []
    market = market_probs if market_probs is not None else devig(common, devig_method)
    nb = n_books or {}

    raw: list[ValueBet] = []
    for sel, odds in common.items():
        if sel not in market:
            continue
        if max_odds is not None and odds > max_odds:
            continue
        books = nb.get(sel, 1)
        if books < min_books:
            continue
        s = effective_shrink(odds, shrink, shrink_odds_scale)
        p = (1.0 - s) * model_probs[sel] + s * market[sel]
        mkt = market[sel]
        edge = p - mkt
        ev = ev_per_unit(p, odds)
        if edge < min_edge or ev <= min_ev:
            continue
        f = min(kelly_frac * kelly_fraction(p, odds), max_stake_frac)
        stake = bankroll * f
        if stake <= 0:
            continue
        raw.append(ValueBet(
            selection=sel, odds=odds, model_prob=round(p, 4), market_prob=round(mkt, 4),
            edge=round(edge, 4), ev=round(ev, 4), n_books=books, stake=stake,
        ))

    # Portfolio exposure cap: scale all stakes down pro-rata if over budget.
    total = sum(b.stake for b in raw)
    cap = bankroll * max_total_exposure_frac
    scale = cap / total if total > cap and total > 0 else 1.0
    bets = [ValueBet(**{**b.__dict__, "stake": round(b.stake * scale, 2)}) for b in raw]
    bets.sort(key=lambda b: b.ev, reverse=True)
    return bets


# --------------------------------------------------------------------------
# Arbitrage / dutching: descriptive market-inefficiency analysis (paper only)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArbResult:
    is_arb: bool
    overround: float            # sum of implied probabilities
    guaranteed_return: float    # profit per 1 unit total staked if arb (else <0)
    stakes: dict[str, float]    # split of a 1-unit total that equalises payout
    payout: float               # the locked payout per 1 unit total staked


def arbitrage(odds: dict[str, float], total_stake: float = 1.0) -> ArbResult:
    """Describe whether a complete set of mutually-exclusive selections is arbable.

    An arbitrage exists iff the overround (sum of 1/odds) < 1: the market prices
    the full set of outcomes at a combined implied probability below 100%, so a
    stake split in proportion to 1/odds *would* lock the same payout whatever
    happens. This explains why the EK2024 'exactly 1 goal' vs 'over 1.5 goals'
    pair paid regardless once it was 1-0: those two selections partition the
    remaining outcomes and were briefly priced at a combined < 100%.

    Descriptive analysis only. Operators void these under palpable-error clauses,
    limit and close accounts that pursue them; do not execute."""
    book = overround(odds)
    weights = {k: (1.0 / v) / book for k, v in odds.items()}
    stakes = {k: round(total_stake * w, 4) for k, w in weights.items()}
    payout = total_stake / book if book > 0 else 0.0
    return ArbResult(
        is_arb=book < 1.0,
        overround=round(book, 6),
        guaranteed_return=round(payout - total_stake, 6),
        stakes=stakes,
        payout=round(payout, 4),
    )


def dutch_for_target_payout(odds: dict[str, float], target_payout: float) -> dict[str, Any]:
    """Illustrate the stake split that WOULD equalise payout across a complete set
    of mutually-exclusive selections (the user's EK2024 'I matched the payout'
    framing): stake_i = target_payout / odds_i, profit = target_payout - total,
    which is positive iff the combined book (overround) is below 100%.

    Descriptive market analysis only; operators void these under palpable-error
    clauses - do not execute."""
    stakes = {k: round(target_payout / o, 2) for k, o in odds.items()}
    total = round(sum(stakes.values()), 2)
    return {
        "stakes": stakes,
        "total_staked": total,
        "payout_each_outcome": round(target_payout, 2),
        "guaranteed_profit": round(target_payout - total, 2),
        "overround": round(overround(odds), 6),
    }


# --------------------------------------------------------------------------
# Closing-line value (the field-standard validity check) on a paper ledger
# --------------------------------------------------------------------------

def clv(entry_fair_prob: float, closing_fair_prob: float) -> float:
    """Closing-line value in probability terms: closing_fair - entry_fair.

    Positive means you took a price implying a LOWER probability than where the
    market closed - i.e. you beat the close (got a longer price than it settled).
    Mean CLV > 0 over many bets is the best available evidence of a real edge,
    independent of small-sample variance in outcomes."""
    return closing_fair_prob - entry_fair_prob


@dataclass
class PaperBet:
    selection: str
    entry_odds: float
    entry_fair_prob: float
    model_prob: float
    stake: float
    closing_fair_prob: float | None = None
    result: bool | None = None  # True if the selection won

    @property
    def clv(self) -> float | None:
        if self.closing_fair_prob is None:
            return None
        return clv(self.entry_fair_prob, self.closing_fair_prob)

    @property
    def pnl(self) -> float | None:
        if self.result is None:
            return None
        return self.stake * (self.entry_odds - 1.0) if self.result else -self.stake


class PaperLedger:
    """In-memory paper-bet log for CLV and P&L research (no persistence, no account)."""

    def __init__(self) -> None:
        self.bets: list[PaperBet] = []

    def record(self, bet: PaperBet) -> None:
        self.bets.append(bet)

    def summary(self) -> dict[str, float | int]:
        graded = [b for b in self.bets if b.clv is not None]
        clvs = [b.clv for b in graded if b.clv is not None]
        settled = [b for b in self.bets if b.pnl is not None]
        pnls = [b.pnl for b in settled if b.pnl is not None]
        return {
            "n_bets": len(self.bets),
            "n_with_clv": len(clvs),
            "mean_clv": round(sum(clvs) / len(clvs), 4) if clvs else 0.0,
            "pct_positive_clv": round(sum(c > 0 for c in clvs) / len(clvs), 3) if clvs else 0.0,
            "n_settled": len(pnls),
            "pnl": round(sum(pnls), 2) if pnls else 0.0,
        }


# --------------------------------------------------------------------------
# Market aggregation from The Odds API events (probability-space consensus)
# --------------------------------------------------------------------------

def _collect_prices(events: list[Any], market_key: str) -> dict[str, list[float]]:
    """Per-selection list of raw decimal prices across all bookmakers, name-aliased."""
    from .odds import NAME_ALIASES
    prices: dict[str, list[float]] = {}
    for event in events:
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != market_key:
                    continue
                for oc in mk.get("outcomes", []):
                    price = float(oc["price"])
                    if price <= 1.0:
                        continue  # drop malformed/degenerate prices
                    name = NAME_ALIASES.get(oc["name"], oc["name"])
                    prices.setdefault(name, []).append(price)
    return prices


@dataclass(frozen=True)
class MarketLine:
    best_odds: float          # longest price across books (what you'd actually take)
    consensus_implied: float  # mean over books of 1/price (kept un-normalised)
    n_books: int


def aggregate_market(events: list[Any], market_key: str = "outrights") -> dict[str, MarketLine]:
    """Aggregate a market in probability space: per selection, the best price and
    the consensus implied probability (mean of per-book 1/price). The consensus
    booksum (sum of consensus_implied) preserves the real margin, so devigging it
    is valid -- unlike averaging decimal prices."""
    out: dict[str, MarketLine] = {}
    for name, ps in _collect_prices(events, market_key).items():
        implied = sum(1.0 / p for p in ps) / len(ps)
        out[name] = MarketLine(best_odds=max(ps), consensus_implied=implied, n_books=len(ps))
    return out


def market_fair_probs(agg: dict[str, MarketLine], method: str = "proportional") -> dict[str, float]:
    """Devigged fair line from a consensus-implied aggregate (the market signal)."""
    implied = {k: v.consensus_implied for k, v in agg.items()}
    if method == "shin":
        # Shin needs odds; reconstruct representative odds from consensus implied.
        return devig_shin({k: 1.0 / v for k, v in implied.items()})
    return devig_implied(implied)


def best_odds(agg: dict[str, MarketLine]) -> dict[str, float]:
    return {k: v.best_odds for k, v in agg.items()}


def n_books_map(agg: dict[str, MarketLine]) -> dict[str, int]:
    return {k: v.n_books for k, v in agg.items()}


# --------------------------------------------------------------------------
# Reports built from the live system (model probs + aggregated market)
# --------------------------------------------------------------------------

def model_champion_probs(conn: sqlite3.Connection) -> dict[str, float]:
    """Latest model champion probabilities from the stored forecast snapshot."""
    row = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    return {t: p["champion"] for t, p in json.loads(row["payload_json"])["probs"].items()}


def champion_value_report(
    conn: sqlite3.Connection,
    events: list[Any] | None = None,
    *,
    signal_method: str = "shin",
    **kwargs: Any,
) -> list[ValueBet]:
    """Paper value table for the outright (champion) market.

    Signal = devigged consensus-implied market (Shin by default, better on the
    longshot tail); EV = best raw price; full-book devig avoids partial-market
    bias. Pass `events` from `odds.fetch_winner_odds()`; if omitted, fetches live."""
    if events is None:
        from .odds import fetch_winner_odds
        events, _ = fetch_winner_odds()
    agg = aggregate_market(events, "outrights")
    model = model_champion_probs(conn)
    return value_bets(
        model, best_odds(agg),
        market_probs=market_fair_probs(agg, signal_method),
        n_books=n_books_map(agg),
        **kwargs,
    )


def upcoming_matches(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Unplayed matches from the latest forecast snapshot (for match_value)."""
    row = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='forecast' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return []
    matches = json.loads(row["payload_json"]).get("matches", [])
    return [m for m in matches if not m.get("result")]


@dataclass(frozen=True)
class MatchValueBet:
    match: str
    date: str | None
    selection: str        # home | draw | away
    odds: float
    model_prob: float
    market_prob: float
    edge: float
    ev: float
    n_books: int
    stake: float


def match_value(
    matches: list[dict[str, Any]],
    market_odds: dict[str, dict[str, float]],
    **kwargs: Any,
) -> list[MatchValueBet]:
    """Paper value table for match (1X2 / W-D-L) markets. A 1X2 market is complete,
    so devigging the three offered prices is a valid market signal.

    `matches`     forecast payload match list (home, away, p_home/p_draw/p_away).
    `market_odds` "Home vs Away" -> {"home","draw","away": best decimal odds}."""
    rows: list[MatchValueBet] = []
    for m in matches:
        key = f"{m['home']} vs {m['away']}"
        odds = market_odds.get(key)
        if not odds or {"home", "draw", "away"} - set(odds):
            continue
        model = {"home": m["p_home"], "draw": m["p_draw"], "away": m["p_away"]}
        # 1X2 is a complete market: derive the signal by devigging the three prices,
        # and lift the outright-oriented longshot guard (these are not longshots).
        kw = {"max_odds": None, **kwargs}
        for b in value_bets(model, odds, **kw):
            rows.append(MatchValueBet(
                match=key, date=m.get("date"), selection=b.selection, odds=b.odds,
                model_prob=b.model_prob, market_prob=b.market_prob, edge=b.edge,
                ev=b.ev, n_books=b.n_books, stake=b.stake,
            ))
    rows.sort(key=lambda r: r.ev, reverse=True)
    return rows


def match_odds_by_fixture(events: list[Any]) -> dict[str, dict[str, float]]:
    """Shape The Odds API h2h `soccer_fifa_world_cup` events into
    {"Home vs Away": {"home","draw","away": best decimal odds}} using best price."""
    from .odds import NAME_ALIASES
    out: dict[str, dict[str, float]] = {}
    for event in events:
        home = NAME_ALIASES.get(event.get("home_team", ""), event.get("home_team", ""))
        away = NAME_ALIASES.get(event.get("away_team", ""), event.get("away_team", ""))
        if not home or not away:
            continue
        best: dict[str, float] = {}
        for bk in event.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                for oc in mk.get("outcomes", []):
                    name = NAME_ALIASES.get(oc["name"], oc["name"])
                    sel = "home" if name == home else "away" if name == away else "draw"
                    price = float(oc["price"])
                    if price > best.get(sel, 0.0):
                        best[sel] = price
        if {"home", "draw", "away"} <= set(best):
            out[f"{home} vs {away}"] = best
    return out


def main() -> None:
    """CLI: print the paper champion value table. Research only, never betting."""
    _assert_paper_only()
    print(DISCLAIMER + "\n")
    from . import db
    conn = db.connect()
    try:
        bets = champion_value_report(conn)
    except Exception as exc:  # network/key issues degrade gracefully
        print(f"Could not build champion value report: {exc}")
        return
    print("PAPER value analysis (champion market):\n")
    print(f"{'Team':<18}{'Odds':>8}{'Model':>8}{'Market':>8}{'Edge':>8}{'EV':>8}{'Bk':>4}{'Stake':>9}")
    print("-" * 71)
    for b in bets:
        print(f"{b.selection:<18}{b.odds:>8.2f}{b.model_prob*100:>7.1f}%"
              f"{b.market_prob*100:>7.1f}%{b.edge*100:>+7.1f}%{b.ev:>+8.3f}"
              f"{b.n_books:>4}{b.stake:>9.2f}")
    if not bets:
        print("(No positive-EV outrights after the tail guard and calibration shrink.)")
    print("\nEV uses the best raw offered odds; the signal is a Shin-devigged "
          "cross-book consensus; stakes are quarter-Kelly on a PAPER bankroll.")


if __name__ == "__main__":
    main()
