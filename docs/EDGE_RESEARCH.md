# Model-vs-market edge & the EK2024 "guaranteed win", explained

> **Disclaimer — PAPER / RESEARCH ONLY, NOT BETTING ADVICE.** This models
> hypothetical value on a *fictional* bankroll. It does not place bets, connect
> to any account, or automate any operator, and must never be wired to do so.
> Positive expected value still **loses over small samples** (high variance), the
> model is overconfident on extreme favourites and longshots, and operators
> **void, limit, and close accounts** for the patterns described here. Never stake
> money you cannot afford to lose. Help: Loket Kansspel 0800-0177 (NL),
> BeGambleAware.org, or your national gambling helpline.

Research notes for the WC-2026 system. Paper analysis only. The point is to
understand the market, and to turn the model into an honest value *signal* rather
than to exploit any platform.

## 1. What the EK2024 case actually was

The Spain–Albania example was **not a platform bug**. It was an ordinary
**arbitrage / dutch** across two complementary in-play selections whose prices
momentarily summed to a combined implied probability **below 100%**.

Once the score was 1-0 to Spain, two selections partition Spain's remaining
outcomes:

- "Spain score **exactly 1**" — true iff Spain add 0 more goals,
- "Spain score **over 1.5**" — true iff Spain add 1 or more goals.

These are **mutually exclusive and collectively exhaustive** for Spain's final
tally given they were already on 1. So exactly one of them must win — you cannot
lose both. Whether that pair *profits* depends only on the prices.

### The math

For decimal odds `o_i`, the bookmaker-implied probability of a selection is
`1/o_i`. For a complete set of mutually-exclusive outcomes, the **overround**
(a.k.a. booksum) is

```
B = Σ (1 / o_i)
```

A normal book has `B > 1` (the margin — the bookmaker's edge). An **arbitrage
exists exactly when `B < 1`**: the market is pricing the full set of outcomes at
a combined implied probability under 100%. Stake each selection in proportion to
`1/o_i` and every outcome returns the same payout:

```
stake_i = S · (1/o_i) / B          (S = total stake)
payout  = S / B                     (same whichever selection wins)
profit  = payout − S = S · (1/B − 1) > 0   iff B < 1
```

The user framed it as "I matched the payout": pick a target payout `P`, stake
`P / o_i` on each, and

```
total staked = Σ P / o_i = P · B
profit       = P − P·B = P (1 − B) > 0   iff B < 1
```

Plugging in the example (each leg set to pay ≈ €120, staking €70 and €17):

- `o_exact1 ≈ 120/70 = 1.714`, `o_over1.5 ≈ 120/17 = 7.06`
- `B = 1/1.714 + 1/7.06 ≈ 0.583 + 0.142 = 0.725`
- total staked `≈ 120 · 0.725 = 87`, payout `120`, **profit ≈ €33** regardless.

`B = 0.725 < 1` is the whole story: the two complementary in-play lines were, for
a moment, priced at a combined 72.5% — a 27.5% **negative margin**.

(`wc26.edge.dutch_for_target_payout` and `wc26.edge.arbitrage` reproduce this; see
`tests/test_edge.py::test_dutch_reproduces_ek2024_example`.)

## 2. Why a sub-100% book appears (and why it's fragile)

Bookmakers normally keep `B` comfortably above 1 on every market. Sub-100% across
*complementary* lines happens transiently because:

- **Live model lag / leg desync.** After a goal, "exactly N" and "over N.5" are
  repriced by the trading model, but not always at the same instant, and one leg
  can keep a stale price for seconds. During that window the pair can sum < 1.
- **Different markets, same event.** "Exact goals" and "over/under" are priced by
  partly separate models; their margins don't always reconcile in-running.
- **Suspensions lagging play.** A price that should have been pulled is briefly
  still live.

It is fragile precisely because it is a pricing error, not a designed offer.

## 3. Why operators void and limit it (and why this isn't a strategy)

- **Palpable-error / obvious-error clauses.** T&Cs let the operator void bets
  struck at clearly mispriced odds. Arbing a sub-100% book is the textbook case.
- **Latency / arb detection.** Repeated complementary-leg staking right after
  goals is a strong, easily-detected signature. It leads to stake limiting,
  max-bet reductions to cents, and account closure.
- **Withheld / clawed-back winnings.** Operators routinely refuse or reverse
  payouts on voided legs.
- **Legal / agreement risk.** It breaches the operator agreement; depending on
  jurisdiction and scale, deliberate exploitation of pricing errors can be
  treated as fraud.

So it is neither sustainable nor safe. The durable, legitimate version of "having
an edge" is below.

## 4. The legitimate edge: model-vs-market value

If your *probabilities* are better than the market's on some selection, then
betting that selection at the offered price is positive expected value — that's
real, and it's what a forecasting model is for. The module implements it as
**paper research**:

### EV is computed against the RAW offered price

The single most important correctness point: **expected value uses the offered
decimal odds (which include the margin), never the devigged "fair" probability.**

```
EV per unit = p_model · o_offered − 1        (positive ⇔ p_model > 1/o_offered)
```

Devigged market probabilities are only the *signal* — the market's own estimate,
used for the probability **edge** (`p_model − p_market`) and for shrinking the
model toward the market. They are not a price you can bet at.

### The market signal: aggregate in probability space, bet the best price

A subtle but decisive correctness point (raised in review): **never average
decimal prices across books.** By Jensen's inequality `1/mean(o) ≤ mean(1/o)`, so
averaging prices *understates* implied probability and inflates apparent value;
on a many-runner futures book the averaged booksum can even drop below 1, which
silently disables the Shin devig. Instead:

- **Signal:** `aggregate_market` takes the per-book consensus *implied
  probability* (`mean_books(1/o)`), which preserves the real booksum/margin, and
  `market_fair_probs` devigs that whole-book consensus (Shin by default for
  outrights). This is the fair line used for edge and shrink.
- **Price:** EV and staking use the **best** available raw price per selection
  (`best_odds`) — the price you could actually take — not an average no book
  offers.

### Devigging

`wc26.edge.devig` offers **proportional** (`fair_i = (1/o_i)/B`) and **shin**
(removes margin assuming a small insider fraction `z`, pulling longshots **down**
and favourites **up** to correct the favourite-longshot bias; reduces to
proportional with no margin). Shin is the default signal for outrights.

### Staking: fractional Kelly with layered guards

Full Kelly `f* = (p·b − q)/b` maximises log-growth but is brutal on estimation
error, so the tool stakes **quarter-Kelly** with:

- an **odds-scaled shrink** toward the market, `s(o) = base + k·ln(o)` (default
  0.25 + 0.18·ln o, clipped to 0.95): long prices — where `LESSONS.md` shows the
  model is least calibrated — are pulled hardest toward the market line;
- a **max-odds cap** (default 26.0) that drops outright longshots entirely (this
  is why the old Colombia 53.0 row no longer appears);
- a **minimum books** gate, and **minimum edge / EV** gates;
- a **per-bet stake cap** (5% of bankroll) and a **total-exposure cap** (25%,
  pro-rata scaling all stakes) — a coarse proxy for the fact that simultaneous
  WC bets are correlated (outrights are mutually exclusive), which independent
  per-bet Kelly would otherwise over-stake.

### Validating an edge: closing-line value (CLV)

Outcomes are noisy; the field-standard evidence that a bet was *good* is **CLV** —
did you beat the closing line? `edge.clv(entry_fair, closing_fair)` and the
`PaperLedger` record paper bets at strike and score mean CLV and % positive once
the closing line is known. Mean CLV > 0 over many bets is far stronger evidence of
a real edge than a small-sample P&L.

### How to read the output, honestly

- The table is most trustworthy in the **mid-range** where the model is calibrated
  (~83% friendly top-pick accuracy, reliability slope ≈ 1.0) and least trustworthy
  on extreme favourites/longshots. The tail guards above exist precisely because
  raw model "+EV" on a longshot is mostly overconfidence, not money.
- Averaged paper EV still overstates reality: you rarely get the best price in
  size, lines move, and outright limits are small. Treat it as a *ranking signal*,
  validated by CLV, not a P&L forecast.
- Variance is large. Even a genuine +EV edge loses over small samples; the paper
  bankroll and fractional Kelly model that, they don't remove it.

## 5. What this module is and is not

- **Is:** a paper analytics layer — probability-space market aggregation, devig
  (proportional/Shin), EV on the best price, odds-scaled-shrink + capped
  fractional-Kelly sizing, a portfolio exposure cap, a CLV paper ledger, a
  model-vs-market table for outrights **and** matches (h2h), and an arbitrage/dutch
  analyzer that *explains* pricing inefficiencies like the EK2024 case.
- **Is not:** wired to any account, does not place bets, does not automate
  anything against a live operator, and is not advice (`LIVE_BETTING_ENABLED` is a
  hard-`False` invariant, asserted in the CLI). It is for understanding the market
  and stress-testing the model's probabilities.

## 6. Open next steps (from the review panel, not yet built)

- **Persist the CLV ledger** to a `paper_bets` table and auto-backfill closing
  lines from the dated `public_benchmark` snapshots already stored.
- **Wire `match_value_report(conn)`** to the new `odds.fetch_match_odds` so the
  deeper, sharper match market is live (currently only outrights are fetched on a
  schedule).
- **Correlation-aware (simultaneous) Kelly** over mutually-exclusive outright sets,
  rather than the current per-bet Kelly + exposure-cap proxy.
- **Sharp-book consensus** (e.g. Pinnacle) for the signal instead of an equal-book
  consensus, and an EV haircut for un-gettable top-of-board prices.
