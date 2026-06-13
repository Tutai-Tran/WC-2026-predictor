# WC-2026-predictor — Improvement Plan (synthesis of 8 expert reports)

Author: Synthesis Lead. Date: 2026-06-14. Source reports: `raw-findings.md` (same dir).
Scope: make the model materially more accurate, fix the two user complaints
(scoreline-collapse, forecast-too-far-from-reality), and make the always-on agents
coherent and genuinely self-learning. All paths are real and verified against the repo.

The eight reports converge hard. The four diagnoses independently reproduce the same
symptoms from the live DB; the four research reports independently point at the same
fixes. Where reports disagree, this plan adjudicates explicitly (see notes in the
backlog). Use the repo venv for everything: `/Users/tutaitran/wc26/.venv/bin/python`.

---

## DIAGNOSIS — ranked root causes (with evidence)

The two complaints are NOT one problem. Complaint #1 ("too far from reality") is a
**calibration + missing-signal** problem; complaint #2 ("only 1-0/2-0") is **80% a
reporting bug and 20% a too-low base_goals**. They are fixed by different levers.

### RC1 (dominant for complaint #1): published probabilities are NEVER blended with the market

The headline per-match `p_home/p_draw/p_away` are raw model outputs. D2 and R3 both
confirm: `group_match_forecasts` (forecast.py:126-153) emits `model.match_forecast`
directly, no blend, no shrink. The market blend exists **only** for the champion outright
and is published as a *separate* `champion_blend` field at 50/50 (forecast.py:328-332,
`blend_weight: 0.5`), so the un-blended model number can still surface. The h2h match
market is already fetched (`odds.fetch_match_odds`, odds.py:52-65) but never persisted or
blended.

Evidence of the damage: latest snapshot has the model **+9.0pp hot on the single top
team** (Spain model 24.2% vs devigged market 15.2%; Argentina +3.2pp). Both lambdas
derive from one Elo supremacy with no attack/defence prior, so a high-Elo team compounds
its edge across a 7-game simulated bracket (D2, D3). The literature is unanimous that the
devigged (closing) market line is the strongest single signal and beats a results-only
Elo on log-loss (R1: PLOS ELO-Odds 1.3913 vs ELO-Result 1.4032; R3: blending beats Platt;
Opta blends odds+ratings; Groll: bookmaker abilities are the 2nd-most-important variable).

### RC2 (dominant for complaint #2): the headline reports a single outcome-restricted modal cell

`forecast.py:139` builds the headline from `most_likely_scoreline(matrix, modal_outcome)`
(model.py:119-137), which argmaxes the matrix **masked to the winner's triangle**. D1
proves the trap: for a Poisson(lambda) loser, P(0) > P(1) iff lambda < 1.0, and with
base_goals=2.19 the weaker side's lambda < 1.0 in **60/72 fixtures**, so the loser can never
score in the headline and the cell collapses to (k,0). Result: across all 72 group
fixtures the headline takes **only 4 values, all clean sheets** (1-0 x25, 0-1 x23, 2-0
x16, 0-2 x8 — D1, D2). Yet R2/R3 show **no exact score ever exceeds ~16% probability** —
the mode of a flat distribution is a near-useless statistic. The full matrix is already
informative (D1: median fixture top-5 = 1-1 .138, 1-0 .128, 0-0 .113, 0-1 .093, 2-0 .088;
O/U 2.5 = 0.39/0.61; BTTS = 0.46/0.54). `model.top_scorelines` already exists
(model.py:89-96) but is not used for the headline.

### RC3 (amplifies both): base_goals=2.19 is too low, adopted by a WDL-only objective blind to total goals

Real total-goals averages (D1, from results.csv): WC since 2010 = 2.81, neutral since
2010 = 2.79, all since 2018 = 2.73. Model-implied mean total = **2.50** — under-predicts
by ~10%. Model P(BTTS)=0.434 vs real 0.466. The backtest objective optimizes only W/D/L
log-loss (backtest.py:107, 209-227), which is near-insensitive to total goals, so it
picked 2.19 for a **0.3% WDL gain** (2.19->0.8597 vs 2.6->0.8629) at the cost of totals
realism and the all-clean-sheet headline. Raising base_goals to ~2.6 restores the loser's
lambda >= 1.0 in 39/72 fixtures and, on its own, brings back 2-1/1-2/2-2 modal scores (D1:
2-0 x19, 2-1 x15, 1-2 x12, ...). R1 and R4 concur (raise toward 2.5–2.7); R3 cautions:
do it as a goal-volume/scoreline fix, not a W/D/L-calibration fix.

### RC4 (the real source of the residual calibration error): single-supremacy Elo with no team attack/defence and no xG/value signal

Both lambdas are a deterministic function of one number, the Elo gap (model.py:43-52; D3
traces every feature). There are no team-specific attack/defence strengths and no
xG/shot-quality or squad-value signal anywhere (D3: grep returns 0 hits). A results-only
Elo over-rates teams on a hot finishing/result streak (D3, R1, R4), which is the
mechanical cause of the elite-vs-elite overconfidence D1 measured: **both teams Elo>2000
(n=189), predicted fav-win 0.470 vs observed 0.418 (+5.2pp), predicted draw 0.279 vs
observed 0.328 (-4.9pp)**. This is the hard ceiling on accuracy and on scoreline
diversity (every even match has identical lambdas and the same 1-1/1-0 mode). Note: at the
*aggregate* W/D/L match level the model is well-calibrated (D2: reliability slope 1.003,
favourite overconfidence <= +1.9pp); the miss is concentrated in the elite cluster and at
the champion level — which is exactly the deep-knockout class the user watches.

### RC5 (calibration plumbing is broken/dead): temperature is fit-on-train and never applied

`temperature=0.996` is dead code: it is computed in backtest.py (fit_temperature line 163,
295) but **never read** in model.py / forecast.py / simulate.py / update.py / learn.py
(D2, D4). Worse, T<1 *sharpens* probabilities — the wrong direction for an overconfident
model (R1). And it is fit on `train` (backtest.py:295), the same data the params were fit
on, so it collapses to ≈1.0 and never learns the out-of-sample overconfidence (R3) — a
leak that is the proximate reason in-tournament Brier (~0.45) >> friendly Brier (~0.30).

### RC6 (self-learning is frozen AND non-durable): the adoption gate never opens, and CI clobbers anything it adopts

D2 and D4 both confirm `model_params_log` has **0 rows** — the gate has never fired.
Live reason: `insufficient graded matches (tournament: 4/12)` (_MIN_EVAL_GROUP=12,
learn.py:487). The corrections the user wants have **already crossed quorum** (`elo_gap
over` n=5, `draw under` n=5) but are hard-blocked by min_eval. Across 104 matches the gate
can fire ~6–8 times, one param by one fixed step each — too slow to matter in-event. And
the one knob most tied to favourites-overconfidence, `temperature`, is not even in the
gate's tunable set (learn.py:495). Compounding this, **CI destroys learning**: forecast.yml
runs `python -m wc26.backtest` (= `run(write=True)`) which unconditionally rewrites c,
base_goals, rho, goal_scale, home_adv_elo, temperature from the static historical fit
(backtest.py:333-334), preserving only rho_friendly, then `git add data/raw/fitted_params.json`
and pushes to main (forecast.yml:54). launchd rebases onto origin/main and pulls the
reverted params back — any in-tournament adoption is erased within 24h (D4).

### RC7 (secondary, self-healing): grading lag

D3 live-tested the ESPN scraper — it is healthy and returns completed results minutes
after full-time. The lag is the ~3.5h heartbeat cadence (schedule.py:25
HEARTBEAT_HOURS=4), a bounded 0–3.5h grading delay, NOT stale-input "far from reality."
Lowest priority of the root causes.

---

## PRIORITIZED BACKLOG (ordered by accuracy-impact-per-effort)

Effort key: XS (<2h), S (~half day), M (1–2 days), L (multi-day). Every item has a
backtest-measurable gate; nothing is adopted without passing the harness in the last
section. **TOP 5 QUICK WINS = items 1–5; do these first.**

---

### 1. [TOP] Blend the devigged market into the PUBLISHED match probabilities (log-pool, fitted weight)

(a) **What & why.** Replace the raw published `p_home/p_draw/p_away` with a blend of the
model probs and the devigged h2h market. This is the single biggest accuracy win (RC1):
it cools the +9pp-hot favourites toward the sharp market without touching model structure.
The h2h market is already fetched (odds.py:52-65) but neither persisted nor blended.

(b) **Files/functions.**
- `wc26/odds.py`: add `devig_h2h(events)` (mirror `devig_outrights`, odds.py:68-81, but
  3-way per fixture and Shin-devig via the existing `edge.py` helper); add
  `store_h2h_benchmark(conn, fixture_probs)` writing `scope='match'` rows into
  `public_benchmark` (the table already exists; only `scope='champion'` is used today);
  add `latest_market_h2h(conn)` reader; add `blend_wdl(model_probs, market_probs, weight)`
  using a **log (geometric) pool**, not the linear `blend_probs`.
- `wc26/forecast.py:126-153` (`group_match_forecasts`) and `:185-208`
  (`knockout`/friendly match emit): after computing `ph,pdw,pa`, if a market line exists
  for the fixture, replace with `odds.blend_wdl(...)`; keep the raw model probs in a
  `p_home_model/...` sub-field for the postmortem audit trail (so grading can still see
  model-only error).
- `wc26/update.py`: call `odds`'s h2h fetch+store in the cycle (guard on quota and on
  THE_ODDS_API_KEY presence; skip cleanly when absent, exactly like the existing champion
  path).

(c) **Math.** Log/geometric opinion pool (R3 B1, the internally-consistent pool for a
log-loss objective): `p_k ∝ m_k^w · q_k^(1-w)` for k in {H,D,A}, then renormalise.
`w` = model share. Adjudication: R3 says log-pool; D2 says weight ~0.3–0.4. We use the
**log pool at a fitted weight** (item 4 fits w; ship a defensible prior **w=0.35 model /
0.65 market** until fitted). When no market line exists for a fixture (most group games
until odds open), fall back to pure model — never block the forecast on odds.

(d) **Success criterion (backtest).** On the historical odds-matched join (matches with
both a model prob and a devigged closing line), the log-pool blend at the fitted w must
cut **test log-loss by >= 0.05** and **Brier by >= 0.03** vs model-only, and must not lose
to a market-only baseline by more than 0.005 log-loss. Target: in-tournament Brier
trajectory toward 0.30–0.33, log-loss toward 0.62–0.66 (R3 estimate). Add the
odds-matched slice to `backtest_report.json`.

(e) **Effort.** S (plumbing) + the weight-fit is item 4.

(f) **Risk / leak-freeness.** Risk: per-match odds quota on The Odds API — mitigate by
fetching h2h only for fixtures inside the next ~48h and caching. Leak-free: w is fit on
*historical* odds only with a temporal split (item 4); live blending uses only odds
available **before** kickoff (the snapshot is frozen pre-match by `snapshot_upcoming`,
learn.py:43, before results enter — preserve that ordering). Stability: log pool with
w>=0.3 cannot produce degenerate 0/1 probs.

---

### 2. [TOP] Report the scoreline DISTRIBUTION, not a single modal cell

(a) **What & why.** Kills complaint #2 directly with zero model change (RC2). The mode of
a flat low-mean Poisson is uninformative (<=16% mass); the matrix already holds everything
the user wants. All four relevant reports agree this is the cheapest highest-leverage fix
for #2 (D1, D2, R2, R3).

(b) **Files/functions.**
- `wc26/model.py`: add `derived_markets(matrix)` returning over/under 2.5, BTTS
  (`1 - P(h=0) - P(a=0) + P(0,0)`), total-goals bands (0-1 / 2 / 3 / 4+), and clean-sheet
  probs — each a sum over the existing matrix (R2 F). Add `expected_score(la, lb)` =
  `(round(la), round(lb))`.
- `wc26/forecast.py:140-149` and `:200-207`: keep `top_scoreline` for backward compat but
  ADD `top_scorelines` (use existing `model.top_scorelines(matrix, 5)` with masses),
  `top_scoreline_p` made explicit, `expected_score`, and the `derived_markets` dict to the
  per-match dict.
- `wc26/vaultgen.py`: render top-5 scorelines with probabilities + O/U + BTTS + expected
  score instead of (or above) the lone modal cell, and label the modal score with its
  ~12–15% probability so it reads as "most likely *single* score," not "the prediction."

(c) **Math.** Pure aggregation over the existing normalised matrix; no new parameters.

(d) **Success criterion.** Reporting change, so the gate is a **distributional sanity
check**, not a scoring-rule delta: (i) headline scoreline *set* across 72 group fixtures
expands from 4 distinct values to >= 10; (ii) top-5 cumulative mass reported per fixture is
0.50–0.62 (matches D1's measured 0.561); (iii) BTTS/O-U mean across fixtures within ±0.02
of the matrix-implied values (regression test). No W/D/L number may change (it is the same
matrix).

(e) **Effort.** XS.

(f) **Risk.** Near-zero — additive fields, no model change. Keep the old field name so
nothing downstream breaks; add new fields alongside.

---

### 3. [TOP] Un-leak temperature: fit on a held-out fold AND actually apply it

(a) **What & why.** RC5. Today temperature is fit-on-train (collapses to ≈1.0) and never
applied. Fixing the leak makes T learn the true out-of-sample overconfidence (R3 expects
T>1, de-sharpening); applying it flattens overconfident favourites globally and cheaply.
This is R3's "cheapest single win."

(b) **Files/functions.**
- `wc26/backtest.py:295`: fit `temperature` on a **held-out validation fold** (a slice of
  the test window not used to fit goal_scale, or k-fold by tournament), not on `train`.
  Also move `goal_scale` (backtest.py:300-306) to that same validation fold to remove the
  minor in-sample touch D2 flagged.
- `wc26/model.py`: add an optional `temperature` to `ModelParams` and apply it in
  `match_forecast` AFTER `outcome_probs`: `p_k ∝ p_k^(1/T)`, renormalise. (Apply to W/D/L
  only, after the matrix is built, so scorelines are unaffected.)
- `wc26/forecast.py`: ensure `ModelParams` is constructed with the loaded `temperature`
  (it currently isn't passed through — D2 confirms it's never read).

(c) **Math.** Temperature scaling `p_cal ∝ p^(1/T)` (Guo et al.; R1, R3). No bias term, so
it cannot flip the argmax — the displayed W/D/L winner never contradicts the headline.

(d) **Success criterion.** On the held-out fold, calibrated test log-loss must improve
over uncalibrated by a real margin (`log_loss_calibrated < log_loss`, which is currently a
tie at 0.8597 — D2). Fitted T must be > 1.0 (de-sharpening) or the change is rejected as a
no-op. On the elite-vs-elite slice (both Elo>2000), the favourite-win overconfidence gap
must shrink from +5.2pp (D1) toward <= +3pp.

(e) **Effort.** S.

(f) **Risk.** Order-of-operations: temperature and the market blend (item 1) both de-
sharpen — do NOT double-count. Adjudication: **apply temperature first, then the market
blend** (R3's stack: matrix -> temperature -> log-pool blend). Re-fit w (item 4) on
*temperature-scaled* model probs so the two are jointly leak-free. Stability: bound T to
[0.8, 2.0] as fit_temperature already does (backtest.py:175).

---

### 4. [TOP] Fit the blend weight w leak-free (replaces the hard-coded 0.5) + EB shrinkage in-tournament

(a) **What & why.** RC1/RC6. The champion blend weight is hard-coded 0.5 (odds.py:116,
forecast.py:332); the literature says market-heavy ~0.3 (R3 B2). Fit it; do not re-fit it
on the handful of live WC matches (overfits noise — R3 D1).

(b) **Files/functions.**
- `wc26/backtest.py`: add `fit_blend_weight(feats_with_odds)` — golden-section
  `minimize_scalar(bounded)` over w in [0,1] minimising OOS log-pool log-loss (reuse the
  exact pattern at backtest.py:174-175). Write `blend_weight` into `fitted_params.json`.
- `wc26/forecast.py:332`: read `blend_weight` from params for BOTH the champion blend and
  the new match blend (item 1); stop hard-coding 0.5.
- `wc26/learn.py`: in-tournament, update w by **empirical-Bayes / beta-binomial
  shrinkage** with a large pseudo-count: `w_live = (n_prior·w0 + n_obs·w_obs)/(n_prior +
  n_obs)`, `n_prior ≈ 300` (R3 D1), so the first ~20 WC results barely move it.

(c) **Math.** 1-D convex search on OOS log-loss for w0; precision-weighted posterior for
w_live.

(d) **Success criterion.** Fitted w0 lands in [0.2, 0.45] (sanity); the fitted-w blend
beats the 0.5-blend on the odds-matched test slice by >= 0.01 log-loss; w_live moves by
<= 0.03 over the first 20 graded WC matches (shrinkage working, not thrashing).

(e) **Effort.** S–M.

(f) **Risk / leak.** Temporal split for w0; never fit on WC data. The EB pseudo-count is
the thrash guard. Keep it inside the same adoption-gate rollback discipline (item 8).

---

### 5. [TOP] Raise/refit base_goals to ~2.6 under a goal-aware objective

(a) **What & why.** RC3. Fixes the ~10% goal under-prediction and is the *model-side*
half of the scoreline-diversity fix (item 2 is the reporting half). Adjudication: R3 warns
not to raise base_goals for *W/D/L calibration* reasons — so we change the **objective**
that picks it, not just the value.

(b) **Files/functions.**
- `wc26/backtest.py:209-227` (`fit`): change the fit objective for `base_goals` (and only
  base_goals) from pure W/D/L log-loss to a **combined objective**: W/D/L log-loss + a
  total-goals calibration penalty `lambda_tot · (mean_pred_total - mean_actual_total)^2`
  (or a Poisson deviance on totals). This lets base_goals find the value that matches real
  goal volume (~2.73–2.81) without W/D/L regressing.
- Leave `c`, `gamma`, `rho` on their existing W/D/L objective (they are well-fit for
  outcomes — D1, D2).

(c) **Math.** Multi-term objective; `lambda_tot` tuned so total-goals mean lands within
±0.05 of actual while W/D/L log-loss rises by <= 0.003 (the cost D1 measured for 2.19->2.6).

(d) **Success criterion.** Model mean total goals moves from 2.50 to within ±0.05 of the
WC/neutral actual (2.79–2.81); model BTTS moves from 0.434 toward 0.46; **W/D/L test
log-loss does not rise by more than 0.004**; headline scoreline-set diversity (item 2's
metric) increases further. If W/D/L log-loss regresses beyond the cap, reject and keep the
lower base_goals (item 2's reporting fix already addresses complaint #2 on its own).

(e) **Effort.** S.

(f) **Risk.** The known tradeoff (D1) is a tiny WDL cost for a large totals gain; the cap
in (d) bounds it. Leak-free: same temporal train/test split.

---

### 6. Negative-Binomial marginal to tame the favourite's blowout tail

(a) **What & why.** RC4 (scoreline side). gamma=0.456 scales only |supremacy|, fattening
the favourite's lambda and over-producing 4-0/5-0 on mismatches (D1, R2 H; live P(>2.5)=66.6%
for top-vs-weak). A NegBin marginal adds overdispersion that pulls the extreme tail in
without changing the mean. R2 D shows family swaps are ~0.02% RPS overall, so this is
**conditional**: backtest it, adopt only if it helps the tail without hurting RPS.

(b) **Files/functions.** `wc26/model.py:65-78` (`scoreline_matrix`) — optional NegBin
marginals via `scipy.stats.nbinom` behind a param flag; `wc26/simulate.py:148-154` —
swap `rng.poisson` for `rng.negative_binomial` when the flag is on (one-line, R2 H).

(c) **Math.** NegBin with mean lambda, variance `lambda + lambda^2/r`; fit r on historical goals.

(d) **Success criterion.** RPS on the test set must not regress (<= +0.0002, the bake-off
noise floor); P(>2.5) on top-vs-weak fixtures drops by a measurable amount; calibration of
the 4+ total-goals band improves. **If RPS regresses at all, do not adopt** (R2's decisive
finding).

(e) **Effort.** S. (f) **Risk.** Low; gated entirely on no-RPS-regression. Skip bivariate
Poisson / ZIP / CMP — R2 shows them flat-or-worse.

---

### 7. Add squad market value (Transfermarkt) as an Elo regularizer

(a) **What & why.** RC4. The single most-cited missing feature in academic WC models
(R1: Groll #1 predictor; R4 #1 by accuracy-per-effort). Squad value de-correlates from
results-Elo enough to temper over-rated hot favourites — the structural cause of complaint
#1 that temperature/blend only patch post-hoc.

(b) **Files/functions.** New `wc26/squadvalue.py` ingesting the weekly Kaggle
`davidcariboo/player-scores` CSV (no scraping, dodges Cloudflare — R4 §2) into a new
`team_squad_value` table (joins on the existing `players` squads — D3, R4); in
`wc26/model.py:43-52` blend `log(squad_value)` into the supremacy input with a small,
shrunk weight, OR (simpler, lower-risk) a regularizer that pulls extreme Elo gaps toward
the mean when the value gap is smaller.

(c) **Math.** `eff_supremacy = (1-k)·elo_supremacy + k·value_supremacy`, k small (~0.15–
0.25), fit by backtest.

(d) **Success criterion.** On the historical test set, the elite-vs-elite favourite
overconfidence gap (D1's +5.2pp) shrinks AND overall test log-loss does not regress.
Adopt only if both hold.

(e) **Effort.** M. (f) **Risk.** Overfit on sparse data — keep k small and shrunk; data
freshness (weekly) is fine. Leak-free: value snapshots are pre-tournament.

---

### 8. Make the self-learning loop genuinely self-correcting in-tournament (gate reachability + temperature knob)

(a) **What & why.** RC6. The gate has the right out-of-sample + anti-ratchet discipline but
is tuned so conservatively it never opens (0 adoptions; the exact corrections wanted are
quorum-met but min_eval-blocked — D2, D4). And the one knob most tied to the complaint,
temperature, is unreachable.

(b) **Files/functions.**
- `wc26/learn.py:495-496`: add `temperature` to `_GATE_STEP` (e.g. ±0.03) and `_GATE_RANGE`
  (e.g. [0.8, 2.0]); add a factor mapping so an "elo_gap over" / "favourites overconfident"
  signal raises temperature toward de-sharpening (D4 Fix 2).
- `wc26/learn.py:487, 387`: lower `_MIN_EVAL_GROUP` 12 -> 8 (still > half a matchday) and
  `_QUORUM_GROUP` 5 -> 3 **for the global calibration params only** (temperature, small c
  step); keep 5 for structural params (base_goals, rho) (D4 Fix 3).
- Keep `_NET_MARGIN=0.005`, `_HIST_CAP=0.01`, `_MIN_NEW_BETWEEN_ADOPT` (the thrash guards
  — do NOT loosen these).

(c) **Math/approach.** Unchanged gate math; only the count thresholds and the tunable set
change. The eval set `_feats_2026` already pulls graded matches leak-free (learn.py:500,
D4 Fix 4) — no change needed there.

(d) **Success criterion.** In a replayed-history dry run, the gate adopts a temperature
nudge after ~one matchday of (synthetic) overconfident-favourite evidence, and every
adopted step still satisfies `ll_new < base_new and regress < _HIST_CAP and improve -
max(0,regress) > _NET_MARGIN` (learn.py:639-640). No adoption may raise historical
held-out log-loss by > _HIST_CAP.

(e) **Effort.** S–M. (f) **Risk.** Loosening counts risks thrash — bounded because the
out-of-sample re-fit gate and anti-ratchet are untouched, and rollback (learn.py:692)
still guards. Keep `_MIN_EVAL` (non-group) at 20.

---

### 9. Stage- and context-aware adjustments (knockout base_goals, rest/travel/altitude)

(a) **What & why.** RC4 (tournament realism). Knockouts are lower-scoring; 2026 has
extreme travel/altitude (R4 §4). Zero new data feed — computed from fixture venue coords +
dates already in the DB.

(b) **Files/functions.** New `wc26/context.py` computing per-team `rest_days`, `travel_km`,
`altitude_m`; a stage-conditional `base_goals` (group higher, knockout lower) in
`model.match_lambdas`; small Elo/lambda nudges. `wc26/simulate.py` adds a separate
extra-time goal rate + near-coin-flip penalties for knockouts (R1 §1; `shootout.py` exists
already — extend it).

(c) **Math.** Small additive lambda/Elo terms, each fit or bounded by backtest.

(d) **Success criterion.** Knockout-stage total-goals calibration improves vs a flat
base_goals on the historical knockout subset; no regression on group-stage log-loss.

(e) **Effort.** M. (f) **Risk.** Low; additive and bounded. Leak-free (geometry is known
pre-match).

---

### 10. Next-day xG-form correction (FBref via soccerdata) — strategic, last

(a) **What & why.** RC4 root fix. xG corrects an Elo-only lambda faster than scorelines
alone (R4 §1). Deliberately last: highest cost, ToS/rate-limit risk, and items 1+3+7
already absorb most of the overconfidence post-hoc.

(b) **Files/functions.** New `wc26/xg.py` (soccerdata FBref, hard-throttled to ~1 req/3s,
aggressive cache) -> rolling `team_xg_form` -> small shrunk weight on lambda in `model.py`.
StatsBomb open data (one-off) for historical attack/defence **priors** and to validate
base_goals/DC (R4 §1A) — note it is released months late, never live.

(c) **Math.** `lambda_adj = lambda · (1 + w_xg·(xg_form_z))`, w_xg small, shrunk.

(d) **Success criterion.** On a held-out historical slice with FBref xG, the xG-nudged
lambdas reduce test log-loss vs Elo-only without regressing totals. Adopt only if it beats
items 1+3+7 combined on the odds-matched slice.

(e) **Effort.** L. (f) **Risk.** ToS/rate-limit (throttle+cache); ~1-day latency (next-day
signal only — do not design the fast loop around it). Leak-free: only pre-match xG.

---

## KEEPING THE RUNNING AGENTS COHERENT (launchd + CI) AND SELF-LEARNING LEAK-FREE

The two always-on agents currently overwrite each other's intent (RC6, D4 §B). The fix is
to make **launchd the single source of truth for learned state during the tournament** and
demote CI to a deterministic-compute fallback. Required changes:

1. **CI must stop writing `fitted_params.json` during the tournament (the critical fix).**
   In `.github/workflows/forecast.yml`: drop `data/raw/fitted_params.json` from the
   `git add` (line 54) so CI commits only `WC vault` + `replayed_elo.json`; OR extend
   `backtest.run(write=...)` (backtest.py:308-334) to preserve gate-owned keys (c,
   base_goals, rho, temperature, blend_weight) the way it already preserves rho_friendly,
   when a newer `model_params_log` adoption exists. Cleanest: **during
   [TOURNAMENT_START, TOURNAMENT_END] the local adoption gate is the sole writer of
   fitted_params.json.** Without this, every backlog item that adopts a param is erased
   within 24h.

2. **Disable the scheduled CI job during the tournament; keep `workflow_dispatch`.**
   (D4 Fix 5a.) This makes launchd the single publisher of `WC vault` + params and removes
   the git race entirely (one writer). CI remains available for manual deterministic
   rebuilds from `data/raw`. Re-enable the daily cron after the tournament.

3. **Persist the learning substrate so it is not local-only.** `wc26.db` (prediction_log,
   postmortems, model_params_log, grading) is gitignored and lives only on the Mac. Either
   accept launchd-as-sole-publisher (item 2 makes this safe), or export the four learning
   tables to `data/raw` each cycle so a CI rebuild grades/adopts identically. Prefer item 2
   (smaller, removes the race).

4. **Preserve the leak-free snapshot ordering.** Everything above must keep the existing
   discipline: `snapshot_upcoming` (learn.py:43) freezes pre-match probs BEFORE results
   enter Elo; `adopt_adjustments` runs BEFORE the forecast so an adopted change sharpens
   the same cycle; the forecast skip-fingerprint includes `fitted_params.json` bytes so an
   adopted param busts the skip (D4 §A). The market blend (item 1) must blend only odds
   available pre-kickoff. Do not break this ordering.

Net coherent set (D4's recommendation, adopted): CI stops touching params/vault during the
tournament (1+2) + temperature is gate-tunable and thresholds are reachable (item 8). The
loop can then adopt a real, durable, forecast-changing calibration correction after ~one
matchday while keeping all leak-free and anti-ratchet guards intact.

---

## MEASUREMENT HARNESS — how each change is validated before adoption

No change ships without passing the harness. The harness is `backtest.py` plus two new
slices. **Never adopt a regression.**

### Baseline (capture once, before any change)
Run `/.venv/bin/python -m wc26.backtest` and freeze `data/raw/backtest_report.json` as
`backtest_report.baseline.json`. Current reference numbers (from the reports): test
log-loss 0.860, Brier 0.505, RPS, reliability slope 1.003; tournament-only (is_major,
n=1200) log-loss 0.9334, Brier 0.5515; elite-vs-elite (both Elo>2000) favourite gap
+5.2pp; mean total goals 2.50; BTTS 0.434; headline scoreline distinct-value count = 4.

### New slices to add to the report (small, leak-free)
- **Odds-matched slice** (for items 1, 3, 4, 10): historical internationals with both a
  model prob and a devigged closing line; reports model-only vs market-only vs blend
  log-loss/Brier on a temporal hold-out. This is the gate for all market-blend work.
- **Elite-vs-elite slice** (for items 3, 7): both teams Elo>2000; reports the favourite-win
  overconfidence gap. This is the gate for de-sharpening work.
- **Totals/scoreline slice** (for items 2, 5, 6, 9): mean total goals, BTTS, O/U-2.5
  calibration, total-goals-band calibration, and the headline scoreline distinct-value
  count across the 72 group fixtures.
- **Live bootstrap CI** (R3 D2): extend `bootstrap_log_loss_ci` (backtest.py:153) to the
  in-tournament slice so the user sees that the ~0.45 Brier ± a wide band overlaps 0.30 —
  separating real signal from small-sample noise. Use **RPS** as the primary in-tournament
  metric (order-aware, less noisy — R3 D2).

### Per-item gate (must hold on the temporal hold-out, never on WC data)
| Item | Primary gate | Hard guardrail |
|---|---|---|
| 1 market blend | blend log-loss <= model-only − 0.05 on odds slice | not worse than market-only by > 0.005 |
| 2 distribution report | distinct headline values >= 10; top-5 mass 0.50–0.62 | W/D/L numbers unchanged (identical matrix) |
| 3 temperature | log_loss_calibrated < log_loss; T > 1.0 | elite gap +5.2pp -> <= +3pp; no overall regression |
| 4 fit w | fitted-w blend <= 0.5-blend − 0.01; w0 in [0.2,0.45] | w_live moves <= 0.03 over first 20 WC matches |
| 5 base_goals | mean total within ±0.05 of actual; BTTS toward 0.46 | W/D/L test log-loss rise <= 0.004 |
| 6 NegBin | P(>2.5) on mismatches drops; 4+ band calib improves | RPS regress <= +0.0002 else reject |
| 7 squad value | elite gap shrinks | overall test log-loss not worse |
| 8 gate reachability | adopts in dry-run after ~1 matchday | every step satisfies _NET_MARGIN/_HIST_CAP |
| 9 context/stage | knockout totals calib improves | group log-loss not worse |
| 10 xG | beats items 1+3+7 on odds slice | totals not worse; throttle/cache verified |

### Adoption discipline
1. Capture baseline. 2. Implement one item. 3. Re-run `backtest.py` + the relevant new
slice. 4. Compare against the per-item gate table above. 5. Adopt only if the primary gate
passes AND no guardrail is breached. 6. The live gate (`learn.adopt_adjustments`) keeps its
existing out-of-sample re-fit + `_HIST_CAP` + anti-ratchet + rollback — the in-tournament
analogue of this harness. Every adopted param is logged to `model_params_log` (which must
go from 0 rows to non-zero once item 8 ships) for the audit trail.

---

## SUMMARY: what to do first

Ship items **1–5** (the top quick wins) as the first batch, in this dependency order:
**3 (un-leak + apply temperature) -> 1 (log-pool match blend) -> 4 (fit w on
temperature-scaled probs) -> 2 (distribution reporting) -> 5 (goal-aware base_goals)**,
each gated by the harness above. In parallel, land the coherence fixes (CI stops writing
params, disable scheduled CI during the tournament) so the learning is durable, and item 8
so the loop can actually adopt. Items 6, 7, 9, 10 are the deeper structural follow-ups that
attack the root cause (single-supremacy, no xG/value) once the cheap post-hoc fixes are
verified.

Expected outcome: complaint #2 dissolved immediately by item 2; complaint #1 materially
reduced by items 1+3+4 (in-tournament Brier ~0.45 -> ~0.30–0.33, log-loss ~0.75 -> ~0.62–
0.66 per R3), with items 5–10 closing the structural gap and the coherence fixes making the
self-learning durable and genuinely live.
