# WC-2026-predictor — Raw expert findings (audit trail)

Captured 2026-06-14. Eight expert reports: four code/DB diagnoses (D1–D4) and four
web-research reports (R1–R4). Verbatim, for the audit trail. The synthesized
improvement plan derived from these is in `PLAN.md` (same directory).

---

## D1 — diag:scorelines (scoreline distribution & goal model)

### Headline finding (the "1-0/2-0 only" symptom reproduced and explained)

Running all 72 WC2026 group fixtures through `forecast.group_match_forecasts`, the
headline `top_scoreline` takes only 4 values, ever:

- `1-0`: 25 (34.7%), `0-1`: 23 (31.9%), `2-0`: 16 (22.2%), `0-2`: 8 (11.1%)
- 100% of headlines are clean sheets. Never a 1-1, 2-1, 1-2, 2-2, or 3-1.

This is NOT primarily a "low-mean Poisson mode is structurally low" problem. It is
the product of three compounding causes.

### Root cause 1 (dominant): the outcome-restricted modal readout forces clean sheets

`forecast.py:139` calls `model.most_likely_scoreline(matrix, outcome_label(...))`
(model.py:119-137). This takes argmax over the matrix masked to the modal-outcome
triangle (e.g. home-win cells only, excluding the diagonal).

The structural trap: for a Poisson(lambda) loser, P(0 goals) > P(1 goal) iff lambda <
1.0. Measured: lambda=0.7 -> P(0)=0.497 vs P(1)=0.348; lambda=0.9 -> 0.407 vs 0.366. Only
at lambda>=1.0 does "1 goal" win. So whenever the weaker side's lambda < 1.0, the most-likely
loser-score is 0, and because the home-win mask also forbids the global mode (usually
1-1), the argmax collapses to (k, 0). Worked example, Mexico vs South Africa (la=2.34,
lb=0.63): the home-win cells rank 2-0(.140) > 1-0(.115) > 3-0(.109) > 2-1(.089). A 2-1
can never win because P(away=1) < P(away=0). The reported probability is tiny: P(top-1
exact) is only 0.114–0.168 even for the favourite.

### Root cause 2 (major): base_goals=2.19 is too low; refitting 2.6->2.19 worsened the symptom

Real total-goals averages from `results.csv`: all matches 2.94; since 2010 2.73; since
2018 2.73; World Cup since 2010 2.81; neutral-venue since 2010 2.79. Model-implied mean
total across WC fixtures: 2.50. Under-predicts by ~0.25–0.30 goals/match (~10%). Model
mean P(BTTS)=0.434 vs real intl BTTS=0.466; model P(over2.5)=0.455.

Counting fixtures where the weaker side's lambda >= 1.0 (the condition for a headline
loser-goal): base_goals=2.19 -> 12/72 (median weaker-lambda 0.82); base_goals=2.6 -> 39/72
(median 1.02); base_goals=2.8 -> 58/72 (median 1.12). Re-running headlines at
base_goals=2.6 yields a diverse distribution: 2-0x19, 2-1x15, 1-2x12, 0-2x12, 1-0x7,
0-1x5, 3-0x2.

Why the optimizer picked 2.19: on W/D/L test log loss, lower base is marginally better
(2.19->0.8597, 2.6->0.8629, 2.8->0.8654). The backtest objective (`backtest.py:107
log_loss`, fit at `backtest.py:209-227`) optimizes only W/D/L, which is nearly
insensitive to total goals; it is blind to scoreline/total-goals realism. An
objective-mismatch bug, not a modelling necessity.

### Root cause 3 (structural ceiling): single-supremacy goal model with no team attack/defence

Both lambda come from ONE Elo diff + a global base (`model.py:43-52`): la=(base+sup)/2,
lb=(base-sup)/2. A team's lambda is purely a function of the Elo gap. Every even match looks
identical (la≈lb≈base/2), so every even match produces the same 1-0 headline. No
mechanism for high-scoring even games (3-2, 2-2). Team-specific attack/defence (or a
bivariate Poisson, or xG-informed lambda) would let even matches differ and lift BTTS toward
0.47.

### Root cause 4 (confirms complaint #1): elite-vs-elite overconfidence

Full historical favourite calibration is excellent (gaps ±0.4–1.5pp). The
overconfidence is specifically elite-vs-elite: both teams Elo>2000 (n=189) ->
predicted favourite-win 0.470 vs observed 0.418 (gap +5.2pp), predicted draw 0.279 vs
observed 0.328 (gap -4.9pp). At least one team Elo>2050 (n=1136) -> fav-win gap +3.2pp.
A structural property of the single-supremacy Poisson between two strong evenly-matched
teams.

### Secondary parameter assessment

- rho (Dixon-Coles) = -0.06: useful but tiny; best ~-0.10 on the test set; does not fix
  the elite-draw under-prediction or change the headline.
- goal_scale = 1.0023: essentially a no-op; bounded [0.90,1.10] can't claw back the ~10%
  shortfall because base_goals already sits below reality.
- gamma = 0.456: does useful WDL work, but by design only scales |supremacy|, inflating
  the favourite's lambda (4-0/5-0 more likely), never the loser's — reinforces clean sheets.

### Recommended reporting (consumes existing matrix, no model change)

Full W/D/L + top-5 exact scorelines with masses + most-likely total-goals band +
over/under 2.5 + BTTS. `model.top_scorelines` already exists (model.py:89) but the
surfaced headline field uses the single restricted cell.

### Ranked root causes for "headline always 1-0/2-0"

1. Outcome-restricted single-cell readout (most_likely_scoreline, model.py:119 /
   forecast.py:139). Fix = report distribution. Cheapest, no model risk.
2. base_goals=2.19 too low (real ~2.73–2.81). Caused by a WDL-only fit objective blind
   to total goals.
3. Single-supremacy goal model, no team attack/defence (model.py:43-52). Hard ceiling.
4. gamma scales only |supremacy|: amplifies favourite blowouts without giving the loser
   goals.

Separately confirmed for complaint #1: elite-vs-elite overconfidence +5.2pp on the
favourite / -4.9pp on the draw (Elo>2000 both sides), invisible in aggregate backtest.

Files: model.py:20-27, :43-52, :55-78, :89-96, :119-137, :140-172; forecast.py:139,
:148; backtest.py:74-80, :107-113, :209-227, :300-306.

---

## D2 — diag:calibration (calibration, overconfidence & market blend)

### 1. Measured calibration: tournament vs friendly

Live graded log (`prediction_log`, n=28 graded):

| Segment | n | Brier | Log-loss | Pick acc |
|---|---|---|---|---|
| Friendly | 24 | 0.306 | 0.570 | 0.833 |
| Tournament (group) | 4 | 0.577 | 0.908 | 0.750 |

Live tournament Brier (0.577) ~89% worse than friendly (0.306), matching the complaint.
But n=4 is statistically meaningless — variance, not yet a proven defect.

Historical leak-free backtest (test 2021+, n=5584): Brier 0.505, LL 0.860, reliability
slope 1.003 (near-perfect); beats climatology (1.049) and favourite (0.942).
Tournament-only historical (is_major, n=1200): LL 0.9334, Brier 0.5515 vs climatology
1.064. Tournaments ARE genuinely harder historically too, but the model still beats both
baselines.

Overconfidence on favourites — measured, SMALL historically: <= +1.9pp in every bin
(worst Elo-gap 200–350 at +1.9pp); at extreme favourites it is slightly UNDER-confident
(-1.2pp). On tournament subset <= +0.7pp. Conclusion: at W/D/L match level the model is
well-calibrated historically. The "runs hotter than the market on top teams" is real but
lives at the champion/outright level, not the match level.

### 2. Temperature, shrinkage and market blend (biggest finding)

temperature=0.996 does literally nothing — and is dead code. It appears ONLY in
backtest.py (fit_temperature at line 295, written at 314). Never read in model.py,
simulate.py, forecast.py, update.py, or learn.py. Published probabilities are never
temperature-scaled. Inside the backtest, log_loss_calibrated (0.8597) == log_loss
(0.8597). Re-fitting T on the tournament subset gives optimal T = 1.027 (negligible
gain). Blending toward climatology makes tournament LL strictly worse. Global shrinkage
/ climatology blend is NOT the fix.

Does the model blend with devigged market odds for published probs? Almost not.
Match W/D/L probs (p_home/p_draw/p_away): NO blend, NO shrink (forecast.py:126-153).
edge.value_bets has an odds-scaled shrink and odds.blend_probs exists, but used only in
the paper value/edge path. Champion outright: blended, but only 50/50 and published as a
SEPARATE field (forecast.py:323-332).

Latest snapshot:

| Team | Model | Market (devig) | Model − Market | 50/50 Blend |
|---|---|---|---|---|
| Spain | 24.2% | 15.2% | +9.0pp | 19.7% |
| Argentina | 11.1% | 7.9% | +3.2pp | 9.5% |
| Colombia | 6.6% | (unpriced top8) | — | 4.1% |
| France | 13.1% | 15.6% | −2.5pp | 14.4% |

Model +9pp hot on Spain because both lambdas derive from one Elo supremacy with no
attack/defence priors, compounding across a 7-game bracket. Fix: publish the market
blend as the headline champion number, lower model weight (~0.3–0.4, not 0.5).

### 3. Scorelines "always 1-0/2-0" — confirmed structural

Published top_scoreline takes only 4 distinct group values (1-0 x25, 0-1 x23, 2-0 x16,
0-2 x8). Modal scoreline probability: mean 13.6%, max 16.9%. Actual graded scorelines
dominated by 2-1 x6, 1-1 x4, 1-2 x4 — scorelines the model structurally never emits as
modal. Presentation/structural artifact; fix at the headline layer.

### 4. Leak-freeness of the backtest

Genuinely leak-free. prematch_pass records PRE-match Elo before update_pair; temporal
train/test split at 2021. Two minor notes: goal_scale fit on the test window
(strictly should use a validation fold; impact tiny at 1.0023); the live adoption gate's
historical guard rebuilds over all results then filters year>=2021 (overlaps the test
window; acceptable guardrail).

### 5. Adoption gate too slow/strict to learn in-tournament

Gate: group mode needs _MIN_EVAL_GROUP=12 graded tournament matches, quorum
_QUORUM_GROUP=5, per-step gain _NET_MARGIN=0.005 LL, hist regression < _HIST_CAP=0.01,
_MIN_NEW_BETWEEN_ADOPT=12. Current state: model_params_log has 0 rows. Only 4 tournament
matches graded (< 12 min_eval), so hard-blocked. Postmortem evidence has ALREADY crossed
quorum: `elo_gap over` n=5 and `draw under` n=5 — the exact corrections wanted — but
cannot be adopted because n_eval=4<12. Across 104 matches the gate can fire ~6–8 times
total, each one parameter by one fixed step (c±8, base_goals±0.06, rho±0.015). Cannot
learn fast enough to matter within the event.

### Ranked fixes for "forecast too far from reality"

1. Publish a market blend as the headline match AND champion probabilities (biggest
   win). Model weight ~0.3–0.4. Requires storing per-match h2h odds.
2. Either wire up temperature or delete it (currently misleading; optimal tournament
   T≈1.03 means a global temperature gives ~0 benefit).
3. Fix the scoreline headline (presentation): top-3, expected score, or explicit ~14%.
4. Loosen the adoption gate for in-tournament learning (carefully).
5. Give goal_scale its own validation fold (low priority).

Key refs: forecast.py:323-332, :126-153, :139; model.py:119-137; backtest.py:163-176,
295, 314; edge.py:148-152, 221-222; odds.py:115-139; learn.py:486-497, 600-621.
DB: model_params_log 0 rows; 4 graded tournament vs 24 friendly; Spain model 24.2% vs
market 15.2%.

---

## D3 — diag:data-features (data, features & freshness)

### 1. Feature inventory — three narrow channels

| Signal | Source | Entry | Drives |
|---|---|---|---|
| Elo (results-only) | base_elo.json + replay | forecast.py:60-67; model.py:43-52 | ~95% of forecast |
| Host/home advantage | config.py:59-60; home_adv_elo=115.3 | added to Elo | host lambda split |
| Availability (LLM news) | news.py -> overrides.py:175-212 | forecast.py:100; model.py:155-157 | scales attacking lambda |
| Goalscorer shares | players.club_goals; scorers.py | scorers.py only | cosmetic scorer list |
| H2H delta | h2h.py | model.py:44 | tiny supremacy nudge |

W/D/L and scoreline outputs are almost entirely a deterministic function of one number,
the Elo gap. base_goals=2.19 sets total volume; gamma adds a little variation in
mismatches only. NO team-specific attack/defence (deliberate).

### 2. Missing professional signals

xG / shot data (FBref / Understat / Opta): ENTIRELY ABSENT (grep returns 0 hits). The
model has never seen a shot, xG, or chance-quality number. A results-only Elo over-rates
teams on a hot finishing/result streak — the mechanical cause of complaint #1. The Elo
top-6 (Spain 2225, Argentina 2187, France 2131) sit ~150–200 pts above market/xG-blended
ratings.

Squad market value (Transfermarkt): ABSENT as a signal (named only in the news LLM
prompt, news.py:71). players.minutes and players.goal_share columns exist but are 0/1246
populated (dead). Pros blend xG/value precisely to temper Elo on top teams.

### 3. Scoreline complaint — feature-poverty symptom

28 graded snapshots: predicted headline 5 distinct values (2-0 x14, 1-0 x7, 0-1 x4, 0-2
x2, 3-0 x1) vs 12 distinct actual. 2-0/1-0 was the headline in 21/28 (75%). pick=draw
happened 0/28 despite 18% of results being draws. The actual modal scoreline is 2-1.

### 4. Favourite overconfidence — quantified from the frozen pre-match log

```
pmax 0.50-0.65 (n=14):  Brier 0.458, hit-rate 0.76   <- the damage is here
pmax 0.65-0.80 (n=9):   hit-rate 0.89
pmax 0.80+    (n=5):    hit-rate 1.00
strong favs pmax>=0.70 (n=11): Brier 0.168
close pmax<0.70 (n=17):        Brier 0.458   (2.7x worse)
```

Big Brier hits all came from Elo-favourites who drew: Canada-Bosnia (P(home)=0.75, drew
1-1, Brier 1.235), Panama-Bosnia (0.59, drew, 0.944), Saudi-Senegal (0-0, 0.919),
Morocco-Norway (0.34, drew, 0.673). Mean P(draw)=0.22 vs actual 0.18, never modal.

### 5. Freshness / ingestion-lag

Scrape source is healthy (live-tested ESPN returns completed Qatar 1-1 Switzerland). The
lag is the update CADENCE (~3.5h heartbeat, schedule.py:25 HEARTBEAT_HOURS=4), not the
scraper. A freshly-finished match waits up to ~3.5h for the next cycle. The staleness is
a 0–3.5h grading delay, self-healing and bounded — secondary, not the cause of "far from
reality." The feature gap is primary.

### 6. Elo quality

K-factors (elo.py:33-45): friendly 20, WC finals 60, qualifiers 40, continental 50. WC
K=60 with goal-diff multiplier (×1.5 at GD 2, ×1.75 at GD 3) is reactive — a 4-0 win can
move a team ~+30–50 pts in one match; arguably too reactive given no xG damping.
Pre-tournament Elo is dominated by friendlies (105 friendly rows since 2026-05-16 vs 4
WC). Elo is reactive enough — the problem is that results-only Elo is the only
team-strength signal.

### Ranked data gaps

1. No xG / shot-quality signal (FBref/Understat/Opta). Causes complaint #1. Highest
   leverage.
2. No squad market value (Transfermarkt).
3. base_goals=2.19 too low + no team-attack signal -> complaint #2.
4. Update cadence ~3.5h. Self-healing, secondary.
5. Dead players.minutes / players.goal_share columns; club_goals used only cosmetically.

Refs: model.py:43-52, :46-48; config.py:53-60; news.py:71; scorers.py:24-31;
schedule.py:25; scrape.py:17,86; update.py:19-54; db.py:36-46.

---

## D4 — diag:learning-loop (self-learning loop & multi-agent coherence)

### A. Does the predictor actually learn and update published predictions?

Verdict: mechanism fully built and wired, but as of now (~4 of 72 group matches played)
it has adopted ZERO parameter changes and is effectively FROZEN on pre-tournament params.
model_params_log has 0 rows. Even when the gate does fire in-window, the only parameter
it can currently move (rho_friendly) has no effect on group/knockout forecasts, and any
tournament param it adopts is silently clobbered by GitHub Actions.

The loop, traced (update.py:57-150): snapshot_upcoming (freeze pre-match probs) ->
scrape.update_results + recompute_elo -> grade_newly_played -> run_news_scan +
run_postmortems -> adopt_adjustments (the gate, runs BEFORE the forecast, good) ->
check_rollback -> forecast (skipped if input_fingerprint unchanged; fitted_params.json
bytes are in the hash so an adopted param busts the skip) -> vaultgen.generate. Wiring is
genuinely closed-loop. The problem is the gate never opens.

Live: `{"adopted": 0, "reason": "insufficient graded matches (tournament: 4/12)"}`.
Four gates block in compounding order:

- Stall 1 — _MIN_EVAL_GROUP=12 (learn.py:487). Only 4 graded. Binding today.
- Stall 2 — quorum per factor _QUORUM_GROUP=5 (learn.py:387). Tournament postmortems
  exist for only 1 wrong group match. tournament quorum factors: [].
- Stall 3 — the friendly pool can fire (24 graded) but _PARAM_MAP_FRIENDLY maps only
  draw -> rho_friendly; that factor is at n=4 (one short of quorum 5); and even if
  adopted, rho_friendly only feeds friendly_forecasts (forecast.py:307) — changes
  NOTHING about the group/knockout/champion forecast.
- Stall 4 — anti-ratchet + net-margin: 12 new graded matches between adoptions; each step
  small (c±8, base_goals±0.06, rho±0.015).

Net: in-tournament feedback is effectively frozen on pre-tournament params during the
group stage. Biases are mined into the vault (LESSONS.md) but the forecast does not move.

### B. Multi-agent coherence (launchd vs GitHub Actions) — NOT coherent

| | launchd (Mac, start.sh) | GitHub Actions (forecast.yml) |
|---|---|---|
| Cadence | event-driven ~3-4h | daily 06:00 UTC |
| Runs | wc26.update (full learn loop) | ingest + backtest + ingest + forecast |
| wc26.db | persists (gitignored) | rebuilt from data/raw, thrown away |
| fitted_params.json | written ONLY by adoption gate | OVERWRITTEN EVERY run by backtest.run(write=True) |
| Commits to main | yes [skip ci] | yes [skip ci] |

Coherence problem 1 (critical): CI clobbers any adopted tournament param. `python -m
wc26.backtest` calls run(write=True) which unconditionally rewrites c, base_goals, rho,
goal_scale, home_adv_elo, temperature from the static historical fit (backtest.py:312-334),
preserving only rho_friendly (308-315), then commits to main. The next daily CI run
reverts a gate-adopted nudge; launchd rebases onto origin/main and pulls the reverted
params back. The learning gets erased within 24h. Single most damaging coherence bug.
Hasn't bitten only because the gate has never adopted.

Coherence problem 2: the persistent learning DB exists only on one machine (wc26.db
gitignored). CI rebuilds a throwaway DB from data/raw, so CI re-snapshots/re-grades
against an empty prediction_log and regenerates the vault from static params. Two
published surfaces can disagree.

Coherence problem 3: git race / double-write on WC vault + fitted_params.json. launchd
rebases before pushing and aborts on conflict (a cycle's commit can be silently dropped).
CI concurrency only serializes CI against itself. No cross-agent lock. Both [skip ci] (no
loop, but CI never re-runs on launchd's data pushes).

What is coherent / safe: _write_params atomic; WAL checkpoint each cycle; publish_db.sh
uses SQLite backup API and refuses empty DBs; [skip ci] prevents trigger loops; start.sh
kills orphaned loops.

Bottom line: they will not corrupt each other, but they overwrite each other's intent. CI
is authoritative for fitted_params.json (static fit) and launchd for everything learned —
and CI's authority destroys launchd's learning.

### C. What is actually changing forecasts in-tournament right now?

Moves: played results -> Elo recompute -> conditional simulation (who advances, champion
odds); availability events; champion market blend (50/50). Does NOT move: the W/D/L and
scoreline model parameters (c, base_goals, rho, gamma) — frozen on the pre-tournament
fit. temperature is not even in the gate's tunable set (learn.py:495). The one knob most
directly tied to favourites overconfidence is unreachable by the learner.

### D. Smallest changes to make the loop self-correcting in-tournament

- Fix 1 (coherence-critical): stop CI clobbering learned params. CI should not write
  fitted_params.json during the tournament; the local gate is the sole writer.
- Fix 2 (high): make temperature a gate-tunable parameter. A temperature nudge is global,
  low-variance, and directly flattens overconfident favourites.
- Fix 3 (high): lower the in-window evidence bar AND aggregate quorum across factors,
  leak-free (drop _MIN_EVAL_GROUP to ~8; reduce _QUORUM_GROUP to 3 for global calibration
  params; keep 5 for structural ones).
- Fix 4 (medium): the eval set _feats_2026 already pulls graded matches leak-free; blocker
  is purely the count thresholds.
- Fix 5 (medium, coherence): give the learning DB a home that survives CI, OR make CI
  learning-aware. Simplest: disable the scheduled CI job during the tournament window;
  keep workflow_dispatch.
- Fix 6 (low): friendly pool can't move the real forecast by design.

Recommended minimal coherent set: Fix 1 + Fix 5(a) + Fix 2 + Fix 3.

Refs: learn.py:386-388, :486-491, :495-497, :550-680, :403-405; forecast.py:307;
backtest.py:308-334; .github/workflows/forecast.yml:50-56; scripts/start.sh; .gitignore.
DB: model_params_log 0 rows; prediction_log 99 (28 graded); postmortems 34, 1 wrong group
match; friendly draw at n=4.

---

## R1 — research:pro-systems (how professional forecasters work)

wc26 today is an Elo supremacy -> single goal-expectation pair -> Dixon-Coles Poisson, no
team attack/defence, base_goals=2.19, plus market devig but the market is not yet blended
into the rating. The two complaints map onto two documented failure modes of exactly this
archetype. A Towards Data Science 2026 WC model is structurally near-identical (Elo,
GOALS_PER_400_ELO, GOALS_BASE split in half, max(0.15, ...) floor, independent Poisson
outer-product, 10k sims) — so wc26's design is a recognised archetype and its limitations
are the textbook limitations of that archetype.

### 1. FiveThirtyEight SPI

Two ratings per team on a goals scale: offensive (expected goals scored vs average on a
neutral field) and defensive (expected conceded). Each team's match expectation combines
its own offence with the opponent's defence + home advantage, then two Poisson
distributions are generated. wc26's single supremacy collapses both teams onto one axis.
Adjusted goals: down-weight goals scored with a man advantage or late while leading.
World Cup ratings: 75% match-based + 25% roster-based (club contributions mapped to
international level) for sparse samples. Knockouts: separate Poisson regression on
extra-time goals since 2005; penalties near-coin-flip.

### 2. Elo families + Opta

eloratings.net: We = 1/(10^(-dr/400)+1), dr = rating diff + 100 home; importance-weighted
K (60 WC finals, 50 continental, 40, 30, 20 friendlies); goal-difference multiplier on K
(×1.5 won by 2, ×1.75 by 3). wc26 has one c=208.86 (goal-conversion constant), not a
per-competition K — friendlies and WC treated identically in updates. Opta Power Rankings:
0–1000 Elo-derived; goal-diff scaling with diminishing returns above ~3; competition
quality multiplier; match probabilities from betting-market odds AND Power Rankings
combined, then 25,000 sims. 2026 Opta champion benchmark: Spain 16.1%, France 13.0%,
England 11.2%, Argentina 10.4%, Portugal 7.0%, Brazil 6.6%, Germany 5.1%.

### 3. Bookmaker / market models

Betting Odds Rating System (PLOS One / PMC5988281): an Elo fed by devigged betting
probabilities ("ELO-Odds", high k=175) beats an Elo fed by goal difference, which beats
raw results. Log-loss: Betting odds 1.3795 < ELO-Odds 1.3913 < ELO-Goals 1.4008 <
ELO-Result 1.4032. "Betting odds prior to a match possess more information than the result
known after the match." Closing odds are the hardest benchmark; expect to match, not beat.
Direct relevance to complaint #1: the market's implied strength of elite sides already
embeds regression that a pure-Elo supremacy does not.

### 4. Academic WC models

Groll/Ley/Schauberger/Zeileis hybrid random forest (arXiv:1806.03208; 2026 R-bloggers):
RF fed with ability estimates -> goals -> independent Poisson -> 100,000 sims.
Variable-importance: market value (Transfermarkt) is the single most important predictor,
then bookmaker-consensus abilities, then plus-minus ratings; FIFA rank is weak. 2026
hybrid: Spain 14.5%, England 12.4%, France 12.4%, Germany 11.2% — less top-heavy than
wc26. Good full-distribution models land Brier ≈ 0.18–0.27; RF-only ≈ 0.46;
1/3,1/3,1/3 Brier is 0.667. wc26's 0.45 in-tournament barely beats chance.
Overconfidence fix — temperature scaling (Guo et al.): divide logits by learned T>1;
note T=0.996 < 1 actually SHARPENS, the wrong direction. Equivalent: shrink toward the
base rate / market.

### 5. Complaint #2 directly

Not a bug — a structural property of a low-mean Poisson; pros never report the modal score
as the headline. The overall modal scoreline in real football, 1-1, occurs only ~11–12%.
Pros report the full correct-score matrix / top-N with probabilities (538 ranked grid;
bookmakers publish a correct-score market, never a single number). To genuinely spread:
(a) raise base_goals toward ~2.5–2.7; (b) add team-specific attack/defence; (c) bivariate
Poisson / diagonal inflation (Karlis-Ntzoufras) raises draw mass realistically. Dixon-Coles
tau only nudges the four lowest cells and does not broaden the distribution.

### Ranked recommendations

1. Blend devigged (closing) market odds into the rating / final probabilities (very high
   impact, medium effort). 2. Temperature scaling T>1, separate for in-tournament vs
friendly (high, low). 3. Report top-N scorelines + cumulative prob (high perceived, low).
4. Team-specific attack/defence offsets (high, high; overfit risk — heavy shrinkage /
club priors). 5. Raise base_goals to ~2.5–2.7 (medium, very low). 6.
Match-importance-weighted K + goal-difference-weighted updates (medium, low-med). 7.
Knockout realism: extra-time rate + coin-flip penalties (medium, low-med). 8.
Sparse-sample club/value prior (medium, medium).

Sources: 538 methodology pages; eloratings.net; Opta 2026 supercomputer; PLOS/PMC5988281;
Groll arXiv:1806.03208 and 2026 R-bloggers; Dixon-Coles 1997 + dashee87; Karlis-Ntzoufras
2003; Guo et al. temperature scaling; arXiv:2308.01222; arXiv:1705.04356; QMUL eval; TDS
2026 WC.

---

## R2 — research:scoreline-models (scoreline / goal models)

### A. Reframing complaint #2

The "always 1-0/2-0" headline is the mathematically correct mode of a low-mean Poisson,
and the mode is a near-useless statistic for football scores. Verified against the live
model:

| Matchup (Elo) | la, lb | Modal | Modal mass | 4 low-scores mass |
|---|---|---|---|---|
| Top vs weak (2100–1650) | 3.07, 0.36 | 3-0 | 15.6% | 18.0% |
| Strong fav (2050–1800) | 2.37, 0.62 | 2-0 | 14.1% | 27.4% |
| Mild fav (1950–1880) | 1.74, 0.86 | 1-0 | 12.3% | 37.8% |
| Even (1900–1900) | 1.10, 1.10 | 1-1 | 14.2% | 49.0% |

No exact score ever exceeds ~16%. The model does NOT actually only say 1-0/2-0 — for
favourites 2-0/3-0, for even games 1-1. If the user sees only 1-0/2-0, the cause is
most_likely_scoreline() restricting to the modal-outcome region combined with low
base_goals=2.19. The genuine fix is communication.

### B. Dixon-Coles — already implemented (low-score tweak only)

_dc_tau (model.py:55-62) is textbook-correct. Typical fitted rho ≈ -0.13 (EPL); the repo
uses -0.06, weaker than literature. The repo does NOT use DC's per-team attack/defence
strengths (deliberate). Full DC strengths: marginal expected benefit.

### C. Bivariate Poisson (Karlis-Ntzoufras 2003)

Cov(X,Y)=lambda_3 >= 0 can only model POSITIVE correlation; football is often slightly
negative (handled by DC's negative rho). So it can be the wrong sign. Worst of six in the
bake-off. Not recommended.

### D. The decisive empirical bake-off (penaltyblog 2025)

| Model | RPS |
|---|---|
| Dixon-Coles | 0.1914 |
| Weibull-count + copula | 0.1914 |
| Poisson (baseline) | 0.1915 |
| Zero-inflated Poisson | 0.1915 |
| Negative Binomial | 0.1916 |
| Bivariate Poisson | 0.1916 |

The entire spread is ~0.02% RPS. Changing the scoreline distribution family will NOT fix
the calibration complaint. The error budget is in the lambda inputs (the Elo->goals
mapping, the gamma blowup, the lack of team strengths), not the count distribution. The
single most important finding: family-swaps are low-leverage.

### E. Overdispersion fixes (relevant to complaint #1, fat tails)

Negative Binomial: didn't help RPS, BUT specifically dampens the extreme blowout scores
(4-0, 5-0, 6-0) the live model over-produces for top-vs-weak (P(>2.5)=66.6%, modal 3-0).
Low effort (nbinom swap + 1 param). Worth a backtest. CMP: high effort, marginal — skip.
ZIP: no gain — skip. Skellam (goal difference): trivial, gives clean margin/spread
markets, doesn't help scorelines.

### F. xG-based lambda — the real lever

Replace/blend the single Elo-supremacy lambda with team-specific attack/defence trained on
xG-for / xG-against (less noisy than goals on sparse data). Highest-leverage change for
both complaints; harder part is sourcing international xG. And report the distribution, not
the mode: top 3–5 scorelines with probabilities (top_scorelines already exists,
model.py:89-96), plus derived markets that aggregate the matrix (each a sum over the
existing matrix, near-zero effort): Over/Under 2.5, BTTS = 1 - P(home=0) - P(away=0) +
P(0,0), score bands, clean-sheet probabilities.

### G. Ranked recommendations

1. Report distribution + derived markets (trivial, high — do first). 2. Soften the lambda
map for top favourites (revisit gamma=0.456 and c; low effort, high). 3. Negative-Binomial
marginal to tame blowout tail (low, medium — backtest it). 4. Skellam margin/spread market
(trivial, optional). 5. Tune rho toward -0.10/-0.13 (trivial, low-med). 6. xG attack/defence
lambda (high, highest if data exists). 7. Bivariate Poisson — do NOT. 8. CMP/ZIP — skip.

### H. Code-grounded notes

model.py:44-52 — the entire lambda from one supremacy; base = base_goals*goal_scale +
gamma*|supremacy|; gamma=0.456 is the blowout amplifier. model.py:55-78 — DC tau correct,
single global rho=-0.06. model.py:89-96 / 119-137 — top_scorelines and
most_likely_scoreline already exist; the matrix already contains everything for O/U, BTTS,
bands. simulate.py:65,148-154 — Monte Carlo draws rng.poisson(la/lb) independently;
swapping to NegBin is a one-line change.

Bottom line: the scoreline distribution family is NOT the problem; complaint #2 is a
reporting bug; complaint #1's scoreline symptom traces to gamma*|supremacy| fattening the
favourite's tail. Genuine model fixes: shrink the supremacy->lambda map, optionally a
NegBin marginal, and (higher cost) xG-based attack/defence lambda.

Sources: dashee87 Dixon-Coles; Karlis-Ntzoufras 2003; penaltyblog bake-off; CMP PMC10193358;
Skellam; gamblingcalc / football-bet-prediction "think in distributions".

---

## R3 — research:calibration-ensemble (calibration, market blending & ensembling)

### A. Repo state vs missing

Already present: devigging (proportional + Shin, edge.py:92-145; outright devig
odds.py:68-81); linear champion blend weight 0.5 (odds.py:115-124; forecast.py:326-332);
post-hoc temperature scaling p ∝ p^(1/T), T=0.996 (backtest.py:163-176); odds-scaled
shrink for the edge path only; proper scoring rules (log-loss, Brier, RPS, reliability
slope, baselines).

Four gaps causing the complaints:
1. No match-level W/D/L market blend — the published p_home/p_draw/p_away come straight
   from the matrix (model.py:159; forecast.py:143). The blend exists only for the champion
   outright. The h2h match market is even fetched (odds.py:62) but never blended. Biggest
   miss.
2. Blend is linear, weight unfitted at 0.5 — arithmetic pooling cannot pull an
   over-extreme model probability back as hard as a log pool.
3. Temperature is fit on train, not held-out (leak) — backtest.py:295 fits T on the very
   data the params were fit on, so T collapses to ≈1.0. This is WHY in-tournament Brier
   (~0.45) is far worse than the ~0.30 friendly number: T never learned the real
   overconfidence. Fix is near-free, high-impact.
4. "Most likely scoreline" = argmax of a low-mean Poisson matrix — structurally 1-0/2-0/
   1-1. Reporting choice, not a model defect.

### B. Market blending recipe (largest impact)

B1. Use a logarithmic (geometric) opinion pool, not linear: p_k ∝ m_k^w · q_k^(1-w),
renormalise. The log pool pulls an over-extreme component multiplicatively toward the
market and the correction accelerates as the model gets more extreme/wrong — exactly the
regime the user reports. The (extremized) geometric mean of odds robustly beats the
arithmetic mean of probabilities on Brier (Tetlock/Metaculus). Log pool uniquely satisfies
external Bayesianity. Wheatcroft: in football, blending consistently beats Platt scaling.

B2. Fit the blend weight w leak-free on historical matches with both a model probability
and a closing devigged market line (same join edge.py builds): 1-D golden-section search
(reuse minimize_scalar pattern, backtest.py:174-175) over w in [0,1] minimising OOS log-
loss of the log-pool blend, with a temporal split. Expect the optimum w ≈ 0.2–0.4 on the
model (market-heavy). Defensible fixed prior: w ≈ 0.3 model / 0.7 market.

B3. Extremization (o^a, a>1) helps underconfident ensembles; the user is overconfident,
so keep a <= 1.

Expected impact: moves in-tournament Brier from ~0.45 toward ~0.30–0.33 and log-loss ~0.75
-> ~0.62–0.66. Low effort; the h2h market is already fetched.

### C. Calibration methods for multiclass W/D/L, small samples

C1. Fix the temperature leak FIRST (near-zero effort, high value): fit on a held-out fold
(backtest.py:295). Temperature has no bias term — it only makes predictions more/less
extreme without flipping argmax. An honestly held-out fit lands T>1, directly attacking
favourite overconfidence. Cheapest single win.
C2. Beta / Dirichlet calibration (Kull et al. NeurIPS 2019) on historical internationals,
not the 104 WC matches — fit offline, apply live.
C3. Isotonic — AVOID (data-hungry, overfits small data).
C4. Platt — dominated by Beta/Dirichlet.
Recommended stack: model matrix -> temperature T (held-out fit) -> log-pool blend with
devigged market at fitted w -> (optional offline) Dirichlet map.

### D. Evaluating & shrinking during a 104-match tournament

D1. Do NOT re-fit T or w on the handful of completed WC matches (overfits noise). Treat
historical-fitted T0, w0 as a prior; form a precision-weighted (empirical-Bayes / beta-
binomial) posterior: w_live = (n_prior·w0 + n_obs·w_obs)/(n_prior + n_obs) with large
pseudo-count (200–400) so the first ~20 WC results barely move it.
D2. Report bootstrap CIs on in-tournament Brier/log-loss (extend bootstrap_log_loss_ci,
backtest.py:153) — 0.45 ± a wide band likely overlaps 0.30. Use RPS as the primary
in-tournament metric. Track vs the market as the benchmark, not vs perfection.
D3. Shrink the Elo supremacy toward market-implied supremacy before forming lambdas
(input-side regularizer), OR accept it and let the market blend + de-sharpening
temperature absorb it post-hoc (cheaper, recommended first).

### E. Ensembling / stacking (after A–D)

Log-pool stack 2–3 weak models (current Elo-DC; a draw-inflated variant; a market-only
model), weights fit by the same leak-free CV log-loss minimisation. Egidi et al.: scoring
rates a convex combination of historical-data params and betting-odds params. Treat the
market as one "expert" in the pool. Brier ↔ arithmetic pool, log score ↔ log pool; since
you optimise log-loss, the log pool is the internally consistent choice.

### F. Scoreline complaint — a reporting fix

most_likely_scoreline returns argmax; for mean total 2.19 the modal cell is almost always
1-0/1-1/2-1. Report outcome (W/D/L %) as primary + modal score labelled "~12-15% likely"
+ top-3 scorelines (top_scorelines, model.py:89). Optionally expected score. Do NOT inflate
goals for calibration reasons — that trades away the deliberately-refit-down volume; do it
only as a display choice.

### G. Prioritised action list

1. Fit temperature on held-out fold (backtest.py:295) — High, Trivial. 2. Match-level
W/D/L blend with devigged h2h market — Highest, Low. 3. Linear -> log pool (odds.py:115-124)
— High, Trivial. 4. Fit blend weight w leak-free (~0.3 model) — High, Low-Med. 5.
Beta-binomial / EB shrinkage of T and w during tournament — Med, Med. 6. Reframe scoreline
headline — Med, Trivial. 7. Bootstrap CIs + RPS on live slice — Med, Low. 8. Offline
Dirichlet calibration — Med, Med. 9. Shrink Elo supremacy toward market supremacy — Med,
Med. 10. Log-pool stack — Low-Med, Med.

The combination of #1 (un-leak temperature) + #2/#3/#4 (log-pool match blend at a fitted
market-heavy weight) closes the gap between the ~0.45 in-tournament Brier and the market's
~0.30. #6 answers the scoreline complaint with no model change.

Sources: Egidi et al. arXiv:1802.08848; Wheatcroft arXiv:2106.14345; "geometric mean of
odds" (EA Forum); Kull et al. arXiv:1910.12656; scikit-learn calibration; Robinson
beta-binomial EB; Wilkens Bundesliga simple models; dashee87 Dixon-Coles.

---

## R4 — research:data-sources (data sources, xG & in-tournament signals)

### 0. Current pipeline

Results: scrape.py:17 ESPN hidden site API; odds: odds.py:17 The Odds API v4 (devig 68,
blend 115); availability: news.py LLM scan; Elo: eloratings.net-style international only.
NO xG, NO shots, NO squad market values anywhere. The model has zero attack/defence
quality signal beyond one Elo supremacy and a flat base_goals=2.19.

### 1. xG / shot data

A. StatsBomb Open Data — BEST free international xG. Free men's WC 2022 (comp 43, season
106), historically WC 2018, Women's WC 2023, AFCON 2024, Euros. pip install statsbombpy,
plain JSON on GitHub. CRITICAL: released months-to-years AFTER a tournament, never live —
NOT usable for in-tournament grading of WC-2026. Value is historical/training: deriving
team-specific finishing & shot-suppression priors and validating base_goals/DC.
B. FBref (StatsBomb-powered) — best free CURRENT-FORM international xG. xG, xA, xGA, npxG,
shots, SoT per match and per competition, including national-team competitions. pip
install soccerdata. ToS technically prohibits scraping; rate-limit aggressively (~1 req/3s).
Latency: match advanced stats within hours to ~1 day — usable as a next-day signal.
C. Understat — club leagues only, no internationals. Skip.
D. Opta / Hudl — paid. Out of scope.
Ranking: FBref-via-soccerdata (current-form, next-day) > StatsBomb open (historical priors)
>> Understat >> Opta.

### 2. Squad strength / market values

The single most-cited missing feature in academic WC models and the cleanest fix for
overconfident-on-top-Elo. A. Transfermarkt — per-player and squad value. Two routes:
(1) pre-built Kaggle dump davidcariboo/player-scores, CSV refreshed ~weekly (lowest
effort, dodges Cloudflare); (2) transfermarkt-scraper (Scrapy). Pros regress goals on log
market value + average rating alongside Elo. The repo already has per-player squads
(players table), so values join cleanly. B. SoFIFA ratings via soccerdata — cheap squad-
quality proxy. Ranking: Kaggle Transfermarkt dump > SoFIFA > live scrape.

### 3. Live results & fixtures

1. ESPN hidden site API (already in use) — free, no key, minutes after final whistle.
Keep as primary. 2. openfootball/worldcup.json — public-domain JSON, no key, 2026 schedule
present, community-updated (hours–days lag) — fixtures backup + reconciliation cross-check.
3. football-data.org free tier — WC competition, key, 10 req/min, scores delayed/next-day
— redundant grader. 4. Paid (Sportmonks/TheStatsAPI/API-Football) — only for live xG.
Realistic cadence: post-match (~1–2h) grade via ESPN, reconciled next-loop against
openfootball/football-data.org. Do NOT chase live-minute data.

### 4. Tournament-specific signals

Host advantage — already modelled (home_adv_elo=115.3); 2026 has three hosts and mostly US
venues, so host boost should be venue/travel-aware. Travel/altitude/rest — 2026 is the big
one (Mexico City 2240m; Vancouver<->Miami). Compute from fixture venue coordinates + match
dates (already derivable). New context.py adding per-team rest_days, travel_km, altitude_m.
Motivation/dead rubbers — 48-team format (12 groups of 4, 8 best third-placed advance) —
final-group-game stakes vary. Knockout vs group goal rates — knockouts lower-scoring; worth
a stage-conditional base_goals. Small-sample in-tournament regime — heavy shrinkage to
pre-tournament priors and pooling; the current Brier blowup is consistent with
under-shrinkage + overconfidence (market-blend / temperature widening is the lever).

### 5. Structural fix for complaint #2

Confirmed structural; DC only nudges the 0-0/1-0/0-1/1-1 cells, does not move the headline
mode. Remedies: Negative-binomial marginals (overdispersion spreads mass off the 1-0/2-0
spike); reporting fix (top-5 / band, stop surfacing only the mode); xG/squad data pushes
some modes to 2-1/3-1.

### 6. Consolidated ranking (accuracy-per-effort)

1. Transfermarkt squad value (Kaggle weekly dump) — regularizes overconfident favourites +
team-specific lambdas; Low effort. 2. FBref xG via soccerdata — in-tournament next-day xG
form correction; Med. 3. Context features (rest/travel/altitude/stage) from existing
fixtures — Low, no new feed. 4. StatsBomb open data — historical priors, validation; Low.
5. football-data.org / openfootball — reconciliation grader; Low. 6. SoFIFA via soccerdata
— cheap squad proxy; Low. Skip Understat and paid.

Bottom line: the two complaints are a missing-signal problem, not a sourcing impossibility.
(1) Add squad market value (free, weekly) to regularize Elo-only overconfidence. (2) Add
FBref/StatsBomb xG for team-specific, faster-correcting goal expectations. (3) Add
zero-new-data context features. Keep ESPN as the fast grader, add a free reconciliation
feed. None requires a paid API.

Sources: StatsBomb open-data + statsbombpy; soccerdata FBref; transfermarkt-scraper +
Kaggle player-scores; football-data.org; openfootball/worldcup.json; TDS 2026; RSD Poisson
WC2026; 48-team format arXiv:2502.08565; Women's WC hybrid arXiv:1906.01131; Bayesian
state-space JRSS-C; DC+NegBin arXiv:2307.02139.

Relevant files: wc26/scrape.py, wc26/ingest.py, wc26/news.py, wc26/learn.py, wc26/model.py,
wc26/odds.py.
