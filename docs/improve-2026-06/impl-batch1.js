export const meta = {
  name: 'wc26-implement-batch1',
  description: 'Sequentially implement + backtest-gate the top WC-2026 improvements (distribution reporting, goal-aware base_goals, temperature calibration, live market blend, durable self-learning) in the dev worktree; commit on pass, revert on regression',
  phases: [{ title: 'Implement', detail: '5 gated items, sequential, each committed-on-pass / reset-on-regression in the worktree' }],
}

const WT = '/Users/tutaitran/wc26-dev'
const PY = '/Users/tutaitran/wc26/.venv/bin/python'

const PREAMBLE = `
You are a senior ML/stats engineer improving the WC-2026 football predictor. You work ONLY in the git worktree at ${WT} (branch improve/accuracy-2026-06). NEVER touch /Users/tutaitran/wc26 (main) and NEVER push. Always start bash with: cd ${WT}. Run Python with: ${PY} (cwd=${WT} so it imports the worktree's wc26 package).

The full plan is at ${WT}/docs/improve-2026-06/PLAN.md (read the relevant item before coding). Frozen baseline backtest is at ${WT}/data/raw/backtest_report.baseline.json. Baseline reference: test log_loss 0.8597, brier 0.5049, rps 0.1672; tournament_only log_loss 0.9334; reliability home_win_slope 1.003; mean implied total goals 2.50; model BTTS 0.434; headline scoreline distinct-value count = 4.

DISCIPLINE (mandatory):
1. Before coding: 'cd ${WT} && git status --short' must be clean (if not, 'git stash' scratch). Read the exact files/functions named in your item.
2. Implement the SMALLEST change that satisfies the item. Match the existing code style (PEP8, type hints, dataclasses). Keep backward-compatible field names; add new fields alongside.
3. Add/extend pytest unit tests under tests/ for your change.
4. Run the gate: 'cd ${WT} && ${PY} -m pytest -q' (ALL must pass) and 'cd ${WT} && ${PY} -m wc26.backtest' (writes data/raw/backtest_report.json). Compare the new report to backtest_report.baseline.json against YOUR item's gate.
5. ADOPT ONLY IF the gate passes AND no guardrail is breached. On adopt: 'git add -A && git commit -m "<msg>"' and report the sha. On reject/regression: 'git reset --hard HEAD && git clean -fd' (so the next item starts from the last good state) and report adopted=false with why.
6. NEVER adopt a regression. The overall test log_loss must not rise above 0.8637 (baseline +0.004) for any item except where the item explicitly allows it; if it does and your item does not justify it, revert.
You report structured output for the orchestrator; the metrics string must contain concrete before->after numbers.
`

const items = [
  {
    label: 'item2:distribution-report',
    prompt: `${PREAMBLE}

ITEM 2 — REPORT THE SCORELINE DISTRIBUTION, NOT A SINGLE MODAL CELL (fixes complaint #2; purely additive, XS).
Read PLAN.md item 2, and model.py (top_scorelines exists at ~89-96, most_likely_scoreline ~119-137, match_forecast ~140-172), forecast.py (group_match_forecasts ~126-153 and the knockout/friendly emit ~185-208), vaultgen.py (where the lone modal scoreline is rendered).
Implement:
- model.py: add derived_markets(matrix) -> dict with over_2_5/under_2_5, btts_yes/btts_no (= 1 - P(h=0) - P(a=0) + P(0,0)), total_goals_bands ({"0-1","2","3","4+"}), clean_sheet_home/away. All are sums over the existing normalised matrix (no new params). Add expected_score(la, lb) = (round(la), round(lb)).
- forecast.py: in BOTH the group and knockout/friendly per-match dicts, KEEP the existing 'top_scoreline'/'likely_scoreline' field for backward compat, and ADD: top_scorelines (list of [[i,j],prob] from model.top_scorelines(matrix,5)), top_scoreline_p (explicit), expected_score, and derived_markets.
- vaultgen.py: render the top-5 scorelines with probabilities + over/under 2.5 + BTTS + expected score, and label the single modal score WITH its ~12-15% probability so it reads as "most likely single score", not "the prediction". Keep AUTO-block markers intact.
- tests/: add tests asserting derived_markets sums are consistent (btts+cs identities), top_scorelines is sorted desc and length 5, and that W/D/L numbers are byte-identical before/after (same matrix).
GATE (item 2): (i) across the 72 group fixtures the distinct headline-scoreline SET expands from 4 to >= 10 (write a tiny script using forecast internals or model over the fixtures to count it; report the number); (ii) top-5 cumulative mass per fixture ~0.50-0.62; (iii) W/D/L probabilities UNCHANGED vs baseline (identical matrix) — pytest must prove this; (iv) full pytest green; (v) backtest log_loss byte-identical to baseline (this change does not touch the matrix/params). If W/D/L or backtest changed at all, you broke something — revert.
Commit message: 'feat(report): publish scoreline distribution (top-5, O/U, BTTS, expected score) instead of lone modal cell'.`,
  },
  {
    label: 'item5:goal-aware-base_goals',
    prompt: `${PREAMBLE}

ITEM 5 — RAISE/REFIT base_goals UNDER A GOAL-AWARE OBJECTIVE (fixes ~10% goal under-prediction; model-side half of scoreline diversity, S).
Read PLAN.md item 5, backtest.py (fit ~209-227, run ~283-334, objective ~107/209-227), model.py match_lambdas.
Problem: the fit picks base_goals purely on W/D/L log-loss (blind to totals) and chose 2.19; real WC/neutral total-goals avg ~2.73-2.81; model-implied mean total = 2.50.
Implement: change ONLY the objective that selects base_goals to a combined objective = W/D/L log-loss + lambda_tot * (mean_pred_total - mean_actual_total)^2 (or a Poisson deviance on totals). Tune lambda_tot so the fitted mean total lands within +/-0.05 of the actual neutral/WC average while W/D/L test log-loss rises by <= 0.004. Leave c, gamma, rho on their existing W/D/L objective. Write the new base_goals to data/raw/fitted_params.json via the normal run(write=True) path.
GATE (item 5): mean implied total goals moves from 2.50 to within +/-0.05 of actual (~2.73-2.81) — compute and report both; model BTTS moves from 0.434 toward ~0.46; W/D/L test log_loss rise <= 0.004 (hard cap, else revert and keep lower base_goals); headline distinct-scoreline count increases further (report it). Full pytest green.
Report the fitted base_goals and lambda_tot. Commit: 'feat(model): refit base_goals under goal-aware objective (totals realism without WDL regression)'.`,
  },
  {
    label: 'item3:temperature-calibration',
    prompt: `${PREAMBLE}

ITEM 3 — UN-LEAK temperature: FIT ON A HELD-OUT FOLD AND ACTUALLY APPLY IT (de-sharpen overconfident favourites, S).
Read PLAN.md item 3, backtest.py (fit_temperature ~163-175, its call ~295, goal_scale fit ~300-306, run ~283-334), model.py (ModelParams, match_forecast ~140-172, outcome_probs), forecast.py (where ModelParams is constructed/loaded).
Problem: temperature=0.996 is computed on TRAIN (leak -> collapses to ~1.0) and NEVER read anywhere. T<1 sharpens (wrong direction for an overconfident model).
Implement:
- backtest.py: fit temperature on a HELD-OUT validation fold (a slice of the test window not used to fit goal_scale, or k-fold by tournament), not on train. Move goal_scale's fit to that same validation fold too. Keep the [0.8,2.0] bound.
- model.py: add optional temperature: float = 1.0 to ModelParams; in match_forecast, AFTER outcome_probs, apply p_k ∝ p_k^(1/T) to the W/D/L triple and renormalise (W/D/L only; do NOT alter the scoreline matrix). No bias term (cannot flip the argmax).
- forecast.py: construct ModelParams WITH the loaded temperature so it is actually used (it currently is not).
GATE (item 3): on the held-out fold, log_loss_calibrated < log_loss by a real margin (currently a tie 0.8597) — report both; fitted T MUST be > 1.0 (de-sharpening) or reject as a no-op; on the elite-vs-elite slice (both Elo>2000) the favourite-win overconfidence gap shrinks from +5.2pp toward <= +3pp (add/compute this slice and report it); no OVERALL test log-loss regression. Full pytest green.
NOTE order-of-ops for later: temperature is applied BEFORE any market blend. Commit: 'feat(calib): fit temperature out-of-sample and apply it to W/D/L (de-sharpen favourites)'.`,
  },
  {
    label: 'item1:live-market-blend',
    prompt: `${PREAMBLE}

ITEM 1 — LIVE MARKET BLEND PLUMBING + DEFENSIBLE PRIOR WEIGHT (biggest accuracy lever in production; cannot be historically backtested because we have NO historical odds, so it is a NO-OP on the historical backtest and therefore safe).
Read PLAN.md item 1, odds.py (fetch_match_odds ~52-65, devig_outrights ~68-81, blend_probs, champion blend ~116), forecast.py (group_match_forecasts ~126-153, champion blend ~328-332), edge.py (Shin/devig helpers), update.py (cycle), db.py (public_benchmark table).
Implement:
- odds.py: add devig_h2h(events) -> per-fixture {home,draw,away} devigged probs (3-way; reuse the Shin/devig helper from edge.py if present, else proportional devig); add store_h2h_benchmark(conn, fixture_probs) writing scope='match' rows into public_benchmark; add latest_market_h2h(conn) reader; add blend_wdl(model_probs, market_probs, weight) using a LOG (geometric) opinion pool: p_k ∝ m_k^w * q_k^(1-w), renormalised, where w = MODEL share. Default prior w = 0.35 (model) / 0.65 (market). Read blend_weight from fitted_params.json if present, else 0.35.
- forecast.py: after computing temperature-scaled p_home/p_draw/p_away for a match, IF a market line exists for that fixture, replace published probs with odds.blend_wdl(...); ALWAYS keep the raw pre-blend probs as p_home_model/p_draw_model/p_away_model for the postmortem audit trail. When NO market line exists (all historical data, and most live group games until odds open), fall back to PURE MODEL — never block the forecast on odds. This guarantees the historical backtest is unchanged.
- update.py: fetch + store h2h odds for fixtures within ~48h each cycle (guard on THE_ODDS_API_KEY presence and quota; skip cleanly when absent, exactly like the existing champion path). This also starts ACCUMULATING odds so the weight can be fit later (item 4, deferred).
- tests/: unit tests for devig_h2h (sums to 1, removes vig), blend_wdl (log-pool correctness; w=1 -> model, w=0 -> market; a model-hot favourite e.g. model 0.60/0.25/0.15 vs market 0.45/0.30/0.25 blends to between them and cools the favourite), and that forecast falls back to pure model when no odds.
GATE (item 1): full pytest green; the historical backtest MUST be byte-identical to baseline (no odds in history -> pure model -> unchanged) — verify and report; a unit test demonstrates a hot favourite is cooled toward market. Adoption does not depend on a backtest improvement (it cannot, historically). Commit: 'feat(odds): log-pool blend of published match W/D/L with devigged market (prior w=0.35), live-only, model fallback'.`,
  },
  {
    label: 'item8:durable-self-learning',
    prompt: `${PREAMBLE}

ITEM 8 + COHERENCE — MAKE THE SELF-LEARNING LOOP REACHABLE AND DURABLE (the user explicitly wants the agents coherent and the predictor to keep learning + update predictions).
Read PLAN.md item 8 and the 'KEEPING THE RUNNING AGENTS COHERENT' section, learn.py (gate thresholds ~387/487/495-496, _GATE_STEP/_GATE_RANGE, _MIN_EVAL_GROUP, _QUORUM_GROUP, _NET_MARGIN, _HIST_CAP, anti-ratchet, rollback ~692, adopt_adjustments), .github/workflows/forecast.yml (the git add of fitted_params.json at ~line 54; the scheduled cron).
Implement (ON THE BRANCH ONLY — this will be human-reviewed before it ever reaches main; do NOT enable/disable any live launchd/CI yourself, only edit the repo files):
- learn.py: add 'temperature' to the gate's tunable set (_GATE_STEP e.g. +/-0.03, _GATE_RANGE e.g. [0.8,2.0]) and add a factor mapping so an 'elo_gap over' / favourites-overconfident signal raises temperature toward de-sharpening. Lower _MIN_EVAL_GROUP 12 -> 8 and _QUORUM_GROUP 5 -> 3 FOR GLOBAL CALIBRATION PARAMS ONLY (temperature, small c step); keep 5 for structural params (base_goals, rho). KEEP _NET_MARGIN=0.005, _HIST_CAP=0.01, and the anti-ratchet/_MIN_NEW_BETWEEN_ADOPT and rollback guards UNCHANGED (these prevent thrash).
- .github/workflows/forecast.yml: stop CI clobbering learned params during the tournament. Cleanest: remove data/raw/fitted_params.json from the 'git add' so CI commits only the WC vault + replayed_elo.json; AND gate the scheduled run so during the tournament window the local launchd gate is the SOLE writer of fitted_params.json (keep workflow_dispatch for manual rebuilds). Add a clear comment explaining why (prevents the 24h revert of in-tournament adoptions).
- Preserve leak-free ordering: snapshot_upcoming freezes pre-match probs before results enter Elo; adopt_adjustments runs before the forecast; the skip-fingerprint includes fitted_params.json bytes. Do not break this.
- tests/: add/adjust tests for the new thresholds and that temperature is now in the tunable set; a dry-run-style test showing the gate CAN adopt a temperature nudge after ~8 graded matches of synthetic overconfident-favourite evidence while still satisfying _NET_MARGIN/_HIST_CAP.
GATE (item 8): full pytest green; a dry-run/test demonstrates the gate adopts a temperature step under sufficient synthetic evidence and REJECTS it when the out-of-sample re-fit shows regression > _HIST_CAP (the guard still works). The historical backtest is unaffected by this item (config/learn only) — verify unchanged.
Report exactly what you changed in forecast.yml and learn.py thresholds (this is the highest-review-priority change). Commit: 'feat(learn): make calibration gate reachable + temperature-tunable; stop CI clobbering in-tournament params'.`,
  },
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    item: { type: 'string' },
    implemented: { type: 'boolean' },
    tests_pass: { type: 'boolean' },
    gate_passed: { type: 'boolean' },
    adopted: { type: 'boolean' },
    committed_sha: { type: ['string', 'null'] },
    metrics: { type: 'string', description: 'concrete before->after numbers for this item gate' },
    files_changed: { type: 'string' },
    summary: { type: 'string' },
    follow_ups: { type: 'string' },
  },
  required: ['item', 'implemented', 'tests_pass', 'gate_passed', 'adopted', 'metrics', 'summary'],
}

// Strictly sequential: shared files + dependency order. Each agent commits-on-pass so the next starts clean.
const results = []
for (const it of items) {
  log(`Implementing ${it.label} ...`)
  const r = await agent(it.prompt, { label: it.label, phase: 'Implement', schema: SCHEMA })
  results.push(r || { item: it.label, adopted: false, summary: 'agent returned null (died)' })
  if (r) log(`${it.label}: adopted=${r.adopted} gate=${r.gate_passed} ${r.committed_sha ? '@' + r.committed_sha : ''} | ${r.metrics || ''}`)
}

const adopted = results.filter(r => r && r.adopted)
return {
  adopted_count: adopted.length,
  total: items.length,
  results,
}
