export const meta = {
  name: 'wc26-implement-batch2',
  description: 'Gated structural improvements: supremacy-conditional de-sharpen for elite-vs-elite overconfidence, Negative-Binomial blowout tail, and stage-aware scoring; sequential, committed-on-pass / reverted-on-regression in the dev worktree',
  phases: [{ title: 'Implement', detail: '3 gated structural items, sequential' }],
}

const WT = '/Users/tutaitran/wc26-dev'
const PY = '/Users/tutaitran/wc26/.venv/bin/python'

const PRE = `
You are a senior ML/stats engineer improving the WC-2026 football predictor. Work ONLY in the git worktree ${WT} (branch improve/accuracy-2026-06). NEVER touch /Users/tutaitran/wc26 (main); NEVER push. Start bash with: cd ${WT}. Run python with ${PY} (cwd=${WT}).
Plan: ${WT}/docs/improve-2026-06/PLAN.md. Frozen batch2 baseline: ${WT}/data/raw/backtest_report.baseline.json (test log_loss 0.8599, brier 0.5049, rps 0.1672, tournament_only 0.9333; an elite-vs-elite favourite-win overconfidence gap of ~6.1pp lives in the report's 'calibration' slice). Batch 1 already shipped: scoreline-distribution reporting, goal-aware base_goals=2.234, an out-of-sample temperature=1.021 applied to W/D/L AFTER outcome_probs and BEFORE the (live-only) market blend, and a reachable+rollback-safe learning gate.
DISCIPLINE: clean tree before coding; read the named files; smallest correct change; PEP8+type hints; add pytest tests; GATE = full pytest green AND backtest compared to the frozen baseline against YOUR item's gate. ADOPT only if the gate passes and no guardrail breaks: overall test log_loss must not rise above 0.8639 (baseline+0.004) for any item. On pass: git commit, report sha. On regression: git reset --hard HEAD && git clean -fdq -e docs, report adopted=false with the measured numbers and why. NEVER adopt a regression. Report concrete before->after numbers in metrics.
`

const items = [
  { label: 'A:elite-desharpen', prompt: `${PRE}
ITEM A — SUPREMACY-CONDITIONAL DE-SHARPEN (fix the elite-vs-elite overconfidence a single global temperature could not reach).
Evidence: both Elo>2000 -> predicted fav-win ~0.47 vs observed ~0.42 (model too confident on the favourite) and predicted draw under-weighted; the global temperature=1.021 only moved the gap 6.4->6.11pp. Item-3's own recommendation: make de-sharpening a function of |supremacy| or an elite bucket, NOT a single scalar.
Read model.py (match_forecast/temper_wdl, match_lambdas, ModelParams), backtest.py (fit_temperature, the 'calibration'/elite slice, run()), forecast.py (param load).
Implement a matchup-conditional effective temperature, e.g. T_eff = T_base + T_elite * g(matchup), where g rises for elite-vs-elite close games (both Elo high and |supremacy| small) and is ~0 for ordinary matches, so non-elite calibration is unchanged. Choose a simple, bounded, fittable parameterization (e.g. a logistic in min(elo_a,elo_b) and/or |supremacy|), add the new param(s) to ModelParams + fitted_params.json, fit them leak-free on a HELD-OUT fold (split the major slice into fit/score sub-folds so the gate is genuinely out-of-sample, addressing the in-sample-gate weakness reviewers flagged in batch 1), and apply T_eff in match_forecast (W/D/L only, after outcome_probs, before the market blend; cannot flip argmax).
GATE: on a genuinely held-out fold, the elite-vs-elite favourite-win overconfidence gap drops from ~6.1pp toward <= 3pp; OVERALL test log_loss does NOT regress (<= baseline+0.004, ideally improves); non-elite calibration unchanged within noise; pytest green. If you cannot reach <=3pp without overall regression, ship the largest elite-gap reduction that still strictly does not regress overall, and report the achieved gap. Commit: 'feat(calib): supremacy-conditional de-sharpen for elite-vs-elite overconfidence'.` },
  { label: '6:negbin-tail', prompt: `${PRE}
ITEM 6 — NEGATIVE-BINOMIAL MARGINAL to tame the favourite blowout tail (PLAN item 6). gamma=0.456 fattens the favourite's lambda and over-produces 4-0/5-0 on mismatches.
Read model.py (scoreline_matrix ~65-78, the Poisson/DC path), simulate.py (the rng.poisson goal draw ~148-154), backtest.py (RPS + totals-band calibration).
Implement optional NegBin marginals (scipy.stats.nbinom, mean lambda, variance lambda + lambda^2/r) behind a ModelParams flag/param r; fit r on historical goals; in simulate.py swap rng.poisson for rng.negative_binomial when enabled. Keep Poisson as default if NegBin does not help.
GATE (decisive, per PLAN R2): RPS on the test set must NOT regress (<= +0.0002, the bake-off noise floor) — if RPS regresses AT ALL, do NOT adopt; P(>2.5 goals) on top-vs-weak mismatch fixtures drops measurably; calibration of the 4+ total-goals band improves; overall log_loss within guardrail; pytest green. Report r, RPS before->after, P(>2.5) before->after. Commit: 'feat(model): optional NegBin marginal to tame favourite blowout tail'.` },
  { label: '9:stage-aware', prompt: `${PRE}
ITEM 9 — STAGE-AWARE SCORING (PLAN item 9). Knockouts are lower-scoring and tighter than group games; a single base_goals misfits both.
Read model.py (match_lambdas, ModelParams), simulate.py (knockout path + shootout.py), backtest.py (stage-conditional fit + a knockout subset slice), forecast.py.
Implement a stage-conditional base_goals (group higher, knockout lower) — a small additive/multiplicative knockout adjustment to base_goals threaded through match_lambdas via a stage argument — fit/bounded on the historical knockout subset. (Travel/altitude/rest are optional and only if cleanly derivable from existing fixture venue+date data; skip if not.)
GATE: knockout-stage total-goals calibration improves vs a flat base_goals on the historical knockout subset; group-stage log_loss NOT worse; overall test log_loss within guardrail; pytest green. Report knockout mean-total before->after vs actual, and group log_loss unchanged. Commit: 'feat(model): stage-aware base_goals (lower-scoring knockouts)'.` },
]

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    item: { type: 'string' }, implemented: { type: 'boolean' }, tests_pass: { type: 'boolean' },
    gate_passed: { type: 'boolean' }, adopted: { type: 'boolean' }, committed_sha: { type: ['string', 'null'] },
    metrics: { type: 'string' }, files_changed: { type: 'string' }, summary: { type: 'string' }, follow_ups: { type: 'string' },
  },
  required: ['item', 'implemented', 'tests_pass', 'gate_passed', 'adopted', 'metrics', 'summary'],
}

const results = []
for (const it of items) {
  log(`Implementing ${it.label} ...`)
  const r = await agent(it.prompt, { label: it.label, phase: 'Implement', schema: SCHEMA })
  results.push(r || { item: it.label, adopted: false, summary: 'agent returned null' })
  if (r) log(`${it.label}: adopted=${r.adopted} gate=${r.gate_passed} ${r.committed_sha ? '@' + r.committed_sha : ''} | ${r.metrics || ''}`)
}
return { adopted_count: results.filter(r => r && r.adopted).length, total: items.length, results }
