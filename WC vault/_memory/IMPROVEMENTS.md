---
type: wc-improvement-plan
created: 2026-06-09
status: IMPLEMENTED 2026-06-09 (see "Implementation outcome" below)
---

# Accuracy & Efficiency Audit (2026-06-09, 2 days before kickoff)

## Implementation outcome (same day)

Everything below was implemented, tested (137 tests pass), reviewed by a 4-agent
panel (statistician, code reviewer, ops engineer, football domain expert), and the
panel's feedback was implemented in a second pass. Key deltas vs the original plan:

- **Adoption gate**: group-stage thresholds 12/5 as planned, PLUS an anti-ratchet
  (a pool needs 12 newly graded matches between adoptions) and stage-objective
  evidence pools (pooled by the match's actual stage, not the LLM's segment tag).
  Matchday-3 group draws are excluded from 'draw' evidence (draw-suits-both games).
- **Rollback**: judged only on matches graded AFTER the adoption (selection-bias fix),
  threshold 8 new matches.
- **Draw fix verdict**: rho_friendly nudges change graded-friendly log loss by
  ~0.0001, far below the 0.005 margin. No change made: the bias the LLM panel
  flagged does not survive the math. Working as designed.
- **Shootout tilt**: fitted 0.145 (se 0.058) on 540 shootouts since 1990 (favourites
  won 54.8%); configured 0.12 (shrunk toward coin-flip). Old 0.05 was too timid.
- **Host advantage**: finals-only refit (Nations League excluded, n=200) = 129.9 Elo.
  Per-host config: Mexico 110 / Canada 90 / USA 75 (crowd dilution). Knockout venues
  now carry host advantage through the whole bracket (Azteca R16, Vancouver R16,
  QF-Final all in USA) instead of stopping after R32.
- **Fitted params**: time-decay refit adopted (half-life 10y): c=208.86,
  base_goals=2.19, gamma=0.456, test LL 0.8597 (was 0.8600), now the backtest default.
- **Challenger**: bivariate Poisson LOST to the champion (paired bootstrap CI
  excludes 0 in champion's favour). Champion stays; harness stays for future challengers.
- **Injury handling**: expected_return + 72h news expiry live; failed news scans now
  PRESERVE existing events (only a successful scan replaces them); confidence =
  source reliability (suspended 1.0 / out 0.9 / doubtful 0.85); rotation is
  kickoff-aware (teams playing within 36h jump the queue), 10 teams/cycle in-tournament.
- **Efficiency**: input-fingerprint skip (markers excluded, expiry-aware, odds
  included), WAL checkpoint, busy_timeout, atomic+fsync param writes.
- **Ops**: git push had been silently failing since June 8 (remote diverged); resolved,
  and the refresh loop now fetches/rebases and LOGS git failures to logs/git.log.
  start.sh also kills orphaned streamlit instances before binding :8501.

Full system review (3 parallel review agents over model stack, learning loop, dashboard/DB; key findings hand-verified in code). Status: system is healthy, well-tested, and honestly calibrated. The biggest gap is not the math, it is that the self-learning loop is throttled so hard it will barely act during the tournament, and several data streams (availability, h2h, odds) are collected but never fed back into learning.

## P1: Do before June 11 (kickoff)

1. **Unlock the learning loop for the group stage** (`learn.py:430,374`)
   - `_MIN_EVAL = 20` tournament-graded matches + `_QUORUM = 8` wrong-per-factor means the first possible parameter adoption is ~June 16, after a quarter of the group stage. With 84% accuracy, only ~11 wrong picks are expected in all 72 group matches, so quorum 8 on a single factor may NEVER fire.
   - Fix: `_MIN_EVAL = 12` (one full matchday) and `_QUORUM = 5` during the group stage, revert after June 27. Keep the validation margin (`_NET_MARGIN`) as is, that is the real safety.

2. **Log availability multipliers + h2h delta into prediction snapshots** (`learn.py:71`, from forecast payload)
   - Post-mortems currently can't see that a lambda was already injury-reduced, so they misattribute misses to `goal_volume` instead of `availability`. Add `mult_home`, `mult_away`, `h2h_delta` to `inputs_json`. 3 lines, makes every tournament post-mortem more truthful.

3. **Expire stale injury events** (`overrides.py` / forecast load)
   - `expected_return` is stored but never checked. A player marked out on June 12 who returns June 15 stays "out" forever unless his team is rescanned. Auto-treat events as expired when `expected_return < today`, and hard-expire news-llm events older than ~72h.

4. **Speed up news rotation during the tournament** (`update.py:78`, limit param)
   - 4 teams per 3h refresh = 36h to cover 48 teams. During group stage raise to 10-12 per cycle (12-15h full rotation). Config-only change.

5. **Pre-test the draw fix offline** (`learn.py`, rho)
   - Draw underprediction is the strongest detected bias (strength 0.459). Run the adoption gate's log-loss check manually with `rho -0.06 -> -0.045`; adopt only if the margin passes. "LLM proposes, the math disposes" still holds, this just runs the math now instead of mid-tournament.

6. **Widen ESPN scrape window during tournament** (`scrape.py`, days_back 12 -> 30) so late corrections are never silently lost.

7. **Fix stale-banner age math** (`app.py:234`): `.date() - .date()).days * 24` loses hour precision; use `total_seconds()/3600`.

8. **Add "Kyrgyz Republic" (and similar) to ESPN_ALIASES** (`scrape.py:23`), and log unmatched_teams into CONTEXT.md each cycle so alias drift is visible.

## P2: During group stage (June 11-27)

9. **Decide the odds blend's role** (`odds.py:98`, `forecast.py`): `blended_champion()` is display-only; the stored headline forecast is pure model. PLAN.md said the published forecast should be the blend. Either wire the blend into the stored payload or document pure-model as intentional. Also: use model-vs-market drift as a sanity check inside the adoption gate (reject a parameter change that widens disagreement with the market).
10. **Segment home advantage by context** (`backtest.py:252`): home_adv_elo ~118 was fit on all non-neutral history; 2026 is mostly neutral venues with 3 co-hosts. Refit per tier (WC vs qualifier vs friendly) and consider a partial (0.6-0.8x) host boost.
11. **Conditional Elo mid-tournament**: conditional re-simulation fixes played results but uses pre-tournament Elo for later rounds. Fold played group results into ratings before simulating remaining matchdays.
12. **Adoption rollback**: if calibration regresses sharply (>0.1 log loss) right after an adoption, auto-revert and log it.
13. **Input-hash skip** (`forecast.py`): skip the 50k-run sim when (Elo, availability, played-matches) hash is unchanged; saves most idle refreshes. Also acceptable: 10k runs pre-kickoff, 50k after.
14. **WAL checkpoint** after heavy write cycles (`db.py`); WAL is already ~same size as the DB.

## P3: Post-tournament / v2 (machine-learning upgrades)

15. **Challenger harness**: implement bivariate Poisson (e.g. penaltyblog) as a challenger, evaluated on the same held-out set with paired bootstrap; adopt only on a significant log-loss win. This is the missing piece of "keep learning from itself" at the model-selection level.
16. **Shootout tilt validation**: SHOOTOUT_FAVOURITE_TILT=0.05 is untested; fit on 2010-2022 shootout history (likely near-random).
17. **Time-decay weighting** in the backtest loss so 2006 matches don't count like 2024 ones.
18. **Availability elasticity**: replace the fixed 0.4/0.6 caps and FW/MF/DF priors with squad-specific goal-share elasticity learned from lineups.
19. **Odds-movement signal**: public_benchmark history (39 snapshots) is stored but unused; market drift is a free re-evaluation signal.
20. **News confidence learning**: confidence is hardcoded 0.8; grade news events against whether the player actually played and learn per-source reliability.

## Validated as solid (no action)

- Vault HUMAN override parse-back: robust, tested, feeds the model correctly.
- Leak-free backtest + frozen-snapshot grading: sound design.
- Parameter freeze with gated adoption: right idea, just tuned too conservatively for a 72-match window (see P1.1).
