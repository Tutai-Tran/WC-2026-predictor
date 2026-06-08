---
type: wc-memory
---

# WC-2026 Project Memory

Running log of context, decisions, and what was done each task. Newest entries are
appended at the bottom. This file is the project's source of truth for "what did we
do and why".

## Project background (seeded 2026-06-05)

**Goal:** forecast the FIFA World Cup 2026 (match winners, scorelines, goalscorers, progression, champion) as well-calibrated probabilities, fed by scraped data, improving over time, with a standalone Obsidian vault (this vault) and a Streamlit dashboard.

**Architecture built (v1):**
- `wc26/elo.py` — World Football Elo, replayable from a results ledger.
- `wc26/model.py` — Elo-driven goal expectation -> Poisson + Dixon-Coles scoreline matrix -> W/D/L, top scorelines. Supports attacking-xG multipliers (availability).
- `wc26/rules.py` — 2026 rules engine: head-to-head-first group ranking (with recursive re-application), best-8-of-12 thirds.
- `wc26/simulate.py` — Monte Carlo (groups -> Annex C routing -> knockouts with ET + shootouts); conditional sim (fix played matches); `decide_knockout` pure function.
- `wc26/backtest.py` — leak-free calibration (fitted c=219, base 2.63); baselines (uniform/climatology/favourite), bootstrap CI, tournament-only slice, temperature, reliability slope (1.019). Writes fitted_params.json, replayed_elo.json, backtest_report.json.
- `wc26/scorers.py` — anytime-scorer model (goal-share shrunk to role prior + penalty stream).
- `wc26/ingest.py` — loads data/raw into SQLite (48 teams, 72 fixtures, 1246 players); prefers self-replayed Elo.
- `wc26/forecast.py` — orchestrator + CLI (`python -m wc26.forecast`), self-ingests on empty DB.
- `wc26/overrides.py` — manual result/availability entry + parse-back of vault HUMAN override blocks; availability -> attacking-xG multiplier.
- `wc26/update.py` — fast loop (sync overrides -> log calibration -> re-forecast -> vault).
- `wc26/odds.py` — The Odds API winner odds, devigged, as a prediction benchmark only (never betting).
- `wc26/vaultgen.py` — generates the vault (clobber-proof AUTO blocks; HUMAN blocks preserved).
- `wc26/app.py` — read-only Streamlit dashboard (Champion, Matches, Groups, Scorers, Calibration).
- GitHub Actions compute pipeline; repo at github.com/Tutai-Tran/WC-2026-predictor.

**Data sources (one-time snapshot dated 2026-06-04):** Wikipedia (2026 groups/fixtures/squads), martj42/international_results (49,378 historical results), eloratings.net, FIFA/ESPN bracket + 495-combo Annex C routing. Odds live via The Odds API key (in gitignored .env).

**Testing:** 6 adversarial test agents + 4 verification agents + 3 sign-off agents; 10 bugs fixed; 56 tests passing at the start of this session.

**Headline forecast (50k sims, seed 20260611):** Spain ~25%, Argentina ~17%, France ~12%, England ~6%.

**Honest framing:** probabilities not certainties; model runs hotter than the market on the very top Elo teams.

## 2026-06-04 23:14 UTC — Data layer: vault memory, schema v2, knockout fixtures, friendlies

Added the vault memory system (wc26/memory.py + this file); update.py logs each refresh here.
Schema v2: matches gained match_no / home_slot / away_slot; stage now spans friendly, group, R32, R16, QF, SF, Final.
Ingest now also loads 31 knockout fixtures (R32 with real dates+slots, plus the R16->Final tree by match number) and 62 pre-WC warm-up friendlies (33 already played) from data/raw/friendlies.json, scraped by an agent from football365 / SI / ESPN.
Played friendlies are folded into current Elo (backtest.apply_played_friendlies), and the forecast now predicts friendlies. The model's top pick was correct on 21/33 (64%) of the played friendlies.
56 tests pass; calibration unchanged (test log loss 0.861, reliability slope 1.019).

## 2026-06-04 23:19 UTC — Dashboard overhaul + bracket diagram + daily view

Rebuilt the Streamlit dashboard (wc26/app.py) to be user-friendly: emoji title, freshness/STALE banner, headline favourite metric cards, and 7 tabs.
- Daily: all matches (friendly/group/knockout) grouped by date with a date slider + full schedule.
- Matches: split by competition (Group stage / Friendlies / Knockouts) with dates and clear type labels.
- Bracket: R32->Final structure with projected qualifiers, fills with real teams/winners as results arrive.
- Plus Champion & stages, Groups, Scorers, Calibration (now includes model-vs-market).
Forecast payload gained: dates on group matches, friendly forecasts, knockout fixtures (slots + dates + projected teams), and a bracket projection that auto-sharpens toward actual qualifiers (reads conditional-sim probs).
vaultgen now writes WC vault/_bracket.md (Mermaid knockout bracket). Verified the dashboard renders in a real browser (Playwright). 56 tests pass.

## 2026-06-04 23:25 UTC — Live results scraper (ESPN) + self-improving Elo loop

Added wc26/scrape.py: reads completed friendly + World Cup results from ESPN's public site JSON (keyless, valid public data) over a rolling 12-day window, records genuinely new results into the results ledger (dedup by teams+date), and recompute_elo() rebuilds current Elo = base (results.csv replay) + the ledger, then updates the ratings table.
backtest now also writes base_elo.json (stable anchor). update.py refresh loop now: sync overrides -> scrape results -> recompute Elo -> log calibration -> re-forecast -> regenerate vault. This is the real self-improving loop: new results flow in and sharpen the ratings/forecast automatically.
First live run added 26 new results; forecast stays sane (Spain ~23%, sum 1.0). 56 tests pass.

## 2026-06-04 23:32 UTC — LLM news agent (Claude CLI + web) for injuries/availability

Added wc26/news.py: an agent that calls the local Claude CLI (Max subscription) with WebSearch/WebFetch to research current injuries/suspensions per team and return structured, source-quoted availability events. Guards: player must be in the official squad (enum), source quote required, a team's prior LLM events are replaced each scan (no double-count), and a rotation marker advances coverage. Validated live: it correctly found Spain's Lamine Yamal and Nico Williams doubtful (hamstrings) with ESPN sources. Events feed the availability -> attacking-xG channel, so scraped injuries move the forecast. Wired into update.py (rotating batch of 3 teams per refresh, best-effort/never blocks). 59 tests pass.

## 2026-06-04 23:34 UTC — Vault match notes split by type + dated

vaultgen now writes match notes into Matches/Group, Matches/Friendly, Matches/Knockout, each with the date and a clear type label in the body and frontmatter (filenames are date-prefixed for sorting). Knockout notes show slots + projected teams + result-TBD. Removed the old flat match notes. HUMAN post-match blocks preserved on regeneration. Counts: 72 group, 62 friendly, 31 knockout. 59 tests pass.

## 2026-06-04 23:44 UTC — Fixed 7 issues from the 4-agent session test

HIGH: (1) scraper dedup never matched because add_result stored today's date not the match date -> ledger duplicated every run and recompute_elo double-counted; add_result now takes played_on and stores the real date (run2 now adds 0). (2) scraped friendlies were labelled 'FIFA World Cup' -> Elo used K=60 not K=20; add_result now derives tournament from stage. (3) bracket_projection could assign the same team to 1X and 2X of a group; runner-up now excludes the winner (regression test added).
MEDIUM: (4) friendly-opponent seed ratings were dated at the future TOURNAMENT_START so recompute_elo was ignored; now seeded at 2000-01-01 so the recompute wins. (5) vaultgen meta query now selects the latest rating (MAX valid_from) like load_tournament.
LOW: (6) news._extract_json_array now uses raw_decode (robust to stray brackets in prose). (7) autostart dashboard binds 127.0.0.1 (matches the install message).
62 tests pass.

## 2026-06-04 23:49 UTC — Visual bracket diagram + Monte Carlo CIs + ESPN parser tests

Dashboard Bracket tab now renders a graphviz knockout diagram (R32->Final, projected qualifiers) in addition to the text breakdown and the vault Mermaid. Champion tab shows a Monte Carlo 95% CI per team (honest uncertainty). Refactored scrape.fetch_espn into a pure parse_espn() with unit tests (completed-only, alias mapping, bad-score handling). Verified the bracket renders in a real browser (0 console errors). 65 tests pass.

## 2026-06-04 23:53 UTC — Availability tab + friendly-accuracy validation panel

Added a dashboard Availability tab showing scraped injuries/availability (news-LLM + vault + manual), and a Calibration-tab panel showing our forecast vs actual played-friendly results (top-pick accuracy + log loss) as a real out-of-sample check. Launched a background news scan to populate availability across teams. 65 tests pass; dashboard renders clean.

## 2026-06-05 00:07 UTC — Session wrap-up: injuries scraped, vault refreshed, auto-start next

News agent scanned 5 teams and found real injuries (South Africa: Mbokazi suspended + 3 doubtful; Canada: Davies + others doubtful), now reflected in the forecast (availability -> attacking-xG) and the country vault notes. Regenerated the 50k forecast + full vault. SUMMARY.md added. 65 tests pass. Next: activate auto-start so the dashboard + self-improving refresh loop persist across reboots.

## 2026-06-05 00:11 UTC — Auto-start + persistence finalized

macOS TCC blocks launchd from reading the project under ~/Documents (Operation not permitted), so the login-time LaunchAgent needs a one-time Full Disk Access grant for /bin/bash. Removed the broken auto-loading plist to avoid boot errors; updated install_autostart.sh + SUMMARY with the FDA step. For now the dashboard + 3-hourly self-improving refresh loop run via 'nohup bash scripts/start.sh' (works without special permission; persists this session on http://localhost:8501). 65 tests pass.

## 2026-06-05 00:14 UTC — Team explorer tab

Added a Team explorer tab (selectbox) to drill into any country: champion/final/advance probs, Elo, group, FIFA rank, current scraped injuries, all its matches (group + friendly) with predictions, and most-likely scorers. Verified all 9 tabs render with 0 console errors. 65 tests pass.

## 2026-06-05 00:21 UTC — Session complete — strong, tested, self-running state

Final 2-agent sign-off: GREEN on correctness (all session fixes verified holding; 65 tests; 9 dashboard tabs render; secrets safe; payload/Elo/bracket/availability all correct). Fixed the one finding: gitignored runtime logs and made the refresh loop auto-commit+push, so the tree stays clean and the self-improving updates are saved+pushed automatically.
Delivered this session: vault memory; Daily view + match dates/types/split; 62 warm-up friendlies (predicted + Elo-feeding); knockout fixtures + projected bracket (graphviz + Mermaid) that fills with winners; ESPN live results scraper + self-improving Elo loop; LLM news agent (Max subscription, web) for injuries -> forecast; odds blend; Availability + Team explorer tabs; Monte Carlo CIs; friendly-accuracy panel; auto-start (run-now works; boot-start needs one-time Full Disk Access since the project is in ~/Documents); SUMMARY.md; four multi-agent test rounds (16 agents) with all findings fixed.
The dashboard + 3-hourly self-improving refresh (scrape -> news -> recompute Elo -> re-forecast -> regenerate vault -> commit+push) are running and will continue until scripts/stop.sh.

## 2026-06-05 00:59 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 59, 'teams_updated': 76}; news: {'teams_scanned': ['Bosnia and Herzegovina', 'Qatar', 'Switzerland'], 'events': 7}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T005940Z.

## 2026-06-05 04:07 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Brazil', 'Morocco', 'Haiti'], 'events': 8}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T040751Z.

## 2026-06-05 09:57 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Scotland', 'United States', 'Paraguay'], 'events': 5}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T095750Z.

## 2026-06-05 10:19 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Australia', 'Turkey', 'Germany'], 'events': 14}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T101900Z.

## 2026-06-05 11:30 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Curacao', 'Ivory Coast', 'Ecuador'], 'events': 1}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T113038Z.

## 2026-06-05 11:47 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Netherlands', 'Japan', 'Sweden'], 'events': 3}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T114726Z.

## 2026-06-05 12:19 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': ['Tunisia', 'Belgium', 'Egypt'], 'events': 8}; overrides synced: 0 events; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T121917Z.

## 2026-06-05 13:15 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 61, 'teams_updated': 76}; news: {'teams_scanned': [], 'events': 0}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 0, 'accuracy': None, 'brier': None, 'log_loss': None}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T131547Z.

## 2026-06-05 13:32 UTC — Automated refresh (update.py)

scraped results: {'added': 1}; elo: {'recomputed': True, 'ledger_matches': 62, 'teams_updated': 76}; news: {'teams_scanned': ['Iran', 'New Zealand', 'Spain'], 'events': 8}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 0, 'accuracy': None, 'brier': None, 'log_loss': None}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T133215Z.

## 2026-06-05 — Session summary (full record in [[sessions/2026-06-05 session]])

Delivered this session: played/today row highlights + Amsterdam kickoff times; better score model (head-to-head term + supremacy-dependent goals γ=0.454 + outcome-consistent likely score + "our call" ✅/❌); front-page accuracy scorecard; cloud hosting Option B2 (Mac publishes a WAL-merged wc26.db to the `live` GitHub Release, Streamlit Cloud app pulls it on a 60s TTL — reachable when the Mac is off, updates the moment it's on); UX redesign.

Biggest piece — the **self-learning post-match loop** (wired into every update.py refresh, forever): snapshot frozen pre-match predictions (leak-free, before results enter Elo) → grade once played → LLM post-mortem on wrong picks (enum-validated, `variance` upsets excluded) → aggregate ranked systematic biases → regenerate LESSONS.md + 🧠 Learning tab → propose **candidate** parameter nudges once a factor is tagged by ≥8 distinct wrong matches (audit-only, never auto-applied). Key decision: **parameters are frozen during the tournament** because single-tournament learning chases noise (this is how the old flat goal_scale=1.10 overfit); applying a nudge is gated behind the deferred phase 3b out-of-sample re-fit gate. "LLM proposes, the math disposes." See [[LESSONS]].

Also fixed a **live-down `sqlite3.DatabaseError`**: cloud-side DB-sync bug (unvalidated downloads + stale WAL sidecars after the file swap). Fix = validate snapshot before install + clear sidecars + graceful self-healing degradation (commit 973bf74). Verified recovered in-browser. 103 tests pass. All pushed to main.

## 2026-06-05 16:39 UTC — Automated refresh (update.py)

scraped results: {'added': 4}; elo: {'recomputed': True, 'ledger_matches': 66, 'teams_updated': 76}; news: {'teams_scanned': ['Cape Verde', 'Saudi Arabia', 'Uruguay'], 'events': 4}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 0, 'accuracy': None, 'brier': None, 'log_loss': None}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T163939Z.

## 2026-06-05 19:46 UTC — Automated refresh (update.py)

scraped results: {'added': 7}; elo: {'recomputed': True, 'ledger_matches': 73, 'teams_updated': 76}; news: {'teams_scanned': ['France', 'Senegal', 'Iraq'], 'events': 2}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 0, 'accuracy': None, 'brier': None, 'log_loss': None}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T194614Z.

## 2026-06-05 22:54 UTC — Automated refresh (update.py)

scraped results: {'added': 1}; elo: {'recomputed': True, 'ledger_matches': 74, 'teams_updated': 76}; news: {'teams_scanned': ['Norway', 'Argentina', 'Algeria'], 'events': 17}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 0, 'accuracy': None, 'brier': None, 'log_loss': None}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260605T225409Z.

## 2026-06-06 02:01 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 76, 'teams_updated': 76}; news: {'teams_scanned': ['Austria', 'Jordan', 'Portugal'], 'events': 7}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 99}; graded: {'graded': 2, 'wrong': 1}; post-mortems: {'analyzed': 1, 'errors': 0}; lessons: {'n': 2, 'accuracy': 0.5, 'brier': 0.4701, 'log_loss': 0.785}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T020155Z.

## 2026-06-06 05:11 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 78, 'teams_updated': 76}; news: {'teams_scanned': ['DR Congo', 'Uzbekistan', 'Colombia'], 'events': 3}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 97}; graded: {'graded': 1, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 3, 'accuracy': 0.667, 'brier': 0.488, 'log_loss': 0.8214}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T051101Z.

## 2026-06-06 08:19 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 78, 'teams_updated': 76}; news: {'teams_scanned': ['England', 'Croatia', 'Ghana'], 'events': 4}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 96}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; lessons: {'n': 3, 'accuracy': 0.667, 'brier': 0.488, 'log_loss': 0.8214}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T081949Z.

## 2026-06-06 10:58 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 78, 'teams_updated': 76}; news: {'teams_scanned': ['Panama', 'Mexico', 'South Africa'], 'events': 11}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 96}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 3, 'accuracy': 0.667, 'brier': 0.488, 'log_loss': 0.8214}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T105855Z.

## 2026-06-06 14:09 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 80, 'teams_updated': 76}; news: {'teams_scanned': ['South Korea', 'Czech Republic', 'Canada'], 'events': 10}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 96}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 3, 'accuracy': 0.667, 'brier': 0.488, 'log_loss': 0.8214}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T140900Z.

## 2026-06-06 16:53 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 82, 'teams_updated': 76}; news: {'teams_scanned': ['Bosnia and Herzegovina', 'Qatar', 'Switzerland'], 'events': 5}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 96}; graded: {'graded': 1, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 4, 'accuracy': 0.75, 'brier': 0.4166, 'log_loss': 0.7292}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T165314Z.

## 2026-06-06 20:02 UTC — Automated refresh (update.py)

scraped results: {'added': 4}; elo: {'recomputed': True, 'ledger_matches': 86, 'teams_updated': 76}; news: {'teams_scanned': ['Brazil', 'Morocco', 'Haiti'], 'events': 6}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 95}; graded: {'graded': 1, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 5, 'accuracy': 0.8, 'brier': 0.3669, 'log_loss': 0.6637}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T200251Z.

## 2026-06-06 21:29 UTC — Automated refresh (update.py)

scraped results: {'added': 4}; elo: {'recomputed': True, 'ledger_matches': 90, 'teams_updated': 76}; news: {'teams_scanned': ['Scotland', 'United States', 'Paraguay'], 'events': 2}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 94}; graded: {'graded': 2, 'wrong': 1}; post-mortems: {'analyzed': 1, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 7, 'accuracy': 0.714, 'brier': 0.4427, 'log_loss': 0.7639}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260606T212901Z.

## 2026-06-07 00:37 UTC — Automated refresh (update.py)

scraped results: {'added': 6}; elo: {'recomputed': True, 'ledger_matches': 96, 'teams_updated': 76}; news: {'teams_scanned': ['Australia', 'Turkey', 'Germany'], 'events': 10}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 92}; graded: {'graded': 3, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 10, 'accuracy': 0.8, 'brier': 0.3698, 'log_loss': 0.6655}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T003727Z.

## 2026-06-07 03:46 UTC — Automated refresh (update.py)

scraped results: {'added': 2}; elo: {'recomputed': True, 'ledger_matches': 98, 'teams_updated': 76}; news: {'teams_scanned': ['Curacao', 'Ivory Coast', 'Ecuador'], 'events': 3}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 89}; graded: {'graded': 2, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 12, 'accuracy': 0.833, 'brier': 0.3185, 'log_loss': 0.59}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T034601Z.

## 2026-06-07 06:53 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 98, 'teams_updated': 76}; news: {'teams_scanned': ['Netherlands', 'Japan', 'Sweden'], 'events': 5}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 87}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 12, 'accuracy': 0.833, 'brier': 0.3185, 'log_loss': 0.59}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T065340Z.

## 2026-06-07 10:03 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 98, 'teams_updated': 76}; news: {'teams_scanned': ['Tunisia', 'Belgium', 'Egypt'], 'events': 3}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 87}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 12, 'accuracy': 0.833, 'brier': 0.3185, 'log_loss': 0.59}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T100301Z.

## 2026-06-07 19:26 UTC — Automated refresh (update.py)

scraped results: {'added': 5}; elo: {'recomputed': True, 'ledger_matches': 103, 'teams_updated': 76}; news: {'teams_scanned': ['Iran', 'New Zealand', 'Spain'], 'events': 5}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 87}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 12, 'accuracy': 0.833, 'brier': 0.3185, 'log_loss': 0.59}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T192632Z.

## 2026-06-07 22:36 UTC — Automated refresh (update.py)

scraped results: {'added': 5}; elo: {'recomputed': True, 'ledger_matches': 108, 'teams_updated': 76}; news: {'teams_scanned': ['Cape Verde', 'Saudi Arabia', 'Uruguay'], 'events': 1}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 87}; graded: {'graded': 3, 'wrong': 1}; post-mortems: {'analyzed': 1, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 15, 'accuracy': 0.8, 'brier': 0.32, 'log_loss': 0.5948}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260607T223604Z.

## 2026-06-08 01:42 UTC — Automated refresh (update.py)

scraped results: {'added': 1}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['France', 'Senegal', 'Iraq'], 'events': 7}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 84}; graded: {'graded': 1, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T014213Z.

## 2026-06-08 04:51 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['Norway', 'Argentina', 'Algeria'], 'events': 15}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 83}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T045133Z.

## 2026-06-08 07:57 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['Austria', 'Jordan', 'Portugal'], 'events': 4}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 83}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T075736Z.

## 2026-06-08 11:06 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['DR Congo', 'Uzbekistan', 'Colombia'], 'events': 3}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 83}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T110628Z.

## 2026-06-08 14:14 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['England', 'Croatia', 'Ghana'], 'events': 8}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 83}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T141449Z.

## 2026-06-08 17:21 UTC — Automated refresh (update.py)

scraped results: {'added': 0}; elo: {'recomputed': True, 'ledger_matches': 109, 'teams_updated': 76}; news: {'teams_scanned': ['Mexico', 'South Africa', 'Panama'], 'events': 14}; overrides synced: 0 events; prediction snapshots: {'snapshotted': 83}; graded: {'graded': 0, 'wrong': 0}; post-mortems: {'analyzed': 0, 'errors': 0}; param adoption: {'adopted': 0, 'reason': 'insufficient graded matches', 'n': 0}; lessons: {'n': 16, 'accuracy': 0.812, 'brier': 0.3098, 'log_loss': 0.5813}; running calibration: {'n': 0}; vault: {'countries': 48, 'groups': 12, 'matches': {'group': 72, 'friendly': 62, 'knockout': 31}, 'bracket': 'written'}; run_id run-20260608T172117Z.
