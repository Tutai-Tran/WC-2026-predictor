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
