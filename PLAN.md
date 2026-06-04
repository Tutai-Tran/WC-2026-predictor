# World Cup 2026 Prediction System: Plan + Full Context

> Self-contained handoff document. It holds the final plan (v4, after nine review passes plus a consolidation), every decision made, the research context behind those decisions, and the review history. A fresh session can pick this up cold and start building.

- Status: planning complete, not yet built.
- Project location: `/Users/tutaitran/Documents/own projects/WC-2026-predictor/`
- Knowledge base: a standalone Obsidian vault named `WC vault` inside this folder, separate from existing vaults.
- Created from research + adversarial review on 2026-06-04.

---

## 1. TL;DR: what we are building

A system that forecasts the FIFA World Cup 2026: match outcomes and scorelines, most likely goalscorers, and progression (Round of 32, QF, SF, Final, and the winner), as well calibrated probabilities. It keeps a continuously updated knowledge base for every country in the `WC vault`, is fed by scrapers plus an LLM agent that reads football news, and improves its own predictions over time as results arrive. Predictions are the product; there is no betting layer.

Core engine: an Elo-driven goal-expectation model feeding a Poisson scoreline model, blended with vig-removed market odds for calibration, run through a Monte Carlo simulation of the real 48-team 2026 format. A correct forecast works on manually entered data first; scraping and the deeper self-training layer build on top so a matchday forecast never stalls.

---

## 2. Decisions log (locked with the user)

1. No fixed deadline. Build to a working, complete product; treat it as a living, self-feeding system that is never truly finished.
2. No paid feeds. The only external services are free: a free LLM key (Claude Haiku) for news parsing and The Odds API free tier for odds.
3. Data via scraping football sources (433, news articles, plus structured public sources), with an LLM agent turning unstructured news into structured facts.
4. No betting or staking layer. Odds are used ONLY as a prediction signal and calibration benchmark, never for bet sizing.
5. Odds ARE included (they improve prediction): The Odds API free tier (500 credits/month, no card), devigged and blended into the forecast.
6. Dashboard: Streamlit (read-only, local/private).
7. Automation: GitHub Actions cron for compute, with scrapers running on the Mac (see architecture). Local cron as fallback.
8. Knowledge base: a standalone Obsidian vault named `WC vault` inside the project folder; the pipeline never writes into the user's main `The vault/`.
9. Prediction quality is the number one goal; the system must improve itself over time.

---

## 3. The plan (v4)

### 3.1 Operating model: living, self-improving, no fixed deadline

The tournament calendar (kickoff 11 June 2026, final 19 July 2026, hosts USA/Canada/Mexico) sets cadence: light now, daily during the group stage, re-simulation after every matchday. Real self-improvement signal only exists once results arrive in mid-June, so the heavy learning machinery is sequenced late, not first.

Two timescales for "improve itself": ratings update fast from each result; structural model parameters change slowly and only when a proper statistical test says a change is real.

### 3.2 Honesty is a design requirement (matters more without a betting market to argue with)

The honest goal is well calibrated probabilities (when we say 60%, it happens about 60% of the time), not certainty. Enforced three ways:
- Data-freshness / STALE banner on every forecast view (highest-leverage honesty feature): each forecast shows what it ran on (latest result ingested, squad-lock date, count of known vs unknown availability facts, red STALE flag if any input is older than a threshold). A forecast on a half-failed scrape must look different from one on fresh, corroborated data.
- Market odds as the primary calibration benchmark and a blend input: vig-removed (devigged) odds from The Odds API free tier are the strongest single signal. The published forecast is a model-plus-market blend (weight tuned where historical odds allow, otherwise set conservatively toward the market for 1X2 and outrights), and model-vs-market divergence is logged per match. Odds are used ONLY for prediction and calibration, never for betting. A scraped public-model number (for example Opta) is a secondary non-promoting sanity check ("we disagree with the best public estimate by more than X, surface for review"); we never train toward either.
- Input quality flows into forecast uncertainty: stale, single-source, or low-confidence inputs WIDEN the predictive distribution (a flatter W/D/L triple), and a per-match data-confidence badge says why.

UX rules: never present a single predicted score or single scorer as certainty (the modal scoreline is typically 8 to 12 percent likely); lead each match view with the W/D/L triple and the top 3 scorelines with probabilities; goalscorers shown as P(anytime scorer) for the top N with a caption that even the top pick misses about half the time; a calibration view reachable everywhere.

### 3.3 The model (prediction core)

Engine. Primary and only v1 engine: Elo-driven goal expectation. Derive each team's expected goals from the Elo rating difference via a calibrated link, estimating a few global parameters (Elo-to-goals link, low-score correction rho, home/host term, time decay xi). This is identifiable on sparse international data and generalises across confederations. Do not gate v1 on a multi-model bake-off; Dixon-Coles, hierarchical Bayesian Poisson, and an ML ensemble are LATER challengers (see self-improvement), each adopted only if it wins a proper out-of-sample significance test. The published match probabilities are a blend of this engine with vig-removed market odds (see Honesty); the pure-model output is retained for diagnostics and for the model-vs-market divergence log.

Elo is replayable, not mutated. Elo = deterministic function of (one-time seed + ordered results + frozen params), recomputed from the results ledger each run, never edited in place. A later-corrected result self-heals on the next recompute. Results are the highest-stakes fact: require two independent sources to agree (or a manual confirmation) before a scoreline updates Elo; disagreements go to a human-review queue.

Player goal-share table (shared dependency, built with the engine). A small table of each player's shrunk share of team goals (toward a position/role prior) is built alongside the Elo engine because BOTH the availability channel and the goalscorer model consume it. Building it early removes an ordering hazard.

How form and availability enter the core (this is the channel all the scraping feeds). Availability applies a BOUNDED multiplicative adjustment to team expected goals using the player goal-share table: a key attacker out reduces attacking xG by that player's estimated share; defensive absences raise opponent xG. The total adjustment is capped, and large swings route to human review. Form is carried by Elo (its job); any extra short-term form term must be small, bounded, and proven to add out-of-sample skill over Elo-alone before inclusion (no double-counting).

Tournament rules engine (highest bug risk, fully unit-tested pure functions). Within-group ranking is head-to-head first (points, then head-to-head points/GD/goals on the mini-table of only the tied teams, then overall GD, overall goals, fair play, FIFA ranking). Best-8-of-12 thirds ranked correctly AND routed via the official Annex C combination lookup table (495 combinations, officially defined) to the right Round of 32 slots. Knockouts cannot draw: 90 minutes, then extra time as a shortened independent draw (xG scaled by about 1/3), then a near 50/50 shootout with at most a tiny favourite tilt; never advance the higher seed on a draw. Conditional simulation is first-class for mid-tournament re-sims (fix played matches, simulate the remainder, never re-randomise completed fixtures). ET/penalty parameters are frozen priors validated on historical knockouts, never tuned online.

Goalscorers. P(anytime scorer) from team xG x player goal share x (expected minutes / 90) x opponent defensive factor, plus a separate penalty stream (P(pen awarded) x P(taker on pitch) x conversion about 0.75). Shrink club-derived shares toward a position/role prior; never trust tiny-cap samples. Validate against realised scorers with a proper scoring rule. Lowest-confidence tier; ship the honesty caption even if the model is crude.

Simulation. Monte Carlo at least 50k runs; fixed, logged RNG seed; report Monte Carlo standard error; common random numbers when comparing model versions so measured "improvement" is not RNG jitter.

### 3.4 Self-improvement, right-sized to the data

Two timescales, because re-tuning 4 global parameters on 4 to 16 new matches per matchday is noise:
- Fast loop (v1, every matchday): ingest results, update Elo, re-run the simulation, refresh the post-hoc calibration map, and LOG each pre-match forecast's Brier/log loss against the realised result. The running calibration trajectory is the honest "is it getting better" signal. This delivers the large majority of real in-tournament improvement and is small.
- Slow loop (v2): re-fit structural parameters (xi, rho, host, link) on the LARGE historical corpus on a slow cadence, not per matchday. A challenger model (DC, hierarchical Bayes, or ML ensemble) replaces the champion ONLY via a paired, significance-aware test (paired bootstrap CI or Diebold-Mariano with common random numbers) clearing a stated minimum margin over a powered sample (hundreds of pre-tournament held-out matches, not one matchday). Define "beats the champion" numerically wherever it appears. Hysteresis prevents ping-ponging. During the tournament the default is "champion stays unless clearly broken."

The ML ensemble is an optional research spike, not a milestone: sparse noisy scraped features make overfitting the likely outcome, so it must use nested CV, leak-free time-series splits, monotonic constraints, and an ablation proving any feature (especially scraped sentiment) earns its place; the interpretable core stays the permanent default.

### 3.5 Data acquisition: manual-entry-first, scraping as the accelerator

Manual entry is a first-class path, not a fallback afterthought. A tracked command/insert for "Group A match 3: 2-1" and for the handful of genuinely material injuries means the forecast core works from day one and never stalls when a scraper breaks. For ~104 matches this is minutes per matchday.

Scraping architecture (the reviewers' key correction): scrapers run on the Mac, compute runs on GitHub Actions, synced via the git data repo. GitHub Actions datacenter IPs are blocked by FBref/Transfermarkt/Cloudflare (demonstrated live: a 403 to a datacenter fetch), so scraping from Actions does not work. The Mac has a residential IP, can run headed/headless Chrome (soccerdata's FBref path now drives Chrome and ships proxy support precisely because blocking is expected), and persists a cache. The split is drawn at "does this touch a hostile site": Mac scrapes/validates/parses and commits structured facts + cache to the repo; Actions pulls, updates Elo, simulates, runs the gate, generates vault AUTO blocks, commits/pushes. Either side can run the whole pipeline if the other is down; staleness is flagged.

Source choices (favour low-block, structured sources over brittle prose):
- Historical results seed + backtest: a free static dataset (the community-maintained `martj42/international_results` GitHub repo), one-time load, re-pulled post-tournament to true-up. Being GitHub-hosted it is not IP-blocked, so it can load from either the Mac or Actions; it is seed-only and lags during the tournament.
- Live results/fixtures: two peer public sources read from the Mac, ESPN's public site JSON and the Wikipedia tournament page, with two-source agreement required before an Elo update. (These are keyless, contract-free public site data: scraping, not a paid API integration, consistent with the no-paid-feed rule.)
- Team strength seed: scrape eloratings.net ONCE now, commit the snapshot, never depend on it live (Elo is self-maintained after seeding).
- Injuries/availability: Transfermarkt per-club injury/suspension pages and Wikipedia squad pages (low anti-bot, semi-structured) feed the LLM parser. 433 and open-web news are a nice-to-have/manual-paste source, not the reliable backbone (JS-heavy, anti-bot, copyright-muddy, low yield).
- Lineups: treat as post-match reconciliation (who actually played, for goalscorer validation) rather than a tight pre-match real-time scrape.
- Odds (prediction signal + calibration benchmark, NOT betting): The Odds API free tier (500 credits/month, no card, no payment), soccer 1X2 + tournament-winner markets, fetched from the Mac and cached aggressively (odds move slowly, poll a couple of times pre-match). Strip the bookmaker margin (devig, proportional method) before blending. This is the single cleanest data source (a keyed free API, not a fragile scrape), so it is the first accelerator to wire on top of the manual-entry core.

Robustness: schema validation (columns, row-count floors, value ranges) AND explicit challenge/interstitial detection (a rendered Cloudflare page is a block, not data), retries with backoff on transient errors, hard-fail (not retry) on layout change, recorded `source` per value, cross-source reconciliation on results, a degraded mode (fall back to last-good snapshot + manual entry, mark STALE, still publish), and failure alerting. Respect robots.txt, FBref's 10 req/min limit (so the delay is about 6s, not 1 to 2s), low cadence on Transfermarkt from the residential IP, attribute StatsBomb/FBref, store only short source-quotes and links from news (never full article bodies). Pin scraper library versions (no auto-upgrade mid-tournament).

Tooling note: pin a Python 3.12 or 3.13 virtual environment on day one and verify `penaltyblog` + numpy/scipy install BEFORE writing model code (the system Python 3.14 fights the scientific-stack wheels).

### 3.6 Knowledge layer: SQLite source of truth + the WC vault

Two kinds of facts, routed by origin so the model never parses prose for its numbers, yet the vault is genuinely where country info lives and is updated:
- Machine facts (ratings, fixtures, results, parsed events, model outputs): SQLite owns them; the vault shows them in marker-delimited AUTO blocks.
- Human facts (your judgment, a manual availability/result override, a presser note): the vault owns them in untouched HUMAN blocks that sync BACK into the DB as `manual_override` events.

v1 scope: one-directional generation (DB to read-only per-country/group/match notes + `_forecast.md`) PLUS the HUMAN-override block parsed back to the DB (needed for manual result/injury entry). Full bidirectional sync and provenance change-logs are v2.

Mechanics: persistence + transport = one dedicated private git repo at `/Users/tutaitran/Documents/own projects/WC-2026-predictor/` holding the SQLite DB, the scrape cache, and the `WC vault` markdown. The Obsidian MCP is never used from CI (no Obsidian there); it is reserved for interactive use on the Mac, whose `WC vault` is a clone of the repo (Obsidian Git plugin auto-pulls). Clobber-proofing: CI replaces only content between `<!-- WC26:AUTO:section START -->` / `END` markers, never a byte outside; missing markers means skip and log; automation never deletes notes (mark `status: eliminated`). Schema mirrors the user's `_project-template` conventions; frontmatter carries `champion_prob` etc. for Dataview; generated notes use `[[wikilinks]]` so an injury connects to every match it affects. Per the user's preference the `WC vault` is a STANDALONE vault, separate from the existing vaults; the pipeline does not write into `The vault/`. Optionally a single one-line pointer can be added to the main `_index.md` for discoverability, but that is opt-in and off by default.

Streamlit (read-only, compute-free, local/private) is where you CONSUME forecasts, calibration, and the improvement trajectory; the vault is where you RECORD facts and judgment that feed the system.

### 3.7 Robustness and operations

All-Python pipeline (Node `node-cron` only as an optional local scheduler). Idempotency via natural keys + `INSERT ... ON CONFLICT DO UPDATE` in transactions. Append-only `model_runs` (run id, timestamp, git SHA, config, input snapshot hash, RNG seed, model version, metrics, data-completeness score) and append-only forecast snapshots, so calibration-over-time and a "why did this move" diff work; the vault `## Change log` mirrors this with a source tag per fact. Staleness guard + freshness score feeding the STALE banner. LLM parser guards: forced JSON schema, input-text hashing to dedupe and bound cost (hard per-run token and per-day call caps; on breach skip parsing, the model runs fine on Elo + results), official squad as an enum to reject hallucinated players, a required source-quote that must be a literal substring of the fetched text (cheap anti-hallucination check) plus URL and fetch timestamp, last-write-wins per (player, match) with recency decay, and a human-review queue for low-confidence/roster-mismatched events. Secrets: the LLM key and the free The Odds API key (both free, no paid feeds) in Actions secrets and a gitignored local `.env`. Timestamps stored UTC with venue timezone separate; cron in UTC; multi-timezone matchday tested. Schema migrations via `schema_version` + ordered files. Unit/property tests for the high-bug-risk pure functions (within-group mini-table ranking, Annex C routing, ET/penalties, conditional simulation) and invariants (per-group probabilities sum to 1, each team in exactly one R32 slot, a dominant team wins about 100 percent, calibration reliability slope about 1 within CI).

### 3.8 Sequencing

v1 (definition of done: one local command, `python -m wc26.forecast`, produces a credible, calibrated full forecast that runs on manually entered data).
1. Pinned Python 3.12/3.13 venv; verify `penaltyblog` + numpy/scipy install; stub the `python -m wc26.forecast` entry point.
2. One-time historical results + Elo seed + 2026 group draw into SQLite (no playoff-placeholder teams).
3. Elo-driven goal-expectation engine (W/D/L + scoreline matrix; one model) AND the shrunk player goal-share table that the availability channel and goalscorer model both consume.
4. Leak-free time-series backtest + post-hoc calibration + reliability diagram, reported with bootstrapped CIs, including a tournament-only held-out slice (tournament football is non-stationary vs friendlies/qualifiers). Meets the numeric acceptance bar below; the public-model comparison here is a one-time MANUAL paste of Opta's published numbers (the automated divergence tracker is v1.5).
5. Tournament simulator with the fully unit-tested rules engine; Monte Carlo (50k, logged seed, MC standard error); sanity-checked once against that manually recorded public-model snapshot.
6. Goalscorer model (penalty stream + the step-3 goal-share table), lowest-confidence; shippable last.
7. Manual result + injury entry path (HUMAN-override to DB) and the fast loop (Elo update on result + re-simulate + log running Brier/log loss).
8. Minimal read-only Streamlit: forecast view leading with W/D/L + top-3 scorelines, a calibration view that includes the running-improvement (Brier/log-loss) trajectory, and a mandatory STALE/freshness banner; plus one-directional vault generation + front-door note.

v1.5 (accelerators on top of the manual-entry core). The Odds API fetch + devig + model-market blend + logged model-vs-market divergence (the cleanest accelerator, a free API not a scrape, do this first); ESPN/Wikipedia live results scraper with two-source agreement; eloratings one-time seed scrape; Transfermarkt/Wikipedia injury scrape feeding the guarded LLM parser; at least one concrete "agent reads a news source and emits a structured fact (with source-quote + link)" deliverable so the news-reading loop the user asked for is visibly working; secondary public-model (Opta) benchmark scrape + logged divergence; data-completeness score surfaced. Manual entry remains the guaranteed fallback.

v2 (deeper). Slow-loop structural re-fit + significance-gated champion/challenger + model-version trajectory dashboard; ML ensemble research spike; full bidirectional two-tier vault + provenance change-logs.

### 3.9 Verification

- Numeric acceptance bar (v1 is "credible" only if it clears this): on leak-free time-series CV the engine beats two trivial baselines (always-favourite-by-Elo, uniform-by-rank) on log loss with bootstrapped CIs, and top-contender champion probabilities sit within a sane band of a public model (Opta).
- Calibration reported on a tournament-only slice separately from the global number; reliability slope about 1 within CI is an automated test, not just a chart.
- Rules engine: all unit/property tests pass before any forecast is published.
- Self-improvement: the fast loop logs a visible calibration trajectory; in v2, a challenger is promoted only when it clears the stated paired-significance margin; a deliberately corrupted result/scrape does NOT get promoted and is caught by two-source agreement.
- Degraded mode: with scrapers forced to fail, the pipeline falls back to last-good + manual entry, marks the forecast STALE, and still publishes (no hard stop on matchday).
- Vault: editing a HUMAN block updates the model via a `manual_override` and never touches human text; AUTO regeneration never clobbers annotations.
- Honesty UX: match views lead with probabilities, no single score/scorer shown as certainty, STALE banner and calibration reachable everywhere.

### 3.10 Reuse map (do not hand-roll)

- Match models + ratings: `penaltyblog` (PyPI; Poisson, Dixon-Coles, bivariate Poisson, hierarchical Bayes, Elo/Massey/Colley).
- Scraping: `soccerdata` (pinned) for FBref/ESPN/eloratings + thin custom scrapers for ESPN JSON / Wikipedia / Transfermarkt, run on the Mac.
- Anthropic SDK + SQLite (WAL/FTS5) pattern for the news parser: jarvis (`/Users/tutaitran/Documents/jarvis/`, see `memory.py`).
- Vault conventions: `/Users/tutaitran/The vault/Claude Setup/templates/_project-template.md`.
- Scraper prior art: `/Users/tutaitran/The vault/Own/Flight scraper/_project.md`.
- Optional local scheduler only: Agent Dashboard `backend/src/scheduler.ts`.

---

## 4. Research context

### 4.1 Forecasting methodology (how analysts actually do this)

- Team ratings: World Football Elo (eloratings.net) outperforms the FIFA ranking for prediction; both are Elo-derived. Opta Power Rankings and the retired FiveThirtyEight SPI are reference points.
- Match outcome: Poisson goal models, with the Dixon-Coles correction for low-score correlation, and bivariate Poisson. They produce a full scoreline probability matrix from which W/D/L and the modal scoreline are read. xG stabilises faster than goals but international xG is sparse. For sparse international data, deriving goal expectation from Elo difference (few global params) is more identifiable than fitting per-team attack/defence.
- Player scorers: P(anytime scorer) ~ team xG x player's share of goals x minutes x opponent defence, with a separate penalty stream. Bookmaker anytime-scorer markets (about 45 percent overround) are only a directional sanity check.
- Tournament: Monte Carlo over the 48-team format (12 groups of 4, top 2 + 8 best thirds into a Round of 32, then knockouts, 104 matches). Knockouts need extra time + shootouts. Third-place routing uses the official Annex C table; tie-breakers end in FIFA ranking.
- Evaluation: Brier score, log loss, ranked probability score; vig-removed market odds (~Brier 0.21 to 0.22) are the gold-standard benchmark and a strong blend signal; beating the market is very hard, so calibration is the honest goal.
- Useful references: dashee87 Dixon-Coles + time-weighting writeup; Opta supercomputer public predictions; Groll/Ley and related academic World Cup ML papers.

### 4.2 Data sources and tooling

- Ratings: eloratings.net (scrape or a Kaggle/GitHub dataset); FIFA ranking (official). ClubElo is club-only.
- Results/fixtures: ESPN public site JSON, Wikipedia tournament page, the `martj42/international_results` dataset for history.
- Stats/xG: FBref (StatsBomb) and Understat are largely club-only and require scraping; international xG is sparse.
- Squads/injuries: Transfermarkt (squad value + injuries, scrape at low cadence), Wikipedia, ESPN injury tracker, club/news pages.
- Odds: The Odds API free tier (500 credits/month, no card) for 1X2 + outright winner; devig before use.
- Python libraries: `penaltyblog` (match models + ratings; reuse, do not hand-roll the models), `soccerdata` (unified scraper for FBref/ESPN/Understat/ClubElo/eloratings), `socceraction` (event-level, not needed for prediction). Monte Carlo tournament simulation must be written by us.
- Legality: prefer keyless public data and free APIs; for scraping use no authentication, respect robots.txt and stated rate limits (FBref 10 req/min), attribute StatsBomb/FBref, store only short quotes from news.

### 4.3 Local conventions and reusable infrastructure

- Vault conventions: `_project.md` per project with `created/updated/status/type` frontmatter and `## Info / Paths / Tech Stack / Open Items / Notes / Session Logs`; template at `The vault/Claude Setup/templates/_project-template.md`. (The WC vault is standalone and mirrors these conventions, but is separate from `The vault/`.)
- jarvis (`/Users/tutaitran/Documents/jarvis/`): Anthropic SDK + SQLite (WAL/FTS5) + structured LLM output patterns to reuse for the news parser.
- Agent Dashboard (`/Users/tutaitran/Documents/own projects/Agent dashboard/`): a `node-cron` in-process scheduler; useful only as an optional LOCAL fallback, not for the cloud pipeline (GitHub Actions cron is the scheduler).
- Flight scraper (`The vault/Own/Flight scraper/_project.md`): existing scraper prior art worth mining.
- Architecture lesson: do NOT run one always-on agent per country. Use a small number of role-based scheduled jobs that iterate all teams; deterministic scrapers for structured data, an LLM only for unstructured news parsing.

---

## 5. Review history (nine passes + consolidation)

Round 1 (five lenses on early API-based drafts) caught: three tournament-rules bugs (within-group head-to-head ordering, missing extra-time/penalties, the best-8-thirds routing table), free-parameter Dixon-Coles overfitting sparse international data (switched primary engine to Elo-driven goal expectation), GitHub Actions ephemeral-storage breaking a local SQLite source of truth, free-tier rate limits breaking mid-tournament, and the betting layer's adverse-selection and Dutch legal exposure.

Round 2 (three lenses on the scraping-first direction) caught: the self-training loop was over-engineered for ~104 matches (split into a fast Elo loop and a slow significance-gated loop), scraping from GitHub Actions gets IP-blocked (move scrapers to the Mac, compute on Actions, manual-entry fallback), 433/open-web news is low-yield/brittle (favour structured sources), the need for a STALE/data-freshness banner and a public-model benchmark, the need to route form/availability into the interpretable core, and pinning Python 3.12/3.13.

Consolidation pass confirmed coherence and removed two hidden v1-to-v1.5 dependencies (public-model comparison made a manual paste in v1; the player goal-share table pulled forward to build with the engine), kept a visible "agent reads news" deliverable, and named the entry-point command.

Post-consolidation user corrections folded in: no paid APIs; odds re-added as a free prediction signal + benchmark via The Odds API free tier; standalone `WC vault`; project saved under `own projects/`.

---

## 6. How to continue

This document is the source of truth for the build. Suggested next command: kick off implementation of v1 in order (Section 3.8), starting with the pinned Python 3.12/3.13 environment and verifying `penaltyblog` installs, then the Elo engine + goal-share table, the backtest/calibration, and the rules-engine + Monte Carlo simulator. Manual data entry first; The Odds API blend and the scrapers are the v1.5 accelerators.
