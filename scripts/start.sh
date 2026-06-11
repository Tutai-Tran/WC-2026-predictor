#!/bin/bash
# Starts the WC2026 dashboard (foreground) and a periodic scrape/refresh loop.
# Invoked by the LaunchAgent on login (RunAtLoad + KeepAlive).
set -u
PROJ="/Users/tutaitran/wc26"
PY="$PROJ/.venv/bin/python"
ST="$PROJ/.venv/bin/streamlit"
# launchd gives us a minimal PATH; add Homebrew so child tools (gh, git) resolve.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"
cd "$PROJ" || exit 1
mkdir -p logs

# stop any prior refresh loop AND any orphaned dashboard from a previous launch
# (a leftover streamlit would hold :8501, crash the new one, and launchd would
# relaunch in a tight loop, each relaunch burning a full refresh cycle)
[ -f logs/refresh.pid ] && kill "$(cat logs/refresh.pid)" 2>/dev/null || true
pkill -f "wc26.update" 2>/dev/null || true
pkill -f "streamlit run wc26/app.py" 2>/dev/null || true

# One full self-improving cycle: scrape + LLM news + odds + re-forecast, publish
# the DB snapshot for the cloud dashboard, then push the repo updates (vault,
# ratings, memory).
refresh_cycle() {
  "$PY" -m wc26.update >> logs/update.log 2>&1 || true
  "$PY" -m wc26.odds   >> logs/odds.log   2>&1 || true
  bash "$PROJ/scripts/publish_db.sh" >> logs/publish.log 2>&1 || true
  git -C "$PROJ" add -A >/dev/null 2>&1 || true
  git -C "$PROJ" commit -q -m "chore: scheduled self-improving refresh [skip ci]" >> logs/git.log 2>&1 || true
  # the remote can gain commits; rebase before pushing so a divergence never
  # silently breaks every future push, and LOG failures instead of hiding them
  git -C "$PROJ" fetch -q origin >> logs/git.log 2>&1 || true
  if ! git -C "$PROJ" rebase -q origin/main >> logs/git.log 2>&1; then
    git -C "$PROJ" rebase --abort >> logs/git.log 2>&1 || true
    echo "$(date -u +%FT%TZ) rebase FAILED — local and remote diverged" >> logs/git.log
  fi
  if ! git -C "$PROJ" push -q origin main >> logs/git.log 2>&1; then
    echo "$(date -u +%FT%TZ) push FAILED" >> logs/git.log
  fi
}

# Refresh + publish IMMEDIATELY on launch, then on an EVENT-DRIVEN cadence: sleep
# until the next pre-kickoff (lineups) / post-match (result) trigger computed from
# the fixture list, with a slow heartbeat when nothing is near. Recomputed each
# cycle. Far more timely around matches than a flat timer, and it avoids pointless
# mid-match re-runs (the model is pre-match; the scrape is completed-results-only;
# the input-fingerprint skip makes a mid-match run a no-op anyway).
( refresh_cycle
  while true; do
    SLEEP_SEC="$("$PY" -m wc26.schedule sleep 2>>logs/schedule.log)"
    case "$SLEEP_SEC" in (*[!0-9]*|"") SLEEP_SEC=10800 ;; esac   # fallback if CLI failed
    sleep "$SLEEP_SEC"
    refresh_cycle
  done ) &
LOOP_PID=$!
echo "$LOOP_PID" > logs/refresh.pid

# OPTIONAL in-play odds/CLV research snapshots (opt-in: WC26_INPLAY_ODDS=1). This
# is the ONLY thing that wakes during the 90 minutes; each tick is a no-op unless
# a match is live and the daily budget allows. Spends Odds API quota.
SNAP_PID=""
if [ "${WC26_INPLAY_ODDS:-}" = "1" ]; then
  ( while true; do
      "$PY" -m wc26.marketwatch >> logs/marketwatch.log 2>&1 || true
      sleep "${WC26_INPLAY_INTERVAL:-900}"
    done ) &
  SNAP_PID=$!
fi

# dashboard alongside the loop
"$ST" run wc26/app.py --server.headless true --server.port 8501 --server.address 127.0.0.1 &
DASH_PID=$!

# Supervise BOTH: if either the refresh loop or the dashboard dies, exit non-zero so
# launchd KeepAlive relaunches the whole thing. This keeps the device-only system
# self-healing with no Claude session involved — only the device needs to be on.
while kill -0 "$LOOP_PID" 2>/dev/null && kill -0 "$DASH_PID" 2>/dev/null; do
  sleep 30
done
kill "$LOOP_PID" "$DASH_PID" ${SNAP_PID:+"$SNAP_PID"} 2>/dev/null || true
exit 1
