#!/bin/bash
# Starts the WC2026 dashboard (foreground) and a periodic scrape/refresh loop.
# Invoked by the LaunchAgent on login (RunAtLoad + KeepAlive).
set -u
PROJ="/Users/tutaitran/Documents/own projects/WC-2026-predictor"
PY="$PROJ/.venv/bin/python"
ST="$PROJ/.venv/bin/streamlit"
cd "$PROJ" || exit 1
mkdir -p logs

# stop any prior refresh loop from a previous launch
[ -f logs/refresh.pid ] && kill "$(cat logs/refresh.pid)" 2>/dev/null || true
pkill -f "wc26.update" 2>/dev/null || true

# periodic refresh: scrape live data, re-forecast, fetch odds — every 3 hours.
# First run is delayed so a fresh login does not immediately churn the repo.
(
  sleep 1800
  while true; do
    "$PY" -m wc26.update >> logs/update.log 2>&1 || true
    "$PY" -m wc26.odds   >> logs/odds.log   2>&1 || true
    sleep 10800
  done
) &
echo $! > logs/refresh.pid

# dashboard in the foreground (KeepAlive restarts it if it ever exits)
exec "$ST" run wc26/app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
