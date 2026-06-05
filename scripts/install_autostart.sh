#!/bin/bash
# Installs the LaunchAgent so the dashboard + scraper start on every login and
# stay running until you run scripts/stop.sh. Safe to re-run.
#
# IMPORTANT (macOS): this project lives under ~/Documents, which macOS protects
# from background (launchd) processes. For the LaunchAgent to read the project you
# must grant Full Disk Access ONCE:
#   System Settings > Privacy & Security > Full Disk Access > add /bin/bash (and bash via the + > Cmd+Shift+G > /bin/bash)
# Without it the agent fails with "Operation not permitted". For the current
# session you can instead just run:  nohup bash scripts/start.sh >logs/run.log 2>&1 &
set -e
PROJ="/Users/tutaitran/Documents/own projects/WC-2026-predictor"
DEST="$HOME/Library/LaunchAgents/com.tutai.wc26.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PROJ/scripts/com.tutai.wc26.plist" "$DEST"
launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"
sleep 5
if curl -s -o /dev/null http://localhost:8501; then
  echo "Installed and running: com.tutai.wc26 (dashboard on http://localhost:8501)."
else
  echo "Installed, but the dashboard is not responding. This is almost certainly the"
  echo "macOS Full Disk Access restriction (project is under ~/Documents)."
  echo "Grant Full Disk Access to /bin/bash (see the header of this script), then re-run."
  echo "Meanwhile, run it for this session:  nohup bash scripts/start.sh >logs/run.log 2>&1 &"
fi
echo "Stop with: bash scripts/stop.sh"
