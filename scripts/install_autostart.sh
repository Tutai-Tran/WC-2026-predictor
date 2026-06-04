#!/bin/bash
# Installs the LaunchAgent so the dashboard + scraper start on every login and
# stay running until you run scripts/stop.sh. Safe to re-run.
set -e
PROJ="/Users/tutaitran/Documents/own projects/WC-2026-predictor"
DEST="$HOME/Library/LaunchAgents/com.tutai.wc26.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PROJ/scripts/com.tutai.wc26.plist" "$DEST"
launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"
echo "Installed and loaded com.tutai.wc26 (dashboard on http://localhost:8501)."
echo "Stop with: bash scripts/stop.sh"
