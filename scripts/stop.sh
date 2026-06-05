#!/bin/bash
# Stops the WC2026 dashboard + refresh loop and disables auto-start.
PROJ="/Users/tutaitran/wc26"
launchctl unload "$HOME/Library/LaunchAgents/com.tutai.wc26.plist" 2>/dev/null || true
[ -f "$PROJ/logs/refresh.pid" ] && kill "$(cat "$PROJ/logs/refresh.pid")" 2>/dev/null || true
pkill -f "streamlit run.*wc26/app.py" 2>/dev/null || true
pkill -f "wc26.update" 2>/dev/null || true
echo "WC2026 dashboard + refresh stopped, and auto-start disabled."
echo "To re-enable auto-start: bash scripts/install_autostart.sh"
