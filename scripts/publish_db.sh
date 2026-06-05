#!/bin/bash
# Publish the latest wc26.db to a PUBLIC data repo's "live" release, so the
# cloud-hosted (read-only) dashboard can pull it without any credentials.
#
# Why a snapshot: the live wc26.db is WAL-mode, so recent commits may sit in the
# -wal sidecar. We use SQLite's backup API to write a consistent single-file copy
# (WAL merged) and upload THAT, named wc26.db so the asset URL stays stable:
#   https://github.com/<DATA_REPO>/releases/download/live/wc26.db
# Safe to re-run; creates the release on first use.
set -u
PROJ="/Users/tutaitran/wc26"
PY="$PROJ/.venv/bin/python"
DATA_REPO="Tutai-Tran/WC-2026-data"

[ -f "$PROJ/wc26.db" ] || { echo "no wc26.db yet; nothing to publish"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# consistent snapshot (merges WAL) named wc26.db
"$PY" - "$PROJ/wc26.db" "$TMP/wc26.db" <<'PYEOF'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)
dst.close()
src.close()
PYEOF

if gh release upload live "$TMP/wc26.db" --clobber -R "$DATA_REPO" 2>/dev/null; then
    echo "published wc26.db to $DATA_REPO release 'live'"
else
    gh release create live "$TMP/wc26.db" -R "$DATA_REPO" \
        -t "live data" -n "Rolling latest wc26.db, auto-published by the local refresh loop." \
        && echo "created release 'live' on $DATA_REPO and uploaded wc26.db"
fi
