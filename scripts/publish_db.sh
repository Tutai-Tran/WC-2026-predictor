#!/bin/bash
# Publish the latest wc26.db to a PUBLIC data repo's "live" release, so the
# cloud-hosted (read-only) dashboard can pull it without any credentials.
#
# Why a snapshot: the live wc26.db is WAL-mode, so recent commits may sit in the
# -wal sidecar. We use SQLite's backup API to write a consistent single-file copy
# (WAL merged) and upload THAT, named wc26.db so the asset URL stays stable:
#   https://github.com/<DATA_REPO>/releases/download/live/wc26.db
# Safe to re-run; creates the release on first use. Fails soft from start.sh
# (called with `|| true`), but logs real errors to logs/publish.log.
set -eu
PROJ="/Users/tutaitran/wc26"
PY="$PROJ/.venv/bin/python"
DATA_REPO="Tutai-Tran/WC-2026-data"

[ -f "$PROJ/wc26.db" ] || { echo "no wc26.db yet; nothing to publish"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# consistent snapshot (merges WAL), named wc26.db for a stable asset URL
"$PY" - "$PROJ/wc26.db" "$TMP/wc26.db" <<'PYEOF'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1])
dst = sqlite3.connect(sys.argv[2])
try:
    # refuse to publish a DB with no forecast (e.g. a half-built/empty file)
    n = src.execute("SELECT COUNT(*) FROM predictions WHERE scope='forecast'").fetchone()[0]
    if n == 0:
        raise SystemExit("source DB has no forecast snapshot; refusing to publish")
    src.backup(dst)
finally:
    dst.close()
    src.close()
PYEOF

# never upload an empty/broken snapshot
[ -s "$TMP/wc26.db" ] || { echo "snapshot is empty; aborting publish" >&2; exit 1; }

if gh release upload live "$TMP/wc26.db" --clobber -R "$DATA_REPO"; then
    echo "published wc26.db to $DATA_REPO release 'live'"
elif gh release view live -R "$DATA_REPO" >/dev/null 2>&1; then
    # release exists but the upload failed -> real error (auth/network), surface it
    echo "ERROR: 'live' release exists but upload failed (auth/network?); not updated" >&2
    exit 1
else
    # first run: the release does not exist yet -> create it with the asset
    gh release create live "$TMP/wc26.db" -R "$DATA_REPO" \
        -t "live data" -n "Rolling latest wc26.db, auto-published by the local refresh loop."
    echo "created release 'live' on $DATA_REPO and uploaded wc26.db"
fi
