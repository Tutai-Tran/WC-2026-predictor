"""Tests for the hosted DB-snapshot install path (wc26/db.py).

These guard the live dashboard against the two ways the remote DB sync used to take
the page down with a redacted `sqlite3.DatabaseError`:
  1. a partial / non-database download being swapped straight in, and
  2. stale `-wal`/`-shm` sidecars left over a freshly-swapped main file
     ("database disk image is malformed").
"""

import sqlite3
from pathlib import Path

from wc26 import db


def _db_bytes(tmp_path, name, with_forecast=True) -> bytes:
    """Build a self-contained (non-WAL) SQLite snapshot and return its raw bytes."""
    p = tmp_path / name
    conn = sqlite3.connect(p)                     # default rollback journal -> all in main file
    db.init_db(conn)
    if with_forecast:
        conn.execute(
            "INSERT INTO predictions (run_id, scope, ref, payload_json, computed_at) "
            "VALUES (?,?,?,?,?)", ("r1", "forecast", "x", "{}", "2026-06-05T00:00:00+00:00"))
    conn.commit()
    conn.close()
    return p.read_bytes()


def test_is_valid_snapshot_accepts_real_forecast_db(tmp_path):
    good = tmp_path / "g.db"
    good.write_bytes(_db_bytes(tmp_path, "g0.db", with_forecast=True))
    assert db.is_valid_snapshot(good) is True


def test_is_valid_snapshot_rejects_non_database(tmp_path):
    html = tmp_path / "page.html"
    html.write_bytes(b"<!DOCTYPE html><title>404 Not Found</title>")
    assert db.is_valid_snapshot(html) is False
    assert db.is_valid_snapshot(tmp_path / "does-not-exist.db") is False


def test_is_valid_snapshot_rejects_truncated_download(tmp_path):
    full = _db_bytes(tmp_path, "full.db", with_forecast=True)
    partial = tmp_path / "partial.db"
    partial.write_bytes(full[:200])               # right magic header, truncated body
    assert db.is_valid_snapshot(partial) is False


def test_is_valid_snapshot_rejects_db_without_forecast(tmp_path):
    empty = tmp_path / "empty.db"
    empty.write_bytes(_db_bytes(tmp_path, "e0.db", with_forecast=False))
    assert db.is_valid_snapshot(empty) is False   # valid SQLite, but nothing to show


def test_install_snapshot_clears_stale_wal_sidecars(tmp_path):
    dest = tmp_path / "wc26.db"
    dest.write_bytes(_db_bytes(tmp_path, "old.db"))
    # sidecars left behind by a previous WAL connection -> would corrupt the new file
    Path(str(dest) + "-wal").write_bytes(b"stale-wal")
    Path(str(dest) + "-shm").write_bytes(b"stale-shm")

    assert db.install_snapshot(_db_bytes(tmp_path, "new.db"), dest) is True
    assert not Path(str(dest) + "-wal").exists()
    assert not Path(str(dest) + "-shm").exists()

    # the live read path must now open cleanly (no "disk image is malformed")
    conn = db.connect(dest)
    try:
        n = conn.execute("SELECT COUNT(*) FROM predictions WHERE scope='forecast'").fetchone()[0]
    finally:
        conn.close()
    assert n >= 1


def test_is_valid_snapshot_handles_path_with_spaces(tmp_path):
    spaced = tmp_path / "my data dir"
    spaced.mkdir()
    p = spaced / "wc 26.db"
    p.write_bytes(_db_bytes(tmp_path, "src.db", with_forecast=True))
    assert db.is_valid_snapshot(p) is True            # file: URI must be percent-encoded


def test_corrupt_db_read_raises_catchable_sqlite_error(tmp_path):
    """Reproduce the exact live failure: a corrupt wc26.db makes the connect+query
    path raise a sqlite3.Error (DatabaseError/OperationalError both subclass it), which
    is what load_latest now catches to degrade gracefully instead of crashing the page.
    Also confirm such a file is rejected by validation so it's never installed."""
    bad = tmp_path / "wc26.db"
    bad.write_bytes(b"definitely not a sqlite database " * 300)   # right size, wrong bytes

    raised = False
    try:
        conn = db.connect(bad)                # mirrors load_latest's connect+query flow
        try:
            conn.execute(
                "SELECT payload_json FROM predictions WHERE scope='forecast' "
                "ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:                     # the base class load_latest catches
        raised = True
    assert raised, "a corrupt DB must raise a catchable sqlite3.Error"
    assert db.is_valid_snapshot(bad) is False  # ...and validation would never install it


def test_install_snapshot_rejects_bad_download_keeps_existing(tmp_path):
    dest = tmp_path / "wc26.db"
    good = _db_bytes(tmp_path, "good.db")
    dest.write_bytes(good)

    assert db.install_snapshot(b"<html>502 Bad Gateway</html>", dest) is False
    assert dest.read_bytes() == good                       # existing DB untouched
    assert not Path(str(dest) + ".download").exists()      # temp file cleaned up
