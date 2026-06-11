"""Tests for the in-play market snapshotter (paper/research, flag-gated)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wc26 import db, edge, marketwatch, schedule

UTC = timezone.utc


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def _events():
    return [{
        "bookmakers": [
            {"markets": [{"key": "outrights", "outcomes": [
                {"name": "Spain", "price": 5.0}, {"name": "Brazil", "price": 7.0}]}]},
            {"markets": [{"key": "outrights", "outcomes": [
                {"name": "Spain", "price": 6.0}, {"name": "Brazil", "price": 9.0}]}]},
        ]
    }]


KO = datetime(2026, 6, 11, 19, 0, tzinfo=UTC)
LIVE = KO + timedelta(minutes=30)       # match in play
OFFLINE = KO - timedelta(hours=3)        # no match in play


# --- enable flag ------------------------------------------------------------

def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("WC26_INPLAY_ODDS", raising=False)
    assert marketwatch.enabled() is False
    monkeypatch.setenv("WC26_INPLAY_ODDS", "1")
    assert marketwatch.enabled() is True
    monkeypatch.setenv("WC26_INPLAY_ODDS", "no")
    assert marketwatch.enabled() is False


# --- run_snapshot gating ----------------------------------------------------

def test_run_snapshot_disabled_is_noop(conn, monkeypatch):
    monkeypatch.delenv("WC26_INPLAY_ODDS", raising=False)
    res = marketwatch.run_snapshot(conn, LIVE, [KO], _events)
    assert res["snapshotted"] is False and res["reason"] == "disabled"


def test_run_snapshot_not_live(conn):
    res = marketwatch.run_snapshot(conn, OFFLINE, [KO], _events, force=True)
    assert res["snapshotted"] is False and res["reason"] == "no match in play"


def test_run_snapshot_stores_when_live(conn):
    res = marketwatch.run_snapshot(conn, LIVE, [KO], _events, force=True)
    assert res["snapshotted"] is True and res["selections"] == 2
    assert marketwatch.snapshots_today(conn, "champion", LIVE.date().isoformat()) == 1
    snap = marketwatch.latest_snapshot(conn, "champion")
    assert snap["Spain"].best_odds == 6.0 and snap["Spain"].n_books == 2


def test_run_snapshot_respects_daily_budget(conn):
    for _ in range(3):
        marketwatch.run_snapshot(conn, LIVE, [KO], _events, force=True, max_per_day=3)
    res = marketwatch.run_snapshot(conn, LIVE, [KO], _events, force=True, max_per_day=3)
    assert res["snapshotted"] is False and res["reason"] == "daily budget reached"
    assert marketwatch.snapshots_today(conn, "champion", LIVE.date().isoformat()) == 3


def test_run_snapshot_handles_fetch_error(conn):
    def boom():
        raise RuntimeError("network down")
    res = marketwatch.run_snapshot(conn, LIVE, [KO], boom, force=True)
    assert res["snapshotted"] is False and "fetch error" in res["reason"]


def test_latest_snapshot_none_when_empty(conn):
    assert marketwatch.latest_snapshot(conn, "champion") is None


# --- CLV vs the latest (closing-proxy) snapshot -----------------------------

def test_clv_table_vs_latest_snapshot(conn):
    marketwatch.run_snapshot(conn, LIVE, [KO], _events, force=True)
    closing = edge.market_fair_probs(marketwatch.latest_snapshot(conn, "champion"), "shin")
    # We "entered" Spain at a fair prob 5pp below where it now sits -> positive CLV.
    entries = {"Spain": closing["Spain"] - 0.05}
    rows = marketwatch.clv_table(conn, "champion", entries, method="shin")
    spain = next(r for r in rows if r["selection"] == "Spain")
    assert spain["clv"] == pytest.approx(0.05, abs=1e-4)
    assert spain["closing_fair"] == pytest.approx(round(closing["Spain"], 4))


def test_clv_table_empty_when_no_snapshot(conn):
    assert marketwatch.clv_table(conn, "champion", {"Spain": 0.2}) == []
