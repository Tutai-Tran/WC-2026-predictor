"""Tests for the event-driven refresh scheduler (pure, deterministic in `now`)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from wc26 import schedule

UTC = timezone.utc


def dt(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


# --- parsing ----------------------------------------------------------------

def test_parse_kickoff_iso_z():
    assert schedule.parse_kickoff("2026-06-11T19:00:00Z") == dt(2026, 6, 11, 19, 0)


def test_parse_kickoff_rejects_dateonly_and_garbage():
    assert schedule.parse_kickoff("2026-05-28") is None       # no time component
    assert schedule.parse_kickoff("") is None
    assert schedule.parse_kickoff(None) is None
    assert schedule.parse_kickoff("not-a-date") is None


def test_kickoffs_from_rows_sorts_and_drops_dateonly():
    rows = [
        {"date_utc": "2026-06-11T19:00:00Z"},
        {"date_utc": "2026-05-28"},               # dropped (date only)
        {"date_utc": "2026-06-11T16:00:00Z"},
        {"date_utc": None},                       # dropped
    ]
    ks = schedule.kickoffs_from_rows(rows)
    assert ks == [dt(2026, 6, 11, 16, 0), dt(2026, 6, 11, 19, 0)]


# --- trigger times ----------------------------------------------------------

def test_trigger_times_pre_and_post():
    ko = dt(2026, 6, 11, 19, 0)
    trig = schedule.trigger_times([ko], pre=75, post=110)
    assert (ko - timedelta(minutes=75), "pre-kickoff") in trig
    assert (ko + timedelta(minutes=110), "post-match") in trig
    # sorted ascending by time
    assert [t for t, _ in trig] == sorted(t for t, _ in trig)


# --- next trigger / sleep ---------------------------------------------------

def test_next_trigger_pre_kickoff():
    ko = dt(2026, 6, 11, 19, 0)
    now = dt(2026, 6, 11, 17, 0)            # 2h before KO; pre-kickoff at 17:45
    t, why = schedule.next_trigger(now, [ko], heartbeat_hours=4)
    assert why == "pre-kickoff" and t == dt(2026, 6, 11, 17, 45)


def test_next_trigger_post_match_after_lineups_passed():
    ko = dt(2026, 6, 11, 19, 0)
    now = dt(2026, 6, 11, 18, 0)            # past the 17:45 pre trigger
    t, why = schedule.next_trigger(now, [ko], heartbeat_hours=6)
    assert why == "post-match" and t == dt(2026, 6, 11, 20, 50)


def test_next_trigger_heartbeat_when_nothing_close():
    ko = dt(2026, 6, 20, 19, 0)            # far away
    now = dt(2026, 6, 11, 12, 0)
    t, why = schedule.next_trigger(now, [ko], heartbeat_hours=4)
    assert why == "heartbeat" and t == now + timedelta(hours=4)


def test_next_trigger_no_fixtures_is_heartbeat():
    now = dt(2026, 6, 11, 12, 0)
    t, why = schedule.next_trigger(now, [], heartbeat_hours=4)
    assert why == "heartbeat" and t == now + timedelta(hours=4)


def test_seconds_until_next_floor_and_value():
    ko = dt(2026, 6, 11, 19, 0)
    now = dt(2026, 6, 11, 17, 0)
    secs, why = schedule.seconds_until_next(now, [ko], min_sleep=300)
    assert why == "pre-kickoff" and secs == 45 * 60
    # right on top of a trigger -> floored to min_sleep, not 0/negative
    now2 = dt(2026, 6, 11, 17, 45)
    secs2, _ = schedule.seconds_until_next(now2, [ko], min_sleep=300, heartbeat_hours=6)
    assert secs2 >= 300


# --- is_due -----------------------------------------------------------------

def test_is_due_initial_when_no_last_run():
    due, why = schedule.is_due(dt(2026, 6, 11, 12, 0), [], None)
    assert due and why == "initial"


def test_is_due_crosses_trigger():
    ko = dt(2026, 6, 11, 19, 0)            # post-match trigger at 20:50
    last = dt(2026, 6, 11, 20, 40)
    now = dt(2026, 6, 11, 20, 55)          # crossed 20:50 since last run
    due, why = schedule.is_due(now, [ko], last)
    assert due and why == "post-match"


def test_is_due_heartbeat_elapsed():
    last = dt(2026, 6, 11, 8, 0)
    now = dt(2026, 6, 11, 12, 30)          # 4.5h later, no triggers between
    due, why = schedule.is_due(now, [], last, heartbeat_hours=4)
    assert due and why == "heartbeat"


def test_is_due_wait():
    last = dt(2026, 6, 11, 11, 0)
    now = dt(2026, 6, 11, 12, 0)            # 1h later, nothing crossed
    due, why = schedule.is_due(now, [], last, heartbeat_hours=4)
    assert not due and why == "wait"


# --- live detection ---------------------------------------------------------

def test_is_live_during_match():
    ko = dt(2026, 6, 11, 19, 0)
    assert schedule.is_live(dt(2026, 6, 11, 19, 30), [ko])      # 30' in
    assert schedule.is_live(dt(2026, 6, 11, 20, 50), [ko])      # within 120' window
    assert not schedule.is_live(dt(2026, 6, 11, 18, 59), [ko])  # before KO
    assert not schedule.is_live(dt(2026, 6, 11, 21, 30), [ko])  # after window


def test_live_kickoffs_lists_in_play():
    kos = [dt(2026, 6, 11, 19, 0), dt(2026, 6, 11, 16, 0)]
    live = schedule.live_kickoffs(dt(2026, 6, 11, 19, 30), kos)
    assert live == [dt(2026, 6, 11, 19, 0)]
