"""Event-driven refresh scheduling for the tournament.

The forecast model is PRE-MATCH: nothing it consumes changes during the 90
minutes (the scrape is completed-results-only and the input-fingerprint skip
makes a mid-match re-run a no-op). The moments that actually change its inputs
are:

  * ~75 min before kickoff  -> confirmed lineups feed the availability channel;
  * ~110 min after kickoff  -> the result is final, so it folds into Elo, the
    forecast re-runs, and the prediction is graded.

So we trigger refreshes around those events and otherwise idle on a slow
heartbeat, instead of a flat fast timer or polling during play. All scheduling
logic here is pure and deterministic in `now`, so it is fully testable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

PRE_KICKOFF_MIN = 75       # run this many minutes before kickoff (lineups)
POST_MATCH_MIN = 110       # run this many minutes after kickoff (result final)
HEARTBEAT_HOURS = 4        # never idle longer than this without a run
MIN_SLEEP_SEC = 300        # floor on the computed sleep, so we never busy-loop
MATCH_DURATION_MIN = 120   # KO .. KO+this is treated as "in play" (covers ET)


def parse_kickoff(value: str | None) -> datetime | None:
    """Parse an ISO `date_utc` into an aware UTC datetime, or None.

    Rejects date-only values (no time component) since they cannot anchor a
    precise kickoff/full-time trigger."""
    if not value or "T" not in value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def kickoffs_from_rows(rows: list) -> list[datetime]:
    """Sorted kickoff datetimes from match rows that have a parseable time."""
    out = [parse_kickoff(r["date_utc"]) for r in rows]
    return sorted(k for k in out if k is not None)


def trigger_times(
    kickoffs: list[datetime], *, pre: int = PRE_KICKOFF_MIN, post: int = POST_MATCH_MIN
) -> list[tuple[datetime, str]]:
    """The (time, reason) refresh triggers implied by a list of kickoffs:
    one `pre-kickoff` trigger before each, one `post-match` trigger after."""
    trig: list[tuple[datetime, str]] = []
    for ko in kickoffs:
        trig.append((ko - timedelta(minutes=pre), "pre-kickoff"))
        trig.append((ko + timedelta(minutes=post), "post-match"))
    trig.sort(key=lambda t: t[0])
    return trig


def next_trigger(
    now: datetime,
    kickoffs: list[datetime],
    *,
    pre: int = PRE_KICKOFF_MIN,
    post: int = POST_MATCH_MIN,
    heartbeat_hours: float = HEARTBEAT_HOURS,
) -> tuple[datetime, str]:
    """The earliest trigger strictly after `now` that falls within the heartbeat
    window; otherwise a `heartbeat` run at now + heartbeat_hours."""
    hb = now + timedelta(hours=heartbeat_hours)
    for t, why in trigger_times(kickoffs, pre=pre, post=post):
        if now < t <= hb:
            return t, why
    return hb, "heartbeat"


def seconds_until_next(
    now: datetime,
    kickoffs: list[datetime],
    *,
    pre: int = PRE_KICKOFF_MIN,
    post: int = POST_MATCH_MIN,
    heartbeat_hours: float = HEARTBEAT_HOURS,
    min_sleep: int = MIN_SLEEP_SEC,
) -> tuple[int, str]:
    """Seconds to sleep until the next trigger (floored at `min_sleep`)."""
    t, why = next_trigger(now, kickoffs, pre=pre, post=post, heartbeat_hours=heartbeat_hours)
    secs = int((t - now).total_seconds())
    return max(min_sleep, secs), why


def is_due(
    now: datetime,
    kickoffs: list[datetime],
    last_run: datetime | None,
    *,
    pre: int = PRE_KICKOFF_MIN,
    post: int = POST_MATCH_MIN,
    heartbeat_hours: float = HEARTBEAT_HOURS,
) -> tuple[bool, str]:
    """Whether a refresh is due now: True if a trigger time was crossed since
    `last_run`, or the heartbeat interval has elapsed. None last_run => initial."""
    if last_run is None:
        return True, "initial"
    for t, why in trigger_times(kickoffs, pre=pre, post=post):
        if last_run < t <= now:
            return True, why
    if now - last_run >= timedelta(hours=heartbeat_hours):
        return True, "heartbeat"
    return False, "wait"


def live_kickoffs(
    now: datetime, kickoffs: list[datetime], *, duration_min: int = MATCH_DURATION_MIN
) -> list[datetime]:
    """Kickoffs whose match is currently in play (KO <= now <= KO+duration)."""
    return [ko for ko in kickoffs if ko <= now <= ko + timedelta(minutes=duration_min)]


def is_live(now: datetime, kickoffs: list[datetime], *, duration_min: int = MATCH_DURATION_MIN) -> bool:
    """True if at least one match is in play right now."""
    return bool(live_kickoffs(now, kickoffs, duration_min=duration_min))


def upcoming_kickoffs(conn: sqlite3.Connection, now: datetime | None = None) -> list[datetime]:
    """Kickoffs of unplayed fixtures, keeping recently-finished ones so their
    `post-match` trigger can still fire after a restart."""
    rows = conn.execute(
        "SELECT date_utc FROM matches WHERE played=0 AND date_utc IS NOT NULL"
    ).fetchall()
    ks = kickoffs_from_rows(rows)
    if now is not None:
        floor = now - timedelta(hours=MATCH_DURATION_MIN / 60 + 1)
        ks = [k for k in ks if k >= floor]
    return ks


def main() -> None:
    """CLI used by the refresh loop. Subcommands:
    `sleep` -> seconds to next trigger; `next` -> time + reason; `live` -> 1/0."""
    import sys

    from . import db

    cmd = sys.argv[1] if len(sys.argv) > 1 else "next"
    now = datetime.now(timezone.utc)
    conn = db.connect()
    kickoffs = upcoming_kickoffs(conn, now)
    if cmd == "sleep":
        secs, why = seconds_until_next(now, kickoffs)
        print(secs)
    elif cmd == "live":
        print(1 if is_live(now, kickoffs) else 0)
    else:
        t, why = next_trigger(now, kickoffs)
        print(f"{t.isoformat()}  {why}  (in {int((t - now).total_seconds())}s)")


if __name__ == "__main__":
    main()
