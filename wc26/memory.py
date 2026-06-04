"""Persistent project memory stored in the WC vault.

A running, human-readable log of context, decisions, and what was done each task,
so the project's history lives alongside the predictions. Append-only.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from . import config

MEMORY_DIR = config.VAULT / "_memory"
CONTEXT = MEMORY_DIR / "CONTEXT.md"

_HEADER = """---
type: wc-memory
---

# WC-2026 Project Memory

Running log of context, decisions, and what was done each task. Newest entries are
appended at the bottom. This file is the project's source of truth for "what did we
do and why".
"""


def log_task(title: str, body: str = "") -> None:
    """Append a timestamped entry to the vault memory."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not CONTEXT.exists():
        CONTEXT.write_text(_HEADER)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with CONTEXT.open("a") as f:
        f.write(f"\n## {ts} — {title}\n\n{body.strip()}\n")


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "note"
    body = sys.argv[2] if len(sys.argv) > 2 else ""
    log_task(title, body)
    print(f"logged: {title}")


if __name__ == "__main__":
    main()
