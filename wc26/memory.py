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


LESSONS = MEMORY_DIR / "LESSONS.md"


def _pct(x) -> str:
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "—"


def write_lessons(summary: dict) -> None:
    """Regenerate the curated 'what the model has learned' note from the aggregate.

    Single, fully-regenerated file (a projection of the current aggregate), so it
    never accumulates stale duplicates. Hypotheses/audit trail — the parameters are
    where the model actually changes."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    o = summary.get("overall", {}) or {}
    out = [
        "---", "type: wc-lessons", f"updated: {str(summary.get('computed_at', ''))[:10]}", "---", "",
        "# What the model has learned", "",
        f"> Leak-free over **{o.get('n', 0)}** graded matches: outcome accuracy "
        f"**{_pct(o.get('accuracy'))}**, Brier {o.get('brier')}, log loss {o.get('log_loss')}. "
        "The biases below are hypotheses (the audit trail); the model's parameters are "
        "where learning is actually applied, only after it validates out-of-sample.", "",
    ]
    segs = summary.get("segments", {}) or {}
    if segs:
        out += ["## Accuracy by segment"]
        out += [f"- {name}: {_pct(s['accuracy'])} ({s['n']} matches)" for name, s in segs.items()]
        out += [""]
    biases = summary.get("biases", []) or []
    if biases:
        out += ["## Systematic biases found (ranked by evidence)"]
        out += [f"- **{b['factor']}** ({b['segment']}): model {b['direction']}-rated it "
                f"(strength {b['strength']}, over {b['n']} wrong matches)" for b in biases[:8]]
    else:
        out += ["_No systematic bias detected yet — needs more played matches._"]
    LESSONS.write_text("\n".join(out) + "\n")


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "note"
    body = sys.argv[2] if len(sys.argv) > 2 else ""
    log_task(title, body)
    print(f"logged: {title}")


if __name__ == "__main__":
    main()
