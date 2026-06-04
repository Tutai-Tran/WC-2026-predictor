"""Generate the WC vault from a forecast result.

Only machine-managed AUTO blocks (between WC26:AUTO markers) are ever rewritten;
HUMAN blocks are never touched. Missing markers are skipped, never re-injected.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from . import config

VAULT = config.VAULT
COUNTRIES = VAULT / "Countries"
GROUPS = VAULT / "Groups"
MATCHES = VAULT / "Matches"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _safe(name: str) -> str:
    return name.replace("/", "-")


# DB team name -> on-disk note filename stem (scrapers used the cedilla form)
NOTE_FILENAME_ALIASES = {"Curacao": "Curaçao"}


def _link(name: str) -> str:
    """Wikilink target stem, honouring the on-disk filename alias."""
    return NOTE_FILENAME_ALIASES.get(name, _safe(name))


def replace_auto_block(text: str, name: str, content: str, source: str = "model") -> str:
    """Replace the body between WC26:AUTO:<name> START/END markers. No-op if absent."""
    # Tempered body so a START never spans across another same-name START
    # (protects HUMAN content if a note is corrupted with a missing END marker).
    pattern = re.compile(
        r"(<!-- WC26:AUTO:" + re.escape(name) + r" START)[^\n]*?(-->)"
        r"((?:(?!<!-- WC26:AUTO:" + re.escape(name) + r" START).)*?)"
        r"(<!-- WC26:AUTO:" + re.escape(name) + r" END -->)",
        re.DOTALL,
    )
    new_start = f"<!-- WC26:AUTO:{name} START | generated {_now()} | source {source} -->"
    end = f"<!-- WC26:AUTO:{name} END -->"
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: f"{new_start}\n{content.strip()}\n{end}", text)


def set_frontmatter(text: str, key: str, value) -> str:
    """Replace `key: ...` ONLY inside the leading YAML frontmatter (no-op if missing).

    Anchoring to the frontmatter block avoids rewriting a HUMAN body line that
    happens to start with the same key text.
    """
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    fm = parts[1]
    pat = re.compile(r"(?m)^(" + re.escape(key) + r":).*$")
    if not pat.search(fm):
        return text
    fm = pat.sub(f"{key}: {value}", fm, count=1)
    return parts[0] + "---" + fm + "---" + parts[2]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _freshness(result: dict) -> str:
    as_of = result.get("data_as_of") or "unknown"
    stale = ""
    try:
        age_days = (datetime.now(timezone.utc).date()
                    - datetime.fromisoformat(as_of).date()).days
        if age_days * 24 > config.STALE_HOURS:
            stale = f"  **STALE** (inputs {age_days}d old)"
    except Exception:
        pass
    return f"Data as of {as_of}{stale} | runs {result.get('n_runs')} | seed {result.get('seed')}"


def write_dashboard(result: dict) -> None:
    probs = result["probs"]
    champ = sorted(probs.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    lines = [f"> {_freshness(result)}", "",
             "These are probabilities, not certainties.", "",
             "## Champion ladder", "",
             "| Team | Win | Final | SF | QF | R16 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, p in champ[:24]:
        lines.append(f"| [[Countries/{_link(name)}\\|{name}]] | {_pct(p['champion'])} | "
                     f"{_pct(p['final'])} | {_pct(p['sf'])} | {_pct(p['qf'])} | {_pct(p['r16'])} |")
    lines += ["", "## Most likely scorers (expected group-stage goals)", ""]
    for b in result.get("golden_boot", [])[:15]:
        lines.append(f"- {b['player']} ({b['team']}): {b['exp_group_goals']}")
    body = "\n".join(lines)
    text = (f"---\ntype: wc-dashboard\nupdated: {datetime.now(timezone.utc).date()}\n---\n\n"
            f"# World Cup 2026 Forecast\n\n"
            f"<!-- WC26:AUTO:dashboard START -->\n{body}\n<!-- WC26:AUTO:dashboard END -->\n")
    (VAULT / "_forecast.md").write_text(text)


def update_country_notes(result: dict, meta: dict) -> int:
    probs = result["probs"]
    n = 0
    for name, p in probs.items():
        stem = NOTE_FILENAME_ALIASES.get(name, _safe(name))
        path = COUNTRIES / f"{stem}.md"
        if not path.exists():
            continue
        text = path.read_text()
        m = meta.get(name, {})
        snapshot = (f"Group {m.get('group','?')} | FIFA rank {m.get('fifa_rank','?')} | "
                    f"Elo {round(m.get('elo',0))}\n"
                    f"Champion {_pct(p['champion'])} | Final {_pct(p['final'])} | "
                    f"SF {_pct(p['sf'])} | QF {_pct(p['qf'])} | Advance {_pct(p['advance'])}")
        text = replace_auto_block(text, "snapshot", snapshot, source="elo-goal-v1")
        text = set_frontmatter(text, "champion_prob", round(p["champion"], 4))
        if m.get("group"):
            text = set_frontmatter(text, "group", f"\"{m['group']}\"")
        text = set_frontmatter(text, "elo", round(m.get("elo", 0)))
        if m.get("fifa_rank"):
            text = set_frontmatter(text, "fifa_rank", m["fifa_rank"])
        text = set_frontmatter(text, "updated", datetime.now(timezone.utc).date())
        change = f"- {datetime.now(timezone.utc).date()}: champion {_pct(p['champion'])} (run {result.get('n_runs')} sims)"
        text = replace_auto_block(text, "changelog", change, source="elo-goal-v1")
        path.write_text(text)
        n += 1
    return n


def update_group_notes(result: dict, meta: dict) -> int:
    probs = result["probs"]
    by_group: dict[str, list[str]] = {}
    for name, m in meta.items():
        by_group.setdefault(m.get("group"), []).append(name)
    n = 0
    for letter, teams in by_group.items():
        path = GROUPS / f"Group {letter}.md"
        if not path.exists():
            continue
        ranked = sorted(teams, key=lambda t: probs[t]["advance"], reverse=True)
        rows = ["| Team | Win group | Top 2 | Advance |", "| --- | --- | --- | --- |"]
        for t in ranked:
            p = probs[t]
            rows.append(f"| [[Countries/{_link(t)}\\|{t}]] | {_pct(p['win_group'])} | "
                        f"{_pct(p['top2'])} | {_pct(p['advance'])} |")
        text = path.read_text()
        text = replace_auto_block(text, "standings", "\n".join(rows), source="elo-goal-v1")
        path.write_text(text)
        n += 1
    return n


def write_match_notes(result: dict) -> int:
    MATCHES.mkdir(exist_ok=True)
    n = 0
    for m in result["matches"]:
        scorers = " ".join(
            f"{s['player']} {_pct(s['p_anytime'])};"
            for s in (m["top_scorers_home"] + m["top_scorers_away"]))
        body = (f"Win {m['home']}: {_pct(m['p_home'])} | Draw: {_pct(m['p_draw'])} | "
                f"Win {m['away']}: {_pct(m['p_away'])}\n"
                f"Most likely scoreline: {m['top_scoreline']} ({_pct(m['top_scoreline_p'])}) "
                f"(modal only; many outcomes possible)\n"
                f"Top scorers: {scorers}\n{_freshness(result)}")
        fname = MATCHES / f"{m['group']} - {_safe(m['home'])} vs {_safe(m['away'])}.md"
        head = (f"---\ntype: wc-match\nstage: group\ngroup: {m['group']}\n"
                f"home: \"{m['home']}\"\naway: \"{m['away']}\"\n"
                f"updated: {datetime.now(timezone.utc).date()}\n---\n\n"
                f"# {m['home']} vs {m['away']}\n\n"
                f"<!-- WC26:AUTO:forecast START -->\n{body}\n<!-- WC26:AUTO:forecast END -->\n\n"
                f"## Post-match\n")
        # preserve human content keyed on the marker (not the cosmetic heading)
        human = "<!-- WC26:HUMAN:notes -->\n"
        if fname.exists():
            idx = fname.read_text().find("<!-- WC26:HUMAN:notes -->")
            if idx != -1:
                human = fname.read_text()[idx:]
        fname.write_text(head + human)
        n += 1
    return n


def generate(conn, result: dict) -> dict:
    meta = {}
    for r in conn.execute(
        "SELECT t.name n, t.group_letter g, t.fifa_rank fr, rt.elo e "
        "FROM teams t JOIN ratings rt ON rt.team_id=t.id"
    ):
        meta[r["n"]] = {"group": r["g"], "fifa_rank": r["fr"], "elo": r["e"]}
    write_dashboard(result)
    return {
        "countries": update_country_notes(result, meta),
        "groups": update_group_notes(result, meta),
        "matches": write_match_notes(result),
    }
