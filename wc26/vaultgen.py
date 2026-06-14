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


def _scoreline_block(m: dict) -> str:
    """Render the scoreline distribution: the most-likely single score (with its
    small probability so it reads as one possibility, not 'the prediction'), the
    top-5 scorelines, and the headline side-markets (O/U 2.5, BTTS, expected score)."""
    lines = []
    tops = m.get("top_scorelines") or []
    if m.get("top_scoreline") is not None and m.get("top_scoreline_p") is not None:
        lines.append(f"Most likely single score: {m['top_scoreline']} "
                     f"({_pct(m['top_scoreline_p'])}) (one of many possible, not a prediction)")
    if tops:
        cum = sum(p for _, p in tops)
        dist = ", ".join(f"{i}-{j} {_pct(p)}" for (i, j), p in tops)
        lines.append(f"Top-5 scorelines: {dist} (cumulative {_pct(cum)})")
    if m.get("expected_score"):
        lines.append(f"Expected score: {m['expected_score']}")
    dm = m.get("derived_markets") or {}
    if dm:
        lines.append(f"Over 2.5: {_pct(dm['over_2_5'])} | Under 2.5: {_pct(dm['under_2_5'])} | "
                     f"BTTS yes: {_pct(dm['btts_yes'])} | BTTS no: {_pct(dm['btts_no'])}")
    return "\n".join(lines)


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
        inj = m.get("injuries") or []
        inj_line = ("Availability concerns: " + ", ".join(inj[:8])) if inj \
            else "Availability: no concerns recorded"
        snapshot = (f"Group {m.get('group','?')} | FIFA rank {m.get('fifa_rank','?')} | "
                    f"Elo {round(m.get('elo',0))}\n"
                    f"Champion {_pct(p['champion'])} | Final {_pct(p['final'])} | "
                    f"SF {_pct(p['sf'])} | QF {_pct(p['qf'])} | Advance {_pct(p['advance'])}\n"
                    f"{inj_line}")
        text = replace_auto_block(text, "snapshot", snapshot, source="elo-goal-v1")
        text = set_frontmatter(text, "champion_prob", round(p["champion"], 4))
        if m.get("group"):
            text = set_frontmatter(text, "group", f"\"{m['group']}\"")
        if m.get("fifa_code"):
            text = set_frontmatter(text, "fifa_code", f"\"{m['fifa_code']}\"")
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


def _dprefix(d) -> str:
    return str(d)[:10] if d else "TBD"


def _emit_match_note(path, frontmatter: dict, title: str, auto_body: str) -> None:
    head = "---\n" + "\n".join(f"{k}: {v}" for k, v in frontmatter.items()) + "\n---\n\n"
    head += (f"# {title}\n\n<!-- WC26:AUTO:forecast START -->\n{auto_body.strip()}\n"
             f"<!-- WC26:AUTO:forecast END -->\n\n## Post-match\n")
    human = "<!-- WC26:HUMAN:notes -->\n"
    if path.exists():
        idx = path.read_text().find("<!-- WC26:HUMAN:notes -->")
        if idx != -1:
            human = path.read_text()[idx:]
    path.write_text(head + human)


def write_match_notes(result: dict) -> dict:
    """Match notes split into Matches/{Group,Friendly,Knockout}, each dated and typed."""
    counts = {"group": 0, "friendly": 0, "knockout": 0}

    gdir = MATCHES / "Group"; gdir.mkdir(parents=True, exist_ok=True)
    for m in result.get("matches", []):
        scorers = " ".join(f"{s['player']} {_pct(s['p_anytime'])};"
                           for s in (m["top_scorers_home"] + m["top_scorers_away"]))
        body = (f"**Group {m['group']} match** · {_dprefix(m.get('date'))}\n\n"
                f"Win {m['home']}: {_pct(m['p_home'])} | Draw: {_pct(m['p_draw'])} | "
                f"Win {m['away']}: {_pct(m['p_away'])}\n"
                f"{_scoreline_block(m)}\n"
                f"Top scorers: {scorers}\n{_freshness(result)}")
        fm = {"type": "wc-match", "stage": "group", "group": m["group"],
              "date": _dprefix(m.get("date")), "home": f'"{m["home"]}"', "away": f'"{m["away"]}"',
              "updated": datetime.now(timezone.utc).date()}
        path = gdir / f"{_dprefix(m.get('date'))} {_safe(m['home'])} vs {_safe(m['away'])}.md"
        _emit_match_note(path, fm, f"{m['home']} vs {m['away']}", body)
        counts["group"] += 1

    fdir = MATCHES / "Friendly"; fdir.mkdir(parents=True, exist_ok=True)
    for f in result.get("friendlies", []):
        status = f"played {f['result']}" if f["played"] else "scheduled"
        body = (f"**Warm-up friendly** · {_dprefix(f.get('date'))} · {status}\n\n"
                f"Win {f['home']}: {_pct(f['p_home'])} | Draw: {_pct(f['p_draw'])} | "
                f"Win {f['away']}: {_pct(f['p_away'])}\n"
                f"{_scoreline_block(f)}\n{_freshness(result)}")
        fm = {"type": "wc-match", "stage": "friendly", "date": _dprefix(f.get("date")),
              "home": f'"{f["home"]}"', "away": f'"{f["away"]}"',
              "updated": datetime.now(timezone.utc).date()}
        path = fdir / f"{_dprefix(f.get('date'))} {_safe(f['home'])} vs {_safe(f['away'])}.md"
        _emit_match_note(path, fm, f"{f['home']} vs {f['away']} (friendly)", body)
        counts["friendly"] += 1

    kdir = MATCHES / "Knockout"; kdir.mkdir(parents=True, exist_ok=True)
    for k in result.get("knockout", []):
        h = k.get("home_proj") or k["home_slot"]
        a = k.get("away_proj") or k["away_slot"]
        body = (f"**{k['stage']} (match #{k['match_no']})** · {_dprefix(k.get('date'))}\n\n"
                f"Slots: {k['home_slot']} vs {k['away_slot']}\n"
                f"Projected: {h} vs {a}\n"
                f"Result: {k.get('result') or 'TBD'}\n{_freshness(result)}")
        fm = {"type": "wc-match", "stage": k["stage"], "match_no": k["match_no"],
              "date": _dprefix(k.get("date")), "updated": datetime.now(timezone.utc).date()}
        path = kdir / f"{k['match_no']:03d} {k['stage']} {_safe(k['home_slot'])} vs {_safe(k['away_slot'])}.md"
        _emit_match_note(path, fm, f"{k['stage']} #{k['match_no']}: {k['home_slot']} vs {k['away_slot']}", body)
        counts["knockout"] += 1

    return counts


def write_bracket(result: dict) -> None:
    """Mermaid knockout bracket. Shows the structure with projected qualifiers now,
    and fills with real teams/winners as the group stage and knockouts complete."""
    from .simulate import KNOCKOUT_TREE
    ko = {k["match_no"]: k for k in result.get("knockout", [])}

    def label(mno: int) -> str:
        k = ko.get(mno)
        if not k:
            return f"#{mno}"
        h = k.get("home_proj") or k.get("home_slot") or "?"
        a = k.get("away_proj") or k.get("away_slot") or "?"
        res = f" [{k['result']}]" if k.get("result") else ""
        return f"{k['stage']} #{mno}<br>{h} vs {a}{res}".replace('"', "'")

    lines = ["```mermaid", "flowchart LR"]
    all_nos = set(range(73, 89)) | set(KNOCKOUT_TREE)
    for mno in sorted(all_nos):
        lines.append(f'  M{mno}["{label(mno)}"]')
    for mno, (a, b) in KNOCKOUT_TREE.items():
        lines.append(f"  M{a} --> M{mno}")
        lines.append(f"  M{b} --> M{mno}")
    lines.append("```")
    text = (f"---\ntype: wc-bracket\nupdated: {datetime.now(timezone.utc).date()}\n---\n\n"
            f"# Knockout bracket\n\n"
            f"Empty for now: teams are projected (current most-likely qualifier per slot) and "
            f"fill in with real teams as the group stage finishes and with winners as knockout "
            f"matches are played.\n\n"
            f"<!-- WC26:AUTO:bracket START -->\n" + "\n".join(lines) + "\n<!-- WC26:AUTO:bracket END -->\n")
    (VAULT / "_bracket.md").write_text(text)


def generate(conn, result: dict) -> dict:
    meta = {}
    for r in conn.execute(
        "SELECT t.name n, t.group_letter g, t.fifa_rank fr, t.fifa_code fc, rt.elo e "
        "FROM teams t JOIN ratings rt ON rt.team_id=t.id "
        "WHERE rt.valid_from = (SELECT MAX(r2.valid_from) FROM ratings r2 WHERE r2.team_id=t.id)"
    ):
        meta[r["n"]] = {"group": r["g"], "fifa_rank": r["fr"], "fifa_code": r["fc"], "elo": r["e"]}
    for r in conn.execute(
        "SELECT t.name n, ae.player_name p, ae.status s FROM availability_events ae "
        "JOIN teams t ON t.id=ae.team_id "
        "WHERE ae.player_name != '(scan-marker)' AND ae.status != 'fit'"
    ):
        if r["n"] in meta:
            meta[r["n"]].setdefault("injuries", []).append(f"{r['p']} ({r['s']})")
    write_dashboard(result)
    write_bracket(result)
    return {
        "countries": update_country_notes(result, meta),
        "groups": update_group_notes(result, meta),
        "matches": write_match_notes(result),
        "bracket": "written",
    }
