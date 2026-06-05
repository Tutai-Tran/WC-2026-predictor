"""Streamlit dashboard (read-only). Renders the latest forecast snapshot from the DB.

Run:  streamlit run wc26/app.py
All computation happens in the pipeline; this only displays precomputed results.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# `streamlit run wc26/app.py` puts wc26/ on sys.path, not the repo root, so the
# package import fails. Add the repo root so `import wc26` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from wc26 import accuracy, config, db, model

st.set_page_config(page_title="World Cup 2026 Forecast", page_icon="⚽", layout="wide")


def _secret_or_env(key: str):
    """Read config from Streamlit secrets (hosted) or the environment, else None."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


# Set ONLY on the hosted deployment (Streamlit secret / env). When present, the
# app pulls the latest DB snapshot the Mac publishes; when absent (local run),
# the app keeps using its own live wc26.db and never downloads anything.
_DB_URL = _secret_or_env("WC26_DB_URL")


@st.cache_data(ttl=60, show_spinner="Syncing latest data…")
def _sync_remote_db() -> str | None:
    """Pull the published DB snapshot (hosted only). Returns a short content
    fingerprint so load_latest reloads when the data changes; None = no-op."""
    if not _DB_URL:
        return None
    try:
        import requests
        r = requests.get(_DB_URL, timeout=30)
        r.raise_for_status()
        tmp = str(config.DB_PATH) + ".download"
        Path(tmp).write_bytes(r.content)
        os.replace(tmp, config.DB_PATH)            # atomic swap
        return hashlib.sha256(r.content).hexdigest()[:16]
    except Exception:
        return None                                # keep whatever DB is already present


@st.cache_data(ttl=60)
def load_latest(db_fingerprint=None):
    # db_fingerprint participates in the cache key: a new published snapshot
    # changes it and forces a reload (hosted). Unused value otherwise.
    conn = db.connect()
    db.init_db(conn)
    row = conn.execute(
        "SELECT payload_json, computed_at, run_id FROM predictions "
        "WHERE scope='forecast' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload_json"])
    payload["_computed_at"] = row["computed_at"]
    return payload


def _date(s):
    if not s:
        return "TBD"
    return str(s)[:10]


def _pct(x):
    return f"{x*100:.0f}%"


# All match scheduling is shown in Amsterdam local time. Group-stage and knockout
# fixtures carry a UTC kickoff (e.g. "2026-06-11T19:00:00Z"); friendlies in the data
# have a date only, so their kickoff time shows as "—".
_AMS = ZoneInfo("Europe/Amsterdam")


def _ams_parts(value) -> tuple[str, str]:
    """Return (Amsterdam date, Amsterdam "HH:MM") for a stored kickoff.

    Date-only values (no time in the data) yield the date and an empty time."""
    if not value:
        return "TBD", ""
    s = str(value)
    if "T" in s:
        try:
            local = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(_AMS)
            return local.date().isoformat(), local.strftime("%H:%M")
        except ValueError:
            pass
    return s[:10], ""


# Row highlights: blue = the match is today, green = already played, none = other
# upcoming. A match is "played" once its status/result cell holds a score
# (e.g. "2-1" / "played 2-1") rather than a placeholder ("scheduled" / "-" / "TBD").
# "Today" takes precedence so the day's fixtures always stand out; the ✅ marker in
# the status cell still flags played matches regardless of row colour.
_PLAYED_BG = "background-color: rgba(38, 166, 91, 0.18)"   # green
_TODAY_BG = "background-color: rgba(56, 132, 255, 0.20)"    # blue


def _today_iso() -> str:
    return datetime.now(_AMS).date().isoformat()


def _is_played(value) -> bool:
    return str(value).strip().lower() not in ("", "scheduled", "-", "tbd", "nan", "none")


def _highlight_rows(frame: pd.DataFrame, date_col: str | None = None,
                    status_col: str | None = None, today: str | None = None):
    """Tint whole rows blue if the row's date is today, else green if played."""
    today = today or _today_iso()

    def _row(r):
        if date_col is not None and str(r[date_col])[:10] == today:
            css = _TODAY_BG
        elif status_col is not None and _is_played(r[status_col]):
            css = _PLAYED_BG
        else:
            css = ""
        return [css] * len(r)

    return frame.style.apply(_row, axis=1)


_SIDE_LABEL = {"home": "Home win", "draw": "Draw", "away": "Away win"}


def _called(p_home, p_draw, p_away, result):
    """Did our top W/D/L pick match the actual result of a played match?

    Returns (correct: bool, predicted_side, actual_side) or None when the match
    has no result or no probabilities to judge against."""
    if result is None or p_home is None:
        return None
    try:
        hs, as_ = map(int, str(result).split("-"))
    except (ValueError, AttributeError):
        return None
    pred = _SIDE_LABEL[model.outcome_label(p_home, p_draw, p_away)]
    actual = _SIDE_LABEL[model.result_label(hs, as_)]
    return pred == actual, pred, actual


def _called_cell(p_home, p_draw, p_away, result):
    """Compact 'did we get the result right' cell: '' when not yet played."""
    c = _called(p_home, p_draw, p_away, result)
    if c is None:
        return ""
    correct, pred, _actual = c
    return "✅ correct" if correct else f"❌ said {pred.split()[0]}"


data = load_latest(_sync_remote_db())
st.title("⚽ World Cup 2026 Forecast")

if data is None:
    if _DB_URL:
        st.info("Waiting for the first live data snapshot from the source device. "
                "This page updates automatically once data is published.")
    else:
        st.warning("No forecast yet. Run:  python -m wc26.forecast")
    st.stop()

# ---- freshness banner ----
as_of = data.get("data_as_of", "unknown")
stale = False
try:
    age_h = (datetime.now(timezone.utc).date() - datetime.fromisoformat(as_of).date()).days * 24
    stale = age_h > config.STALE_HOURS
except Exception:
    pass
banner = (f"Data as of **{as_of}**  ·  {data['n_runs']:,} simulations  ·  "
          f"updated {str(data['_computed_at'])[:16].replace('T', ' ')}")
# quiet status when fresh (the 99% case), a loud alert only when actually stale
if stale:
    st.error("⚠️ Data may be stale — " + banner)
else:
    st.caption("🟢 " + banner)
st.caption("These are probabilities, not certainties. A single most-likely scoreline is usually "
           "only ~10% likely, and even the top scorer pick misses about half the time.")

probs = data["probs"]
champ_sorted = sorted(probs.items(), key=lambda kv: kv[1]["champion"], reverse=True)

# ---- headline favourites ----
with st.container(border=True):
    st.markdown("##### 🏆 Title favourites")
    cols = st.columns(4)
    for rank, (col, (name, p)) in enumerate(zip(cols, champ_sorted[:4]), start=1):
        col.metric(f"{rank}. {name}", _pct(p["champion"]),
                   help="Probability of winning the tournament")
    st.caption("Top 4 by simulated title probability. Full table on the **Champion & stages** tab.")

# rate metrics share a ~50% coin-flip baseline and are graded green/amber/red;
# exact score & goal difference are low-ceiling by nature (a single scoreline is
# ~10% likely even for a perfect model), so they are shown neutral, not "failing".
_RATE_METRICS = {"outcome", "favourite", "ou25", "btts"}
_NEUTRAL = "#4a78c4"
_DONUT_BG = "#E7E9E4"
# darker text variants so the centred % meets WCAG contrast on the light hole
# (the lighter arc-fill colour can stay vivid).
_TEXT_COLOR = {"#00875A": "#00875A", "#E8A33D": "#B26A00", "#D1495B": "#D1495B",
               _NEUTRAL: "#33558f"}


def _donut_color(key: str, pct: float) -> str:
    if key not in _RATE_METRICS:
        return _NEUTRAL
    if pct >= 58:
        return "#00875A"   # green: clearly beating a coin flip
    if pct >= 48:
        return "#E8A33D"   # amber: around chance
    return "#D1495B"       # red: below chance


def _accuracy_donut(key: str, correct: int, total: int):
    """A thin ring with a big centred % — the focal number the eye wants."""
    if not total:
        return alt.Chart(pd.DataFrame({"v": [1]})).mark_arc().encode(theta="v:Q")
    pct = correct / total * 100.0
    color = _donut_color(key, pct)
    ring = pd.DataFrame({"k": ["correct", "wrong"], "v": [correct, total - correct],
                         "o": [0, 1]})
    arc = (
        alt.Chart(ring).mark_arc(innerRadius=46, outerRadius=62, cornerRadius=3).encode(
            theta=alt.Theta("v:Q", stack=True), order=alt.Order("o:Q"),
            color=alt.Color("k:N", scale=alt.Scale(domain=["correct", "wrong"],
                                                    range=[color, _DONUT_BG]), legend=None),
            tooltip=[alt.Tooltip("k:N", title=""), alt.Tooltip("v:Q", title="matches")],
        )
    )
    center = (alt.Chart(pd.DataFrame({"t": [f"{pct:.0f}%"]}))
              .mark_text(fontSize=26, fontWeight="bold", color=_TEXT_COLOR.get(color, color))
              .encode(text="t:N"))
    return alt.layer(arc, center).properties(height=140).configure_view(strokeWidth=0)


# ---- model accuracy scorecard (how often we are right, per metric) ----
_played = accuracy.played_from_payload(data)
if _played:
    _rows = [r for r in accuracy.metrics(_played) if r["total"]]
    _n = max((r["total"] for r in _rows), default=0)
    with st.container(border=True):
        st.markdown(f"##### 🎯 How accurate has the model been? · {_n} played matches")
        st.caption(
            "Share of played matches we called right, per prediction type. We nail the **result** "
            "(win/draw/loss) far more often than the **exact score**, which is inherently "
            "low-probability in football, so judge the model on the green metrics (outcome, "
            "over/under, both-teams-to-score). Goal difference and exact score (blue) are "
            "low-ceiling by nature. Parameters are fit leak-free on 14k+ historical internationals "
            "and the ratings update from every result, so it keeps learning."
        )
        _cols = st.columns(len(_rows))
        for _c, _r in zip(_cols, _rows):
            _c.markdown(f"<div style='text-align:center;font-weight:600;font-size:0.9rem;"
                        f"min-height:2.6em;line-height:1.2;display:flex;align-items:center;"
                        f"justify-content:center'>{_r['label']}</div>", unsafe_allow_html=True)
            _c.altair_chart(_accuracy_donut(_r["key"], _r["correct"], _r["total"]),
                            use_container_width=True)
            _c.markdown(f"<div style='text-align:center;color:#4B5563;font-size:0.82rem;"
                        f"margin-top:-0.6em'>{_r['correct']} / {_r['total']} correct</div>",
                        unsafe_allow_html=True)
        st.write("")   # breathing room so the rings don't touch the card border
else:
    st.caption("🎯 Accuracy scoring will appear here once matches have been played.")

st.divider()

# friendly column labels + percent formatting for the match tables (the row
# highlighting from _highlight_rows still applies; this only renames/formats).
_MATCH_COLCFG = {
    "Kickoff (AMS)": st.column_config.TextColumn("Kickoff"),
    "Grp": st.column_config.TextColumn("Group"),
    "H%": st.column_config.NumberColumn("Home win", format="%d%%"),
    "D%": st.column_config.NumberColumn("Draw", format="%d%%"),
    "A%": st.column_config.NumberColumn("Away win", format="%d%%"),
    "Likely": st.column_config.TextColumn("Likely score"),
    "Top scorers": st.column_config.TextColumn("Top scorers", width="large"),
}

_DAILY_COLCFG = {
    "date": st.column_config.TextColumn("Date"),
    "time (AMS)": st.column_config.TextColumn("Kickoff"),
    "type": st.column_config.TextColumn("Stage"),
    "match": st.column_config.TextColumn("Match", width="medium"),
    "forecast": st.column_config.TextColumn("Win / Draw / Away %",
                                            help="Model probabilities: home win / draw / away win"),
    "likely": st.column_config.TextColumn("Likely score"),
    "status": st.column_config.TextColumn("Status"),
    "our call": st.column_config.TextColumn("Our call",
                                            help="✅ if our win/draw/loss pick matched the result"),
}

tabs = st.tabs(["📅 Daily", "🏆 Champion & stages", "⚽ Matches", "🔀 Bracket",
                "👥 Groups", "🥅 Scorers", "📋 Availability", "📈 Calibration", "🔎 Team"])

# =============================== Daily ===============================
with tabs[0]:
    st.subheader("Matches by day")
    rows = []
    for m in data.get("matches", []):
        d, t = _ams_parts(m.get("date"))
        gres = m.get("result")
        rows.append({"date": d, "time (AMS)": t or "—", "type": f"Group {m['group']}",
                     "match": f"{m['home']} vs {m['away']}",
                     "forecast": f"{_pct(m['p_home'])} / {_pct(m['p_draw'])} / {_pct(m['p_away'])}",
                     "likely": m["top_scoreline"],
                     "status": (f"✅ played {gres}" if gres else "scheduled"),
                     "our call": _called_cell(m.get("p_home"), m.get("p_draw"), m.get("p_away"), gres)})
    for f in data.get("friendlies", []):
        d, t = _ams_parts(f.get("date"))
        rows.append({"date": d, "time (AMS)": t or "—", "type": "Friendly",
                     "match": f"{f['home']} vs {f['away']}",
                     "forecast": f"{_pct(f['p_home'])} / {_pct(f['p_draw'])} / {_pct(f['p_away'])}",
                     "likely": f["top_scoreline"],
                     "status": (f"✅ played {f['result']}" if f["played"] else "scheduled"),
                     "our call": _called_cell(f.get("p_home"), f.get("p_draw"), f.get("p_away"), f.get("result"))})
    for k in data.get("knockout", []):
        if not k.get("date"):
            continue
        d, t = _ams_parts(k["date"])
        home = k.get("home_proj") or k["home_slot"]
        away = k.get("away_proj") or k["away_slot"]
        rows.append({"date": d, "time (AMS)": t or "—", "type": k["stage"],
                     "match": f"{home} vs {away}", "forecast": "(projected teams)",
                     "likely": "-", "status": (f"✅ {k['result']}" if k.get("result") else "scheduled"),
                     "our call": ""})
    df = pd.DataFrame(rows).sort_values(["date", "time (AMS)", "type"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    today = _today_iso()
    upcoming = [d for d in dates if d >= today]
    default = upcoming[0] if upcoming else (dates[-1] if dates else None)
    pick = st.select_slider("Pick a date", options=dates,
                            value=default) if dates else None
    if pick:
        day = df[df["date"] == pick]
        st.markdown(f"### {pick} — {len(day)} match(es)" + ("  ·  📍 today" if pick == today else ""))
        st.caption("Forecast column is Win / Draw / Win (home/away). The 'our call' column shows "
                   "✅ if our forecast got the result (win/draw/loss) right and ❌ (with what we "
                   "said) if not. Kickoff times are Amsterdam local (CEST); matches without a "
                   "scheduled time show —. 🔵 Today's matches are highlighted blue, ✅ played green.")
        day_disp = day.drop(columns=["date"])
        if pick == today:
            styled = day_disp.style.apply(lambda r: [_TODAY_BG] * len(r), axis=1)
        else:
            styled = _highlight_rows(day_disp, status_col="status")
        st.caption("↔ On a phone, swipe the table sideways to see every column.")
        st.dataframe(styled, hide_index=True, use_container_width=True,
                     column_config=_DAILY_COLCFG)
    with st.expander("See the full schedule (all days)"):
        st.dataframe(_highlight_rows(df, date_col="date", status_col="status", today=today),
                     hide_index=True, use_container_width=True, height=400,
                     column_config=_DAILY_COLCFG)

# ======================== Champion & stages =========================
with tabs[1]:
    blend = {}
    try:
        from wc26 import odds
        blend = odds.blended_champion(db.connect())
    except Exception:
        blend = {}
    n_runs = max(1, data.get("n_runs", 1))
    rows = []
    for n, p in probs.items():
        c = p["champion"]
        ci = round(1.96 * (c * (1 - c) / n_runs) ** 0.5 * 100, 2)  # Monte Carlo 95% CI
        row = {"Team": n, "Win %": round(c * 100, 1), "± (95%)": ci}
        if blend:
            row["Win % (model+market)"] = round(blend.get(n, p["champion"])*100, 1)
        row.update({"Final %": round(p["final"]*100, 1), "SF %": round(p["sf"]*100, 1),
                    "QF %": round(p["qf"]*100, 1), "R16 %": round(p["r16"]*100, 1),
                    "Advance %": round(p["advance"]*100, 1)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Win %", ascending=False).reset_index(drop=True)
    if blend:
        st.caption("'Model + market' blends the model with devigged market odds "
                   "(usually better calibrated). Plain 'Champion' is the pure model.")
    _champ_cfg = {
        "Win %": st.column_config.ProgressColumn(
            "Champion", format="%.1f%%", min_value=0.0,
            max_value=float(df["Win %"].max()) if len(df) else 1.0,
            help="Probability of winning the tournament"),
        "± (95%)": st.column_config.NumberColumn("± 95% CI", format="%.1f",
            help="Monte-Carlo 95% confidence interval on the champion %"),
        "Win % (model+market)": st.column_config.NumberColumn("Model + market", format="%.1f%%"),
        "Final %": st.column_config.NumberColumn("Reach final", format="%.1f%%"),
        "SF %": st.column_config.NumberColumn("Semi-final", format="%.1f%%"),
        "QF %": st.column_config.NumberColumn("Quarter-final", format="%.1f%%"),
        "R16 %": st.column_config.NumberColumn("Round of 16", format="%.1f%%"),
        "Advance %": st.column_config.NumberColumn("Advance (R32)", format="%.1f%%",
            help="Probability of getting out of the group"),
    }
    st.dataframe(df, use_container_width=True, height=600, hide_index=True,
                 column_config=_champ_cfg)

# ============================== Matches =============================
with tabs[2]:
    kind = st.radio("Competition", ["Group stage", "Friendlies (warm-up)", "Knockouts"],
                    horizontal=True)
    if kind == "Group stage":
        mrows = []
        for m in data.get("matches", []):
            sc = ", ".join(f"{s['player']} {_pct(s['p_anytime'])}"
                           for s in (m["top_scorers_home"][:2] + m["top_scorers_away"][:2]))
            d, t = _ams_parts(m.get("date"))
            gres = m.get("result")
            mrows.append({"Date": d, "Kickoff (AMS)": t or "—", "Grp": m["group"],
                          "Home": m["home"], "Away": m["away"],
                          "H%": round(m["p_home"]*100), "D%": round(m["p_draw"]*100),
                          "A%": round(m["p_away"]*100), "Likely": m["top_scoreline"],
                          "Our call": _called_cell(m.get("p_home"), m.get("p_draw"), m.get("p_away"), gres),
                          "Status": (f"✅ played {gres}" if gres else "scheduled"),
                          "Top scorers": sc})
        st.caption("Kickoff times are Amsterdam local (CEST). 🔵 Today's matches are highlighted blue, "
                   "✅ played green; 'Our call' shows ✅/❌ once a match has a result.")
        st.dataframe(_highlight_rows(pd.DataFrame(mrows).sort_values(["Date", "Kickoff (AMS)"]),
                                     date_col="Date", status_col="Status"),
                     hide_index=True, use_container_width=True, height=560,
                     column_config=_MATCH_COLCFG)
    elif kind == "Friendlies (warm-up)":
        frows = []
        n_played = n_correct = 0
        for f in data.get("friendlies", []):
            d, t = _ams_parts(f.get("date"))
            c = _called(f.get("p_home"), f.get("p_draw"), f.get("p_away"), f.get("result"))
            if c is not None:
                n_played += 1
                n_correct += int(c[0])
            frows.append({"Date": d, "Kickoff (AMS)": t or "—", "Home": f["home"], "Away": f["away"],
                          "H%": round(f["p_home"]*100), "D%": round(f["p_draw"]*100),
                          "A%": round(f["p_away"]*100), "Likely": f["top_scoreline"],
                          "Our call": _called_cell(f.get("p_home"), f.get("p_draw"), f.get("p_away"), f.get("result")),
                          "Status": (f"✅ played {f['result']}" if f["played"] else "scheduled")})
        if n_played:
            st.success(f"Our win/draw/loss call was correct in **{n_correct}/{n_played}** played "
                       f"friendlies (**{n_correct/n_played*100:.0f}%**). Exact scorelines are far "
                       "less predictable than the result, so the 'Our call' column judges the model "
                       "on the outcome it picked, not the exact score.")
            st.caption("Heads-up: these forecasts use the current Elo ratings, which already include "
                       "these friendly results, so this figure is an optimistic upper bound rather "
                       "than a clean out-of-sample score. The leak-free skill estimate (historical "
                       "backtest) is on the 📈 Calibration tab.")
        st.caption("Warm-up friendlies in the run-up to the World Cup. Kickoff times are Amsterdam "
                   "local (CEST) where the data has them, else —. 🔵 Today's matches are highlighted "
                   "blue; played results (✅, green) also update the model's Elo when the refresh runs.")
        st.dataframe(_highlight_rows(pd.DataFrame(frows).sort_values("Date"),
                                     date_col="Date", status_col="Status"),
                     hide_index=True, use_container_width=True, height=560,
                     column_config=_MATCH_COLCFG)
    else:
        krows = []
        for k in data.get("knockout", []):
            d, t = _ams_parts(k.get("date"))
            krows.append({"#": k["match_no"], "Stage": k["stage"], "Date": d,
                          "Kickoff (AMS)": t or "—",
                          "Home slot": k["home_slot"], "Away slot": k["away_slot"],
                          "Projected": f"{k.get('home_proj') or '?'} vs {k.get('away_proj') or '?'}",
                          "Result": (f"✅ {k['result']}" if k.get("result") else "-")})
        st.caption("Knockout fixtures by slot. Kickoff times are Amsterdam local (CEST). Teams are "
                   "TBD until the group stage finishes; the 'Projected' column shows the current "
                   "most-likely qualifiers. 🔵 Today's matches are highlighted blue, ✅ played green.")
        st.dataframe(_highlight_rows(pd.DataFrame(krows), date_col="Date", status_col="Result"),
                     hide_index=True, use_container_width=True, height=560)

# ============================== Bracket =============================
with tabs[3]:
    st.subheader("Knockout bracket — who reaches the later rounds")
    st.caption("The bracket fills in with real teams as the group stage finishes and with "
               "winners as knockout matches are played. For now it shows the structure and the "
               "current most-likely qualifier for each slot.")
    ko = {k["match_no"]: k for k in data.get("knockout", [])}

    try:
        from wc26.simulate import KNOCKOUT_TREE

        def _nl(mno):
            k = ko.get(mno)
            if not k:
                return f"#{mno}"
            h = k.get("home_proj") or k["home_slot"]
            a = k.get("away_proj") or k["away_slot"]
            return f"{h}\\nvs {a}".replace('"', "'")

        dot = ["digraph B { rankdir=LR; bgcolor=transparent; "
               "node [shape=box style=rounded fontsize=9]; ranksep=0.6; nodesep=0.12;"]
        for mno in sorted(set(range(73, 89)) | set(KNOCKOUT_TREE)):
            dot.append(f'M{mno} [label="{_nl(mno)}"];')
        for mno, (a, b) in KNOCKOUT_TREE.items():
            dot.append(f"M{a} -> M{mno}; M{b} -> M{mno};")
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
    except Exception as e:
        st.caption(f"(diagram unavailable: {e})")

    def line(mno):
        k = ko.get(mno)
        if not k:
            return f"Match {mno}: TBD"
        h = k.get("home_proj") or k["home_slot"]
        a = k.get("away_proj") or k["away_slot"]
        res = f"  →  {k['result']}" if k.get("result") else ""
        d, t = _ams_parts(k.get("date"))
        when = f"{d} {t} AMS" if t else d
        return f"**{k['stage']}** (#{mno}, {when}): {h}  vs  {a}{res}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Round of 32")
        for mno in range(73, 89):
            st.markdown(line(mno))
    with c2:
        st.markdown("#### Round of 16")
        for mno in range(89, 97):
            st.markdown(line(mno))
        st.markdown("#### Quarter-finals")
        for mno in range(97, 101):
            st.markdown(line(mno))
        st.markdown("#### Semi-finals")
        for mno in (101, 102):
            st.markdown(line(mno))
        st.markdown("#### Final")
        st.markdown(line(104))
    st.info("A visual bracket diagram (Mermaid) is also generated in the vault at "
            "`WC vault/_bracket.md` — open it in Obsidian.")

# ============================== Groups =============================
with tabs[4]:
    groups: dict[str, list] = {}
    conn = db.connect()
    for r in conn.execute("SELECT name, group_letter FROM teams WHERE group_letter IS NOT NULL"):
        groups.setdefault(r["group_letter"], []).append(r["name"])
    cols = st.columns(3)
    for i, letter in enumerate(sorted(groups)):
        teams = sorted(groups[letter], key=lambda t: probs[t]["advance"], reverse=True)
        with cols[i % 3]:
            st.subheader(f"Group {letter}")
            st.dataframe(pd.DataFrame([
                {"Team": t, "Win grp %": round(probs[t]["win_group"]*100),
                 "Advance %": round(probs[t]["advance"]*100)} for t in teams
            ]), hide_index=True, use_container_width=True)

# ============================== Scorers =============================
with tabs[5]:
    st.subheader("Most likely scorers (expected group-stage goals)")
    st.dataframe(pd.DataFrame(data.get("golden_boot", [])), hide_index=True,
                 use_container_width=True)

# ============================ Availability ==========================
with tabs[6]:
    st.subheader("Current injuries & availability (scraped)")
    st.caption("From the LLM news agent (web search on your Max subscription), your vault "
               "overrides, and manual entries. These reduce a team's attacking strength in the forecast.")
    conn = db.connect()
    rows = conn.execute(
        "SELECT t.name team, ae.player_name player, ae.status, ae.source, ae.source_quote quote, ae.url "
        "FROM availability_events ae JOIN teams t ON t.id=ae.team_id "
        "WHERE ae.player_name != '(scan-marker)' AND ae.status != 'fit' "
        "ORDER BY t.name, ae.player_name"
    ).fetchall()
    if rows:
        st.dataframe(pd.DataFrame([
            {"Team": r["team"], "Player": r["player"], "Status": r["status"],
             "Source": r["source"], "Note": (r["quote"] or "")[:90]} for r in rows
        ]), hide_index=True, use_container_width=True)
    else:
        st.caption("No current availability concerns recorded yet. The news agent scans teams "
                   "on a rotating schedule; results appear here as it finds them.")
    scanned = conn.execute(
        "SELECT COUNT(DISTINCT team_id) c FROM availability_events WHERE source LIKE 'news-llm%'"
    ).fetchone()["c"]
    st.caption(f"Teams scanned by the news agent so far: {scanned}/48.")


# ============================ Calibration ===========================
with tabs[7]:
    st.subheader("How good is the model? (leak-free historical backtest)")
    rp = config.DATA_RAW / "backtest_report.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        b = r["baselines"]
        st.caption("Log loss on held-out matches, lower is better. The model should beat all baselines.")
        st.dataframe(pd.DataFrame([
            {"forecast": "our model", "log loss": r["test"]["log_loss"]},
            {"forecast": "favourite baseline", "log loss": b["favourite_log_loss"]},
            {"forecast": "climatology baseline", "log loss": b["climatology_log_loss"]},
            {"forecast": "uniform baseline", "log loss": b["uniform_log_loss"]},
        ]), hide_index=True, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Log loss", r["test"]["log_loss"],
                  help=f"95% CI {r['test']['log_loss_ci95']} · lower is better")
        c2.metric("Brier score", r["test"]["brier"], help="Lower is better")
        c3.metric("RPS", r["test"]["rps"],
                  help="Ranked Probability Score: the standard ordinal football metric (lower is better)")
        c4.metric("Reliability slope", r["reliability"]["home_win_slope"], help="1.0 = perfectly calibrated")
        st.caption(f"Tournament-only matches: model {r['tournament_only']['log_loss']} vs "
                   f"climatology {r['tournament_only']['climatology_log_loss']}.")
    else:
        st.info("Run `python -m wc26.backtest` to populate calibration metrics.")

    st.subheader("Running calibration vs played results")
    conn = db.connect()
    rows = conn.execute(
        "SELECT payload_json FROM predictions WHERE scope='calibration' ORDER BY id").fetchall()
    if rows:
        traj = pd.DataFrame([json.loads(r["payload_json"]) for r in rows])
        st.line_chart(traj.set_index("computed_at")[["brier", "log_loss"]])
    else:
        st.caption("No played results yet — the running-improvement trajectory fills in once matches start.")

    st.subheader("Model vs market (devigged World Cup winner odds)")
    try:
        from wc26 import odds
        comp = odds.model_vs_market(conn)
        if comp:
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)
            st.caption("Benchmark only, never betting. Edge = model minus market.")
        else:
            st.caption("Run `python -m wc26.odds` to fetch and compare market odds.")
    except Exception as e:
        st.caption(f"Market odds unavailable: {e}")

    st.subheader("Pre-tournament friendly accuracy (real out-of-sample check)")
    played = [f for f in data.get("friendlies", []) if f.get("played") and f.get("result")]
    if played:
        import math
        picks = ["Home win", "Draw", "Away win"]
        hit, ll, frows = 0, 0.0, []
        for f in played:
            try:
                hs, as_ = map(int, str(f["result"]).split("-"))
            except Exception:
                continue
            outcome = 0 if hs > as_ else (1 if hs == as_ else 2)
            p = [f["p_home"], f["p_draw"], f["p_away"]]
            pred = p.index(max(p))
            hit += pred == outcome
            ll += -math.log(max(1e-9, p[outcome]))
            frows.append({"Date": _date(f.get("date")), "Match": f"{f['home']} vs {f['away']}",
                          "Our pick": picks[pred], "Result": f["result"],
                          "✓": "✅" if pred == outcome else "❌"})
        n = len(frows)
        if n:
            c1, c2 = st.columns(2)
            c1.metric("Top-pick accuracy", f"{hit}/{n} ({hit/n*100:.0f}%)")
            c2.metric("Avg log loss", round(ll / n, 3))
            st.dataframe(pd.DataFrame(frows), hide_index=True, use_container_width=True)
    else:
        st.caption("No played friendlies in the latest snapshot yet.")

# ============================== Team ==============================
with tabs[8]:
    sel = st.selectbox("Select a team", sorted(probs.keys()))
    p = probs[sel]
    conn = db.connect()
    info = conn.execute(
        "SELECT t.group_letter g, t.fifa_rank fr, "
        "(SELECT elo FROM ratings WHERE team_id=t.id ORDER BY valid_from DESC LIMIT 1) elo "
        "FROM teams t WHERE t.name=?", (sel,)).fetchone()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion", _pct(p["champion"]))
    c2.metric("Reach final", _pct(p["final"]))
    c3.metric("Advance (R32)", _pct(p["advance"]))
    c4.metric("Elo", round(info["elo"]) if info and info["elo"] else "?")
    st.caption(f"Group {info['g'] if info else '?'} · FIFA rank {info['fr'] if info else '?'}")

    st.subheader("Availability")
    inj = conn.execute(
        "SELECT player_name, status, source_quote FROM availability_events ae "
        "JOIN teams t ON t.id=ae.team_id WHERE t.name=? AND player_name!='(scan-marker)' "
        "AND status!='fit'", (sel,)).fetchall()
    if inj:
        st.dataframe(pd.DataFrame([{"Player": r["player_name"], "Status": r["status"],
                                    "Note": (r["source_quote"] or "")[:90]} for r in inj]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No concerns recorded.")

    st.subheader("Matches")
    mrows = []
    for m in data.get("matches", []):
        if sel in (m["home"], m["away"]):
            d, t = _ams_parts(m.get("date"))
            mrows.append({"Date": d, "Kickoff (AMS)": t or "—", "Type": f"Group {m['group']}",
                          "Match": f"{m['home']} vs {m['away']}",
                          "Forecast": f"{_pct(m['p_home'])}/{_pct(m['p_draw'])}/{_pct(m['p_away'])}",
                          "Likely": (m.get("result") or m["top_scoreline"])})
    for f in data.get("friendlies", []):
        if sel in (f["home"], f["away"]):
            d, t = _ams_parts(f.get("date"))
            mrows.append({"Date": d, "Kickoff (AMS)": t or "—", "Type": "Friendly",
                          "Match": f"{f['home']} vs {f['away']}",
                          "Forecast": f"{_pct(f['p_home'])}/{_pct(f['p_draw'])}/{_pct(f['p_away'])}",
                          "Likely": (f["result"] if f["played"] else f["top_scoreline"])})
    if mrows:
        st.caption("Kickoff times are Amsterdam local (CEST); matches without a scheduled time show —.")
        st.dataframe(pd.DataFrame(mrows).sort_values(["Date", "Kickoff (AMS)"]), hide_index=True,
                     use_container_width=True)

    st.subheader("Most likely scorers")
    scs = None
    for m in data.get("matches", []):
        if m["home"] == sel:
            scs = m["top_scorers_home"]; break
        if m["away"] == sel:
            scs = m["top_scorers_away"]; break
    if scs:
        st.dataframe(pd.DataFrame([{"Player": s["player"], "P(anytime)": _pct(s["p_anytime"])}
                                   for s in scs]), hide_index=True, use_container_width=True)
    else:
        st.caption("Scorer estimates appear for teams in the group stage.")
