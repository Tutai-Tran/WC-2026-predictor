"""Streamlit dashboard (read-only). Renders the latest forecast snapshot from the DB.

Run:  streamlit run wc26/app.py
All computation happens in the pipeline; this only displays precomputed results.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# `streamlit run wc26/app.py` puts wc26/ on sys.path, not the repo root, so the
# package import fails. Add the repo root so `import wc26` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from wc26 import config, db

st.set_page_config(page_title="World Cup 2026 Forecast", page_icon="⚽", layout="wide")


@st.cache_data(ttl=60)
def load_latest():
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


data = load_latest()
st.title("⚽ World Cup 2026 Forecast")

if data is None:
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
(st.error if stale else st.success)(("⚠️ STALE — " if stale else "✅ ") + banner)
st.caption("These are probabilities, not certainties. A single most-likely scoreline is usually "
           "only ~10% likely, and even the top scorer pick misses about half the time.")

probs = data["probs"]
champ_sorted = sorted(probs.items(), key=lambda kv: kv[1]["champion"], reverse=True)

# ---- headline favourites ----
cols = st.columns(4)
for col, (name, p) in zip(cols, champ_sorted[:4]):
    col.metric(name, _pct(p["champion"]), help="Probability of winning the tournament")

tabs = st.tabs(["📅 Daily", "🏆 Champion & stages", "⚽ Matches", "🔀 Bracket",
                "👥 Groups", "🥅 Scorers", "📈 Calibration"])

# =============================== Daily ===============================
with tabs[0]:
    st.subheader("Matches by day")
    rows = []
    for m in data.get("matches", []):
        rows.append({"date": _date(m.get("date")), "type": f"Group {m['group']}",
                     "match": f"{m['home']} vs {m['away']}",
                     "forecast": f"{_pct(m['p_home'])} / {_pct(m['p_draw'])} / {_pct(m['p_away'])}",
                     "likely": m["top_scoreline"], "status": "scheduled"})
    for f in data.get("friendlies", []):
        rows.append({"date": _date(f.get("date")), "type": "Friendly",
                     "match": f"{f['home']} vs {f['away']}",
                     "forecast": f"{_pct(f['p_home'])} / {_pct(f['p_draw'])} / {_pct(f['p_away'])}",
                     "likely": f["top_scoreline"],
                     "status": (f"played {f['result']}" if f["played"] else "scheduled")})
    for k in data.get("knockout", []):
        if not k.get("date"):
            continue
        home = k.get("home_proj") or k["home_slot"]
        away = k.get("away_proj") or k["away_slot"]
        rows.append({"date": _date(k["date"]), "type": k["stage"],
                     "match": f"{home} vs {away}", "forecast": "(projected teams)",
                     "likely": "-", "status": k.get("result") or "scheduled"})
    df = pd.DataFrame(rows).sort_values(["date", "type"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    today = datetime.now(timezone.utc).date().isoformat()
    upcoming = [d for d in dates if d >= today]
    default = upcoming[0] if upcoming else (dates[-1] if dates else None)
    pick = st.select_slider("Pick a date", options=dates,
                            value=default) if dates else None
    if pick:
        day = df[df["date"] == pick]
        st.markdown(f"### {pick} — {len(day)} match(es)")
        st.caption("Forecast column is Win / Draw / Win (home/away).")
        st.dataframe(day.drop(columns=["date"]), hide_index=True, use_container_width=True)
    with st.expander("See the full schedule (all days)"):
        st.dataframe(df, hide_index=True, use_container_width=True, height=400)

# ======================== Champion & stages =========================
with tabs[1]:
    blend = {}
    try:
        from wc26 import odds
        blend = odds.blended_champion(db.connect())
    except Exception:
        blend = {}
    rows = []
    for n, p in probs.items():
        row = {"Team": n, "Win %": round(p["champion"]*100, 1)}
        if blend:
            row["Win % (model+market)"] = round(blend.get(n, p["champion"])*100, 1)
        row.update({"Final %": round(p["final"]*100, 1), "SF %": round(p["sf"]*100, 1),
                    "QF %": round(p["qf"]*100, 1), "R16 %": round(p["r16"]*100, 1),
                    "Advance %": round(p["advance"]*100, 1)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Win %", ascending=False).reset_index(drop=True)
    if blend:
        st.caption("'Win % (model+market)' blends the model with devigged market odds "
                   "(usually better calibrated). Plain 'Win %' is the pure model.")
    st.dataframe(df, use_container_width=True, height=600)

# ============================== Matches =============================
with tabs[2]:
    kind = st.radio("Competition", ["Group stage", "Friendlies (warm-up)", "Knockouts"],
                    horizontal=True)
    if kind == "Group stage":
        mrows = []
        for m in data.get("matches", []):
            sc = ", ".join(f"{s['player']} {_pct(s['p_anytime'])}"
                           for s in (m["top_scorers_home"][:2] + m["top_scorers_away"][:2]))
            mrows.append({"Date": _date(m.get("date")), "Grp": m["group"],
                          "Home": m["home"], "Away": m["away"],
                          "H%": round(m["p_home"]*100), "D%": round(m["p_draw"]*100),
                          "A%": round(m["p_away"]*100), "Likely": m["top_scoreline"],
                          "Top scorers": sc})
        st.dataframe(pd.DataFrame(mrows).sort_values("Date"), hide_index=True,
                     use_container_width=True, height=560)
    elif kind == "Friendlies (warm-up)":
        frows = [{"Date": _date(f.get("date")), "Home": f["home"], "Away": f["away"],
                  "H%": round(f["p_home"]*100), "D%": round(f["p_draw"]*100),
                  "A%": round(f["p_away"]*100), "Likely": f["top_scoreline"],
                  "Status": (f"played {f['result']}" if f["played"] else "scheduled")}
                 for f in data.get("friendlies", [])]
        st.caption("Warm-up friendlies in the run-up to the World Cup. Played results also "
                   "update the model's Elo ratings.")
        st.dataframe(pd.DataFrame(frows).sort_values("Date"), hide_index=True,
                     use_container_width=True, height=560)
    else:
        krows = [{"#": k["match_no"], "Stage": k["stage"], "Date": _date(k.get("date")),
                  "Home slot": k["home_slot"], "Away slot": k["away_slot"],
                  "Projected": f"{k.get('home_proj') or '?'} vs {k.get('away_proj') or '?'}",
                  "Result": k.get("result") or "-"}
                 for k in data.get("knockout", [])]
        st.caption("Knockout fixtures by slot. Teams are TBD until the group stage finishes; "
                   "the 'Projected' column shows the current most-likely qualifiers.")
        st.dataframe(pd.DataFrame(krows), hide_index=True, use_container_width=True, height=560)

# ============================== Bracket =============================
with tabs[3]:
    st.subheader("Knockout bracket — who reaches the later rounds")
    st.caption("The bracket fills in with real teams as the group stage finishes and with "
               "winners as knockout matches are played. For now it shows the structure and the "
               "current most-likely qualifier for each slot.")
    ko = {k["match_no"]: k for k in data.get("knockout", [])}

    def line(mno):
        k = ko.get(mno)
        if not k:
            return f"Match {mno}: TBD"
        h = k.get("home_proj") or k["home_slot"]
        a = k.get("away_proj") or k["away_slot"]
        res = f"  →  {k['result']}" if k.get("result") else ""
        return f"**{k['stage']}** (#{mno}, {_date(k.get('date'))}): {h}  vs  {a}{res}"

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

# ============================ Calibration ===========================
with tabs[6]:
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
        c1, c2 = st.columns(2)
        c1.metric("Reliability slope (1.0 = perfect)", r["reliability"]["home_win_slope"])
        c2.metric("Model log loss (95% CI)", f"{r['test']['log_loss']} {r['test']['log_loss_ci95']}")
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
