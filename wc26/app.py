"""Streamlit dashboard (read-only). Renders the latest forecast snapshot from the DB.

Run:  streamlit run wc26/app.py
All computation happens in the pipeline; this only displays precomputed results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from wc26 import config, db

st.set_page_config(page_title="World Cup 2026 Forecast", layout="wide")


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
    payload["_run_id"] = row["run_id"]
    return payload


data = load_latest()
st.title("World Cup 2026 Forecast")

if data is None:
    st.warning("No forecast yet. Run:  python -m wc26.forecast")
    st.stop()

as_of = data.get("data_as_of", "unknown")
stale = False
try:
    age_h = (datetime.now(timezone.utc).date() - datetime.fromisoformat(as_of).date()).days * 24
    stale = age_h > config.STALE_HOURS
except Exception:
    pass
banner = f"Data as of **{as_of}** | {data['n_runs']} sims | seed {data['seed']} | computed {data['_computed_at']}"
(st.error if stale else st.info)(("STALE — " if stale else "") + banner)
st.caption("These are probabilities, not certainties. The single most likely scoreline is usually only ~10% likely, and the top scorer misses about half the time.")

probs = data["probs"]
tab_f, tab_m, tab_g, tab_s, tab_c = st.tabs(
    ["Champion & stages", "Matches", "Groups", "Scorers", "Calibration"])

with tab_f:
    rows = [{"Team": n, "Win %": round(p["champion"] * 100, 1), "Final %": round(p["final"] * 100, 1),
             "SF %": round(p["sf"] * 100, 1), "QF %": round(p["qf"] * 100, 1),
             "R16 %": round(p["r16"] * 100, 1), "Advance %": round(p["advance"] * 100, 1)}
            for n, p in probs.items()]
    df = pd.DataFrame(rows).sort_values("Win %", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, height=600)

with tab_m:
    mrows = []
    for m in data["matches"]:
        sc = ", ".join(f"{s['player']} {round(s['p_anytime']*100)}%"
                       for s in m["top_scorers_home"][:2] + m["top_scorers_away"][:2])
        mrows.append({"Grp": m["group"], "Home": m["home"], "Away": m["away"],
                      "Home %": round(m["p_home"] * 100), "Draw %": round(m["p_draw"] * 100),
                      "Away %": round(m["p_away"] * 100), "Likely score": m["top_scoreline"],
                      "Top scorers": sc})
    st.dataframe(pd.DataFrame(mrows), use_container_width=True, height=600)

with tab_g:
    groups: dict[str, list] = {}
    conn = db.connect()
    for r in conn.execute("SELECT name, group_letter FROM teams"):
        groups.setdefault(r["group_letter"], []).append(r["name"])
    cols = st.columns(3)
    for i, letter in enumerate(sorted(groups)):
        teams = sorted(groups[letter], key=lambda t: probs[t]["advance"], reverse=True)
        with cols[i % 3]:
            st.subheader(f"Group {letter}")
            st.dataframe(pd.DataFrame([
                {"Team": t, "Win grp %": round(probs[t]["win_group"] * 100),
                 "Advance %": round(probs[t]["advance"] * 100)} for t in teams
            ]), hide_index=True, use_container_width=True)

with tab_s:
    st.subheader("Most likely scorers (expected group-stage goals)")
    st.dataframe(pd.DataFrame(data.get("golden_boot", [])), hide_index=True, use_container_width=True)

with tab_c:
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
        c1.metric("Reliability slope (1.0 = perfect calibration)", r["reliability"]["home_win_slope"])
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
