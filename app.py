"""
app.py Indiaruns AI HuggingFace Spaces demo.
Streamlit interactive leaderboard for Redrob India Data & AI Challenge.

Run locally:   streamlit run app.py
Deploy:        Push to HuggingFace Space (SDK: Streamlit)
"""

import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndiaRuns AI Redrob AI Ranker",
    page_icon="🏆",
    layout="wide",
)

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    art = Path("artifacts")
    df = pd.read_csv("submission.csv")
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)

    with open(art / "features.pkl", "rb") as f:
        features: dict = pickle.load(f)

    return df, features

df, features = load_data()

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏆 Indiaruns AI Redrob AI Ranker")
st.caption(
    "Redrob India Data & AI Challenge · Track 1 · Team: **Indiaruns AI** · "
    "100,000 candidates → top 100 · CPU-only · no LLM calls"
)
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_board, tab_inspect, tab_demo, tab_about = st.tabs(["🏆 Leaderboard", "🔍 Candidate Inspector", "▶ Live Demo", "⚙️ How It Works"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_board:
    st.subheader("Top 100 Candidates")

    col_filter, col_info = st.columns([3, 1])
    with col_filter:
        search = st.text_input("🔎 Filter by candidate ID or keyword in reasoning", "")
    with col_info:
        st.metric("Total candidates ranked", "100,000")
        st.metric("Output size", "100")

    # CSV download
    st.download_button(
        label="⬇️ Download submission.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

    display_df = df.copy()
    if search.strip():
        mask = (
            display_df["candidate_id"].str.contains(search, case=False, na=False)
            | display_df["reasoning"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    # HTML table with horizontal + vertical scroll, full reasoning visible
    rows_list = []
    for _, row in display_df.iterrows():
        rows_list.append(f"""<tr style='border-bottom:1px solid #eee'>
            <td style='padding:8px 12px;text-align:center;font-weight:bold;white-space:nowrap'>{int(row['rank'])}</td>
            <td style='padding:8px 12px;white-space:nowrap;font-family:monospace'>{row['candidate_id']}</td>
            <td style='padding:8px 12px;text-align:right;white-space:nowrap'>{float(row['score']):.2f}</td>
            <td style='padding:8px 12px;min-width:500px;white-space:normal;word-break:break-word;line-height:1.5'>{row['reasoning']}</td>
        </tr>""")

    final_html = f"""
    <div style='overflow-x:auto;overflow-y:auto;max-height:600px;border:1px solid #444;border-radius:6px'>
    <table style='width:100%;border-collapse:collapse;font-size:13px'>
        <thead>
            <tr style='background:#1e3a5f;position:sticky;top:0;z-index:1'>
                <th style='padding:10px 12px;text-align:center;border-bottom:2px solid #555;white-space:nowrap;color:#ffffff;font-weight:700'>Rank</th>
                <th style='padding:10px 12px;text-align:left;border-bottom:2px solid #555;white-space:nowrap;color:#ffffff;font-weight:700'>Candidate ID</th>
                <th style='padding:10px 12px;text-align:right;border-bottom:2px solid #555;white-space:nowrap;color:#ffffff;font-weight:700'>Score</th>
                <th style='padding:10px 12px;text-align:left;border-bottom:2px solid #555;min-width:500px;color:#ffffff;font-weight:700'>Reasoning</th>
            </tr>
        </thead>
        <tbody>{''.join(rows_list)}</tbody>
    </table>
    </div>"""

    st.markdown(final_html, unsafe_allow_html=True)

    # Full reasoning viewer
    st.subheader("Full Reasoning")
    selected = st.selectbox(
        "Select candidate",
        options=display_df["candidate_id"].tolist(),
        format_func=lambda cid: f"#{df.loc[df.candidate_id==cid,'rank'].values[0]}  {cid}",
        key="board_select",
    )
    if selected:
        rank_val  = df.loc[df["candidate_id"] == selected, "rank"].values[0]
        score_val = df.loc[df["candidate_id"] == selected, "score"].values[0]
        full      = df.loc[df["candidate_id"] == selected, "reasoning"].values[0]
        st.markdown(f"**Rank {rank_val}** · Score `{score_val:.2f}`")
        st.text_area("Reasoning", value=full, height=200, disabled=True, label_visibility="collapsed")

    # Score distribution chart
    st.subheader("Score Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=df["rank"],
        y=df["score"],
        marker_color="steelblue",
        hovertemplate="Rank %{x}<br>Score %{y:.2f}<br>%{customdata}",
        customdata=df["candidate_id"],
    ))
    fig_dist.update_layout(
        xaxis_title="Rank",
        yaxis_title="Score",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CANDIDATE INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    st.subheader("Inspect a Candidate")

    all_ids = df.sort_values("rank")["candidate_id"].tolist()
    selected_id = st.selectbox(
        "Select candidate (sorted by rank)",
        options=all_ids,
        format_func=lambda cid: f"{df.loc[df.candidate_id==cid,'rank'].values[0]:>3}. {cid}",
    )

    if selected_id:
        row = df[df["candidate_id"] == selected_id].iloc[0]
        feat = features.get(selected_id, {})

        # ── Summary metrics ────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank", int(row["rank"]))
        c2.metric("Score", f"{row['score']:.2f}")
        c3.metric("YOE", f"{feat.get('yoe') or 0.0:.1f} yr")
        c4.metric("ML Product Years", f"{feat.get('ml_product_years') or 0.0:.1f} yr")

        # ── Reasoning ─────────────────────────────────────────────────────
        with st.expander("📝 Full Reasoning", expanded=True):
            st.write(row["reasoning"])

        # ── Signal breakdown chart ─────────────────────────────────────────
        st.subheader("Signal Breakdown")

        raw_skill  = feat.get("skill_quality_score", 0.0)   # 0–100 raw
        raw_traj   = feat.get("trajectory_score", 0.0)       # 0–100 raw
        behavioral = feat.get("behavioral_gate", 1.0)        # 0.5–1.2 multiplier

        yoe_val = feat.get("yoe") or 0.0
        if yoe_val >= 5.0:
            yoe_pct = 100.0 if yoe_val <= 9.0 else max(85.0, 100.0 - (yoe_val - 9.0) * 2.0)
        else:
            yoe_pct = max(40.0, yoe_val / 5.0 * 100.0)

        ml_yrs = feat.get("ml_product_years") or 0.0
        ml_pct = min(100.0, (0.50 + (ml_yrs / 4.0) * 0.50) * 100.0)

        signals = {
            "Skill Quality": raw_skill,
            "Trajectory":    raw_traj,
            "YOE Fit":       yoe_pct,
            "ML Prod Yrs":   ml_pct,
            "Behavioral Gate (×)": (behavioral - 0.5) / 0.7 * 100.0,  # normalise to 0–100
        }

        fig_bar = go.Figure(go.Bar(
            x=list(signals.values()),
            y=list(signals.keys()),
            orientation="h",
            marker_color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"],
            text=[f"{v:.1f}" for v in signals.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 110], title="Score (normalised 0–100)"),
            height=280,
            margin=dict(l=140, r=60, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Gate flags ────────────────────────────────────────────────────
        st.subheader("Gate Flags")
        flags = {
            "is_honeypot":      feat.get("is_honeypot", False),
            "all_consulting":   feat.get("all_consulting", False),
            "cv_speech_primary":feat.get("cv_speech_primary", False),
            "pure_research":    feat.get("pure_research", False),
            "framework_only":   feat.get("framework_only", False),
            "title_chaser":     feat.get("title_chaser", False),
            "startup_experience":feat.get("startup_experience", False),
        }
        flag_cols = st.columns(len(flags))
        for col, (name, val) in zip(flag_cols, flags.items()):
            emoji = "🔴" if val and name not in ("startup_experience",) else ("🟢" if val else "⚪")
            col.metric(name, emoji)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    import json

    st.subheader("▶ Live Demo — Run Ranker on Sample Candidates")
    st.info(
        "Ranks the 20 bundled sample candidates end-to-end using pre-computed "
        "skill quality and trajectory scores from `features.pkl`. "
        "BM25 and FAISS signals require 300MB+ indexes not bundled here; "
        "the full pipeline uses all 4 signals."
    )

    SAMPLE_PATH = Path("sample_candidates.json")

    if not SAMPLE_PATH.exists():
        st.error("sample_candidates.json not found. Make sure it's in the repo root.")
    else:
        if st.button("▶ Run Demo Ranking", type="primary"):
            with st.spinner("Ranking 20 sample candidates..."):
                with open(SAMPLE_PATH) as f:
                    sample_cands = json.load(f)

                results = []
                for cand in sample_cands:
                    cid  = cand.get("candidate_id")
                    feat = features.get(cid, {})

                    if feat.get("is_honeypot"):
                        continue

                    yoe = feat.get("yoe") or 0.0
                    if yoe < 3.5:
                        continue

                    # Hard gate
                    if feat.get("all_consulting"):   gate = 0.20
                    elif feat.get("cv_speech_primary"): gate = 0.15
                    elif feat.get("pure_research"):  gate = 0.15
                    elif feat.get("framework_only"): gate = 0.20
                    else:                            gate = 1.00

                    # YOE factor
                    if yoe >= 5.0:
                        yoe_factor = 1.0 if yoe <= 9.0 else max(0.85, 1.0 - (yoe - 9.0) * 0.02)
                    else:
                        yoe_factor = max(0.40, yoe / 5.0)

                    # ML years factor
                    ml_yrs = feat.get("ml_product_years") or 0.0
                    ml_yrs_factor = min(1.0, 0.50 + (ml_yrs / 4.0) * 0.50)

                    # S3 + S4 (BM25 + FAISS excluded — indexes not bundled)
                    s3 = feat.get("skill_quality_score", 0) / 100.0
                    s4 = feat.get("trajectory_score", 0) / 100.0
                    raw = (0.40 * s3 + 0.60 * s4) * yoe_factor * ml_yrs_factor

                    behavioral = feat.get("behavioral_gate", 1.0)
                    final = raw * gate * behavioral

                    profile = cand.get("profile") or {}
                    results.append({
                        "candidate_id": cid,
                        "_score": final,
                        "title":  profile.get("current_title") or "Engineer",
                        "yoe":    round(yoe, 1),
                        "ml_yrs": round(ml_yrs, 1),
                        "skill":  round(feat.get("skill_quality_score", 0), 1),
                        "traj":   round(feat.get("trajectory_score", 0), 1),
                        "gate":   gate,
                        "behavioral": round(behavioral, 2),
                    })

                results.sort(key=lambda x: -x["_score"])

                # Rescale 10-100
                max_s = results[0]["_score"]  if results else 1.0
                min_s = results[-1]["_score"] if results else 0.0
                for i, r in enumerate(results):
                    r["rank"] = i + 1
                    r["score"] = round(
                        10 + 90 * (r["_score"] - min_s) / (max_s - min_s + 1e-9), 2
                    )

                # Display table
                display_cols = ["rank", "candidate_id", "title", "yoe", "ml_yrs",
                                "skill", "traj", "gate", "behavioral", "score"]
                demo_df = pd.DataFrame(results)[display_cols]
                st.dataframe(demo_df, use_container_width=True, hide_index=True)

                # CSV download
                csv_rows = ["candidate_id,rank,score"]
                for r in results:
                    csv_rows.append(f"{r['candidate_id']},{r['rank']},{r['score']}")
                st.download_button(
                    "⬇️ Download Demo CSV",
                    "\n".join(csv_rows).encode("utf-8"),
                    file_name="demo_ranking.csv",
                    mime="text/csv",
                )

                st.success(f"✅ Ranked {len(results)} candidates in < 1 second")
                st.caption(
                    "Full pipeline (precompute.py + rank.py) on 100K candidates: ~1.7 hours precompute, "
                    "< 2 minutes rank. See README for reproduce commands."
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Pipeline Overview")

    st.markdown("""
### Two-stage CPU-only pipeline. No LLM API calls. No external services.

---

#### Stage 1 — `precompute.py` (run once, ~1.5 hr, 100K candidates)

| Step | What happens |
|------|-------------|
| JD encoding | `all-MiniLM-L6-v2` embeds the job description |
| Candidate encoding | Same model encodes each candidate's career text → FAISS `IndexFlatIP` |
| BM25 index | `BM25Okapi` index over career descriptions |
| Feature extraction | Per-candidate: skill quality, trajectory score, behavioral gate, boolean gates |

---

#### Stage 2 — `rank.py` (< 5 min)

Weighted combination of 4 signals:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| **BM25** | 0.20 | Keyword match — career text vs JD tokens |
| **FAISS cosine** | 0.30 | Semantic similarity — candidate embedding vs JD embedding |
| **Skill quality** | 0.20 | Relevance × verified proficiency × tenure duration |
| **Trajectory** | 0.30 | Production evidence + pre-LLM depth + still-coding signal |

Then multiplied by:
- **YOE fit** — soft penalty below 5yr, soft penalty above 9yr  
- **ML product years fit** — rewards "4yr applied ML at product companies" (JD spec)  
- **Hard gates** — honeypot ×0.0, all-consulting ×0.20, cv-speech ×0.15  
- **Behavioral multiplier** — activity, notice period, location, github (range 0.50–1.20)  

---

#### Models used

| Model | Source | Usage |
|-------|--------|-------|
| `all-MiniLM-L6-v2` | HuggingFace (sentence-transformers) | Text embedding for FAISS + skill matching |

No OpenAI · No Anthropic · No cloud inference · Runs on CPU
    """)

    st.info(
        "📦 **Artifacts bundled in this repo:** `submission.csv`, `artifacts/features.pkl`, "
        "`artifacts/meta.pkl`  \n"
        "🚫 **Not bundled (too large):** `artifacts/faiss.index` (146MB), "
        "`artifacts/bm25_index.pkl` (167MB)"
    )
