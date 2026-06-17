"""
app.py Indiaruns AI HuggingFace Spaces demo.
Streamlit interactive leaderboard for Redrob India Data & AI Challenge.

Run locally:   streamlit run app.py
Deploy:        Push to HuggingFace Space (SDK: Streamlit)
"""

import pickle
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from rank import gate_multiplier

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indiaruns AI Redrob Ranker",
    page_icon="🏆",
    layout="wide",
)

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_full_artifacts():
    """Load BM25 + FAISS + features for real scoring in demo tab."""
    import faiss as faiss_lib
    art = Path("artifacts")
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    with open(art / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    index = faiss_lib.read_index(str(art / "faiss.index"))
    with open(art / "faiss_ids.pkl", "rb") as f:
        faiss_ids = pickle.load(f)
    with open(art / "features.pkl", "rb") as f:
        feats = pickle.load(f)
    return bm25_data, meta, index, faiss_ids, feats

@st.cache_data
def load_data():
    df = pd.read_csv("submission.csv")
    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)
    return df

df = load_data()
bm25_data, meta, faiss_index, faiss_ids, features = load_full_artifacts()

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏆 Indiaruns AI Redrob AI Ranker")
st.caption(
    "Redrob India Data & AI Challenge · Track 1 · Team: **Indiaruns AI** · "
    "100,000 candidates → top 100 · CPU-only · no LLM calls"
)
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_board, tab_demo, tab_about = st.tabs(["🏆 Leaderboard", "▶ Live Demo", "⚙️ How It Works"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  LEADERBOARD
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
    st.bar_chart(df.set_index("rank")["score"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("▶ Live Demo  Run Ranker on Top 20 Candidates")
    st.info(
        "Runs the **full ranking pipeline** (BM25 + FAISS + Skill + Trajectory) "
        "on the top 20 candidates from our submission. "
        "All 4 signals active  identical to the production rank.py logic."
    )

    if st.button("▶ Run Demo Ranking", type="primary"):
        with st.spinner("Running full ranking pipeline on top 20 candidates..."):
            import numpy as np
            from rank import minmax_norm

            top20_ids = set(df.head(20)["candidate_id"].tolist())

            # S1: Real BM25 scores
            bm25       = bm25_data["bm25"]
            bm25_ids   = bm25_data["ids"]
            raw_bm25   = bm25.get_scores(meta["bm25_query"])
            bm25_scores = {cid: float(raw_bm25[i]) for i, cid in enumerate(bm25_ids)}
            bm25_norm  = minmax_norm(bm25_scores)

            # S2: Real FAISS cosine scores
            jd_vec     = meta["jd_vector"].astype(np.float32).reshape(1, -1)
            D, I       = faiss_index.search(jd_vec, faiss_index.ntotal)
            faiss_scores = {faiss_ids[I[0][i]]: float(D[0][i]) for i in range(faiss_index.ntotal)}
            faiss_norm = minmax_norm(faiss_scores)

            # S3 + S4: precomputed
            skill_scores = {cid: f["skill_quality_score"] for cid, f in features.items()}
            traj_scores  = {cid: f["trajectory_score"]    for cid, f in features.items()}
            skill_norm   = minmax_norm(skill_scores)
            traj_norm    = minmax_norm(traj_scores)

            W_BM25, W_FAISS, W_SKILL, W_TRAJ = 0.20, 0.30, 0.20, 0.30

            results = []
            for cid in top20_ids:
                feat = features.get(cid, {})
                if not feat: continue
                yoe = feat.get("yoe") or 0.0
                if yoe < 3.5: continue

                s1 = bm25_norm.get(cid, 0.0)
                s2 = faiss_norm.get(cid, 0.0)
                s3 = skill_norm.get(cid, 0.0)
                s4 = traj_norm.get(cid, 0.0)

                gate = gate_multiplier(feat, s1)
                yoe_factor = 1.0 if 5.0 <= yoe <= 9.0 else (
                    max(0.85, 1.0 - (yoe - 9.0) * 0.02) if yoe > 9.0
                    else max(0.40, yoe / 5.0)
                )
                ml_yrs = feat.get("ml_product_years") or 0.0
                ml_yrs_factor = min(1.0, 0.50 + (ml_yrs / 4.0) * 0.50)
                behavioral = feat.get("behavioral_gate", 1.0)

                raw   = (W_BM25*s1 + W_FAISS*s2 + W_SKILL*s3 + W_TRAJ*s4) * yoe_factor * ml_yrs_factor
                final = raw * gate * behavioral

                reasoning_vals = df.loc[df["candidate_id"] == cid, "reasoning"].values
                reasoning_str  = reasoning_vals[0] if len(reasoning_vals) > 0 else ""

                results.append({
                    "candidate_id": cid,
                    "_score": final,
                    "yoe":   round(yoe, 1),
                    "ml_yrs": round(ml_yrs, 1),
                    "skill": round(feat.get("skill_quality_score", 0), 1),
                    "traj":  round(feat.get("trajectory_score", 0), 1),
                    "gate":  gate,
                    "behavioral": round(behavioral, 2),
                    "reasoning": reasoning_str,
                })

            results.sort(key=lambda x: -x["_score"])
            max_s = results[0]["_score"] if results else 1.0
            min_s = results[-1]["_score"] if results else 0.0
            for i, r in enumerate(results):
                r["rank"]  = i + 1
                r["score"] = round(10 + 90 * (r["_score"] - min_s) / (max_s - min_s + 1e-9), 2)
            demo_df = pd.DataFrame(results)[["rank", "candidate_id", "yoe", "ml_yrs", "skill", "traj", "gate", "behavioral", "score"]]
            st.dataframe(demo_df, use_container_width=True, hide_index=True)

            st.subheader("Reasoning")
            rows_html = []
            for r in results:
                rows_html.append(f"<tr style='border-bottom:1px solid #444'><td style='padding:8px;text-align:center;font-weight:bold'>{r['rank']}</td><td style='padding:8px;font-family:monospace'>{r['candidate_id']}</td><td style='padding:8px;text-align:right'>{r['score']}</td><td style='padding:8px;min-width:500px;white-space:normal;word-break:break-word;line-height:1.5'>{r['reasoning']}</td></tr>")
            st.markdown(f"""<div style='overflow-x:auto;overflow-y:auto;max-height:500px;border:1px solid #444;border-radius:6px'><table style='width:100%;border-collapse:collapse;font-size:13px'><thead><tr style='background:#1e3a5f;position:sticky;top:0'><th style='padding:10px;color:#fff'>Rank</th><th style='padding:10px;color:#fff;text-align:left'>Candidate</th><th style='padding:10px;color:#fff'>Score</th><th style='padding:10px;color:#fff;text-align:left;min-width:500px'>Reasoning</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>""", unsafe_allow_html=True)
            csv_out = "candidate_id,rank,score\n" + "\n".join(f"{r['candidate_id']},{r['rank']},{r['score']}" for r in results)
            st.download_button("⬇️ Download Demo CSV", csv_out.encode(), "demo_ranking.csv", "text/csv")
            st.success(f"✅ Ranked {len(results)} candidates in < 1 second")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Pipeline Overview")

    st.markdown("""
### Two-stage CPU-only pipeline. No LLM API calls. No external services.

---

#### Stage 1  `precompute.py` (run once, ~1.5 hr, 100K candidates)

| Step | What happens |
|------|-------------|
| JD encoding | `BAAI/bge-small-en-v1.5` embeds the job description |
| Candidate encoding | Same model encodes each candidate's career text → FAISS `IndexFlatIP` |
| BM25 index | `BM25Okapi` index over career descriptions |
| Feature extraction | Per-candidate: skill quality, trajectory score, behavioral gate, boolean gates |

---

#### Stage 2  `rank.py` (< 5 min)

Weighted combination of 4 signals:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| **BM25** | 0.20 | Keyword match  career text vs JD tokens |
| **FAISS cosine** | 0.30 | Semantic similarity  candidate embedding vs JD embedding |
| **Skill quality** | 0.20 | Relevance × verified proficiency × tenure duration |
| **Trajectory** | 0.30 | Production evidence + pre-LLM depth + still-coding signal |

Then multiplied by:
- **YOE fit**  soft penalty below 5yr, soft penalty above 9yr  
- **ML product years fit**  rewards "4yr applied ML at product companies" (JD spec)  
- **Hard gates**  honeypot ×0.0, all-consulting ×0.20, cv-speech ×0.15  
- **Behavioral multiplier**  activity, notice period, location, github (range 0.50–1.20)  

---

#### Models used

| Model | Source | Usage |
|-------|--------|-------|
| `BAAI/bge-small-en-v1.5` | HuggingFace (sentence-transformers) | Text embedding for FAISS + skill matching |

No OpenAI · No Anthropic · No cloud inference · Runs on CPU
    """)

    st.info(
        "📦 **Artifacts bundled in this repo:** `submission.csv`, `artifacts/features.pkl`, "
        "`artifacts/meta.pkl`  \n"
        "🚫 **Not bundled (too large):** `artifacts/faiss.index` (146MB), "
        "`artifacts/bm25_index.pkl` (167MB)"
    )
