#!/usr/bin/env python3
"""
rank.py  Main ranking script. Must finish < 5 min. CPU only. No network.

Usage:
    python rank.py \
        --candidates /path/to/candidates.jsonl \
        --artifacts artifacts/ \
        --out submission.csv
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
W_BM25  = 0.20   # keyword match  useful but easily gamed
W_FAISS = 0.30   # semantic similarity  stronger signal for technical JD
W_SKILL = 0.20
W_TRAJ  = 0.30   # trajectory  prestige + seniority + production depth

# ─────────────────────────────────────────────────────────────────────────────
# GATE MULTIPLIERS
# ─────────────────────────────────────────────────────────────────────────────
def gate_multiplier(feat: dict, bm25_score_norm: float) -> float:
    if feat.get("is_honeypot"):
        return 0.00
    if feat.get("title_class") == "ANTI_TITLE" and bm25_score_norm < 0.05:
        return 0.05
    if feat.get("all_consulting"):
        return 0.20
    if feat.get("cv_speech_primary"):
        return 0.15
    if feat.get("pure_research"):
        return 0.15
    if feat.get("framework_only"):
        return 0.20
    return 1.00

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifacts", default="artifacts/")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    art = Path(args.artifacts)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("[1/5] Loading artifacts...")
    with open(art / "features.pkl", "rb") as f:
        features: Dict[str, dict] = pickle.load(f)
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25        = bm25_data["bm25"]
    bm25_ids    = bm25_data["ids"]
    with open(art / "faiss_ids.pkl", "rb") as f:
        faiss_ids: List[str] = pickle.load(f)
    with open(art / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    jd_vector   = meta["jd_vector"].astype(np.float32)
    bm25_query  = meta["bm25_query"]

    import faiss
    index = faiss.read_index(str(art / "faiss.index"))
    print(f"  → {index.ntotal} vectors in FAISS, {len(features)} feature records")

    # ── S1: BM25 ─────────────────────────────────────────────────────────────
    print("[2/5] Computing BM25 scores...")
    raw_bm25 = bm25.get_scores(bm25_query)   # np.ndarray len = n_docs
    bm25_scores: Dict[str, float] = {
        cid: float(raw_bm25[i]) for i, cid in enumerate(bm25_ids)
    }
    bm25_norm = minmax_norm(bm25_scores)

    # ── S2: FAISS cosine ─────────────────────────────────────────────────────
    print("[3/5] Computing FAISS cosine scores...")
    jd_vec_row = jd_vector.reshape(1, -1)
    n_total = index.ntotal
    D, I = index.search(jd_vec_row, n_total)
    faiss_scores: Dict[str, float] = {
        faiss_ids[I[0][i]]: float(D[0][i])
        for i in range(n_total)
    }
    faiss_norm = minmax_norm(faiss_scores)

    # ── S3: Skill quality (precomputed) ──────────────────────────────────────
    skill_scores_raw = {cid: feat["skill_quality_score"]
                        for cid, feat in features.items()}
    skill_norm = minmax_norm(skill_scores_raw)

    # ── S4: Trajectory (precomputed) ─────────────────────────────────────────
    traj_scores_raw = {cid: feat["trajectory_score"]
                       for cid, feat in features.items()}
    traj_norm = minmax_norm(traj_scores_raw)

    # ── Combine + gate ────────────────────────────────────────────────────────
    print("[4/5] Combining scores and applying gates...")
    all_ids = list(features.keys())
    final_scores: Dict[str, float] = {}

    for cid in tqdm(all_ids, desc="  scoring", unit="cand"):
        feat = features[cid]

        s1 = bm25_norm.get(cid, 0.0)
        s2 = faiss_norm.get(cid, 0.0)
        s3 = skill_norm.get(cid, 0.0)
        s4 = traj_norm.get(cid, 0.0)

        # Hard YOE gate  JD says 5-9yr, exclude clearly under-qualified
        yoe = feat.get("yoe") or 0.0
        if yoe < 3.5:
            continue

        # YOE soft multiplier  JD says 5-9yr range, ideal 6-8yr
        if yoe >= 5.0:
            yoe_factor = 1.0 if yoe <= 9.0 else max(0.85, 1.0 - (yoe - 9.0) * 0.02)
        else:
            yoe_factor = max(0.40, yoe / 5.0)  # 4yr=0.80, 3yr=0.60, 2yr=0.40

        # ML product years multiplier  JD: "4-5yr in applied ML at product companies"
        # Captures quantity of relevant experience; complements trajectory_score (quality).
        ml_yrs = feat.get("ml_product_years") or 0.0
        ml_yrs_factor = min(1.0, 0.50 + (ml_yrs / 4.0) * 0.50)  # 0yr=0.50 → 4yr+=1.0

        raw = (W_BM25 * s1 + W_FAISS * s2 + W_SKILL * s3 + W_TRAJ * s4) * yoe_factor * ml_yrs_factor

        gate = gate_multiplier(feat, s1)
        gated = raw * gate

        behavioral = feat.get("behavioral_gate", 1.0)
        final = gated * behavioral

        final_scores[cid] = final

    # ── Rank, pick top 100 ────────────────────────────────────────────────────
    print("[5/5] Sorting and writing CSV...")
    sorted_candidates = sorted(final_scores.items(), key=lambda x: -x[1])

    # Exclude hard-zeroed honeypots from top 100
    top_100 = [
        (cid, score) for cid, score in sorted_candidates
        if not features[cid].get("is_honeypot")
    ][:100]

    # Ensure scores are non-increasing (fix floating point ties)
    adjusted_scores = []
    prev = None
    for rank_idx, (cid, score) in enumerate(top_100):
        if prev is not None and score > prev:
            score = prev
        prev = score
        adjusted_scores.append((cid, score))

    # Normalize to 0-100 range for submission
    max_s = adjusted_scores[0][1] if adjusted_scores else 1.0
    min_s = adjusted_scores[-1][1] if adjusted_scores else 0.0
    def rescale(s):
        if max_s - min_s < 1e-9:
            return 50.0
        return 10.0 + 90.0 * (s - min_s) / (max_s - min_s)

    # Load candidates for explain
    print("  Loading candidates for reasoning...")
    top_ids = {c for c, _ in adjusted_scores}   # build once outside the loop
    cand_lookup: Dict[str, dict] = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                cid = cand.get("candidate_id")
                if cid in top_ids:
                    cand_lookup[cid] = cand
            except:
                continue

    # Generate CSV
    from explain import generate_reason
    seen_descs: set = set()
    out_path = args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_num, (cid, raw_score) in enumerate(adjusted_scores, start=1):
            score_out = round(rescale(raw_score), 4)
            cand = cand_lookup.get(cid, {})
            reason = generate_reason(cand, features.get(cid, {}), rank_num, score_out, seen_descs)
            writer.writerow([cid, rank_num, score_out, reason])

    print(f"\n✓ Submission written to: {out_path}")
    print(f"  Rank 1: {adjusted_scores[0][0]}  score={rescale(adjusted_scores[0][1]):.2f}")
    print(f"  Rank 100: {adjusted_scores[-1][0]}  score={rescale(adjusted_scores[-1][1]):.2f}")


if __name__ == "__main__":
    main()
