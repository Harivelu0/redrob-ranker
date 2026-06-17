#!/usr/bin/env python3
"""
weight_experiment.py  Tries 4 weight combinations, shows top 10 for each.
No files modified. Pure comparison.

Run: python weight_experiment.py --artifacts artifacts/ --candidates path/to/candidates.jsonl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rank import minmax_norm


EXPERIMENTS = [
    # label,           W_BM25, W_FAISS, W_SKILL, W_TRAJ
    ("Current",        0.35,   0.25,    0.20,    0.20),
    ("Boost FAISS",    0.30,   0.35,    0.20,    0.15),
    ("Boost Skill",    0.30,   0.25,    0.30,    0.15),
    ("Boost Traj",     0.30,   0.25,    0.15,    0.30),
    ("Equal",          0.25,   0.25,    0.25,    0.25),
]


def gate_mult(feat: dict, s1: float) -> float:
    if feat.get("is_honeypot"):              return 0.00
    if feat.get("title_class") == "ANTI_TITLE" and s1 < 0.05: return 0.05
    if feat.get("all_consulting"):           return 0.20
    if feat.get("cv_speech_primary"):        return 0.15
    if feat.get("pure_research"):            return 0.15
    if feat.get("framework_only"):           return 0.20
    return 1.00


def yoe_factor(yoe: float) -> float:
    if yoe >= 5.0:
        return 1.0 if yoe <= 9.0 else max(0.85, 1.0 - (yoe - 9.0) * 0.02)
    return max(0.40, yoe / 5.0)


def rank_with_weights(features, bm25_norm, faiss_norm, skill_norm, traj_norm,
                      w1, w2, w3, w4) -> List[str]:
    scores = {}
    for cid, feat in features.items():
        s1 = bm25_norm.get(cid, 0.0)
        s2 = faiss_norm.get(cid, 0.0)
        s3 = skill_norm.get(cid, 0.0)
        s4 = traj_norm.get(cid, 0.0)
        yf = yoe_factor(feat.get("yoe") or 0.0)
        raw = (w1*s1 + w2*s2 + w3*s3 + w4*s4) * yf
        gate = gate_mult(feat, s1) * feat.get("behavioral_gate", 1.0)
        scores[cid] = raw * gate
    return sorted(scores, key=lambda c: -scores[c])


def signal_agreement(cid, bm25_norm, faiss_norm, skill_norm, traj_norm) -> int:
    vals = [bm25_norm.get(cid,0), faiss_norm.get(cid,0),
            skill_norm.get(cid,0), traj_norm.get(cid,0)]
    return sum(1 for v in vals if v > 0.70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/")
    parser.add_argument("--candidates", required=True)
    args, _ = parser.parse_known_args()
    art = Path(args.artifacts)

    print("Loading artifacts...")
    with open(art / "features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(art / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    with open(art / "faiss_ids.pkl", "rb") as f:
        faiss_ids = pickle.load(f)
    with open(art / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    import faiss
    index = faiss.read_index(str(art / "faiss.index"))
    bm25 = bm25_data["bm25"]
    bm25_ids = bm25_data["ids"]
    jd_vector = meta["jd_vector"].astype(np.float32)
    bm25_query = meta["bm25_query"]

    print("Computing base signals...")
    bm25_raw = bm25.get_scores(bm25_query)
    bm25_scores_all = {cid: float(bm25_raw[i]) for i, cid in enumerate(bm25_ids)}
    bm25_norm = minmax_norm(bm25_scores_all)

    n = index.ntotal
    D, I = index.search(jd_vector.reshape(1, -1), n)
    faiss_scores_all = {faiss_ids[I[0][i]]: float(D[0][i]) for i in range(n)}
    faiss_norm = minmax_norm(faiss_scores_all)

    skill_raw = {cid: f["skill_quality_score"] for cid, f in features.items()}
    skill_norm = minmax_norm(skill_raw)

    traj_raw = {cid: f["trajectory_score"] for cid, f in features.items()}
    traj_norm = minmax_norm(traj_raw)

    # Load candidate names for display
    print("Loading candidate names...")
    cand_titles: Dict[str, str] = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                p = c.get("profile") or {}
                title = p.get("current_title") or "?"
                company = p.get("current_company") or "?"
                yoe = p.get("years_of_experience") or 0
                cand_titles[cid] = f"{title} @ {company} ({yoe:.1f}yr)"
            except:
                continue

    # Run experiments
    print("\n" + "="*80)
    print("  WEIGHT EXPERIMENT  TOP 10 COMPARISON")
    print("="*80)
    results = {}
    for label, w1, w2, w3, w4 in EXPERIMENTS:
        ranked = rank_with_weights(
            features, bm25_norm, faiss_norm, skill_norm, traj_norm,
            w1, w2, w3, w4
        )
        top10 = ranked[:10]
        agree_scores = [signal_agreement(cid, bm25_norm, faiss_norm,
                                          skill_norm, traj_norm)
                        for cid in top10]
        avg_agree = sum(agree_scores) / len(agree_scores)
        full_agree = sum(1 for a in agree_scores if a == 4)
        results[label] = (top10, avg_agree, full_agree)

    # Print comparison table
    for label, w1, w2, w3, w4 in EXPERIMENTS:
        top10, avg_agree, full_agree = results[label]
        print(f"\n── {label} (BM25={w1} FAISS={w2} Skill={w3} Traj={w4}) ──")
        print(f"   4/4 agree: {full_agree}/10 | avg agree: {avg_agree:.1f}")
        for i, cid in enumerate(top10, 1):
            agree = signal_agreement(cid, bm25_norm, faiss_norm,
                                      skill_norm, traj_norm)
            name = cand_titles.get(cid, cid)
            marker = "★" if agree == 4 else " "
            print(f"   {marker} {i:2}. {name}")

    # Summary table
    print("\n" + "="*80)
    print("  SUMMARY")
    print(f"  {'Label':<20} {'4/4 agree':>10} {'avg agree':>10}")
    print(f"  {'-'*40}")
    for label, w1, w2, w3, w4 in EXPERIMENTS:
        _, avg, full = results[label]
        marker = " ◄ BEST" if full == max(r[2] for r in results.values()) else ""
        print(f"  {label:<20} {full:>10}/10 {avg:>10.2f}{marker}")
    print("="*80)
    print("\nPick the weights with highest 4/4 agree count.")
    print("If tied, pick the one whose top 10 looks most defensible.")


if __name__ == "__main__":
    main()
