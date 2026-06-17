#!/usr/bin/env python3
"""
eval_strict.py — Strict pre-submission eval. No ground truth needed.
Tests ranking quality, stability, signal agreement, and anti-patterns.

Run: python eval_strict.py \
        --submission submission.csv \
        --artifacts artifacts/ \
        --candidates path/to/candidates.jsonl \
        --sample_submission path/to/sample_submission.csv
"""

import argparse, csv, json, pickle, random, sys
from pathlib import Path

# Allow imports from project root (rank.py, explain.py etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
results = []

def check(name, passed, detail="", warn=False):
    label = WARN if warn else (PASS if passed else FAIL)
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    results.append((name, status, detail))
    print(f"  [{label}] {name}")
    if detail:
        print(f"         → {detail}")
    return passed


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--artifacts", default="artifacts/")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sample_submission", default=None)
    args, _ = parser.parse_known_args()  # ignore unknown args from validate.py
    art = Path(args.artifacts)
    rows = load_csv(args.submission)
    top_ids  = [r["candidate_id"] for r in rows]
    scores   = [float(r["score"]) for r in rows]
    top10_ids = set(top_ids[:10])
    top20_ids = set(top_ids[:20])

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
    bm25       = bm25_data["bm25"]
    bm25_ids   = bm25_data["ids"]
    jd_vector  = meta["jd_vector"].astype(np.float32)
    bm25_query = meta["bm25_query"]

    # ── CHECK 1: SAMPLE SUBMISSION TRAP ──────────────────────────────────────
    print("\n" + "="*60)
    print("  CHECK 1 — SAMPLE SUBMISSION TRAP")
    print("="*60)

    if args.sample_submission and Path(args.sample_submission).exists():
        sample_rows = load_csv(args.sample_submission)
        # sample_submission ranks HR/marketing types #1-19 (the trap)
        sample_top20 = {r["candidate_id"] for r in sample_rows[:20]}
        overlap = sample_top20 & set(top_ids)
        check("None of sample_submission's top 20 in our top 100",
              len(overlap) == 0,
              f"overlap: {overlap}")

        # The trap candidates should be near the BOTTOM of our ranking
        sample_top5 = [r["candidate_id"] for r in sample_rows[:5]]
        our_ranks = {r["candidate_id"]: int(r["rank"]) for r in rows}
        trap_in_our_bottom = [cid for cid in sample_top5
                               if our_ranks.get(cid, 999) > 80]
        check("Sample submission's top 5 appear in our bottom 20 (or absent)",
              len(trap_in_our_bottom) >= 3 or
              all(cid not in our_ranks for cid in sample_top5),
              f"trap candidates in our bottom: {len(trap_in_our_bottom)}/5")
    else:
        print("  [SKIP] No sample_submission provided — skipping trap check")

    # ── CHECK 2: SCORE DIFFERENTIATION ───────────────────────────────────────
    print("\n" + "="*60)
    print("  CHECK 2 — SCORE DIFFERENTIATION")
    print("="*60)

    # Rank 1 vs rank 2 gap should be meaningful
    gap_1_2 = scores[0] - scores[1]
    check("Rank 1 vs rank 2 gap > 0.5",
          gap_1_2 > 0.5,
          f"gap = {gap_1_2:.4f}")

    # Top 10 should span at least 25 points
    top10_spread = scores[0] - scores[9]
    check("Top 10 score spread > 25",
          top10_spread > 25,
          f"spread = {top10_spread:.2f}")

    # No score ties in top 20
    top20_scores = scores[:20]
    ties = sum(1 for i in range(len(top20_scores)-1)
               if abs(top20_scores[i] - top20_scores[i+1]) < 0.001)
    check("No score ties in top 20",
          ties == 0,
          f"{ties} ties found")

    # Score distribution — should not be clustered (std check)
    score_std = float(np.std(scores))
    check("Score distribution has healthy spread (std > 18)",
          score_std > 18,
          f"std = {score_std:.2f}")

    # ── CHECK 3: CROSS-SIGNAL AGREEMENT FOR TOP 10 ───────────────────────────
    print("\n" + "="*60)
    print("  CHECK 3 — CROSS-SIGNAL AGREEMENT (top 10)")
    print("="*60)

    # Compute all 4 raw signals for every candidate
    from rank import minmax_norm

    bm25_raw = bm25.get_scores(bm25_query)
    bm25_scores_all = {cid: float(bm25_raw[i]) for i, cid in enumerate(bm25_ids)}
    bm25_norm = minmax_norm(bm25_scores_all)

    n_total = index.ntotal
    D, I = index.search(jd_vector.reshape(1, -1), n_total)
    faiss_scores_all = {faiss_ids[I[0][i]]: float(D[0][i]) for i in range(n_total)}
    faiss_norm = minmax_norm(faiss_scores_all)

    skill_raw = {cid: f["skill_quality_score"] for cid, f in features.items()}
    skill_norm = minmax_norm(skill_raw)

    traj_raw = {cid: f["trajectory_score"] for cid, f in features.items()}
    traj_norm = minmax_norm(traj_raw)

    print(f"\n  Signal breakdown for top 10:")
    print(f"  {'Rank':<5} {'CID':<16} {'BM25':>6} {'FAISS':>6} {'Skill':>6} {'Traj':>6} {'Agree':>6}")
    print(f"  {'-'*55}")

    disagreements = 0
    for row in rows[:10]:
        cid = row["candidate_id"]
        rank = row["rank"]
        s1 = bm25_norm.get(cid, 0)
        s2 = faiss_norm.get(cid, 0)
        s3 = skill_norm.get(cid, 0)
        s4 = traj_norm.get(cid, 0)
        # Agreement = how many signals are in top 30% (>0.70)
        agree = sum(1 for s in [s1, s2, s3, s4] if s > 0.70)
        if agree < 2:
            disagreements += 1
        print(f"  {rank:<5} {cid:<16} {s1:>6.3f} {s2:>6.3f} {s3:>6.3f} {s4:>6.3f} {agree:>5}/4")

    check("Top 10 candidates have ≥2 signals in top 30%",
          disagreements == 0,
          f"{disagreements} candidates with <2 agreeing signals")

    # ── CHECK 4: ANTI-PATTERN DETECTION ──────────────────────────────────────
    print("\n" + "="*60)
    print("  CHECK 4 — JD ANTI-PATTERN CHECK (top 20)")
    print("="*60)

    # Load top 20 candidates from jsonl
    top20_data = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid in top20_ids:
                    top20_data[cid] = c
            except:
                continue

    # JD explicit do-not-want list
    framework_only_in_top20 = []
    pure_research_in_top20  = []
    cv_speech_in_top20      = []
    title_chaser_in_top20   = []
    closed_source_in_top20  = []

    for cid in top_ids[:20]:
        feat = features.get(cid, {})
        if feat.get("framework_only"):
            framework_only_in_top20.append(cid)
        if feat.get("pure_research"):
            pure_research_in_top20.append(cid)
        if feat.get("cv_speech_primary"):
            cv_speech_in_top20.append(cid)
        if feat.get("title_chaser"):
            title_chaser_in_top20.append(cid)

    check("No framework-only candidates in top 20",
          len(framework_only_in_top20) == 0,
          f"found: {framework_only_in_top20}")
    check("No pure-research candidates in top 20",
          len(pure_research_in_top20) == 0,
          f"found: {pure_research_in_top20}")
    check("No CV/speech-primary candidates in top 20",
          len(cv_speech_in_top20) == 0,
          f"found: {cv_speech_in_top20}")
    check("No title-chasers in top 10",
          not any(features.get(cid, {}).get("title_chaser")
                  for cid in top_ids[:10]),
          f"title-chasers in top 10: {[cid for cid in top_ids[:10] if features.get(cid,{}).get('title_chaser')]}")

    # ── CHECK 5: RANK STABILITY ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  CHECK 5 — RANK STABILITY")
    print("="*60)
    print("  (Re-scoring top 100 with ±1% noise on each signal)")

    W_BM25, W_FAISS, W_SKILL, W_TRAJ = 0.20, 0.30, 0.20, 0.30  # must match rank.py

    def rescore(noise=0.0):
        final = {}
        for cid in top_ids:
            feat = features.get(cid, {})
            s1 = bm25_norm.get(cid, 0) * (1 + random.uniform(-noise, noise))
            s2 = faiss_norm.get(cid, 0) * (1 + random.uniform(-noise, noise))
            s3 = skill_norm.get(cid, 0) * (1 + random.uniform(-noise, noise))
            s4 = traj_norm.get(cid, 0) * (1 + random.uniform(-noise, noise))
            yoe = feat.get("yoe") or 0.0
            yoe_f = 1.0 if 5<=yoe<=9 else (max(0.85,1-(yoe-9)*0.02) if yoe>9 else max(0.40,yoe/5))
            raw = (W_BM25*s1 + W_FAISS*s2 + W_SKILL*s3 + W_TRAJ*s4) * yoe_f
            gate = feat.get("behavioral_gate", 1.0)
            final[cid] = raw * gate
        return sorted(final, key=lambda c: -final[c])

    random.seed(42)
    original_top10 = set(top_ids[:10])
    trials = 10
    instability_count = 0
    for _ in range(trials):
        noisy = rescore(noise=0.01)
        noisy_top10 = set(noisy[:10])
        overlap = len(original_top10 & noisy_top10)
        if overlap < 8:  # if more than 2 change, unstable
            instability_count += 1

    check("Top 10 stable under ±1% score noise (8/10 same across 10 trials)",
          instability_count == 0,
          f"{instability_count}/10 trials had <8 overlap in top 10",
          warn=instability_count > 0)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    total  = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    warned = sum(1 for _, s, _ in results if s == "WARN")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print(f"  TOTAL: {total}  |  PASS: {passed}  |  WARN: {warned}  |  FAIL: {failed}")
    if failed:
        print("\n  FAILURES:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"    ✗ {name}: {detail}")

    confidence = int((passed + warned*0.5) / total * 100)
    print(f"\n  STRICT EVAL CONFIDENCE: {confidence}%")
    if failed == 0 and warned == 0:
        print("  → SUBMIT.")
    elif failed == 0:
        print("  → SUBMIT WITH CAUTION — review warnings.")
    else:
        print("  → FIX FAILURES BEFORE SUBMITTING.")
    print("="*60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
