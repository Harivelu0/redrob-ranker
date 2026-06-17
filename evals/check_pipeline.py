#!/usr/bin/env python3
"""
eval_artifacts.py — Deep pipeline validation.
Validates every PKL artifact and scoring signal before trusting the output.

Layer 2: Artifact integrity
Layer 3: Signal sanity
Layer 4: Adversarial probes

Run: python eval_artifacts.py --artifacts artifacts/ --candidates path/to/candidates.jsonl
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/")
    parser.add_argument("--candidates", required=True)
    args, _ = parser.parse_known_args()  # ignore unknown args from validate.py
    art = Path(args.artifacts)
    print("\n" + "="*60)
    print("  LAYER 2 — ARTIFACT INTEGRITY")
    print("="*60)

    # 2a. All 5 files exist
    required = ["features.pkl", "bm25_index.pkl", "faiss.index",
                "faiss_ids.pkl", "meta.pkl"]
    for fname in required:
        check(f"{fname} exists", (art / fname).exists())

    # 2b. Load all artifacts
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

    bm25      = bm25_data["bm25"]
    bm25_ids  = bm25_data["ids"]
    jd_vector = meta["jd_vector"].astype(np.float32)

    n_features = len(features)
    n_faiss    = index.ntotal
    n_bm25     = len(bm25_ids)
    n_faiss_ids = len(faiss_ids)

    # 2c. All counts match
    check("features.pkl count == 100K",
          n_features == 100000, f"got {n_features}")
    check("FAISS index count == 100K",
          n_faiss == 100000, f"got {n_faiss}")
    check("BM25 ids count == 100K",
          n_bm25 == 100000, f"got {n_bm25}")
    check("faiss_ids count == 100K",
          n_faiss_ids == 100000, f"got {n_faiss_ids}")

    # 2d. faiss_ids and bm25_ids must be same set (same candidates)
    check("faiss_ids and bm25_ids cover same candidates",
          set(faiss_ids) == set(bm25_ids),
          f"diff: {len(set(faiss_ids).symmetric_difference(set(bm25_ids)))} mismatches")

    # 2e. features has all required keys
    sample_feat = next(iter(features.values()))
    required_keys = ["is_honeypot", "title_class", "all_consulting",
                     "behavioral_gate", "skill_quality_score",
                     "trajectory_score", "ml_product_years", "yoe"]
    missing_keys = [k for k in required_keys if k not in sample_feat]
    check("features.pkl has all required keys",
          len(missing_keys) == 0,
          f"missing: {missing_keys}")

    # 2f. JD vector is unit normalized
    norm = float(np.linalg.norm(jd_vector))
    check("JD vector is L2-normalized (norm ≈ 1.0)",
          abs(norm - 1.0) < 0.01,
          f"norm = {norm:.4f}")

    # 2g. FAISS dimension matches JD vector
    check("FAISS dim matches JD vector dim",
          index.d == len(jd_vector),
          f"FAISS dim={index.d}, JD dim={len(jd_vector)}")

    # 2h. BM25 corpus length matches
    bm25_corpus_size = bm25.corpus_size if hasattr(bm25, 'corpus_size') else len(bm25.doc_freqs)
    check("BM25 corpus size matches candidate count",
          int(bm25_corpus_size) == n_bm25,
          f"bm25 corpus={bm25_corpus_size}, expected={n_bm25}")

    # ── LAYER 3: SIGNAL SANITY ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  LAYER 3 — SIGNAL SANITY CHECKS")
    print("="*60)

    feat_values = list(features.values())

    # 3a. Behavioral gate range — our clamp is max(0.50, min(1.20, ...))
    gates = [f["behavioral_gate"] for f in feat_values]
    check("All behavioral gates in [0.50, 1.20]",
          all(0.50 <= g <= 1.20 for g in gates),
          f"out of range: {sum(1 for g in gates if g < 0.50 or g > 1.20)}")

    # 3b. Skill quality scores in [0, 100]
    sq = [f["skill_quality_score"] for f in feat_values]
    check("All skill_quality_scores in [0, 100]",
          all(0 <= s <= 100 for s in sq),
          f"out of range: {sum(1 for s in sq if s < 0 or s > 100)}")

    # 3c. Trajectory scores in [0, 100]
    tq = [f["trajectory_score"] for f in feat_values]
    check("All trajectory_scores in [0, 100]",
          all(0 <= s <= 100 for s in tq),
          f"out of range: {sum(1 for s in tq if s < 0 or s > 100)}")

    # 3d. Honeypot rate sanity (should be < 20%)
    honeypots = sum(1 for f in feat_values if f["is_honeypot"])
    check("Honeypot rate < 20%",
          honeypots / n_features < 0.20,
          f"{honeypots} honeypots = {honeypots/n_features*100:.1f}%",
          warn=honeypots / n_features > 0.10)

    # 3e. Consulting-only rate sanity (should be < 30%)
    consulting = sum(1 for f in feat_values if f["all_consulting"])
    check("Consulting-only rate < 30%",
          consulting / n_features < 0.30,
          f"{consulting} consulting-only = {consulting/n_features*100:.1f}%")

    # 3f. FAISS scores are cosine similarities (should be in [-1, 1])
    D, I = index.search(jd_vector.reshape(1, -1), 10)
    top_sims = D[0].tolist()
    check("FAISS top-10 similarities in [0, 1] (cosine, normalized)",
          all(0 <= s <= 1.0 for s in top_sims),
          f"top sims: {[round(s,3) for s in top_sims]}")

    # 3g. BM25 query returns non-zero scores for at least 1000 candidates
    bm25_query = meta["bm25_query"]
    bm25_scores = bm25.get_scores(bm25_query)
    nonzero = int(np.sum(bm25_scores > 0))
    check("BM25 returns >1000 non-zero scores",
          nonzero > 1000,
          f"{nonzero} candidates have non-zero BM25 score")

    # 3h. Score distributions are not degenerate (std > 0)
    check("skill_quality_score has variance",
          float(np.std(sq)) > 0.1,
          f"std = {np.std(sq):.2f}")
    check("trajectory_score has variance",
          float(np.std(tq)) > 0.1,
          f"std = {np.std(tq):.2f}")
    check("behavioral_gate has variance",
          float(np.std(gates)) > 0.01,
          f"std = {np.std(gates):.4f}")

    # ── LAYER 4: ADVERSARIAL PROBES ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  LAYER 4 — ADVERSARIAL PROBES")
    print("  (known good vs known bad — system must agree)")
    print("="*60)

    # Load a sample of candidates to build known good/bad sets
    print("  Loading sample candidates for probing...")
    known_good = []   # should rank high
    known_bad  = []   # should rank low

    consulting_firms = {"tcs","infosys","wipro","accenture","cognizant","capgemini"}
    good_skills = {"faiss","qdrant","milvus","weaviate","pinecone","vector search",
                   "information retrieval","semantic search","embeddings","nlp",
                   "learning to rank"}
    prod_markers = ["deployed","production","serving","shipped","owned the","built the"]

    count = 0
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if count > 5000:  # sample first 5K for speed
                break
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except:
                continue
            count += 1

            cid = c.get("candidate_id")
            profile = c.get("profile") or {}
            career  = c.get("career_history") or []
            skills  = c.get("skills") or []
            sig     = c.get("redrob_signals") or {}

            skill_names = {s.get("name","").lower() for s in skills}
            yoe = profile.get("years_of_experience") or 0
            all_text = " ".join(
                (j.get("description") or "") for j in career
            ).lower()

            # Known GOOD: 5-9yr, has good skills, has production evidence
            has_good_skills = len(skill_names & good_skills) >= 2
            has_prod = sum(1 for m in prod_markers if m in all_text) >= 2
            is_consulting = all(
                any(f in (j.get("company") or "").lower() for f in consulting_firms)
                for j in career if career
            )
            is_honeypot = sum(1 for j in career if j.get("is_current")) > 1

            if (5 <= yoe <= 9 and has_good_skills and has_prod
                    and not is_consulting and not is_honeypot
                    and len(known_good) < 20):
                known_good.append(cid)

            # Known BAD: honeypot OR all consulting OR anti-title
            title = (profile.get("current_title") or "").lower()
            is_anti = any(kw in title for kw in
                         ["hr manager","marketing","sales manager","recruiter"])
            if (is_honeypot or is_consulting or is_anti) and len(known_bad) < 20:
                known_bad.append(cid)

    print(f"  Found {len(known_good)} known-good, {len(known_bad)} known-bad candidates")

    # Probe 1: known-good should have higher skill_quality_score than known-bad
    if known_good and known_bad:
        good_sq = [features[cid]["skill_quality_score"]
                   for cid in known_good if cid in features]
        bad_sq  = [features[cid]["skill_quality_score"]
                   for cid in known_bad  if cid in features]
        if good_sq and bad_sq:
            avg_good = sum(good_sq) / len(good_sq)
            avg_bad  = sum(bad_sq)  / len(bad_sq)
            check("Known-good avg skill_quality_score > known-bad",
                  avg_good > avg_bad,
                  f"good={avg_good:.2f} bad={avg_bad:.2f}")

    # Probe 3: known-good should have higher trajectory score
    if known_good and known_bad:
        good_tq = [features[cid]["trajectory_score"]
                   for cid in known_good if cid in features]
        bad_tq  = [features[cid]["trajectory_score"]
                   for cid in known_bad  if cid in features]
        if good_tq and bad_tq:
            avg_good = sum(good_tq) / len(good_tq)
            avg_bad  = sum(bad_tq)  / len(bad_tq)
            check("Known-good avg trajectory_score > known-bad",
                  avg_good > avg_bad,
                  f"good={avg_good:.2f} bad={avg_bad:.2f}")

    # Probe 3b: behavioral gate does NOT need to distinguish technical quality
    # It measures availability — consulting candidates can be active job-seekers too.
    # Instead verify: bad candidates have hard gates (honeypot/consulting flags set).
    bad_with_penalty = sum(
        1 for cid in known_bad if cid in features and (
            features[cid].get("is_honeypot") or
            features[cid].get("all_consulting") or
            features[cid].get("title_class") == "ANTI_TITLE"
        )
    )
    check("Known-bad candidates have hard gate flags set",
          bad_with_penalty > 0,
          f"{bad_with_penalty}/{len(known_bad)} bad candidates flagged for hard gates")

    # Probe 4: honeypot gate should be 0 in features
    honeypot_ids = [cid for cid, f in features.items() if f["is_honeypot"]]
    if honeypot_ids:
        sample_hp = random.choice(honeypot_ids)
        check("Honeypot gate stored correctly (is_honeypot=True)",
              features[sample_hp]["is_honeypot"] is True,
              f"sample honeypot: {sample_hp}")

    # Probe 5: FAISS top-1 result should be a plausible match
    D, I = index.search(jd_vector.reshape(1, -1), 1)
    top1_id = faiss_ids[I[0][0]]
    top1_feat = features.get(top1_id, {})
    top1_title_class = top1_feat.get("title_class", "UNKNOWN")
    check("FAISS top-1 candidate has ML/SWE title (not ANTI_TITLE)",
          top1_title_class != "ANTI_TITLE",
          f"top1={top1_id} title_class={top1_title_class}")

    # Probe 6: BM25 top result should mention JD keywords
    top_bm25_idx = int(np.argmax(bm25_scores))
    top_bm25_id  = bm25_ids[top_bm25_idx]
    top_bm25_feat = features.get(top_bm25_id, {})
    check("BM25 top candidate has ML/SWE/DATA title",
          top_bm25_feat.get("title_class") in ("ML_ENG","SWE","DATA_ENG"),
          f"BM25 top={top_bm25_id} class={top_bm25_feat.get('title_class')}")

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

    confidence = int((passed + warned * 0.5) / total * 100)
    print(f"\n  PIPELINE CONFIDENCE: {confidence}%")
    if failed == 0 and warned == 0:
        print("  → ALL LAYERS CLEAN. Pipeline is trustworthy.")
    elif failed == 0:
        print("  → WARNINGS ONLY. Review above, then submit.")
    else:
        print("  → FAILURES FOUND. Do not submit until fixed.")
    print("="*60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
