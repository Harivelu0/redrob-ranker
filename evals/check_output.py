#!/usr/bin/env python3
"""
eval_submission.py — Pre-submission confidence checks.
Cross-references submission.csv against candidates.jsonl.
Run: python eval_submission.py --submission submission.csv --candidates path/to/candidates.jsonl
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ── JD-relevant terms for title check ────────────────────────────────────────
GOOD_TITLE_KW = [
    "machine learning", "ml", "nlp", "ai engineer", "data scientist",
    "applied scientist", "search engineer", "ranking", "recommendation",
    "deep learning", "research scientist", "information retrieval",
]
BAD_TITLE_KW = [
    "hr", "marketing", "sales", "recruiter", "operations manager",
    "product manager", "scrum master", "finance", "content writer",
    "business analyst", "account manager",
]
CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"
WARN  = "\033[93mWARN\033[0m"

results = []

def check(name, passed, detail="", warn=False):
    label = WARN if warn else (PASS if passed else FAIL)
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    results.append((name, status, detail))
    print(f"  [{label}] {name}")
    if detail:
        print(f"         {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--candidates", required=True)
    args, _ = parser.parse_known_args()  # ignore unknown args from validate.py

    print("\n" + "="*60)
    print("  REDROB SUBMISSION EVAL")
    print("="*60)

    # ── Load submission ───────────────────────────────────────────────────────
    rows = []
    with open(args.submission, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\n[1] FORMAT CHECKS")
    check("Exactly 100 rows", len(rows) == 100,
          f"got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    check("Ranks are 1-100 complete",
          sorted(ranks) == list(range(1, 101)),
          f"missing: {set(range(1,101)) - set(ranks)}" if sorted(ranks) != list(range(1,101)) else "")

    ids = [r["candidate_id"] for r in rows]
    check("No duplicate candidate_ids",
          len(ids) == len(set(ids)),
          f"{len(ids)-len(set(ids))} duplicates found")

    scores = [float(r["score"]) for r in rows]
    monotone = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    check("Scores non-increasing", monotone,
          f"violation at rank {next((i+1 for i in range(len(scores)-1) if scores[i]<scores[i+1]), '?')}" if not monotone else "")

    check("All reasoning non-empty",
          all(r["reasoning"].strip() for r in rows),
          "empty reasoning rows: " + str([r["rank"] for r in rows if not r["reasoning"].strip()]))

    check("Score range 10-100",
          min(scores) >= 10 and max(scores) <= 100,
          f"min={min(scores):.2f} max={max(scores):.2f}")

    # ── Load top 100 from candidates.jsonl ───────────────────────────────────
    print(f"\n[2] CROSS-REFERENCE WITH CANDIDATES.JSONL")
    top_ids = set(ids)
    cand_data = {}
    print(f"  Scanning candidates.jsonl for top 100...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid in top_ids:
                    cand_data[cid] = c
            except:
                continue

    check("All 100 candidates found in jsonl",
          len(cand_data) == 100,
          f"missing {100-len(cand_data)} candidates from jsonl")

    # ── Per-candidate checks ──────────────────────────────────────────────────
    print(f"\n[3] CANDIDATE QUALITY CHECKS")

    # Honeypot check
    honeypots = []
    for cid, c in cand_data.items():
        career = c.get("career_history") or []
        skills = c.get("skills") or []
        if sum(1 for j in career if j.get("is_current")) > 1:
            honeypots.append(cid)
        for sk in skills:
            if sk.get("proficiency") == "expert" and (sk.get("duration_months") or 0) == 0:
                honeypots.append(cid)
    check("No honeypots in top 100", len(honeypots) == 0,
          f"honeypots found: {honeypots}")

    # Consulting-only check
    consulting_only = []
    for cid, c in cand_data.items():
        career = c.get("career_history") or []
        if not career:
            continue
        if all(any(firm in (j.get("company") or "").lower() for firm in CONSULTING)
               for j in career):
            consulting_only.append(cid)
    check("No consulting-only careers in top 100",
          len(consulting_only) == 0,
          f"consulting-only: {consulting_only}")

    # Title relevance
    bad_titles = []
    for cid, c in cand_data.items():
        title = (c.get("profile") or {}).get("current_title") or ""
        title_lower = title.lower()
        if any(kw in title_lower for kw in BAD_TITLE_KW):
            bad_titles.append((cid, title))
    check("No ANTI_TITLE candidates in top 100",
          len(bad_titles) == 0,
          f"bad titles: {bad_titles}")

    # YOE distribution
    yoe_vals = [(cid, (c.get("profile") or {}).get("years_of_experience") or 0)
                for cid, c in cand_data.items()]
    sub5 = [(cid, yoe) for cid, yoe in yoe_vals if yoe < 5]
    sub4 = [(cid, yoe) for cid, yoe in yoe_vals if yoe < 4]
    over15 = [(cid, yoe) for cid, yoe in yoe_vals if yoe > 15]

    # Check sub-5yr in top 10
    top10_ids = {r["candidate_id"] for r in rows if int(r["rank"]) <= 10}
    sub5_top10 = [(cid, yoe) for cid, yoe in sub5 if cid in top10_ids]
    check("No sub-5yr candidates in top 10",
          len(sub5_top10) == 0,
          f"sub-5yr in top 10: {sub5_top10}")

    check("Sub-5yr candidates in top 100 ≤ 20",
          len(sub5) <= 20,
          f"{len(sub5)} sub-5yr candidates: {[(c,round(y,1)) for c,y in sub5[:5]]}...",
          warn=len(sub5) > 10)

    check("No sub-4yr candidates in top 50",
          not any(r["candidate_id"] in {c for c,_ in sub4} and int(r["rank"]) <= 50
                  for r in rows),
          f"sub-4yr in top 50: {[r['candidate_id'] for r in rows if r['candidate_id'] in {c for c,_ in sub4} and int(r['rank'])<=50]}")

    # Location distribution
    print(f"\n[4] LOCATION & AVAILABILITY")
    locations = []
    outside_india = []
    for cid, c in cand_data.items():
        profile = c.get("profile") or {}
        loc = (profile.get("location") or "").lower()
        country = (profile.get("country") or "").lower()
        locations.append(loc)
        if country not in ("india", "in") and "india" not in loc:
            outside_india.append((cid, profile.get("location")))

    check("Outside-India candidates ≤ 15 in top 100",
          len(outside_india) <= 15,
          f"{len(outside_india)} outside India: {[x[1] for x in outside_india]}",
          warn=len(outside_india) > 8)

    # Check outside-India in top 10
    top10_outside = [(cid, loc) for cid, loc in outside_india if cid in top10_ids]
    check("No outside-India in top 10",
          len(top10_outside) == 0,
          f"outside-India in top 10: {top10_outside}",
          warn=len(top10_outside) > 0)

    # Notice period
    sig_data = {cid: c.get("redrob_signals") or {} for cid, c in cand_data.items()}
    long_notice = [(cid, sig.get("notice_period_days")) for cid, sig in sig_data.items()
                   if (sig.get("notice_period_days") or 0) > 90]
    check("Long notice period (>90d) candidates ≤ 30 in top 100",
          len(long_notice) <= 30,
          f"{len(long_notice)} candidates with >90d notice",
          warn=len(long_notice) > 20)

    # Skill relevance
    print(f"\n[5] REASONING QUALITY")
    JD_SKILLS = {"weaviate","qdrant","pinecone","milvus","faiss","opensearch",
                 "elasticsearch","vector search","nlp","pytorch","tensorflow",
                 "semantic search","embeddings","information retrieval","learning to rank",
                 "scikit-learn","python","lora","fine-tuning","bm25"}

    bad_reasoning = []
    for row in rows[:20]:  # check top 20 thoroughly
        reasoning = row["reasoning"].lower()
        has_jd_skill = any(sk in reasoning for sk in JD_SKILLS)
        if not has_jd_skill:
            bad_reasoning.append(row["rank"])

    check("Top 20 reasoning all mention JD-relevant skills",
          len(bad_reasoning) == 0,
          f"ranks without JD skills in reasoning: {bad_reasoning}")

    # Check no generic "limited signal" in top 50
    generic = [r["rank"] for r in rows if int(r["rank"]) <= 50
               and "limited signal" in r["reasoning"].lower()]
    check("No generic reasoning in top 50",
          len(generic) == 0,
          f"generic reasoning at ranks: {generic}")

    # Score distribution health
    print(f"\n[6] SCORE DISTRIBUTION")
    top10_scores = scores[:10]
    score_spread = top10_scores[0] - top10_scores[-1]
    check("Top 10 scores well spread (spread > 20)",
          score_spread > 20,
          f"spread = {score_spread:.2f} (rank1={top10_scores[0]:.1f} rank10={top10_scores[-1]:.1f})",
          warn=score_spread < 30)

    bottom10_scores = scores[-10:]
    check("Bottom 10 scores above 10",
          min(bottom10_scores) >= 10,
          f"min bottom score = {min(bottom10_scores):.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    warned = sum(1 for _, s, _ in results if s == "WARN")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print(f"  TOTAL : {total}  |  PASS: {passed}  |  WARN: {warned}  |  FAIL: {failed}")

    confidence = int((passed + warned * 0.5) / total * 100)
    print(f"  CONFIDENCE SCORE: {confidence}%")

    if failed == 0 and warned <= 2:
        print("  → READY TO SUBMIT")
    elif failed == 0:
        print("  → SUBMIT WITH CAUTION — review warnings above")
    else:
        print("  → DO NOT SUBMIT — fix failures first")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
