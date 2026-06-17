#!/usr/bin/env python3
"""
precompute.py  One-time preprocessing for Redrob Hackathon Ranker.
No time limit. Run once before rank.py.

Usage:
    python precompute.py \
        --candidates "C:/path/to/candidates.jsonl" \
        --jd "C:/path/to/job_description.docx" \
        --out artifacts/
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Heavy ML imports are lazy (inside main()) so boolean detectors are testable
# without installing sentence-transformers / faiss / sklearn.
try:
    import numpy as np
    from sklearn.preprocessing import normalize as sk_normalize
    from tqdm import tqdm
except ImportError:
    np = None  # type: ignore
    sk_normalize = None  # type: ignore
    tqdm = None  # type: ignore

# ── Date reference ───────────────────────────────────────────────────────────
TODAY = date.today()

# ── Consulting firms (exact match against lowercased company name) ────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

# ── Company prestige: structural signal (industry + size) ────────────────────
# Minimal hardcoded list  only unambiguous global T1 for ML roles.
# Everything else derived from industry + company_size fields in the data.
KNOWN_T1_COMPANIES = {
    "google", "meta", "apple", "microsoft", "amazon",
    "openai", "anthropic", "deepmind", "netflix", "nvidia",
}

# Product-tech industries (vs IT services/consulting which get no prestige)
PRODUCT_TECH_INDUSTRIES = {
    "Software", "Fintech", "AI/ML", "SaaS", "AdTech", "Gaming",
    "E-commerce", "EdTech", "HealthTech", "HealthTech AI",
    "Conversational AI", "AI Services", "Food Delivery",
    "Insurance Tech", "Voice AI", "Internet", "Consumer Electronics",
    "Technology",  # generic catch-all used by some data sources
}
SERVICES_INDUSTRIES = {"IT Services", "Consulting"}


def compute_prestige_bonus(job: dict) -> float:
    """
    Data-driven prestige: industry + company_size, with a tiny hardcoded list
    for unambiguous global T1. This is how production hiring systems work 
    structured metadata first, exact names only as a last-resort override.
    Returns 0.0 – 0.20.
    """
    company  = (job.get("company") or "").lower()
    industry = (job.get("industry") or "").strip()
    size     = (job.get("company_size") or "")

    # Hard override for known global T1 (too few to derive structurally)
    if any(t in company for t in KNOWN_T1_COMPANIES):
        return 0.20

    # IT services / consulting → no prestige bonus (already gated separately)
    if industry in SERVICES_INDUSTRIES:
        return 0.00

    # Product-tech companies: bonus scales with size
    if industry in PRODUCT_TECH_INDUSTRIES:
        if size in ("5001-10000", "10001+"):  return 0.15
        if size in ("1001-5000"):             return 0.10
        if size in ("501-1000", "201-500"):   return 0.07
        return 0.03  # small startup  some credit

    # Unknown industry  weak size proxy only
    if size in ("5001-10000", "10001+"):  return 0.05
    return 0.00

# ── Seniority level keywords ──────────────────────────────────────────────────
SENIOR_TITLE_KW = [
    "staff", "principal", "lead", "senior", "sr.", "sr ", "head of",
    "director", "vp", "vice president", "distinguished", "fellow",
]

# ── Title classifiers ─────────────────────────────────────────────────────────
ML_TITLE_KW = [
    "machine learning", "ml engineer", "ai engineer", "deep learning",
    "data scientist", "research scientist", "nlp engineer", "applied scientist",
    "ai researcher", "search engineer", "ranking engineer",
    "recommendation engineer", "information retrieval engineer",
    "computer vision engineer",
]
SWE_TITLE_KW = [
    "software engineer", "backend engineer", "frontend engineer",
    "fullstack engineer", "full stack engineer", "platform engineer",
    "software developer", "swe",
]
DATA_TITLE_KW = [
    "data engineer", "analytics engineer", "big data engineer",
    "data infrastructure", "data platform engineer", "etl engineer",
]
ANTI_TITLE_KW = [
    "operations manager", "hr manager", "human resources", "marketing manager",
    "sales manager", "finance manager", "project manager", "scrum master",
    "recruiter", "content writer", "support agent", "customer success",
    "business development", "account manager", "product manager",
]

# ── Skill buckets ─────────────────────────────────────────────────────────────
CV_SPEECH_SKILLS = {
    "image classification", "computer vision", "object detection",
    "image segmentation", "semantic segmentation", "pose estimation",
    "optical flow", "speech recognition", "asr", "tts", "text-to-speech",
    "speaker diarization", "speaker recognition", "robotics", "slam",
    "lidar", "depth estimation",
}
NLP_IR_SKILLS = {
    "nlp", "natural language processing", "information retrieval",
    "vector search", "embeddings", "semantic search", "ranking",
    "recommendation", "bert", "transformers", "text classification",
    "named entity recognition", "question answering", "search",
    "retrieval", "reranking", "dense retrieval", "sparse retrieval",
    "bm25", "faiss", "approximate nearest neighbor", "learning to rank",
}
PRE_LLM_ML_SKILLS = {
    "sklearn", "scikit-learn", "pytorch", "tensorflow", "keras",
    "xgboost", "lightgbm", "catboost", "spark ml", "mlflow",
    "kubeflow", "gradient boosting", "random forest", "feature engineering",
    "recommender system", "collaborative filtering", "matrix factorization",
    "neural network", "deep learning",
}
LANGCHAIN_ONLY_SKILLS = {
    "langchain", "llamaindex", "llama index", "llama-index",
    "autogen", "crewai", "crew ai", "dspy", "haystack",
}

# ── Research vs Production markers ───────────────────────────────────────────
RESEARCH_MARKERS = [
    "published", "arxiv", "paper", "dataset", "benchmark",
    "novel approach", "proposed method", "ablation study", "sota",
    "state of the art", "conference", "journal", "phd", "research lab",
    "intern at research", "academic",
]
PRODUCTION_MARKERS = [
    "deployed", "production", "serving", "latency", "throughput",
    "traffic", "api endpoint", "inference service", "real-time",
    "shipped", "launched", "owned the", "built the", "served",
    "a/b test", "users", "customers", "revenue", "scale",
]

# ── JD embedding text (used for FAISS + skill relevance) ─────────────────────
JD_EMBED_TEXT = """
Senior AI Engineer Redrob AI production ML systems information retrieval vector search
semantic search embeddings ranking systems recommendation systems evaluation metrics
NDCG MRR MAP A/B testing production deployment serving infrastructure latency optimization
real-time inference Python vector databases Pinecone Weaviate Qdrant Milvus FAISS
pre-LLM machine learning end-to-end ML pipelines training production serving
shipped deployed ranking search recommendation system at scale
search quality learning to rank offline evaluation online evaluation experiment design
pytorch tensorflow scikit-learn transformers NLP natural language processing
"""

# ── BM25 query tokens ─────────────────────────────────────────────────────────
BM25_QUERY = (
    "production deployed shipped serving inference latency throughput traffic "
    "vector search embeddings semantic search information retrieval ranking recommendation "
    "NDCG MRR MAP evaluation offline online AB test experiment "
    "owned built end-to-end responsible full pipeline "
    "Python pytorch tensorflow sklearn scikit-learn "
    "faiss pinecone weaviate qdrant milvus vector database "
    "machine learning deep learning NLP natural language processing "
    "search quality ranking model learning to rank "
    "senior engineer AI ML"
).lower().split()

# ── Proficiency numeric values ────────────────────────────────────────────────
PROFICIENCY_MAP = {
    "beginner": 0.25,
    "intermediate": 0.60,
    "advanced": 0.85,
    "expert": 1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def read_jd_text(jd_path: Optional[str]) -> str:
    """Read JD from docx, fall back to embedded JD_EMBED_TEXT."""
    if jd_path and Path(jd_path).exists():
        try:
            from docx import Document
            doc = Document(jd_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"[WARN] Could not read JD docx ({e}), using fallback.")
    return JD_EMBED_TEXT


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d: Optional[date]) -> int:
    if d is None:
        return 9999
    return (TODAY - d).days


def get_career_text(career: List[dict]) -> str:
    """All career descriptions concatenated."""
    parts = []
    for job in career:
        desc = (job.get("description") or "").strip()
        title = (job.get("title") or "").strip()
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)
    return " ".join(parts)


def tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def classify_title(title: str) -> str:
    t = title.lower()
    if any(kw in t for kw in ANTI_TITLE_KW):
        return "ANTI_TITLE"
    if any(kw in t for kw in ML_TITLE_KW):
        return "ML_ENG"
    if any(kw in t for kw in DATA_TITLE_KW):
        return "DATA_ENG"
    if any(kw in t for kw in SWE_TITLE_KW):
        return "SWE"
    return "OTHER"

# ─────────────────────────────────────────────────────────────────────────────
# BOOLEAN FLAG DETECTORS
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(cand: dict) -> bool:
    """Impossible data signatures → honeypot."""
    career = cand.get("career_history") or []
    skills = cand.get("skills") or []

    # Two jobs claiming is_current=True
    if sum(1 for j in career if j.get("is_current")) > 1:
        return True

    # expert skill with 0 duration_months
    for sk in skills:
        if sk.get("proficiency") == "expert" and (sk.get("duration_months") or 0) == 0:
            return True

    # Total tenure months > claimed experience * 14 (allow 2 months gap noise)
    claimed_yrs = cand.get("profile", {}).get("years_of_experience") or 0
    total_months = sum(j.get("duration_months") or 0 for j in career)
    if claimed_yrs > 0 and total_months > claimed_yrs * 14:
        return True

    return False


def detect_all_consulting(career: List[dict]) -> bool:
    if not career:
        return False
    for job in career:
        company = (job.get("company") or "").lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return False
    return True


def detect_cv_speech_primary(skills: List[dict], career_text: str) -> bool:
    """Primary domain is CV/speech/robotics without NLP/IR counterbalance."""
    cv_months = sum(
        (sk.get("duration_months") or 0)
        for sk in skills
        if sk.get("name", "").lower() in CV_SPEECH_SKILLS
    )
    nlp_months = sum(
        (sk.get("duration_months") or 0)
        for sk in skills
        if sk.get("name", "").lower() in NLP_IR_SKILLS
    )
    ct = career_text.lower()
    nlp_in_career = any(kw in ct for kw in ["nlp", "embedding", "retrieval", "ranking", "search"])
    if cv_months > 0 and cv_months > nlp_months * 1.5 and not nlp_in_career:
        return True
    return False


def detect_pure_research(career_text: str) -> bool:
    ct = career_text.lower()
    research_hits = sum(1 for m in RESEARCH_MARKERS if m in ct)
    production_hits = sum(1 for m in PRODUCTION_MARKERS if m in ct)
    return research_hits >= 3 and production_hits <= 1


def detect_framework_only(skills: List[dict]) -> bool:
    """Only recent AI experience is LangChain-type, no pre-LLM ML skills."""
    has_framework = any(
        sk.get("name", "").lower() in LANGCHAIN_ONLY_SKILLS
        for sk in skills
    )
    if not has_framework:
        return False
    # Check if ANY pre-LLM ML skills exist
    has_pre_llm = any(
        sk.get("name", "").lower() in PRE_LLM_ML_SKILLS
        for sk in skills
    )
    return not has_pre_llm


def detect_title_chaser(career: List[dict]) -> bool:
    if len(career) < 3:
        return False
    durations = [j.get("duration_months") or 0 for j in career]
    avg_tenure = sum(durations) / len(durations)
    return avg_tenure < 18

# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL GATE (multiplier 0.10 – 1.20)
# ─────────────────────────────────────────────────────────────────────────────

LOCATION_TIER1 = {"pune", "noida", "gurugram", "gurgaon"}
LOCATION_TIER2 = {
    "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore", "chennai",
    "trivandrum", "thiruvananthapuram", "kochi", "cochin", "coimbatore",
    "indore", "ahmedabad", "jaipur", "bhubaneswar", "kolkata",
    "nagpur", "visakhapatnam", "vizag", "mysuru", "mysore",
}

def compute_behavioral_gate(cand: dict) -> float:
    sig = cand.get("redrob_signals") or {}
    profile = cand.get("profile") or {}
    delta = 0.0

    # last_active_date
    last_active = parse_date(sig.get("last_active_date"))
    d = days_since(last_active)
    if d <= 30:    delta += 0.15
    elif d <= 90:  delta += 0.05
    elif d <= 180: delta += 0.00
    elif d <= 365: delta -= 0.20
    else:          delta -= 0.40   # JD: down-weight 6-month inactivity

    # open_to_work
    if sig.get("open_to_work_flag"):
        delta += 0.10

    # recruiter response rate
    rr = sig.get("recruiter_response_rate") or 0.0
    if rr >= 0.70:    delta += 0.10
    elif rr >= 0.40:  delta += 0.05
    elif rr <= 0.05:  delta -= 0.15   # JD: explicit 5% rate down-weight

    # avg response time
    rt = sig.get("avg_response_time_hours") or 999
    if rt < 24:    delta += 0.05
    elif rt < 48:  delta += 0.02
    elif rt > 120: delta -= 0.05

    # notice period
    np_days = sig.get("notice_period_days") or 90
    if np_days <= 30:    delta += 0.15
    elif np_days <= 60:  delta += 0.05
    elif np_days <= 90:  delta += 0.00
    elif np_days <= 120: delta -= 0.05
    else:                delta -= 0.10

    # location
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = sig.get("willing_to_relocate") or False
    if any(c in location for c in LOCATION_TIER1):
        delta += 0.20
    elif any(c in location for c in LOCATION_TIER2):
        delta += 0.10
    elif country in ("india", "in"):
        delta += 0.05 if relocate else 0.00
    else:
        delta += -0.10 if relocate else -0.30

    # github
    gh = sig.get("github_activity_score") or -1
    if gh >= 70:    delta += 0.10
    elif gh >= 40:  delta += 0.05
    elif 0 <= gh < 10: delta -= 0.05

    # interview completion
    ic = sig.get("interview_completion_rate") or 0.5
    if ic >= 0.80:  delta += 0.05
    elif ic < 0.30: delta -= 0.05

    # offer acceptance  explicit None check to avoid 0.0 or -1 = -1 bug
    oa = sig.get("offer_acceptance_rate")
    if oa is None:
        oa = -1
    if oa >= 0.70:  delta += 0.05
    elif 0 <= oa < 0.30: delta -= 0.05

    # verification
    v = int(bool(sig.get("verified_email"))) + int(bool(sig.get("verified_phone")))
    if v == 2:   delta += 0.05
    elif v == 0: delta -= 0.05

    # work mode (flexible/hybrid slight bonus, onsite slight minus)
    wm = (sig.get("preferred_work_mode") or "").lower()
    if wm == "flexible": delta += 0.05
    elif wm == "onsite":  delta -= 0.02

    return max(0.50, min(1.20, 1.0 + delta))  # 1.20 = full +20% boost for ideal behavioral profile

# ─────────────────────────────────────────────────────────────────────────────
# SKILL QUALITY SCORE (S3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_quality_score(skills: List[dict], skill_relevance: Dict[str, float],
                                  assessment_scores: Dict[str, float]) -> float:
    """
    Returns 0-100.
    For each skill: relevance × actual × duration_factor
    actual = assessment_score/100 if test exists, else proficiency × 0.75
    """
    total = 0.0
    for sk in skills:
        name = sk.get("name") or ""
        name_lower = name.lower()
        relevance = skill_relevance.get(name_lower, 0.0)
        if relevance < 0.05:  # irrelevant skill → skip
            continue

        prof_val = PROFICIENCY_MAP.get(sk.get("proficiency") or "beginner", 0.25)
        if name in assessment_scores:
            actual = assessment_scores[name] / 100.0
        elif name_lower in assessment_scores:
            actual = assessment_scores[name_lower] / 100.0
        else:
            actual = prof_val * 0.75

        dur = sk.get("duration_months") or 0
        duration_factor = min(1.0, dur / 24.0) if dur > 0 else 0.30

        total += relevance * actual * duration_factor

    # Normalize: a candidate with 8 perfect relevant skills → 100
    return min(100.0, total * (100.0 / 8.0))


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY SCORE (S4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_trajectory_score(cand: dict, career_text: str) -> float:
    """Returns 0-100."""
    career = cand.get("career_history") or []
    skills = cand.get("skills") or []
    sig = cand.get("redrob_signals") or {}
    ct = career_text.lower()

    # 1. Production deployment evidence (0-1)
    prod_hits = sum(1 for m in PRODUCTION_MARKERS if m in ct)
    production_score = min(1.0, prod_hits / 6.0)

    # 2. Pre-LLM skill depth: soft linear ramp over total months (no hard cutoff)
    deep_months = sum(
        (sk.get("duration_months") or 0)
        for sk in skills
        if sk.get("name", "").lower() in PRE_LLM_ML_SKILLS
    )
    pre_llm_score = min(1.0, deep_months / 48.0)  # 48 months (4yr) = full score, matches JD

    # 3. Still coding signal (0-1)
    gh = max(0.0, float(sig.get("github_activity_score") or 0))
    gh_score = gh / 100.0
    current_jobs = [j for j in career if j.get("is_current")]
    recent_desc = (current_jobs[0].get("description") or "") if current_jobs else ""
    coding_kw = ["built", "implemented", "engineered", "wrote", "developed", "trained", "deployed"]
    coding_hits = sum(1 for kw in coding_kw if kw in recent_desc.lower())
    coding_score = min(1.0, coding_hits / 3.0)
    still_coding = 0.5 * gh_score + 0.5 * coding_score

    # 4. Title progression with seniority level (0-1)
    title_scores = []
    sorted_jobs = sorted(career, key=lambda j: j.get("start_date") or "", reverse=True)
    for i, job in enumerate(sorted_jobs):
        recency_weight = 1.0 / (i + 1)
        title = (job.get("title") or "").lower()
        is_ml = classify_title(title) == "ML_ENG"
        is_senior = any(kw in title for kw in SENIOR_TITLE_KW)
        level_multiplier = 1.30 if is_senior else 1.0
        title_scores.append(recency_weight * (level_multiplier if is_ml else 0.0))
    title_progression = min(1.0, sum(title_scores) / 1.5) if title_scores else 0.0

    # 5. Company prestige bonus  data-driven via industry + size
    prestige_bonus = compute_prestige_bonus(sorted_jobs[0]) if sorted_jobs else 0.0

    # Penalty for title chaser
    title_chaser = detect_title_chaser(career)
    penalty = 0.15 if title_chaser else 0.0

    raw = (
        0.35 * production_score
        + 0.25 * pre_llm_score
        + 0.15 * still_coding
        + 0.10 * title_progression
        + prestige_bonus          # already scaled 0.0–0.20
        - penalty
    )
    return max(0.0, min(1.0, raw)) * 100.0


def compute_ml_product_years(career: List[dict]) -> float:
    total = 0.0
    for job in career:
        title_class = classify_title(job.get("title") or "")
        company = (job.get("company") or "").lower()
        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)
        is_large_it_services = job.get("company_size") == "10001+" and (
            job.get("industry") or ""
        ).lower() in ("it services", "consulting", "it consulting")
        if title_class == "ML_ENG" and not is_consulting and not is_large_it_services:
            total += (job.get("duration_months") or 0) / 12.0
    return round(total, 2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute features + indices.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--jd", default=None, help="Path to job_description.docx")
    parser.add_argument("--out", default="artifacts/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load JD ───────────────────────────────────────────────────────
    print("[1/6] Loading JD text...")
    jd_text = read_jd_text(args.jd)
    jd_embed_text = JD_EMBED_TEXT + "\n" + jd_text[:2000]  # cap to avoid noise

    # ── Step 2: Load model ────────────────────────────────────────────────────
    print("[2/6] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Step 3: Encode JD ─────────────────────────────────────────────────────
    print("[3/6] Encoding JD...")
    jd_vector = model.encode([jd_embed_text], show_progress_bar=False)[0]
    jd_vector = jd_vector / np.linalg.norm(jd_vector)  # L2 normalize

    # ── Step 4: First pass  collect texts, candidate data, unique skills ─────
    print("[4/6] First pass: reading candidates.jsonl...")
    candidate_ids: List[str] = []
    career_texts: List[str] = []
    all_candidates: List[dict] = []
    all_skill_names: set = set()

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  reading", unit="cand"):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = cand.get("candidate_id")
            if not cid:
                continue
            candidate_ids.append(cid)
            career = cand.get("career_history") or []
            skills = cand.get("skills") or []
            career_text = get_career_text(career)
            summary = (cand.get("profile") or {}).get("summary") or ""
            embed_text = (summary + " " + career_text).strip()
            career_texts.append(embed_text)
            all_candidates.append(cand)
            for sk in skills:
                name = (sk.get("name") or "").strip().lower()
                if name:
                    all_skill_names.add(name)

    n = len(candidate_ids)
    print(f"  → {n} candidates loaded, {len(all_skill_names)} unique skills")

    # ── Step 5: Encode skills + compute skill_relevance_map ───────────────────
    print("[5/6] Encoding skill names for relevance scoring...")
    skill_name_list = sorted(all_skill_names)
    skill_vectors = model.encode(skill_name_list, batch_size=512,
                                  show_progress_bar=True)
    skill_vectors = sk_normalize(skill_vectors)  # L2 normalize
    # Cosine similarity = dot product after L2 normalization
    similarities = (skill_vectors @ jd_vector).tolist()
    skill_relevance_map: Dict[str, float] = {
        name: float(sim) for name, sim in zip(skill_name_list, similarities)
    }

    # ── Step 5b: Encode all career texts for FAISS ────────────────────────────
    print("  Encoding career texts for FAISS (this takes ~30 min CPU)...")
    BATCH = 256
    embeddings = []
    for i in tqdm(range(0, n, BATCH), desc="  encoding", unit="batch"):
        batch = career_texts[i:i+BATCH]
        vecs = model.encode(batch, show_progress_bar=False)
        embeddings.append(vecs)
    embeddings = np.vstack(embeddings).astype(np.float32)
    embeddings = sk_normalize(embeddings)  # L2 normalize → cosine via dot product

    # ── Step 5c: Build FAISS index ────────────────────────────────────────────
    import faiss
    print("  Building FAISS IndexFlatIP...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss_path = str(out_dir / "faiss.index")
    faiss.write_index(index, faiss_path)
    ids_path = str(out_dir / "faiss_ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(candidate_ids, f)
    print(f"  → FAISS index saved ({n} vectors, dim={dim})")

    # ── Step 5d: Build BM25 index ─────────────────────────────────────────────
    print("  Building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(t) for t in tqdm(career_texts, desc="  tokenizing", unit="doc")]
    bm25 = BM25Okapi(tokenized)
    bm25_path = str(out_dir / "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": candidate_ids}, f)
    print(f"  → BM25 index saved")

    # ── Step 6: Compute all features ──────────────────────────────────────────
    print("[6/6] Computing features for all candidates...")
    features: Dict[str, dict] = {}

    for cand in tqdm(all_candidates, desc="  features", unit="cand"):
        cid = cand["candidate_id"]
        profile = cand.get("profile") or {}
        career = cand.get("career_history") or []
        skills = cand.get("skills") or []
        sig = cand.get("redrob_signals") or {}
        career_text = get_career_text(career)

        assessment_raw = sig.get("skill_assessment_scores") or {}
        # Normalize keys to lowercase for lookup
        assessment_scores = {k.lower(): v for k, v in assessment_raw.items()}
        assessment_scores.update(assessment_raw)  # also keep original keys

        # Boolean flags
        is_honeypot = detect_honeypot(cand)
        title_class = classify_title(profile.get("current_title") or profile.get("headline") or "")
        all_consulting = detect_all_consulting(career)
        cv_speech_primary = detect_cv_speech_primary(skills, career_text)
        pure_research = detect_pure_research(career_text)
        framework_only = detect_framework_only(skills)
        title_chaser = detect_title_chaser(career)
        startup_experience = any(
            (j.get("company_size") or "") in ("1-10", "11-50")
            for j in career
        )

        # Numeric features
        skill_quality_score = compute_skill_quality_score(
            skills, skill_relevance_map, assessment_scores
        )
        trajectory_score = compute_trajectory_score(cand, career_text)
        ml_product_years = compute_ml_product_years(career)
        behavioral_gate = compute_behavioral_gate(cand)

        features[cid] = {
            "is_honeypot": is_honeypot,
            "title_class": title_class,
            "all_consulting": all_consulting,
            "cv_speech_primary": cv_speech_primary,
            "pure_research": pure_research,
            "framework_only": framework_only,
            "title_chaser": title_chaser,
            "startup_experience": startup_experience,
            "skill_quality_score": skill_quality_score,
            "trajectory_score": trajectory_score,
            "ml_product_years": ml_product_years,
            "behavioral_gate": behavioral_gate,
            "yoe": profile.get("years_of_experience") or 0.0,
        }

    features_path = str(out_dir / "features.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(features, f)
    print(f"  → Features saved for {len(features)} candidates")

    # Save JD vector and query tokens for rank.py
    meta = {
        "jd_vector": jd_vector,
        "bm25_query": BM25_QUERY,
        "candidate_ids": candidate_ids,
    }
    with open(str(out_dir / "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n✓ Precompute complete. Artifacts in:", str(out_dir))
    print(f"  features.pkl    {len(features)} candidates")
    print(f"  bm25_index.pkl  BM25 index")
    print(f"  faiss.index     {n} vectors")
    print(f"  faiss_ids.pkl   candidate ID order")
    print(f"  meta.pkl        JD vector + query tokens")


if __name__ == "__main__":
    main()
