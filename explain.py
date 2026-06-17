#!/usr/bin/env python3
"""
explain.py — Generates 1-2 sentence reasoning for each ranked candidate.
No hallucination. References specific facts from the candidate profile.
"""

from typing import Optional


PROFICIENCY_MAP = {
    "beginner": 0.25, "intermediate": 0.60, "advanced": 0.85, "expert": 1.00,
}

# Skills that are actually relevant to the JD
JD_RELEVANT_SKILLS = {
    "nlp", "natural language processing", "information retrieval", "vector search",
    "embeddings", "semantic search", "ranking", "recommendation", "bert", "transformers",
    "text classification", "search", "retrieval", "reranking", "dense retrieval",
    "sparse retrieval", "bm25", "faiss", "learning to rank", "pytorch", "tensorflow",
    "sklearn", "scikit-learn", "xgboost", "lightgbm", "deep learning", "python",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "mlflow", "kubeflow", "feature engineering", "lora", "fine-tuning llms",
    "weights & biases", "bentoml", "apache beam", "recommender system",
    "collaborative filtering", "matrix factorization", "gradient boosting",
}


def _top_skills(skills: list, n: int = 3) -> str:
    """Top N JD-relevant skills by proficiency × duration. Falls back to any skill."""
    scored = []
    for sk in skills:
        name = sk.get("name") or ""
        if name.lower() not in JD_RELEVANT_SKILLS:
            continue
        p = PROFICIENCY_MAP.get(sk.get("proficiency") or "beginner", 0.25)
        d = min(1.0, (sk.get("duration_months") or 0) / 24.0)
        scored.append((p * d, name))
    if not scored:
        for sk in skills:
            p = PROFICIENCY_MAP.get(sk.get("proficiency") or "beginner", 0.25)
            d = min(1.0, (sk.get("duration_months") or 0) / 24.0)
            scored.append((p * d, sk.get("name") or ""))
    top = [name for _, name in sorted(scored, reverse=True)[:n] if name]
    return ", ".join(top) if top else "general software engineering"


def _most_recent_role(career: list) -> Optional[dict]:
    current = [j for j in career if j.get("is_current")]
    if current:
        return current[0]
    if career:
        return sorted(career, key=lambda j: j.get("start_date") or "", reverse=True)[0]
    return None


def _location_str(profile: dict, sig: dict) -> str:
    loc = profile.get("location") or profile.get("country") or "unknown location"
    relocate = sig.get("willing_to_relocate")
    np_days = sig.get("notice_period_days")
    parts = [loc]
    if np_days is not None and np_days <= 30:
        parts.append(f"immediately available ({np_days}d notice)")
    elif np_days is not None:
        parts.append(f"{np_days}d notice")
    if relocate:
        parts.append("open to relocation")
    return "; ".join(parts)


def generate_reason(cand: dict, feat: dict, rank: int, score: float, seen_descs: set = None) -> str:
    """Generate 1-2 sentence reasoning referencing specific candidate facts."""
    if not cand:
        return f"Ranked {rank} based on composite scoring across skills, career trajectory, and availability."

    profile = cand.get("profile") or {}
    career  = cand.get("career_history") or []
    skills  = cand.get("skills") or []
    sig     = cand.get("redrob_signals") or {}

    yoe     = profile.get("years_of_experience") or 0
    title   = profile.get("current_title") or profile.get("headline") or "Engineer"
    company = profile.get("current_company") or "current company"
    top_sk  = _top_skills(skills)
    loc_str = _location_str(profile, sig)
    ml_yrs  = feat.get("ml_product_years") or 0.0
    gh_score = sig.get("github_activity_score") or -1

    recent = _most_recent_role(career)
    recent_snippet = ""
    if recent:
        desc = (recent.get("description") or "").strip()
        if desc:
            if seen_descs is not None and desc in seen_descs:
                desc = ""  # duplicate — skip recent snippet
            else:
                if seen_descs is not None:
                    seen_descs.add(desc)
                if len(desc) > 400:
                    cut = desc[:400]
                    last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
                    if last_stop > 80:
                        desc = desc[:last_stop + 1]
                    else:
                        desc = cut.rsplit(" ", 1)[0] + "..."
                recent_snippet = f" Recent: {desc}"

    # Flags for concerns
    concerns = []
    if yoe > 12.0:
        concerns.append(f"experience significantly above JD range ({yoe:.1f}yr vs 5-9yr target)")
    if feat.get("all_consulting"):
        concerns.append("career entirely in IT services")
    if feat.get("framework_only"):
        concerns.append("no pre-LLM ML depth detected")
    if feat.get("pure_research"):
        concerns.append("limited production deployment evidence")
    if feat.get("title_chaser"):
        concerns.append("high job-hop rate (avg tenure <18 months)")
    if feat.get("cv_speech_primary"):
        concerns.append("primary domain is CV/speech, not NLP/IR")
    concern_str = ("; concern: " + "; ".join(concerns)) if concerns else ""

    if rank <= 10:
        ml_str = f"{ml_yrs:.1f}yr at ML product companies" if ml_yrs >= 1 else ""
        gh_str = f" github_activity={gh_score:.0f}" if gh_score >= 0 else ""
        return (
            f"{yoe:.1f}yr exp, currently {title} at {company}; "
            f"top skills: {top_sk}; {ml_str}{gh_str}; {loc_str}{concern_str}."
            + recent_snippet
        ).strip()

    elif rank <= 50:
        return (
            f"{title} ({yoe:.1f}yr) with strengths in {top_sk}; {loc_str}{concern_str}."
            + recent_snippet
        ).strip()

    else:
        # Rank 51-100: specific about gaps, honest about concerns
        gap_parts = []
        if not feat.get("ml_product_years") or feat.get("ml_product_years", 0) < 2:
            gap_parts.append("limited ML product company tenure")
        if feat.get("title_chaser") and "high job-hop rate" not in " ".join(concerns):
            gap_parts.append("high job-hop rate")
        if feat.get("pure_research") and "production deployment" not in " ".join(concerns):
            gap_parts.append("research-heavy profile")
        if feat.get("framework_only") and "pre-LLM" not in " ".join(concerns):
            gap_parts.append("no pre-LLM ML depth")
        if feat.get("cv_speech_primary") and "CV/speech" not in " ".join(concerns):
            gap_parts.append("CV/speech primary domain")
        if not gap_parts:
            gap_parts.append("lower semantic match to JD on retrieval/ranking signals")
        # Combine concerns and gaps — don't swallow one when both exist
        all_issues = []
        if concerns:
            all_issues.append("concern: " + "; ".join(concerns))
        if gap_parts:
            all_issues.append("gap: " + "; ".join(gap_parts))
        gap_str = ("; " + "; ".join(all_issues)) if all_issues else ""
        return (
            f"{title} ({yoe:.1f}yr), {loc_str}{gap_str}. "
            f"Relevant skills: {top_sk}."
        ).strip()
