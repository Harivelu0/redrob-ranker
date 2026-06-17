"""
Shared fixtures for redrob-ranker tests.
"""
import sys
from pathlib import Path
import pytest

# Add project root to path so we can import precompute, rank, explain
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_candidate(override: dict = None) -> dict:
    """
    Base valid candidate: Senior ML Engineer in Pune.
    Override any field via the override dict (shallow-merged at top level).
    """
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Senior ML Engineer",
            "current_title": "Senior ML Engineer",
            "current_company": "GoodStartup",
            "current_company_size": "51-200",
            "current_industry": "Technology",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "summary": "ML engineer with 6 years of production ML experience.",
        },
        "career_history": [
            {
                "company": "GoodStartup",
                "title": "Senior ML Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 29,
                "is_current": True,
                "industry": "Technology",
                "company_size": "51-200",
                "description": (
                    "Built and deployed production embedding retrieval system. "
                    "Owned the vector search pipeline end-to-end, serving 50K queries/day. "
                    "Ran A/B tests to validate NDCG improvements."
                ),
            },
            {
                "company": "AnotherTech",
                "title": "ML Engineer",
                "start_date": "2019-06-01",
                "end_date": "2021-12-31",
                "duration_months": 30,
                "is_current": False,
                "industry": "E-commerce",
                "company_size": "201-500",
                "description": "Trained and shipped ranking models for product search.",
            },
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 72},
            {"name": "NLP", "proficiency": "advanced", "endorsements": 25, "duration_months": 30},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-10",
            "open_to_work_flag": True,
            "profile_views_received_30d": 40,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.65,
            "avg_response_time_hours": 18.0,
            "skill_assessment_scores": {"Python": 88.0, "PyTorch": 75.0},
            "connection_count": 400,
            "endorsements_received": 60,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 72.0,
            "search_appearance_30d": 200,
            "saved_by_recruiters_30d": 8,
            "interview_completion_rate": 0.85,
            "offer_acceptance_rate": 0.75,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    if override:
        base.update(override)
    return base
