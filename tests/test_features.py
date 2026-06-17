"""
test_features.py — Boolean flag detectors + classify_title
TDD vertical slice: one test at a time.
"""
import sys
from pathlib import Path
import copy
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import make_candidate
from precompute import (
    detect_honeypot,
    detect_all_consulting,
    detect_cv_speech_primary,
    detect_pure_research,
    detect_framework_only,
    detect_title_chaser,
    classify_title,
)

# ─── TRACER BULLET ───────────────────────────────────────────────────────────

def test_honeypot_two_current_jobs():
    """Candidate with 2 is_current=True jobs is flagged as honeypot."""
    cand = make_candidate()
    cand["career_history"][1]["is_current"] = True  # now both are current
    assert detect_honeypot(cand) is True

# ─── HONEYPOT ─────────────────────────────────────────────────────────────────

def test_honeypot_expert_zero_duration():
    """expert proficiency with 0 duration_months → honeypot."""
    cand = make_candidate()
    cand["skills"] = [{"name": "PyTorch", "proficiency": "expert", "duration_months": 0}]
    assert detect_honeypot(cand) is True


def test_honeypot_not_triggered_on_valid_candidate():
    """Clean candidate with valid timeline → not a honeypot."""
    cand = make_candidate()
    assert detect_honeypot(cand) is False


def test_honeypot_tenure_exceeds_claimed_experience():
    """Sum of career months >> claimed years → honeypot."""
    cand = make_candidate()
    # Claimed 2 years but career has 60 months (5 years)
    cand["profile"]["years_of_experience"] = 2.0
    cand["career_history"][0]["duration_months"] = 36
    cand["career_history"][1]["duration_months"] = 36
    # 72 months vs 2 * 14 = 28 months ceiling → honeypot
    assert detect_honeypot(cand) is True


# ─── ALL CONSULTING ───────────────────────────────────────────────────────────

def test_all_consulting_when_entire_career_is_consulting():
    """All jobs at TCS/Infosys/Wipro → all_consulting."""
    cand = make_candidate()
    cand["career_history"] = [
        {"company": "TCS", "title": "Software Engineer", "duration_months": 24, "is_current": False},
        {"company": "Infosys", "title": "Analyst", "duration_months": 36, "is_current": True},
    ]
    assert detect_all_consulting(cand["career_history"]) is True


def test_all_consulting_false_when_one_product_company():
    """One non-consulting job → not all_consulting."""
    cand = make_candidate()
    cand["career_history"] = [
        {"company": "TCS", "title": "Software Engineer", "duration_months": 24, "is_current": False},
        {"company": "SomeStartup", "title": "ML Engineer", "duration_months": 30, "is_current": True},
    ]
    assert detect_all_consulting(cand["career_history"]) is False


def test_all_consulting_false_on_empty_career():
    """Empty career history → not all_consulting."""
    assert detect_all_consulting([]) is False


# ─── CV SPEECH PRIMARY ────────────────────────────────────────────────────────

def test_cv_speech_primary_when_cv_dominates():
    """Heavy CV skills, no NLP → cv_speech_primary."""
    cand = make_candidate()
    cand["skills"] = [
        {"name": "Image Classification", "proficiency": "expert", "duration_months": 48},
        {"name": "Computer Vision", "proficiency": "expert", "duration_months": 36},
        {"name": "Object Detection", "proficiency": "advanced", "duration_months": 24},
    ]
    # Replace all career descriptions so no NLP/IR keywords leak in
    cand["career_history"] = [
        {
            "company": "VisionCo",
            "title": "Computer Vision Engineer",
            "duration_months": 48,
            "is_current": True,
            "description": "Built image classification and object detection systems.",
        }
    ]
    career_text = " ".join(j.get("description", "") for j in cand["career_history"])
    assert detect_cv_speech_primary(cand["skills"], career_text) is True


def test_cv_speech_primary_false_when_nlp_in_career():
    """CV skills but NLP keyword in career text → not cv_speech_primary."""
    cand = make_candidate()
    cand["skills"] = [
        {"name": "Image Classification", "proficiency": "expert", "duration_months": 48},
    ]
    cand["career_history"][0]["description"] = "Built image and NLP embedding pipelines."
    career_text = " ".join(j.get("description", "") for j in cand["career_history"])
    assert detect_cv_speech_primary(cand["skills"], career_text) is False


# ─── PURE RESEARCH ────────────────────────────────────────────────────────────

def test_pure_research_when_academic_no_production():
    """3+ research markers and ≤1 production marker → pure_research."""
    text = (
        "Published paper on arxiv about novel dataset. "
        "Presented at NeurIPS conference. Ran ablation study on benchmark."
    )
    assert detect_pure_research(text) is True


def test_pure_research_false_when_production_evidence():
    """Research markers but strong production evidence → not pure_research."""
    text = (
        "Published arxiv paper. Deployed model to production serving 10K users. "
        "Owned the inference API endpoint."
    )
    assert detect_pure_research(text) is False


# ─── FRAMEWORK ONLY ──────────────────────────────────────────────────────────

def test_framework_only_when_langchain_no_pre_llm():
    """LangChain skill, no pre-LLM ML skills → framework_only."""
    cand = make_candidate()
    cand["skills"] = [
        {"name": "LangChain", "proficiency": "intermediate", "duration_months": 10},
        {"name": "FastAPI", "proficiency": "advanced", "duration_months": 24},
    ]
    assert detect_framework_only(cand["skills"]) is True


def test_framework_only_false_when_pytorch_present():
    """LangChain + PyTorch → not framework_only."""
    cand = make_candidate()
    cand["skills"] = [
        {"name": "LangChain", "proficiency": "intermediate", "duration_months": 10},
        {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36},
    ]
    assert detect_framework_only(cand["skills"]) is False


def test_framework_only_false_when_no_framework_skills():
    """No LangChain-type skills at all → not framework_only."""
    cand = make_candidate()
    assert detect_framework_only(cand["skills"]) is False


# ─── TITLE CHASER ─────────────────────────────────────────────────────────────

def test_title_chaser_when_many_short_stints():
    """3 jobs, avg tenure < 18 months → title_chaser."""
    career = [
        {"duration_months": 12, "is_current": False, "company": "A", "title": "ML Eng"},
        {"duration_months": 10, "is_current": False, "company": "B", "title": "ML Eng"},
        {"duration_months": 14, "is_current": True, "company": "C", "title": "ML Eng"},
    ]
    assert detect_title_chaser(career) is True


def test_title_chaser_false_when_fewer_than_three_jobs():
    """Only 2 jobs → guard clause → not title_chaser."""
    career = [
        {"duration_months": 10, "is_current": False, "company": "A", "title": "ML Eng"},
        {"duration_months": 10, "is_current": True, "company": "B", "title": "ML Eng"},
    ]
    assert detect_title_chaser(career) is False


def test_title_chaser_false_when_long_tenures():
    """3 jobs but avg > 18 months → not title_chaser."""
    career = [
        {"duration_months": 30, "is_current": False, "company": "A", "title": "ML Eng"},
        {"duration_months": 28, "is_current": False, "company": "B", "title": "ML Eng"},
        {"duration_months": 24, "is_current": True, "company": "C", "title": "ML Eng"},
    ]
    assert detect_title_chaser(career) is False


# ─── CLASSIFY TITLE ──────────────────────────────────────────────────────────

def test_classify_title_ml_engineer():
    assert classify_title("Senior Machine Learning Engineer") == "ML_ENG"


def test_classify_title_anti_title():
    assert classify_title("Operations Manager") == "ANTI_TITLE"


def test_classify_title_swe():
    assert classify_title("Senior Software Engineer") == "SWE"


def test_classify_title_data_engineer():
    assert classify_title("Data Engineer") == "DATA_ENG"


def test_classify_title_other():
    assert classify_title("Technical Program Manager") == "OTHER"
