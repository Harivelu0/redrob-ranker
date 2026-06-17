"""
test_explain.py  explain.py output correctness.
Tests that reasoning is fact-anchored, rank-appropriate, and non-identical.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import make_candidate
from explain import generate_reason


def _features(overrides=None):
    base = {
        "is_honeypot": False,
        "title_class": "ML_ENG",
        "all_consulting": False,
        "cv_speech_primary": False,
        "pure_research": False,
        "framework_only": False,
        "title_chaser": False,
        "startup_experience": True,
        "skill_quality_score": 75.0,
        "trajectory_score": 80.0,
        "ml_product_years": 4.5,
        "behavioral_gate": 1.15,
    }
    if overrides:
        base.update(overrides)
    return base


# ─── RANK 1-10 ───────────────────────────────────────────────────────────────

def test_rank1_mentions_company():
    cand = make_candidate()
    reason = generate_reason(cand, _features(), rank=1, score=95.0)
    assert "GoodStartup" in reason


def test_rank1_mentions_years_of_experience():
    cand = make_candidate()
    reason = generate_reason(cand, _features(), rank=1, score=95.0)
    assert "6" in reason   # yoe = 6.0


def test_rank1_mentions_top_skills():
    cand = make_candidate()
    reason = generate_reason(cand, _features(), rank=1, score=95.0)
    # Top skill by proficiency × duration: Python (expert × 72mo) should appear
    assert any(skill in reason for skill in ["Python", "PyTorch", "FAISS", "NLP"])


# ─── RANK 51-100 ─────────────────────────────────────────────────────────────

def test_rank75_mentions_gap_when_consulting():
    cand = make_candidate()
    feat = _features({"all_consulting": True})
    reason = generate_reason(cand, feat, rank=75, score=22.0)
    assert "concern" in reason.lower() or "it services" in reason.lower() or "consulting" in reason.lower()


def test_rank75_mentions_lower_match():
    cand = make_candidate()
    reason = generate_reason(cand, _features(), rank=75, score=22.0)
    assert any(kw in reason.lower() for kw in
               ["lower match", "limited", "gaps", "gap:", "lower semantic"])


# ─── NO HALLUCINATION ────────────────────────────────────────────────────────

def test_empty_candidate_does_not_crash():
    """generate_reason should handle an empty cand dict without raising."""
    reason = generate_reason({}, _features(), rank=50, score=50.0)
    assert isinstance(reason, str)
    assert len(reason) > 0


def test_no_two_candidates_get_identical_reasoning():
    """Two different candidates must produce different reasoning strings."""
    cand1 = make_candidate()

    cand2 = make_candidate()
    cand2["candidate_id"] = "CAND_0000002"
    cand2["profile"]["current_company"] = "DifferentCorp"
    cand2["profile"]["years_of_experience"] = 9.5
    cand2["profile"]["location"] = "Hyderabad"

    reason1 = generate_reason(cand1, _features(), rank=1, score=90.0)
    reason2 = generate_reason(cand2, _features(), rank=2, score=88.0)

    assert reason1 != reason2


# ─── CONCERN FLAGGING ────────────────────────────────────────────────────────

def test_framework_only_concern_in_rank1_output():
    """Even top-ranked candidate has framework_only concern noted."""
    cand = make_candidate()
    feat = _features({"framework_only": True})
    reason = generate_reason(cand, feat, rank=1, score=91.0)
    assert "concern" in reason.lower() or "pre-llm" in reason.lower() or "framework" in reason.lower()


def test_title_chaser_concern_noted():
    cand = make_candidate()
    feat = _features({"title_chaser": True})
    reason = generate_reason(cand, feat, rank=5, score=82.0)
    assert "concern" in reason.lower() or "hop" in reason.lower() or "tenure" in reason.lower()
