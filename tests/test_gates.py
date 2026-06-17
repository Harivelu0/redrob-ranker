"""
test_gates.py  Gate multipliers, minmax normalization, behavioral gate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import make_candidate
from rank import gate_multiplier, minmax_norm
from precompute import compute_behavioral_gate


def _neutral_candidate():
    """
    Candidate with gate == 1.0  all signals at zero-contribution midpoints.
    Use for comparison tests that need to isolate the effect of a single variable.
    """
    cand = make_candidate()
    sig = cand["redrob_signals"]
    cand["profile"]["location"] = "Coimbatore"  # other India, not tier 1/2
    cand["profile"]["country"] = "India"
    sig["willing_to_relocate"] = False
    sig["last_active_date"] = "2026-03-04"       # ~101 days → d <= 180 → +0
    sig["open_to_work_flag"] = False             # 0
    sig["recruiter_response_rate"] = 0.30        # 0.05 < x < 0.40 → 0
    sig["avg_response_time_hours"] = 80          # 48 < x < 120 → 0
    sig["notice_period_days"] = 90               # 60 < x <= 90 → 0
    sig["github_activity_score"] = 30            # 10 <= x < 40 → 0
    sig["interview_completion_rate"] = 0.60      # 0.30 <= x < 0.80 → 0
    sig["offer_acceptance_rate"] = -1            # no history → 0
    sig["verified_email"] = True
    sig["verified_phone"] = False                # only one → 0
    sig["preferred_work_mode"] = "hybrid"        # 0
    return cand


# ─── GATE MULTIPLIER ─────────────────────────────────────────────────────────

def test_gate_honeypot_zeroes_score():
    feat = {"is_honeypot": True}
    assert gate_multiplier(feat, bm25_score_norm=0.5) == 0.0


def test_gate_all_consulting():
    feat = {"is_honeypot": False, "all_consulting": True}
    assert gate_multiplier(feat, bm25_score_norm=0.5) == 0.20


def test_gate_cv_speech():
    feat = {"is_honeypot": False, "cv_speech_primary": True}
    assert gate_multiplier(feat, bm25_score_norm=0.5) == 0.15


def test_gate_pure_research():
    feat = {"is_honeypot": False, "pure_research": True}
    assert gate_multiplier(feat, bm25_score_norm=0.5) == 0.15


def test_gate_framework_only():
    feat = {"is_honeypot": False, "framework_only": True}
    assert gate_multiplier(feat, bm25_score_norm=0.5) == 0.20


def test_gate_anti_title_low_bm25():
    """Anti-title + BM25 < 0.05 → severely discounted."""
    feat = {"is_honeypot": False, "title_class": "ANTI_TITLE"}
    assert gate_multiplier(feat, bm25_score_norm=0.02) == 0.05


def test_gate_anti_title_high_bm25_passes():
    """Anti-title but BM25 >= 0.05 → not penalised (rare edge case)."""
    feat = {"is_honeypot": False, "title_class": "ANTI_TITLE"}
    assert gate_multiplier(feat, bm25_score_norm=0.10) == 1.0


def test_gate_clean_candidate():
    feat = {
        "is_honeypot": False, "all_consulting": False, "cv_speech_primary": False,
        "pure_research": False, "framework_only": False, "title_class": "ML_ENG",
    }
    assert gate_multiplier(feat, bm25_score_norm=0.8) == 1.0


# ─── MINMAX NORMALIZATION ─────────────────────────────────────────────────────

def test_minmax_norm_basic():
    scores = {"a": 0.0, "b": 0.5, "c": 1.0}
    result = minmax_norm(scores)
    assert result["a"] == 0.0
    assert result["c"] == 1.0
    assert abs(result["b"] - 0.5) < 1e-6


def test_minmax_norm_single_value_returns_half():
    """All equal values → returns 0.5 for all (no division by zero)."""
    scores = {"a": 7.0, "b": 7.0, "c": 7.0}
    result = minmax_norm(scores)
    assert all(v == 0.5 for v in result.values())


def test_minmax_norm_preserves_order():
    scores = {"low": 1.0, "mid": 5.0, "high": 10.0}
    result = minmax_norm(scores)
    assert result["low"] < result["mid"] < result["high"]


# ─── BEHAVIORAL GATE ─────────────────────────────────────────────────────────

def test_behavioral_gate_active_pune_candidate():
    """Recently active, Pune, open to work, short notice → gate >= 1.0."""
    cand = make_candidate()
    gate = compute_behavioral_gate(cand)
    assert gate >= 1.0


def test_behavioral_gate_inactive_low_response():
    """Many negative signals from neutral base → gate clamps at minimum (0.50)."""
    cand = _neutral_candidate()
    sig = cand["redrob_signals"]
    sig["last_active_date"] = "2025-05-01"      # 408 days → -0.40
    sig["recruiter_response_rate"] = 0.04       # <= 0.05 → -0.15
    sig["notice_period_days"] = 120             # > 90 → -0.05 (graduated penalty)
    sig["avg_response_time_hours"] = 200        # > 120 → -0.05
    sig["github_activity_score"] = 3            # < 10 → -0.05
    sig["interview_completion_rate"] = 0.2      # < 0.3 → -0.05
    sig["offer_acceptance_rate"] = 0.2          # < 0.3 → -0.05
    sig["verified_email"] = False               # neither verified → -0.05
    # delta ≈ -0.85 → gate = max(0.50, 0.15) = 0.50 (floor changed from 0.10 → 0.50)
    gate = compute_behavioral_gate(cand)
    assert gate <= 0.60  # 0.50 floor + small margin for signal variance


def test_behavioral_gate_always_clamps_to_minimum():
    """Worst-case candidate never goes below 0.10."""
    cand = make_candidate()
    sig = cand["redrob_signals"]
    sig["last_active_date"] = "2024-01-01"
    sig["open_to_work_flag"] = False
    sig["recruiter_response_rate"] = 0.01
    sig["avg_response_time_hours"] = 500
    sig["notice_period_days"] = 180
    cand["profile"]["country"] = "USA"
    sig["willing_to_relocate"] = False
    sig["github_activity_score"] = 0
    sig["interview_completion_rate"] = 0.1
    sig["offer_acceptance_rate"] = 0.1
    sig["verified_email"] = False
    sig["verified_phone"] = False
    gate = compute_behavioral_gate(cand)
    assert gate >= 0.10


def test_behavioral_gate_clamps_to_maximum():
    """Best-case candidate never exceeds 1.20."""
    cand = make_candidate()
    gate = compute_behavioral_gate(cand)
    assert gate <= 1.20


def test_behavioral_gate_outside_india_no_relocate_penalized():
    """Outside India (no relocate) scores lower than other-India city with same signals."""
    cand_india = _neutral_candidate()   # other India, no relocate → +0 location delta
    gate_india = compute_behavioral_gate(cand_india)

    cand_abroad = _neutral_candidate()
    cand_abroad["profile"]["location"] = "San Francisco"
    cand_abroad["profile"]["country"] = "USA"
    # willing_to_relocate already False from _neutral_candidate
    gate_abroad = compute_behavioral_gate(cand_abroad)

    assert gate_abroad < gate_india


def test_behavioral_gate_low_notice_period_bonus():
    """0-day notice has higher gate than 120-day notice (from same neutral base)."""
    cand_fast = _neutral_candidate()
    cand_fast["redrob_signals"]["notice_period_days"] = 0    # <= 30 → +0.15
    gate_fast = compute_behavioral_gate(cand_fast)

    cand_slow = _neutral_candidate()
    cand_slow["redrob_signals"]["notice_period_days"] = 120  # > 90 → -0.10
    gate_slow = compute_behavioral_gate(cand_slow)

    assert gate_fast > gate_slow
