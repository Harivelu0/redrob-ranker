#!/usr/bin/env python3
"""
run_all.py — Master eval runner. Runs all 3 eval layers in sequence.

Usage:
    python evals/run_all.py \
        --submission submission.csv \
        --artifacts artifacts/ \
        --candidates path/to/candidates.jsonl \
        --sample_submission path/to/sample_submission.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_eval(script: str, extra_args: list) -> int:
    """Run an eval script and return exit code."""
    cmd = [sys.executable, str(ROOT / "evals" / script)] + extra_args
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission",       default="submission.csv")
    parser.add_argument("--artifacts",        default="artifacts/")
    parser.add_argument("--candidates",       required=True)
    parser.add_argument("--sample_submission",default=None)
    args = parser.parse_args()

    base_args = [
        "--submission",  args.submission,
        "--artifacts",   args.artifacts,
        "--candidates",  args.candidates,
    ]
    if args.sample_submission:
        base_args += ["--sample_submission", args.sample_submission]

    print("\n" + "█"*60)
    print("  REDROB RANKER — FULL EVAL SUITE")
    print("█"*60)

    results = {}

    print("\n\n━━━  CHECK 1/3: OUTPUT FORMAT + QUALITY  ━━━")
    results["check_output"]   = run_eval("check_output.py",   base_args)

    print("\n\n━━━  CHECK 2/3: ARTIFACT INTEGRITY + SIGNALS  ━━━")
    results["check_pipeline"] = run_eval("check_pipeline.py", base_args)

    print("\n\n━━━  CHECK 3/3: RANKING QUALITY  ━━━")
    results["check_ranking"]  = run_eval("check_ranking.py",  base_args)

    # Combined verdict
    print("\n" + "█"*60)
    print("  COMBINED VERDICT")
    print("█"*60)
    all_pass = all(code == 0 for code in results.values())
    for name, code in results.items():
        status = "\033[92mPASS\033[0m" if code == 0 else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {name}")

    if all_pass:
        print("\n  ✓ ALL 3 EVALS PASSED — SUBMISSION IS READY")
    else:
        print("\n  ✗ SOME EVALS FAILED — DO NOT SUBMIT YET")
    print("█"*60 + "\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
