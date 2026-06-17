# Redrob Hackathon Ranker

**Redrob AI × Hack2Skill — India Runs Track 1**
Deadline: July 2, 2026

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Step 1 — Precompute (run once, no time limit):**
```bash
python precompute.py \
  --candidates "C:/Users/harip/OneDrive/Desktop/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
  --jd "C:/Users/harip/OneDrive/Desktop/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx" \
  --out artifacts/
```

**Step 2 — Rank (< 5 min, CPU, no network):**
```bash
python rank.py \
  --candidates "C:/Users/harip/OneDrive/.../candidates.jsonl" \
  --artifacts artifacts/ \
  --out submission.csv
```

**Validate:**
```bash
python validate_submission.py submission.csv
```

## Architecture

```
S1 BM25   (0.35) — keyword match on career descriptions
S2 FAISS  (0.25) — MiniLM cosine similarity to JD
S3 Skills (0.20) — skill relevance × actual proficiency × duration
S4 Traj   (0.20) — production evidence + pre-LLM depth + still coding + title

Hard gates: honeypot(0.0), anti-title(0.05), consulting(0.20),
            cv/speech(0.15), pure research(0.15), framework-only(0.20)

Behavioral multiplier (0.10–1.20): last_active, open_to_work,
  recruiter_response_rate, notice_period, location, github, etc.
```

## Scoring
`0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10`
