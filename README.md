# IndiaRuns AI Redrob India Data & AI Challenge

**Track 1 · Team: Indiaruns AI · CPU-only hybrid ranking pipeline · 100,000 candidates → top 100**

---

## Live Demo

👉 [HuggingFace Space  Interactive Leaderboard + Live Demo](https://huggingface.co/spaces/Haripvelu/redrob-ranker)

- **Tab 1  Leaderboard**: Search and filter top 100 candidates, read full reasoning
- **Tab 2  Live Demo**: Run full ranking pipeline on top 20 candidates
- **Tab 3  How It Works**: Full pipeline walkthrough

---

## Pipeline Overview

Two-stage, CPU-only. No LLM API calls. No external services.

### Stage 1  `precompute.py` (one-time, ~1.7 hours)

| Step | What |
|---|---|
| JD encoding | `BAAI/bge-small-en-v1.5` embeds the job description (with BGE query prefix) |
| Candidate encoding | Same model embeds 100K career texts → FAISS `IndexFlatIP` |
| BM25 index | `BM25Okapi` over career descriptions |
| Feature extraction | Per-candidate: skill quality, trajectory score, behavioral gate, boolean gates |

### Stage 2  `rank.py` (< 5 minutes)

Weighted combination of 4 signals:

| Signal | Weight | What it measures |
|---|---|---|
| BM25 | 0.20 | Keyword match  career text vs JD tokens |
| FAISS cosine | 0.30 | Semantic similarity  candidate embedding vs JD embedding |
| Skill quality | 0.20 | Relevance × verified proficiency × tenure duration |
| Trajectory | 0.30 | Production evidence + pre-LLM depth + seniority + company prestige |

Then multiplied by:
- **YOE fit**  soft penalty below 5yr, taper above 9yr
- **ML product years**  rewards 4yr+ applied ML at product companies (JD requirement)
- **Hard gates**  honeypot ×0.0, all-consulting ×0.20, cv-speech ×0.15, pure-research ×0.15, framework-only ×0.20
- **Behavioral multiplier**  notice period, location, github activity, availability (range: 0.50–1.20)

---

## Trajectory Score  5 Components

```
0.35 × production_score     shipped/deployed/serving/a-b-test markers in career text
0.25 × pre_llm_score        months on pre-LLM ML skills (scikit-learn, PyTorch, XGBoost...)
0.15 × still_coding         github activity + coding keywords in current role
0.10 × title_progression    recent ML titles; Lead/Principal/Staff get 1.30× multiplier
0.20 × prestige_bonus       data-driven: industry + company_size (FAANG override = 0.20)
```

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1  precompute (run once, ~1.7 hours on CPU)
python precompute.py \
    --candidates path/to/candidates.jsonl \
    --jd path/to/job_description.docx \
    --out artifacts/

# Stage 2  rank (< 5 minutes)
python rank.py \
    --candidates path/to/candidates.jsonl \
    --artifacts artifacts/ \
    --out submission.csv

# Validate
python validate_submission.py submission.csv

# Run full eval suite
python evals/validate.py \
    --candidates path/to/candidates.jsonl \
    --submission submission.csv \
    --artifacts artifacts/

# Run tests
python -m pytest tests/ -v
```

---

## Repo Structure

```
redrob-ranker/
├── precompute.py              # Stage 1: one-time preprocessing (~1.7 hr)
├── rank.py                    # Stage 2: ranking (< 5 min)
├── explain.py                 # Reasoning generator for CSV column
├── app.py                     # Streamlit demo (HuggingFace Space)
├── submission.csv             # Final submission: top 100 ranked candidates
├── submission_metadata.yaml   # Hackathon metadata
├── requirements.txt           # Full dependencies
├── hf_requirements.txt        # HuggingFace Space dependencies
├── spot_check.py              # Manual spot-check tool
├── artifacts/
│   ├── features.pkl           # Per-candidate precomputed features (9.4 MB)
│   ├── faiss_ids.pkl          # FAISS index candidate ID mapping (1.4 MB)
│   └── meta.pkl               # JD vector + BM25 query tokens (1.4 MB)
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_gates.py
│   └── test_explain.py
└── evals/
    ├── check_output.py
    ├── check_pipeline.py
    ├── check_ranking.py
    └── validate.py
```

---

## Models Used

| Model | Source | Usage |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | HuggingFace sentence-transformers | Text embedding for FAISS + skill matching |

No OpenAI · No Anthropic · No cloud inference · Runs fully on CPU

---

## Test Results

```
48 passed in 7.4s
```
