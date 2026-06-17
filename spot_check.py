import csv, json, random

CANDIDATES = "C:/Users/harip/OneDrive/Desktop/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"

with open("submission.csv") as f:
    rows = list(csv.DictReader(f))

sample = random.sample(rows[:20], 5)
target_ids = {r["candidate_id"] for r in sample}

cands = {}
with open(CANDIDATES) as f:
    for line in f:
        c = json.loads(line.strip())
        if c["candidate_id"] in target_ids:
            cands[c["candidate_id"]] = c

for row in sample:
    cid = row["candidate_id"]
    c = cands.get(cid, {})
    p = c.get("profile", {})
    skills = [s["name"] for s in (c.get("skills") or [])[:5]]
    print("=" * 60)
    print("Rank:", row["rank"], "|", cid)
    print("ACTUAL title  :", p.get("current_title"))
    print("ACTUAL company:", p.get("current_company"))
    print("ACTUAL yoe    :", p.get("years_of_experience"))
    print("ACTUAL skills :", skills)
    print("CSV reasoning :", row["reasoning"][:250])
    print()
