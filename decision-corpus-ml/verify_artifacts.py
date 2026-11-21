"""Reload both saved models and list failed decisions. Usage: python verify_artifacts.py [artifacts_dir]"""
from pathlib import Path
from collections import defaultdict
import json
import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

p = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
SentenceTransformer(str(p/"bi_encoder"), device="cpu", local_files_only=True)
ce = AutoModelForSequenceClassification.from_pretrained(p/"cross_encoder_arbitrator", local_files_only=True)
AutoTokenizer.from_pretrained(p/"cross_encoder_arbitrator", local_files_only=True)
assert ce.config.num_labels == 2
report = json.loads((p/"v1_ml_eval_report.json").read_text())
rows = report["predictions"]
assert len(rows) == 42
groups = defaultdict(list)
for r in rows:
    if r["query_type"] == "arbitration_pair":
        groups[r["pair_group"]].append(r)
    if r["abstained"] != (r["query_type"] == "abstention"):
        print("Scope error", r["query_id"], r["primary"], max(r["abstention"].values()), max(r["relevance"].values()))
assert len(groups) == 14 and all(len(g) == 2 for g in groups.values())
for group, pair in groups.items():
    if any(r["primary"] != r["gold_primary_id"] for r in pair):
        for r in pair:
            gold = r["gold_primary_id"]
            print(group, r["query_id"], "gold", gold, "primary", r["primary"], "seed hit", gold in r["seeds"],
                  "pool hit", gold in r["pool"], "raw relevance", r["relevance"].get(gold), "final top three", r["top3"])
print(json.dumps(report["metrics"], indent=2))
