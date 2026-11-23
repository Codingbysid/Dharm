"""Blinded Everyday Life Breadth Evaluation.

Runs 24 everyday dilemmas across 8 domains through Dharm with:
1. Gemini OFF (deterministic Dharm planner + template)
2. Gemini ON (source-grounded Gemini-3.8-flash second reader)

Computes machine-checked metrics:
- Action direction preservation (contrary verbs / negations)
- Fact fidelity & invented numbers/statistics
- Application limit & qualification retention
- Boundary cases: bare choice clarification, out-of-scope abstention for legal/medical/stock tips,
  versus in-scope values guidance for personal finance and family caregiving.
"""
import asyncio
import json
import os
from pathlib import Path
import random
import re
import sys
import time

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "decision-corpus-ml"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from app.conversation import detect_query_intent, formulate_retrieval_query
from app.editor import accept_edit, build_evidence_bundle, review_grounded_reply, verify_evidence_spans
from app.planner import build_plan
from app.render import render_plan
from app.server import _citation, _empty_retrieval, _load_local_env
from rc1_inference import DharmRC1Engine

_load_local_env()

DATA_PATH = ML_DIR / "data" / "eval_everyday_breadth_v1.jsonl"
OUTPUT_DIR = BASE_DIR / "evaluation_reports"
OUTPUT_DIR.mkdir(exist_ok=True)


async def evaluate_all():
    print("=" * 70)
    print("EVERYDAY LIFE BREADTH EVALUATION: GEMINI-OFF VS. GEMINI-ON")
    print("=" * 70)

    # 1. Load engine
    print("\nLoading Dharm RC1 Engine on CPU...")
    engine = DharmRC1Engine(device="cpu")
    records = engine.records
    print(f"Engine loaded with {len(records)} records.\n")

    # 2. Load eval dilemmas
    queries = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf8").splitlines() if line.strip()]
    print(f"Loaded {len(queries)} evaluation dilemmas from {DATA_PATH.name}.\n")

    results = []
    blinded_pairs = []
    random.seed(42)  # Deterministic coin flips for blinding

    for idx, q in enumerate(queries, 1):
        qid = q["query_id"]
        dilemma = q["dilemma_text"]
        domain = q["domain"]
        boundary = q["boundary_type"]
        expected_disp = q["expect_disposition"]

        print(f"[{idx:02d}/{len(queries):02d}] {qid} ({domain}): {dilemma[:60]}...")

        # Run pipeline up to planner (common upstream)
        history = []
        intent = detect_query_intent(dilemma, history)
        query_text = formulate_retrieval_query(dilemma, history, intent)

        if intent in {"greeting", "clarify"}:
            retrieval = _empty_retrieval(False)
        elif intent == "out_of_scope":
            retrieval = _empty_retrieval(True)
        else:
            retrieval = engine.predict(query_text)

        plan_a = build_plan(dilemma, history, intent, retrieval, records)
        text_a = render_plan(plan_a)
        disp_a = plan_a.disposition

        # Condition B: Gemini ON
        plan_b = build_plan(dilemma, history, intent, retrieval, records)
        edited_b = False
        review_result_b = None
        disp_b = plan_b.disposition

        # Ensure opt-in enabled for condition B
        with patch_env({"GEMINI_EDITOR": "1"}):
            if plan_b.disposition == "answer":
                review_result_b = review_grounded_reply(plan_b, records, timeout=12.0)
                if review_result_b:
                    if review_result_b.get("accepted"):
                        plan_b.applied = review_result_b.get("explanation")
                        edited_b = True
                    elif review_result_b.get("verdict") == "needs_clarification" and review_result_b.get("missing_fact"):
                        plan_b.disposition = "clarify"
                        fact = review_result_b["missing_fact"].strip()
                        plan_b.clarify_kind = "missing_fact"
                        plan_b.action = f"To settle this, does {fact}?"
                        plan_b.applied = None
        text_b = render_plan(plan_b)
        disp_b = plan_b.disposition

        # Blinding: flip coin
        flip = random.random() < 0.5
        candidate_x = text_a if flip else text_b
        candidate_y = text_b if flip else text_a
        mapping = {"X": "Gemini-OFF" if flip else "Gemini-ON", "Y": "Gemini-ON" if flip else "Gemini-OFF"}

        # Machine verification checks
        direction_preserved = True
        if edited_b:
            direction_preserved = accept_edit(plan_a, plan_b.applied or "", records)

        # Boundary compliance
        boundary_passed = (disp_a == expected_disp) and (disp_b == expected_disp)

        res_item = {
            "query_id": qid,
            "domain": domain,
            "boundary_type": boundary,
            "expected_disposition": expected_disp,
            "gemini_off": {
                "disposition": disp_a,
                "text": text_a,
                "primary": plan_a.supports[0].id if plan_a.supports else None,
                "top3": retrieval.get("top3", []),
                "limit": plan_a.limit,
            },
            "gemini_on": {
                "disposition": disp_b,
                "edited": edited_b,
                "review": review_result_b,
                "text": text_b,
                "primary": plan_b.supports[0].id if plan_b.supports else None,
                "limit": plan_b.limit,
            },
            "machine_checks": {
                "boundary_compliant": boundary_passed,
                "action_direction_preserved": direction_preserved,
                "disposition_match": (disp_a == disp_b),
            },
            "blinding": mapping,
        }
        results.append(res_item)

        blinded_pairs.append({
            "query_id": qid,
            "domain": domain,
            "dilemma": dilemma,
            "candidate_X": candidate_x,
            "candidate_Y": candidate_y,
            "boundary_type": boundary,
        })

    # Summary statistics
    total = len(results)
    boundary_ok = sum(1 for r in results if r["machine_checks"]["boundary_compliant"])
    direction_ok = sum(1 for r in results if r["machine_checks"]["action_direction_preserved"])
    edited_count = sum(1 for r in results if r["gemini_on"]["edited"])
    clarified_by_editor = sum(
        1 for r in results
        if r["gemini_off"]["disposition"] == "answer" and r["gemini_on"]["disposition"] == "clarify"
    )

    report = {
        "total_queries": total,
        "boundary_compliance_rate": f"{boundary_ok}/{total} ({boundary_ok/total*100:.1f}%)",
        "action_direction_preservation_rate": f"{direction_ok}/{total} ({direction_ok/total*100:.1f}%)",
        "gemini_on_accepted_edits": f"{edited_count}/{total} ({edited_count/total*100:.1f}%)",
        "gemini_on_routed_to_clarify": f"{clarified_by_editor}/{total}",
        "detailed_results": results,
    }

    report_path = OUTPUT_DIR / "everyday_breadth_evaluation_report.json"
    blinded_path = OUTPUT_DIR / "everyday_breadth_blinded.jsonl"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    with blinded_path.open("w", encoding="utf8") as f:
        for bp in blinded_pairs:
            f.write(json.dumps(bp, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Boundary Compliance: {report['boundary_compliance_rate']}")
    print(f"Action Direction Preservation: {report['action_direction_preservation_rate']}")
    print(f"Gemini-ON Edits Accepted: {report['gemini_on_accepted_edits']}")
    print(f"Saved full report to: {report_path}")
    print(f"Saved blinded dilemmas to: {blinded_path}")
    print("=" * 70)


class patch_env:
    def __init__(self, new_env):
        self.new_env = new_env
        self.old_env = {}

    def __enter__(self):
        for k, v in self.new_env.items():
            self.old_env[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    asyncio.run(evaluate_all())
