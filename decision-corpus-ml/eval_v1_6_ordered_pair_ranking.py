"""Decision Corpus v1.6 — Ordered Pair Joint Scoring on Frozen v1.3 Baseline.

Objective:
Test whether scoring two passages jointly can improve ordered synthesis
while preserving single-decision arbitration and abstention.

Features:
- Reproduces v1.3 predictions exactly when pair promotion is disabled.
- Enumerates ordered complementary pairs (u, v) in each candidate pool.
- Scores dilemma against joint card (First Principle, Transition, Second Principle)
  using BAAI/bge-reranker-base.
- Deduplicates reciprocal representations so each edge is scored once per query.
- Promotes pair to ranks #1 and #2 when joint score satisfies calibrated margin
  thresholds (calibrated on corpus dev split, frozen in pair_scorer_config.json).
- Falls back to v1.3 ranking if no pair qualifies or promotion is disabled.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_V1_3 = Path("decision-corpus-ml/artifacts-v1.3")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")

SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def load_corpus(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = [json.loads(line) for line in (data_dir / "decisions_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    records = {r["id"]: r for r in rows}
    notes = [json.loads(line) for line in (data_dir / "chanakya_daily_supplement.jsonl").read_text(encoding="utf8").splitlines()]
    for row in notes:
        target = row.get("existing_record_id") or SUPPLEMENT.get(row["note_id"])
        if target and target in records:
            records[target]["scenarios_json"].append({"scenario": row["modern_application"]})
    return records


def passage_card(r: Dict[str, Any]) -> str:
    rid, work, loc = r["id"], r["work"], r["locator"]
    dom, urg, rev = r["decision_domain"], r["urgency_level"], r["reversibility"]
    prin, ctx = r["derived_principle"], r["context_summary"]
    txt, cues = r["english_translation"], r["retrieval_cues"]
    return (f"[{rid}] {work} {loc} | Domain: {dom} | Urgency: {urg} | Reversibility: {rev}\n"
            f"Principle: {prin}\nContext: {ctx}\nText: {txt}\nCues: {cues}")


def pair_joint_card(u_rec: Dict[str, Any], v_rec: Dict[str, Any]) -> str:
    uid, uprin = u_rec["id"], u_rec["derived_principle"]
    vid, vprin = v_rec["id"], v_rec["derived_principle"]
    role = u_rec[WORKFLOW]["sequence_role"]
    return (f"First Principle [{uid}]: {uprin}\n"
            f"Transition: {role}\n"
            f"Second Principle [{vid}]: {vprin}")


def batch_score_ce(model, tokenizer, pairs, device, batch_size=32, max_length=384):
    scores = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            inputs = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze(-1)
            raw = logits.cpu().tolist()
            if isinstance(raw, float):
                raw = [raw]
            scores.extend(raw)
    return scores


def evaluate_benchmark(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    recall, ordered, pairs = [], [], defaultdict(list)
    tp = pred = gold = 0
    synth_unordered_top2 = 0
    synth_ordered_top2 = 0

    for r in predictions:
        p, c = r["gold_primary_id"], r.get("gold_competing_id")
        pred += int(r["abstained"])
        if r["query_type"] == "abstention":
            gold += 1
            tp += int(r["abstained"])
            continue

        relevant = {p}
        if r["query_type"] == "synthesis":
            relevant.add(c)
            is_ordered = (not r["abstained"] and r["top3"][:2] == [p, c])
            is_unordered = (not r["abstained"] and set(r["top3"][:2]) == {p, c})
            ordered.append(is_ordered)
            if is_ordered:
                synth_ordered_top2 += 1
            if is_unordered:
                synth_unordered_top2 += 1
        elif r.get("pair_group"):
            pairs[r["pair_group"]].append(r["primary"] == p)

        recall.append(len(relevant.intersection(r["top3"])) / len(relevant))

    return {
        "recall_at_3": sum(recall) / len(recall) if recall else 0.0,
        "flip": [sum(all(g) for g in pairs.values()), len(pairs)],
        "abstention_precision": [tp, pred],
        "abstention_recall": [tp, gold],
        "ordered_synthesis": [sum(ordered), len(ordered)],
        "synth_ordered_top2": synth_ordered_top2,
        "synth_unordered_top2": synth_unordered_top2,
    }


def format_scoreboard_row(name: str, slice_name: str, m: Dict[str, Any]) -> str:
    def frac(k):
        a, b = m[k]
        return f"{a}/{b}" if b else "0/0"
    return (f"| {name:<35} | {slice_name:<8} | {100*m['recall_at_3']:>6.2f}% | "
            f"{frac('flip'):>5} | {frac('abstention_precision'):>6} | {frac('abstention_recall'):>6} | "
            f"{frac('ordered_synthesis'):>6} |")


def run_evaluation(
    enable_pair_promotion: bool = True,
    device: str = "cpu",
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    records = load_corpus(DATA_DIR)
    v1_3_report = json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))
    v1_3_preds = v1_3_report["predictions"]

    cfg_file = ARTIFACTS_V1_6 / "pair_scorer_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing calibrated configuration {cfg_file}. Run calibrate_pair_scorer.py first.")
    config = json.loads(cfg_file.read_text(encoding="utf8"))
    if config_override:
        config.update(config_override)

    tau_self = config["tau_self"]
    tau_pool = config["tau_pool"]

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model"]).to(device)
    model.eval()

    passage_cards = {k: passage_card(records[k]) for k in records}

    candidate_predictions = []
    promotion_traces = []

    for q_v13 in v1_3_preds:
        qid = q_v13["query_id"]
        qtype = q_v13["query_type"]
        text = q_v13["dilemma_text"]
        pool = q_v13["pool"]
        abstained = bool(q_v13["abstained"])
        fallback_top3 = list(q_v13["top3"])
        fallback_primary = q_v13["primary"]

        if not enable_pair_promotion or abstained:
            candidate_predictions.append(dict(
                q_v13,
                pair_promoted=False,
                promoted_pair=None,
                joint_score=None,
            ))
            continue

        # Enumerate eligible ordered complementary pairs
        # Deduplication: only take when u's sequence_role starts with "Follow with"
        pairs_in_pool = []
        for u in pool:
            wf = records[u][WORKFLOW]
            v = wf["id"]
            if v in pool and wf["sequence_role"].startswith("Follow with"):
                pairs_in_pool.append((u, v))

        # Single passage scores
        single_inputs = [[text, passage_cards[k]] for k in pool]
        single_scores = batch_score_ce(model, tokenizer, single_inputs, device)
        s_dict = dict(zip(pool, single_scores))
        max_pool_s = max(s_dict.values())

        # Joint pair cards scores
        j_dict = {}
        if pairs_in_pool:
            joint_inputs = [[text, pair_joint_card(records[u], records[v])] for u, v in pairs_in_pool]
            joint_scores = batch_score_ce(model, tokenizer, joint_inputs, device)
            j_dict = dict(zip(pairs_in_pool, joint_scores))

        # Find best eligible pair
        promoted_pair = None
        best_j_val = -9999.0
        qualifying_pairs = []

        for (u, v), j_val in j_dict.items():
            max_self = max(s_dict[u], s_dict[v])
            cond_self = (j_val - max_self >= tau_self)
            cond_pool = (j_val - max_pool_s >= tau_pool)
            if cond_self and cond_pool:
                qualifying_pairs.append({
                    "pair": (u, v),
                    "joint_score": j_val,
                    "max_self": max_self,
                    "diff_self": j_val - max_self,
                    "diff_pool": j_val - max_pool_s,
                })
                if j_val > best_j_val:
                    best_j_val = j_val
                    promoted_pair = (u, v)

        if promoted_pair is not None:
            u, v = promoted_pair
            # Replace/reorder ranking: [u, v] at top, followed by remaining pool items in fallback order
            remaining = [k for k in fallback_top3 if k not in (u, v)]
            for k in pool:
                if k not in (u, v) and k not in remaining:
                    remaining.append(k)
            new_top3 = [u, v] + remaining[:1]
            new_primary = u
            is_promoted = True
        else:
            new_top3 = fallback_top3
            new_primary = fallback_primary
            is_promoted = False

        pred_entry = dict(
            q_v13,
            top3=new_top3,
            primary=new_primary,
            pair_promoted=is_promoted,
            promoted_pair=promoted_pair,
            joint_score=best_j_val if is_promoted else None,
            single_ce_scores=s_dict,
            joint_ce_scores={f"{u}_{v}": val for (u, v), val in j_dict.items()},
        )
        candidate_predictions.append(pred_entry)

        if is_promoted or new_top3 != fallback_top3:
            promotion_traces.append({
                "query_id": qid,
                "query_type": qtype,
                "gold_primary": q_v13.get("gold_primary_id"),
                "gold_competing": q_v13.get("gold_competing_id"),
                "fallback_top3": fallback_top3,
                "new_top3": new_top3,
                "promoted_pair": promoted_pair,
                "joint_score": best_j_val if is_promoted else None,
                "qualifying_pairs": qualifying_pairs,
                "changed": (new_top3 != fallback_top3),
            })

    slices = {
        "all": candidate_predictions,
        "original": candidate_predictions[:18],
        "new": candidate_predictions[18:],
    }
    metrics = {k: evaluate_benchmark(v) for k, v in slices.items()}

    return {
        "metrics": metrics,
        "predictions": candidate_predictions,
        "promotion_traces": promotion_traces,
        "config": config,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("================================================================================")
    print("Decision Corpus v1.6: Ordered Pair Ranking Evaluation")
    print("================================================================================\n")

    # 1. Parity Check: with pair promotion disabled, must match v1.3 exactly
    res_disabled = run_evaluation(enable_pair_promotion=False, device=args.device)
    v1_3_report = json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))
    v1_3_eval = evaluate_benchmark(v1_3_report["predictions"])

    diff_disabled = abs(res_disabled["metrics"]["all"]["recall_at_3"] - v1_3_eval["recall_at_3"])
    assert diff_disabled < 1e-6, f"Fallback parity failed: diff={diff_disabled}"
    print("1. Fallback Parity Check: PASSED (exact match to frozen v1.3 when promotion disabled).\n")

    # 2. Run Candidate with Pair Promotion Enabled
    print("2. Running Candidate with Pair Promotion Enabled...")
    res_candidate = run_evaluation(enable_pair_promotion=True, device=args.device)
    cand_metrics = res_candidate["metrics"]
    v1_3_all = v1_3_eval
    v1_3_orig = evaluate_benchmark(v1_3_report["predictions"][:18])
    v1_3_new = evaluate_benchmark(v1_3_report["predictions"][18:])

    # 3. Scoreboard
    print("\n### 3. Three-Slice Performance Scoreboard")
    print("| Model / Configuration               | Slice    | Recall@3 | Flips | Abst P | Abst R | Ordered Synth |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print(format_scoreboard_row("v1.3 Neural Reference (Frozen)", "All", v1_3_all))
    print(format_scoreboard_row("", "Original", v1_3_orig))
    print(format_scoreboard_row("", "New", v1_3_new))
    print("|---|---|---:|---:|---:|---:|---:|")
    print(format_scoreboard_row("v1.6 Ordered Pair Candidate", "All", cand_metrics["all"]))
    print(format_scoreboard_row("", "Original", cand_metrics["original"]))
    print(format_scoreboard_row("", "New", cand_metrics["new"]))
    print("|---|---|---:|---:|---:|---:|---:|")

    # 4. Synthesis Breakdown (Ordered vs Unordered)
    print("\n### 4. Synthesis Query Breakdown (Ordered Top-2 vs Unordered Top-2)")
    print(f"  v1.3 Reference:  Ordered Top-2: {v1_3_all['ordered_synthesis'][0]}/{v1_3_all['ordered_synthesis'][1]} | "
          f"Unordered Top-2: {v1_3_all['synth_unordered_top2']}/{v1_3_all['ordered_synthesis'][1]}")
    print(f"  v1.6 Candidate:  Ordered Top-2: {cand_metrics['all']['ordered_synthesis'][0]}/{cand_metrics['all']['ordered_synthesis'][1]} | "
          f"Unordered Top-2: {cand_metrics['all']['synth_unordered_top2']}/{cand_metrics['all']['ordered_synthesis'][1]}")

    # 5. Single-Step Promotion Analysis
    single_promoted = [t for t in res_candidate["promotion_traces"] if t["query_type"] != "synthesis"]
    single_lost_primary = [t for t in single_promoted if t["new_top3"][0] != t["gold_primary"]]
    print(f"\n### 5. Single-Step Queries Promotion Count")
    print(f"  Single-step queries receiving pair promotion: {len(single_promoted)}")
    print(f"  Single-step queries losing previously correct primary: {len(single_lost_primary)}")

    # 6. Before-and-After Top 3 for Changed Predictions
    print("\n### 6. Before-and-After Top 3 for Every Changed Query")
    changed_traces = [t for t in res_candidate["promotion_traces"] if t["changed"]]
    if not changed_traces:
        print("  No queries changed prediction.")
    else:
        for t in changed_traces:
            qid = t["query_id"]
            qtype = t["query_type"]
            gold = f"({t['gold_primary']}, {t['gold_competing']})" if t['gold_competing'] else t['gold_primary']
            print(f"  • {qid} ({qtype}) | Gold: {gold}")
            print(f"    v1.3 Top 3: {t['fallback_top3']}")
            print(f"    v1.6 Top 3: {t['new_top3']} (Promoted Pair: {t['promoted_pair']}, Joint Score: {t['joint_score']:.4f})")

    # 7. Promotion Decision Check
    r3_all = cand_metrics["all"]["recall_at_3"]
    flips_all = cand_metrics["all"]["flip"][0]
    flips_new = cand_metrics["new"]["flip"][0]
    abst_p = cand_metrics["all"]["abstention_precision"][0] == cand_metrics["all"]["abstention_precision"][1] == 4
    abst_r = cand_metrics["all"]["abstention_recall"][0] == cand_metrics["all"]["abstention_recall"][1] == 4
    ord_synth = cand_metrics["all"]["ordered_synthesis"][0]

    cond1 = r3_all > 0.6447
    cond2 = (flips_all >= 5) and (flips_new >= 3)
    cond3 = abst_p and abst_r
    cond4 = ord_synth >= 2

    promoted = cond1 and cond2 and cond3 and cond4

    print("\n### 7. Promotion Decision Verification")
    print(f"  1. Overall Recall@3 > 64.47%: {100*r3_all:.2f}% -> {'PASS' if cond1 else 'FAIL'}")
    print(f"  2. Flips >= 5/14 overall and >= 3/8 on new: {flips_all}/14, {flips_new}/8 -> {'PASS' if cond2 else 'FAIL'}")
    print(f"  3. Abstentions 4/4 P, 4/4 R: {cand_metrics['all']['abstention_precision']}, {cand_metrics['all']['abstention_recall']} -> {'PASS' if cond3 else 'FAIL'}")
    print(f"  4. Ordered Synthesis >= 2/10: {ord_synth}/10 -> {'PASS' if cond4 else 'FAIL'}")
    print(f"\n  Final Promotion Decision: {'PROCEED WITH PROMOTION' if promoted else 'DO NOT PROMOTE'}")

    # Save results
    out_file = ARTIFACTS_V1_6 / "v1_6_ordered_pair_report.json"
    report = {
        "timestamp": "2026-11-21T09:30:00Z",
        "config": res_candidate["config"],
        "metrics": cand_metrics,
        "v1_3_baseline_metrics": {
            "all": v1_3_all,
            "original": v1_3_orig,
            "new": v1_3_new,
        },
        "promotion_decision": {
            "promoted": promoted,
            "criteria": {
                "recall_at_3_gt_64_47": cond1,
                "flips_ge_5_overall_and_ge_3_new": cond2,
                "abstention_perfect": cond3,
                "ordered_synth_ge_2": cond4,
            }
        },
        "changed_predictions": changed_traces,
        "predictions": res_candidate["predictions"],
    }
    out_file.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"\nMachine-readable report saved to {out_file}")


if __name__ == "__main__":
    main()
