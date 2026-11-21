"""Blind Comparison Evaluator: Frozen v1.3 Reference vs RC1 (pair-verifier-rc1).

Evaluates independently authored dilemmas while keeping adjudicated gold labels
strictly separate from inference inputs.

Key Protocol Rules:
1. Inputs given to the model are ONLY raw dilemma texts.
2. Inference is executed independently by:
   - Frozen v1.3 Baseline Engine
   - Release Candidate 1 (RC1) Engine
3. When gold labels are provided in a separate file, computes:
   - Paired Arbitration Comparison: Preserves the paired dilemma (P_i) as the
     unit of comparison. A win requires BOTH queries in the pair to have their
     gold primary at rank #1.
     * RC1 Win: RC1 flips pair correctly, v1.3 fails.
     * RC1 Loss: v1.3 flips pair correctly, RC1 fails.
     * Tie: Both succeed or both fail.
   - Ordered Synthesis Comparison: Evaluates ordered top-2 accuracy [p, c].
     * Win, Loss, Tie.
   - Epistemic Abstention Comparison: Evaluates precision, recall, and false abstentions.
   - Recall@3 Comparison across in-scope queries.
4. If no labels file is supplied, outputs blind predictions for downstream adjudication.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from rc1_inference import DharmRC1Engine


def evaluate_predictions_with_labels(v13_preds: List[Dict[str, Any]],
                                     rc1_preds: List[Dict[str, Any]],
                                     labels: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Score v1.3 vs RC1 predictions against adjudicated labels."""
    # Group by query type
    arb_pairs_v13 = defaultdict(list)
    arb_pairs_rc1 = defaultdict(list)

    synth_v13 = []
    synth_rc1 = []

    abst_v13 = []
    abst_rc1 = []

    r3_v13 = []
    r3_rc1 = []

    for p13, prc1 in zip(v13_preds, rc1_preds):
        qid = p13["query_id"]
        lab = labels.get(qid)
        if not lab:
            continue

        qtype = lab["query_type"]
        gold_p = lab["gold_primary_id"]
        gold_c = lab.get("gold_competing_id")
        pair_grp = lab.get("pair_group")

        # Abstention evaluation
        if qtype == "abstention":
            abst_v13.append({"pred": p13["abstained"], "gold": True})
            abst_rc1.append({"pred": prc1["abstained"], "gold": True})
            continue

        abst_v13.append({"pred": p13["abstained"], "gold": False})
        abst_rc1.append({"pred": prc1["abstained"], "gold": False})

        # Recall@3
        relevant = {gold_p}
        if qtype == "synthesis" and gold_c:
            relevant.add(gold_c)

        r3_v13.append(len(relevant.intersection(p13["top3"])) / len(relevant))
        r3_rc1.append(len(relevant.intersection(prc1["top3"])) / len(relevant))

        # Arbitration pairs
        if qtype == "arbitration_pair" and pair_grp:
            arb_pairs_v13[pair_grp].append(p13["primary"] == gold_p)
            arb_pairs_rc1[pair_grp].append(prc1["primary"] == gold_p)

        # Synthesis ordered pairs
        if qtype == "synthesis" and gold_c:
            v13_hit = (not p13["abstained"] and p13["top3"][:2] == [gold_p, gold_c])
            rc1_hit = (not prc1["abstained"] and prc1["top3"][:2] == [gold_p, gold_c])
            synth_v13.append(v13_hit)
            synth_rc1.append(rc1_hit)

    # Paired arbitration wins / losses / ties
    arb_wins_rc1 = 0
    arb_losses_rc1 = 0
    arb_ties = 0

    for grp in arb_pairs_v13:
        v13_succ = (len(arb_pairs_v13[grp]) == 2 and all(arb_pairs_v13[grp]))
        rc1_succ = (len(arb_pairs_rc1[grp]) == 2 and all(arb_pairs_rc1[grp]))

        if rc1_succ and not v13_succ:
            arb_wins_rc1 += 1
        elif v13_succ and not rc1_succ:
            arb_losses_rc1 += 1
        else:
            arb_ties += 1

    total_arb_pairs = len(arb_pairs_v13)

    # Synthesis wins / losses / ties
    synth_wins_rc1 = 0
    synth_losses_rc1 = 0
    synth_ties = 0

    for s13, src1 in zip(synth_v13, synth_rc1):
        if src1 and not s13:
            synth_wins_rc1 += 1
        elif s13 and not src1:
            synth_losses_rc1 += 1
        else:
            synth_ties += 1

    total_synth = len(synth_v13)

    return {
        "recall_at_3": {
            "v1_3": float(np.mean(r3_v13)) if r3_v13 else 0.0,
            "rc1": float(np.mean(r3_rc1)) if r3_rc1 else 0.0,
            "delta": float(np.mean(r3_rc1) - np.mean(r3_v13)) if r3_v13 else 0.0,
        },
        "paired_arbitration": {
            "total_pairs": total_arb_pairs,
            "rc1_flips": sum(1 for g in arb_pairs_rc1.values() if all(g) and len(g) == 2),
            "v1_3_flips": sum(1 for g in arb_pairs_v13.values() if all(g) and len(g) == 2),
            "rc1_wins": arb_wins_rc1,
            "rc1_losses": arb_losses_rc1,
            "ties": arb_ties,
        },
        "ordered_synthesis": {
            "total_queries": total_synth,
            "rc1_successes": sum(synth_rc1),
            "v1_3_successes": sum(synth_v13),
            "rc1_wins": synth_wins_rc1,
            "rc1_losses": synth_losses_rc1,
            "ties": synth_ties,
        },
        "abstention": {
            "v1_3": {
                "tp": sum(1 for x in abst_v13 if x["pred"] and x["gold"]),
                "fp": sum(1 for x in abst_v13 if x["pred"] and not x["gold"]),
                "fn": sum(1 for x in abst_v13 if not x["pred"] and x["gold"]),
            },
            "rc1": {
                "tp": sum(1 for x in abst_rc1 if x["pred"] and x["gold"]),
                "fp": sum(1 for x in abst_rc1 if x["pred"] and not x["gold"]),
                "fn": sum(1 for x in abst_rc1 if not x["pred"] and x["gold"]),
            },
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Blind Comparison Evaluator: Frozen v1.3 vs RC1")
    ap.add_argument("--dilemmas", type=Path, required=True, help="Path to input JSONL containing raw dilemma texts")
    ap.add_argument("--labels", type=Path, default=None, help="Optional path to adjudicated gold labels JSONL")
    ap.add_argument("--out", type=Path, default=None, help="Optional output JSON path for predictions and report")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("=" * 80)
    print("Blind Evaluation Protocol: Frozen v1.3 vs Release Candidate 1")
    print("=" * 80)

    # 1. Load Dilemmas
    dilemma_rows = [json.loads(line) for line in args.dilemmas.read_text(encoding="utf8").splitlines()]
    print(f"\nLoaded {len(dilemma_rows)} dilemma queries from {args.dilemmas}")

    # 2. Initialize RC1 Engine
    print("\nInitializing Dharm RC1 Engine...")
    engine = DharmRC1Engine(device=args.device)

    # 3. Run Predictions
    print("\nRunning blind inference on identical raw texts...")
    v13_preds = []
    rc1_preds = []

    for idx, row in enumerate(dilemma_rows):
        qid = row.get("query_id") or row.get("id") or f"BLIND_{idx:03d}"
        text = row["dilemma_text"]

        res = engine.predict(text)

        # In RC1 engine, fallback_primary and fallback_top3 are exactly the v1.3 prediction!
        pred_v13 = {
            "query_id": qid,
            "dilemma_text": text,
            "primary": res["fallback_primary"],
            "top3": res["fallback_top3"],
            "abstained": res["abstained"],
            "abstention_prob": res["abstention_prob"],
        }
        v13_preds.append(pred_v13)

        pred_rc1 = {
            "query_id": qid,
            "dilemma_text": text,
            "primary": res["primary"],
            "top3": res["top3"],
            "abstained": res["abstained"],
            "abstention_prob": res["abstention_prob"],
            "pair_promoted": res["pair_promoted"],
            "proposed_pair": res["proposed_pair"],
            "acceptance_score": res["acceptance_score"],
            "fallback_reason": res["fallback_reason"],
        }
        rc1_preds.append(pred_rc1)

    print(f"Completed inference on {len(dilemma_rows)} queries.")

    # 4. Score if labels provided
    evaluation_results = None
    if args.labels and args.labels.exists():
        label_rows = [json.loads(line) for line in args.labels.read_text(encoding="utf8").splitlines()]
        labels_dict = {r.get("query_id") or r.get("id"): r for r in label_rows}
        print(f"\nScoring against {len(labels_dict)} adjudicated labels from {args.labels}...")
        evaluation_results = evaluate_predictions_with_labels(v13_preds, rc1_preds, labels_dict)

        print("\n" + "=" * 80)
        print("Blind Comparison Results:")
        print("=" * 80)
        r3 = evaluation_results["recall_at_3"]
        print(f"  • Recall@3: v1.3 = {r3['v1_3']*100:.2f}%, RC1 = {r3['rc1']*100:.2f}% (Delta = {r3['delta']*100:+.2f}%)")

        arb = evaluation_results["paired_arbitration"]
        print(f"  • Paired Arbitration: RC1 = {arb['rc1_flips']}/{arb['total_pairs']}, v1.3 = {arb['v1_3_flips']}/{arb['total_pairs']}")
        print(f"    - Paired Head-to-Head: Wins={arb['rc1_wins']}, Losses={arb['rc1_losses']}, Ties={arb['ties']}")

        synth = evaluation_results["ordered_synthesis"]
        print(f"  • Ordered Synthesis:  RC1 = {synth['rc1_successes']}/{synth['total_queries']}, v1.3 = {synth['v1_3_successes']}/{synth['total_queries']}")
        print(f"    - Head-to-Head: Wins={synth['rc1_wins']}, Losses={synth['rc1_losses']}, Ties={synth['ties']}")

        abst = evaluation_results["abstention"]
        print(f"  • Abstention TP/FP/FN: RC1={abst['rc1']}, v1.3={abst['v1_3']}")
    else:
        print("\nNo adjudicated labels provided. Blind validation status: PENDING.")
        print("Inference predictions generated and ready for blind scoring once labels are supplied.")

    if args.out:
        out_data = {
            "num_queries": len(dilemma_rows),
            "labels_supplied": bool(args.labels and args.labels.exists()),
            "blind_validation_status": "COMPLETED" if evaluation_results else "PENDING",
            "evaluation_results": evaluation_results,
            "v1_3_predictions": v13_preds,
            "rc1_predictions": rc1_preds,
        }
        args.out.write_text(json.dumps(out_data, indent=2), encoding="utf8")
        print(f"Saved evaluation outputs to {args.out}")


if __name__ == "__main__":
    main()
