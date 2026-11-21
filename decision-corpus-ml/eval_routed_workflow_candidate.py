"""End-to-End Evaluation of Routed Workflow Intent Candidate (v1.6-routed).

Comparison:
1. v1.3 Neural Reference (Frozen)
2. Ungated v1.6 (Frozen Pair Scorer without Router)
3. Diagnostic Routing Oracle (Label-aware diagnostic: v1.6 on synthesis, v1.3 on others)
4. Deployable Router Candidate (Gated by frozen BGE + LogisticRegression intent classifier)

Pipeline Rules:
- Apply existing epistemic abstention gate first.
- Default to complete v1.3 ranking.
- Permit v1.6 pair promotion ONLY when:
    (a) Router gate is open: router_prob >= theta* (0.3300, frozen)
    AND
    (b) Frozen pair scorer accepts a pair (tau_self = -0.5, tau_pool = -1.0)
- Verify bit-for-bit v1.3 fallback whenever either condition fails.
- Exclude query ID, query type, pair group, and gold IDs from inference.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_V1_3 = Path("decision-corpus-ml/artifacts-v1.3")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")

SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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


def run_pipeline(device: str = "cpu") -> Dict[str, Any]:
    records = load_corpus(DATA_DIR)
    v1_3_report = json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))
    v1_3_preds = v1_3_report["predictions"]

    v1_6_report = json.loads((ARTIFACTS_V1_6 / "v1_6_ordered_pair_report.json").read_text(encoding="utf8"))
    v1_6_preds = v1_6_report["predictions"]

    # Load frozen pair scorer config
    pair_cfg = json.loads((ARTIFACTS_V1_6 / "pair_scorer_config.json").read_text(encoding="utf8"))
    tau_self = pair_cfg["tau_self"]
    tau_pool = pair_cfg["tau_pool"]

    # Load frozen router model and config
    router_cfg = json.loads((ARTIFACTS_V1_6 / "router_config.json").read_text(encoding="utf8"))
    router_clf = joblib.load(ARTIFACTS_V1_6 / "workflow_router.joblib")
    router_thresh = router_cfg["calibrated_threshold"]

    # Load models
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    tokenizer = AutoTokenizer.from_pretrained(pair_cfg["model"])
    ce_model = AutoModelForSequenceClassification.from_pretrained(pair_cfg["model"]).to(device)
    ce_model.eval()

    passage_cards = {k: passage_card(records[k]) for k in records}

    # 1. Diagnostic Routing Oracle: v1.6 on 10 synthesis queries, v1.3 everywhere else
    oracle_predictions = []
    for q13, q16 in zip(v1_3_preds, v1_6_preds):
        if q13["query_type"] == "synthesis":
            oracle_predictions.append(dict(q16))
        else:
            oracle_predictions.append(dict(q13))

    # 2. Deployable Router Candidate Pipeline
    routed_predictions = []
    traces = []

    # Batch compute router probabilities for all dilemma texts
    dilemma_texts = [q["dilemma_text"] for q in v1_3_preds]
    query_embs = bge_model.encode([QUERY_PREFIX + t for t in dilemma_texts], normalize_embeddings=True, show_progress_bar=False)
    router_probs = router_clf.predict_proba(query_embs)[:, 1]

    for idx, q_v13 in enumerate(v1_3_preds):
        qid = q_v13["query_id"]
        qtype = q_v13["query_type"]
        text = q_v13["dilemma_text"]
        pool = q_v13["pool"]
        abstained = bool(q_v13["abstained"])
        fallback_top3 = list(q_v13["top3"])
        fallback_primary = q_v13["primary"]
        r_prob = float(router_probs[idx])

        # Inference logic: strict encapsulation
        gate_open = (not abstained) and (r_prob >= router_thresh)
        pair_promoted = False
        promoted_pair = None
        best_j_val = None
        decline_reason = None

        if not gate_open:
            if abstained:
                decline_reason = "abstained"
            else:
                decline_reason = "router_closed"
            final_top3 = fallback_top3
            final_primary = fallback_primary
        else:
            # Router opened! Evaluate pair scorer on candidates in pool
            pairs_in_pool = []
            for u in pool:
                wf = records[u][WORKFLOW]
                v = wf["id"]
                if v in pool and wf["sequence_role"].startswith("Follow with"):
                    pairs_in_pool.append((u, v))

            single_inputs = [[text, passage_cards[k]] for k in pool]
            single_scores = batch_score_ce(ce_model, tokenizer, single_inputs, device)
            s_dict = dict(zip(pool, single_scores))
            max_pool_s = max(s_dict.values())

            j_dict = {}
            if pairs_in_pool:
                joint_inputs = [[text, pair_joint_card(records[u], records[v])] for u, v in pairs_in_pool]
                joint_scores = batch_score_ce(ce_model, tokenizer, joint_inputs, device)
                j_dict = dict(zip(pairs_in_pool, joint_scores))

            qualifying_pairs = []
            for (u, v), j_val in j_dict.items():
                max_self = max(s_dict[u], s_dict[v])
                cond_self = (j_val - max_self >= tau_self)
                cond_pool = (j_val - max_pool_s >= tau_pool)
                if cond_self and cond_pool:
                    qualifying_pairs.append({
                        "pair": (u, v),
                        "joint_score": j_val,
                        "diff_self": j_val - max_self,
                        "diff_pool": j_val - max_pool_s,
                    })

            if qualifying_pairs:
                qualifying_pairs.sort(key=lambda x: -x["joint_score"])
                promoted_pair = qualifying_pairs[0]["pair"]
                best_j_val = qualifying_pairs[0]["joint_score"]
                pair_promoted = True

                u, v = promoted_pair
                remaining = [k for k in fallback_top3 if k not in (u, v)]
                for k in pool:
                    if k not in (u, v) and k not in remaining:
                        remaining.append(k)
                final_top3 = [u, v] + remaining[:1]
                final_primary = u
            else:
                decline_reason = "pair_scorer_declined"
                final_top3 = fallback_top3
                final_primary = fallback_primary

        # Bit-for-bit check: if not pair_promoted, final_top3 MUST equal fallback_top3 exactly
        if not pair_promoted:
            assert final_top3 == fallback_top3, f"Bit-for-bit fallback failed on {qid}"
            assert final_primary == fallback_primary, f"Primary fallback failed on {qid}"

        pred_entry = dict(
            q_v13,
            top3=final_top3,
            primary=final_primary,
            router_prob=r_prob,
            router_gate_open=gate_open,
            pair_promoted=pair_promoted,
            promoted_pair=promoted_pair,
            joint_score=best_j_val,
            decline_reason=decline_reason,
        )
        routed_predictions.append(pred_entry)

        traces.append({
            "query_id": qid,
            "query_type": qtype,
            "gold_primary": q_v13.get("gold_primary_id"),
            "gold_competing": q_v13.get("gold_competing_id"),
            "router_prob": r_prob,
            "router_gate_open": gate_open,
            "pair_promoted": pair_promoted,
            "promoted_pair": promoted_pair,
            "fallback_top3": fallback_top3,
            "fallback_primary": fallback_primary,
            "final_top3": final_top3,
            "final_primary": final_primary,
            "decline_reason": decline_reason,
            "primary_changed": (final_primary != fallback_primary),
            "top3_changed": (final_top3 != fallback_top3),
        })

    return {
        "v1_3_predictions": v1_3_preds,
        "v1_6_predictions": v1_6_preds,
        "oracle_predictions": oracle_predictions,
        "routed_predictions": routed_predictions,
        "traces": traces,
        "router_threshold": router_thresh,
        "router_config": router_cfg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("================================================================================")
    print("End-to-End Evaluation: Routed Workflow Intent Candidate vs Baselines")
    print("================================================================================\n")

    res = run_pipeline(device=args.device)
    v13_preds = res["v1_3_predictions"]
    v16_preds = res["v1_6_predictions"]
    oracle_preds = res["oracle_predictions"]
    routed_preds = res["routed_predictions"]
    traces = res["traces"]

    def compute_slices(preds):
        return {
            "all": evaluate_benchmark(preds),
            "original": evaluate_benchmark(preds[:18]),
            "new": evaluate_benchmark(preds[18:]),
        }

    m_v13 = compute_slices(v13_preds)
    m_v16 = compute_slices(v16_preds)
    m_oracle = compute_slices(oracle_preds)
    m_routed = compute_slices(routed_preds)

    # 1. Four-Way Three-Slice Scoreboard
    print("### 1. Four-Way Three-Slice Scoreboard")
    print("| Model / Configuration               | Slice    | Recall@3 | Flips | Abst P | Abst R | Ordered Synth |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name, m_dict in [
        ("v1.3 Neural Reference (Frozen)", m_v13),
        ("Ungated v1.6 Pair Candidate", m_v16),
        ("Diagnostic Routing Oracle", m_oracle),
        ("Deployable Router Candidate", m_routed),
    ]:
        print(format_scoreboard_row(name, "All", m_dict["all"]))
        print(format_scoreboard_row("", "Original", m_dict["original"]))
        print(format_scoreboard_row("", "New", m_dict["new"]))
        print("|---|---|---:|---:|---:|---:|---:|")

    # 2. Gate Opening Statistics
    synth_traces = [t for t in traces if t["query_type"] == "synthesis"]
    arb_traces = [t for t in traces if t["query_type"] == "arbitration_pair"]
    abst_traces = [t for t in traces if t["query_type"] == "abstention"]

    synth_opened = sum(1 for t in synth_traces if t["router_gate_open"])
    arb_opened = sum(1 for t in arb_traces if t["router_gate_open"])
    abst_opened = sum(1 for t in abst_traces if t["router_gate_open"])

    synth_promoted = sum(1 for t in synth_traces if t["pair_promoted"])
    arb_promoted = sum(1 for t in arb_traces if t["pair_promoted"])

    arb_changed_primary = sum(1 for t in arb_traces if t["primary_changed"])
    synth_changed_primary = sum(1 for t in synth_traces if t["primary_changed"])

    print("\n### 2. Router Gate Opening & Promotion Statistics")
    print(f"  Synthesis queries gate opened:   {synth_opened}/10 ({synth_opened/10*100:.1f}%)")
    print(f"  Arbitration queries gate opened: {arb_opened}/28 ({arb_opened/28*100:.1f}%)")
    print(f"  Abstention queries gate opened:  {abst_opened}/4 ({abst_opened/4*100:.1f}%)")
    print(f"  Synthesis pairs promoted:        {synth_promoted}/10")
    print(f"  Arbitration pairs promoted:      {arb_promoted}/28")
    print(f"  Arbitration primaries changed:   {arb_changed_primary}/28")

    # 3. Primary Gains and Losses in Arbitration Queries
    corr_to_incorr = []
    incorr_to_corr = []
    for t in arb_traces:
        gold = t["gold_primary"]
        p_v13 = t["fallback_primary"]
        p_final = t["final_primary"]
        if p_v13 == gold and p_final != gold:
            corr_to_incorr.append(t["query_id"])
        elif p_v13 != gold and p_final == gold:
            incorr_to_corr.append(t["query_id"])

    print("\n### 3. Arbitration Primary Gains & Losses")
    print(f"  Correct -> Incorrect Primaries ({len(corr_to_incorr)}): {corr_to_incorr}")
    print(f"  Incorrect -> Correct Primaries ({len(incorr_to_corr)}): {incorr_to_corr}")

    # 4. Diagnosis of Every Synthesis Miss
    print("\n### 4. Diagnosis for Each of the 10 Synthesis Queries")
    for t in synth_traces:
        qid = t["query_id"]
        gold_p = t["gold_primary"]
        gold_c = t["gold_competing"]
        is_hit = (t["final_top3"][:2] == [gold_p, gold_c])
        status = "HIT" if is_hit else "MISS"
        prob = t["router_prob"]
        gate = "OPEN" if t["router_gate_open"] else "CLOSED"
        prom = f"Promoted {t['promoted_pair']}" if t["pair_promoted"] else f"No ({t['decline_reason']})"
        print(f"  • {qid} [{status}]: Gold=({gold_p}, {gold_c}) | Top3={t['final_top3']} | "
              f"Router Prob={prob:.4f} (Gate: {gate}) | Pair Promotion: {prom}")
        if not is_hit:
            if not t["router_gate_open"]:
                print(f"    -> Miss Cause: ROUTER CLOSED (router_prob={prob:.4f} < {res['router_threshold']:.4f})")
            elif not t["pair_promoted"]:
                print(f"    -> Miss Cause: PAIR SCORER DECLINED")
            else:
                print(f"    -> Miss Cause: WRONG PAIR WON ({t['promoted_pair']} instead of ({gold_p}, {gold_c}))")

    # 5. Promotion Decision
    r3_all = m_routed["all"]["recall_at_3"]
    flips_all = m_routed["all"]["flip"][0]
    flips_new = m_routed["all"]["flip"][0]
    flips_new_slice = m_routed["new"]["flip"][0]
    abst_p = m_routed["all"]["abstention_precision"][0] == m_routed["all"]["abstention_precision"][1] == 4
    abst_r = m_routed["all"]["abstention_recall"][0] == m_routed["all"]["abstention_recall"][1] == 4
    ord_synth = m_routed["all"]["ordered_synthesis"][0]

    cond1 = r3_all > 0.6447
    cond2 = (flips_all >= 5) and (flips_new_slice >= 3)
    cond3 = abst_p and abst_r
    cond4 = ord_synth >= 2

    promoted = cond1 and cond2 and cond3 and cond4

    print("\n### 5. Formal Promotion Criteria Check")
    print(f"  1. Overall Recall@3 > 64.47%: {100*r3_all:.2f}% -> {'PASS' if cond1 else 'FAIL'}")
    print(f"  2. Flips >= 5/14 overall & >= 3/8 on new: {flips_all}/14, {flips_new_slice}/8 -> {'PASS' if cond2 else 'FAIL'}")
    print(f"  3. Abstentions 4/4 P, 4/4 R: {m_routed['all']['abstention_precision']}, {m_routed['all']['abstention_recall']} -> {'PASS' if cond3 else 'FAIL'}")
    print(f"  4. Ordered Synthesis >= 2/10: {ord_synth}/10 -> {'PASS' if cond4 else 'FAIL'}")
    print(f"\n  Final Promotion Decision: {'PROCEED WITH PROMOTION' if promoted else 'DO NOT PROMOTE'}")

    # Save machine-readable evaluation report
    out_file = ARTIFACTS_V1_6 / "workflow_intent_router_eval_report.json"
    report = {
        "timestamp": "2026-11-21T09:45:00Z",
        "router_threshold": res["router_threshold"],
        "router_config": res["router_config"],
        "metrics": {
            "v1_3_baseline": m_v13,
            "ungated_v1_6": m_v16,
            "diagnostic_oracle": m_oracle,
            "deployable_router": m_routed,
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
        "gate_statistics": {
            "synthesis_opened": [synth_opened, 10],
            "arbitration_opened": [arb_opened, 28],
            "abstention_opened": [abst_opened, 4],
            "synthesis_promoted": [synth_promoted, 10],
            "arbitration_promoted": [arb_promoted, 28],
        },
        "arbitration_impact": {
            "corr_to_incorr": corr_to_incorr,
            "incorr_to_corr": incorr_to_corr,
        },
        "predictions": routed_preds,
    }
    out_file.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
