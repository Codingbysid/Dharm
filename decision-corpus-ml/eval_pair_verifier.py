"""End-to-End Evaluation of Pair-Promotion Verifier Candidate (v1.6-verifier).

Compares:
1. v1.3 Neural Reference (Frozen Baseline)
2. Ungated v1.6 (Frozen Pair Scorer without Verifier)
3. Diagnostic Routing Oracle (Label-aware diagnostic ceiling)
4. Deployable Pair Verifier Candidate (Gated by frozen Pair-Promotion Verifier)

Pipeline Rules:
- Check epistemic abstention gate first. If abstained -> primary = "Abstain", no pair promotion.
- Default ranking is complete v1.3 ranking.
- Run frozen v1.6 pair scorer on pool (tau_self = -0.5, tau_pool = -1.0).
- If no pair proposed -> reject (exact bit-for-bit v1.3 fallback).
- If pair proposed (u*, v*) -> extract 16 features:
    diff_self, diff_pool, diff_min, endpoint_span, j_score, max_self, min_self,
    rank_u, rank_v, min_rank, max_rank, rank_sum, both_in_top2, both_in_top3,
    both_in_top5, router_prob.
- Score with frozen pair_verifier_model.joblib. If P(accept) >= tau_v* (frozen in verifier_config.json):
    promote (u*, v*) to ranks #1 and #2. Primary = u*.
  Otherwise:
    reject proposal and fallback bit-for-bit to v1.3 ranking.
- Strictly exclude query ID, query type, pair group, and gold IDs from inference.
"""
import argparse
from collections import defaultdict
import hashlib
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


def run_pipeline(device: str = "cpu", recompute: bool = False) -> Dict[str, Any]:
    records = load_corpus(DATA_DIR)
    v1_3_report = json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))
    v1_3_preds = v1_3_report["predictions"]

    v1_6_report = json.loads((ARTIFACTS_V1_6 / "v1_6_ordered_pair_report.json").read_text(encoding="utf8"))
    v1_6_preds = v1_6_report["predictions"]

    # Load pair scorer config
    pair_cfg = json.loads((ARTIFACTS_V1_6 / "pair_scorer_config.json").read_text(encoding="utf8"))
    tau_self = pair_cfg["tau_self"]
    tau_pool = pair_cfg["tau_pool"]

    # Load router config and model
    router_clf = joblib.load(ARTIFACTS_V1_6 / "workflow_router.joblib")
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # Load verifier config and model
    verifier_cfg = json.loads((ARTIFACTS_V1_6 / "verifier_config.json").read_text(encoding="utf8"))
    verifier_clf = joblib.load(ARTIFACTS_V1_6 / "pair_verifier_model.joblib")
    verifier_thresh = verifier_cfg["calibrated_threshold"]
    feature_names = verifier_cfg["features"]

    # Optional cross-encoder model loading
    ce_model = None
    tokenizer = None
    passage_cards = None
    if recompute:
        tokenizer = AutoTokenizer.from_pretrained(pair_cfg["model"])
        ce_model = AutoModelForSequenceClassification.from_pretrained(pair_cfg["model"]).to(device)
        ce_model.eval()
        passage_cards = {k: passage_card(records[k]) for k in records}

    # Precompute router intent probabilities for all queries
    dilemma_texts = [q["dilemma_text"] for q in v1_3_preds]
    query_embs = bge_model.encode([QUERY_PREFIX + t for t in dilemma_texts], normalize_embeddings=True, show_progress_bar=False)
    router_probs = router_clf.predict_proba(query_embs)[:, 1]

    # 1. Diagnostic Routing Oracle: v1.6 on 10 synthesis queries, v1.3 everywhere else
    oracle_predictions = []
    for q13, q16 in zip(v1_3_preds, v1_6_preds):
        if q13["query_type"] == "synthesis":
            oracle_predictions.append(dict(q16))
        else:
            oracle_predictions.append(dict(q13))

    # 2. Deployable Pair Verifier Candidate Pipeline
    verified_predictions = []
    traces = []

    for idx, (q_v13, q_v16) in enumerate(zip(v1_3_preds, v1_6_preds)):
        qid = q_v13["query_id"]
        qtype = q_v13["query_type"]
        text = q_v13["dilemma_text"]
        pool = list(q_v13["pool"])
        abstained = bool(q_v13["abstained"])
        fallback_top3 = list(q_v13["top3"])
        fallback_primary = q_v13["primary"]
        r_prob = float(router_probs[idx])

        pair_promoted = False
        promoted_pair = None
        decline_reason = None
        verifier_prob = None
        feat_dict = None

        if abstained:
            decline_reason = "abstained"
            final_top3 = fallback_top3
            final_primary = fallback_primary
        else:
            # Get single and joint scores
            if recompute:
                pairs_in_pool = []
                for u in pool:
                    wf = records[u][WORKFLOW]
                    v = wf["id"]
                    if v in pool and wf["sequence_role"].startswith("Follow with"):
                        pairs_in_pool.append((u, v))

                single_inputs = [[text, passage_cards[k]] for k in pool]
                single_scores = batch_score_ce(ce_model, tokenizer, single_inputs, device)
                s_dict = dict(zip(pool, single_scores))

                j_dict = {}
                if pairs_in_pool:
                    joint_inputs = [[text, pair_joint_card(records[u], records[v])] for u, v in pairs_in_pool]
                    joint_scores = batch_score_ce(ce_model, tokenizer, joint_inputs, device)
                    j_dict = dict(zip(pairs_in_pool, joint_scores))
            else:
                s_dict = q_v16.get("single_ce_scores", {})
                j_raw = q_v16.get("joint_ce_scores", {})
                j_dict = {tuple(k.split("_")): v for k, v in j_raw.items()}

            if not s_dict:
                decline_reason = "no_scores"
                final_top3 = fallback_top3
                final_primary = fallback_primary
            else:
                max_pool_s = max(s_dict.values())
                qualifying = []
                for (u, v), j_val in j_dict.items():
                    if u not in s_dict or v not in s_dict:
                        continue
                    max_self = max(s_dict[u], s_dict[v])
                    min_self = min(s_dict[u], s_dict[v])
                    diff_self = j_val - max_self
                    diff_pool = j_val - max_pool_s
                    diff_min = j_val - min_self
                    span = max_self - min_self
                    if diff_self >= tau_self and diff_pool >= tau_pool:
                        qualifying.append({
                            "pair": (u, v),
                            "j_score": j_val,
                            "max_self": max_self,
                            "min_self": min_self,
                            "diff_self": diff_self,
                            "diff_pool": diff_pool,
                            "diff_min": diff_min,
                            "endpoint_span": span,
                        })

                if not qualifying:
                    decline_reason = "pair_scorer_declined"
                    final_top3 = fallback_top3
                    final_primary = fallback_primary
                else:
                    qualifying.sort(key=lambda x: -x["j_score"])
                    best = qualifying[0]
                    u_prop, v_prop = best["pair"]

                    # Rank features in pool
                    fallback_rank = sorted(pool, key=lambda k: (-s_dict[k], k))
                    ru = fallback_rank.index(u_prop) + 1 if u_prop in fallback_rank else 99
                    rv = fallback_rank.index(v_prop) + 1 if v_prop in fallback_rank else 99
                    min_r = min(ru, rv)
                    max_r = max(ru, rv)
                    rank_sum = ru + rv
                    both_in_top2 = float(ru <= 2 and rv <= 2)
                    both_in_top3 = float(ru <= 3 and rv <= 3)
                    both_in_top5 = float(ru <= 5 and rv <= 5)

                    feat_vector = [
                        best["diff_self"],
                        best["diff_pool"],
                        best["diff_min"],
                        best["endpoint_span"],
                        best["j_score"],
                        best["max_self"],
                        best["min_self"],
                        float(ru),
                        float(rv),
                        float(min_r),
                        float(max_r),
                        float(rank_sum),
                        both_in_top2,
                        both_in_top3,
                        both_in_top5,
                        r_prob,
                    ]
                    feat_dict = dict(zip(feature_names, feat_vector))

                    v_prob = float(verifier_clf.predict_proba([feat_vector])[0, 1])
                    verifier_prob = v_prob

                    if v_prob >= verifier_thresh:
                        pair_promoted = True
                        promoted_pair = [u_prop, v_prop]
                        remaining = [k for k in fallback_top3 if k not in (u_prop, v_prop)]
                        for k in pool:
                            if k not in (u_prop, v_prop) and k not in remaining:
                                remaining.append(k)
                        final_top3 = [u_prop, v_prop] + remaining[:1]
                        final_primary = u_prop
                    else:
                        decline_reason = "verifier_rejected"
                        final_top3 = fallback_top3
                        final_primary = fallback_primary

        # Bit-for-bit assertion: if proposal is rejected or absent, top3 and primary MUST match v1.3 exactly
        if not pair_promoted:
            assert final_top3 == fallback_top3, f"Fallback top3 diverged on {qid}"
            assert final_primary == fallback_primary, f"Fallback primary diverged on {qid}"

        pred_entry = dict(
            q_v13,
            top3=final_top3,
            primary=final_primary,
            pair_promoted=pair_promoted,
            promoted_pair=promoted_pair,
            router_prob=r_prob,
            verifier_prob=verifier_prob,
            decline_reason=decline_reason,
        )
        verified_predictions.append(pred_entry)

        traces.append({
            "query_id": qid,
            "query_type": qtype,
            "pair_group": q_v13.get("pair_group"),
            "gold_primary": q_v13.get("gold_primary_id"),
            "gold_competing": q_v13.get("gold_competing_id"),
            "router_prob": r_prob,
            "verifier_prob": verifier_prob,
            "pair_promoted": pair_promoted,
            "promoted_pair": promoted_pair,
            "fallback_top3": fallback_top3,
            "fallback_primary": fallback_primary,
            "final_top3": final_top3,
            "final_primary": final_primary,
            "decline_reason": decline_reason,
            "primary_changed": (final_primary != fallback_primary),
            "top3_changed": (final_top3 != fallback_top3),
            "features": feat_dict,
        })

    return {
        "v1_3_predictions": v1_3_preds,
        "v1_6_predictions": v1_6_preds,
        "oracle_predictions": oracle_predictions,
        "verified_predictions": verified_predictions,
        "traces": traces,
        "verifier_threshold": verifier_thresh,
        "verifier_config": verifier_cfg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--recompute", action="store_true", help="Recompute neural cross-encoder scores instead of cached.")
    args = ap.parse_args()

    print("================================================================================")
    print("End-to-End Evaluation: Pair-Promotion Verifier Candidate vs Baselines")
    print("================================================================================\n")

    res = run_pipeline(device=args.device, recompute=args.recompute)
    v13_preds = res["v1_3_predictions"]
    v16_preds = res["v1_6_predictions"]
    oracle_preds = res["oracle_predictions"]
    verified_preds = res["verified_predictions"]
    traces = res["traces"]
    v_thresh = res["verifier_threshold"]

    def compute_slices(preds):
        return {
            "all": evaluate_benchmark(preds),
            "original": evaluate_benchmark(preds[:18]),
            "new": evaluate_benchmark(preds[18:]),
        }

    m_v13 = compute_slices(v13_preds)
    m_v16 = compute_slices(v16_preds)
    m_oracle = compute_slices(oracle_preds)
    m_verified = compute_slices(verified_preds)

    # 1. Four-Way Three-Slice Scoreboard
    print("### 1. Four-Way Three-Slice Scoreboard")
    print("| Model / Configuration               | Slice    | Recall@3 | Flips | Abst P | Abst R | Ordered Synth |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name, m_dict in [
        ("v1.3 Neural Reference (Frozen)", m_v13),
        ("Ungated v1.6 Pair Candidate", m_v16),
        ("Diagnostic Routing Oracle", m_oracle),
        ("Deployable Pair Verifier Candidate", m_verified),
    ]:
        print(format_scoreboard_row(name, "All", m_dict["all"]))
        print(format_scoreboard_row("", "Original", m_dict["original"]))
        print(format_scoreboard_row("", "New", m_dict["new"]))
        print("|---|---|---:|---:|---:|---:|---:|")

    # 2. Detailed Diagnostic Counts
    synth_traces = [t for t in traces if t["query_type"] == "synthesis"]
    arb_traces = [t for t in traces if t["query_type"] == "arbitration_pair"]
    abst_traces = [t for t in traces if t["query_type"] == "abstention"]

    correct_promoted = []
    wrong_promoted = []
    correct_blocked = []
    corr_arb_displaced = []

    v16_extra_synth = ["Q14", "Q24", "Q29", "Q30", "Q36"]
    survived_v16_extra = []

    for t in traces:
        qid = t["query_id"]
        qtype = t["query_type"]
        gold_p = t["gold_primary"]
        gold_c = t["gold_competing"]
        prom_pair = tuple(t["promoted_pair"]) if t["promoted_pair"] else None

        is_gold_synth_pair = (qtype == "synthesis" and prom_pair == (gold_p, gold_c))

        if t["pair_promoted"]:
            if is_gold_synth_pair:
                correct_promoted.append((qid, prom_pair))
            else:
                wrong_promoted.append((qid, prom_pair, qtype))

            if qtype == "arbitration_pair":
                v13_corr = (t["fallback_primary"] == gold_p)
                ver_corr = (t["final_primary"] == gold_p)
                if v13_corr and not ver_corr:
                    corr_arb_displaced.append(qid)

            if qid in v16_extra_synth and is_gold_synth_pair:
                survived_v16_extra.append(qid)
        else:
            # Check if frozen v1.6 proposed the correct gold pair but verifier blocked it
            # We can check from v1.6 report
            q16 = next(q for q in v16_preds if q["query_id"] == qid)
            if qtype == "synthesis" and q16.get("promoted_pair") == [gold_p, gold_c]:
                correct_blocked.append((qid, tuple(q16["promoted_pair"])))

    print("\n### 2. Diagnostic Verifier Counts")
    print(f"  • Correct pairs promoted: {len(correct_promoted)} -> {correct_promoted}")
    print(f"  • Wrong pairs promoted:   {len(wrong_promoted)} -> {wrong_promoted}")
    print(f"  • Correct pairs blocked:  {len(correct_blocked)} -> {correct_blocked}")
    print(f"  • Previously correct arbitration primaries displaced: {len(corr_arb_displaced)} -> {corr_arb_displaced}")
    print(f"  • v1.6 Additional Synthesis Successes Survived: {len(survived_v16_extra)}/5 -> {survived_v16_extra}")

    # 3. Trace for Synthesis Queries
    print("\n### 3. Diagnosis for Each Synthesis Query")
    for t in synth_traces:
        qid = t["query_id"]
        gold_p = t["gold_primary"]
        gold_c = t["gold_competing"]
        is_hit = (t["final_top3"][:2] == [gold_p, gold_c])
        status = "HIT" if is_hit else "MISS"
        r_p = t["router_prob"]
        v_p = f"{t['verifier_prob']:.4f}" if t["verifier_prob"] is not None else "N/A"
        prom = f"Promoted {t['promoted_pair']}" if t["pair_promoted"] else f"Declined ({t['decline_reason']})"
        print(f"  • {qid} [{status}]: Gold=({gold_p}, {gold_c}) | Top3={t['final_top3']} | "
              f"r_prob={r_p:.3f} v_prob={v_p} (tau={v_thresh:.2f}) | {prom}")

    # 4. Formal Promotion Criteria Check
    r3_all = m_verified["all"]["recall_at_3"]
    flips_all = m_verified["all"]["flip"][0]
    flips_new = m_verified["new"]["flip"][0]
    abst_p = m_verified["all"]["abstention_precision"][0] == m_verified["all"]["abstention_precision"][1] == 4
    abst_r = m_verified["all"]["abstention_recall"][0] == m_verified["all"]["abstention_recall"][1] == 4
    ord_synth = m_verified["all"]["ordered_synthesis"][0]

    cond1 = r3_all > 0.6447
    cond2 = (flips_all >= 5) and (flips_new >= 3)
    cond3 = abst_p and abst_r
    cond4 = ord_synth >= 2

    promoted = cond1 and cond2 and cond3 and cond4

    print("\n### 4. Formal Promotion Criteria Evaluation")
    print(f"  1. Overall Recall@3 > 64.47%: {100*r3_all:.2f}% -> {'PASS' if cond1 else 'FAIL'}")
    print(f"  2. Flips >= 5/14 overall & >= 3/8 on new: {flips_all}/14, {flips_new}/8 -> {'PASS' if cond2 else 'FAIL'}")
    print(f"  3. Abstentions 4/4 P, 4/4 R: {m_verified['all']['abstention_precision']}, {m_verified['all']['abstention_recall']} -> {'PASS' if cond3 else 'FAIL'}")
    print(f"  4. Ordered Synthesis >= 2/10: {ord_synth}/10 -> {'PASS' if cond4 else 'FAIL'}")
    print(f"\n  Final Promotion Decision: {'PROCEED WITH PROMOTION' if promoted else 'DO NOT PROMOTE'}")

    # 5. Save Machine-Readable Evaluation Report
    out_file = ARTIFACTS_V1_6 / "pair_verifier_eval_report.json"
    report = {
        "timestamp": "2026-11-21T13:00:00Z",
        "verifier_threshold": v_thresh,
        "verifier_config": res["verifier_config"],
        "metrics": {
            "v1_3_baseline": m_v13,
            "ungated_v1_6": m_v16,
            "diagnostic_oracle": m_oracle,
            "deployable_verifier": m_verified,
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
        "diagnostic_counts": {
            "correct_pairs_promoted": [t for t in correct_promoted],
            "wrong_pairs_promoted": [t for t in wrong_promoted],
            "correct_pairs_blocked": [t for t in correct_blocked],
            "corr_arb_displaced": corr_arb_displaced,
            "v1_6_extra_synth_survived": survived_v16_extra,
        },
        "predictions": verified_preds,
    }
    out_file.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
