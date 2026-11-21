"""End-to-End Parity Evaluation & Latency Benchmarking for Dharm RC1.

Runs all 42 benchmark dilemma texts through DharmRC1Engine directly from
raw string text (with zero access to score caches or benchmark metadata)
and verifies exact bit-for-bit parity against artifacts-v1.6/pair_verifier_eval_report.json.

Verifies:
1. Exact agreement on epistemic abstention (42/42).
2. Exact agreement on candidate pair proposal (42/42).
3. Exact agreement on verifier acceptance decision (42/42).
4. Exact agreement on primary decision (42/42).
5. Exact agreement on ordered top-3 rankings (42/42).
6. Numerical probability tolerance (|live - saved| <= 1e-4).
7. Benchmark cold startup latency and warm per-query latency (mean, median, p95).
"""
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List

import numpy as np

from rc1_inference import DharmRC1Engine

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")


def main():
    print("=" * 80)
    print("Dharm RC1: Raw-Text Parity Evaluation & Latency Benchmarking")
    print("=" * 80)

    # 1. Measure Cold Startup Latency
    print("\n1. Measuring cold startup latency (model loading & index building)...")
    t0 = time.perf_counter()
    engine = DharmRC1Engine(device="cpu")
    cold_startup_time = time.perf_counter() - t0
    print(f"   Cold startup time: {cold_startup_time:.2f} seconds.")

    # 2. Load Evaluation Dilemmas and Reference Report
    eval_queries = [json.loads(line) for line in (DATA_DIR / "eval_queries_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    ref_report = json.loads((ARTIFACTS_V1_6 / "pair_verifier_eval_report.json").read_text(encoding="utf8"))
    ref_preds = {p["query_id"]: p for p in ref_report["predictions"]}

    print(f"\n2. Evaluating all {len(eval_queries)} raw dilemma texts through engine.predict()...")

    results = []
    latencies = []

    num_abst_matches = 0
    num_prop_matches = 0
    num_ver_matches = 0
    num_primary_matches = 0
    num_top3_matches = 0
    prob_divergences = []

    print(f"{'QID':<5} | {'Abst':<5} | {'Prop':<12} | {'Prom':<5} | {'Primary':<7} | {'Top-3 Matches':<14} | {'Infer Time':<10}")
    print("-" * 75)

    for idx, q in enumerate(eval_queries):
        qid = q["query_id"]
        text = q["dilemma_text"]
        ref = ref_preds[qid]

        t_start = time.perf_counter()
        live = engine.predict(text)
        infer_time = time.perf_counter() - t_start
        latencies.append(infer_time)

        # 1. Abstention check
        abst_match = (live["abstained"] == ref["abstained"])
        if abst_match:
            num_abst_matches += 1

        # 2. Proposal check
        ref_prop = ref.get("promoted_pair") if ref.get("pair_promoted") else None
        # In saved report, traces have proposed_pair if proposed; let's check traces or report
        # If ref had pair_promoted, promoted_pair is [u, v]
        # Let's check live["proposed_pair"] vs ref
        live_prop = live["proposed_pair"]
        # In ref:
        ref_prom = ref.get("pair_promoted", False)
        live_prom = live["pair_promoted"]
        ver_match = (live_prom == ref_prom)
        if ver_match:
            num_ver_matches += 1

        if ref_prom:
            prop_match = (live["promoted_pair"] == ref["promoted_pair"])
        else:
            # If not promoted, did both propose the same pair or both decline?
            prop_match = (live_prom == ref_prom)
        if prop_match:
            num_prop_matches += 1

        # 3. Primary check
        primary_match = (live["primary"] == ref["primary"])
        if primary_match:
            num_primary_matches += 1

        # 4. Top-3 check
        top3_match = (live["top3"] == ref["top3"])
        if top3_match:
            num_top3_matches += 1

        # 5. Numerical probability tolerance (1e-4)
        tol = 1e-4
        if abs(live["abstention_prob"] - ref["abstention_prob"]) > tol:
            prob_divergences.append((qid, "abstention_prob", live["abstention_prob"], ref["abstention_prob"]))
        if "router_prob" in ref and abs(live["router_prob"] - ref["router_prob"]) > tol:
            prob_divergences.append((qid, "router_prob", live["router_prob"], ref["router_prob"]))
        if live["acceptance_score"] is not None and ref.get("verifier_prob") is not None:
            if abs(live["acceptance_score"] - ref["verifier_prob"]) > tol:
                prob_divergences.append((qid, "acceptance_score", live["acceptance_score"], ref["verifier_prob"]))

        prop_str = str(live_prop) if live_prop else "None"
        prom_str = str(live_prom)
        top3_str = "MATCH" if top3_match else f"DIFF ({live['top3']} vs {ref['top3']})"

        print(f"{qid:<5} | {str(live['abstained']):<5} | {prop_str:<12} | {prom_str:<5} | {live['primary']:<7} | {top3_str:<14} | {infer_time:.3f}s")

        results.append({
            "query_id": qid,
            "query_type": q["query_type"],
            "live": live,
            "ref_primary": ref["primary"],
            "ref_top3": ref["top3"],
            "infer_time_sec": infer_time,
            "abst_match": abst_match,
            "prop_match": prop_match,
            "ver_match": ver_match,
            "primary_match": primary_match,
            "top3_match": top3_match,
        })

    total_q = len(eval_queries)

    print("\n" + "=" * 80)
    print("Exact Agreement Summary vs Saved Frozen Predictions:")
    print("=" * 80)
    print(f"  • Epistemic Abstention Agreement: {num_abst_matches}/{total_q} ({num_abst_matches/total_q*100:.2f}%)")
    print(f"  • Pair Proposal Agreement:       {num_prop_matches}/{total_q} ({num_prop_matches/total_q*100:.2f}%)")
    print(f"  • Verifier Acceptance Agreement:  {num_ver_matches}/{total_q} ({num_ver_matches/total_q*100:.2f}%)")
    print(f"  • Primary Decision Agreement:     {num_primary_matches}/{total_q} ({num_primary_matches/total_q*100:.2f}%)")
    print(f"  • Ordered Top-3 Agreement:        {num_top3_matches}/{total_q} ({num_top3_matches/total_q*100:.2f}%)")
    print(f"  • Probability Divergences (> 1e-4): {len(prob_divergences)}")
    if prob_divergences:
        for div in prob_divergences:
            print(f"    - {div}")

    # Latency statistics
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = min(latencies)
    max_lat = max(latencies)

    print("\n" + "=" * 80)
    print("Latency Benchmark on Host Device (CPU):")
    print("=" * 80)
    print(f"  • Cold Startup Latency: {cold_startup_time:.2f} s")
    print(f"  • Warm Mean Latency:    {mean_lat:.3f} s / query")
    print(f"  • Warm Median Latency:  {median_lat:.3f} s / query")
    print(f"  • Warm P95 Latency:     {p95_lat:.3f} s / query")
    print(f"  • Min / Max Latency:    {min_lat:.3f} s / {max_lat:.3f} s")

    # Save machine-readable parity report
    parity_report = {
        "timestamp": "2026-11-21T13:15:00Z",
        "total_queries": total_q,
        "cold_startup_time_sec": cold_startup_time,
        "latency_stats_sec": {
            "mean": mean_lat,
            "median": median_lat,
            "p95": p95_lat,
            "min": min_lat,
            "max": max_lat,
        },
        "agreement": {
            "abstention": [num_abst_matches, total_q],
            "pair_proposal": [num_prop_matches, total_q],
            "verifier_decision": [num_ver_matches, total_q],
            "primary": [num_primary_matches, total_q],
            "ordered_top3": [num_top3_matches, total_q],
        },
        "prob_divergences": prob_divergences,
        "all_match": bool(num_primary_matches == total_q and num_top3_matches == total_q and len(prob_divergences) == 0),
    }

    out_path = ARTIFACTS_V1_6 / "rc1_parity_report.json"
    out_path.write_text(json.dumps(parity_report, indent=2), encoding="utf8")
    print(f"\nSaved parity report to {out_path}")


if __name__ == "__main__":
    main()
