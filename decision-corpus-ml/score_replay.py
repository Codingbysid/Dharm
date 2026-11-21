"""Deterministic score replay and edge tracing engine for Decision Corpus.

Reconstructs all ranking stages (pre-graph, post-flip, post-workflow, final top3)
from saved numeric scores without loading neural network models or external dependencies.
Supports both legacy (v1.4) and corrected directed-edge graph semantics (v1.5).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def load_corpus_graph(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads corpus records and directed graph structure."""
    decisions_file = data_dir / "decisions_v0.2.jsonl"
    rows = [json.loads(line) for line in decisions_file.read_text(encoding="utf8").splitlines()]
    records = {r["id"]: r for r in rows}
    return records


def min_max_normalize(scores_dict: Dict[str, float], subset_keys: List[str]) -> Dict[str, float]:
    """Min-max normalizes scores across a subset of candidate keys."""
    vals = [scores_dict[k] for k in subset_keys]
    mi, ma = min(vals), max(vals)
    if ma > mi:
        return {k: (scores_dict[k] - mi) / (ma - mi) for k in subset_keys}
    return {k: 1.0 for k in subset_keys}


def deterministic_rank(scores: Dict[str, float]) -> List[str]:
    """Sort candidates descending by score, breaking ties ascending by passage ID."""
    return sorted(scores.keys(), key=lambda k: (-scores[k], k))


def replay_query(
    q_entry: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    w_rrf: float = 0.55,
    w_ce: float = 0.30,
    w_nli: float = 0.15,
    rule_mode: str = "retain_only",  # 'retain_only', 'composite', 'none'
    edge_mode: str = "directed_v1_5",  # 'directed_v1_5', 'legacy_v1_4'
    workflow_active: bool = True,
    flip_gate: float = 0.68,
    flip_delta: float = 0.04,
    wf_max_thresh: float = 0.68,
    wf_min_thresh: float = 0.35,
) -> Dict[str, Any]:
    """Executes deterministic ranking replay for a single dilemma query."""
    qid = q_entry["query_id"]
    pool = q_entry["pool"]
    seeds = q_entry["seeds"]
    abstained = bool(q_entry["abstained"])

    # Stage 0: Select rule alignment scores
    if rule_mode == "composite":
        rule_norm = q_entry["nli_rule_composite_norm"]
    elif rule_mode == "retain_only":
        rule_norm = q_entry["nli_rule_retain_norm"]
    elif rule_mode == "none":
        rule_norm = {k: 0.0 for k in pool}
    else:
        raise ValueError(f"Unknown rule_mode: {rule_mode}")

    # Stage 1: Base 3-way blended score
    base_scores = {
        k: w_rrf * q_entry["rrf_norm"][k] + w_ce * q_entry["ce_norm"][k] + w_nli * rule_norm[k]
        for k in pool
    }
    pre_graph_ranks = deterministic_rank(base_scores)

    # Stage 2: Directional flip pass
    final_scores = base_scores.copy()
    edge_traces: List[Dict[str, Any]] = []

    r_scores = q_entry["r_scores"]
    f_scores = q_entry["f_scores"]

    if edge_mode == "legacy_v1_4":
        # Legacy: greedy in-place mutation over unordered pairs
        processed_pairs = set()
        for u in pool:
            v = records[u][CONTRAST]["id"]
            if v in pool:
                pair_key = tuple(sorted([u, v]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                topical_max = max(base_scores[u], base_scores[v])
                gate_passed = bool(topical_max >= flip_gate)
                r_u, f_u = r_scores[u], f_scores[u]
                r_v, f_v = r_scores[v], f_scores[v]
                net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))

                delta_applied = 0.0
                u_before, v_before = final_scores[u], final_scores[v]
                if gate_passed:
                    if net_uv > flip_delta:
                        final_scores[v] = max(final_scores[v], topical_max + 0.18 * net_uv)
                        final_scores[u] = min(final_scores[u], topical_max - 0.06 * net_uv)
                        delta_applied = net_uv
                    elif net_uv < -flip_delta:
                        net_vu = -net_uv
                        final_scores[u] = max(final_scores[u], topical_max + 0.18 * net_vu)
                        final_scores[v] = min(final_scores[v], topical_max - 0.06 * net_vu)
                        delta_applied = -net_vu

                edge_traces.append({
                    "source_id": u,
                    "target_id": v,
                    "is_reciprocal": (records[v][CONTRAST]["id"] == u),
                    "raw_r_source": r_u,
                    "raw_f_source": f_u,
                    "raw_r_target": r_v,
                    "raw_f_target": f_v,
                    "topical_max": topical_max,
                    "gate_passed": gate_passed,
                    "net_delta": net_uv,
                    "source_before": u_before,
                    "source_after": final_scores[u],
                    "target_before": v_before,
                    "target_after": final_scores[v],
                })

    elif edge_mode == "directed_v1_5":
        # Corrected directed semantics: compute from immutable snapshot, order-invariant
        snapshot = base_scores.copy()
        planned_transfers: List[Tuple[str, str, float, float]] = []

        for u in sorted(pool):
            v = records[u][CONTRAST]["id"]
            if v in pool:
                is_reciprocal = (records[v][CONTRAST]["id"] == u)
                topical_max = max(snapshot[u], snapshot[v])
                gate_passed = bool(topical_max >= flip_gate)

                r_u, f_u = r_scores[u], f_scores[u]
                if is_reciprocal:
                    r_v, f_v = r_scores[v], f_scores[v]
                    net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))
                else:
                    # Nonreciprocal: evaluate only u's conditions to move from u to v
                    r_v, f_v = None, None
                    net_uv = f_u - r_u

                delta_applied = False
                if gate_passed and net_uv > flip_delta:
                    delta_applied = True
                    planned_transfers.append((u, v, topical_max, net_uv))

                edge_traces.append({
                    "source_id": u,
                    "target_id": v,
                    "is_reciprocal": is_reciprocal,
                    "raw_r_source": r_u,
                    "raw_f_source": f_u,
                    "raw_r_target": r_v,
                    "raw_f_target": f_v,
                    "topical_max": topical_max,
                    "gate_passed": gate_passed,
                    "net_delta": net_uv,
                    "delta_applied": delta_applied,
                    "source_before": snapshot[u],
                    "target_before": snapshot[v],
                })

        # Apply transfers deterministically from snapshot
        # 1. Apply all target promotions
        for u, v, tmax, net in planned_transfers:
            final_scores[v] = max(final_scores[v], tmax + 0.18 * net)

        # 2. Apply all source demotions
        for u, v, tmax, net in planned_transfers:
            final_scores[u] = min(final_scores[u], tmax - 0.06 * net)

        # Update trace after-scores
        for tr in edge_traces:
            tr["source_after"] = final_scores[tr["source_id"]]
            tr["target_after"] = final_scores[tr["target_id"]]
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    # Immutable post-flip snapshot before workflow pass
    post_flip_scores = final_scores.copy()
    post_flip_ranks = deterministic_rank(post_flip_scores)

    # Stage 3: Workflow synthesis ordering pass
    workflow_traces: List[Dict[str, Any]] = []
    if workflow_active:
        workflow_pairs = set()
        for u in pool:
            c = records[u][WORKFLOW]["id"]
            if c in pool:
                pair_key = tuple(sorted([u, c]))
                if pair_key in workflow_pairs:
                    continue
                workflow_pairs.add(pair_key)

                max_sc = max(base_scores[u], base_scores[c])
                min_sc = min(base_scores[u], base_scores[c])
                gate_passed = bool(max_sc >= wf_max_thresh and min_sc >= wf_min_thresh)

                seq = records[u][WORKFLOW]["sequence_role"]
                first, second = (u, c) if seq.startswith("Follow with") else (c, u)
                f_before, s_before = final_scores[first], final_scores[second]

                if gate_passed:
                    pair_top = max(base_scores[first], base_scores[second])
                    final_scores[first] = max(final_scores[first], pair_top + 0.10)
                    final_scores[second] = max(final_scores[second], pair_top + 0.06)

                workflow_traces.append({
                    "first": first,
                    "second": second,
                    "max_base": max_sc,
                    "min_base": min_sc,
                    "gate_passed": gate_passed,
                    "first_before": f_before,
                    "first_after": final_scores[first],
                    "second_before": s_before,
                    "second_after": final_scores[second],
                })

    post_workflow_ranks = deterministic_rank(final_scores)
    final_top3 = post_workflow_ranks[:3]
    primary = "Abstain" if abstained else final_top3[0]

    return {
        "query_id": qid,
        "query_type": q_entry["query_type"],
        "pair_group": q_entry.get("pair_group"),
        "gold_primary_id": q_entry.get("gold_primary_id"),
        "gold_competing_id": q_entry.get("gold_competing_id"),
        "seeds": seeds,
        "pool": pool,
        "abstained": abstained,
        "abstention_prob": q_entry["abstention_prob"],
        "rrf_norm": q_entry["rrf_norm"],
        "ce_norm": q_entry["ce_norm"],
        "nli_rule_norm": rule_norm,
        "pre_graph_scores": base_scores,
        "pre_graph_ranks": pre_graph_ranks,
        "edge_traces": edge_traces,
        "post_flip_scores": post_flip_scores,
        "post_flip_ranks": post_flip_ranks,
        "workflow_traces": workflow_traces,
        "post_workflow_scores": final_scores,
        "post_workflow_ranks": post_workflow_ranks,
        "top3": final_top3,
        "primary": primary,
    }


def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes benchmark metrics: Recall@3, flips, abstention, ordered synthesis, pool coverage."""
    recall, ordered, pairs = [], [], defaultdict(list)
    tp = pred = gold = 0
    pool_primary = 0
    pool_both = 0

    for r in predictions:
        p, c = r["gold_primary_id"], r.get("gold_competing_id")
        pred += int(r["abstained"])
        if r["query_type"] == "abstention":
            gold += 1
            tp += int(r["abstained"])
            continue

        relevant = {p}
        in_p = p in r["pool"]
        in_c = c in r["pool"] if c else True
        if in_p:
            pool_primary += 1
        if in_p and in_c:
            pool_both += 1

        if r["query_type"] == "synthesis":
            relevant.add(c)
            ordered.append(not r["abstained"] and r["top3"][:2] == [p, c])
        elif r.get("pair_group"):
            pairs[r["pair_group"]].append(r["primary"] == p)

        recall.append(len(relevant.intersection(r["top3"])) / len(relevant))

    return {
        "recall_at_3": sum(recall) / len(recall) if recall else 0.0,
        "flip": [sum(all(g) for g in pairs.values()), len(pairs)],
        "abstention_precision": [tp, pred],
        "abstention_recall": [tp, gold],
        "ordered_synthesis": [sum(ordered), len(ordered)],
        "pool_primary_coverage": [pool_primary, len(recall)],
        "pool_both_coverage": [pool_both, len(recall)],
    }


def format_metrics(m: Dict[str, Any]) -> str:
    """Formats metrics row as Markdown string."""
    def frac(k):
        a, b = m[k]
        return f"{a}/{b}" if b else "undefined"
    return (
        f"{m['recall_at_3'] * 100:6.2f}% | "
        f"{frac('flip'):<5} | "
        f"{frac('abstention_precision'):<5} | "
        f"{frac('abstention_recall'):<5} | "
        f"{frac('ordered_synthesis'):<5} | "
        f"{frac('pool_primary_coverage'):<5}"
    )


def verify_replay_parity(cache: Dict[str, Any], records: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verifies that legacy_v1_4 mode strictly reproduces saved v1.4 predictions."""
    mismatches = []
    for q in cache["queries"]:
        res = replay_query(
            q,
            records,
            w_rrf=0.45,
            w_ce=0.35,
            w_nli=0.20,
            rule_mode="composite",
            edge_mode="legacy_v1_4",
            workflow_active=True,
        )
        qid = q["query_id"]
        if res["top3"] != q["v1_4_top3"]:
            mismatches.append(f"{qid} top3: replay={res['top3']} vs expected={q['v1_4_top3']}")
        max_diff = max(abs(res["post_workflow_scores"][k] - q["v1_4_final_scores"][k]) for k in q["pool"])
        if max_diff > 1e-6:
            mismatches.append(f"{qid} score diff: max_diff={max_diff:.3e}")
    return len(mismatches) == 0, mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_file", type=Path, default=Path("decision-corpus-ml/artifacts-v1.5/score_replay_data.json"))
    ap.add_argument("--data_dir", type=Path, default=Path("decision-corpus-ml/data"))
    ap.add_argument("--query", type=str, default=None, help="Inspect a specific query (e.g. Q01)")
    ap.add_argument("--trace_edges", action="store_true", help="Print all edge trace rows for inspected query")
    ap.add_argument("--weights", type=str, default="0.55,0.30,0.15", help="Weights as w_rrf,w_ce,w_nli")
    ap.add_argument("--rule_mode", choices=["composite", "retain_only", "none"], default="retain_only")
    ap.add_argument("--wf_min_thresh", type=float, default=0.35, help="Minimum endpoint score threshold for workflow ordering pass")
    ap.add_argument("--wf_max_thresh", type=float, default=0.68, help="Maximum endpoint score threshold for workflow ordering pass")
    ap.add_argument("--no_workflow", action="store_true", help="Disable workflow synthesis ordering")
    ap.add_argument("--verify_parity", action="store_true", help="Verify exact parity against v1.4 baseline")
    args = ap.parse_args()

    records = load_corpus_graph(args.data_dir)
    cache = json.loads(args.cache_file.read_text(encoding="utf8"))
    weights = [float(x) for x in args.weights.split(",")]

    if args.verify_parity:
        passed, mismatches = verify_replay_parity(cache, records)
        if passed:
            print("✓ Parity verification PASSED: Replay exactly matches 42/42 v1.4 predictions (max diff < 1e-6).")
            sys.exit(0)
        else:
            print("✗ Parity verification FAILED with mismatches:")
            for m in mismatches:
                print(" ", m)
            sys.exit(1)

    if args.query:
        target_q = next((q for q in cache["queries"] if q["query_id"] == args.query), None)
        if not target_q:
            print(f"Query {args.query} not found in cache.")
            sys.exit(1)

        res = replay_query(
            target_q,
            records,
            w_rrf=weights[0],
            w_ce=weights[1],
            w_nli=weights[2],
            rule_mode=args.rule_mode,
            edge_mode=args.edge_mode,
            workflow_active=not args.no_workflow,
            wf_min_thresh=args.wf_min_thresh,
            wf_max_thresh=args.wf_max_thresh,
        )

        print(f"=== Score Replay for {res['query_id']} ({res['query_type']}) ===")
        print(f"Seeds: {res['seeds'][:5]}... | Pool size: {len(res['pool'])}")
        print(f"Gold primary: {res['gold_primary_id']} | Gold competing: {res['gold_competing_id']}")
        print(f"Abstained: {res['abstained']} (prob={res['abstention_prob']:.4f})")
        print("\nRanking Progression:")
        print(f"  Pre-graph rank #1..3:     {res['pre_graph_ranks'][:3]}")
        print(f"  Post-flip rank #1..3:     {res['post_flip_ranks'][:3]}")
        print(f"  Post-workflow rank #1..3: {res['post_workflow_ranks'][:3]}")
        print(f"  Final top3:               {res['top3']}")
        print(f"  Primary:                  {res['primary']}")

        if args.trace_edges:
            print("\nScored Directed Edge Traces:")
            print("| Source | Target | Recip | Raw r(u) | Raw f(u) | Raw r(v) | Raw f(v) | Topical Max | Gate | Net Delta | Applied | u: Before->After | v: Before->After |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for t in res["edge_traces"]:
                r_v_str = f"{t['raw_r_target']:.3f}" if t.get("raw_r_target") is not None else "N/A"
                f_v_str = f"{t['raw_f_target']:.3f}" if t.get("raw_f_target") is not None else "N/A"
                app_str = "YES" if t.get("delta_applied", t.get("gate_passed")) else "NO"
                print(
                    f"| {t['source_id']} | {t['target_id']} | {t['is_reciprocal']} | "
                    f"{t['raw_r_source']:+.3f} | {t['raw_f_source']:+.3f} | {r_v_str} | {f_v_str} | "
                    f"{t['topical_max']:.3f} | {t['gate_passed']} | {t['net_delta']:+.3f} | {app_str} | "
                    f"{t['source_before']:.3f}->{t['source_after']:.3f} | "
                    f"{t['target_before']:.3f}->{t['target_after']:.3f} |"
                )
        return

    # Run replay on all 42 queries and evaluate
    preds = [
        replay_query(
            q,
            records,
            w_rrf=weights[0],
            w_ce=weights[1],
            w_nli=weights[2],
            rule_mode=args.rule_mode,
            edge_mode=args.edge_mode,
            workflow_active=not args.no_workflow,
            wf_min_thresh=args.wf_min_thresh,
            wf_max_thresh=args.wf_max_thresh,
        )
        for q in cache["queries"]
    ]

    slices = {
        "all": preds,
        "original": preds[:18],
        "new": preds[18:],
    }
    metrics = {k: evaluate(v) for k, v in slices.items()}

    print(f"\nReplay Configuration: weights={weights}, rule_mode={args.rule_mode}, edge_mode={args.edge_mode}, workflow={not args.no_workflow}")
    print("| Slice    | Recall@3 | Flips | Abst P | Abst R | Ordered Synth | Primary Pool |")
    print("|---|---|---|---|---|---|---|")
    for s in ("all", "original", "new"):
        print(f"| {s:<8} | {format_metrics(metrics[s])} |")


if __name__ == "__main__":
    main()
