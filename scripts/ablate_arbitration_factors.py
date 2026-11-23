"""Independent ablation of arbitration factors on fresh development pairs.
Tests both_transfer_supported (on/off) and NLI net threshold (0.05, 0.10, 0.15)
on pairs strictly disjoint from Q01–Q42 and L01–L24.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "decision-corpus-ml"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ML_DIR))

from rc1_inference import DharmRC1Engine, CONTRAST, WORKFLOW

def build_dev_arbitration_pairs(records):
    # Benchmark pair groups use: D001/D010, D007/D023, D015/D017, D025/D026, D035/D036,
    # D003/D006, D039/D040, D049/D050, D057/D058, D067/D068, D085/D086, D089/D090, D093/D094, D105/D106.
    benchmark_ids = {
        "D001", "D010", "D007", "D023", "D015", "D017", "D025", "D026", "D035", "D036",
        "D003", "D006", "D039", "D040", "D049", "D050", "D057", "D058", "D067", "D068",
        "D085", "D086", "D089", "D090", "D093", "D094", "D105", "D106",
        # Synthesis and abstention ids
        "D030", "D031", "D016", "D029", "D024", "D019", "D032", "D033", "D041", "D051",
        "D064", "D065", "D066", "D062", "D073", "D075", "D082", "D083",
    }
    # Also exclude transfer evaluation ids L01-L24
    transfer_eval = ROOT / "decision-corpus-ml" / "data" / "eval_transfer_v1.jsonl"
    if transfer_eval.exists():
        for line in transfer_eval.read_text(encoding="utf8").splitlines():
            if line.strip():
                for acc in json.loads(line).get("acceptable_ids", []):
                    benchmark_ids.add(acc)

    # Find reciprocal pairs completely disjoint from benchmark and transfer
    dev_pairs = []
    seen = set()
    for u, r in records.items():
        v = r[CONTRAST]["id"]
        if records[v][CONTRAST]["id"] == u and u < v:
            if u not in benchmark_ids and v not in benchmark_ids:
                dev_pairs.append((u, v))

    print(f"Found {len(dev_pairs)} fresh, untouched reciprocal pairs for dev evaluation.")

    # Create paired queries for each dev pair:
    # Query 1 targets u using scenario or retain_self_when
    # Query 2 targets v using scenario or flip_to_competing_when
    dev_queries = []
    for pair_idx, (u, v) in enumerate(dev_pairs[:12], 1):
        u_rec = records[u]
        v_rec = records[v]
        u_cond = u_rec[CONTRAST].get("retain_self_when") or u_rec["scenarios_json"][0]["scenario"]
        v_cond = u_rec[CONTRAST].get("flip_to_competing_when") or v_rec["scenarios_json"][0]["scenario"]

        q1 = f"An administrative body must decide: {u_rec['derived_principle']} Situation: {u_cond}"
        q2 = f"An administrative body must decide: {v_rec['derived_principle']} Situation: {v_cond}"
        dev_queries.append({"pair_id": f"DEV_{pair_idx}", "dilemma_text": q1, "gold_primary": u, "gold_comp": v})
        dev_queries.append({"pair_id": f"DEV_{pair_idx}", "dilemma_text": q2, "gold_primary": v, "gold_comp": u})

    return dev_queries


def run_ablation():
    engine = DharmRC1Engine(device="cpu")
    records = engine.records
    dev_queries = build_dev_arbitration_pairs(records)
    print(f"Constructed {len(dev_queries)} dev queries across {len(dev_queries)//2} pairs.\n")

    configs = [
        # (name, both_transfer_supported, nli_threshold)
        ("A: TransferFilter=ON,  NLI_thresh=0.15 (Candidate default)", True, 0.15),
        ("B: TransferFilter=ON,  NLI_thresh=0.10", True, 0.10),
        ("C: TransferFilter=ON,  NLI_thresh=0.05", True, 0.05),
        ("D: TransferFilter=OFF, NLI_thresh=0.15", False, 0.15),
        ("E: TransferFilter=OFF, NLI_thresh=0.10", False, 0.10),
        ("F: TransferFilter=OFF, NLI_thresh=0.05 (Legacy v1.3 style)", False, 0.05),
    ]

    print(f"{'Configuration':<45} | {'Primary Top-1':<15} | {'Both-Sides Pairs':<18}")
    print("-" * 82)

    for name, use_transfer_filter, nli_thresh in configs:
        # Evaluate dev queries
        p1_hits = 0
        pair_hits = {}

        for q in dev_queries:
            text = q["dilemma_text"]
            gold = q["gold_primary"]
            pid = q["pair_id"]

            # Run engine search and ranking manually simulating config
            search_res = engine.hybrid_index.search_with_scores([text], top_k=7)
            seeds, rrf_scores, dense_map, scen_map = search_res[0]
            from rc1_inference import expand_pool, min_max_normalize, batch_score_ce, batch_score_nli, both_transfer_supported
            pool = expand_pool(seeds, records)
            rrf_norm = min_max_normalize(rrf_scores, pool)

            ce_pairs = [[text, engine.rank_cards[k]] for k in pool]
            ce_raw = batch_score_ce(engine.ce_model, engine.ce_tok, ce_pairs, engine.device)
            s_dict = dict(zip(pool, ce_raw))
            ce_norm = min_max_normalize(s_dict, pool)
            base = {k: 0.55 * rrf_norm[k] + 0.45 * ce_norm[k] for k in pool}
            final = base.copy()

            r_nli_vals = batch_score_nli(engine.nli_model, engine.nli_tok, text, [engine.retain_texts[k] for k in pool],
                                         engine.entail_idx, engine.contra_idx, engine.device)
            f_nli_vals = batch_score_nli(engine.nli_model, engine.nli_tok, text, [engine.flip_texts[k] for k in pool],
                                         engine.entail_idx, engine.contra_idx, engine.device)
            r_scores = dict(zip(pool, r_nli_vals))
            f_scores = dict(zip(pool, f_nli_vals))

            processed_pairs = set()
            for u in pool:
                v = records[u][CONTRAST]["id"]
                if v in pool:
                    pair_key = tuple(sorted([u, v]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    topical_max = max(base[u], base[v])
                    cond = topical_max >= 0.72
                    if use_transfer_filter:
                        cond = cond and both_transfer_supported(dense_map, u, v)
                    if cond:
                        r_u, f_u = r_scores[u], f_scores[u]
                        r_v, f_v = r_scores[v], f_scores[v]
                        net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))
                        if net_uv > nli_thresh:
                            final[v] = max(final[v], topical_max)
                            final[u] = min(final[u], topical_max - 0.05 * net_uv)
                        elif net_uv < -nli_thresh:
                            net_vu = -net_uv
                            final[u] = max(final[u], topical_max)
                            final[v] = min(final[v], topical_max - 0.05 * net_vu)

            top1 = sorted(pool, key=lambda k: (-final[k], k))[0]
            is_hit = (top1 == gold)
            if is_hit:
                p1_hits += 1
            pair_hits.setdefault(pid, []).append(is_hit)

        both_sides = sum(1 for hits in pair_hits.values() if len(hits) == 2 and all(hits))
        total_p = len(pair_hits)
        print(f"{name:<45} | {p1_hits:>2}/{len(dev_queries)} ({p1_hits/len(dev_queries)*100:>5.1f}%) | {both_sides:>2}/{total_p} ({both_sides/total_p*100:>5.1f}%)")

if __name__ == "__main__":
    run_ablation()
