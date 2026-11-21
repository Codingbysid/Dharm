"""Corpus-Derived Development Split & Pair Promotion Calibration.

Strict Isolation:
Keeps all benchmark queries (eval_queries_v0.2.jsonl), benchmark query text,
IDs, query types, and gold labels strictly outside training, calibration,
and threshold selection.

Uses only:
  - decisions_v0.2.jsonl
  - chanakya_daily_supplement.jsonl

Workflow:
1. Extract 54 complementary pairs from corpus.
2. Form 54 grouped dev folds (GroupKFold by pair_group).
3. Generate positive 2-step synthesis dilemmas and negative single-step dilemmas.
4. Run cross-encoder scoring for single cards and joint pair cards.
5. Grid search promotion threshold across 5 folds.
6. Output frozen configuration and SHA-256 hash.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_DIR = Path("decision-corpus-ml/artifacts-v1.6")

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


def build_dev_dataset(records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for r in records.values():
        u = r["id"]
        role = r[WORKFLOW]["sequence_role"]
        v = r[WORKFLOW]["id"]
        if role.startswith("Follow with"):
            pairs.append((u, v))
    pairs.sort()

    dev_groups = []
    for p_idx, (u, v) in enumerate(pairs):
        u_rec, v_rec = records[u], records[v]
        u_scens = [s["scenario"] for s in u_rec["scenarios_json"]]
        v_scens = [s["scenario"] for s in v_rec["scenarios_json"]]

        # Two positive 2-step synthesis dilemmas
        q_pos_1 = (f"{u_scens[0]} After resolving this initial matter, "
                   f"{v_scens[0][:1].lower() + v_scens[0][1:]} "
                   f"What sequence of actions should be taken?")
        q_pos_2 = (f"{u_scens[1]} Following that initial step, "
                   f"{v_scens[1][:1].lower() + v_scens[1][1:]} "
                   f"How should the team proceed in order?")

        # Two negative single-step dilemmas
        q_neg_u = u_scens[0]
        q_neg_v = v_scens[0]

        # One negative arbitration dilemma
        comp_u = u_rec[CONTRAST]["id"]
        cond_u = u_rec[CONTRAST]["condition"]
        q_neg_arb = f"{u_scens[1]} Note that {cond_u[:1].lower() + cond_u[1:]}. Which principle applies?"

        # Candidate pool for this pair group: {u, v, comp_u, comp_v, and 1-hop neighbors}
        pool = set([u, v, comp_u, records[v][CONTRAST]["id"]])
        for node in list(pool):
            pool.add(records[node][WORKFLOW]["id"])
            pool.add(records[node][CONTRAST]["id"])
        pool_list = sorted(pool)

        dev_groups.append({
            "group_id": f"PAIR_{p_idx:02d}",
            "first_id": u,
            "second_id": v,
            "pool": pool_list,
            "queries": [
                {"id": f"DEV_{p_idx:02d}_P1", "type": "synthesis", "text": q_pos_1, "gold": (u, v)},
                {"id": f"DEV_{p_idx:02d}_P2", "type": "synthesis", "text": q_pos_2, "gold": (u, v)},
                {"id": f"DEV_{p_idx:02d}_N1", "type": "single_u", "text": q_neg_u, "gold": u},
                {"id": f"DEV_{p_idx:02d}_N2", "type": "single_v", "text": q_neg_v, "gold": v},
                {"id": f"DEV_{p_idx:02d}_N3", "type": "single_arb", "text": q_neg_arb, "gold": u},
            ]
        })
    return dev_groups


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


def main():
    device = "cpu"
    print(f"Loading corpus from {DATA_DIR}...")
    records = load_corpus(DATA_DIR)
    dev_groups = build_dev_dataset(records)
    print(f"Constructed {len(dev_groups)} pair groups with {sum(len(g['queries']) for g in dev_groups)} queries.")

    print("Loading cross-encoder model BAAI/bge-reranker-base...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
    model.eval()

    passage_cards = {k: passage_card(records[k]) for k in records}

    # Pre-score all queries against their pool candidates and pool complementary pairs
    print("Scoring development queries with cross-encoder...")
    scored_data = []

    for g in dev_groups:
        pool = g["pool"]
        # Enumerate ordered pairs in pool
        pairs_in_pool = []
        for u in pool:
            wf = records[u][WORKFLOW]
            v = wf["id"]
            if v in pool and wf["sequence_role"].startswith("Follow with"):
                pairs_in_pool.append((u, v))

        for q in g["queries"]:
            q_text = q["text"]
            # Single passage pairs
            single_inputs = [[q_text, passage_cards[k]] for k in pool]
            single_scores = batch_score_ce(model, tokenizer, single_inputs, device)
            s_dict = dict(zip(pool, single_scores))

            # Joint pair card inputs
            joint_inputs = [[q_text, pair_joint_card(records[u], records[v])] for u, v in pairs_in_pool]
            joint_scores = batch_score_ce(model, tokenizer, joint_inputs, device) if joint_inputs else []
            j_dict = dict(zip(pairs_in_pool, joint_scores))

            scored_data.append({
                "group_id": g["group_id"],
                "query_id": q["id"],
                "type": q["type"],
                "gold": q["gold"],
                "pool": pool,
                "single_scores": s_dict,
                "pairs_in_pool": pairs_in_pool,
                "joint_scores": j_dict,
            })

    print(f"Scoring complete. Total scored queries: {len(scored_data)}.")

    # 5-Fold GroupKFold cross-validation
    group_ids = sorted(list({d["group_id"] for d in scored_data}))
    np.random.seed(42)
    shuffled_groups = group_ids.copy()
    np.random.shuffle(shuffled_groups)

    k_folds = 5
    folds = [shuffled_groups[i::k_folds] for i in range(k_folds)]

    # Grid search candidate parameters
    # Rule formulation:
    # A pair (u, v) is candidate for promotion if:
    #   1. s_joint(u, v) - max(s_u, s_v) >= tau_self
    #   2. s_joint(u, v) - max_pool_s >= tau_pool
    # If multiple pairs qualify, pick the one with highest s_joint.
    # If a pair qualifies:
    #   reorder ranking: [u, v] at top, followed by remaining candidates in fallback order.
    # If none qualifies:
    #   fallback to single ranking.

    tau_self_candidates = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]
    tau_pool_candidates = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5]

    best_cv_score = -999.0
    best_params = None
    results_grid = []

    print("\nRunning 5-Fold GroupKFold Grid Search...")
    for t_self in tau_self_candidates:
        for t_pool in tau_pool_candidates:
            fold_scores = []
            fold_synth_accs = []
            fold_neg_preserves = []

            for fold_idx in range(k_folds):
                val_groups = set(folds[fold_idx])
                val_items = [d for d in scored_data if d["group_id"] in val_groups]

                synth_correct = 0
                synth_total = 0
                neg_preserved = 0
                neg_total = 0

                for item in val_items:
                    s_dict = item["single_scores"]
                    j_dict = item["joint_scores"]
                    pool = item["pool"]
                    max_pool_s = max(s_dict.values())
                    fallback_rank = sorted(pool, key=lambda k: (-s_dict[k], k))

                    promoted_pair = None
                    best_j_val = -999.0
                    for (u, v), j_val in j_dict.items():
                        max_self = max(s_dict[u], s_dict[v])
                        if (j_val - max_self >= t_self) and (j_val - max_pool_s >= t_pool):
                            if j_val > best_j_val:
                                best_j_val = j_val
                                promoted_pair = (u, v)

                    if promoted_pair is not None:
                        u, v = promoted_pair
                        final_rank = [u, v] + [k for k in fallback_rank if k not in (u, v)]
                    else:
                        final_rank = fallback_rank

                    if item["type"] == "synthesis":
                        synth_total += 1
                        gold_u, gold_v = item["gold"]
                        if final_rank[:2] == [gold_u, gold_v]:
                            synth_correct += 1
                    else:
                        neg_total += 1
                        gold_single = item["gold"]
                        if final_rank[0] == gold_single:
                            neg_preserved += 1

                synth_acc = synth_correct / synth_total if synth_total else 0.0
                neg_acc = neg_preserved / neg_total if neg_total else 0.0
                # Balanced objective: reward ordered synthesis while penalizing negative corruption
                score = synth_acc + 1.5 * neg_acc
                fold_scores.append(score)
                fold_synth_accs.append(synth_acc)
                fold_neg_preserves.append(neg_acc)

            avg_score = float(np.mean(fold_scores))
            avg_synth = float(np.mean(fold_synth_accs))
            avg_neg = float(np.mean(fold_neg_preserves))
            results_grid.append({
                "tau_self": t_self,
                "tau_pool": t_pool,
                "avg_score": avg_score,
                "avg_synth_acc": avg_synth,
                "avg_neg_preserved": avg_neg,
            })

            if avg_score > best_cv_score:
                best_cv_score = avg_score
                best_params = (t_self, t_pool, avg_score, avg_synth, avg_neg)

    print("\nTop 5 Parameter Configurations:")
    results_grid.sort(key=lambda r: -r["avg_score"])
    for r in results_grid[:5]:
        print(f"  tau_self={r['tau_self']:+.2f}, tau_pool={r['tau_pool']:+.2f} | "
              f"CV Score: {r['avg_score']:.4f} | Synth Acc: {r['avg_synth_acc']*100:.1f}% | "
              f"Neg Preserved: {r['avg_neg_preserved']*100:.1f}%")

    t_self_opt, t_pool_opt, opt_score, opt_synth, opt_neg = best_params
    print(f"\nOptimal Calibrated Parameters: tau_self={t_self_opt:+.2f}, tau_pool={t_pool_opt:+.2f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "model": "BAAI/bge-reranker-base",
        "card_format": "pair_joint_card_v1",
        "promotion_rule": "joint_margin_promotion",
        "tau_self": float(t_self_opt),
        "tau_pool": float(t_pool_opt),
        "cv_score": float(opt_score),
        "cv_synth_acc": float(opt_synth),
        "cv_neg_preserved": float(opt_neg),
        "cross_validation": "5-fold-GroupKFold-by-pair_group",
        "calibration_dataset": "decisions_v0.2-scenarios-54-pairs",
    }
    config_json = json.dumps(config_dict, sort_keys=True, indent=2)
    config_hash = hashlib.sha256(config_json.encode("utf8")).hexdigest()
    config_dict["config_sha256"] = config_hash

    out_file = ARTIFACTS_DIR / "pair_scorer_config.json"
    out_file.write_text(json.dumps(config_dict, indent=2), encoding="utf8")
    print(f"\nSaved calibrated configuration to {out_file}")
    print(f"Configuration SHA256: {config_hash}")


if __name__ == "__main__":
    main()
