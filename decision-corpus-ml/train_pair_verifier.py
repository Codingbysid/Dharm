"""Train and calibrate the Pair-Promotion Verifier.

Strict Isolation:
Builds development proposals solely from corpus materials (decisions_v0.2.jsonl
and chanakya_daily_supplement.jsonl).
All 42 benchmark queries, query texts, IDs, query types, and gold labels
are strictly kept outside fitting, calibration, and threshold selection.

Design:
- Upstream Intent Feature: Out-of-fold query-intent probabilities from 5-fold GroupKFold router.
- Frozen v1.6 Pair Scorer: Generates candidate pair proposals using BAAI/bge-reranker-base (tau_self = -0.5, tau_pool = -1.0).
- Verifier Input Features:
  * Score margins: diff_self, diff_pool, diff_min, endpoint_span, j_score, max_self, min_self
  * Endpoint ranks: rank_u, rank_v, min_rank, max_rank, rank_sum, both_in_top2, both_in_top3, both_in_top5
  * Query-intent score: router_prob
- Ground Truth Target:
  * y = 1: Synthesis query where proposed pair equals gold ordered complementary pair.
  * y = 0: Single-step dilemma (spurious proposal) OR synthesis query where wrong pair was proposed.
- Grouping: 54 complementary pair groups (GroupKFold by pair_group). Zero leakage verified.
- Calibration: Search decision threshold tau_verifier on out-of-fold predictions to maximize F1 / balanced accuracy while strictly controlling false promotions.
- Artifacts: Saved to artifacts-v1.6/ with configuration SHA256 hash.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_DIR = Path("decision-corpus-ml/artifacts-v1.6")

SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FEATURE_NAMES = [
    "diff_self",
    "diff_pool",
    "diff_min",
    "endpoint_span",
    "j_score",
    "max_self",
    "min_self",
    "rank_u",
    "rank_v",
    "min_rank",
    "max_rank",
    "rank_sum",
    "both_in_top2",
    "both_in_top3",
    "both_in_top5",
    "router_prob",
]


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


def generate_dev_dataset(records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build development examples strictly from corpus materials with matched contrasts."""
    pairs = []
    for r in records.values():
        u = r["id"]
        role = r[WORKFLOW]["sequence_role"]
        v = r[WORKFLOW]["id"]
        if role.startswith("Follow with"):
            pairs.append((u, v))
    pairs.sort()

    examples = []
    for p_idx, (u, v) in enumerate(pairs):
        group_id = f"PAIR_{p_idx:02d}"
        u_rec, v_rec = records[u], records[v]
        u_scens = [s["scenario"] for s in u_rec["scenarios_json"]]
        v_scens = [s["scenario"] for s in v_rec["scenarios_json"]]
        comp_u = u_rec[CONTRAST]["id"]
        cond_u = u_rec[CONTRAST]["condition"]
        comp_v = v_rec[CONTRAST]["id"]
        cond_v = v_rec[CONTRAST]["condition"]

        s_u0, s_u1 = u_scens[0], u_scens[1] if len(u_scens) > 1 else u_scens[0]
        s_v0, s_v1 = v_scens[0], v_scens[1] if len(v_scens) > 1 else v_scens[0]

        # 1. Standard Positives
        pos_std_1 = (f"{s_u0} Once this immediate issue is resolved, "
                     f"{s_v0[:1].lower() + s_v0[1:]} "
                     f"What sequence of actions should be implemented?")
        pos_std_2 = (f"{s_u1} Handling this initial situation provides the baseline. Thereafter, "
                     f"{s_v1[:1].lower() + s_v1[1:]} "
                     f"How should the team sequence these two steps?")

        # 2. Hard Positives (no explicit sequencing words)
        pos_hard_1 = (f"A complex institutional challenge involves two interconnected phases: "
                      f"{s_u0} Resolving the underlying friction establishes shared ground so that "
                      f"{s_v0[:1].lower() + s_v0[1:]} "
                      f"Which integrated multi-phase strategy governs this dilemma?")
        pos_hard_2 = (f"Leadership must coordinate across dependent responsibilities: "
                      f"{s_u1} Completing this preliminary stabilization enables the leadership to address "
                      f"{s_v1[:1].lower() + s_v1[1:]} "
                      f"What coordinated strategy applies?")

        # 3. Standard Negatives
        neg_std_u = f"{s_u0} What core principle should govern this decision?"
        neg_std_v = f"{s_v0} What core principle should guide this action?"
        neg_std_arb = f"{s_u1} Although {cond_u[:1].lower() + cond_u[1:]}, what single choice is warranted?"

        # 4. Hard Negatives (with misleading sequence cues)
        neg_hard_1 = (f"Before the quarterly deadline next month, an organization must resolve one matter first: "
                      f"{s_u0} Although earlier discussions were held then, what primary decision should be taken now?")
        neg_hard_2 = (f"Two proposals were submitted first, and a formal review is planned next week. "
                      f"In the current situation, {s_v1[:1].lower() + s_v1[1:]} "
                      f"What single authoritative decision must be made?")
        neg_hard_3 = (f"The committee met first to review background facts, then identified an impasse: "
                      f"{s_u1} What single ruling resolves this conflict?")

        # Candidate pool: {u, v, comp_u, comp_v, and 1-hop neighbors}
        pool = set([u, v, comp_u, records[v][CONTRAST]["id"]])
        for node in list(pool):
            pool.add(records[node][WORKFLOW]["id"])
            pool.add(records[node][CONTRAST]["id"])
        pool_list = sorted(pool)

        for text in [pos_std_1, pos_std_2]:
            examples.append({"text": text, "is_synthesis": True, "gold_pair": (u, v), "group_id": group_id, "pool": pool_list})
        for text in [pos_hard_1, pos_hard_2]:
            examples.append({"text": text, "is_synthesis": True, "gold_pair": (u, v), "group_id": group_id, "pool": pool_list})

        for text in [neg_std_u, neg_std_v, neg_std_arb]:
            examples.append({"text": text, "is_synthesis": False, "gold_pair": None, "group_id": group_id, "pool": pool_list})
        for text in [neg_hard_1, neg_hard_2, neg_hard_3]:
            examples.append({"text": text, "is_synthesis": False, "gold_pair": None, "group_id": group_id, "pool": pool_list})

    return examples


def batch_score_ce(model, tokenizer, pairs, device, batch_size=64, max_length=384):
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
    print("================================================================================")
    print("Training & Calibrating Pair-Promotion Verifier")
    print("================================================================================\n")

    records = load_corpus(DATA_DIR)
    dev_examples = generate_dev_dataset(records)
    print(f"Constructed {len(dev_examples)} development queries across 54 pair groups.")

    # 1. Compute Out-of-Fold Query Intent Probabilities
    print("\n1. Computing Out-of-Fold query intent probabilities via 5-Fold GroupKFold...")
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    texts = [e["text"] for e in dev_examples]
    query_embs = bge_model.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True, show_progress_bar=False)

    groups = np.array([e["group_id"] for e in dev_examples])
    labels = np.array([int(e["is_synthesis"]) for e in dev_examples])

    gkf = GroupKFold(n_splits=5)
    oof_router_probs = np.zeros(len(dev_examples), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(query_embs, labels, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups.intersection(val_groups)) == 0, f"Leakage in fold {fold_idx}"

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(query_embs[train_idx], labels[train_idx])
        oof_router_probs[val_idx] = clf.predict_proba(query_embs[val_idx])[:, 1]

    print("   Out-of-fold query intent probabilities computed. Zero group overlap verified.")

    # 2. Score Single Passages and Pairs with Frozen Cross-Encoder
    print("\n2. Scoring candidate passages and pairs with BAAI/bge-reranker-base...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    ce_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
    ce_model.eval()

    passage_cards = {k: passage_card(records[k]) for k in records}

    # Flatten all pairs across all queries for efficient vectorized scoring
    pair_scorer_cfg = json.loads((ARTIFACTS_DIR / "pair_scorer_config.json").read_text(encoding="utf8"))
    tau_self = pair_scorer_cfg["tau_self"]
    tau_pool = pair_scorer_cfg["tau_pool"]

    proposals = []

    # Process query by query
    print("   Evaluating frozen v1.6 pair proposals on development queries...")
    for idx, ex in enumerate(dev_examples):
        q_text = ex["text"]
        pool = ex["pool"]
        is_synth = ex["is_synthesis"]
        gold_pair = ex["gold_pair"]
        group_id = ex["group_id"]
        router_p = float(oof_router_probs[idx])

        # Enumerate ordered pairs in pool
        pairs_in_pool = []
        for u in pool:
            wf = records[u][WORKFLOW]
            v = wf["id"]
            if v in pool and wf["sequence_role"].startswith("Follow with"):
                pairs_in_pool.append((u, v))

        # Score single cards
        single_inputs = [[q_text, passage_cards[k]] for k in pool]
        single_scores = batch_score_ce(ce_model, tokenizer, single_inputs, device, batch_size=64)
        s_dict = dict(zip(pool, single_scores))
        max_pool_s = max(s_dict.values())

        # Score joint pair cards
        j_dict = {}
        if pairs_in_pool:
            joint_inputs = [[q_text, pair_joint_card(records[u], records[v])] for u, v in pairs_in_pool]
            joint_scores = batch_score_ce(ce_model, tokenizer, joint_inputs, device, batch_size=64)
            j_dict = dict(zip(pairs_in_pool, joint_scores))

        # Evaluate frozen v1.6 proposal rule
        qualifying_pairs = []
        for (u, v), j_val in j_dict.items():
            max_self = max(s_dict[u], s_dict[v])
            cond_self = (j_val - max_self >= tau_self)
            cond_pool = (j_val - max_pool_s >= tau_pool)
            if cond_self and cond_pool:
                qualifying_pairs.append({
                    "pair": (u, v),
                    "j_score": j_val,
                    "max_self": max_self,
                    "min_self": min(s_dict[u], s_dict[v]),
                    "diff_self": j_val - max_self,
                    "diff_pool": j_val - max_pool_s,
                    "diff_min": j_val - min(s_dict[u], s_dict[v]),
                    "endpoint_span": max_self - min(s_dict[u], s_dict[v]),
                })

        if not qualifying_pairs:
            continue  # No proposal made

        qualifying_pairs.sort(key=lambda x: -x["j_score"])
        best = qualifying_pairs[0]
        u_prop, v_prop = best["pair"]

        # Endpoint ranks in fallback ranking (descending single score)
        fallback_rank = sorted(pool, key=lambda k: (-s_dict[k], k))
        ru = fallback_rank.index(u_prop) + 1 if u_prop in fallback_rank else 99
        rv = fallback_rank.index(v_prop) + 1 if v_prop in fallback_rank else 99
        min_r = min(ru, rv)
        max_r = max(ru, rv)
        rank_sum = ru + rv

        both_in_top2 = float(ru <= 2 and rv <= 2)
        both_in_top3 = float(ru <= 3 and rv <= 3)
        both_in_top5 = float(ru <= 5 and rv <= 5)

        # Ground truth target: y=1 only if query is synthesis AND proposal matches gold
        is_correct_pair = bool(is_synth and gold_pair is not None and (u_prop, v_prop) == gold_pair)
        y = 1 if is_correct_pair else 0

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
            router_p,
        ]

        proposals.append({
            "group_id": group_id,
            "query_idx": idx,
            "is_synthesis": is_synth,
            "gold_pair": gold_pair,
            "proposed_pair": (u_prop, v_prop),
            "y": y,
            "features": feat_vector,
        })

    print(f"   Generated {len(proposals)} proposals from {len(dev_examples)} dev queries.")
    pos_props = sum(1 for p in proposals if p["y"] == 1)
    neg_props = sum(1 for p in proposals if p["y"] == 0)
    print(f"   Correct proposals (y=1): {pos_props}")
    print(f"   Spurious / wrong proposals (y=0): {neg_props}")

    # 3. Train and Calibrate Verifier Classifier
    print("\n3. Training & Calibrating Verifier Classifier via 5-Fold GroupKFold...")
    X = np.array([p["features"] for p in proposals])
    y = np.array([p["y"] for p in proposals])
    p_groups = np.array([p["group_id"] for p in proposals])

    verifier_gkf = GroupKFold(n_splits=5)
    oof_verifier_probs = np.zeros(len(proposals), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(verifier_gkf.split(X, y, p_groups)):
        train_g = set(p_groups[train_idx])
        val_g = set(p_groups[val_idx])
        assert len(train_g.intersection(val_g)) == 0, f"Verifier leakage in fold {fold_idx}"

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        oof_verifier_probs[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]

    # Grid search threshold to maximize balanced metric while constraining false accepts on y=0
    threshold_grid = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_score = -999.0
    best_precision = 0.0
    best_recall = 0.0
    best_fp = 0
    best_tp = 0

    total_pos_p = np.sum(y == 1)
    total_neg_p = np.sum(y == 0)

    for th in threshold_grid:
        preds = (oof_verifier_probs >= th).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = total_pos_p - tp

        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / total_pos_p if total_pos_p else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        # Optimization objective: Reward high precision and recall on true proposals,
        # while heavily penalizing spurious pair acceptances (to protect arbitration flips)
        # Goal: false accept rate on spurious proposals <= 10%
        spurious_accept_rate = fp / total_neg_p if total_neg_p else 0.0
        opt_score = f1 - 1.5 * spurious_accept_rate

        if opt_score > best_score:
            best_score = opt_score
            best_thresh = th
            best_precision = prec
            best_recall = rec
            best_tp = tp
            best_fp = fp

    print(f"\n   Calibrated Verifier Threshold: tau_verifier* = {best_thresh:.4f}")
    print(f"   CV True Proposal Recall: {best_recall*100:.2f}% ({best_tp}/{total_pos_p})")
    print(f"   CV Precision on Proposals: {best_precision*100:.2f}%")
    print(f"   CV Spurious Proposal Rejection Rate: {(total_neg_p - best_fp)/total_neg_p*100:.2f}% "
          f"({total_neg_p - best_fp}/{total_neg_p} rejected, only {best_fp} false accepts)")

    # 4. Fit Final Deployable Verifier Pipeline
    print("\n4. Fitting final deployable verifier pipeline on full proposal dataset...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    final_pipeline.fit(X, y)

    # Inspect feature importances
    coefs = final_pipeline.named_steps["clf"].coef_[0]
    print("\n   Verifier Feature Weights (Standardized Logistic Regression):")
    for fname, w in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -abs(x[1])):
        print(f"     {fname:<16s}: {w:+.4f}")

    # 5. Save Artifacts and Configuration Hash
    model_path = ARTIFACTS_DIR / "pair_verifier_model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"\nSaved verifier model to {model_path}")

    verifier_config = {
        "model_type": "StandardScaler + LogisticRegression(C=1.0)",
        "features": FEATURE_NAMES,
        "calibrated_threshold": float(best_thresh),
        "cv_metrics": {
            "total_proposals": len(proposals),
            "total_true_proposals": int(total_pos_p),
            "total_spurious_proposals": int(total_neg_p),
            "cv_tp": int(best_tp),
            "cv_fp": int(best_fp),
            "cv_recall": float(best_recall),
            "cv_precision": float(best_precision),
            "cv_spurious_rejection_rate": float((total_neg_p - best_fp) / total_neg_p),
        },
        "upstream_router": "artifacts-v1.6/workflow_router.joblib",
        "pair_scorer": "BAAI/bge-reranker-base (tau_self=-0.5, tau_pool=-1.0)",
        "fallback": "v1.3 exact bit-for-bit",
    }
    cfg_json = json.dumps(verifier_config, sort_keys=True, indent=2)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf8")).hexdigest()
    verifier_config["config_sha256"] = cfg_hash

    cfg_path = ARTIFACTS_DIR / "verifier_config.json"
    cfg_path.write_text(json.dumps(verifier_config, indent=2), encoding="utf8")
    print(f"Saved verifier configuration to {cfg_path}")
    print(f"Configuration SHA256: {cfg_hash}")


if __name__ == "__main__":
    main()
