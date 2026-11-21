"""Train and calibrate the Deployable Query-Text Workflow Intent Router.

Strict Isolation:
Builds development examples solely from corpus materials (decisions_v0.2.jsonl
and chanakya_daily_supplement.jsonl).
All 42 benchmark queries, query texts, IDs, query types, and gold labels
are strictly kept outside fitting, calibration, and threshold selection.

Design:
- Query embeddings: Frozen BAAI/bge-small-en-v1.5 dense encoder.
- Classifier: Regularized Logistic Regression.
- Positive examples: Two-stage sequential decisions where result of first informs second.
- Negative examples: Single-step decisions (including multi-event context).
- Matched contrasts on the same topic:
  * Hard negatives: single choice with 'first', 'then', 'before', 'next' in background.
  * Hard positives: genuine two-stage decision with NO explicit sequencing words.
- Grouping: 54 complementary pair groups (GroupKFold by pair_group).
  Zero overlap across folds verified.
- Threshold selection:
  Predeclared constraint: max 5% false workflow-routing rate (FPR <= 0.05) on single-step cases.
  Maximize workflow recall within that budget.
  Always-fallback operating point included.
"""
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_DIR = Path("decision-corpus-ml/artifacts-v1.6")

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

        # 1. Standard Positives: two actions where result of first informs second
        pos_std_1 = (f"{s_u0} Once this immediate issue is resolved, "
                     f"{s_v0[:1].lower() + s_v0[1:]} "
                     f"What sequence of actions should be implemented?")
        pos_std_2 = (f"{s_u1} Handling this initial situation provides the baseline. Thereafter, "
                     f"{s_v1[:1].lower() + s_v1[1:]} "
                     f"How should the team sequence these two steps?")

        # 2. Hard Positives: genuine two-stage decision with NO explicit sequencing words (no 'first', 'then', 'sequence', 'next')
        pos_hard_1 = (f"A complex institutional challenge involves two interconnected phases: "
                      f"{s_u0} Resolving the underlying friction establishes shared ground so that "
                      f"{s_v0[:1].lower() + s_v0[1:]} "
                      f"Which integrated multi-phase strategy governs this dilemma?")
        pos_hard_2 = (f"Leadership must coordinate across dependent responsibilities: "
                      f"{s_u1} Completing this preliminary stabilization enables the leadership to address "
                      f"{s_v1[:1].lower() + s_v1[1:]} "
                      f"What coordinated strategy applies?")

        # 3. Standard Negatives: single decision
        neg_std_u = f"{s_u0} What core principle should govern this decision?"
        neg_std_v = f"{s_v0} What core principle should guide this action?"
        neg_std_arb = f"{s_u1} Although {cond_u[:1].lower() + cond_u[1:]}, what single choice is warranted?"

        # 4. Hard Negatives: single choice with 'first', 'then', 'before', 'next' in background context, but only ONE decision asked
        neg_hard_1 = (f"Before the quarterly deadline next month, an organization must resolve one matter first: "
                      f"{s_u0} Although earlier discussions were held then, what primary decision should be taken now?")
        neg_hard_2 = (f"Two proposals were submitted first, and a formal review is planned next week. "
                      f"In the current situation, {s_v1[:1].lower() + s_v1[1:]} "
                      f"What single authoritative decision must be made?")
        neg_hard_3 = (f"The committee met first to review background facts, then identified an impasse: "
                      f"{s_u1} What single ruling resolves this conflict?")

        for text in [pos_std_1, pos_std_2]:
            examples.append({"text": text, "label": 1, "group_id": group_id, "type": "pos_std"})
        for text in [pos_hard_1, pos_hard_2]:
            examples.append({"text": text, "label": 1, "group_id": group_id, "type": "pos_hard_no_cue"})

        for text in [neg_std_u, neg_std_v, neg_std_arb]:
            examples.append({"text": text, "label": 0, "group_id": group_id, "type": "neg_std"})
        for text in [neg_hard_1, neg_hard_2, neg_hard_3]:
            examples.append({"text": text, "label": 0, "group_id": group_id, "type": "neg_hard_with_cue"})

    return examples


def main():
    print("================================================================================")
    print("Training & Calibrating Deployable Workflow Intent Router")
    print("================================================================================\n")

    records = load_corpus(DATA_DIR)
    dev_examples = generate_dev_dataset(records)
    print(f"Total development examples: {len(dev_examples)}")
    pos_count = sum(1 for e in dev_examples if e["label"] == 1)
    neg_count = sum(1 for e in dev_examples if e["label"] == 0)
    print(f"  Positives: {pos_count} (std: {sum(1 for e in dev_examples if e['type'] == 'pos_std')}, "
          f"hard-no-cue: {sum(1 for e in dev_examples if e['type'] == 'pos_hard_no_cue')})")
    print(f"  Negatives: {neg_count} (std: {sum(1 for e in dev_examples if e['type'] == 'neg_std')}, "
          f"hard-with-cue: {sum(1 for e in dev_examples if e['type'] == 'neg_hard_with_cue')})")

    groups = np.array([e["group_id"] for e in dev_examples])
    labels = np.array([e["label"] for e in dev_examples])
    texts = [e["text"] for e in dev_examples]

    print("\nEncoding query texts with frozen BAAI/bge-small-en-v1.5...")
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    embeddings = bge_model.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
    print(f"Embeddings shape: {embeddings.shape}")

    # 5-Fold GroupKFold Cross-Validation
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(dev_examples), dtype=float)
    fold_splits_info = []

    print("\nRunning 5-Fold GroupKFold Cross-Validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(embeddings, labels, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Leakage detected in fold {fold_idx}: {overlap}"

        fold_splits_info.append({
            "fold": fold_idx,
            "train_groups_count": len(train_groups),
            "val_groups_count": len(val_groups),
            "val_groups": sorted(list(val_groups)),
            "overlap_count": 0,
        })

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(embeddings[train_idx], labels[train_idx])
        val_probs = clf.predict_proba(embeddings[val_idx])[:, 1]
        oof_probs[val_idx] = val_probs

    print("5-Fold Cross-Validation completed. Verified zero group overlap across all folds.")

    # Threshold Search under Predeclared Budget: FPR <= 0.05
    # Include operating points across grid
    threshold_grid = np.linspace(0.0, 1.0, 101)
    operating_points = []

    best_thresh = 1.0
    best_recall = -1.0
    best_fpr = 0.0
    best_tp = 0
    best_fp = 0

    neg_mask = (labels == 0)
    pos_mask = (labels == 1)
    total_neg = np.sum(neg_mask)
    total_pos = np.sum(pos_mask)

    for th in threshold_grid:
        preds = (oof_probs >= th).astype(int)
        tp = int(np.sum((preds == 1) & pos_mask))
        fp = int(np.sum((preds == 1) & neg_mask))
        fn = total_pos - tp
        tn = total_neg - fp

        fpr = fp / total_neg if total_neg else 0.0
        recall = tp / total_pos if total_pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0

        operating_points.append({
            "threshold": round(float(th), 4),
            "fp": fp,
            "total_neg": int(total_neg),
            "fpr": round(float(fpr), 4),
            "tp": tp,
            "total_pos": int(total_pos),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
        })

        if fpr <= 0.05:
            if recall > best_recall:
                best_recall = recall
                best_thresh = th
                best_fpr = fpr
                best_tp = tp
                best_fp = fp

    print(f"\nOperating Points Summary (FPR budget <= 0.05):")
    print(f"  Selected Threshold: theta* = {best_thresh:.4f}")
    print(f"  False Positive Rate (single-step false routing): {best_fpr*100:.2f}% ({best_fp}/{total_neg})")
    print(f"  Workflow Intent Recall: {best_recall*100:.2f}% ({best_tp}/{total_pos})")
    always_fallback_op = next(op for op in operating_points if op["threshold"] == 1.0)
    print(f"  Always-fallback Operating Point (theta=1.0): FPR = {always_fallback_op['fpr']*100:.2f}% "
          f"({always_fallback_op['fp']}/{total_neg}), Recall = {always_fallback_op['recall']*100:.2f}% "
          f"({always_fallback_op['tp']}/{total_pos})")

    # Fit final deployable classifier on all development data
    print("\nFitting final deployable LogisticRegression on full development set...")
    final_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_clf.fit(embeddings, labels)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "workflow_router.joblib"
    joblib.dump(final_clf, model_path)
    print(f"Saved router model to {model_path}")

    # Build frozen configuration hash
    router_config = {
        "router_model_type": "LogisticRegression(C=1.0)",
        "query_encoder": "BAAI/bge-small-en-v1.5",
        "query_prefix": QUERY_PREFIX,
        "predeclared_max_fpr": 0.05,
        "calibrated_threshold": float(best_thresh),
        "calibration_metrics": {
            "total_positives": int(total_pos),
            "total_negatives": int(total_neg),
            "selected_tp": int(best_tp),
            "selected_fp": int(best_fp),
            "selected_fpr": float(best_fpr),
            "selected_recall": float(best_recall),
        },
        "operating_points_sample": [
            op for op in operating_points if op["threshold"] in [0.0, 0.2, 0.4, 0.5, round(best_thresh, 4), 0.7, 0.8, 0.9, 1.0]
        ],
        "split_verification": {
            "folds_count": 5,
            "all_overlaps_zero": True,
            "folds": fold_splits_info,
        },
        "materials": "decisions_v0.2.jsonl, chanakya_daily_supplement.jsonl (benchmark excluded)",
    }
    cfg_json = json.dumps(router_config, sort_keys=True, indent=2)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf8")).hexdigest()
    router_config["config_sha256"] = cfg_hash

    config_path = ARTIFACTS_DIR / "router_config.json"
    config_path.write_text(json.dumps(router_config, indent=2), encoding="utf8")
    print(f"Saved router configuration to {config_path}")
    print(f"Configuration SHA256: {cfg_hash}")


if __name__ == "__main__":
    main()
