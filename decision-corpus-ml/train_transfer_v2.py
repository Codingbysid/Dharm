"""Train the cross-domain transfer projection and the local suitability heads.

Q01–Q42 and eval_transfer_v1.jsonl are not used as training text.
The neural encoders stay frozen. Only a residual projection and sklearn heads are fit.
"""
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rc1_inference import (  # noqa: E402
    CONTRAST,
    QUERY_PREFIX,
    WORKFLOW,
    TransferProjection,
    load_corpus,
    transfer_card,
)
from train_workflow_router import generate_dev_dataset

DATA = ROOT / "data"
ART = ROOT / "artifacts-v1.6"
ART11 = ROOT / "artifacts-v1.1"
BI = "BAAI/bge-small-en-v1.5"


def _encode(model, texts):
    return model.encode(
        [QUERY_PREFIX + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _out_of_scope():
    rows = []
    for i in range(1, 13):
        rows.append(f"What is {17 * i} multiplied by {23 + i}?")
        rows.append(f"In which calendar year was ledger {800 + i} first bound?")
        rows.append(f"Which workshop cast bronze artifact {1400 + i}?")
    rows.extend([
        "How do I replace a worn kitchen washer this afternoon?",
        "What color should I dye my hair before the wedding?",
        "Give me a recipe for plain rice.",
        "Write a python script that prints today's date.",
        "What is the exact weight of an unlabeled stone on a desk?",
        "Which quarry supplied this unlabeled ceramic tile?",
        "What second did an unrecorded bell stop vibrating yesterday?",
        "My thermometer shows two numbers. Which is the true temperature?",
        "How can I stop a dripping tap before guests arrive?",
        "Which screw size fits this loose cabinet hinge?",
        "The faucet will not shut. What tool do I use?",
        "How many grams of salt are in this unmarked bowl?",
        "No one measured it. What was the precise duration of a sound in an empty hall yesterday?",
        "Which unlabeled jar is heavier if I cannot weigh either jar?",
    ])
    return rows


def train_projection(card_emb, scen_emb, owners, competing_idx):
    model = TransferProjection()
    with torch.no_grad():
        model.up.weight.zero_()
        model.up.bias.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cards = torch.tensor(card_emb, dtype=torch.float32)
    scens = torch.tensor(scen_emb, dtype=torch.float32)
    owners_t = torch.tensor(owners, dtype=torch.long)
    comp_t = torch.tensor(competing_idx, dtype=torch.long)
    idx = np.arange(len(owners))
    for epoch in range(8):
        random.Random(10 + epoch).shuffle(idx.tolist())
        order = idx.copy()
        random.Random(10 + epoch).shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), 32):
            batch = torch.tensor(order[start:start + 32], dtype=torch.long)
            q = model(scens[batch])
            docs = model(cards)
            logits = (q @ docs.T) / 0.07
            loss = F.cross_entropy(logits, owners_t[batch])
            pos = (q * docs[owners_t[batch]]).sum(-1)
            neg = (q * docs[comp_t[batch]]).sum(-1)
            loss = loss + 0.3 * F.relu(0.15 - (pos - neg)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"projection epoch {epoch + 1} loss {total / max(steps, 1):.4f}")
    model.eval()
    return model


def apply_proj(model, embs):
    with torch.inference_mode():
        tensor = torch.tensor(np.asarray(embs), dtype=torch.float32)
        return model(tensor).cpu().numpy()


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    records = load_corpus(DATA)
    keys = sorted(records.keys())
    key_index = {k: i for i, k in enumerate(keys)}
    bi = SentenceTransformer(BI, device="cpu")
    cards = [transfer_card(records[k]) for k in keys]
    card_emb = bi.encode(cards, normalize_embeddings=True, show_progress_bar=False)
    scenarios = []
    owners = []
    for k in keys:
        for item in records[k]["scenarios_json"]:
            scenarios.append(item["scenario"])
            owners.append(key_index[k])
    scen_emb = _encode(bi, scenarios)
    competing_idx = [key_index[records[keys[o]][CONTRAST]["id"]] for o in owners]
    print(f"training projection on {len(scenarios)} scenarios / {len(keys)} passages")
    proj = train_projection(card_emb, scen_emb, owners, competing_idx)
    ART.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": proj.state_dict()}, ART / "transfer_proj.pt")

    card_p = apply_proj(proj, card_emb)
    scen_p = apply_proj(proj, scen_emb)
    in_scope = scenarios
    out_scope = _out_of_scope()
    in_emb = scen_p
    out_emb = apply_proj(proj, _encode(bi, out_scope))
    x = np.vstack([in_emb, out_emb])
    y = np.array([0] * len(in_emb) + [1] * len(out_emb))
    abst = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    abst.fit(x, y)
    # Class 1 is out-of-scope. Threshold keeps corpus scenarios in scope.
    in_prob = abst.predict_proba(in_emb)[:, 1]
    out_prob = abst.predict_proba(out_emb)[:, 1]
    # Prefer 0.50 when training still separates how-to and measurement questions there.
    # A lower threshold abstains on real dilemmas whose wording is merely concrete.
    threshold = 0.35
    for candidate in (0.50, 0.45, 0.40, 0.35):
        if (in_prob > candidate).mean() <= 0.03 and (out_prob > candidate).mean() >= 0.95:
            threshold = float(candidate)
            break
    print(f"abstention threshold {threshold:.2f} in-scope FPR {(in_prob > threshold).mean():.3f} "
          f"out recall {(out_prob > threshold).mean():.3f}")
    joblib.dump(abst, ART / "abstention_clf.joblib")
    joblib.dump(abst, ART11 / "abstention_clf.joblib")
    (ART / "abstention_config.json").write_text(json.dumps({"threshold": threshold}), encoding="utf8")

    # Suitability uses the same projected embedding plus dense margins, without a CE pass.
    # The inference feature names stay stable; CE margins are filled with the dense margins
    # at training time and recomputed from the cross-encoder at inference only as extras
    # when both sides share a scale. Here both sides use dense margins so the head matches.
    def margins(emb):
        sims = emb @ card_p.T
        order = np.argsort(-sims, axis=1)
        top1 = sims[np.arange(len(emb)), order[:, 0]]
        top2 = sims[np.arange(len(emb)), order[:, 1]]
        return top1, top1 - top2, top1

    def pack(emb, abst_prob):
        top1, gap, scen = margins(emb)
        return np.column_stack([abst_prob, top1, gap, top1, scen])

    in_pack = pack(in_emb, abst.predict_proba(in_emb)[:, 1])
    out_pack = pack(out_emb, abst.predict_proba(out_emb)[:, 1])
    suit_x = np.vstack([in_pack, out_pack])
    suit_y = np.array([1] * len(in_pack) + [0] * len(out_pack))
    suit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500)),
    ])
    suit.fit(suit_x, suit_y)
    in_s = suit.predict_proba(in_pack)[:, 1]
    out_s = suit.predict_proba(out_pack)[:, 1]
    suit_threshold = 0.5
    for candidate in np.linspace(0.25, 0.7, 19):
        if (in_s < candidate).mean() <= 0.02 and (out_s < candidate).mean() >= 0.85:
            suit_threshold = float(candidate)
            break
    print(f"suitability threshold {suit_threshold:.2f} "
          f"in-scope drop {(in_s < suit_threshold).mean():.3f} "
          f"out reject {(out_s < suit_threshold).mean():.3f}")
    joblib.dump(suit, ART / "suitability_clf.joblib")
    (ART / "suitability_config.json").write_text(
        json.dumps({
            "threshold": suit_threshold,
            "features": ["abst_prob", "top1_ce", "top1_minus_top2_ce", "dense_transfer_sim", "max_scen_sim"],
            "note": "Trained on dense margins with the same five-feature layout used at inference.",
        }),
        encoding="utf8",
    )

    dev = generate_dev_dataset(records)
    dev_emb = apply_proj(proj, _encode(bi, [row["text"] for row in dev]))
    router = LogisticRegression(C=1.0, max_iter=500)
    router.fit(dev_emb, np.array([row["label"] for row in dev]))
    joblib.dump(router, ART / "workflow_router.joblib")
    print(f"router fit on {len(dev)} corpus-only examples, positive rate {np.mean([r['label'] for r in dev]):.2f}")

    # Pair verifier: features from projected dense ranks, not a fresh cross-encoder sweep.
    # The 16-feature layout stays what RC1 already extracts so the saved pipeline still loads.
    examples = []
    labels = []
    groups = []
    for row_i, row in enumerate(dev):
        emb = dev_emb[row_i]
        sims = emb @ card_p.T
        order = list(np.argsort(-sims))
        rank = {keys[int(j)]: r for r, j in enumerate(order, 1)}
        pairs = []
        for u in keys:
            role = records[u][WORKFLOW]["sequence_role"]
            v = records[u][WORKFLOW]["id"]
            if role.startswith("Follow with"):
                pairs.append((u, v))
        if not pairs:
            continue
        scored = []
        for u, v in pairs:
            iu, iv = key_index[u], key_index[v]
            joint = 0.5 * (float(sims[iu]) + float(sims[iv]))
            scored.append((joint, u, v, float(sims[iu]), float(sims[iv])))
        scored.sort(reverse=True)
        joint, u, v, su, sv = scored[0]
        max_self = max(su, sv)
        min_self = min(su, sv)
        ru, rv = rank[u], rank[v]
        feat = [
            joint - max_self,
            joint - float(sims.max()),
            joint - min_self,
            max_self - min_self,
            joint,
            max_self,
            min_self,
            float(ru), float(rv), float(min(ru, rv)), float(max(ru, rv)), float(ru + rv),
            float(ru <= 2 and rv <= 2),
            float(ru <= 3 and rv <= 3),
            float(ru <= 5 and rv <= 5),
            float(router.predict_proba([emb])[0, 1]),
        ]
        gold_positive = row["label"] == 1 and row.get("type", "").startswith("pos")
        examples.append(feat)
        labels.append(1 if gold_positive else 0)
        groups.append(row["group_id"])
    verifier = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500)),
    ])
    verifier.fit(np.array(examples), np.array(labels))
    # Keep the previous acceptance bar unless this fit separates cleanly lower.
    threshold_v = 0.57
    probs = verifier.predict_proba(np.array(examples))[:, 1]
    labels_a = np.array(labels)
    for candidate in np.linspace(0.4, 0.75, 15):
        pred = probs >= candidate
        if labels_a.sum() == 0:
            break
        precision = (pred & (labels_a == 1)).sum() / max(pred.sum(), 1)
        recall = (pred & (labels_a == 1)).sum() / max((labels_a == 1).sum(), 1)
        if precision >= 0.55 and recall >= 0.35:
            threshold_v = float(candidate)
            break
    print(f"verifier examples {len(examples)} threshold {threshold_v:.2f}")
    joblib.dump(verifier, ART / "pair_verifier_model.joblib")
    cfg_path = ART / "verifier_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf8"))
    cfg["calibrated_threshold"] = threshold_v
    cfg["transfer_v2"] = True
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf8")
    print("saved transfer_proj, abstention, suitability, router, verifier")


if __name__ == "__main__":
    main()
