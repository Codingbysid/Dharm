"""Dharm Decision Corpus Release Candidate 1 (RC1) Inference Engine & CLI.

Local inference entry point that accepts raw dilemma text and executes the
full frozen pipeline end-to-end without requiring benchmark files or per-query
score caches.

Pipeline Stages:
1. Epistemic Abstention Gate (BAAI/bge-small-en-v1.5 + abstention_clf.joblib)
   Threshold: 0.6500. If abstained -> primary = "Abstain", no pair promotion.
2. Multi-Vector Stage-1 Hybrid RRF Retrieval (Dense Max-Sim + Lexical TF-IDF)
   Top-7 seeds expanded via 1-hop dual-graph (competing + complementary).
3. Frozen v1.3 Fallback Candidate Ranking:
   - RRF norm + CE norm (0.55/0.45)
   - Gated NLI directional arbitration (cross-encoder/nli-deberta-v3-small)
   - v1.3 workflow ordering bonus
4. Frozen v1.6 Complementary Pair Proposer (BAAI/bge-reranker-base):
   - Proposes top pair satisfying tau_self = -0.5, tau_pool = -1.0.
5. Frozen Pair-Promotion Verifier (StandardScaler + LogisticRegression):
   - Extracts 16 features from proposed pair, pool CE ranks, and query intent.
   - Accepts promotion if P(accept) >= 0.5700.
   - Falls back bit-for-bit to v1.3 ranking if rejected or no proposal made.

All neural models and indices are loaded ONCE per process.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR_DEFAULT = Path(__file__).resolve().parent / "data"
ARTIFACTS_V1_1_DEFAULT = Path(__file__).resolve().parent / "artifacts-v1.1"
ARTIFACTS_V1_6_DEFAULT = Path(__file__).resolve().parent / "artifacts-v1.6"

BI_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "BAAI/bge-reranker-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def read_rows(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), f"Duplicate {key} in {path}")
    return dict(sorted(result.items()))


def load_corpus(folder: Path) -> Dict[str, Dict[str, Any]]:
    records = read_rows(folder / "decisions_v0.2.jsonl", "id")
    require(set(records) == {f"D{i:03}" for i in range(1, 109)}, "Expected D001–D108")
    for k, r in records.items():
        require(len(r) == 24 and len(r["scenarios_json"]) == 2, f"Unexpected schema for {k}")
        for edge in (CONTRAST, WORKFLOW):
            require(r[edge]["id"] in records and r[edge]["id"] != k, f"Broken {edge} at {k}")
        peer = records[r[WORKFLOW]["id"]][WORKFLOW]
        role = r[WORKFLOW]["sequence_role"]
        require(peer["id"] == k and
                ((role.startswith("Follow with") and peer["sequence_role"].startswith("Precede with"))
                 or (role.startswith("Precede with") and peer["sequence_role"].startswith("Follow with"))),
                f"Inconsistent workflow at {k}")
    notes = read_rows(folder / "chanakya_daily_supplement.jsonl", "note_id")
    for note, row in notes.items():
        target = row.get("existing_record_id") or SUPPLEMENT.get(note)
        if target and target in records:
            records[target]["scenarios_json"].append({"scenario": row["modern_application"]})
    return records


def passage_card(r: Dict[str, Any]) -> str:
    return (f"[{r['id']}] {r['work']} {r['locator']} | Domain: {r['decision_domain']} | "
            f"Urgency: {r['urgency_level']} | Reversibility: {r['reversibility']}\n"
            f"Principle: {r['derived_principle']}\nContext: {r['context_summary']}\n"
            f"Text: {r['english_translation']}\nCues: {r['retrieval_cues']}")


def tfidf_doc(r: Dict[str, Any]) -> str:
    scenarios_text = " ".join(s["scenario"] for s in r["scenarios_json"])
    comp_cond = r[CONTRAST].get("condition", "")
    seq_role = r[WORKFLOW].get("sequence_role", "")
    return f"{passage_card(r)}\n{scenarios_text}\n{comp_cond}\n{seq_role}"


def pair_joint_card(u_rec: Dict[str, Any], v_rec: Dict[str, Any]) -> str:
    uid, uprin = u_rec["id"], u_rec["derived_principle"]
    vid, vprin = v_rec["id"], v_rec["derived_principle"]
    role = u_rec[WORKFLOW]["sequence_role"]
    return (f"First Principle [{uid}]: {uprin}\n"
            f"Transition: {role}\n"
            f"Second Principle [{vid}]: {vprin}")


def expand_pool(seeds: List[str], records: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted({j for k in seeds for j in (k, records[k][CONTRAST]["id"], records[k][WORKFLOW]["id"])})


def min_max_normalize(scores_dict: Dict[str, float], subset_keys: List[str]) -> Dict[str, float]:
    vals = [scores_dict[k] for k in subset_keys]
    mi, ma = min(vals), max(vals)
    if ma > mi:
        return {k: (scores_dict[k] - mi) / (ma - mi) for k in subset_keys}
    return {k: 1.0 for k in subset_keys}


class MultiVectorHybridRRFIndex:
    """Stage 1: Multi-Vector Max-Sim Dense Retrieval + Lexical TF-IDF fused with RRF."""
    def __init__(self, bi: SentenceTransformer, records: Dict[str, Dict[str, Any]]):
        self.bi = bi
        self.records = records
        self.keys = sorted(records.keys())

        # Passage card dense embeddings
        passage_cards = [passage_card(records[k]) for k in self.keys]
        self.card_embs = self.bi.encode(passage_cards, normalize_embeddings=True, show_progress_bar=False)

        # Scenario dense embeddings with owner index mapping
        scenario_texts = []
        scenario_owner_idx = []
        for idx, k in enumerate(self.keys):
            for s in records[k]["scenarios_json"]:
                scenario_texts.append(s["scenario"])
                scenario_owner_idx.append(idx)
        self.scen_embs = self.bi.encode(scenario_texts, normalize_embeddings=True, show_progress_bar=False)
        self.scenario_owner_idx = np.array(scenario_owner_idx)

        # Lexical TF-IDF index
        tfidf_docs = [tfidf_doc(records[k]) for k in self.keys]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(tfidf_docs)

    def search_with_scores(self, query_texts: List[str], top_k: int = 7) -> List[Tuple[List[str], Dict[str, float]]]:
        q_embs = self.bi.encode([QUERY_PREFIX + t for t in query_texts], normalize_embeddings=True, show_progress_bar=False)

        card_sims = q_embs @ self.card_embs.T  # (num_q, 108)
        scen_sims = q_embs @ self.scen_embs.T  # (num_q, num_scenarios)

        max_scen_sims = np.zeros_like(card_sims)
        for p_idx in range(len(self.keys)):
            mask = (self.scenario_owner_idx == p_idx)
            max_scen_sims[:, p_idx] = np.max(scen_sims[:, mask], axis=1)

        dense_scores = 0.65 * card_sims + 0.35 * max_scen_sims

        tfidf_q = self.vectorizer.transform(query_texts)
        tfidf_scores = (tfidf_q @ self.tfidf_matrix.T).toarray()

        results = []
        for i in range(len(query_texts)):
            d_order = np.argsort(-dense_scores[i])
            t_order = np.argsort(-tfidf_scores[i])
            rank_d = {self.keys[idx]: r for r, idx in enumerate(d_order, 1)}
            rank_t = {self.keys[idx]: r for r, idx in enumerate(t_order, 1)}

            rrf_scores = {k: 1.0 / (20.0 + rank_d[k]) + 1.0 / (20.0 + rank_t[k]) for k in self.keys}
            seeds = sorted(self.keys, key=lambda k: (-rrf_scores[k], k))[:top_k]
            results.append((seeds, rrf_scores))
        return results


def batch_score_ce(model, tokenizer, pairs: List[List[str]], device: str, batch_size: int = 32, max_length: int = 384) -> List[float]:
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


def batch_score_nli(model, tokenizer, premise: str, hypotheses: List[str], entail_idx: int, contra_idx: int,
                    device: str, batch_size: int = 32, max_length: int = 256) -> List[float]:
    nli_scores = []
    with torch.inference_mode():
        for i in range(0, len(hypotheses), batch_size):
            chunk = hypotheses[i : i + batch_size]
            pairs = [[premise, h] for h in chunk]
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            diff = (probs[:, entail_idx] - probs[:, contra_idx]).cpu().tolist()
            if isinstance(diff, float):
                diff = [diff]
            nli_scores.extend(diff)
    return nli_scores


class DharmRC1Engine:
    """Production Release Candidate 1 Inference Engine.

    Encapsulates all models, indexes, and logic for cold-start initialization
    and low-latency per-query raw text scoring.
    """
    def __init__(self,
                 data_dir: Path = DATA_DIR_DEFAULT,
                 artifacts_v1_1: Path = ARTIFACTS_V1_1_DEFAULT,
                 artifacts_v1_6: Path = ARTIFACTS_V1_6_DEFAULT,
                 device: str = "cpu"):
        self.device = device
        self.data_dir = Path(data_dir)
        self.artifacts_v1_1 = Path(artifacts_v1_1)
        self.artifacts_v1_6 = Path(artifacts_v1_6)

        # 1. Load Corpus
        self.records = load_corpus(self.data_dir)
        self.keys = sorted(self.records.keys())
        self.passage_cards = {k: passage_card(self.records[k]) for k in self.keys}
        self.retain_texts = {k: self.records[k][CONTRAST]["retain_self_when"] for k in self.keys}
        self.flip_texts = {k: self.records[k][CONTRAST]["flip_to_competing_when"] for k in self.keys}

        # 2. Neural Embedder & Stage-1 Index
        self.bi_model = SentenceTransformer(BI_MODEL, device=self.device)
        self.hybrid_index = MultiVectorHybridRRFIndex(self.bi_model, self.records)

        # 3. Dedicated Abstention Classifier
        abst_path = self.artifacts_v1_1 / "abstention_clf.joblib"
        require(abst_path.exists(), f"Missing abstention model at {abst_path}")
        self.abstention_clf = joblib.load(abst_path)

        # 4. Cross-Encoder Reranker
        self.ce_tok = AutoTokenizer.from_pretrained(CE_MODEL)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1).to(self.device)
        self.ce_model.eval()

        # 5. NLI Cross-Encoder
        self.nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(self.device)
        self.nli_model.eval()
        self.entail_idx = next(i for i, l in self.nli_model.config.id2label.items() if l.lower().startswith("entail"))
        self.contra_idx = next(i for i, l in self.nli_model.config.id2label.items() if l.lower().startswith("contra"))

        # 6. Workflow Intent Router
        router_path = self.artifacts_v1_6 / "workflow_router.joblib"
        require(router_path.exists(), f"Missing router model at {router_path}")
        self.router_clf = joblib.load(router_path)

        # 7. Pair Scorer Margins
        pair_cfg_path = self.artifacts_v1_6 / "pair_scorer_config.json"
        require(pair_cfg_path.exists(), f"Missing pair scorer config at {pair_cfg_path}")
        pair_cfg = json.loads(pair_cfg_path.read_text(encoding="utf8"))
        self.tau_self = float(pair_cfg["tau_self"])  # -0.5
        self.tau_pool = float(pair_cfg["tau_pool"])  # -1.0

        # 8. Pair Verifier Classifier & Config
        verifier_cfg_path = self.artifacts_v1_6 / "verifier_config.json"
        require(verifier_cfg_path.exists(), f"Missing verifier config at {verifier_cfg_path}")
        self.verifier_cfg = json.loads(verifier_cfg_path.read_text(encoding="utf8"))
        self.verifier_thresh = float(self.verifier_cfg["calibrated_threshold"])  # 0.5700
        self.feature_names = self.verifier_cfg["features"]

        verifier_model_path = self.artifacts_v1_6 / "pair_verifier_model.joblib"
        require(verifier_model_path.exists(), f"Missing verifier model at {verifier_model_path}")
        self.verifier_clf = joblib.load(verifier_model_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """Execute complete inference on raw dilemma text string."""
        text = text.strip()
        require(len(text) > 0, "Dilemma text must not be empty.")

        # --- Stage 0: Epistemic Abstention Gate ---
        q_emb = self.bi_model.encode([QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)[0]
        abst_prob = float(self.abstention_clf.predict_proba([q_emb])[0, 1])
        abstained = bool(abst_prob > 0.65)

        # --- Stage 1: Multi-Vector Stage-1 Hybrid RRF Retrieval ---
        search_res = self.hybrid_index.search_with_scores([text], top_k=7)
        seeds, rrf_scores = search_res[0]
        pool = expand_pool(seeds, self.records)

        # --- Stage 2: Frozen v1.3 Fallback Candidate Ranking ---
        rrf_norm = min_max_normalize(rrf_scores, pool)

        # Single cross-encoder passage scores
        ce_pairs = [[text, self.passage_cards[k]] for k in pool]
        ce_raw_scores = batch_score_ce(self.ce_model, self.ce_tok, ce_pairs, self.device)
        s_dict = dict(zip(pool, ce_raw_scores))
        ce_norm = min_max_normalize(s_dict, pool)

        base = {k: 0.55 * rrf_norm[k] + 0.45 * ce_norm[k] for k in pool}
        final = base.copy()

        # NLI directional arbitration on top contenders
        r_nli_vals = batch_score_nli(self.nli_model, self.nli_tok, text, [self.retain_texts[k] for k in pool],
                                     self.entail_idx, self.contra_idx, self.device)
        f_nli_vals = batch_score_nli(self.nli_model, self.nli_tok, text, [self.flip_texts[k] for k in pool],
                                     self.entail_idx, self.contra_idx, self.device)
        r_scores = dict(zip(pool, r_nli_vals))
        f_scores = dict(zip(pool, f_nli_vals))

        processed_pairs = set()
        for u in pool:
            v = self.records[u][CONTRAST]["id"]
            if v in pool:
                pair_key = tuple(sorted([u, v]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                topical_max = max(base[u], base[v])
                if topical_max >= 0.72:
                    r_u, f_u = r_scores[u], f_scores[u]
                    r_v, f_v = r_scores[v], f_scores[v]
                    net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))
                    if net_uv > 0.05:
                        final[v] = max(final[v], topical_max + 0.15 * net_uv)
                        final[u] = min(final[u], topical_max - 0.05 * net_uv)
                    elif net_uv < -0.05:
                        net_vu = -net_uv
                        final[u] = max(final[u], topical_max + 0.15 * net_vu)
                        final[v] = min(final[v], topical_max - 0.05 * net_vu)

        # v1.3 workflow ordering bonus
        workflow_pairs = set()
        for u in pool:
            c = self.records[u][WORKFLOW]["id"]
            if c in pool:
                pair_key = tuple(sorted([u, c]))
                if pair_key in workflow_pairs:
                    continue
                workflow_pairs.add(pair_key)
                if max(base[u], base[c]) >= 0.70 and min(base[u], base[c]) >= 0.35:
                    seq = self.records[u][WORKFLOW]["sequence_role"]
                    if seq.startswith("Follow with"):
                        first, second = u, c
                    else:
                        first, second = c, u
                    pair_top = max(base[first], base[second])
                    final[first] = max(final[first], pair_top + 0.10)
                    final[second] = max(final[second], pair_top + 0.06)

        fallback_top3 = sorted(pool, key=lambda k: (-final[k], k))[:3]
        fallback_primary = "Abstain" if abstained else fallback_top3[0]

        # Upstream router intent probability
        router_prob = float(self.router_clf.predict_proba([q_emb])[0, 1])

        # --- Stage 3: v1.6 Candidate Pair Proposer & Stage 4: Verifier ---
        pair_promoted = False
        promoted_pair = None
        proposed_pair = None
        acceptance_score = None
        fallback_reason = None
        features = None

        if abstained:
            fallback_reason = "abstained"
            final_top3 = fallback_top3
            final_primary = fallback_primary
        else:
            # Enumerate complementary pairs in pool
            pairs_in_pool = []
            for u in pool:
                wf = self.records[u][WORKFLOW]
                v = wf["id"]
                if v in pool and wf["sequence_role"].startswith("Follow with"):
                    pairs_in_pool.append((u, v))

            if not pairs_in_pool:
                fallback_reason = "no_pairs_in_pool"
                final_top3 = fallback_top3
                final_primary = fallback_primary
            else:
                # Score joint cards
                joint_pairs = [[text, pair_joint_card(self.records[u], self.records[v])] for u, v in pairs_in_pool]
                joint_scores = batch_score_ce(self.ce_model, self.ce_tok, joint_pairs, self.device)
                j_dict = dict(zip(pairs_in_pool, joint_scores))

                max_pool_s = max(s_dict.values())
                qualifying = []
                for (u, v), j_val in j_dict.items():
                    max_self = max(s_dict[u], s_dict[v])
                    min_self = min(s_dict[u], s_dict[v])
                    diff_self = j_val - max_self
                    diff_pool = j_val - max_pool_s
                    diff_min = j_val - min_self
                    span = max_self - min_self
                    if diff_self >= self.tau_self and diff_pool >= self.tau_pool:
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
                    fallback_reason = "pair_scorer_declined"
                    final_top3 = fallback_top3
                    final_primary = fallback_primary
                else:
                    qualifying.sort(key=lambda x: -x["j_score"])
                    best = qualifying[0]
                    u_prop, v_prop = best["pair"]
                    proposed_pair = [u_prop, v_prop]

                    # Rank features in pool based on single-passage cross-encoder scores
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
                        router_prob,
                    ]
                    features = dict(zip(self.feature_names, feat_vector))

                    v_prob = float(self.verifier_clf.predict_proba([feat_vector])[0, 1])
                    acceptance_score = v_prob

                    if v_prob >= self.verifier_thresh:
                        pair_promoted = True
                        promoted_pair = [u_prop, v_prop]
                        remaining = [k for k in fallback_top3 if k not in (u_prop, v_prop)]
                        for k in pool:
                            if k not in (u_prop, v_prop) and k not in remaining:
                                remaining.append(k)
                        final_top3 = [u_prop, v_prop] + remaining[:1]
                        final_primary = u_prop
                    else:
                        fallback_reason = "verifier_rejected"
                        final_top3 = fallback_top3
                        final_primary = fallback_primary

        return {
            "primary": final_primary,
            "top3": final_top3,
            "abstained": abstained,
            "abstention_prob": abst_prob,
            "proposed_pair": proposed_pair,
            "promoted_pair": promoted_pair,
            "pair_promoted": pair_promoted,
            "acceptance_score": acceptance_score,
            "fallback_reason": fallback_reason,
            "router_prob": router_prob,
            "seeds": seeds,
            "pool": pool,
            "fallback_top3": fallback_top3,
            "fallback_primary": fallback_primary,
            "features": features,
        }


def main():
    ap = argparse.ArgumentParser(description="Dharm RC1 Standalone Inference CLI")
    ap.add_argument("--text", type=str, help="Raw dilemma text string to evaluate")
    ap.add_argument("--file", type=Path, help="Path to text file containing a dilemma text")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("Must provide either --text or --file.")

    if args.file:
        dilemma_text = args.file.read_text(encoding="utf8").strip()
    else:
        dilemma_text = args.text.strip()

    t0 = time.perf_counter()
    engine = DharmRC1Engine(device=args.device)
    init_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = engine.predict(dilemma_text)
    infer_time = time.perf_counter() - t1

    if args.json:
        out = dict(pred, init_time_sec=init_time, infer_time_sec=infer_time)
        print(json.dumps(out, indent=2))
    else:
        print("=" * 80)
        print("Dharm RC1 Decision Output")
        print("=" * 80)
        print(f"Dilemma: {dilemma_text[:120]}..." if len(dilemma_text) > 120 else f"Dilemma: {dilemma_text}")
        print(f"Abstained:       {pred['abstained']} (prob = {pred['abstention_prob']:.4f})")
        print(f"Primary Decision:{pred['primary']}")
        print(f"Ordered Top-3:   {pred['top3']}")
        print(f"Pair Promoted:   {pred['pair_promoted']}")
        if pred["proposed_pair"]:
            print(f"Proposed Pair:   {pred['proposed_pair']}")
            print(f"Verifier Score:  {pred['acceptance_score']:.4f} (tau = {engine.verifier_thresh:.2f})")
        if pred["fallback_reason"]:
            print(f"Fallback Reason: {pred['fallback_reason']}")
        print(f"Timing:          Startup = {init_time:.2f}s, Inference = {infer_time:.3f}s")
        print("=" * 80)


if __name__ == "__main__":
    main()
