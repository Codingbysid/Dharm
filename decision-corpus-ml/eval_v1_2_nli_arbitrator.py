"""Decision Corpus v1.2 — NLI Directional Flipping + RRF-Blended Reranking.

Exact Changes over v1.1:
1. Reuse v1.1 Abstention Classifier (4/4 P, 4/4 R) & Stage-1 Hybrid RRF (84.2% pool recall).
   Min-max normalize Stage-1 RRF scores within candidate pool to [0.0, 1.0].
2. Blend Stage-1 RRF with pretrained BAAI/bge-reranker-base on passage_card (min-max normalized):
   base[i] = 0.55 * rrf_norm[i] + 0.45 * ce_norm[i].
3. Pretrained NLI Cross-Encoder (cross-encoder/nli-deberta-v3-small) for Directional Flip Pass:
   nli(premise, hypothesis) = P(entailment) - P(contradiction).
   Directional net flip margin: net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v)).
   If net_uv > 0.05: final[v] = max(final[v], max(base[u], base[v]) + 0.30 * net_uv).
   Synthesis Workflow Order Pass: min(base) >= 0.40 and max(base) >= 0.65 -> boost first (+0.08) and second (+0.04).
4. Full evaluation and 4-way comparison table + per-pair flip breakdown & synthesis breakdown.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

BI = "BAAI/bge-small-en-v1.5"
CE = "BAAI/bge-reranker-base"
NLI = "cross-encoder/nli-deberta-v3-small"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), f"Duplicate {key} in {path}")
    return dict(sorted(result.items()))


def load_corpus(folder):
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
    require(set(notes) == {f"CD{i:02}" for i in range(1, 10)}, "Expected CD01–CD09")
    for note, row in notes.items():
        target = row["existing_record_id"]
        require(target == SUPPLEMENT.get(note), f"Unexpected mapping for {note}")
        if target:
            records[target]["scenarios_json"].append({"scenario": row["modern_application"]})
    return records


def passage_card(r):
    return (f"[{r['id']}] {r['work']} {r['locator']} | Domain: {r['decision_domain']} | "
            f"Urgency: {r['urgency_level']} | Reversibility: {r['reversibility']}\n"
            f"Principle: {r['derived_principle']}\nContext: {r['context_summary']}\n"
            f"Text: {r['english_translation']}\nCues: {r['retrieval_cues']}")


def tfidf_doc(r):
    scenarios_text = " ".join(s["scenario"] for s in r["scenarios_json"])
    comp_cond = r[CONTRAST].get("condition", "")
    seq_role = r[WORKFLOW].get("sequence_role", "")
    return f"{passage_card(r)}\n{scenarios_text}\n{comp_cond}\n{seq_role}"


class HybridRRFIndex:
    def __init__(self, bi, records):
        self.bi = bi
        self.records = records
        self.keys = sorted(records.keys())
        
        # Dense index
        passage_cards = [passage_card(records[k]) for k in self.keys]
        self.dense_matrix = self.bi.encode(passage_cards, normalize_embeddings=True, show_progress_bar=False)
        
        # Lexical TF-IDF index
        tfidf_docs = [tfidf_doc(records[k]) for k in self.keys]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(tfidf_docs)
        
    def search_with_scores(self, query_texts, top_k=5):
        dense_q = self.bi.encode([QUERY_PREFIX + t for t in query_texts], normalize_embeddings=True, show_progress_bar=False)
        dense_scores = dense_q @ self.dense_matrix.T
        
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


def expand_pool(seeds, records):
    return sorted({j for k in seeds for j in (k, records[k][CONTRAST]["id"], records[k][WORKFLOW]["id"])})


def min_max_normalize(scores_dict, subset_keys):
    vals = [scores_dict[k] for k in subset_keys]
    mi, ma = min(vals), max(vals)
    if ma > mi:
        return {k: (scores_dict[k] - mi) / (ma - mi) for k in subset_keys}
    return {k: 1.0 for k in subset_keys}


def batch_score_ce(model, tokenizer, query_text, candidate_texts, device, batch_size=16, max_length=384):
    scores = []
    with torch.inference_mode():
        for i in range(0, len(candidate_texts), batch_size):
            chunk = candidate_texts[i : i + batch_size]
            pairs = [[query_text, t] for t in chunk]
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze(-1)
            raw = logits.cpu().tolist()
            if isinstance(raw, float):
                raw = [raw]
            scores.extend(raw)
    return scores


def batch_score_nli(model, tokenizer, premise, hypotheses, entail_idx, contra_idx, device, batch_size=16, max_length=256):
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


def evaluate(rows):
    recall, ordered, pairs = [], [], defaultdict(list)
    tp = pred = gold = 0
    for r in rows:
        p, c = r["gold_primary_id"], r["gold_competing_id"]
        pred += int(r["abstained"])
        if r["query_type"] == "abstention":
            gold += 1
            tp += int(r["abstained"])
            continue
        relevant = {p}
        if r["query_type"] == "synthesis":
            relevant.add(c)
            ordered.append(not r["abstained"] and r["top3"][:2] == [p, c])
        elif r.get("pair_group"):
            pairs[r["pair_group"]].append(r["primary"] == p)
        recall.append(len(relevant.intersection(r["top3"])) / len(relevant))
        
    require(all(len(group) == 2 for group in pairs.values()), "Incomplete arbitration pair")
    return dict(
        recall_at_3=sum(recall) / len(recall),
        flip=[sum(all(g) for g in pairs.values()), len(pairs)],
        abstention_precision=[tp, pred],
        abstention_recall=[tp, gold],
        ordered_synthesis=[sum(ordered), len(ordered)]
    )


def show_row(name, model, m):
    def fraction(k):
        a, b = m[k]
        return f"{a}/{b}" if b else "undefined"
    print(f"| {name} | {model} | {100*m['recall_at_3']:.2f}% | " +
          " | ".join(fraction(k) for k in ("flip", "abstention_precision", "abstention_recall", "ordered_synthesis")) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--v1_1_artifacts", type=Path, default=Path("artifacts-v1.1"))
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts-v1.2"))
    args = ap.parse_args()
    
    device = args.device
    print(f"Running Decision Corpus v1.2 NLI Arbitrator on device: {device}")
    
    records = load_corpus(args.data_dir)
    keys = sorted(records.keys())
    print(f"Loaded {len(records)} records from corpus.")
    
    # 1. Load v1.1 Abstention Classifier and initialize frozen BGE-small
    bi = SentenceTransformer(BI, device=device)
    abst_path = args.v1_1_artifacts / "abstention_clf.joblib"
    require(abst_path.exists(), f"Missing v1.1 abstention classifier at {abst_path}")
    abstention_clf = joblib.load(abst_path)
    print("Loaded v1.1 dedicated Abstention Classifier.")
    
    # Initialize Stage 1 Hybrid RRF Index
    hybrid_index = HybridRRFIndex(bi, records)
    print("Initialized Stage 1 Hybrid RRF Index (Dense + Lexical).")
    
    # 2. Load Pretrained BAAI/bge-reranker-base (zero-shot, passage_card)
    ce_tok = AutoTokenizer.from_pretrained(CE)
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE, num_labels=1).to(device)
    ce_model.eval()
    print("Loaded pretrained BAAI/bge-reranker-base.")
    
    # 3. Load Pretrained NLI Cross-Encoder
    nli_tok = AutoTokenizer.from_pretrained(NLI)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI).to(device)
    nli_model.eval()
    
    # Dynamically find entailment and contradiction indices
    entail_idx = next(i for i, l in nli_model.config.id2label.items() if l.lower().startswith("entail"))
    contra_idx = next(i for i, l in nli_model.config.id2label.items() if l.lower().startswith("contra"))
    print(f"Loaded {NLI} (entailment idx={entail_idx}, contradiction idx={contra_idx}).")
    
    # Load evaluation queries
    eval_queries = [json.loads(line) for line in (args.data_dir / "eval_queries_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    dilemma_texts = [q["dilemma_text"] for q in eval_queries]
    
    # Abstention inference
    q_embs = bi.encode([QUERY_PREFIX + t for t in dilemma_texts], normalize_embeddings=True, show_progress_bar=False)
    abst_probs = abstention_clf.predict_proba(q_embs)[:, 1]
    
    # Hybrid RRF Search
    rrf_results = hybrid_index.search_with_scores(dilemma_texts, top_k=5)
    
    passage_cards = {k: passage_card(records[k]) for k in keys}
    retain_texts = {k: records[k][CONTRAST]["retain_self_when"] for k in keys}
    flip_texts = {k: records[k][CONTRAST]["flip_to_competing_when"] for k in keys}
    
    predictions = []
    print("Executing two-pass RRF-blended reranking + NLI directional arbitration...")
    
    for i, (q, text, (seeds, rrf_scores), abst_prob) in enumerate(zip(eval_queries, dilemma_texts, rrf_results, abst_probs)):
        abstained = bool(abst_prob > 0.65)
        pool = expand_pool(seeds, records)
        
        # 1. Min-max normalize Stage-1 RRF within pool
        rrf_norm = min_max_normalize(rrf_scores, pool)
        
        # 2. Pretrained BGE-reranker on passage_card within pool (min-max normalized)
        ce_raw_scores = batch_score_ce(ce_model, ce_tok, text, [passage_cards[k] for k in pool], device,
                                       batch_size=16, max_length=384)
        ce_scores_dict = dict(zip(pool, ce_raw_scores))
        ce_norm = min_max_normalize(ce_scores_dict, pool)
        
        # Blended base score: 0.55 * rrf_norm + 0.45 * ce_norm
        base = {k: 0.55 * rrf_norm[k] + 0.45 * ce_norm[k] for k in pool}
        final = base.copy()
        
        # 3. NLI Directional Flip Pass
        r_nli_vals = batch_score_nli(nli_model, nli_tok, text, [retain_texts[k] for k in pool],
                                     entail_idx, contra_idx, device, batch_size=16, max_length=256)
        f_nli_vals = batch_score_nli(nli_model, nli_tok, text, [flip_texts[k] for k in pool],
                                     entail_idx, contra_idx, device, batch_size=16, max_length=256)
        r_scores = dict(zip(pool, r_nli_vals))
        f_scores = dict(zip(pool, f_nli_vals))
        
        for u in pool:
            v = records[u][CONTRAST]["id"]
            if v in pool:
                r_u = r_scores[u]
                f_u = f_scores[u]
                r_v = r_scores[v]
                f_v = f_scores[v]
                net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))
                if net_uv > 0.05:
                    final[v] = max(final[v], max(base[u], base[v]) + 0.30 * net_uv)
                    
        # 4. Workflow Order Pass (Synthesis)
        for u in pool:
            c = records[u][WORKFLOW]["id"]
            if c in pool:
                if min(base[u], base[c]) >= 0.40 and max(base[u], base[c]) >= 0.65:
                    seq = records[u][WORKFLOW]["sequence_role"]
                    if seq.startswith("Follow with"):
                        first, second = u, c
                    else:
                        first, second = c, u
                    pair_top = max(base[first], base[second])
                    final[first] = max(final[first], pair_top + 0.08)
                    final[second] = max(final[second], pair_top + 0.04)
                    
        top3 = sorted(pool, key=lambda k: (-final[k], k))[:3]
        primary = "Abstain" if abstained else top3[0]
        
        predictions.append(dict(
            q,
            seeds=seeds,
            pool=pool,
            primary=primary,
            top3=top3,
            abstained=abstained,
            abstention_prob=float(abst_prob),
            rrf_norm=rrf_norm,
            ce_norm=ce_norm,
            base_blended=base,
            final_scores=final
        ))
        
    slices = {
        "original": predictions[:18],
        "new": predictions[18:],
        "all": predictions
    }
    metrics = {k: evaluate(v) for k, v in slices.items()}
    
    # 4-Way Baseline Comparison
    tfidf_baseline = {
        "original": dict(recall_at_3=0.8125, flip=[4, 6], abstention_precision=[2, 2], abstention_recall=[2, 2], ordered_synthesis=[2, 4]),
        "new": dict(recall_at_3=14.5/22, flip=[0, 8], abstention_precision=[0, 0], abstention_recall=[0, 2], ordered_synthesis=[3, 6]),
        "all": dict(recall_at_3=27.5/38, flip=[4, 14], abstention_precision=[2, 2], abstention_recall=[2, 4], ordered_synthesis=[5, 10])
    }
    v1_0_baseline = {
        "original": dict(recall_at_3=0.65625, flip=[1, 6], abstention_precision=[2, 9], abstention_recall=[2, 2], ordered_synthesis=[0, 4]),
        "new": dict(recall_at_3=0.40909, flip=[1, 8], abstention_precision=[2, 10], abstention_recall=[2, 2], ordered_synthesis=[0, 6]),
        "all": dict(recall_at_3=0.51316, flip=[2, 14], abstention_precision=[4, 19], abstention_recall=[4, 4], ordered_synthesis=[0, 10])
    }
    v1_1_baseline = {
        "original": dict(recall_at_3=0.53125, flip=[2, 6], abstention_precision=[2, 2], abstention_recall=[2, 2], ordered_synthesis=[0, 4]),
        "new": dict(recall_at_3=0.52273, flip=[1, 8], abstention_precision=[2, 2], abstention_recall=[2, 2], ordered_synthesis=[1, 6]),
        "all": dict(recall_at_3=0.52632, flip=[3, 14], abstention_precision=[4, 4], abstention_recall=[4, 4], ordered_synthesis=[1, 10])
    }
    
    print("\n### 4-Way Benchmark Comparison Table")
    print("| Slice | Model | Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name in ("original", "new", "all"):
        show_row(name, "TF-IDF v0.2.3", tfidf_baseline[name])
        show_row(name, "v1.0 Broken-Head Baseline", v1_0_baseline[name])
        show_row(name, "v1.1 Hybrid RRF + Margin Arbitrator", v1_1_baseline[name])
        show_row(name, "v1.2 NLI Arbitrator (RRF-Blended)", metrics[name])
        print("|---|---|---:|---:|---:|---:|---:|")
        
    # Per-pair flip breakdown
    print("\n### Arbitration Pairs Breakdown (P1–P14)")
    groups = defaultdict(list)
    for r in predictions:
        if r["query_type"] == "arbitration_pair":
            groups[r["pair_group"]].append(r)
            
    for grp in sorted(groups.keys(), key=lambda g: int(g[1:])):
        pair = groups[grp]
        succ = all(r["primary"] == r["gold_primary_id"] for r in pair)
        mark = "SUCCESS" if succ else "FAIL"
        print(f"\nGroup {grp}: [{mark}]")
        for r in pair:
            gold = r["gold_primary_id"]
            hit = r["primary"] == gold
            print(f"  {r['query_id']} ({'MATCH' if hit else 'MISS'}): gold={gold}, primary={r['primary']}, "
                  f"seeds={r['seeds'][:3]}, in_pool={gold in r['pool']}, top3={r['top3']}")
            
    # Ordered synthesis breakdown
    print("\n### Synthesis Queries Breakdown (10 Queries)")
    for r in predictions:
        if r["query_type"] == "synthesis":
            gold_seq = [r["gold_primary_id"], r["gold_competing_id"]]
            succ = (r["top3"][:2] == gold_seq) and not r["abstained"]
            mark = "SUCCESS" if succ else "FAIL"
            print(f"  {r['query_id']}: [{mark}] gold={gold_seq}, top3={r['top3']}, abstained={r['abstained']}")
            
    # Save outputs to artifacts-v1.2/
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy abstention classifier to v1.2 for self-containment
    joblib.dump(abstention_clf, out_dir / "abstention_clf.joblib")
    
    inputs = ["decisions_v0.2.jsonl", "chanakya_daily_supplement.jsonl", "eval_queries_v0.2.jsonl"]
    fingerprints = {n: hashlib.sha256((args.data_dir / n).read_bytes()).hexdigest() for n in inputs}
    
    report = dict(
        version="v1.2",
        config=vars(args),
        device=device,
        models=[BI, CE, NLI],
        fingerprints=fingerprints,
        baseline_tfidf=tfidf_baseline,
        baseline_v1_0=v1_0_baseline,
        baseline_v1_1=v1_1_baseline,
        metrics=metrics,
        predictions=predictions
    )
    
    report_path = out_dir / "v1_2_ml_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf8")
    
    # Store environment lock and input hashes
    lock_file = Path("environment.lock.txt")
    if lock_file.exists():
        (out_dir / "environment.lock.txt").write_bytes(lock_file.read_bytes())
        
    hash_lines = [f"{digest}  data/{name}\n" for name, digest in fingerprints.items()]
    (out_dir / "input_hashes.sha256").write_text("".join(hash_lines), encoding="utf8")
    
    print(f"\nSaved v1.2 report and artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
