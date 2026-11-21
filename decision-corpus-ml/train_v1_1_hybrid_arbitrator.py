"""Decision Corpus v1.1 — Hybrid RRF Retrieval & Pretrained Margin-Tuned Cross-Encoder Arbitrator.

Upgrades:
1. Decouple epistemic abstention to a dedicated LogisticRegression classifier on frozen BGE-small embeddings.
2. Stage-1 Hybrid retrieval via Reciprocal Rank Fusion (Dense BGE-small + Lexical TF-IDF).
3. Preserve pretrained BAAI/bge-reranker-base (num_labels=1) and fine-tune with MarginRankingLoss on corpus triplets.
   At inference: Two-pass scoring (base relevance pass + neural directional flip pass + workflow synthesis order pass).
4. Full evaluation and 3-way baseline comparison against TF-IDF v0.2.3 and v1.0 Broken-Head baseline.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BI = "BAAI/bge-small-en-v1.5"
CE = "BAAI/bge-reranker-base"
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


def load_training(folder):
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


def arbitrator_card(r):
    edge = r[CONTRAST]
    return (f"Principle: {r['derived_principle']}\nRetain: {edge['retain_self_when']}\n"
            f"Flip: {edge['flip_to_competing_when']}\nLimit: {r['application_limit']}\n"
            f"Workflow: {r[WORKFLOW]['sequence_role']}")


def misapplication_card(r):
    return f"Principle: {r['derived_principle']}\nMisapplication: {r['misapplication_scenario']}"


def tfidf_doc(r):
    scenarios_text = " ".join(s["scenario"] for s in r["scenarios_json"])
    comp_cond = r[CONTRAST].get("condition", "")
    seq_role = r[WORKFLOW].get("sequence_role", "")
    return f"{passage_card(r)}\n{scenarios_text}\n{comp_cond}\n{seq_role}"


def anchors(records):
    return [(s["scenario"], k) for k, r in records.items() for s in r["scenarios_json"]]


def synthetic_empirical_queries():
    """~80 diverse synthetic empirical/measurement/historical-date/physical-provenance queries.
    Zero copying from benchmark Q17, Q18, Q41, Q42."""
    out = []
    # 1. Arithmetic & Calculation
    for i in range(1, 9):
        out.append(f"What is the mathematical product of {13*i} and {29+i}?")
    # 2. Historical & Catalog Dating
    for i in range(1, 9):
        out.append(f"Give the exact calendar year in which manuscript ledger {700+i} was first bound.")
    # 3. Archival Workshop & Provenance
    for i in range(1, 9):
        out.append(f"Which metalworking shop cast bronze artifact accession {1200+i} in the museum collection?")
    # 4. Instrument & Sensor Calibration
    instruments = ["barometer", "galvanometer", "manometer", "pyrometer", "spectrometer", "viscometer", "tachometer", "voltmeter"]
    for i, inst in enumerate(instruments, 1):
        out.append(f"What was the recorded baseline offset of the {inst} during the morning trial?")
    # 5. Material Density & Composition
    materials = ["granite", "basalt", "feldspar", "obsidian", "limestone", "shale", "sandstone", "diorite"]
    for i, mat in enumerate(materials, 1):
        out.append(f"What specific gravity was determined in the assay for core specimen {i} of {mat}?")
    # 6. Physical Provenance & Geological Origin
    geologies = ["clay deposit", "limestone quarry", "alluvial silt", "gravel bed", "marble vein", "flint seam", "iron pit", "copper outcrop"]
    for i, geo in enumerate(geologies, 1):
        out.append(f"From which geological {geo} did the excavated masonry block in sector {i} derive?")
    # 7. Acoustic & Vibration Durations
    for i in range(1, 9):
        out.append(f"How many milliseconds did the acoustic resonant vibration persist in chamber {i} after impulse?")
    # 8. Thermal & Phase Measurements
    for i in range(1, 9):
        out.append(f"At what Celsius temperature did crystal batch {40+i} reach its eutectic melting point?")
    # 9. Chemical & Isotopic Assay
    elements = ["carbon", "strontium", "lead", "argon", "oxygen", "nitrogen", "sulfur", "helium"]
    for i, el in enumerate(elements, 1):
        out.append(f"What numerical {el} isotope ratio was detected in uncalibrated test specimen {i}?")
    # 10. Elapsed Time & Interval Counting
    for i in range(1, 9):
        out.append(f"How many total seconds elapsed between oscillation {10*i} and oscillation {25*i} in the sealed test?")
    return out


def train_abstention_classifier(bi, records):
    """Upgrade 1: Dedicated Abstention Classifier on frozen BGE-small query embeddings."""
    in_scope = [s["scenario"] for r in records.values() for s in r["scenarios_json"]]
    out_of_scope = synthetic_empirical_queries()
    
    X_in = bi.encode([QUERY_PREFIX + q for q in in_scope], normalize_embeddings=True, show_progress_bar=False)
    X_out = bi.encode([QUERY_PREFIX + q for q in out_of_scope], normalize_embeddings=True, show_progress_bar=False)
    
    X = np.vstack([X_in, X_out])
    y = np.array([0] * len(X_in) + [1] * len(X_out))
    
    clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    print(f"Trained Abstention Classifier: {len(X_in)} in-scope (class 0), {len(X_out)} out-of-scope (class 1).")
    return clf


class HybridRRFIndex:
    """Upgrade 2: Stage 1 Hybrid Retrieval via Reciprocal Rank Fusion (Dense + Lexical)."""
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
        
    def search(self, query_texts, top_k=5):
        dense_q = self.bi.encode([QUERY_PREFIX + t for t in query_texts], normalize_embeddings=True, show_progress_bar=False)
        dense_scores = dense_q @ self.dense_matrix.T
        
        tfidf_q = self.vectorizer.transform(query_texts)
        tfidf_scores = (tfidf_q @ self.tfidf_matrix.T).toarray()
        
        all_seeds = []
        for i in range(len(query_texts)):
            d_order = np.argsort(-dense_scores[i])
            t_order = np.argsort(-tfidf_scores[i])
            rank_d = {self.keys[idx]: r for r, idx in enumerate(d_order, 1)}
            rank_t = {self.keys[idx]: r for r, idx in enumerate(t_order, 1)}
            
            rrf_scores = {k: 1.0 / (20.0 + rank_d[k]) + 1.0 / (20.0 + rank_t[k]) for k in self.keys}
            seeds = sorted(self.keys, key=lambda k: (-rrf_scores[k], k))[:top_k]
            all_seeds.append(seeds)
        return all_seeds


def expand_pool(seeds, records):
    """Take top-5 RRF seeds and expand 1 hop along both competing and complementary edges."""
    return sorted({j for k in seeds for j in (k, records[k][CONTRAST]["id"], records[k][WORKFLOW]["id"])})


def build_training_triplets(records):
    """Upgrade 3: Corpus-only training triplets for Pairwise Margin Ranking Loss."""
    triplets = []
    for k, r in records.items():
        cid = r[CONTRAST]["id"]
        c_pos = arbitrator_card(r)
        c_comp = arbitrator_card(records[cid])
        c_mis = misapplication_card(r)
        
        # (scenario, card_pos, card_competing) -> y = +1
        # (scenario, card_pos, misapplication_card) -> y = +1
        for s in r["scenarios_json"]:
            triplets.append((s["scenario"], c_pos, c_comp))
            triplets.append((s["scenario"], c_pos, c_mis))
            
        # (retain_self_when, card_self, card_competing) -> y = +1
        # (flip_to_competing_when, card_competing, card_self) -> y = +1
        ret = r[CONTRAST]["retain_self_when"]
        flp = r[CONTRAST]["flip_to_competing_when"]
        triplets.append((ret, c_pos, c_comp))
        triplets.append((flp, c_comp, c_pos))
        
    return triplets


def train_margin_ce(model, tokenizer, triplets, args, device):
    """Upgrade 3: Fine-tune BAAI/bge-reranker-base (num_labels=1) with MarginRankingLoss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = (len(triplets) * args.epochs_ce) // (args.batch_size * args.gradient_accumulation_steps)
    warmup_steps = math.ceil(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    margin_loss_fn = nn.MarginRankingLoss(margin=1.0)
    
    print(f"Fine-tuning Cross-Encoder Arbitrator: {len(triplets)} triplets, {args.epochs_ce} epochs, "
          f"batch_size={args.batch_size}, accum={args.gradient_accumulation_steps} (effective batch {args.batch_size*args.gradient_accumulation_steps}).")
    
    step = 0
    for epoch in range(args.epochs_ce):
        random.Random(args.seed + epoch).shuffle(triplets)
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for b_start in tqdm(range(0, len(triplets), args.batch_size), desc=f"CE Epoch {epoch+1}"):
            batch = triplets[b_start : b_start + args.batch_size]
            qs, pos_docs, neg_docs = zip(*batch)
            
            pos_inputs = tokenizer(list(qs), list(pos_docs), padding=True, truncation=True,
                                   max_length=args.max_length, return_tensors="pt").to(device)
            neg_inputs = tokenizer(list(qs), list(neg_docs), padding=True, truncation=True,
                                   max_length=args.max_length, return_tensors="pt").to(device)
            
            pos_logits = model(**pos_inputs).logits.squeeze(-1)
            neg_logits = model(**neg_inputs).logits.squeeze(-1)
            
            target = torch.ones_like(pos_logits)
            loss_rank = margin_loss_fn(pos_logits, neg_logits, target)
            loss_bce_pos = nn.functional.binary_cross_entropy_with_logits(pos_logits, target)
            loss_bce_neg = nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            
            # Combine margin ranking with BCE calibration to keep sigmoid calibrated around (0, 1)
            loss = (loss_rank + 0.5 * (loss_bce_pos + loss_bce_neg)) / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            
            step += 1
            if step % args.gradient_accumulation_steps == 0 or (b_start + args.batch_size >= len(triplets)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
        avg_loss = epoch_loss / math.ceil(len(triplets) / args.batch_size)
        print(f"CE Epoch {epoch+1} Complete — Avg Loss: {avg_loss:.4f}")
        
    return model.eval()


def batch_score_ce(model, tokenizer, query_text, candidate_texts, device, batch_size=16, max_length=384):
    """Helper to score (query, candidate) pairs through the cross-encoder and return sigmoids."""
    scores = []
    with torch.inference_mode():
        for i in range(0, len(candidate_texts), batch_size):
            chunk = candidate_texts[i : i + batch_size]
            pairs = [[query_text, t] for t in chunk]
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze(-1)
            sig = torch.sigmoid(logits).cpu().tolist()
            if isinstance(sig, float):
                sig = [sig]
            scores.extend(sig)
    return scores


def predict_v1_1(questions, records, hybrid_index, ce_model, tokenizer, abstention_clf, bi, args, device):
    """Two-Pass Pretrained Cross-Encoder Scoring + Explicit Directional & Workflow Arbitration."""
    dilemma_texts = [q["dilemma_text"] for q in questions]
    
    # 1. Epistemic Abstention check
    q_embs = bi.encode([QUERY_PREFIX + t for t in dilemma_texts], normalize_embeddings=True, show_progress_bar=False)
    abst_probs = abstention_clf.predict_proba(q_embs)[:, 1]
    
    # 2. Stage 1 Hybrid RRF Retrieval
    all_seeds = hybrid_index.search(dilemma_texts, top_k=5)
    
    cards = {k: arbitrator_card(records[k]) for k in records}
    retain_texts = {k: records[k][CONTRAST]["retain_self_when"] for k in records}
    flip_texts = {k: records[k][CONTRAST]["flip_to_competing_when"] for k in records}
    
    results = []
    for q, text, seeds, abst_prob in zip(questions, dilemma_texts, all_seeds, abst_probs):
        pool = expand_pool(seeds, records)
        
        # Pass 1: Base Relevance Pass
        base_scores = batch_score_ce(ce_model, tokenizer, text, [cards[k] for k in pool], device,
                                     batch_size=16, max_length=args.max_length)
        base = dict(zip(pool, base_scores))
        final = base.copy()
        
        # Pass 2: Directional Flip Pass
        r_scores = dict(zip(pool, batch_score_ce(ce_model, tokenizer, text, [retain_texts[k] for k in pool], device,
                                                 batch_size=16, max_length=256)))
        f_scores = dict(zip(pool, batch_score_ce(ce_model, tokenizer, text, [flip_texts[k] for k in pool], device,
                                                 batch_size=16, max_length=256)))
        
        for u in pool:
            v = records[u][CONTRAST]["id"]
            if v in pool:
                r_u = r_scores[u]
                f_u = f_scores[u]
                r_v = r_scores[v]
                f_v = f_scores[v]
                net_uv = 0.5 * ((f_u - r_u) + (r_v - f_v))
                if net_uv > 0.05:
                    final[v] = max(final[v], max(base[u], base[v]) + 0.35 * net_uv)
                    
        # Pass 3: Workflow Order Pass (Synthesis)
        for u in pool:
            c = records[u][WORKFLOW]["id"]
            if c in pool:
                if min(base[u], base[c]) >= 0.35 and max(base[u], base[c]) >= 0.55:
                    seq = records[u][WORKFLOW]["sequence_role"]
                    if seq.startswith("Follow with"):
                        first, second = u, c
                    else:
                        first, second = c, u
                    pair_top = max(base[first], base[second])
                    final[first] = max(final[first], pair_top + 0.08)
                    final[second] = max(final[second], pair_top + 0.04)
                    
        top3 = sorted(pool, key=lambda k: (-final[k], k))[:3]
        abstained = bool(abst_prob > 0.65)
        primary = "Abstain" if abstained else top3[0]
        
        results.append(dict(
            q,
            seeds=seeds,
            pool=pool,
            primary=primary,
            top3=top3,
            abstained=abstained,
            abstention_prob=float(abst_prob),
            base_relevance=base,
            adjusted_relevance=final
        ))
        
    return results


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


def show(name, model, m):
    def fraction(k):
        a, b = m[k]
        return f"{a}/{b}" if b else "undefined"
    print(f"| {name} | {model} | {100*m['recall_at_3']:.2f}% | " +
          " | ".join(fraction(k) for k in ("flip", "abstention_precision", "abstention_recall", "ordered_synthesis")) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs_ce", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts-v1.1"))
    ap.add_argument("--skip_finetune", action="store_true", help="Zero-shot ablation using pretrained weights only.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device
    print(f"Running Decision Corpus v1.1 on {device} (skip_finetune={args.skip_finetune})")
    
    records = load_training(args.data_dir)
    print(f"Loaded {len(records)} passages and {len(anchors(records))} total scenarios.")
    
    # Initialize frozen BGE-small for Stage 1 dense embeddings & epistemic classifier
    bi = SentenceTransformer(BI, device=device)
    
    # Upgrade 1: Train dedicated abstention classifier
    abstention_clf = train_abstention_classifier(bi, records)
    
    # Upgrade 2: Build Stage 1 Hybrid RRF Index
    hybrid_index = HybridRRFIndex(bi, records)
    
    # Upgrade 3: Load pretrained bge-reranker-base preserving num_labels=1
    tokenizer = AutoTokenizer.from_pretrained(CE)
    ce_model = AutoModelForSequenceClassification.from_pretrained(CE, num_labels=1).to(device)
    
    if not args.skip_finetune:
        triplets = build_training_triplets(records)
        ce_model = train_margin_ce(ce_model, tokenizer, triplets, args, device)
    else:
        print("Ablation: Skipping CE fine-tuning (evaluating 100% zero-shot pretrained weights).")
        
    # Benchmark evaluation
    eval_rows = [json.loads(line) for line in (args.data_dir / "eval_queries_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    predictions = predict_v1_1(eval_rows, records, hybrid_index, ce_model, tokenizer, abstention_clf, bi, args, device)
    
    slices = {
        "original": predictions[:18],
        "new": predictions[18:],
        "all": predictions
    }
    
    metrics = {k: evaluate(v) for k, v in slices.items()}
    
    # Baselines for 3-way table
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
    
    print("\n### 3-Way Benchmark Comparison Table")
    print("| Slice | Model | Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for name in ("original", "new", "all"):
        show(name, "TF-IDF v0.2.3", tfidf_baseline[name])
        show(name, "v1.0 Broken-Head Baseline", v1_0_baseline[name])
        label = "v1.1 Hybrid RRF (Zero-Shot Ablation)" if args.skip_finetune else "v1.1 Hybrid RRF + Margin Arbitrator"
        show(name, label, metrics[name])
        print("|---|---|---:|---:|---:|---:|---:|")
        
    # Save artifacts
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ce_model.save_pretrained(out_dir / "cross_encoder_arbitrator")
    tokenizer.save_pretrained(out_dir / "cross_encoder_arbitrator")
    bi.save(str(out_dir / "bi_encoder"))
    joblib.dump(abstention_clf, out_dir / "abstention_clf.joblib")
    
    inputs = ["decisions_v0.2.jsonl", "chanakya_daily_supplement.jsonl", "eval_queries_v0.2.jsonl"]
    fingerprints = {n: hashlib.sha256((args.data_dir / n).read_bytes()).hexdigest() for n in inputs}
    
    report = dict(
        version="v1.1",
        config=vars(args),
        device=device,
        models=[BI, CE],
        torch_version=torch.__version__,
        fingerprints=fingerprints,
        baseline_tfidf=tfidf_baseline,
        baseline_v1_0=v1_0_baseline,
        metrics=metrics,
        predictions=predictions
    )
    
    report_path = out_dir / "v1_1_ml_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf8")
    
    # Store environment lock and input hashes together
    lock_file = Path("environment.lock.txt")
    if lock_file.exists():
        (out_dir / "environment.lock.txt").write_bytes(lock_file.read_bytes())
        
    hash_lines = [f"{digest}  data/{name}\n" for name, digest in fingerprints.items()]
    (out_dir / "input_hashes.sha256").write_text("".join(hash_lines), encoding="utf8")
    
    print(f"\nSaved v1.1 models, report, and locks to: {out_dir}")


if __name__ == "__main__":
    main()
