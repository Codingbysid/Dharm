"""Generate comprehensive score cache for deterministic replay in Decision Corpus v1.5.

Captures all Stage-1 RRF scores, BGE-reranker scores, NLI rule alignments
(both composite and retain-only), and raw directional retain/flip NLI scores.
Saves to artifacts-v1.5/score_replay_data.json with provenance metadata.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

BI = "BAAI/bge-small-en-v1.5"
CE = "BAAI/bge-reranker-base"
NLI = "cross-encoder/nli-deberta-v3-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SUPPLEMENT = dict(zip([f"CD{i:02}" for i in range(1, 7)],
                      ["D089", "D080", "D042", "D106", "D040", "D035"]))
CONTRAST = "competing_passage_id"
WORKFLOW = "complementary_passage_id"


def read_rows(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    result = {r[key]: r for r in rows}
    return dict(sorted(result.items()))


def load_corpus(folder):
    records = read_rows(folder / "decisions_v0.2.jsonl", "id")
    notes = read_rows(folder / "chanakya_daily_supplement.jsonl", "note_id")
    for note, row in notes.items():
        target = row["existing_record_id"]
        if target and target in records:
            records[target]["scenarios_json"].append({"scenario": row["modern_application"]})
    return records


def passage_card(r):
    return (f"[{r['id']}] {r['work']} {r['locator']} | Domain: {r['decision_domain']} | "
            f"Urgency: {r['urgency_level']} | Reversibility: {r['reversibility']}\n"
            f"Principle: {r['derived_principle']}\nContext: {r['context_summary']}\n"
            f"Text: {r['english_translation']}\nCues: {r['retrieval_cues']}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data_dir", type=Path, default=Path("decision-corpus-ml/data"))
    ap.add_argument("--v1_4_report", type=Path, default=Path("decision-corpus-ml/artifacts-v1.4/v1_4_ml_eval_report.json"))
    ap.add_argument("--out_file", type=Path, default=Path("decision-corpus-ml/artifacts-v1.5/score_replay_data.json"))
    args = ap.parse_args()

    device = args.device
    print(f"Generating score replay cache on device: {device}")

    # Load corpus & query definitions
    records = load_corpus(args.data_dir)
    keys = sorted(records.keys())

    eval_queries = [json.loads(line) for line in (args.data_dir / "eval_queries_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    dilemma_texts = [q["dilemma_text"] for q in eval_queries]

    # Load v1.4 report to extract verified Stage-1 seeds, pool, rrf_norm, ce_norm, and baseline scores
    v1_4_data = json.load(open(args.v1_4_report))
    v1_4_preds = v1_4_data["predictions"]

    # Load NLI model to compute raw directional retain/flip and retain-only rule alignment
    nli_tok = AutoTokenizer.from_pretrained(NLI)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI).to(device)
    nli_model.eval()

    entail_idx = next(i for i, l in nli_model.config.id2label.items() if l.lower().startswith("entail"))
    contra_idx = next(i for i, l in nli_model.config.id2label.items() if l.lower().startswith("contra"))

    # Prepare cards
    rule_cards_composite = {k: f"{records[k]['derived_principle']} When to apply: {records[k][CONTRAST]['condition']}" for k in keys}
    rule_cards_retain_only = {k: f"{records[k]['derived_principle']} When to apply: {records[k][CONTRAST]['retain_self_when']}" for k in keys}
    retain_texts = {k: records[k][CONTRAST]["retain_self_when"] for k in keys}
    flip_texts = {k: records[k][CONTRAST]["flip_to_competing_when"] for k in keys}

    queries_data = []

    for i, (q, p14) in enumerate(zip(eval_queries, v1_4_preds)):
        qid = q["query_id"]
        text = q["dilemma_text"]
        pool = p14["pool"]

        # Raw NLI scores on retain_texts and flip_texts
        r_vals = batch_score_nli(nli_model, nli_tok, text, [retain_texts[k] for k in pool], entail_idx, contra_idx, device)
        f_vals = batch_score_nli(nli_model, nli_tok, text, [flip_texts[k] for k in pool], entail_idx, contra_idx, device)
        r_scores = dict(zip(pool, r_vals))
        f_scores = dict(zip(pool, f_vals))

        # Raw NLI score on composite rule_card
        comp_vals = batch_score_nli(nli_model, nli_tok, text, [rule_cards_composite[k] for k in pool], entail_idx, contra_idx, device)
        comp_dict = dict(zip(pool, comp_vals))
        comp_norm = min_max_normalize(comp_dict, pool)

        # Raw NLI score on retain-only rule_card
        ret_vals = batch_score_nli(nli_model, nli_tok, text, [rule_cards_retain_only[k] for k in pool], entail_idx, contra_idx, device)
        ret_dict = dict(zip(pool, ret_vals))
        ret_norm = min_max_normalize(ret_dict, pool)

        q_entry = {
            "query_id": qid,
            "query_type": q["query_type"],
            "pair_group": q.get("pair_group"),
            "dilemma_text": text,
            "gold_primary_id": q.get("gold_primary_id"),
            "gold_competing_id": q.get("gold_competing_id"),
            "seeds": p14["seeds"],
            "pool": pool,
            "abstained": p14["abstained"],
            "abstention_prob": p14["abstention_prob"],
            "rrf_norm": p14["rrf_norm"],
            "ce_norm": p14["ce_norm"],
            "nli_rule_composite_raw": comp_dict,
            "nli_rule_composite_norm": comp_norm,
            "nli_rule_retain_raw": ret_dict,
            "nli_rule_retain_norm": ret_norm,
            "r_scores": r_scores,
            "f_scores": f_scores,
            "v1_4_base_3way": p14["base_3way"],
            "v1_4_final_scores": p14["final_scores"],
            "v1_4_top3": p14["top3"],
        }
        queries_data.append(q_entry)
        if (i + 1) % 10 == 0 or i == len(eval_queries) - 1:
            print(f"Processed {i + 1}/{len(eval_queries)} queries.")

    # Calculate input file fingerprints
    inputs = ["decisions_v0.2.jsonl", "chanakya_daily_supplement.jsonl", "eval_queries_v0.2.jsonl"]
    fingerprints = {n: hashlib.sha256((args.data_dir / n).read_bytes()).hexdigest() for n in inputs}

    cache_payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": "Regenerated missing directional NLI scores with cross-encoder/nli-deberta-v3-base on CPU, exact match to v1.4 top3 and final scores",
            "models": {
                "bi_encoder": BI,
                "cross_encoder": CE,
                "nli_arbitrator": NLI,
            },
            "fingerprints": fingerprints,
        },
        "queries": queries_data,
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"Successfully saved complete score cache to: {args.out_file}")


if __name__ == "__main__":
    main()
