# Dharm: Decision Corpus ML

A machine learning framework and benchmark suite for ethical dilemma arbitration, legal/moral retrieval, and epistemic abstention over ancient and classical jurisprudence corpora (**108 passages, 222 anchors, 840 contrastive pairs, 42 evaluation dilemmas**).

The chat app runs locally with `./scripts/run_local.sh` and is documented in `app/BETA.md`.

---

## 📊 Benchmark Progression

| Version | Model Architecture | Macro Recall@3 (All 38) | Recall@3 (Original 18) | Correct Flips (All 14) | Flips on Unseen Pairs (8) | Abstention Precision | Abstention Recall | Ordered Synthesis (10) |
|---|---|---|---|---|---|---|---|---|
| **TF-IDF Baseline** | BM25/TF-IDF + Heuristic Gate | 72.37% | 81.25% | 4/14 (28.6%) | 0/8 (0.0%) | 2/2 (100%) | 2/4 (50%) | 5/10 (50%) |
| **v1.0 (ML)** | Fine-tuned BGE Bi-Encoder + BCE Cross-Encoder | 51.32% | 56.25% | 2/14 (14.3%) | 1/8 (12.5%) | 4/19 (21.1%) | 4/4 (100%) | 1/10 (10%) |
| **v1.1 (Hybrid)** | Decoupled Classifier + RRF + Margin-tuned Reranker | 52.63% | 56.25% | 3/14 (21.4%) | 1/8 (12.5%) | **4/4 (100%)** | **4/4 (100%)** | 1/10 (10%) |
| **v1.2 (NLI)** | Stage-1 RRF + Zero-shot BGE-Reranker + DeBERTa-v3-small NLI | 55.26% | 59.38% | 4/14 (28.6%) | 2/8 (25.0%) | **4/4 (100%)** | **4/4 (100%)** | 2/10 (20%) |
| **v1.3 (Gated NLI)** | Multi-Vector MaxSim Dense + Top-7 Dual Graph + Gated NLI | 64.47% | 68.75% | **5/14 (35.7%)** | **3/8 (37.5%)** | **4/4 (100%)** | **4/4 (100%)** | 2/10 (20%) |
| **v1.4 (DeBERTa Base)** | Condition-Aware Multi-Vector + DeBERTa-v3-base Arbiter | 63.16% | **71.88%** | 4/14 (28.6%) | 2/8 (25.0%) | **4/4 (100%)** | **4/4 (100%)** | 3/10 (30%) |

### Key Achievements
- **Epistemic Abstention**: Decoupled TF-IDF + Logistic Regression classifier achieved perfect **100% Precision (4/4)** and **100% Recall (4/4)** on impossible/unanswerable empirical dilemmas, eliminating the 15 false abstentions from v1.0.
- **Unseen Directional Flips**: Advanced from **0/8 (0%)** on the TF-IDF baseline to **3/8 (37.5%)** in v1.3 on completely unseen dilemma pairs via Natural Language Inference (entailment vs. contradiction).
- **Stage-1 Pool Coverage**: Multi-vector passage-scenario indexing elevated candidate pool recall from **84.2%** to **94.7% (36 of 38 gold passages)**.

---

## 📂 Repository Structure

```
.
├── README.md                                # Comprehensive overview and benchmarks
├── README_v1.0.md                           # Detailed v1.0 specification and diagnostic notes
├── requirements.txt                         # Top-level dependencies
├── decisions_v0.2.csv                       # CSV copy kept for review; no runtime reader
├── train_retriever_and_arbitrator.py        # Top-level training script
│
│   The chat runtime reads only decision-corpus-ml/data. The root JSONL copies matched
│   those files by SHA-256 and were removed:
│   decisions_v0.2.jsonl 18d35df30acf268c76088f477e328688bd56220d18abb0cbf9b8de4ec260043f
│   chanakya_daily_supplement.jsonl ca7e5ebb9ddab447a8f874ade4da05a0d82db691d825ba50dcf9097af94232c8
│   eval_queries_v0.2.jsonl 6a284a3250bba9ad08c5aee9a0ce8ec0032290ab8982758726498a9ba6bd96e9
│
└── decision-corpus-ml/                      # Main ML development directory
    ├── requirements.txt                     # Pinned package requirements
    ├── environment.lock.txt                 # Exact environment lockfile
    ├── verify_artifacts.py                  # SHA-256 and reload verification tool
    │
    ├── train_retriever_and_arbitrator.py    # v1.0: Bi-encoder + Cross-encoder baseline
    ├── train_v1_1_hybrid_arbitrator.py      # v1.1: Decoupled abstention + Margin ranking
    ├── eval_v1_2_nli_arbitrator.py          # v1.2: RRF-blended NLI directional flipping
    ├── eval_v1_3_gated_nli.py               # v1.3: Multi-vector MaxSim retrieval + Gated NLI
    ├── eval_v1_4_deberta_base_arbiter.py    # v1.4: DeBERTa-v3-base NLI + Condition-aware alignment
    │
    ├── data/                                # Local benchmark data
    │   ├── decisions_v0.2.jsonl
    │   ├── chanakya_daily_supplement.jsonl
    │   └── eval_queries_v0.2.jsonl
    │
    ├── logs/                                # Complete run logs across all iterations
    │   ├── reference_run_cpu_bs4.log
    │   ├── train_v1_1.log
    │   └── ...
    │
    └── artifacts-*/                         # Evaluation reports, configs, and models
        ├── artifacts-cpu-bs4/               # v1.0 ML run reports & configs
        ├── artifacts-v1.1/                  # v1.1 reports, configs & abstention classifier
        ├── artifacts-v1.1-ablation-zeroshot/# v1.1 zero-shot ablation
        ├── artifacts-v1.2/                  # v1.2 evaluation report & metrics
        ├── artifacts-v1.3/                  # v1.3 evaluation report & metrics
        └── artifacts-v1.4/                  # v1.4 evaluation report & metrics
```

---

## 🚀 Setup & Reproduction

### 1. Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r decision-corpus-ml/requirements.txt
```

### 2. Run Verification
Verify all data hashes and artifact checksums:
```bash
python decision-corpus-ml/verify_artifacts.py
```

### 3. Run Evaluations
- **v1.1 Hybrid Arbitrator**:
  ```bash
  python decision-corpus-ml/train_v1_1_hybrid_arbitrator.py \
    --data_dir decision-corpus-ml/data \
    --output_dir decision-corpus-ml/artifacts-v1.1
  ```
- **v1.2 NLI Directional Flipping**:
  ```bash
  python decision-corpus-ml/eval_v1_2_nli_arbitrator.py
  ```
- **v1.3 Multi-Vector Gated NLI**:
  ```bash
  python decision-corpus-ml/eval_v1_3_gated_nli.py
  ```
- **v1.4 DeBERTa-v3-base Arbiter**:
  ```bash
  python decision-corpus-ml/eval_v1_4_deberta_base_arbiter.py
  ```

---

## 📜 Evaluation Criteria

- **Recall@3**: Fraction of gold passages appearing in top-3 candidates for answerable dilemmas.
- **Flip Accuracy**: Both variants of a contrast dilemma pair ($P_i$) must retrieve their respective gold passage at rank 1.
- **Epistemic Abstention**: Correctly refusing to answer empirical dilemmas lacking factual evidence in the corpus without producing false abstentions on answerable queries.
- **Ordered Synthesis**: Both supporting passages retrieved at ranks 1 and 2 in prescribed order for complex multi-principle dilemmas.
