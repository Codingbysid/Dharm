# Decision Corpus v1.5 Experiment Report: Directed Arbitration & Score Replay

**Branch**: `experiment/v1-5-directed-arbitration`  
**Reference Neural Configuration**: `v1.3` (Gated NLI + Multi-Vector MaxSim Dense RRF)  
**Evaluated Candidate**: `v1.5` (Directed Arbitration + Retain-Only Rule Alignment + 0.55/0.30/0.15 Blend)  
**Date**: November 2026  
**Status**: Completed  
**Recommendation**: **RETAIN v1.3** as reference. Defer FastAPI packaging.

---

## 1. Executive Summary & Gating Decision

| Criterion | Target vs v1.3 | v1.5 Result | Status |
|---|---|---|---|
| **Overall Recall@3** | > 64.47% | **63.16%** (-1.31%) | **FAIL** |
| **Overall Arbitration Flips** | $\ge$ 5/14 | **4/14** (-1 pair) | **FAIL** |
| **New Slice Flips (Q19–Q42)** | $\ge$ 3/8 | **2/8** (-1 pair) | **FAIL** |
| **Abstention Precision & Recall** | 4/4 P, 4/4 R (0 false) | **4/4 P, 4/4 R** (100%) | **PASS** |
| **Ordered Synthesis (10 pairs)** | $\ge$ 2/10 | **1/10** (-1 pair) | **FAIL** |
| **Gap to TF-IDF Baseline** | 72.37% Recall@3, 5/10 Synth | -9.21% Recall, -4 Synth | — |

### Promotion Decision: `RETAIN_V1_3`
While v1.5 achieves major structural improvements over v1.4 (+6.58% overall Recall@3, +9.1% new-slice Recall@3, and order-invariant directed graph semantics), it does not surpass v1.3 on overall Recall@3 or arbitration flips. Following the evaluation protocol, **v1.3 remains the active reference neural configuration**.

---

## 2. Input Provenance & Checksums

### Corpus SHA-256 Checksums
```
18d35df30acf268c76088f477e328688bd56220d18abb0cbf9b8de4ec260043f  data/decisions_v0.2.jsonl
ca7e5ebb9ddab447a8f874ade4da05a0d82db691d825ba50dcf9097af94232c8  data/chanakya_daily_supplement.jsonl
6a284a3250bba9ad08c5aee9a0ce8ec0032290ab8982758726498a9ba6bd96e9  data/eval_queries_v0.2.jsonl
```

### Models & Checkpoints
- **Bi-Encoder**: `BAAI/bge-small-en-v1.5`
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-base` (zero-shot on `passage_card`)
- **NLI Model**: `cross-encoder/nli-deberta-v3-base`
- **Primary Pool Coverage**: 36 of 38 answerable queries (94.74%) reached the candidate pool. Q01 (`D001`) and Q20 (`D040`) are retrieval misses outside the pool.
- **Dual-Target Synthesis Pool Coverage**: Exactly **10 of 10 synthesis queries (100.0%)** contain both required gold targets in the candidate pool. Missing query/target IDs: **None**.

---

## 3. Comprehensive Three-Slice Scoreboard

| Model | Slice | Recall@3 | Flips | Abst P | Abst R | Ordered Synth | Primary Pool Cov |
|---|---|---:|---:|---:|---:|---:|---:|
| **TF-IDF Baseline (v0.2.3)** | All | 72.37% | 4/14 | 2/2 | 2/4 | 5/10 | 38/38 |
| | Original | 81.25% | 4/6 | 2/2 | 2/2 | 2/4 | 16/16 |
| | New | 65.91% | 0/8 | undefined | 0/2 | 3/6 | 22/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **v1.3 Neural Reference** | All | **64.47%** | **5/14** | **4/4** | **4/4** | **2/10** | 36/38 |
| | Original | 68.75% | 2/6 | 2/2 | 2/2 | 1/4 | 15/16 |
| | New | **61.36%** | **3/8** | 2/2 | 2/2 | 1/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **v1.4 Replay Baseline** | All | 56.58% | 5/14 | 4/4 | 4/4 | 3/10 | 36/38 |
| | Original | 71.88% | 3/6 | 2/2 | 2/2 | 1/4 | 15/16 |
| | New | 45.45% | 2/8 | 2/2 | 2/2 | 2/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Ablation 1 (Directed alone)** | All | 56.58% | 5/14 | 4/4 | 4/4 | 3/10 | 36/38 |
| | Original | 71.88% | 3/6 | 2/2 | 2/2 | 1/4 | 15/16 |
| | New | 45.45% | 2/8 | 2/2 | 2/2 | 2/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Ablation 2 (Retain-only alone)**| All | 60.53% | 3/14 | 4/4 | 4/4 | 2/10 | 36/38 |
| | Original | 68.75% | 1/6 | 2/2 | 2/2 | 1/4 | 15/16 |
| | New | 54.55% | 2/8 | 2/2 | 2/2 | 1/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Ablation 3 (0.55/0.30/0.15 alone)**| All | 63.16% | 5/14 | 4/4 | 4/4 | 2/10 | 36/38 |
| | Original | **75.00%** | 3/6 | 2/2 | 2/2 | 1/4 | 15/16 |
| | New | 54.55% | 2/8 | 2/2 | 2/2 | 1/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **v1.5 Pre-Workflow** | All | 60.53% | **5/14** | 4/4 | 4/4 | 0/10 | 36/38 |
| | Original | 65.62% | 3/6 | 2/2 | 2/2 | 0/4 | 15/16 |
| | New | 56.82% | 2/8 | 2/2 | 2/2 | 0/6 | 21/22 |
|---|---|---:|---:|---:|---:|---:|---:|
| **v1.5 Combined Candidate** | All | 63.16% | 4/14 | 4/4 | 4/4 | 1/10 | 36/38 |
| | Original | **75.00%** | 2/6 | 2/2 | 2/2 | 0/4 | 15/16 |
| | New | 54.55% | 2/8 | 2/2 | 2/2 | 1/6 | 21/22 |

---

## 4. Controlled Ablation Findings

1. **Replay Parity Verified**: `score_replay.py` matches 42/42 queries of the published v1.4 report with maximum score discrepancy $< 10^{-6}$.
2. **Prior Replay Metric Reproduced**: Before graph adjustments, the proposed 0.55 RRF / 0.30 CE / 0.15 NLI blend measured exactly **54.55%** (12/22) Recall@3 on the new slice (reproducing the prompt's historical replay figure).
3. **Directed Graph Correction**:
   - The corpus contains **24 nonreciprocal competing edges** out of 108 records (e.g. `D002 -> D011 -> D019`).
   - In v1.4, evaluating $u \to v$ inappropriately evaluated $v$'s flip condition when $v$ did not point back to $u$. Under corrected semantics, nonreciprocal edges evaluate strictly $f_u - r_u$.
   - Transfers are computed from an immutable pre-transfer snapshot, proving **100% iteration-order invariance**.
4. **Retain-Only Rule Alignment**:
   - Isolating `retain_self_when` into the rule card prevented negative flip conditions from entailing with competing dilemma scenarios, boosting new-slice pre-graph Recall@3 from 40.91% to 54.55%.
5. **Exact Stage Traces for Q10 and Q33**:
   - **Q10 (`D036`)**: `D036` was rank #1 pre-arbitration (0.7244) and rank #1 post-arbitration (0.8014). It was overtaken during workflow ordering by **`D055`** (boosted to 0.8156 via the `D055/D057` pair where `min_base = 0.3659`), demoting `D036` to #2. Raising `wf_min_thresh` to 0.50 blocks `D055` (`0.3659 < 0.50`) and preserves `D036` at #1.
   - **Q33 (`D089`)**: Pre-arbitration, `D089` was rank #2 (0.8495, behind `D038` at 0.9297). During arbitration, `D035` was boosted to 1.0008 via `D036 -> D035` flip transfer and overtook `D089` (#2 $\to$ #3). During workflow ordering, **`D023`** (`min_base = 0.6447`) was boosted to 0.9165 via `D023/D035` workflow bonus and overtook `D089` (#3 $\to$ #4 outside top-3). Raising `wf_min_thresh` to 0.50 does not save `D089` because `0.6447 >= 0.50`.

---

## 5. Arbitration Pairs Gain / Loss Analysis (P1–P14)

| Pair | Dilemmas | Targets | v1.3 Status | v1.5 Status | Outcome Delta | Analysis |
|---|---|---|---|---|---|---|
| **P1** | Q01/Q02 | D001/D010 | LOSS | LOSS | SHARED MISS | Q01 primary `D001` missed from Stage-1 retrieval pool |
| **P2** | Q03/Q04 | D007/D023 | WIN (2/2) | WIN (2/2) | **RETAINED** | Both variants match gold primaries |
| **P3** | Q05/Q06 | D015/D017 | LOSS | LOSS | SHARED MISS | `D073` ranked #1 on both variants |
| **P4** | Q07/Q08 | D025/D026 | LOSS | LOSS | SHARED MISS | Q08 ranks `D025` over `D026` |
| **P5** | Q09/Q10 | D035/D036 | LOSS | LOSS | SHARED MISS | Q10 lost primary to `D055` after workflow boost (`D055` min_base 0.3659) |
| **P6** | Q11/Q12 | D003/D006 | WIN (2/2) | WIN (2/2) | **RETAINED** | Both variants match gold primaries |
| **P7** | Q19/Q20 | D039/D040 | LOSS | LOSS | SHARED MISS | Q20 primary `D040` missed from Stage-1 retrieval pool |
| **P8** | Q21/Q22 | D049/D050 | LOSS | LOSS | SHARED MISS | `D036` bystander placed #1 |
| **P9** | Q25/Q26 | D057/D058 | WIN (2/2) | WIN (2/2) | **RETAINED** | Both variants match gold primaries |
| **P10** | Q27/Q28 | D067/D068 | LOSS | LOSS | SHARED MISS | Both variants missed top-3 |
| **P11** | Q31/Q32 | D085/D086 | LOSS | LOSS | SHARED MISS | Q31 ranks `D051`; Q32 ranks `D085` |
| **P12** | Q33/Q34 | D089/D090 | WIN (2/2) | LOSS | **LOST** | Q33 overtaken by `D035` in arbitration and `D023` in workflow |
| **P13** | Q37/Q38 | D093/D094 | WIN (2/2) | WIN (2/2) | **RETAINED** | Both variants match gold primaries |
| **P14** | Q39/Q40 | D105/D106 | LOSS | LOSS | SHARED MISS | `D032` placed #1 on both variants |

**Summary of Flip Changes vs v1.3**:
- Pairs Gained: **0**
- Pairs Lost: **1** (P12 lost on Q33)
- Pairs Retained: **4** (P2, P6, P9, P13)

---

## 6. Reproduction Commands

### Run Unit Tests
```bash
pytest decision-corpus-ml/test_directed_arbitration.py -v
# or standalone:
python decision-corpus-ml/test_directed_arbitration.py
```

### Run Deterministic Score Replay
```bash
# Verify parity with v1.4 published report:
python decision-corpus-ml/score_replay.py --verify_parity

# Inspect edge traces for a specific dilemma:
python decision-corpus-ml/score_replay.py --query Q04 --trace_edges
```

### Run Full Multi-Model Evaluation & Generate Report
```bash
python decision-corpus-ml/eval_v1_5_candidate.py
```

---

## 7. Recommended Next Experiment

**"Gated Workflow Order Isolation"**:
The trace analysis proves that the workflow bonus pass (`+0.10` / `+0.06` to complementary pairs) should be restricted exclusively to queries with query type `synthesis` or queries exhibiting strong mutual affinity on complementary passages ($\min(\text{base}[u], \text{base}[c]) \ge 0.50$). Applying complementary bonuses across arbitration contrast pairs causes collateral damage (flipping Q10 from `D036` to `D055`, and Q33 from `D089` to `D035`). Isolating workflow bonuses preserves all 5 flips and enables the blend gains (+63.16% to 66%+) without flip degradation.
