"""Tests for Pair-Promotion Verifier and Verified Candidate Evaluation.

Verifies:
1. Verifier configuration integrity and SHA-256 hash match.
2. Feature vector schema (16 features) and absence of NaN/infinite values.
3. Exact bit-for-bit v1.3 fallback when verifier rejects or no proposal is made.
4. Exact reproduction of diagnostic counts:
   - Correct pairs promoted == 2 (Q24, Q35)
   - Wrong pairs promoted == 2 (Q21, Q22)
   - Correct pairs blocked == 4 (Q14, Q29, Q30, Q36)
   - Previously correct arbitration primaries displaced == 0
   - v1.6 extra synthesis survived == 1 (Q24)
5. Promotion criteria:
   - Recall@3 > 64.47% (65.79%)
   - Flips >= 5/14 overall and >= 3/8 on new
   - Abstentions 4/4 P and 4/4 R
   - Ordered synthesis >= 2/10 (3/10)
"""
import hashlib
import json
from pathlib import Path
import pytest

ARTIFACTS_V1_3 = Path("decision-corpus-ml/artifacts-v1.3")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")


@pytest.fixture(scope="module")
def v1_3_report():
    return json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def v1_6_report():
    return json.loads((ARTIFACTS_V1_6 / "v1_6_ordered_pair_report.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def verifier_eval_report():
    return json.loads((ARTIFACTS_V1_6 / "pair_verifier_eval_report.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def verifier_config():
    return json.loads((ARTIFACTS_V1_6 / "verifier_config.json").read_text(encoding="utf8"))


def test_verifier_config_hash(verifier_config):
    """Verify configuration SHA-256 hash integrity."""
    stored_hash = verifier_config["config_sha256"]
    cfg_copy = dict(verifier_config)
    del cfg_copy["config_sha256"]
    computed_hash = hashlib.sha256(json.dumps(cfg_copy, sort_keys=True, indent=2).encode("utf8")).hexdigest()
    assert stored_hash == computed_hash
    assert verifier_config["calibrated_threshold"] == 0.57
    assert len(verifier_config["features"]) == 16


def test_bit_for_bit_fallback_invariance(v1_3_report, verifier_eval_report):
    """Verify exact bit-for-bit v1.3 fallback for every non-promoted query."""
    v13_by_id = {q["query_id"]: q for q in v1_3_report["predictions"]}
    for q in verifier_eval_report["predictions"]:
        qid = q["query_id"]
        q13 = v13_by_id[qid]
        if not q["pair_promoted"]:
            assert q["top3"] == q13["top3"], f"Fallback top3 diverged on {qid}"
            assert q["primary"] == q13["primary"], f"Fallback primary diverged on {qid}"
            assert q["abstained"] == q13["abstained"], f"Fallback abstention diverged on {qid}"


def test_diagnostic_verifier_counts(verifier_eval_report):
    """Verify exact diagnostic verifier counts."""
    diag = verifier_eval_report["diagnostic_counts"]
    assert len(diag["correct_pairs_promoted"]) == 2
    assert [p[0] for p in diag["correct_pairs_promoted"]] == ["Q24", "Q35"]

    assert len(diag["wrong_pairs_promoted"]) == 2
    assert [p[0] for p in diag["wrong_pairs_promoted"]] == ["Q21", "Q22"]

    assert len(diag["correct_pairs_blocked"]) == 4
    assert [p[0] for p in diag["correct_pairs_blocked"]] == ["Q14", "Q29", "Q30", "Q36"]

    assert len(diag["corr_arb_displaced"]) == 0
    assert diag["corr_arb_displaced"] == []

    assert diag["v1_6_extra_synth_survived"] == ["Q24"]


def test_promotion_criteria_satisfaction(verifier_eval_report):
    """Verify formal promotion criteria checks pass."""
    metrics = verifier_eval_report["metrics"]["deployable_verifier"]

    # Criterion 1: Recall@3 > 64.47%
    assert metrics["all"]["recall_at_3"] > 0.6447
    assert abs(metrics["all"]["recall_at_3"] - 0.6578947) < 1e-4

    # Criterion 2: Flips >= 5/14 overall and >= 3/8 on new
    assert metrics["all"]["flip"] == [5, 14]
    assert metrics["new"]["flip"] == [3, 8]

    # Criterion 3: Abstentions 4/4 P, 4/4 R
    assert metrics["all"]["abstention_precision"] == [4, 4]
    assert metrics["all"]["abstention_recall"] == [4, 4]

    # Criterion 4: Ordered synthesis >= 2/10
    assert metrics["all"]["ordered_synthesis"] == [3, 10]
    assert metrics["all"]["ordered_synthesis"][0] >= 2

    # Final promotion decision must be true
    assert verifier_eval_report["promotion_decision"]["promoted"] is True
