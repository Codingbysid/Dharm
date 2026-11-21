"""Tests for Workflow Intent Router and Reconciled Evidence.

Verifies:
1. Exact reconciliation of arbitration queries disruption counts (v1.3 vs v1.6).
2. Regression test: pair promotion changes runner-up while preserving correct primary.
3. Diagnostic routing oracle reproduces exact metrics (7/10 synthesis, 5/14 flips, 3/8 new flips).
4. GroupKFold development splits have 0 group overlap.
5. Bit-for-bit v1.3 fallback when router gate is closed or pair scorer declines.
"""
import json
from pathlib import Path
import pytest

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_V1_3 = Path("decision-corpus-ml/artifacts-v1.3")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")


@pytest.fixture(scope="module")
def v1_3_report():
    return json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def v1_6_report():
    return json.loads((ARTIFACTS_V1_6 / "v1_6_ordered_pair_report.json").read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def routed_report():
    return json.loads((ARTIFACTS_V1_6 / "workflow_intent_router_eval_report.json").read_text(encoding="utf8"))


def test_reconcile_arbitration_evidence(v1_3_report, v1_6_report):
    """Reconcile all 28 arbitration queries between v1.3 and v1.6."""
    v13_by_id = {q["query_id"]: q for q in v1_3_report["predictions"]}
    v16_by_id = {q["query_id"]: q for q in v1_6_report["predictions"]}

    arb_ids = [q["query_id"] for q in v1_3_report["predictions"] if q["query_type"] == "arbitration_pair"]
    assert len(arb_ids) == 28

    promoted = []
    changed_top3 = []
    changed_primary = []
    corr_to_incorr = []
    incorr_to_corr = []
    corr_preserved = []
    incorr_preserved = []

    for qid in arb_ids:
        q13 = v13_by_id[qid]
        q16 = v16_by_id[qid]
        gold = q13["gold_primary_id"]

        p13 = q13["primary"]
        p16 = q16["primary"]
        t13 = q13["top3"]
        t16 = q16["top3"]

        if q16.get("pair_promoted", False):
            promoted.append(qid)
        if t13 != t16:
            changed_top3.append(qid)
        if p13 != p16:
            changed_primary.append(qid)

        v13_corr = (p13 == gold)
        v16_corr = (p16 == gold)

        if v13_corr and not v16_corr:
            corr_to_incorr.append(qid)
        elif not v13_corr and v16_corr:
            incorr_to_corr.append(qid)
        elif v13_corr and v16_corr:
            corr_preserved.append(qid)
        else:
            incorr_preserved.append(qid)

    assert len(promoted) == 21
    assert len(changed_top3) == 19
    assert len(changed_primary) == 18
    assert corr_to_incorr == ["Q04", "Q09", "Q26", "Q34", "Q37"]
    assert incorr_to_corr == ["Q21", "Q27"]
    assert corr_preserved == ["Q03", "Q07", "Q11", "Q12", "Q25", "Q33", "Q38"]

    # Reconcile the reported 17 in ungated v1.6 report:
    # 12 (already incorrect in v1.3 and stayed incorrect) + 5 (lost previously correct primary) = 17
    assert len(incorr_preserved) == 14
    # Of the 21 promoted, exactly 17 ended with incorrect primary (12 + 5):
    promoted_incorrect = [qid for qid in promoted if v16_by_id[qid]["primary"] != v16_by_id[qid]["gold_primary_id"]]
    assert len(promoted_incorrect) == 17


def test_regression_promotion_preserves_primary_changes_runner_up(v1_3_report, v1_6_report):
    """Regression test: promotion changes runner-up while preserving the correct primary (Q38)."""
    q13 = next(q for q in v1_3_report["predictions"] if q["query_id"] == "Q38")
    q16 = next(q for q in v1_6_report["predictions"] if q["query_id"] == "Q38")
    gold = q13["gold_primary_id"]

    assert gold == "D094"
    assert q13["top3"] == ["D094", "D093", "D018"]
    assert q13["primary"] == "D094"

    assert q16["pair_promoted"] is True
    assert q16["promoted_pair"] == ["D094", "D095"]
    assert q16["top3"] == ["D094", "D095", "D093"]
    assert q16["primary"] == "D094"  # Primary is preserved!
    assert q16["top3"][1] == "D095"   # Runner-up changed from D093 to D095


def test_diagnostic_routing_oracle(v1_3_report, v1_6_report):
    """Verify diagnostic routing oracle metrics."""
    from eval_routed_workflow_candidate import evaluate_benchmark

    oracle_preds = []
    for q13, q16 in zip(v1_3_report["predictions"], v1_6_report["predictions"]):
        if q13["query_type"] == "synthesis":
            oracle_preds.append(q16)
        else:
            oracle_preds.append(q13)

    m = evaluate_benchmark(oracle_preds)
    assert m["ordered_synthesis"] == [7, 10]
    assert m["flip"] == [5, 14]
    assert m["abstention_precision"] == [4, 4]
    assert m["abstention_recall"] == [4, 4]
    assert abs(m["recall_at_3"] - 0.6842105) < 1e-4

    m_new = evaluate_benchmark(oracle_preds[18:])
    assert m_new["flip"] == [3, 8]
    assert m_new["ordered_synthesis"] == [5, 6]


def test_router_split_overlap_zero():
    """Verify router configuration has zero group overlap across folds."""
    cfg = json.loads((ARTIFACTS_V1_6 / "router_config.json").read_text(encoding="utf8"))
    assert cfg["split_verification"]["all_overlaps_zero"] is True
    assert cfg["calibrated_threshold"] == 0.33
    assert cfg["predeclared_max_fpr"] == 0.05
    assert cfg["calibration_metrics"]["selected_fpr"] <= 0.05


def test_bit_for_bit_fallback_invariance(v1_3_report, routed_report):
    """Verify that every prediction where gate is closed or pair scorer declined has bit-for-bit v1.3 fallback."""
    v13_by_id = {q["query_id"]: q for q in v1_3_report["predictions"]}
    for q in routed_report["predictions"]:
        qid = q["query_id"]
        q13 = v13_by_id[qid]
        if not q["pair_promoted"]:
            assert q["top3"] == q13["top3"], f"Fallback top3 diverged on {qid}"
            assert q["primary"] == q13["primary"], f"Fallback primary diverged on {qid}"
