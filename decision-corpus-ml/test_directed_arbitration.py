"""Focused test suite for directed arbitration graph semantics.

Verifies:
1. Corpus nonreciprocal edge count (exactly 24).
2. Reciprocal pair bidirectional evidence combination.
3. Nonreciprocal three-node cycle (evidence on B->C never stands in for B->A).
4. Candidate crossing the flip gate (threshold and delta gating).
5. Iteration-order invariance (immutable pre-transfer snapshot).
6. Workflow ordering tested separately.
"""
import itertools
import json
from pathlib import Path
import random

# score_replay import
try:
    from score_replay import (
        load_corpus_graph,
        deterministic_rank,
        replay_query,
        CONTRAST,
        WORKFLOW,
    )
except ImportError:
    from decision_corpus_ml.score_replay import (
        load_corpus_graph,
        deterministic_rank,
        replay_query,
        CONTRAST,
        WORKFLOW,
    )

try:
    import pytest
except ImportError:
    class PytestMock:
        @staticmethod
        def fixture(fn):
            return fn

        @staticmethod
        def approx(expected, abs_tol=1e-6, abs=None):
            tolerance = abs if abs is not None else abs_tol
            class ApproxVal:
                def __eq__(self, actual):
                    diff = actual - expected
                    if diff < 0:
                        diff = -diff
                    return diff <= tolerance
            return ApproxVal()
    pytest = PytestMock()


@pytest.fixture
def corpus_records():
    data_dir = Path("decision-corpus-ml/data")
    if not data_dir.exists():
        data_dir = Path("data")
    return load_corpus_graph(data_dir)


def test_corpus_nonreciprocal_edge_count(corpus_records):
    """The audit found 24 nonreciprocal competing edges in decisions_v0.2.jsonl."""
    nonreciprocal = []
    reciprocal = []
    for u, r in corpus_records.items():
        v = r[CONTRAST]["id"]
        v_comp = corpus_records[v][CONTRAST]["id"]
        if v_comp != u:
            nonreciprocal.append((u, v, v_comp))
        elif u < v:
            reciprocal.append((u, v))

    assert len(corpus_records) == 108
    assert len(reciprocal) == 42  # 42 pairs = 84 reciprocal edges
    assert len(nonreciprocal) == 24, f"Expected 24 nonreciprocal edges, got {len(nonreciprocal)}"


def test_reciprocal_pairs():
    """Verify reciprocal pair evaluates both directions and combines evidence symmetrically."""
    records = {
        "A": {
            CONTRAST: {"id": "B"},
            WORKFLOW: {"id": "B", "sequence_role": "Follow with B"},
        },
        "B": {
            CONTRAST: {"id": "A"},
            WORKFLOW: {"id": "A", "sequence_role": "Precede with A"},
        },
    }

    # Case 1: Evidence favors flipping from A to B
    # r_A = -0.8, f_A = +0.8 (A wants to flip to B)
    # r_B = +0.8, f_B = -0.8 (B wants to retain B, not flip to A)
    q_entry = {
        "query_id": "TEST_Q1",
        "query_type": "arbitration_pair",
        "pool": ["A", "B"],
        "seeds": ["A", "B"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"A": 0.80, "B": 0.40},
        "ce_norm": {"A": 0.80, "B": 0.40},
        "nli_rule_retain_norm": {"A": 0.80, "B": 0.40},
        "r_scores": {"A": -0.80, "B": 0.80},
        "f_scores": {"A": 0.80, "B": -0.80},
    }

    # Under 0.55/0.30/0.15 blend: base[A] = 0.80, base[B] = 0.40
    # topical_max = 0.80 >= 0.68
    # net_A_to_B = 0.5 * ((0.8 - (-0.8)) + (0.8 - (-0.8))) = 0.5 * (1.6 + 1.6) = 1.60
    # B receives boost: max(0.40, 0.80 + 0.18 * 1.60) = 0.80 + 0.288 = 1.088
    # A receives demotion: min(0.80, 0.80 - 0.06 * 1.60) = 0.80 - 0.096 = 0.704
    res = replay_query(
        q_entry,
        records,
        w_rrf=0.55,
        w_ce=0.30,
        w_nli=0.15,
        rule_mode="retain_only",
        edge_mode="directed_v1_5",
        workflow_active=False,
    )

    assert res["top3"] == ["B", "A"]
    assert res["primary"] == "B"
    assert res["post_flip_scores"]["B"] > res["post_flip_scores"]["A"]

    # Verify edge traces recorded symmetry
    traces = {t["source_id"]: t for t in res["edge_traces"]}
    assert traces["A"]["is_reciprocal"] is True
    assert traces["B"]["is_reciprocal"] is True
    assert abs(traces["A"]["net_delta"] - 1.60) < 1e-5
    assert abs(traces["B"]["net_delta"] - (-1.60)) < 1e-5
    assert traces["A"]["delta_applied"] is True
    assert traces["B"]["delta_applied"] is False  # net_delta < 0, no transfer from B to A


def test_nonreciprocal_three_node_cycle():
    """Verify 3-node cycle A -> B -> C -> A where evidence on B->C never stands in for B->A."""
    records = {
        "A": {CONTRAST: {"id": "B"}, WORKFLOW: {"id": "B", "sequence_role": "Follow with B"}},
        "B": {CONTRAST: {"id": "C"}, WORKFLOW: {"id": "C", "sequence_role": "Follow with C"}},
        "C": {CONTRAST: {"id": "A"}, WORKFLOW: {"id": "A", "sequence_role": "Follow with A"}},
    }

    # Suppose A has strong flip evidence toward B: f_A = +0.9, r_A = -0.5 (net = 1.4)
    # B has strong flip evidence toward C: f_B = +0.95, r_B = -0.6
    # In legacy logic, evaluating edge (A, B) mistakenly pulled f_B (which belongs to B->C!)
    # Under directed semantics, evaluating A->B uses ONLY A's conditions because B does not point to A!
    q_entry = {
        "query_id": "TEST_CYCLE",
        "query_type": "arbitration_pair",
        "pool": ["A", "B", "C"],
        "seeds": ["A", "B", "C"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"A": 0.75, "B": 0.50, "C": 0.20},
        "ce_norm": {"A": 0.75, "B": 0.50, "C": 0.20},
        "nli_rule_retain_norm": {"A": 0.75, "B": 0.50, "C": 0.20},
        "r_scores": {"A": -0.50, "B": -0.60, "C": 0.00},
        "f_scores": {"A": 0.90, "B": 0.95, "C": 0.00},
    }

    res = replay_query(
        q_entry,
        records,
        w_rrf=0.55,
        w_ce=0.30,
        w_nli=0.15,
        rule_mode="retain_only",
        edge_mode="directed_v1_5",
        workflow_active=False,
    )

    traces = {f"{t['source_id']}->{t['target_id']}": t for t in res["edge_traces"]}
    # Edge A->B: nonreciprocal, net_delta = f_A - r_A = 0.90 - (-0.50) = 1.40
    # Must NOT depend on f_B (+0.95)
    assert traces["A->B"]["is_reciprocal"] is False
    assert abs(traces["A->B"]["net_delta"] - 1.40) < 1e-5
    assert traces["A->B"]["raw_r_target"] is None
    assert traces["A->B"]["raw_f_target"] is None

    # Edge B->C: nonreciprocal, net_delta = f_B - r_B = 0.95 - (-0.60) = 1.55
    # Topical max = max(base[B], base[C]) = max(0.50, 0.20) = 0.50 < 0.68 -> GATE FAILS
    assert traces["B->C"]["topical_max"] == 0.50
    assert traces["B->C"]["gate_passed"] is False
    assert traces["B->C"]["delta_applied"] is False


def test_candidate_crossing_flip_gate():
    """Verify gating conditions: topical_max threshold and net_delta threshold."""
    records = {
        "U": {CONTRAST: {"id": "V"}, WORKFLOW: {"id": "V", "sequence_role": "Follow with V"}},
        "V": {CONTRAST: {"id": "U"}, WORKFLOW: {"id": "U", "sequence_role": "Precede with U"}},
    }

    # Case A: Below topical gate (0.65 < 0.68)
    q_subgate = {
        "query_id": "TEST_GATE_SUB",
        "query_type": "arbitration_pair",
        "pool": ["U", "V"],
        "seeds": ["U", "V"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"U": 0.65, "V": 0.30},
        "ce_norm": {"U": 0.65, "V": 0.30},
        "nli_rule_retain_norm": {"U": 0.65, "V": 0.30},
        "r_scores": {"U": -0.5, "V": 0.5},
        "f_scores": {"U": 0.5, "V": -0.5},
    }
    res_sub = replay_query(
        q_subgate, records, flip_gate=0.68, flip_delta=0.04, workflow_active=False
    )
    assert res_sub["edge_traces"][0]["gate_passed"] is False
    assert res_sub["post_flip_scores"]["U"] == pytest.approx(0.65)
    assert res_sub["post_flip_scores"]["V"] == pytest.approx(0.30)
    assert res_sub["top3"] == ["U", "V"]

    # Case B: Exceeds topical gate and delta gate (0.75 >= 0.68, net=1.0 > 0.04) -> FLIPS
    q_above = {
        "query_id": "TEST_GATE_ABOVE",
        "query_type": "arbitration_pair",
        "pool": ["U", "V"],
        "seeds": ["U", "V"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"U": 0.75, "V": 0.30},
        "ce_norm": {"U": 0.75, "V": 0.30},
        "nli_rule_retain_norm": {"U": 0.75, "V": 0.30},
        "r_scores": {"U": -0.5, "V": 0.5},
        "f_scores": {"U": 0.5, "V": -0.5},
    }
    res_above = replay_query(
        q_above, records, flip_gate=0.68, flip_delta=0.04, workflow_active=False
    )
    assert res_above["edge_traces"][0]["gate_passed"] is True
    assert res_above["post_flip_scores"]["V"] > res_above["post_flip_scores"]["U"]
    assert res_above["top3"] == ["V", "U"]


def test_iteration_order_invariance():
    """Verify that candidate permutation does not alter final ranking or scores."""
    records = {
        "A": {CONTRAST: {"id": "B"}, WORKFLOW: {"id": "B", "sequence_role": "Follow with B"}},
        "B": {CONTRAST: {"id": "C"}, WORKFLOW: {"id": "C", "sequence_role": "Follow with C"}},
        "C": {CONTRAST: {"id": "D"}, WORKFLOW: {"id": "D", "sequence_role": "Follow with D"}},
        "D": {CONTRAST: {"id": "A"}, WORKFLOW: {"id": "A", "sequence_role": "Follow with A"}},
    }

    base_q = {
        "query_id": "TEST_PERM",
        "query_type": "arbitration_pair",
        "seeds": ["A", "B", "C", "D"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"A": 0.80, "B": 0.70, "C": 0.60, "D": 0.50},
        "ce_norm": {"A": 0.80, "B": 0.70, "C": 0.60, "D": 0.50},
        "nli_rule_retain_norm": {"A": 0.80, "B": 0.70, "C": 0.60, "D": 0.50},
        "r_scores": {"A": -0.2, "B": -0.3, "C": 0.1, "D": 0.2},
        "f_scores": {"A": 0.6, "B": 0.4, "C": -0.5, "D": -0.2},
    }

    reference_res = None
    # Test all 24 permutations of pool iteration order
    for perm in itertools.permutations(["A", "B", "C", "D"]):
        q_perm = dict(base_q, pool=list(perm))
        res = replay_query(
            q_perm,
            records,
            rule_mode="retain_only",
            edge_mode="directed_v1_5",
            workflow_active=False,
        )
        if reference_res is None:
            reference_res = res
        else:
            assert res["top3"] == reference_res["top3"], f"Order changed under permutation {perm}"
            for k in ["A", "B", "C", "D"]:
                assert res["post_flip_scores"][k] == pytest.approx(
                    reference_res["post_flip_scores"][k], abs=1e-9
                )


def test_workflow_ordering_separated():
    """Verify workflow pass modifies scores separately without mutating the flip graph."""
    records = {
        "X": {CONTRAST: {"id": "Z"}, WORKFLOW: {"id": "Y", "sequence_role": "Follow with Y"}},
        "Y": {CONTRAST: {"id": "Z"}, WORKFLOW: {"id": "X", "sequence_role": "Precede with X"}},
        "Z": {CONTRAST: {"id": "X"}, WORKFLOW: {"id": "X", "sequence_role": "Follow with X"}},
    }

    q = {
        "query_id": "TEST_WF",
        "query_type": "synthesis",
        "gold_primary_id": "X",
        "gold_competing_id": "Y",
        "pool": ["X", "Y", "Z"],
        "seeds": ["X", "Y", "Z"],
        "abstained": False,
        "abstention_prob": 0.05,
        "rrf_norm": {"X": 0.70, "Y": 0.40, "Z": 0.85},
        "ce_norm": {"X": 0.70, "Y": 0.40, "Z": 0.85},
        "nli_rule_retain_norm": {"X": 0.70, "Y": 0.40, "Z": 0.85},
        "r_scores": {"X": 0.0, "Y": 0.0, "Z": 0.0},
        "f_scores": {"X": 0.0, "Y": 0.0, "Z": 0.0},
    }

    # Before workflow: Z has highest score (0.85), X is 0.70, Y is 0.40
    res_no_wf = replay_query(q, records, workflow_active=False)
    assert res_no_wf["top3"] == ["Z", "X", "Y"]

    # After workflow: X and Y are complementary pair meeting max(0.70, 0.40)=0.70 >= 0.68
    # and min=0.40 >= 0.35. Role: Follow with Y (first=X, second=Y).
    # X gets boosted by +0.10 -> 0.70 + 0.10 = 0.80
    # Y gets boosted by +0.06 -> 0.70 + 0.06 = 0.76
    # Note post_flip_scores in res_with_wf remain identical to res_no_wf!
    res_with_wf = replay_query(q, records, workflow_active=True)
    assert res_with_wf["post_flip_scores"] == res_no_wf["post_flip_scores"]
    assert res_with_wf["workflow_traces"][0]["gate_passed"] is True
    assert res_with_wf["workflow_traces"][0]["first"] == "X"
    assert res_with_wf["workflow_traces"][0]["second"] == "Y"


if __name__ == "__main__":
    records = corpus_records()
    print("Running test_corpus_nonreciprocal_edge_count...")
    test_corpus_nonreciprocal_edge_count(records)
    print("Running test_reciprocal_pairs...")
    test_reciprocal_pairs()
    print("Running test_nonreciprocal_three_node_cycle...")
    test_nonreciprocal_three_node_cycle()
    print("Running test_candidate_crossing_flip_gate...")
    test_candidate_crossing_flip_gate()
    print("Running test_iteration_order_invariance...")
    test_iteration_order_invariance()
    print("Running test_workflow_ordering_separated...")
    test_workflow_ordering_separated()
    print("\n✓ ALL 6 DIRECTED ARBITRATION TESTS PASSED.")
