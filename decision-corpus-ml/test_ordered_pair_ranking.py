"""Unit and regression tests for Ordered Pair Ranking (v1.6 candidate).

Verifies:
1. Orientation extraction and semantic role interpretation (Follow with vs Precede with).
2. Reciprocal deduplication (each edge is scored once per query).
3. Diagnostic ceilings on synthesis queries (both-in-pool, linked, oriented).
4. Fallback parity: when pair promotion is disabled, output is 100% identical to v1.3.
5. Invariance of corpus records and benchmark metadata.
"""
import json
from pathlib import Path
import pytest

DATA_DIR = Path("decision-corpus-ml/data")
ARTIFACTS_V1_3 = Path("decision-corpus-ml/artifacts-v1.3")
ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")
WORKFLOW = "complementary_passage_id"


@pytest.fixture(scope="module")
def corpus_records():
    rows = [json.loads(line) for line in (DATA_DIR / "decisions_v0.2.jsonl").read_text(encoding="utf8").splitlines()]
    return {r["id"]: r for r in rows}


@pytest.fixture(scope="module")
def v1_3_report():
    return json.loads((ARTIFACTS_V1_3 / "v1_3_ml_eval_report.json").read_text(encoding="utf8"))


def test_orientation_extraction(corpus_records):
    """Verify that every complementary edge has reciprocal roles and valid orientation."""
    follow_count = 0
    precede_count = 0
    for rid, r in corpus_records.items():
        wf = r[WORKFLOW]
        target_id = wf["id"]
        role = wf["sequence_role"]
        peer_wf = corpus_records[target_id][WORKFLOW]
        peer_role = peer_wf["sequence_role"]

        assert peer_wf["id"] == rid, f"Non-reciprocal workflow edge between {rid} and {target_id}"

        if role.startswith("Follow with"):
            follow_count += 1
            assert role.startswith(f"Follow with {target_id}."), f"Unexpected syntax: {role}"
            assert peer_role.startswith(f"Precede with {rid}."), f"Inconsistent peer role: {peer_role}"
        elif role.startswith("Precede with"):
            precede_count += 1
            assert role.startswith(f"Precede with {target_id}."), f"Unexpected syntax: {role}"
            assert peer_role.startswith(f"Follow with {rid}."), f"Inconsistent peer role: {peer_role}"
        else:
            pytest.fail(f"Invalid sequence_role at {rid}: {role}")

    assert follow_count == 54
    assert precede_count == 54


def test_reciprocal_deduplication(corpus_records, v1_3_report):
    """Verify that pair enumeration deduplicates reciprocal representations."""
    for q in v1_3_report["predictions"]:
        pool = q["pool"]
        pairs_in_pool = []
        for u in pool:
            wf = corpus_records[u][WORKFLOW]
            v = wf["id"]
            if v in pool and wf["sequence_role"].startswith("Follow with"):
                pairs_in_pool.append((u, v))

        # Check that no unordered pair appears twice
        unordered_pairs = [tuple(sorted([u, v])) for u, v in pairs_in_pool]
        assert len(unordered_pairs) == len(set(unordered_pairs)), f"Duplicate pair in query {q['query_id']}"


def test_synthesis_diagnostic_ceilings(corpus_records, v1_3_report):
    """Verify the 3 diagnostic ceilings on the 10 synthesis queries."""
    synth_queries = [q for q in v1_3_report["predictions"] if q["query_type"] == "synthesis"]
    assert len(synth_queries) == 10

    both_in_pool = 0
    linked_in_either_direction = 0
    correctly_oriented = 0

    for q in synth_queries:
        p = q["gold_primary_id"]
        c = q["gold_competing_id"]
        pool = set(q["pool"])

        if p in pool and c in pool:
            both_in_pool += 1

        wf_p = corpus_records[p][WORKFLOW]
        is_linked = (wf_p["id"] == c)
        if is_linked:
            linked_in_either_direction += 1
            if wf_p["sequence_role"].startswith("Follow with"):
                correctly_oriented += 1

    assert both_in_pool == 10
    assert linked_in_either_direction == 10
    assert correctly_oriented == 10


def test_fallback_parity(v1_3_report):
    """Verify that candidate evaluation produces exact match to v1.3 when promotion is disabled."""
    from eval_v1_6_ordered_pair_ranking import run_evaluation, evaluate_benchmark

    res_disabled = run_evaluation(enable_pair_promotion=False, device="cpu")
    v1_3_eval = evaluate_benchmark(v1_3_report["predictions"])

    for k in ["recall_at_3", "flip", "abstention_precision", "abstention_recall", "ordered_synthesis"]:
        assert res_disabled["metrics"]["all"][k] == v1_3_eval[k], f"Mismatch on metric {k}"

    for p_cand, p_v13 in zip(res_disabled["predictions"], v1_3_report["predictions"]):
        assert p_cand["top3"] == p_v13["top3"], f"Mismatch in top3 for {p_cand['query_id']}"
        assert p_cand["primary"] == p_v13["primary"], f"Mismatch in primary for {p_cand['query_id']}"


def test_metadata_invariance(v1_3_report):
    """Ensure benchmark queries and IDs are preserved identically."""
    eval_queries_file = DATA_DIR / "eval_queries_v0.2.jsonl"
    raw_queries = [json.loads(line) for line in eval_queries_file.read_text(encoding="utf8").splitlines()]

    assert len(raw_queries) == 42
    assert [q["query_id"] for q in raw_queries] == [q["query_id"] for q in v1_3_report["predictions"]]
