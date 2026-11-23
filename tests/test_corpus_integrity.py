"""Deep integrity tests for Dharm corpus, CSV/JSONL parity, graph structure, and model configurations."""
import csv
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "decisions_v0.2.csv"
JSONL_PATH = ROOT / "decision-corpus-ml" / "data" / "decisions_v0.2.jsonl"
VERIFIER_CFG_PATH = ROOT / "decision-corpus-ml" / "artifacts-v1.6" / "verifier_config.json"


def test_csv_jsonl_deep_field_by_field_parity():
    """Verify 100% field-by-field parity between decisions_v0.2.csv and decisions_v0.2.jsonl."""
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"
    assert JSONL_PATH.exists(), f"Missing {JSONL_PATH}"

    csv_records = {}
    with open(CSV_PATH, mode="r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            row["scenarios_json"] = json.loads(row["scenarios_json"])
            row["competing_passage_id"] = json.loads(row["competing_passage_id"])
            row["complementary_passage_id"] = json.loads(row["complementary_passage_id"])
            csv_records[pid] = row

    jsonl_records = {}
    with open(JSONL_PATH, mode="r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                jsonl_records[r["id"]] = r

    assert len(csv_records) == 120, f"Expected 120 CSV records, got {len(csv_records)}"
    assert len(jsonl_records) == 120, f"Expected 120 JSONL records, got {len(jsonl_records)}"
    assert set(csv_records.keys()) == set(jsonl_records.keys()), "Passage ID mismatch between CSV and JSONL"

    # Compare every single field for all 120 passages
    for pid, c_row in csv_records.items():
        j_row = jsonl_records[pid]
        assert set(c_row.keys()) == set(j_row.keys()), f"Key mismatch at {pid}"
        for k in c_row:
            assert c_row[k] == j_row[k], f"Field mismatch at {pid}.{k}: CSV={c_row[k]!r} != JSONL={j_row[k]!r}"


def test_complete_graph_structure():
    """Verify graph structure: competing reciprocity, workflow sequence roles, and zero broken links."""
    records = {json.loads(line)["id"]: json.loads(line) for line in open(JSONL_PATH, encoding="utf8") if line.strip()}
    assert len(records) == 120

    nonreciprocal_comp = []
    reciprocal_comp = []
    for u, r in records.items():
        v = r["competing_passage_id"]["id"]
        assert v in records and v != u, f"Broken or self competing edge at {u} -> {v}"
        v_comp = records[v]["competing_passage_id"]["id"]
        if v_comp != u:
            nonreciprocal_comp.append((u, v, v_comp))
        elif u < v:
            reciprocal_comp.append((u, v))

    assert len(nonreciprocal_comp) == 24, f"Expected 24 non-reciprocal edges, got {len(nonreciprocal_comp)}"
    assert len(reciprocal_comp) == 48, f"Expected 48 reciprocal pairs, got {len(reciprocal_comp)}"

    follow_count = 0
    precede_count = 0
    for u, r in records.items():
        v = r["complementary_passage_id"]["id"]
        role = r["complementary_passage_id"]["sequence_role"]
        assert v in records and v != u, f"Broken or self workflow edge at {u} -> {v}"
        peer = records[v]["complementary_passage_id"]
        assert peer["id"] == u, f"Non-reciprocal workflow peer at {u} <-> {v}"
        if role.startswith("Follow with"):
            follow_count += 1
            assert role.startswith(f"Follow with {v}."), f"Malformed follow role at {u}: {role}"
            assert peer["sequence_role"].startswith(f"Precede with {u}."), f"Inconsistent peer role at {v}"
        elif role.startswith("Precede with"):
            precede_count += 1
            assert role.startswith(f"Precede with {v}."), f"Malformed precede role at {u}: {role}"
            assert peer["sequence_role"].startswith(f"Follow with {u}."), f"Inconsistent peer role at {v}"
        else:
            pytest.fail(f"Invalid sequence role at {u}: {role}")

    assert follow_count == 60, f"Expected 60 follow edges, got {follow_count}"
    assert precede_count == 60, f"Expected 60 precede edges, got {precede_count}"


def test_verifier_config_provenance_and_hash():
    """Verify verifier configuration hash matches canonical SHA-256."""
    cfg = json.loads(VERIFIER_CFG_PATH.read_text(encoding="utf8"))
    stored_hash = cfg["config_sha256"]
    cfg_copy = dict(cfg)
    del cfg_copy["config_sha256"]
    computed_hash = hashlib.sha256(json.dumps(cfg_copy, sort_keys=True, indent=2).encode("utf8")).hexdigest()
    assert stored_hash == computed_hash, "Verifier config SHA-256 hash mismatch"
    assert cfg["calibrated_threshold"] == 0.57
    assert len(cfg["features"]) == 16
