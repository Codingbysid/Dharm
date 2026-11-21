"""Unit and Regression Tests for Dharm Release Candidate 1 (RC1) Parity.

Verifies:
1. Exact bit-for-bit parity between raw-text inference and saved frozen predictions:
   - 42/42 agreement on abstention
   - 42/42 agreement on primary decision
   - 42/42 agreement on ordered top-3 rankings
   - 42/42 agreement on verifier promotion decisions
2. Numerical probability tolerances:
   - Max divergence <= 1e-4 on abstention, router, and verifier probabilities
3. Latency constraints:
   - Cold startup latency <= 30.0 seconds
   - Warm per-query latency <= 15.0 seconds
"""
import json
from pathlib import Path
import pytest

ARTIFACTS_V1_6 = Path("decision-corpus-ml/artifacts-v1.6")


@pytest.fixture(scope="module")
def parity_report():
    report_path = ARTIFACTS_V1_6 / "rc1_parity_report.json"
    if not report_path.exists():
        pytest.skip(f"Parity report not found at {report_path}. Run eval_rc1_parity.py first.")
    return json.loads(report_path.read_text(encoding="utf8"))


def test_parity_abstention_agreement(parity_report):
    """Verify 42/42 exact agreement on epistemic abstention."""
    m, total = parity_report["agreement"]["abstention"]
    assert m == total == 42


def test_parity_primary_agreement(parity_report):
    """Verify 42/42 exact agreement on primary decisions."""
    m, total = parity_report["agreement"]["primary"]
    assert m == total == 42


def test_parity_top3_agreement(parity_report):
    """Verify 42/42 exact agreement on ordered top-3 rankings."""
    m, total = parity_report["agreement"]["ordered_top3"]
    assert m == total == 42


def test_parity_verifier_decision_agreement(parity_report):
    """Verify 42/42 exact agreement on verifier acceptance decisions."""
    m, total = parity_report["agreement"]["verifier_decision"]
    assert m == total == 42


def test_parity_probability_tolerances(parity_report):
    """Verify zero probability divergences exceeding 1e-4."""
    divs = parity_report.get("prob_divergences", [])
    assert len(divs) == 0, f"Found probability divergences: {divs}"


def test_parity_latency_bounds(parity_report):
    """Verify cold start and warm per-query latency satisfy operational budgets."""
    cold_start = parity_report["cold_startup_time_sec"]
    mean_lat = parity_report["latency_stats_sec"]["mean"]
    assert cold_start < 30.0, f"Cold startup {cold_start:.2f}s exceeded budget"
    assert mean_lat < 15.0, f"Mean query latency {mean_lat:.2f}s exceeded budget"
