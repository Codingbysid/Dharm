"""Floors for the transfer model. This does not reload RC1.

The frozen bit-for-bit parity report stays in test_rc1_parity.py.
These checks read the latest local evaluation of Q01–Q42 and eval_transfer_v1.
"""
import json
from pathlib import Path

REPORT = Path("decision-corpus-ml/artifacts-v1.6/transfer_eval_report.json")


def _report():
    assert REPORT.exists(), "Missing transfer_eval_report.json"
    return json.loads(REPORT.read_text(encoding="utf8"))


def test_transfer_eval_floors():
    report = _report()
    recall, total = report["q_recall_at_3"]
    abstained, abst_total = report["q_abstention"]
    transfer, transfer_total = report["transfer_hit"]
    assert report["records"] == 120
    assert total == 38 and recall >= 28
    assert (abstained, abst_total) == (4, 4)
    assert transfer_total == 24 and transfer >= 22
