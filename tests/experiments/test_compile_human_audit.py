import importlib.util
from pathlib import Path

import pytest

from src.experiments import stable_json_sha256


SCRIPT = Path(__file__).parents[2] / "scripts" / "compile_human_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("compile_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs():
    queue = [
        {"audit_id": "subject-audit", "item_kind": "subject"},
        {"audit_id": "alarm-audit", "item_kind": "alarm"},
    ]
    for item in queue:
        item["item_sha256"] = stable_json_sha256(item)
    sealed = {
        "subject_manifest_sha256": "a" * 64,
        "experiment_plan_sha256": "b" * 64,
        "run_manifest_sha256": "c" * 64,
        "observations_sha256": "d" * 64,
        "queue_item_count": len(queue),
        "queue_audit_ids": sorted(item["audit_id"] for item in queue),
        "queue_sha256": stable_json_sha256(sorted(queue, key=lambda item: item["audit_id"])),
        "mapping": {
            "subject-audit": {"subject_id": "s1"},
            "alarm-audit": {
                "subject_id": "s1",
                "test_id": "1" * 64,
                "scope": "in_contract",
            },
        }
    }
    sealed["sealed_mapping_sha256"] = stable_json_sha256(sealed)
    annotations = [
        {
            "audit_id": "subject-audit",
            "auditor_id": "A",
            "annotation_role": "primary",
            "primary_label": "CONFIRMED_IN_CONTRACT_DEFECT",
            "confidence": "high",
            "rationale": "replayed",
        },
        {
            "audit_id": "subject-audit",
            "auditor_id": "B",
            "annotation_role": "primary",
            "primary_label": "CONFIRMED_IN_CONTRACT_DEFECT",
            "confidence": "high",
            "rationale": "independent replay",
        },
        {
            "audit_id": "alarm-audit",
            "auditor_id": "A",
            "annotation_role": "primary",
            "primary_label": "CONFIRMED_IN_CONTRACT_DISCREPANCY",
            "confidence": "high",
            "rationale": "valid witness",
        },
        {
            "audit_id": "alarm-audit",
            "auditor_id": "B",
            "annotation_role": "primary",
            "primary_label": "INVALID_INPUT",
            "confidence": "medium",
            "rationale": "contract dispute",
        },
        {
            "audit_id": "alarm-audit",
            "auditor_id": "C",
            "annotation_role": "adjudicator",
            "primary_label": "CONFIRMED_IN_CONTRACT_DISCREPANCY",
            "confidence": "high",
            "rationale": "contract confirms input",
        },
    ]
    for row in annotations:
        row.update(
            {
                "fault_class": "numeric_or_contract",
                "contract_clause": "value_domain",
                "evidence_reproduced": True,
            }
        )
    return queue, sealed, annotations


def test_agreement_is_pre_adjudication_and_analysis_labels_are_joined():
    module = _load()
    queue, sealed, annotations = _inputs()

    report, labels = module.compile_audit(
        queue=queue,
        sealed_mapping=sealed,
        annotations=annotations,
        require_complete=True,
    )

    assert report["raw_agreement"] == 0.5
    assert report["disagreements"] == 1
    assert report["adjudicated"] == 1
    assert report["pending"] == []
    assert {row["record_type"] for row in labels} == {"subject", "alarm"}
    assert any(row.get("test_id") == "1" * 64 for row in labels)


def test_complete_mode_fails_closed_on_a_missing_second_auditor():
    module = _load()
    queue, sealed, annotations = _inputs()
    annotations = [row for row in annotations if row["auditor_id"] != "B"]

    with pytest.raises(module.AuditCompileError, match="audit is incomplete"):
        module.compile_audit(
            queue=queue,
            sealed_mapping=sealed,
            annotations=annotations,
            require_complete=True,
        )


def test_queue_cannot_drop_a_sealed_item_before_compilation():
    module = _load()
    queue, sealed, annotations = _inputs()

    with pytest.raises(module.AuditCompileError, match="ID set"):
        module.compile_audit(
            queue=queue[:1],
            sealed_mapping=sealed,
            annotations=[row for row in annotations if row["audit_id"] == "subject-audit"],
            require_complete=True,
        )


def test_alarm_label_must_match_the_detector_scope():
    module = _load()
    queue, sealed, annotations = _inputs()
    sealed.pop("sealed_mapping_sha256")
    sealed["mapping"]["alarm-audit"]["scope"] = "extended_contract"
    sealed["sealed_mapping_sha256"] = stable_json_sha256(sealed)

    with pytest.raises(module.AuditCompileError, match="extended-contract alarm"):
        module.compile_audit(
            queue=queue,
            sealed_mapping=sealed,
            annotations=annotations,
            require_complete=True,
        )


def test_adjudicator_must_be_independent_of_primary_auditors():
    module = _load()
    queue, sealed, annotations = _inputs()
    annotations[-1]["auditor_id"] = "A"

    with pytest.raises(module.AuditCompileError, match="independent"):
        module.compile_audit(
            queue=queue,
            sealed_mapping=sealed,
            annotations=annotations,
            require_complete=True,
        )
