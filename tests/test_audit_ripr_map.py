"""Tests for RIPR escape classification and FaultToStressMap building (M7)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.mapbuild import build_fault_to_stress_map, case_key
from src.audit.ripr import (
    ABSORPTION_FAILURE_TOLERANCE,
    ACTIVATION_FAILURE_VALUE,
    MASKING_FAILURE_PRECISION,
    OBSERVATION_FAILURE_NONDETERMINISM,
    REACHABILITY_FAILURE_CONFIG,
    REACHABILITY_FAILURE_MODE,
    classify_escape,
    dimension_of_case,
)


def _case(policy="iid", mode="eval", **parameters):
    return {"policy": policy, "mode": mode, "parameters": parameters}


def test_dimension_assignment():
    assert dimension_of_case(_case()) == "value"
    assert dimension_of_case(_case(policy="near_zero")) == "value"
    assert dimension_of_case(_case(dtype="float16")) == "dtype"
    assert dimension_of_case(_case(mode="train")) == "training"
    assert dimension_of_case(_case(mode="repeated", repeat_count=10)) == "repeated"
    assert dimension_of_case(_case(mode="config", batch_size=4)) == "config"


def test_escape_decision_tree():
    assert classify_escape(_case(mode="config", batch_size=1))["mechanism"] == \
        REACHABILITY_FAILURE_CONFIG
    assert classify_escape(_case(mode="train", policy="near_zero"))["mechanism"] == \
        REACHABILITY_FAILURE_MODE
    assert classify_escape(_case(mode="repeated", repeat_count=10))["mechanism"] == \
        OBSERVATION_FAILURE_NONDETERMINISM
    assert classify_escape(_case(dtype="bfloat16"))["mechanism"] == \
        MASKING_FAILURE_PRECISION
    assert classify_escape(_case(policy="near_zero"))["mechanism"] == \
        ACTIVATION_FAILURE_VALUE
    assert classify_escape(_case(policy="iid"))["mechanism"] == \
        ABSORPTION_FAILURE_TOLERANCE


def test_escape_consistency_check_with_audit_reason():
    consistent = classify_escape(_case(policy="near_zero"), "value_insensitive")
    assert consistent["audit_consistent"] is True
    conflicting = classify_escape(_case(policy="near_zero"), "requires_config_change")
    assert conflicting["audit_consistent"] is False
    compatible = classify_escape(_case(policy="near_zero"), "predicate_unreachable")
    assert compatible["audit_consistent"] is True


def _record(probe, operator, case, verdict, **extra):
    record = {"probe_id": probe, "operator": operator, "case": case, "verdict": verdict}
    record.update(extra)
    return record


def test_map_building_end_to_end():
    near_zero = _case(policy="near_zero")
    iid = _case()
    records = [
        # probe A (epsilon): killed by near_zero, indistinguished on iid
        _record("pA", "epsilon_modify", near_zero, "SPEC_VIOLATION",
                cost_ms=100.0, counterexample_id="ceA", order=1),
        _record("pA", "epsilon_modify", iid, "INDISTINGUISHED", order=0),
        # probe B (epsilon): survived everything -> not witnessed
        _record("pB", "epsilon_modify", near_zero, "INDISTINGUISHED"),
        _record("pB", "epsilon_modify", iid, "INDISTINGUISHED"),
        # probe C (sync): killed only by repeated
        _record("pC", "sync_remove", _case(mode="repeated", repeat_count=10),
                "SPEC_VIOLATION", cost_ms=300.0, order=0),
        _record("pC", "sync_remove", near_zero, "EXACT_DIVERGENCE_ONLY", order=1),
        # invalid rounds never count as executions
        _record("pA", "epsilon_modify", _case(mode="config", batch_size=64),
                "INVALID_INPUT"),
    ]
    fmap = build_fault_to_stress_map(
        records, map_version="test-1", derived_from_run="run-x")

    assert fmap["witnessed_probe_count"] == 2
    entries = {entry["fault_class"]: entry for entry in fmap["entries"]}
    assert set(entries) == {"F-EPS", "F-SYNC"}

    eps = entries["F-EPS"]
    assert eps["witnessed_probes"] == 1
    best = eps["effective_cases"][0]
    assert best["case"]["policy"] == "near_zero"
    # denominator counts witnessed probes only (pB excluded)
    assert best["executions"] == 1 and best["kills"] == 1
    assert best["closure_rate"] == 1.0
    assert best["sole_detector_count"] == 1  # pA killed only in value dimension
    assert eps["evidence_counterexamples"] == ["ceA"]
    assert eps["escape_mechanisms"][0]["mechanism"] == ACTIVATION_FAILURE_VALUE

    sync = entries["F-SYNC"]
    assert sync["escape_mechanisms"][0]["mechanism"] == OBSERVATION_FAILURE_NONDETERMINISM
    # EXACT_DIVERGENCE_ONLY is an execution but never a kill
    nz_key = case_key(near_zero)
    nz_entries = [e for e in sync["effective_cases"] if case_key(e["case"]) == nz_key]
    assert nz_entries == []  # no kills for near_zero on F-SYNC -> no entry


def test_map_rejects_unknown_operator_and_conflicts():
    with pytest.raises(ValueError):
        build_fault_to_stress_map(
            [_record("p", "not_an_operator", _case(), "SPEC_VIOLATION")],
            map_version="v", derived_from_run="r")
    with pytest.raises(ValueError):
        build_fault_to_stress_map(
            [
                _record("p", "epsilon_modify", _case(), "SPEC_VIOLATION"),
                _record("p", "sync_remove", _case(), "SPEC_VIOLATION"),
            ],
            map_version="v", derived_from_run="r")
