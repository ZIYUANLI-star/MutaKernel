"""Tests for the site-directed stress-plan derivation (M8, torch-free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validator.site_directed import (
    DEFAULT_GENERAL_SEQUENCE,
    derive_seed,
    derive_site_directed_plan,
)


def _map():
    return {
        "map_version": "test-map-1",
        "entries": [
            {
                "fault_class": "F-EPS",
                "effective_cases": [
                    {"case": {"policy": "near_zero", "mode": "eval", "parameters": {}},
                     "closure_rate": 0.9, "mean_cost_ms": 100},
                    {"case": {"policy": "denormals", "mode": "eval", "parameters": {}},
                     "closure_rate": 0.5},
                ],
            },
            {
                "fault_class": "F-SYNC",
                "effective_cases": [
                    {"case": {"policy": "iid", "mode": "repeated",
                               "parameters": {"repeat_count": 4}},
                     "closure_rate": 0.7},
                ],
            },
            {
                "fault_class": "F-BOUND",
                "effective_cases": [
                    {"case": {"policy": "boundary_last_element", "mode": "eval",
                               "parameters": {}},
                     "closure_rate": 0.6},
                ],
            },
        ],
    }


def _fingerprint(present):
    return {"fingerprint_version": "fpv-1", "fault_classes_present": list(present)}


def test_directed_cases_come_first_and_are_labelled():
    result = derive_site_directed_plan(
        subject_id="subj-1",
        fingerprint=_fingerprint(["F-EPS", "F-SYNC"]),
        fault_to_stress_map=_map(),
        budget_candidate_calls=10,
    )
    plan = result["plan"]
    directed = [case for case in plan if case["source"] == "directed"]
    assert directed, "directed section must not be empty when sites are present"
    # best closure rate first: F-EPS near_zero (0.9) before F-SYNC repeated (0.7)
    assert directed[0]["policy"] == "near_zero"
    assert directed[0]["fault_class"] == "F-EPS"
    assert directed[0]["map_closure_rate"] == 0.9
    # absent fault classes are not in the directed section
    assert all(case.get("fault_class") != "F-BOUND" for case in directed)
    # but the general fallback section still exists (no exclusion principle)
    assert any(case["source"] == "general" for case in plan)


def test_budget_accounting_counts_repeated_cost():
    result = derive_site_directed_plan(
        subject_id="subj-1",
        fingerprint=_fingerprint(["F-SYNC"]),
        fault_to_stress_map=_map(),
        budget_candidate_calls=6,
    )
    assert result["budget"]["planned_calls"] <= 6
    repeated = [c for c in result["plan"] if c["mode"] == "repeated"]
    assert all(c["candidate_run_cost"] >= 2 for c in repeated)


def test_plan_is_deterministic_and_seeds_are_stable():
    kwargs = dict(
        subject_id="subj-42",
        fingerprint=_fingerprint(["F-EPS"]),
        fault_to_stress_map=_map(),
        budget_candidate_calls=8,
    )
    first = derive_site_directed_plan(**kwargs)
    second = derive_site_directed_plan(**kwargs)
    assert first == second
    case = {"policy": "near_zero", "mode": "eval", "parameters": {}}
    assert derive_seed("subj-42", case) == derive_seed("subj-42", dict(case))
    assert derive_seed("subj-42", case) != derive_seed("subj-43", case)


def test_contract_gate_records_skipped_cases():
    def deny_repeated(case):
        return case.get("mode") != "repeated"

    result = derive_site_directed_plan(
        subject_id="subj-1",
        fingerprint=_fingerprint(["F-SYNC"]),
        fault_to_stress_map=_map(),
        budget_candidate_calls=8,
        is_authorized=deny_repeated,
    )
    assert all(case["mode"] != "repeated" for case in result["plan"])
    assert any(case["mode"] == "repeated" for case in result["skipped_unauthorized"])


def test_no_sites_degenerates_to_general_sequence_order():
    result = derive_site_directed_plan(
        subject_id="subj-1",
        fingerprint=_fingerprint([]),
        fault_to_stress_map=_map(),
        budget_candidate_calls=5,
    )
    assert all(case["source"] == "general" for case in result["plan"])
    expected_first = DEFAULT_GENERAL_SEQUENCE[0]["policy"]
    assert result["plan"][0]["policy"] == expected_first
