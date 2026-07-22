"""CPU-only tests for the E1/E2 readiness additions (torch-free).

Covers: versioned static equivalence rules + machine_proof, the
INCONCLUSIVE-reason classifier, task-level cross-fitting, the corpus
stable-ID normalization, and the E1 driver's pure adapters.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.crossfit import assign_folds, crossfit_map_evaluation, task_of_probe
from src.audit.inconclusive import (
    REASON_CUDA_INVALID_CONFIG,
    REASON_OOM,
    REASON_OTHER,
    REASON_STATE_SYNC,
    REASON_TIMEOUT,
    classify_inconclusive_text,
    classify_observation,
)
from src.models import Mutant, MutationSite
from src.mutengine.static_equiv_rules import (
    RULE_VERSIONS,
    _RULES,
    machine_proof,
    rules_content_version,
)
from scripts.reconcile_corpora import normalize_source, stable_id
from scripts.run_e1_probe_study import _trials_to_map_records, load_historical_details


def _mutant(original="x = 1\n", mutated="x = 2\n", operator="const_perturb"):
    return Mutant(
        id="L1_P1__" + operator + "__0",
        operator_name=operator,
        operator_category="A",
        site=MutationSite(line_start=1, line_end=1, original_code="x = 1"),
        original_code=original,
        mutated_code=mutated,
    )


# ---------------------------------------------------------------------------
# Versioned static rules / machine proof
# ---------------------------------------------------------------------------

def test_every_rule_has_a_version():
    assert {name for name, _ in _RULES} == set(RULE_VERSIONS)
    assert all(RULE_VERSIONS.values())


def test_rules_content_version_is_stable():
    assert rules_content_version() == rules_content_version()
    assert len(rules_content_version()) == 16


def test_machine_proof_byte_identical():
    proof = machine_proof(_mutant(original="same\n", mutated="same\n"))
    assert proof["proof_kind"] == "byte_identical"
    assert proof["rules_version"] == rules_content_version()


def test_machine_proof_none_for_real_change():
    # A constant that IS consumed by the Model class: no rule may fire.
    original = "class ModelNew:\n    def forward(self):\n        return N\nN = 2048\n"
    mutated = "class ModelNew:\n    def forward(self):\n        return N\nN = 2049\n"
    mutant = _mutant(original=original, mutated=mutated, operator="const_perturb")
    mutant.site.line_start = 4
    assert machine_proof(mutant) is None


def test_machine_proof_static_rule_carries_version():
    # dead_host_constant: module-level constant used only by get_inputs().
    original = "N = 2048\n\ndef get_inputs():\n    return [N]\n"
    mutated = "N = 2049\n\ndef get_inputs():\n    return [N]\n"
    mutant = _mutant(original=original, mutated=mutated, operator="const_perturb")
    proof = machine_proof(mutant)
    assert proof is not None
    assert proof["proof_kind"] == "static_rule"
    assert proof["rule"] == "dead_host_constant"
    assert proof["rule_version"] == RULE_VERSIONS["dead_host_constant"]


# ---------------------------------------------------------------------------
# INCONCLUSIVE classification (E0 lesson 3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("StateSyncError: state_dict keys differ and cannot be aligned", REASON_STATE_SYNC),
    ("strict state synchronization failed: missing=('bn.weight',)", REASON_STATE_SYNC),
    ("CUDA error: invalid configuration argument", REASON_CUDA_INVALID_CONFIG),
    ("RuntimeError: CUDA out of memory. Tried to allocate 10 GiB", REASON_OOM),
    ("worker inconclusive_timeout after 420s", REASON_TIMEOUT),
    ("something entirely new", REASON_OTHER),
    ("", REASON_OTHER),
])
def test_classify_inconclusive_text(text, expected):
    assert classify_inconclusive_text(text) == expected


def test_classify_observation_nested_and_timeout():
    record = {"trials": [{"reason": "state_dict keys differ ..."}]}
    assert classify_observation(record) == REASON_STATE_SYNC
    assert classify_observation({}, timed_out=True) == REASON_TIMEOUT


# ---------------------------------------------------------------------------
# Task-level cross-fitting
# ---------------------------------------------------------------------------

def _record(probe, operator, policy, verdict, order=0):
    return {
        "probe_id": probe,
        "operator": operator,
        "case": {"policy": policy, "mode": "eval", "parameters": {}},
        "verdict": verdict,
        "order": order,
    }


def test_fold_assignment_is_deterministic_and_task_level():
    tasks = [f"L1_P{i}" for i in range(20)]
    first = assign_folds(tasks, 5)
    second = assign_folds(list(reversed(tasks)), 5)
    assert first == second
    assert set(first.values()) <= set(range(5))


def test_task_of_probe_derivation():
    assert task_of_probe({"probe_id": "L1_P39__index_replace__3"}) == "L1_P39"
    assert task_of_probe({"probe_id": "x__y__0", "task_id": "T9"}) == "T9"


def test_crossfit_pools_over_folds():
    records = []
    for i in range(12):
        probe = f"L1_P{i}__arith_replace__0"
        records.append(_record(probe, "arith_replace", "large_magnitude",
                               "SPEC_VIOLATION"))
        records.append(_record(probe, "arith_replace", "near_zero",
                               "INDISTINGUISHED", order=1))
    result = crossfit_map_evaluation(records, k=3, planned_cases=4)
    assert result["pooled"]["witnessed"] == 12
    # Every training fold sees the same killing case, so closure is total.
    assert result["pooled"]["closed"] == 12
    assert result["pooled"]["closure_rate"] == 1.0
    # No probe was judged by a map its own task helped build.
    for fold in result["per_fold"]:
        for task in fold.get("held_tasks", []):
            assert result["fold_assignment"][task] == fold["fold"]


# ---------------------------------------------------------------------------
# Corpus stable IDs
# ---------------------------------------------------------------------------

def test_normalize_source_strips_only_whitespace():
    a = "import torch  \n\nx = 1\n\n\n"
    b = "import torch\n\nx = 1\n"
    assert normalize_source(a) == normalize_source(b)
    assert stable_id(a) == stable_id(b)
    assert stable_id("x = 1\n") != stable_id("x = 2\n")


# ---------------------------------------------------------------------------
# E1 driver pure adapters
# ---------------------------------------------------------------------------

def test_trials_to_map_records_maps_statuses():
    row = {
        "probe_id": "L1_P7__index_replace__9",
        "operator_name": "index_replace",
        "trials": [
            {"status": "pass", "policy": None},
            {"status": "fail", "policy": "head_heavy"},
            {"status": "inconclusive", "policy": "sparse"},
        ],
    }
    records = _trials_to_map_records(row)
    assert [r["verdict"] for r in records] == [
        "INDISTINGUISHED", "SPEC_VIOLATION", "INCONCLUSIVE"]
    assert records[0]["case"]["policy"] == "iid"
    assert records[1]["case"]["policy"] == "head_heavy"
    assert all(r["task_id"] == "L1_P7" for r in records)


def test_load_historical_details_recovers_kernel_source(tmp_path):
    detail = {
        "kernel": {"problem_id": 7, "level": 1, "problem_name": "L1_P7",
                   "language": "cuda"},
        "mutants": [
            {"id": "L1_P7__a__0", "status": "killed",
             "original_code": "SRC", "mutated_code": "MUT1"},
            {"id": "L1_P7__b__1", "status": "survived",
             "original_code": "SRC", "mutated_code": "MUT2"},
        ],
    }
    (tmp_path / "L1_P7.json").write_text(json.dumps(detail), encoding="utf-8")
    kernels = load_historical_details(tmp_path)
    assert len(kernels) == 1
    assert kernels[0]["kernel_source"] == "SRC"
    assert kernels[0]["historical_probe_ids"] == ["L1_P7__a__0", "L1_P7__b__1"]
    assert kernels[0]["historical_status"]["L1_P7__b__1"] == "survived"


def test_load_historical_details_rejects_mixed_sources(tmp_path):
    detail = {
        "kernel": {"problem_id": 8, "level": 1, "problem_name": "L1_P8"},
        "mutants": [
            {"id": "a", "status": "killed", "original_code": "SRC1",
             "mutated_code": "M"},
            {"id": "b", "status": "killed", "original_code": "SRC2",
             "mutated_code": "M"},
        ],
    }
    (tmp_path / "L1_P8.json").write_text(json.dumps(detail), encoding="utf-8")
    with pytest.raises(ValueError):
        load_historical_details(tmp_path)
