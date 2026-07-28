"""CSE falsification lane support (torch-free, CPU-only).

Pins the CSE dual-track contract: lane output files follow the
``cse_observations_lane*`` / ``cse_completed_lane*`` monitoring globs, the
legacy serial filenames are untouched, and a lane driver's skip set folds in
the legacy checkpoint plus every other lane's completed file while
checkpointing only its own probes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_e1_cse_falsify import (
    CSE_MIN_STRESS_ROUNDS,
    FALSIFY_EQUIV_RUNS,
    classify_vram_evidence,
    cse_lane_paths,
    grade_cse_evidence,
    load_cse_skip_set,
    order_kernel_files,
    witness_fields,
)


def test_legacy_serial_paths_unchanged(tmp_path: Path):
    legacy = cse_lane_paths(tmp_path, None)
    assert legacy["obs"].name == "cse_falsify_observations.jsonl"
    assert legacy["done"].name == "cse_falsify_completed.json"


def test_lane_paths_match_monitoring_globs_and_are_isolated(tmp_path: Path):
    lane0 = cse_lane_paths(tmp_path, 0)
    lane1 = cse_lane_paths(tmp_path, 1)
    lane2 = cse_lane_paths(tmp_path, 2)
    assert lane0["obs"].name == "cse_observations_lane0.jsonl"
    assert lane1["done"].name == "cse_completed_lane1.json"
    assert lane0["obs"].match("cse_observations_lane*.jsonl")
    assert lane2["done"].match("cse_completed_lane*.json")
    everything = [p for group in (lane0, lane1, lane2) for p in group.values()]
    assert len(everything) == len(set(everything))


def test_tagged_lane_paths_keep_glob_shape(tmp_path: Path):
    tagged = cse_lane_paths(tmp_path, 0, tag="requeue")
    assert tagged["obs"].name == "cse_observations_lane0_requeue.jsonl"
    assert tagged["obs"].match("cse_observations_lane*.jsonl")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    (tmp_path / "cse_falsify_completed.json").write_text(
        json.dumps(["legacy1", "legacy2"]))
    (tmp_path / "cse_completed_lane0.json").write_text(json.dumps(["h1"]))
    (tmp_path / "cse_completed_lane1.json").write_text(json.dumps(["a1", "a2"]))
    return tmp_path


def test_lane_skip_set_folds_legacy_and_other_lanes(out_dir: Path):
    own, skip = load_cse_skip_set(out_dir, 1)
    assert own == {"a1", "a2"}
    assert skip == {"legacy1", "legacy2", "h1", "a1", "a2"}


def test_lane_without_checkpoint_still_skips_everything_else(out_dir: Path):
    own, skip = load_cse_skip_set(out_dir, 2)
    assert own == set()
    assert skip == {"legacy1", "legacy2", "h1", "a1", "a2"}


def test_serial_mode_reads_only_legacy_checkpoint(out_dir: Path):
    own, skip = load_cse_skip_set(out_dir, None)
    assert own == skip == {"legacy1", "legacy2"}


def test_empty_out_dir_is_tolerated(tmp_path: Path):
    own, skip = load_cse_skip_set(tmp_path, 0)
    assert own == skip == set()


# ---------------------------------------------------------------------------
# VRAM-evidence classification for the heavy-lane split
# ---------------------------------------------------------------------------

def test_resource_events_are_proven_unsafe():
    hist = {"A": {"resource_events": True, "capped45_complete": True,
                  "capped45_timeout": False}}
    assert classify_vram_evidence(hist) == {"A": "PROVEN_UNSAFE"}


def test_known_unsafe_list_overrides_clean_history():
    hist = {"A": {"resource_events": False, "capped45_complete": True,
                  "capped45_timeout": False}}
    assert classify_vram_evidence(hist, known_unsafe=["A"]) == {
        "A": "PROVEN_UNSAFE"}


def test_clean_capped45_completion_is_proven_safe():
    hist = {"A": {"resource_events": False, "capped45_complete": True,
                  "capped45_timeout": False}}
    assert classify_vram_evidence(hist) == {"A": "PROVEN_SAFE"}


def test_capped45_timeout_blocks_safety_proof():
    hist = {"A": {"resource_events": False, "capped45_complete": True,
                  "capped45_timeout": True}}
    assert classify_vram_evidence(hist) == {"A": "UNKNOWN"}


def test_only_high_fraction_history_is_unknown():
    # e.g. ran only under the 0.9 requeue lane: no 0.45 proof either way.
    hist = {"A": {"resource_events": False, "capped45_complete": False,
                  "capped45_timeout": False}}
    assert classify_vram_evidence(hist) == {"A": "UNKNOWN"}


# ---------------------------------------------------------------------------
# CSE partial-evidence grading (mirror of the equiv timeout fix)
# ---------------------------------------------------------------------------

def _cse_trials(random_pass=0, stress_pass=0):
    trials = [{"round_type": "random", "round_index": i, "status": "pass"}
              for i in range(random_pass)]
    trials += [{"round_type": "stress", "sub_index": i, "status": "pass"}
               for i in range(stress_pass)]
    return trials


def test_cse_completed_pass_unchanged():
    result = {"validation_status": "pass",
              "trials": _cse_trials(FALSIFY_EQUIV_RUNS, 63)}
    status, outcome, evidence = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("pass", "STILL_LIKELY_EQUIVALENT")
    assert evidence is None


def test_cse_completed_fail_unchanged():
    result = {"validation_status": "fail",
              "divergence": {"detail": "output_diverged"}}
    status, outcome, evidence = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("fail", "FALSIFIED")
    assert evidence is None


def test_cse_timeout_meeting_threshold_stays_still_likely():
    result = {"partial": True, "validation_status": "inconclusive",
              "trials": _cse_trials(FALSIFY_EQUIV_RUNS, CSE_MIN_STRESS_ROUNDS)}
    status, outcome, evidence = grade_cse_evidence(result, timed_out=True)
    assert (status, outcome) == ("pass", "STILL_LIKELY_EQUIVALENT")
    assert evidence["budget_exhausted"] is True
    assert evidence["random_rounds_passed"] == FALSIFY_EQUIV_RUNS
    assert evidence["stress_rounds_passed"] == CSE_MIN_STRESS_ROUNDS


def test_cse_timeout_below_threshold_is_inconclusive():
    result = {"partial": True, "validation_status": "inconclusive",
              "trials": _cse_trials(FALSIFY_EQUIV_RUNS,
                                    CSE_MIN_STRESS_ROUNDS - 1)}
    status, outcome, evidence = grade_cse_evidence(result, timed_out=True)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")
    assert evidence["budget_exhausted"] is True


def test_cse_timeout_with_witness_stays_falsified():
    result = {"partial": True, "validation_status": "inconclusive",
              "divergence": {"detail": "output_diverged"},
              "trials": _cse_trials(10, 0)}
    status, outcome, _ = grade_cse_evidence(result, timed_out=True)
    assert (status, outcome) == ("fail", "FALSIFIED")


def test_cse_timeout_without_result_is_inconclusive():
    status, outcome, evidence = grade_cse_evidence(None, timed_out=True)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")
    assert evidence["rounds_completed"] == 0


# ---------------------------------------------------------------------------
# Resource-degraded completion (preregistered 2026-07-27)
# ---------------------------------------------------------------------------

def _oom_trial(round_type, idx):
    return {
        "round_type": round_type, "round_index": idx, "status": "inconclusive",
        "reason": ("validation setup failed: InputIsolationError: "
                   "OutOfMemoryError: CUDA out of memory. Tried to allocate "
                   "4.00 GiB"),
        "errors": [{"phase": "setup", "message": "CUDA out of memory"}],
    }


def test_resource_voided_rounds_do_not_poison_completed_probe():
    # Mode A shape (L1_P20__relop_replace__3): 35 random + 63 stress passed,
    # 5 random rounds voided by CUDA OOM, run completed without timeout.
    trials = _cse_trials(random_pass=35, stress_pass=63)
    trials += [_oom_trial("random", 35 + i) for i in range(5)]
    result = {"validation_status": "inconclusive", "trials": trials}
    status, outcome, evidence = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("pass", "STILL_LIKELY_EQUIVALENT")
    assert evidence["resource_degraded"] is True
    assert evidence["environmental_voided_rounds"] == 5
    assert evidence["rounds_completed"] == 98


def test_mass_voided_probe_stays_inconclusive():
    # P96 shape: nearly all rounds OOM-voided -> below evidence floor.
    trials = [_oom_trial("random", i) for i in range(40)]
    trials += [_oom_trial("stress", i) for i in range(9)]
    result = {"validation_status": "inconclusive", "trials": trials}
    status, outcome, _ = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")


def test_non_environmental_void_blocks_rescue():
    trials = _cse_trials(random_pass=35, stress_pass=63)
    trials.append({
        "round_type": "random", "round_index": 35, "status": "inconclusive",
        "reason": "strict state synchronization failed", "errors": [],
    })
    result = {"validation_status": "inconclusive", "trials": trials}
    status, outcome, _ = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")


def test_early_killed_all_pass_below_threshold_stays_inconclusive():
    # Mode B shape (SIGKILLed worker partial snapshot): 44 clean passes only.
    result = {"partial": True, "validation_status": "inconclusive",
              "trials": _cse_trials(random_pass=40, stress_pass=4)}
    status, outcome, _ = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")


def test_rescue_requires_stress_threshold():
    trials = _cse_trials(random_pass=40, stress_pass=41)
    trials += [_oom_trial("stress", 41 + i) for i in range(22)]
    result = {"validation_status": "inconclusive", "trials": trials}
    status, outcome, _ = grade_cse_evidence(result, timed_out=False)
    assert (status, outcome) == ("inconclusive", "INCONCLUSIVE")


def test_rescue_never_applies_with_divergence():
    # Anomalous shape (inconclusive status but divergence recorded): the
    # rescue must never promote it to STILL; it stays on the base grading.
    trials = _cse_trials(random_pass=40, stress_pass=50)
    result = {"validation_status": "inconclusive",
              "divergence": {"detail": "output_diverged"}, "trials": trials}
    status, outcome, _ = grade_cse_evidence(result, timed_out=False)
    assert outcome != "STILL_LIKELY_EQUIVALENT"


# ---------------------------------------------------------------------------
# Falsification witness extraction
# ---------------------------------------------------------------------------

def test_witness_fields_for_stress_round():
    div = {"round_type": "stress", "policy": "head_heavy", "sub_index": 2,
           "seed": 50041, "detail": "output_diverged"}
    assert witness_fields(div) == {
        "witness_policy": "head_heavy", "witness_round": 2,
        "witness_seed": 50041}


def test_witness_fields_for_random_round():
    div = {"round_type": "random", "round_index": 7, "policy": None,
           "seed": 50007}
    assert witness_fields(div) == {
        "witness_policy": "random", "witness_round": 7, "witness_seed": 50007}


def test_witness_fields_without_divergence():
    assert witness_fields(None) == {
        "witness_policy": None, "witness_round": None, "witness_seed": None}


# ---------------------------------------------------------------------------
# Plan-driven execution ordering (queue reordering)
# ---------------------------------------------------------------------------

def _kf(name):
    return {"kernel": {"problem_name": name}}


def test_kernel_files_follow_plan_order():
    files = [_kf("A"), _kf("B"), _kf("C")]
    out = order_kernel_files(files, ["C", "A", "B"])
    assert [kf["kernel"]["problem_name"] for kf in out] == ["C", "A", "B"]


def test_risky_kernels_listed_last_run_last():
    files = [_kf("P21"), _kf("P16"), _kf("P39"), _kf("P22")]
    out = order_kernel_files(files, ["P16", "P22", "P39", "P21"])
    assert [kf["kernel"]["problem_name"] for kf in out] == [
        "P16", "P22", "P39", "P21"]


def test_unlisted_kernels_keep_relative_order_at_end():
    files = [_kf("X"), _kf("A"), _kf("Y")]
    out = order_kernel_files(files, ["A"])
    assert [kf["kernel"]["problem_name"] for kf in out] == ["A", "X", "Y"]
