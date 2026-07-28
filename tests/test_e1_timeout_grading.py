"""Partial-evidence grading for E1 equiv timeouts (torch-free, CPU-only).

The 2026-07-24 timeout fix: a whole-probe timeout (or a watchdog round
timeout inside the worker) no longer voids the completed rounds.  The
orchestrator grades the surviving evidence via
``grade_equiv_evidence``; these tests pin that contract.
"""

from __future__ import annotations

from scripts.run_e1_probe_study import (
    EQUIV_MIN_STRESS_ROUNDS,
    EQUIV_RUNS,
    EQUIV_STRESS_ROUNDS_PLANNED,
    grade_equiv_evidence,
)


def _trials(random_pass=0, stress_pass=0, random_inconclusive=0,
            stress_round_timeouts=0):
    trials = []
    for i in range(random_pass):
        trials.append({"round_type": "random", "round_index": i,
                       "status": "pass"})
    for i in range(random_inconclusive):
        trials.append({
            "round_type": "random", "round_index": random_pass + i,
            "status": "inconclusive",
            "reason": "strict state synchronization failed",
        })
    for i in range(stress_pass):
        trials.append({"round_type": "stress", "sub_index": i,
                       "status": "pass"})
    for i in range(stress_round_timeouts):
        trials.append({
            "round_type": "stress", "sub_index": stress_pass + i,
            "status": "inconclusive", "round_timeout": True,
            "reason": "round timed out after 90s watchdog; skipped to next round",
        })
    return trials


# ---------------------------------------------------------------------------
# Untouched (fully completed) paths keep the historical semantics.
# ---------------------------------------------------------------------------

def test_completed_pass_unchanged():
    result = {"validation_status": "pass",
              "trials": _trials(random_pass=EQUIV_RUNS, stress_pass=12)}
    status, grade, evidence = grade_equiv_evidence(result, timed_out=False)
    assert (status, grade) == ("pass", "LIKELY_EQUIVALENT")
    assert evidence is None


def test_completed_fail_unchanged():
    result = {"validation_status": "fail", "divergence": {"detail": "output_diverged"},
              "trials": _trials(random_pass=3)}
    status, grade, evidence = grade_equiv_evidence(result, timed_out=False)
    assert (status, grade) == ("fail", "WITNESSED_NON_EQUIVALENT")
    assert evidence is None


def test_completed_inconclusive_unchanged():
    result = {"validation_status": "inconclusive",
              "trials": _trials(random_pass=EQUIV_RUNS - 1,
                                random_inconclusive=1, stress_pass=12)}
    status, grade, evidence = grade_equiv_evidence(result, timed_out=False)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")
    assert evidence is None


def test_worker_crash_without_timeout_stays_inconclusive():
    status, grade, evidence = grade_equiv_evidence(None, timed_out=False)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")
    assert evidence is None


# ---------------------------------------------------------------------------
# Whole-probe timeout with a surviving partial snapshot.
# ---------------------------------------------------------------------------

def test_timeout_meeting_threshold_is_likely_equivalent():
    result = {
        "partial": True, "validation_status": "inconclusive",
        "trials": _trials(random_pass=EQUIV_RUNS,
                          stress_pass=EQUIV_MIN_STRESS_ROUNDS),
    }
    status, grade, evidence = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("pass", "LIKELY_EQUIVALENT")
    assert evidence["budget_exhausted"] is True
    assert evidence["partial_result"] is True
    assert evidence["rounds_completed"] == EQUIV_RUNS + EQUIV_MIN_STRESS_ROUNDS
    assert evidence["rounds_planned"] == EQUIV_RUNS + EQUIV_STRESS_ROUNDS_PLANNED
    assert evidence["random_rounds_passed"] == EQUIV_RUNS
    assert evidence["stress_rounds_passed"] == EQUIV_MIN_STRESS_ROUNDS


def test_timeout_below_stress_threshold_is_inconclusive():
    result = {
        "partial": True, "validation_status": "inconclusive",
        "trials": _trials(random_pass=EQUIV_RUNS,
                          stress_pass=EQUIV_MIN_STRESS_ROUNDS - 1),
    }
    status, grade, evidence = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")
    assert evidence["budget_exhausted"] is True
    assert evidence["stress_rounds_passed"] == EQUIV_MIN_STRESS_ROUNDS - 1


def test_timeout_with_incomplete_random_rounds_is_inconclusive():
    result = {
        "partial": True, "validation_status": "inconclusive",
        "trials": _trials(random_pass=EQUIV_RUNS - 1, stress_pass=12),
    }
    status, grade, _ = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")


def test_timeout_random_inconclusive_round_blocks_promotion():
    result = {
        "partial": True, "validation_status": "inconclusive",
        "trials": _trials(random_pass=EQUIV_RUNS - 1, random_inconclusive=1,
                          stress_pass=12),
    }
    status, grade, _ = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")


def test_timeout_with_witnessed_divergence_stays_witnessed():
    result = {
        "partial": True, "validation_status": "inconclusive",
        "divergence": {"detail": "output_diverged"},
        "trials": _trials(random_pass=5),
    }
    status, grade, evidence = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("fail", "WITNESSED_NON_EQUIVALENT")
    assert evidence["budget_exhausted"] is True


def test_timeout_with_final_fail_result_stays_witnessed():
    # Race: the worker finished writing a final FAIL result right as the
    # orchestrator's whole-probe timeout fired.
    result = {"validation_status": "fail",
              "divergence": {"detail": "candidate_crash"},
              "trials": _trials(random_pass=2)}
    status, grade, _ = grade_equiv_evidence(result, timed_out=True)
    assert (status, grade) == ("fail", "WITNESSED_NON_EQUIVALENT")


def test_timeout_without_any_result_is_inconclusive():
    status, grade, evidence = grade_equiv_evidence(None, timed_out=True)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")
    assert evidence["budget_exhausted"] is True
    assert evidence["rounds_completed"] == 0


# ---------------------------------------------------------------------------
# Watchdog round timeouts inside a completed worker.
# ---------------------------------------------------------------------------

def test_round_timeouts_with_threshold_met_promote_to_likely():
    result = {
        "validation_status": "inconclusive", "round_timeouts": 2,
        "trials": _trials(random_pass=EQUIV_RUNS, stress_pass=10,
                          stress_round_timeouts=2),
    }
    status, grade, evidence = grade_equiv_evidence(result, timed_out=False)
    assert (status, grade) == ("pass", "LIKELY_EQUIVALENT")
    assert evidence["budget_exhausted"] is False
    assert evidence["round_timeouts"] == 2
    assert evidence["stress_rounds_passed"] == 10


def test_round_timeout_trials_do_not_count_as_passes():
    result = {
        "validation_status": "inconclusive", "round_timeouts": 6,
        "trials": _trials(random_pass=EQUIV_RUNS, stress_pass=6,
                          stress_round_timeouts=6),
    }
    status, grade, evidence = grade_equiv_evidence(result, timed_out=False)
    assert (status, grade) == ("inconclusive", "INCONCLUSIVE")
    assert evidence["round_timeouts"] == 6
