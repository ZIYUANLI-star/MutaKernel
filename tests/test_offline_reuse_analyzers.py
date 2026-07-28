"""Core-logic tests for the offline round-log reuse analyzers (torch-free,
CPU-only, no remote access).

Pins the conservative-evidence semantics:
  * dedup keeps the latest record per probe;
  * witness extraction reads the single fail trial (or the divergence
    fallback for timeout-truncated logs);
  * closure lower bound counts a witness only when its recorded case is in
    the held-out top-k, and unclosed probes split into evidence-insufficient
    vs all-planned-conclusive;
  * upper bound (full map) dominates the lower bound;
  * sole-detector / cross-confirmation accounting under proxy and strict
    dimension mappings;
  * budget-recall replay and A13 per-round yield accounting;
  * cost percentile statistics.
"""

from __future__ import annotations

import pytest

from scripts.offline_reuse_lib import (
    BLUEPRINT_DIMENSIONS,
    POLICY_DIMENSION_PROXY,
    build_map_records,
    closure_evaluation,
    cost_stats,
    dedup_latest,
    dimension_of_policy_proxy,
    dimension_strict,
    executed_case_status,
    extract_witnesses,
    percentile,
    policy_round_stats,
    recall_curve,
    trial_policy_label,
    witness_budget_indices,
    witness_dimension_summary,
)
from src.audit.mapbuild import case_key


def _trial(status, policy=None, round_type=None, total_ms=None):
    trial = {
        "round_type": round_type or ("random" if policy is None else "stress"),
        "policy": policy,
        "status": status,
    }
    if total_ms is not None:
        trial["timings_ms"] = {"total_ms": total_ms}
    return trial


def _equiv_row(probe_id, operator, trials, grade, wall_ms=1000.0, **extra):
    return {
        "probe_id": probe_id,
        "kernel": probe_id.split("__", 1)[0],
        "operator_name": operator,
        "fault_class": None,
        "evidence_grade": grade,
        "trials": trials,
        "wall_ms": wall_ms,
        **extra,
    }


def _witnessed(probe_id, operator, witness_policy, passes_before=1,
               pass_policies=()):
    trials = [_trial("pass") for _ in range(passes_before)]
    trials += [_trial("pass", policy=p) for p in pass_policies]
    trials.append(_trial("fail", policy=witness_policy))
    return _equiv_row(probe_id, operator, trials, "WITNESSED_NON_EQUIVALENT")


def _likely(probe_id, operator, pass_policies):
    trials = [_trial("pass") for _ in range(2)]
    trials += [_trial("pass", policy=p) for p in pass_policies]
    return _equiv_row(probe_id, operator, trials, "LIKELY_EQUIVALENT")


# ---------------------------------------------------------------------------
# loading / dedup / witness extraction
# ---------------------------------------------------------------------------

def test_dedup_latest_keeps_newest_record():
    rows = [
        {"probe_id": "T1__op__0", "finished_at": "2026-07-24T00:00:00", "v": 1},
        {"probe_id": "T1__op__0", "finished_at": "2026-07-26T00:00:00", "v": 2},
        {"probe_id": "T2__op__0", "finished_at": "2026-07-25T00:00:00", "v": 3},
    ]
    unique, dropped = dedup_latest(rows)
    assert dropped == 1
    assert {r["probe_id"]: r["v"] for r in unique} == {"T1__op__0": 2, "T2__op__0": 3}


def test_extract_witnesses_reads_single_fail_trial():
    equiv = [
        _witnessed("T1__arith_replace__0", "arith_replace", "head_heavy",
                   passes_before=3),
        _likely("T2__arith_replace__0", "arith_replace", ["near_zero"]),
    ]
    cse = [{
        "probe_id": "T3__init_modify__2",
        "kernel": "T3",
        "operator_name": "init_modify",
        "fault_class": "F-INIT",
        "outcome": "FALSIFIED",
        "trials": [_trial("pass")] * 2 + [_trial("fail", policy="near_overflow")],
    }]
    witnesses = extract_witnesses(equiv, cse)
    assert set(witnesses) == {"T1__arith_replace__0", "T3__init_modify__2"}
    w1 = witnesses["T1__arith_replace__0"]
    assert w1["witness_policy_label"] == "head_heavy"
    assert w1["witness_round_1based"] == 4
    assert w1["source"] == "equiv"
    w3 = witnesses["T3__init_modify__2"]
    assert w3["source"] == "cse"
    assert w3["witness_round_1based"] == 3


def test_extract_witnesses_divergence_fallback_when_no_fail_trial():
    row = _equiv_row(
        "T9__arith_replace__0", "arith_replace",
        [_trial("pass")] * 2, "WITNESSED_NON_EQUIVALENT",
        divergence={"round_type": "stress", "policy": "sparse", "seed": 7})
    witnesses = extract_witnesses([row], [])
    wit = witnesses["T9__arith_replace__0"]
    assert wit["witness_policy_label"] == "sparse"
    assert wit["witness_case_key"] == case_key(
        {"policy": "sparse", "mode": "eval", "parameters": {}})


def test_random_rounds_collapse_to_iid_case_but_keep_random_label():
    random_trial = _trial("fail")  # round_type random, policy None
    assert trial_policy_label(random_trial) == "random"
    row = _equiv_row("T1__arith_replace__0", "arith_replace",
                     [random_trial], "WITNESSED_NON_EQUIVALENT")
    wit = extract_witnesses([row], [])["T1__arith_replace__0"]
    assert wit["witness_case_key"] == case_key(
        {"policy": "iid", "mode": "eval", "parameters": {}})


def test_executed_case_status_priority_kill_over_pass():
    rows = [_equiv_row(
        "T1__arith_replace__0", "arith_replace",
        [_trial("pass", policy="sparse"), _trial("fail", policy="sparse")],
        "WITNESSED_NON_EQUIVALENT")]
    status = executed_case_status(rows, [])
    key = case_key({"policy": "sparse", "mode": "eval", "parameters": {}})
    assert status["T1__arith_replace__0"][key] == "kill"


# ---------------------------------------------------------------------------
# closure evaluation (§5.6)
# ---------------------------------------------------------------------------

def _closure_fixture():
    """Six tasks, one operator, one dominant killing policy.

    head_heavy kills the witnessed probes of five tasks; the sixth task's
    probe is witnessed by an off-map policy that no training fold ever saw
    killing anything else, so cross-fitting cannot rank it early.
    """
    equiv = []
    for i in range(5):
        equiv.append(_witnessed(
            f"T{i}__arith_replace__0", "arith_replace", "head_heavy"))
    # witness by a policy never seen killing in any other task
    equiv.append(_witnessed(
        "T5__arith_replace__0", "arith_replace", "denormals",
        passes_before=2, pass_policies=["head_heavy"]))
    # equivalence ballast so folds always have training data
    for i in range(6):
        equiv.append(_likely(
            f"T{i}__arith_replace__1", "arith_replace",
            ["head_heavy", "denormals"]))
    return equiv


def test_closure_lower_bound_and_insufficiency_classification():
    equiv = _closure_fixture()
    witnesses = extract_witnesses(equiv, [])
    executed = executed_case_status(equiv, [])
    records = build_map_records(equiv, [], [])
    result = closure_evaluation(
        records, witnesses, executed, k_folds=3, planned_cases=2,
        curve_max_k=5)

    assert result["witnessed_total"] == 6
    by_probe = {r["probe_id"]: r for r in result["per_probe"]}
    # every head_heavy witness must be closed: head_heavy kills in other
    # folds, so the held-out ranking always contains it near the top
    for i in range(5):
        assert by_probe[f"T{i}__arith_replace__0"]["closed_at_planned_k"], i
    # the denormals witness cannot be closed by any training fold
    odd = by_probe["T5__arith_replace__0"]
    assert not odd["closed_at_planned_k"]
    # ...and it executed all its planned cases conclusively or not: with
    # planned = [head_heavy(pass), iid(pass?)] executed -> classification
    assert odd["classification"] in (
        "unclosed_all_planned_conclusive", "unclosed_evidence_insufficient")
    assert result["pooled_lower_bound"]["closed"] == 5

    # both curves are monotone in k; the full map can rank every witness
    # case (it saw every kill), so the upper-bound curve saturates at 1.0.
    # NOTE the non-cross-fitted curve is an upper bound in the circular-
    # evaluation sense, not pointwise at every k (small-sample closure-rate
    # ranking can reorder cases between fold maps and the full map).
    for curve in (result["closure_curve_lower"], result["closure_curve_upper"]):
        rates = [p["closure_rate"] for p in curve]
        assert rates == sorted(rates)
    assert result["closure_curve_upper"][-1]["closure_rate"] == 1.0
    # with the full map the denormals witness is rankable -> upper closes it
    assert odd["full_map_rank_of_witness"] is not None


def test_closure_marks_never_executed_planned_case_as_insufficient():
    equiv = _closure_fixture()
    # a witnessed probe that never executed head_heavy (killed by random
    # round 1 == iid case): planned top cases contain head_heavy, which was
    # never executed on it -> evidence-insufficient if unclosed
    row = _equiv_row("T6__arith_replace__0", "arith_replace",
                     [_trial("fail")], "WITNESSED_NON_EQUIVALENT")
    equiv.append(row)
    witnesses = extract_witnesses(equiv, [])
    executed = executed_case_status(equiv, [])
    records = build_map_records(equiv, [], [])
    result = closure_evaluation(
        records, witnesses, executed, k_folds=3, planned_cases=2,
        curve_max_k=5)
    target = next(r for r in result["per_probe"]
                  if r["probe_id"] == "T6__arith_replace__0")
    if not target["closed_at_planned_k"]:
        assert target["classification"] == "unclosed_evidence_insufficient"
        assert any(s["recorded"] == "not_executed"
                   for s in target["planned_case_status"])


def test_closure_requires_positive_folds_with_training_data():
    equiv = [_witnessed("T0__arith_replace__0", "arith_replace", "head_heavy")]
    witnesses = extract_witnesses(equiv, [])
    executed = executed_case_status(equiv, [])
    records = build_map_records(equiv, [], [])
    with pytest.raises(ValueError):
        closure_evaluation(records, witnesses, executed, k_folds=2)


# ---------------------------------------------------------------------------
# sole-detector / dimensions (Table 11)
# ---------------------------------------------------------------------------

def test_policy_dimension_proxy_covers_all_21_policies_plus_rounds():
    from src.stress.policy_metadata import POLICY_TARGET_FAULT_CLASSES
    for policy in POLICY_TARGET_FAULT_CLASSES:
        assert policy in POLICY_DIMENSION_PROXY, policy
        assert (POLICY_DIMENSION_PROXY[policy]["dimension"]
                in BLUEPRINT_DIMENSIONS)
    assert dimension_of_policy_proxy("random") == "value"
    assert dimension_of_policy_proxy("iid") == "value"
    assert dimension_of_policy_proxy("head_heavy") == "configuration"
    assert dimension_of_policy_proxy("near_overflow") == "dtype"
    assert dimension_of_policy_proxy("reduction_adversarial") == "repetition"
    assert dimension_of_policy_proxy("init_sensitive") == "training"


def test_dimension_strict_is_value_for_eval_fp32_cases():
    assert dimension_strict(
        {"policy": "head_heavy", "mode": "eval", "parameters": {}}) == "value"
    assert dimension_strict(
        {"policy": "x", "mode": "train", "parameters": {}}) == "training"
    assert dimension_strict(
        {"policy": "x", "mode": "repeated", "parameters": {}}) == "repetition"
    assert dimension_strict(
        {"policy": "x", "mode": "config", "parameters": {}}) == "configuration"


def test_witness_dimension_summary_sole_and_cross_rates():
    equiv = [
        _witnessed("T1__arith_replace__0", "arith_replace", "head_heavy"),
        _witnessed("T2__arith_replace__0", "arith_replace", "near_overflow"),
        _witnessed("T3__arith_replace__0", "arith_replace", "large_magnitude"),
    ]
    summary = witness_dimension_summary(extract_witnesses(equiv, []))
    proxy = summary["proxy"]
    assert proxy["defects_total"] == 3
    assert proxy["sole_detector_defects"]["configuration"] == 1
    assert proxy["sole_detector_defects"]["dtype"] == 1
    assert proxy["sole_detector_defects"]["value"] == 1
    # single-witness logs -> cross-confirmation is a 0 lower bound
    assert proxy["cross_confirmed_ge2_dims"] == 0
    strict = summary["strict"]
    assert strict["sole_detector_defects"]["value"] == 3


# ---------------------------------------------------------------------------
# budget-recall + A13
# ---------------------------------------------------------------------------

def test_witness_budget_indices_charge_cse_with_equiv_budget():
    equiv = [
        _witnessed("T1__arith_replace__0", "arith_replace", "head_heavy",
                   passes_before=4),  # witness at round 5
        _likely("T2__arith_replace__0", "arith_replace",
                ["near_zero"] * 3),   # 5 trials, then CSE witness at round 2
    ]
    cse = [{
        "probe_id": "T2__arith_replace__0",
        "kernel": "T2",
        "operator_name": "arith_replace",
        "outcome": "FALSIFIED",
        "trials": [_trial("pass"), _trial("fail", policy="near_overflow")],
    }]
    indices = witness_budget_indices(extract_witnesses(equiv, cse), equiv)
    by_id = {i["probe_id"]: i for i in indices}
    assert by_id["T1__arith_replace__0"]["phase_round_index"] == 5
    assert by_id["T1__arith_replace__0"]["combined_round_index"] == 5
    assert by_id["T2__arith_replace__0"]["phase_round_index"] == 2
    assert by_id["T2__arith_replace__0"]["combined_round_index"] == 7  # 5 + 2


def test_recall_curve_monotone_and_saturating():
    curve = recall_curve([1, 3, 3, None], max_budget=4)
    assert [p["witnesses"] for p in curve] == [1, 1, 3, 3]
    assert curve[-1]["recall"] == 0.75  # the None witness never lands
    rates = [p["recall"] for p in curve]
    assert rates == sorted(rates)


def test_policy_round_stats_dual_accounting():
    rows = [
        _equiv_row(
            "T1__arith_replace__0", "arith_replace",
            [_trial("pass", total_ms=100.0),               # random
             _trial("pass", total_ms=100.0),               # random
             _trial("fail", policy="head_heavy", total_ms=300.0)],
            "WITNESSED_NON_EQUIVALENT"),
        _equiv_row(
            "T2__arith_replace__0", "arith_replace",
            [_trial("pass", total_ms=100.0),               # random
             _trial("pass", policy="head_heavy", total_ms=300.0),
             _trial("inconclusive", policy="sparse", total_ms=100.0)],
            "LIKELY_EQUIVALENT"),
    ]
    stats = {s["policy"]: s for s in policy_round_stats(rows, "equiv")}
    assert stats["random"]["rounds"] == 3
    assert stats["random"]["witnesses"] == 0
    assert stats["head_heavy"]["rounds"] == 2
    assert stats["head_heavy"]["witnesses"] == 1
    assert stats["head_heavy"]["hit_rate_per_round"] == 0.5
    # inconclusive rounds stay in the budget denominator
    assert stats["sparse"]["rounds"] == 1
    assert stats["sparse"]["conclusive_rounds"] == 0
    total_rounds = sum(s["rounds"] for s in stats.values())
    assert total_rounds == 6
    assert stats["head_heavy"]["budget_share_rounds"] == pytest.approx(2 / 6)
    assert stats["head_heavy"]["budget_share_wall_ms"] == pytest.approx(
        600.0 / 1000.0)


# ---------------------------------------------------------------------------
# cost stats
# ---------------------------------------------------------------------------

def test_percentile_linear_interpolation():
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile(list(map(float, range(1, 101))), 0.95) == pytest.approx(95.05)


def test_cost_stats_totals():
    stats = cost_stats([1000.0, 2000.0, 3000.0])
    assert stats["n"] == 3
    assert stats["median_ms"] == 2000.0
    assert stats["total_ms"] == 6000.0
    # 6 seconds == 6000 ms == 6000 / 3.6e6 hours
    assert stats["total_gpu_hours"] == pytest.approx(6000.0 / 3.6e6, rel=1e-3)
    assert cost_stats([]) == {"n": 0}
