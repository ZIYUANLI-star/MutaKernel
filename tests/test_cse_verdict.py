"""Tests for the five-valued three-way verdict semantics (M6, torch-free)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cse.verdict import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    VERDICT_ACCIDENTAL_REPAIR,
    VERDICT_EXACT_DIVERGENCE_ONLY,
    VERDICT_INCONCLUSIVE,
    VERDICT_INDISTINGUISHED,
    VERDICT_INVALID_INPUT,
    VERDICT_SPEC_VIOLATION,
    three_way_verdict,
    two_way_verdict,
    verdict_from_legacy_stress_record,
)


def test_five_row_table():
    assert three_way_verdict(PASS, FAIL, False) == VERDICT_SPEC_VIOLATION
    assert three_way_verdict(PASS, FAIL, None) == VERDICT_SPEC_VIOLATION
    assert three_way_verdict(PASS, PASS, False) == VERDICT_EXACT_DIVERGENCE_ONLY
    assert three_way_verdict(PASS, PASS, True) == VERDICT_INDISTINGUISHED
    assert three_way_verdict(FAIL, FAIL, None) == VERDICT_INVALID_INPUT
    assert three_way_verdict(FAIL, PASS, None) == VERDICT_ACCIDENTAL_REPAIR


def test_inconclusive_dominates():
    assert three_way_verdict(INCONCLUSIVE, FAIL) == VERDICT_INCONCLUSIVE
    assert three_way_verdict(PASS, INCONCLUSIVE) == VERDICT_INCONCLUSIVE


def test_double_pass_without_exact_information_fails_closed():
    assert three_way_verdict(PASS, PASS, None) == VERDICT_INCONCLUSIVE


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        three_way_verdict("yes", PASS)


def test_two_way_degenerate_form():
    assert two_way_verdict(FAIL) == VERDICT_SPEC_VIOLATION
    assert two_way_verdict(PASS) == VERDICT_INDISTINGUISHED
    assert two_way_verdict(INCONCLUSIVE) == VERDICT_INCONCLUSIVE


def test_legacy_bitwise_only_kill_is_reclassified_not_a_violation():
    # V1 killed on "allclose passes but bitwise differs"; V2 reclassifies.
    record = {
        "ref_ok": True,
        "original_ok": True,
        "mutant_ok": True,
        "observed_bitwise_orig_mut_eq": False,
    }
    assert verdict_from_legacy_stress_record(record) == VERDICT_EXACT_DIVERGENCE_ONLY


def test_legacy_spec_violation_and_invalid_rounds():
    assert verdict_from_legacy_stress_record(
        {"ref_ok": True, "original_ok": True, "mutant_ok": False}
    ) == VERDICT_SPEC_VIOLATION
    assert verdict_from_legacy_stress_record(
        {"ref_ok": True, "original_ok": False, "mutant_ok": False}
    ) == VERDICT_INVALID_INPUT
    assert verdict_from_legacy_stress_record(
        {"ref_ok": False, "original_ok": False, "mutant_ok": False}
    ) == VERDICT_INCONCLUSIVE
    assert verdict_from_legacy_stress_record(
        {"ref_ok": True, "original_ok": True, "mutant_ok": True,
         "validation_status": "inconclusive"}
    ) == VERDICT_INCONCLUSIVE
