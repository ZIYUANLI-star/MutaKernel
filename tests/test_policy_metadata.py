"""Consistency tests for torch-free policy metadata (M2)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mutengine.fault_classes import ALL_FAULT_CLASSES
from src.stress.policy_metadata import (
    EXECUTION_CONTEXT_TARGET_FAULT_CLASSES,
    POLICY_TARGET_FAULT_CLASSES,
    all_value_policy_names,
)


def test_metadata_references_only_known_fault_classes():
    known = set(ALL_FAULT_CLASSES)
    for policy, faults in POLICY_TARGET_FAULT_CLASSES.items():
        unknown = set(faults) - known
        assert not unknown, f"{policy} targets unknown fault classes: {unknown}"
    for context, faults in EXECUTION_CONTEXT_TARGET_FAULT_CLASSES.items():
        unknown = set(faults) - known
        assert not unknown, f"{context} targets unknown fault classes: {unknown}"


def test_every_fault_class_is_targeted_by_something():
    targeted = set()
    for faults in POLICY_TARGET_FAULT_CLASSES.values():
        targeted.update(faults)
    for faults in EXECUTION_CONTEXT_TARGET_FAULT_CLASSES.values():
        targeted.update(faults)
    assert targeted == set(ALL_FAULT_CLASSES)


def test_metadata_matches_executable_policy_registry():
    torch = pytest.importorskip("torch")  # noqa: F841 - registry needs torch
    from src.stress.policy_bank import get_all_policy_names

    assert set(all_value_policy_names()) == set(get_all_policy_names())
