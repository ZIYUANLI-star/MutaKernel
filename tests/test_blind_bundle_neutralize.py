"""Tests for the policy-neutral blinding of audit evidence (M9, torch-free)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_blind_bundles import neutral_id, neutralize_case_config

CASE_CONFIG = {
    "subject_id": "subj-1",
    "reference_path": "refs/task_1.py",
    "candidate_path": "cands/task_1_cand.py",
    "device": "cuda",
    "contract": {"oracle": {"atol": 1e-2, "rtol": 1e-2}},
    "case": {
        "test_id": "abc123",
        "policy": "near_zero",
        "seed": 42,
        "scope": "in_contract",
        "mode": "train",
        "strategy_name": "mutakernel-full",
        "parameters": {
            "dtype": "float16",
            "batch_size": 4,
            "policy_arg_indices": [0],
            "dtype_arg_indices": [0],
        },
    },
}


def test_sensitive_fields_removed_and_sealed():
    blind, sealed = neutralize_case_config(CASE_CONFIG, salt="s3cret")
    text = json.dumps(blind)
    assert "near_zero" not in text
    assert "strategy" not in text
    assert '"seed"' not in text
    assert "policy_arg_indices" not in text
    # sealed side is lossless
    assert sealed["policy"] == "near_zero"
    assert sealed["seed"] == 42
    assert sealed["strategy_name"] == "mutakernel-full"
    assert sealed["test_id"] == "abc123"
    assert sealed["parameters"]["policy_arg_indices"] == [0]


def test_neutral_execution_context_is_preserved():
    blind, _ = neutralize_case_config(CASE_CONFIG, salt="s3cret")
    ctx = blind["execution_context"]
    assert ctx["mode"] == "train"
    assert ctx["dtype"] == "float16"
    assert ctx["batch_size"] == 4
    # contract stays available for in-contract judgement
    assert blind["contract"]["oracle"]["atol"] == 1e-2


def test_neutral_id_is_salted_and_stable():
    blind1, _ = neutralize_case_config(CASE_CONFIG, salt="s3cret")
    blind2, _ = neutralize_case_config(CASE_CONFIG, salt="s3cret")
    blind3, _ = neutralize_case_config(CASE_CONFIG, salt="another")
    assert blind1["neutral_id"] == blind2["neutral_id"]
    assert blind1["neutral_id"] != blind3["neutral_id"]
    assert blind1["neutral_id"] == neutral_id("abc123", "s3cret")
    assert "abc123" not in blind1["neutral_id"]


def test_missing_test_id_is_rejected():
    config = json.loads(json.dumps(CASE_CONFIG))
    del config["case"]["test_id"]
    with pytest.raises(ValueError):
        neutralize_case_config(config, salt="s")
