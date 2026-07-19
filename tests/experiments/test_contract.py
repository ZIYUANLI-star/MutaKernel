import copy

import pytest
import torch

from src.experiments.contract import (
    ContractError,
    assert_case_in_contract,
    validate_call_inputs,
    validate_contract,
)
from tests.experiments.contract_fixture import rich_contract


def test_complete_contract_is_canonical_and_round_trips():
    contract = rich_contract()

    normalized = validate_contract(contract)

    assert normalized == contract
    assert validate_contract(normalized) == normalized


def test_missing_oracle_semantics_fail_closed():
    contract = rich_contract()
    del contract["oracle"]["equal_nan"]

    with pytest.raises(ContractError, match="every oracle field"):
        validate_contract(contract)


def test_unbound_value_policy_cannot_be_called_in_contract():
    contract = rich_contract()
    del contract["policy_bindings"]["near_zero"]

    with pytest.raises(ContractError, match="no argument binding"):
        assert_case_in_contract(
            {"policy": "near_zero", "mode": "eval", "parameters": {}},
            contract,
        )


def test_unsupported_dtype_cannot_be_called_in_contract():
    contract = rich_contract()

    with pytest.raises(ContractError, match="outside tensor argument"):
        assert_case_in_contract(
            {
                "policy": "iid",
                "mode": "eval",
                "parameters": {"dtype": "float64"},
            },
            contract,
        )


def test_planner_inputs_are_not_mutated_by_validation():
    contract = rich_contract()
    before = copy.deepcopy(contract)

    validate_contract(contract)

    assert contract == before


def test_runtime_input_shape_and_domain_are_enforced():
    contract = rich_contract()
    validate_call_inputs((torch.ones(2, 3),), contract)

    with pytest.raises(ContractError, match="dimension"):
        validate_call_inputs((torch.ones(2, 4),), contract)
    with pytest.raises(ContractError, match="non-finite"):
        validate_call_inputs((torch.full((2, 3), float("nan")),), contract)


def test_sparse_coo_layout_uses_the_runtime_layout_name():
    contract = rich_contract()
    contract["tensor_inputs"][0]["layouts"] = ["sparse_coo"]
    contract["policy_bindings"] = {}
    contract["input_adapters"] = {}
    normalized = validate_contract(contract)

    validate_call_inputs((torch.ones(2, 3).to_sparse(),), normalized)


def test_value_policy_binding_rejects_integer_or_complex_noop_targets():
    for dtype in ("int64", "complex64"):
        contract = rich_contract()
        contract["tensor_inputs"][0]["dtypes"] = [dtype]
        with pytest.raises(ContractError, match="silent no-op stress"):
            validate_contract(contract)


def test_case_parameter_typo_and_unimplemented_dtype_fail_closed():
    contract = rich_contract()
    with pytest.raises(ContractError, match="unknown fields"):
        assert_case_in_contract(
            {
                "policy": "iid",
                "mode": "train",
                "parameters": {"requires_backwards": True},
            },
            contract,
        )
    contract["tensor_inputs"][0]["dtypes"].append("complex64")
    with pytest.raises(ContractError, match="not executable"):
        assert_case_in_contract(
            {
                "policy": "iid",
                "mode": "eval",
                "parameters": {"dtype": "complex64"},
            },
            contract,
        )
