"""Complete rich contract shared by experiment-protocol unit tests."""

from __future__ import annotations

import copy

from src.stress.policy_bank import STRESS_POLICIES


def rich_contract() -> dict:
    contract = {
        "schema_version": "1.0",
        "contract_id": "unit-rich-contract-v1",
        "tensor_inputs": [
            {
                "arg_index": 0,
                "dtypes": ["float16", "bfloat16", "float32"],
                "shape": [{"name": "batch", "min": 1, "max": 64}, 3],
                "value_domain": {"kind": "unrestricted_finite"},
                "layouts": ["contiguous", "noncontiguous"],
                "requires_grad": False,
                "aliases": [],
            }
        ],
        "execution": {
            "modes": ["eval", "train", "config", "repeated"],
            "backward": True,
            "deterministic": True,
            "repeat_count_max": 2,
            "streams": ["default"],
            "compare_input_side_effects": True,
            "compare_module_state": True,
            "backward_vjp_count": 3,
        },
        "oracle": {
            "atol": 0.0,
            "rtol": 0.0,
            "equal_nan": True,
            "require_dtype": True,
            "require_device": True,
            "require_layout": True,
            "require_stride": False,
            "require_aliasing": False,
            "dtype_tolerances": {},
        },
        "policy_bindings": {
            policy: [0] for policy in sorted(STRESS_POLICIES)
        },
        "input_adapters": {
            "batch": {
                "arg_indices": [0],
                "dimension": 0,
                "allowed_values": [1, 4, 16, 64],
            },
            "layout": {
                "arg_indices": [0],
                "allowed_values": ["noncontiguous"],
            },
            "dtype": {"arg_indices": [0]},
        },
        "candidate_classes": ["ModelNew", "Model"],
    }
    return copy.deepcopy(contract)
