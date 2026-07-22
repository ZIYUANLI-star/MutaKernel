"""Torch-free metadata for the stress-policy library (M2, MutaKernel V2).

Keeps the policy -> target-fault-class prior importable without torch so that
the online site-directed selector (方法V2_08) and the offline RIPR classifier
(方法V2_07) can run in lightweight environments.

Consistency with the executable policy registry
(``src.stress.policy_bank.STRESS_POLICIES``) is enforced by a regression test
that runs wherever torch is available.

Source of the prior: MutakernelV2/MutakernelV2方法修正/方法V2_02 §4.2.
The prior is a design-time hypothesis; the audited FaultToStressMap
(方法V2_07 §3.5) supersedes it with measured closure rates.
"""

from __future__ import annotations

from typing import Dict, Tuple

POLICY_METADATA_VERSION = "2.0"

# Value-distribution policies (applied by value_stress / training_stress).
POLICY_TARGET_FAULT_CLASSES: Dict[str, Tuple[str, ...]] = {
    # family 1: extreme value distributions
    "large_magnitude": ("F-ARITH", "F-STAB", "F-SYNC"),
    "extreme_magnitude": ("F-STAB", "F-CAST"),
    "near_overflow": ("F-STAB", "F-CAST"),
    "near_zero": ("F-EPS", "F-SCALE", "F-CONST"),
    "denormals": ("F-EPS",),
    "near_epsilon": ("F-EPS",),
    "mixed_extremes": ("F-PREC-ACC", "F-CAST", "F-RED-ORD"),
    # family 2: boundary values and sign
    "all_negative": ("F-INIT",),
    "all_positive": ("F-INIT", "F-STAB"),
    "boundary_last_element": ("F-BOUND", "F-LAUNCH", "F-RELOP"),
    "relop_boundary_hit": ("F-RELOP",),
    # family 3: sparsity gradient
    "dense_nonzero": ("F-ARITH",),
    "sparse": ("F-BOUND", "F-IDX"),
    "sparse_extreme": ("F-BOUND", "F-RELOP"),
    # family 4: structured / position-sensitive
    "structured_ramp": ("F-IDX", "F-BCAST", "F-LAYOUT", "F-LAUNCH"),
    "head_heavy": ("F-IDX",),
    "tail_heavy": ("F-IDX", "F-BOUND"),
    # family 5: reduction adversarial / special behaviour
    "alternating_sign": ("F-RED-ORD",),
    "reduction_adversarial": ("F-RED-ORD", "F-PREC-ACC"),
    "uniform_constant": ("F-SCALE",),
    "init_sensitive": ("F-INIT",),
}

# Execution-context transforms (case ``mode`` / parameters, not value
# policies; see 方法V2_02 §4.3) and the fault classes they primarily target.
EXECUTION_CONTEXT_TARGET_FAULT_CLASSES: Dict[str, Tuple[str, ...]] = {
    "dtype": ("F-PREC-ACC", "F-CAST"),
    "train": ("F-EPS", "F-SCALE", "F-INIT", "F-CONST", "F-ARITH", "F-CAST"),
    "repeated": ("F-SYNC",),
    "config": ("F-LAUNCH", "F-BOUND", "F-IDX"),
    "layout": ("F-LAYOUT",),
}


def all_value_policy_names() -> Tuple[str, ...]:
    return tuple(sorted(POLICY_TARGET_FAULT_CLASSES))
