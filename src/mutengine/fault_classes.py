"""Machine-readable operator-to-fault-class taxonomy (MutaKernel V2, torch-free).

Authoritative definition: MutakernelV2/MutakernelV2方法修正/方法V2_03 §3.1.
The mapping is intentionally 1:1 so that fault classes inherit the operators'
construct-validity evidence from RealismGuard without an extra indirection.
"""

from __future__ import annotations

from typing import Dict, Tuple

TAXONOMY_VERSION = "2.0"

OPERATOR_TO_FAULT_CLASS: Dict[str, str] = {
    # Category A: classical arithmetic
    "arith_replace": "F-ARITH",
    "relop_replace": "F-RELOP",
    "const_perturb": "F-CONST",
    # Category B: GPU parallel semantics
    "index_replace": "F-IDX",
    "sync_remove": "F-SYNC",
    "mask_boundary": "F-BOUND",
    "launch_config_mutate": "F-LAUNCH",
    # Category C: ML numerical semantics
    "stab_remove": "F-STAB",
    "epsilon_modify": "F-EPS",
    "scale_modify": "F-SCALE",
    "init_modify": "F-INIT",
    "acc_downgrade": "F-PREC-ACC",
    "cast_remove": "F-CAST",
    "reduction_reorder": "F-RED-ORD",
    # Category D: LLM-specific patterns
    "broadcast_unsafe": "F-BCAST",
    "layout_assume": "F-LAYOUT",
}

FAULT_CLASS_TO_OPERATOR: Dict[str, str] = {
    fault: operator for operator, fault in OPERATOR_TO_FAULT_CLASS.items()
}

ALL_FAULT_CLASSES: Tuple[str, ...] = tuple(sorted(FAULT_CLASS_TO_OPERATOR))

# ``node_type`` substrings that mark sites which are equivalent by
# construction (see V1 operator design; e.g. a reduction-tail __syncthreads or
# a redundant static_cast).  Such sites carry no blind-spot information and
# are excluded from ``fault_classes_present`` in the site fingerprint.
PRIOR_EQUIVALENT_NODE_TYPE_MARKERS: Tuple[str, ...] = (
    "reduction_tail",
    ":redundant",
)


def is_prior_equivalent_node_type(node_type: str) -> bool:
    text = node_type or ""
    return any(marker in text for marker in PRIOR_EQUIVALENT_NODE_TYPE_MARKERS)
