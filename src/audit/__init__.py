"""Audit-mode analysis components (M7): RIPR escape classification and
FaultToStressMap construction.

Spec: MutakernelV2/MutakernelV2方法修正/方法V2_07.
"""

from .ripr import (  # noqa: F401
    ACTIVATION_FAILURE_VALUE,
    MASKING_FAILURE_PRECISION,
    REACHABILITY_FAILURE_MODE,
    REACHABILITY_FAILURE_CONFIG,
    OBSERVATION_FAILURE_NONDETERMINISM,
    ABSORPTION_FAILURE_TOLERANCE,
    classify_escape,
    dimension_of_case,
)
from .mapbuild import build_fault_to_stress_map  # noqa: F401
