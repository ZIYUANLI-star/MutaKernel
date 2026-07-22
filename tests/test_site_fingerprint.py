"""Tests for M3 fault-class taxonomy and static site fingerprinting (torch-free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mutengine.fault_classes import (
    ALL_FAULT_CLASSES,
    FAULT_CLASS_TO_OPERATOR,
    OPERATOR_TO_FAULT_CLASS,
    is_prior_equivalent_node_type,
)
from src.mutengine.fingerprint import build_site_fingerprint, fingerprint_version
from src.mutengine.operators import get_all_operators

KERNEL_WITH_EPS_AND_SYNC = '''
import torch
import torch.nn as nn

cuda_src = """
__global__ void norm_kernel(const float* x, float* out, int n) {
    __shared__ float sdata[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { sdata[threadIdx.x] = x[i]; }
    __syncthreads();
    float var = sdata[threadIdx.x];
    out[i] = rsqrtf(var + 1e-5f);
}
"""

class ModelNew(nn.Module):
    def forward(self, x):
        eps = 1e-5
        return x * eps
'''

PLAIN_SOURCE = '''
def add(a, b):
    return a
'''


def test_taxonomy_is_one_to_one_and_covers_all_operators():
    operator_names = {op.name for op in get_all_operators()}
    assert set(OPERATOR_TO_FAULT_CLASS) == operator_names
    assert len(FAULT_CLASS_TO_OPERATOR) == len(OPERATOR_TO_FAULT_CLASS)
    assert len(ALL_FAULT_CLASSES) == 16


def test_prior_equivalent_markers():
    assert is_prior_equivalent_node_type("cuda_syncthreads:reduction_tail")
    assert is_prior_equivalent_node_type("cast:cuda_static_cast:redundant")
    assert not is_prior_equivalent_node_type("cuda_syncthreads")
    assert not is_prior_equivalent_node_type("eps_sci")


def test_fingerprint_detects_expected_fault_classes():
    fp = build_site_fingerprint(KERNEL_WITH_EPS_AND_SYNC, subject_id="s1")
    assert fp["subject_id"] == "s1"
    present = set(fp["fault_classes_present"])
    assert "F-EPS" in present       # 1e-5 epsilon sites
    assert "F-SYNC" in present      # non-reduction-tail __syncthreads
    assert "F-SCALE" in present     # rsqrtf(...)
    assert fp["sites"]["epsilon_modify"]["count"] >= 1
    # every registered operator has an entry, even with zero sites
    assert set(fp["sites"]) == {op.name for op in get_all_operators()}


def test_fingerprint_is_deterministic_and_version_stable():
    fp1 = build_site_fingerprint(KERNEL_WITH_EPS_AND_SYNC)
    fp2 = build_site_fingerprint(KERNEL_WITH_EPS_AND_SYNC)
    assert fp1 == fp2
    assert fp1["fingerprint_version"] == fingerprint_version()
    assert len(fingerprint_version()) == 16


def test_fingerprint_on_plain_source_reports_empty_presence():
    fp = build_site_fingerprint(PLAIN_SOURCE)
    assert fp["fault_classes_present"] == []


def test_fingerprint_survives_syntax_errors():
    fp = build_site_fingerprint("def broken(:\n    pass")
    assert "fault_classes_present" in fp  # must not raise
