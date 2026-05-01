#!/usr/bin/env python3
"""Verify _patch_load_inline_pybind_conflict works correctly."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mutengine.mutant_runner import _patch_load_inline_pybind_conflict

# Test case: kernel with PYBIND11_MODULE in cuda_sources AND functions= kwarg
test_source = '''import torch
from torch.utils.cpp_extension import load_inline

_cuda_source = r"""
torch::Tensor swish_forward(torch::Tensor x) {
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish forward");
}"""

_ext = load_inline(
    name="test_kernel",
    cpp_sources="",
    cuda_sources=_cuda_source,
    functions=['forward'],
    verbose=False,
)

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return _ext.forward(x)
'''

patched = _patch_load_inline_pybind_conflict(test_source)

has_functions_before = "functions=['forward']" in test_source
has_functions_after = "functions=" in patched

print(f"Before patch - has functions=: {has_functions_before}")
print(f"After patch  - has functions=: {has_functions_after}")
print()

if has_functions_before and not has_functions_after:
    print("PASS: functions= kwarg correctly removed")
else:
    print("FAIL: patch did not work")
    print("--- Patched source ---")
    print(patched)
    print("--- End ---")

# Test case 2: kernel WITHOUT PYBIND11_MODULE (should not be modified)
test_source_no_pybind = '''import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
torch::Tensor forward(torch::Tensor x) {
    return x;
}"""

_ext = load_inline(
    name="test2",
    cpp_sources="",
    cuda_sources=_cuda_src,
    functions=['forward'],
    verbose=False,
)
'''

patched2 = _patch_load_inline_pybind_conflict(test_source_no_pybind)
has_functions_after2 = "functions=['forward']" in patched2
print(f"\nNo PYBIND case - functions= preserved: {has_functions_after2}")
if has_functions_after2:
    print("PASS: functions= correctly preserved when no PYBIND11_MODULE")
else:
    print("FAIL: functions= incorrectly removed")
