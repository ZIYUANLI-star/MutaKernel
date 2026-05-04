#!/usr/bin/env python3
"""Quick test of how load_inline handles code with embedded PYBIND11_MODULE.

We try two modes and compare:
  Mode A: pass `functions=['forward']`  (current behavior)
  Mode B: don't pass `functions=`        (rely on embedded PYBIND11_MODULE)
"""
from __future__ import annotations
import os, sys, tempfile, traceback
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mul2_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 2.0f;
}

torch::Tensor forward(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul2_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "x*2 (CUDA)");
}
"""

def try_mode(name, kwargs):
    print(f"\n--- Mode: {name} ---")
    try:
        ext = load_inline(
            name=f"_test_{name}",
            cpp_sources="",
            cuda_sources=CUDA_SRC,
            verbose=False,
            extra_cuda_cflags=["-O2"],
            **kwargs,
        )
        x = torch.arange(8, dtype=torch.float32, device="cuda")
        y = ext.forward(x)
        ok = torch.allclose(y, x * 2)
        print(f"  compiled: yes, output ok: {ok}")
        return True
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False

try_mode("with_functions", {"functions": ["forward"]})
try_mode("without_functions", {})
