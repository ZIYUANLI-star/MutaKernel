"""解析器单元测试。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from src.mutengine.parser.triton_parser import TritonParser
from src.mutengine.parser.cuda_parser import CudaParser


TRITON_SOURCE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        output = torch.empty_like(x)
        n = output.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        return output
"""

CUDA_SOURCE = '''\
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                     out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);"
'''


class TestTritonParser(unittest.TestCase):
    def test_parse(self):
        parser = TritonParser()
        result = parser.parse(TRITON_SOURCE)
        self.assertTrue(result.is_triton)
        self.assertTrue(result.has_kernels)
        self.assertEqual(len(result.kernels), 1)
        self.assertEqual(result.kernels[0].name, "add_kernel")

    def test_wrapper_class(self):
        parser = TritonParser()
        result = parser.parse(TRITON_SOURCE)
        self.assertEqual(result.wrapper_class, "ModelNew")

    def test_kernel_lines(self):
        parser = TritonParser()
        result = parser.parse(TRITON_SOURCE)
        k = result.kernels[0]
        self.assertGreater(k.end_line, k.start_line)
        self.assertIn("tl.program_id", k.source)

    def test_non_triton(self):
        parser = TritonParser()
        result = parser.parse("import torch\nx = torch.randn(10)\n")
        self.assertFalse(result.has_kernels)

    def test_extract_mutatable(self):
        parser = TritonParser()
        source, tree = parser.extract_mutatable_source(TRITON_SOURCE)
        self.assertEqual(source, TRITON_SOURCE)
        self.assertIsNotNone(tree)


class TestCudaParser(unittest.TestCase):
    def test_parse(self):
        parser = CudaParser()
        result = parser.parse(CUDA_SOURCE)
        self.assertTrue(result.is_cuda)
        self.assertTrue(result.has_kernels)

    def test_kernel_extraction(self):
        parser = CudaParser()
        result = parser.parse(CUDA_SOURCE)
        self.assertGreater(len(result.cuda_blocks), 0)
        block = result.cuda_blocks[0]
        self.assertGreater(len(block.kernels), 0)
        self.assertEqual(block.kernels[0].name, "add_kernel")

    def test_non_cuda(self):
        parser = CudaParser()
        result = parser.parse("import torch\nx = torch.randn(10)\n")
        self.assertFalse(result.is_cuda)


if __name__ == "__main__":
    unittest.main()
