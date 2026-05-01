"""Verify that A-category operators find sites in CUDA C++ kernel code.

Run:  python -m scripts.verify_a_operators
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_CUDA_KERNEL = r'''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && (t * TILE_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_SIZE + ty) < N && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    return C;
}
"""

matmul_cpp_source = "torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B);"

N = 2048

def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]
'''

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb

def main():
    source = SAMPLE_CUDA_KERNEL

    print("=" * 70)
    print("A1 ArithReplace")
    print("=" * 70)
    op = ArithReplace()
    sites = op.find_sites(source)
    ast_count = sum(1 for s in sites if not s.node_type.startswith("cuda_"))
    cuda_count = sum(1 for s in sites if s.node_type.startswith("cuda_"))
    print(f"  Total sites: {len(sites)}  (AST: {ast_count}, CUDA: {cuda_count})")
    for s in sites:
        line = source.splitlines()[s.line_start - 1].rstrip()
        tag = "CUDA" if s.node_type.startswith("cuda_") else " AST"
        print(f"  [{tag}] L{s.line_start:3d} col={s.col_start:2d}  '{s.original_code}'  "
              f"type={s.node_type:<12s}  {line}")

    print()
    print("=" * 70)
    print("A2 RelOpReplace")
    print("=" * 70)
    op2 = RelOpReplace()
    sites2 = op2.find_sites(source)
    ast_count2 = sum(1 for s in sites2 if not s.node_type.startswith("cuda_"))
    cuda_count2 = sum(1 for s in sites2 if s.node_type.startswith("cuda_"))
    print(f"  Total sites: {len(sites2)}  (AST: {ast_count2}, CUDA: {cuda_count2})")
    for s in sites2:
        line = source.splitlines()[s.line_start - 1].rstrip()
        tag = "CUDA" if s.node_type.startswith("cuda_") else " AST"
        print(f"  [{tag}] L{s.line_start:3d} col={s.col_start:2d}  '{s.original_code}'  "
              f"type={s.node_type:<12s}  {line}")

    print()
    print("=" * 70)
    print("A3 ConstPerturb (sample)")
    print("=" * 70)
    op3 = ConstPerturb()
    sites3 = op3.find_sites(source)
    print(f"  Total sites: {len(sites3)}")
    for s in sites3[:20]:
        line = source.splitlines()[s.line_start - 1].rstrip()
        print(f"  L{s.line_start:3d}  '{s.original_code}'  type={s.node_type}  {line}")
    if len(sites3) > 20:
        print(f"  ... and {len(sites3) - 20} more")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  A1 ArithReplace:  {len(sites):3d} sites  (AST={ast_count}, CUDA={cuda_count})")
    print(f"  A2 RelOpReplace:  {len(sites2):3d} sites  (AST={ast_count2}, CUDA={cuda_count2})")
    print(f"  A3 ConstPerturb:  {len(sites3):3d} sites")

    # Quick apply test
    print()
    print("=" * 70)
    print("APPLY TESTS")
    print("=" * 70)
    cuda_arith = [s for s in sites if s.node_type.startswith("cuda_")]
    if cuda_arith:
        s = cuda_arith[0]
        mutated = op.apply(source, s)
        orig_line = source.splitlines()[s.line_start - 1]
        new_line = mutated.splitlines()[s.line_start - 1]
        print(f"  ArithReplace CUDA site L{s.line_start}: '{s.original_code}' →")
        print(f"    BEFORE: {orig_line.rstrip()}")
        print(f"    AFTER:  {new_line.rstrip()}")

    cuda_rel = [s for s in sites2 if s.node_type.startswith("cuda_")]
    if cuda_rel:
        s = cuda_rel[0]
        mutated = op2.apply(source, s)
        orig_line = source.splitlines()[s.line_start - 1]
        new_line = mutated.splitlines()[s.line_start - 1]
        print(f"  RelOpReplace CUDA site L{s.line_start}: '{s.original_code}' →")
        print(f"    BEFORE: {orig_line.rstrip()}")
        print(f"    AFTER:  {new_line.rstrip()}")


if __name__ == "__main__":
    main()
